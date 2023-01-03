/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/correspondence/correspondence_finder.h"

#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace {

auto ComputeFaceInfo(const std::vector<Eigen::Vector3f>& verts,
                     const std::vector<Eigen::Vector3i>& vert_faces) {
  std::vector<Eigen::Vector3f> face_centroids(vert_faces.size());
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> face_planes(vert_faces.size());

  auto compute_func = [&](size_t idx) {
    const auto& face = vert_faces[idx];
    const Eigen::Vector3f& v0 = verts[face[0]];
    const Eigen::Vector3f& v1 = verts[face[1]];
    const Eigen::Vector3f& v2 = verts[face[2]];
    const auto centroid = (v0 + v1 + v2) / 3.0;
    face_centroids[idx] = centroid;

    Eigen::Vector3f vec10 = v1 - v0;
    Eigen::Vector3f vec20 = v2 - v0;
    Eigen::Vector3f n = vec10.cross(vec20).normalized();
    float d = -1.f * n.dot(v0);
    face_planes[idx] = {n[0], n[1], n[2], d};
  };

  ugu::parallel_for(0u, vert_faces.size(), compute_func);

  return std::make_tuple(face_centroids, face_planes);
}

}  // namespace

namespace ugu {

bool KDTreeCorrespFinder::Init(
    const std::vector<Eigen::Vector3f>& verts,
    const std::vector<Eigen::Vector3i>& verts_faces,
    const std::vector<Eigen::Vector3f>& vert_normals) {
  if (verts.empty() || verts_faces.empty()) {
    return false;
  }

  auto [face_centroids, face_planes] = ComputeFaceInfo(verts, verts_faces);

  m_verts = verts;
  m_verts_faces = verts_faces;
  if (vert_normals.size() != m_verts.size()) {
    const Eigen::Vector3f zero{0.0f, 0.0f, 0.0f};
    m_vert_normals.resize(m_verts.size(), zero);

    std::vector<int> add_count(m_verts.size(), 0);

    for (size_t i = 0; i < m_verts.size(); i++) {
      const auto& face = m_verts_faces[i];
      for (int j = 0; j < 3; j++) {
        int idx = face[j];
        m_vert_normals[idx] += Extract3f(face_planes[i]);
        add_count[idx]++;
      }
    }

    // Get average normal
    for (size_t i = 0; i < m_verts.size(); i++) {
      if (add_count[i] > 0) {
        m_vert_normals[i] /= static_cast<float>(add_count[i]);
        m_vert_normals[i].normalize();
      } else {
        // for unreferenced vertices, set (0, 0, 0)
        m_vert_normals[i].setZero();
      }
    }
  }

  m_face_centroids = std::move(face_centroids);
  m_face_planes = std::move(face_planes);

  m_tree = GetDefaultUniqueKdTree<float, 3>();

  m_tree->SetData(m_face_centroids);

  bool ret = m_tree->Build();

  return ret;
}

void KDTreeCorrespFinder::SetNnNum(uint32_t nn_num) { m_nn_num = nn_num; }

Corresp KDTreeCorrespFinder::Find(const Eigen::Vector3f& src_p,
                                  const Eigen::Vector3f& src_n,
                                  CorrespFinderMode mode) const {
#if 1
  // Check src_n norm
  Eigen::Vector3f src_n_ = src_n;
  bool use_src_n = std::abs(src_n.squaredNorm() - 1.f) < 0.01f;

  // Get the closest src face
  // Roughly get candidates.NN for face center points.
  auto knn_results = m_tree->SearchKnn(src_p, m_nn_num);
  std::vector<size_t> indices;
  std::vector<double> distance_sq;
  std::for_each(knn_results.begin(), knn_results.end(),
                [&](const KdTreeSearchResult& r) {
                  indices.push_back(r.index);
                  distance_sq.push_back(r.dist * r.dist);
                });

  // Check point - plane distance and get the smallest
  float min_dist = std::numeric_limits<float>::max();
  float min_signed_dist = std::numeric_limits<float>::infinity();
  int32_t min_index = -1;
  Eigen::Vector2f min_bary(99.f, 99.f);
  Eigen::Vector3f min_foot(99.f, 99.f, 99.f);
  float min_cos_dist = std::numeric_limits<float>::max();
  float min_angle = std::numeric_limits<float>::max();
  for (const auto& index : indices) {
    const auto& vface = m_verts_faces[index];
    const auto& v0 = m_verts[vface[0]];
    const auto& v1 = m_verts[vface[1]];
    const auto& v2 = m_verts[vface[2]];

    const auto& plane = m_face_planes[index];

    auto [signed_dist, foot, uv] = PointTriangleDistance(
        src_p, v0, v1, v2, plane[0], plane[1], plane[2], plane[3]);
    float abs_dist = std::abs(signed_dist);

    if (!use_src_n) {
      Eigen::Vector3f ray = (src_p - foot).normalized();
      src_n_ = ray;
    }

#if 0
    Eigen::Vector3f n =
        (uv[0] * m_vert_normals[vface[0]] + uv[1] * m_vert_normals[vface[1]] +
         (1.f - uv[0] - uv[1]) * m_vert_normals[vface[2]])
            .normalized();  // Barycentric interpolation of vertex normals
#else
    // Face normal:
    // Foot is inside triangle: cos is zero
    // Foot is on the edge of triangle: cos is non-zero
    Eigen::Vector3f n = Extract3f(m_face_planes[index]);
#endif

    float angle_cos = n.dot(src_n_);
    // Ignore cos  sign because surfaces are close and could be both front or
    // back.
    angle_cos = std::clamp(std::abs(angle_cos), 0.f, 1.f);

    // TODO: better weight method with cos
    // With face normal, below calculation gives minimum value because face
    // normal is uniform on the face. With barycentric interpoltion, possively
    // no...
#if 0
    float cos_abs_dist = abs_dist * (2.f - angle_cos);
#else
    float cos_abs_dist =
        abs_dist / (angle_cos + std::numeric_limits<float>::epsilon());
#endif

    // TODO: CorrespFinderMode::kMinAngle would be unstable because of many
    // angle = zero faces
    float angle = std::acos(angle_cos);

    if ((mode == CorrespFinderMode::kMinDist && abs_dist < min_dist) ||
        (mode == CorrespFinderMode::kMinAngleCosDist &&
         cos_abs_dist < min_cos_dist) ||
        (mode == CorrespFinderMode::kMinAngle && angle <= min_angle &&
         abs_dist < min_dist)) {
      min_dist = abs_dist;
      min_signed_dist = signed_dist;
      min_index = static_cast<int32_t>(index);
      min_bary = uv;
      min_foot = foot;
      min_angle = angle;
      min_cos_dist = cos_abs_dist;
    }
  }

  Corresp corresp;
  corresp.fid = min_index;
  corresp.p = min_foot;
  corresp.signed_dist = min_signed_dist;
  corresp.abs_dist = min_dist;
  corresp.uv = min_bary;
  corresp.angle = min_angle;
  corresp.cos_abs_dist = min_cos_dist;
  // std::cout << corresp.cos_abs_dist << " " << corresp.angle << std::endl;

  return corresp;
#else
  // Slow
  return FindAll(src_p, src_n, mode)[0];

#endif
}

std::vector<Corresp> KDTreeCorrespFinder::FindAll(
    const Eigen::Vector3f& src_p, const Eigen::Vector3f& src_n,
    CorrespFinderMode mode) const {
  // Check src_n norm
  Eigen::Vector3f src_n_ = src_n;
  bool use_src_n = std::abs(src_n.squaredNorm() - 1.f) < 0.01f;

  std::vector<Corresp> corresps;

  // Get the closest src face
  // Roughly get candidates.NN for face center points.
  auto knn_results = m_tree->SearchKnn(src_p, m_nn_num);
  std::vector<size_t> indices;
  std::vector<double> distance_sq;
  std::for_each(knn_results.begin(), knn_results.end(),
                [&](const KdTreeSearchResult& r) {
                  indices.push_back(r.index);
                  distance_sq.push_back(r.dist * r.dist);
                });

  for (const auto& index : indices) {
    const auto& vface = m_verts_faces[index];
    const auto& v0 = m_verts[vface[0]];
    const auto& v1 = m_verts[vface[1]];
    const auto& v2 = m_verts[vface[2]];

    const auto& plane = m_face_planes[index];

    auto [signed_dist, foot, uv] = PointTriangleDistance(
        src_p, v0, v1, v2, plane[0], plane[1], plane[2], plane[3]);
    float abs_dist = std::abs(signed_dist);

    if (!use_src_n) {
      Eigen::Vector3f ray = (src_p - foot).normalized();
      src_n_ = ray;
    }

#if 0
    Eigen::Vector3f n =
        (uv[0] * m_vert_normals[vface[0]] + uv[1] * m_vert_normals[vface[1]] +
         (1.f - uv[0] - uv[1]) * m_vert_normals[vface[2]])
            .normalized();  // Barycentric interpolation of vertex normals
#else
    // Face normal:
    // Foot is inside triangle: cos is zero
    // Foot is on the edge of triangle: cos is non-zero
    Eigen::Vector3f n = Extract3f(m_face_planes[index]);
#endif

    float angle_cos = n.dot(src_n_);
    // Ignore cos  sign because surfaces are close and could be both front or
    // back.
    angle_cos = std::clamp(std::abs(angle_cos), 0.f, 1.f);

    // TODO: better weight method with cos
    // With face normal, below calculation gives minimum value because face
    // normal is uniform on the face. With barycentric interpoltion, possively
    // no...
#if 0
    float cos_abs_dist = abs_dist * (2.f - angle_cos);
#else
    float cos_abs_dist =
        abs_dist / (angle_cos + std::numeric_limits<float>::epsilon());
#endif

    // TODO: CorrespFinderMode::kMinAngle would be unstable because of many
    // angle = zero faces
    float angle = std::acos(angle_cos);

    Corresp corresp;
    corresp.fid = static_cast<int32_t>(index);
    corresp.p = foot;
    corresp.signed_dist = signed_dist;
    corresp.abs_dist = abs_dist;
    corresp.uv = uv;
    corresp.angle = angle;
    corresp.cos_abs_dist = cos_abs_dist;

    corresps.push_back(corresp);
  }

  // Sort
  std::function<bool(const Corresp&, const Corresp&)> sort_func;
  if (mode == CorrespFinderMode::kMinDist) {
    sort_func = [&](const Corresp& a, const Corresp& b) {
      return a.abs_dist < b.abs_dist;
    };
  } else if (mode == CorrespFinderMode::kMinAngleCosDist) {
    sort_func = [&](const Corresp& a, const Corresp& b) {
      return a.cos_abs_dist < b.cos_abs_dist;
    };
  } else if (mode == CorrespFinderMode::kMinAngle) {
    sort_func = [&](const Corresp& a, const Corresp& b) {
      return a.angle < b.angle;
    };
  }

  std::sort(corresps.begin(), corresps.end(), sort_func);

  return corresps;
}

}  // namespace ugu
