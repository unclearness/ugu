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

Corresp KDTreeCorrespFinder::Find(const Eigen::Vector3f& src_p) const {
  return FindKnn(src_p, 1u)[0];
}

std::vector<Corresp> KDTreeCorrespFinder::FindKnn(
    const Eigen::Vector3f& src_p, const uint32_t& nn_num) const {
  std::vector<Corresp> corresps;

  // Get the closest src face
  // Roughly get candidates.NN for face center points.
  auto knn_results = m_tree->SearchKnn(src_p, nn_num);
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

    Corresp corresp;
    corresp.fid = static_cast<int32_t>(index);
    corresp.p = foot;
    corresp.signed_dist = signed_dist;
    corresp.abs_dist = abs_dist;
    corresp.uv = uv;
    corresp.n =
        (corresp.uv[0] * m_vert_normals[vface[0]] +
         corresp.uv[1] * m_vert_normals[vface[1]] +
         (1.f - corresp.uv[0] - corresp.uv[1]) * m_vert_normals[vface[2]])
            .normalized();
    corresps.push_back(corresp);
  }

  // Sort
  std::function<bool(const Corresp&, const Corresp&)> sort_func;
  sort_func = [&](const Corresp& a, const Corresp& b) {
    return a.abs_dist < b.abs_dist;
  };

  std::sort(corresps.begin(), corresps.end(), sort_func);

  return corresps;
}

}  // namespace ugu
