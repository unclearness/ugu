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
    const std::vector<Eigen::Vector3i>& verts_faces) {
  if (verts.empty() || verts_faces.empty()) {
    return false;
  }

  auto [face_centroids, face_planes] = ComputeFaceInfo(verts, verts_faces);

  m_verts = verts;
  m_verts_faces = verts_faces;

  m_face_centroids = std::move(face_centroids);
  m_face_planes = std::move(face_planes);

  m_tree = GetDefaultUniqueKdTree<float, 3>();

  m_tree->SetData(m_face_centroids);

  bool ret = m_tree->Build();

  return ret;
}

void KDTreeCorrespFinder::SetNnNum(uint32_t nn_num) { m_nn_num = nn_num; }

ugu::Corresp KDTreeCorrespFinder::Find(const Eigen::Vector3f& src_p,
                                       const Eigen::Vector3f& src_n) const {
  (void)src_n;

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
  for (const auto& index : indices) {
    const auto& vface = m_verts_faces[index];
    const auto& v0 = m_verts[vface[0]];
    const auto& v1 = m_verts[vface[1]];
    const auto& v2 = m_verts[vface[2]];

    const auto& plane = m_face_planes[index];

    auto [signed_dist, foot, uv] = PointTriangleDistance(
        src_p, v0, v1, v2, plane[0], plane[1], plane[2], plane[3]);
    float abs_dist = std::abs(signed_dist);

    if (abs_dist < min_dist) {
      min_dist = abs_dist;
      min_signed_dist = signed_dist;
      min_index = static_cast<int32_t>(index);
      min_bary = uv;
      min_foot = foot;
    }
  }

  ugu::Corresp corresp;
  corresp.fid = min_index;
  corresp.p = min_foot;
  corresp.signed_dist = min_signed_dist;
  corresp.abs_dist = min_dist;
  corresp.uv = min_bary;

  return corresp;
}

}  // namespace ugu
