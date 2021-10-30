/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */
#ifdef UGU_USE_NANOFLANN

#include "ugu/textrans/correspondence_finder.h"

#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace {

auto ComputeFaceInfo(const std::vector<Eigen::Vector3f>& verts,
                     const std::vector<Eigen::Vector3i>& vert_faces) {
  std::vector<Eigen::Vector3f> face_centroids;
  // ax + by + cz + d = 0
  std::vector<Eigen::Vector4f> face_planes;

  for (const auto& face : vert_faces) {
    const Eigen::Vector3f& v0 = verts[face[0]];
    const Eigen::Vector3f& v1 = verts[face[1]];
    const Eigen::Vector3f& v2 = verts[face[2]];
    const auto centroid = (v0 + v1 + v2) / 3.0;
    face_centroids.emplace_back(centroid);

    Eigen::Vector3f vec10 = v1 - v0;
    Eigen::Vector3f vec20 = v2 - v0;
    Eigen::Vector3f n = vec10.cross(vec20).normalized();
    float d = -1.f * n.dot(v0);
    face_planes.emplace_back(n[0], n[1], n[2], d);
  }

  return std::make_tuple(face_centroids, face_planes);
}

}  // namespace

namespace ugu {

std::tuple<std::vector<size_t>, std::vector<float>> QueryKdTree(
    const Eigen::Vector3f& p, const ugu_kdtree_t& index, int32_t nn_num) {
  std::vector<std::pair<Eigen::Index, float>> ret_matches;
  std::vector<size_t> out_indices(nn_num);
  std::vector<float> out_distance_sq(nn_num);
  index.index->knnSearch(p.data(), nn_num, out_indices.data(),
                         out_distance_sq.data());
  return std::make_tuple(out_indices, out_distance_sq);
}


bool KDTreeCorrespFinder::Init(
    const std::vector<Eigen::Vector3f>& verts,
    const std::vector<Eigen::Vector3i>& verts_faces) {
  auto [face_centroids, face_planes] = ComputeFaceInfo(verts, verts_faces);

  m_verts = verts;
  m_verts_faces = verts_faces;

  m_face_centroids = std::move(face_centroids);
  m_face_planes = std::move(face_planes);

  m_tree = std::make_shared<ugu_kdtree_t>();
  m_tree->init(verts_faces.size(), m_face_centroids, 10 /* max leaf */);

  return true;
}

void KDTreeCorrespFinder::SetNnNum(uint32_t nn_num) { m_nn_num = nn_num; }

ugu::Corresp KDTreeCorrespFinder::Find(const Eigen::Vector3f& src_p,
                                       const Eigen::Vector3f& src_n) const {
  (void)src_n;

  // Get the closest src face
  // Roughly get candidates.NN for face center points.
  auto [indices, distance_sq] = QueryKdTree(src_p, *m_tree, m_nn_num);
  // Check point - plane distance and get the smallest
  float min_dist = std::numeric_limits<float>::max();
  float min_signed_dist = std::numeric_limits<float>::infinity();
  int32_t min_index = -1;
  Eigen::Vector3f min_bary(99.f, 99.f, 99.f);
  Eigen::Vector3f min_foot;
  for (const auto& index : indices) {
    const auto& vface = m_verts_faces[index];
    const auto& v0 = m_verts[vface[0]];
    const auto& v1 = m_verts[vface[1]];
    const auto& v2 = m_verts[vface[2]];

    // Case 1: foot of perpendicular line is inside of target triangle
    const auto& plane = m_face_planes[index];
    const Eigen::Vector3f normal = Extract3f(plane);
    // point-plane distance |ax'+by'+cz'+d|
    const float signed_dist = src_p.dot(normal) + plane[3];

    float dist = std::abs(signed_dist);
    Eigen::Vector3f foot = -signed_dist * normal + src_p;
    float foot_dist = foot.dot(normal) + plane[3];
    if (std::abs(foot_dist) > 0.0001f) {
      // ugu::LOGE("wrong dist %f %f\n", foot, foot_dist);
      throw std::runtime_error("wrong dist");
    }

    auto [isInside, bary] = ugu::IsPoint3dInsideTriangle(foot, v0, v1, v2);

    if (dist < min_dist && isInside) {
      min_dist = dist;
      min_signed_dist = signed_dist;
      min_index = static_cast<int32_t>(index);
      min_bary = bary;
      min_foot = foot;
    }
  }

  // Case 2: foot of perpendicular line is outside of the triangle
  if (min_index < 0) {
    for (const auto& index : indices) {
      const auto& vface = m_verts_faces[index];
      const auto& v0 = m_verts[vface[0]];
      const auto& v1 = m_verts[vface[1]];
      const auto& v2 = m_verts[vface[2]];

      const auto& plane = m_face_planes[index];
      const Eigen::Vector3f normal = Extract3f(plane);

      // Check distance to boundary line segments of triangle
      auto [ldist, lfoot] = ugu::PointTriangleDistance(src_p, v0, v1, v2);
      auto [lIsInside, lbary] = ugu::IsPoint3dInsideTriangle(lfoot, v0, v1, v2);
      if (!lIsInside) {
        // By numerical reason, sometimes becomes little over [0, 1]
        // So just clip
        lbary[0] = std::clamp(lbary[0], 0.f, 1.f);
        lbary[1] = std::clamp(lbary[1], 0.f, 1.f);
      }

      if (ldist < min_dist) {
        min_dist = ldist;
        // TODO: Add sign
        min_signed_dist = ldist;
        min_index = static_cast<int32_t>(index);
        min_bary = lbary;
        min_foot = lfoot;
      }
    }
  }

  // IMPORTANT: Without this line, unexpectable noises may appear...
  min_bary[2] = 1.f - min_bary[0] - min_bary[1];

  ugu::Corresp corresp;
  corresp.fid = min_index;
  corresp.p = min_foot;
  corresp.singed_dist = min_signed_dist;
  corresp.abs_dist = min_dist;
  corresp.uv = min_bary;

  return corresp;
}

}  // namespace ugu

#endif
