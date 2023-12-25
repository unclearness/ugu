#include "ugu/curvature/curvature.h"

#include <cmath>

#include "ugu/face_adjacency.h"
#include "ugu/util/raster_util.h"

namespace {

template <typename DerivedV, typename DerivedF>
bool CurvatureGaussianImpl(const std::vector<DerivedV>& vertices,
                           const std::vector<DerivedF>& faces,
                           std::vector<float>& curvature,
                           std::vector<DerivedV>& internal_angles) {
  internal_angles.resize(faces.size());
  curvature.resize(vertices.size(), static_cast<float>(2 * ugu::pi));

  typedef typename DerivedV::Scalar Scalar;

  auto corner = [](const typename DerivedV& x, const typename DerivedV& y,
                   const typename DerivedV& z) {
    auto v1 = x - y;
    auto v2 = z - y;
    return Scalar(2) * std::atan((v1 / v1.norm() - v2 / v2.norm()).norm() /
                                 (v1 / v1.norm() + v2 / v2.norm()).norm());
  };

  for (size_t i = 0; i < faces.size(); ++i) {
    internal_angles[i][0] = corner(vertices[faces[i][2]], vertices[faces[i][0]],
                                   vertices[faces[i][1]]);
    internal_angles[i][1] = corner(vertices[faces[i][0]], vertices[faces[i][1]],
                                   vertices[faces[i][2]]);
    internal_angles[i][2] = corner(vertices[faces[i][1]], vertices[faces[i][2]],
                                   vertices[faces[i][0]]);
    curvature[faces[i][0]] -= internal_angles[i][0];
    curvature[faces[i][1]] -= internal_angles[i][1];
    curvature[faces[i][2]] -= internal_angles[i][2];
  }

  return true;
};

}  // namespace

namespace ugu {

bool CurvatureGaussian(const std::vector<Eigen::Vector3f>& vertices,
                       const std::vector<Eigen::Vector3i>& faces,
                       std::vector<float>& curvature,
                       std::vector<Eigen::Vector3f>& internal_angles) {
  return CurvatureGaussianImpl(vertices, faces, curvature, internal_angles);
}

std::vector<float> BarycentricCellArea(
    const std::vector<Eigen::Vector3f>& vertices,
    const std::vector<Eigen::Vector3i>& faces) {
  // Adjacency vertex_adjacency = GenerateVertexAdjacency(
  //     faces, static_cast<uint32_t>(vertices.size()));

  auto v2f = GenerateVertex2FaceMap(faces, vertices.size());

  std::vector<float> area(vertices.size(), 0.f);

  for (size_t v_idx = 0; v_idx < vertices.size(); v_idx++) {
    const auto& f_idxs = v2f.at(static_cast<int>(v_idx));

    for (const auto& f_idx : f_idxs) {
      const auto& face = faces[f_idx];
      area[v_idx] +=
          TriArea(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    }
    area[v_idx] /= 3.f;
  }

  return area;
}

}  // namespace ugu