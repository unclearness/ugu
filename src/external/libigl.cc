/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/external/external.h"

#ifdef UGU_USE_LIBGIL

//#include "igl/boundary_loop.h"
#include "igl/lscm.h"

#endif

namespace ugu {

bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices,
                const std::vector<int>& boundary,
                std::vector<Eigen::Vector2f>& uvs) {
#ifdef UGU_USE_LIBGIL

  Eigen::MatrixXd V(vertices.size(), 3);
  Eigen::MatrixXi F(vertex_indices.size(), 3);

  for (size_t i = 0; i < vertices.size(); i++) {
    V.row(i) = vertices[i].cast<double>();
  }

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    F.row(i) = vertex_indices[i];
  }

  Eigen::MatrixXd V_uv;

  Eigen::VectorXi b(2, 1);
  // igl::boundary_loop(F, bnd);
  b(0) = boundary[0];
  b(1) = boundary[boundary.size() / 2];
  Eigen::MatrixXd bc(2, 2);
  bc << 0, 0, 1, 0;

  // LSCM parameterization
  igl::lscm(V, F, b, bc, V_uv);

  uvs.clear();
  Eigen::Vector2f invalid;
  invalid.Constant(std::numeric_limits<float>::lowest());
  uvs.resize(vertices.size(), invalid);

  Eigen::Vector2d min_uv = V_uv.colwise().minCoeff();
  Eigen::Vector2d max_uv = V_uv.colwise().maxCoeff();
  Eigen::Vector2d diff = max_uv - min_uv;

  for (size_t i = 0; i < uvs.size(); i++) {
    uvs[i] = (V_uv.row(i) - min_uv.transpose())
                 .cwiseProduct(diff.cwiseInverse().transpose())
                 .cast<float>();
  }

  return true;
#else
  (void)vertices;
  (void)vertex_indices;
  (void)boundary;
  (void)uvs;
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

#if 0
bool LibiglLscm(const std::vector<Eigen::Vector3f>& vertices,
                const std::vector<Eigen::Vector3i>& vertex_indices,
                const std::vector<std::vector<int>>& boundaries,
                std::vector<Eigen::Vector2f>& uvs,
                std::vector<Eigen::Vector3i>& uv_indices) {
#ifdef UGU_USE_LIBGIL
  bool ret = true;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXd V_uv;

  // Eigen::VectorXib  = Eigen::VectorXi::Map(a.data(), a.size());
  // b(fix, 1);
  // igl::boundary_loop(F, bnd);
  // b(0) = bnd(0);
  // b(1) = bnd(bnd.size() / 2);
  Eigen::MatrixXd bc(2, 2);
  bc << 0, 0, 1, 0;

  // LSCM parameterization
  igl::lscm(V, F, b, bc, V_uv);

  return ret;
#else
  (void)src;
  (void)target_face_num;
  (void)decimated;
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}
#endif

}  // namespace ugu
