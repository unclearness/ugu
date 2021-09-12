/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/decimation/decimation.h"

#include <queue>

#include "ugu/face_adjacency.h"

namespace {

using VertexAttr = Eigen::VectorXd;
using VertexAttrs = std::vector<VertexAttr>;
using VertexAttrsPtr = std::shared_ptr<VertexAttrs>;

using QSlimEdge = std::array<int32_t, 2>;

using Quadric = Eigen::MatrixXd;
using Quadrics = std::vector<Quadric>;
using QuadricPtr = std::shared_ptr<Quadric>;
using QuadricsPtr = std::shared_ptr<Quadrics>;

bool ComputeOptimalContraction(const VertexAttr& v1, const Quadric& q1,
                               const VertexAttr& v2, const Quadric& q2,
                               VertexAttr& v, double& error) {
  VertexAttr zero(v1.size());
  zero.setZero();
  zero[zero.size() - 1] = 1.0;

  bool ret = true;

  Quadric q = q1 + q2;

  if (std::abs(q.determinant()) < 0.00001) {
    // Not ivertible case
    ret = false;

    // Select best one from v1, v2 and (v1+v2)/2
    std::array<VertexAttr, 3> candidates = {v1, v2, (v1 + v2) * 0.5};
    double min_error = std::numeric_limits<double>::max();
    VertexAttr min_vert = v1;

    for (int i = 0; i < 3; i++) {
      double tmp_error = candidates[i].transpose() * q * candidates[i];
      if (tmp_error < min_error) {
        min_error = tmp_error;
        candidates[i];
        min_vert = candidates[i];
      }
    }

    error = min_error;
    v = min_vert;

  } else {
    // Invertible case
    // Eq. (1) in the paper
    Quadric q_inv = q.inverse();
    v = q_inv * zero;
    error = v.transpose() * q * v;
  }

  return ret;
}

struct QSlimEdgeInfo {
  QSlimEdge edge = {-1, -1};
  int org_vid = -1;
  double error = std::numeric_limits<double>::max();
  QuadricsPtr quadrics;
  VertexAttrsPtr vert_attrs;
  VertexAttr decimated_v;
  QSlimEdgeInfo(QSlimEdge edge_, int org_vid_, double error_,
                VertexAttrsPtr vert_attrs_, QuadricsPtr quadrics_) {
    edge = edge_;
    org_vid = org_vid_;
    error = error_;
    quadrics = quadrics_;
    vert_attrs = vert_attrs_;
    decimated_v.setZero();
  }

  void ComputeError() {
    int v0 = edge[0];
    int v1 = edge[1];
    ComputeOptimalContraction(vert_attrs->at(v0), quadrics->at(v0),
                              vert_attrs->at(v1), quadrics->at(v1), decimated_v,
                              error);
  }

  QSlimEdgeInfo(){};
  ~QSlimEdgeInfo(){};
};

bool operator<(const QSlimEdgeInfo& l, const QSlimEdgeInfo& r) {
  return l.error > r.error;
};

using QSlimHeap =
    std::priority_queue<QSlimEdgeInfo, std::vector<QSlimEdgeInfo>>;

}  // namespace

namespace ugu {

bool QSlim(MeshPtr mesh, QSlimType type, bool keep_geom_boundary,
           bool keep_uv_boundary, bool accept_non_edge) {
  // Initialize quadrics
  Quadrics quadrics;

  QSlimHeap heap;

  // Main loop
  while (!heap.empty()) {
    // Find the lowest error edge
  }

  return true;
}

}  // namespace ugu
