/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/accel/kdtree.h"
#include "ugu/mesh.h"

namespace ugu {

// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

// "Least-squares estimation of transformation parameters between two point
// patterns ", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
// implementation reference:
// https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/_geometric.py#L63
bool FindSimilarityTransformFromPointCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, Eigen::MatrixXd& R,
    Eigen::MatrixXd& t, Eigen::MatrixXd& scale, Eigen::MatrixXd& T);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);

enum class IcpLossType { kPointToPoint = 0, kPointToPlane = 1 };

struct IcpTerminateCriteria {
  int iter_max = 100;
  double loss_min = 0.01;
  double loss_eps = 0.001;

};

struct IcpOutput {
  std::vector<Eigen::Affine3d> transform_histry;
  std::vector<double> loss_histroty;
};


bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3f>& src,
                          const std::vector<Eigen::Vector3f>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          KdTreePtr<float, 3> kdtree = nullptr);

bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3d>& src,
                          const std::vector<Eigen::Vector3d>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          KdTreePtr<double, 3> kdtree = nullptr);

bool RigidIcp(const Mesh& src, const Mesh& dst, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              IcpOutput& output, bool with_scale,
              KdTreePtr<float, 3> kdtree = nullptr);

}  // namespace ugu