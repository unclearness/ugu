/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#pragma once

#include "ugu/accel/kdtree.h"
#include "ugu/correspondence/correspondence_finder.h"
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
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst,
    Eigen::MatrixXd& R_similarity, Eigen::MatrixXd& t_similarity,
    Eigen::MatrixXd& scale, Eigen::MatrixXd& T_similarity,
    Eigen::MatrixXd& R_rigid, Eigen::MatrixXd& t_rigid,
    Eigen::MatrixXd& T_rigid);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst);

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst);

void DecomposeRts(const Eigen::Affine3f& T, Eigen::Matrix3f& R,
                  Eigen::Vector3f& t, Eigen::Vector3f& s);
void DecomposeRts(const Eigen::Affine3d& T, Eigen::Matrix3d& R,
                  Eigen::Vector3d& t, Eigen::Vector3d& s);

enum class IcpLossType { kPointToPoint = 0, kPointToPlane = 1 };

struct IcpTerminateCriteria {
  int iter_max = 20;
  double loss_min = 0.001;
  double loss_eps = 0.0001;
};

struct IcpOutput {
  std::vector<Eigen::Affine3d> transform_histry;
  std::vector<double> loss_histroty;
};

using IcpCallbackFunc =
    std::function<void(const IcpTerminateCriteria&, const IcpOutput&)>;

// WARNING: with_scale = true is not recommended. In most cases, the
// transformation will incorrectly converge to zero scale and a certain
// translation, which corresponds to a certain target point.
bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3f>& src,
                          const std::vector<Eigen::Vector3f>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale = false,
                          KdTreePtr<float, 3> kdtree = nullptr,
                          IcpCallbackFunc callback = nullptr);

bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3d>& src,
                          const std::vector<Eigen::Vector3d>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale = false,
                          KdTreePtr<double, 3> kdtree = nullptr,
                          IcpCallbackFunc callback = nullptr);

bool RigidIcpPointToPlane(const std::vector<Eigen::Vector3f>& src_points,
                          const std::vector<Eigen::Vector3f>& dst_points,
                          const std::vector<Eigen::Vector3i>& dst_faces,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale = false,
                          CorrespFinderPtr corresp_finder = nullptr,
                          IcpCallbackFunc callback = nullptr);

bool RigidIcpPointToPlane(const std::vector<Eigen::Vector3d>& src_points,
                          const std::vector<Eigen::Vector3d>& dst_points,
                          const std::vector<Eigen::Vector3i>& dst_faces,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale = false,
                          CorrespFinderPtr corresp_finder = nullptr,
                          IcpCallbackFunc callback = nullptr);

bool RigidIcp(const Mesh& src, const Mesh& dst, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria, IcpOutput& output,
              bool with_scale = false, KdTreePtr<float, 3> kdtree = nullptr,
              CorrespFinderPtr corresp_finder = nullptr,
              IcpCallbackFunc callback = nullptr);

}  // namespace ugu