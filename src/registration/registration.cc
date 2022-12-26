/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/registration/registration.h"

#include <iostream>

#include "Eigen/Geometry"
#include "ugu/accel/kdtree.h"

namespace {
using namespace ugu;
template <typename T>
bool RigidIcpPointToPointImpl(const std::vector<Eigen::Vector3<T>>& src,
                              const std::vector<Eigen::Vector3<T>>& dst,
                              const IcpTerminateCriteria& terminate_criteria,
                              IcpOutput& output, bool with_scale,
                              KdTreePtr<T, 3> kdtree = nullptr) {
  if (kdtree == nullptr || !kdtree->IsInitialized()) {
    kdtree = GetDefaultKdTree<T, 3>();
    kdtree->SetData(dst);
    if (!kdtree->Build()) {
      return false;
    }
  }

  int iter = 0;
  double loss_ave = std::numeric_limits<double>::max();
  Eigen::Transform<T, 3, Eigen::Affine> accum_transform =
      Eigen::Transform<T, 3, Eigen::Affine>::Identity();
  double prev_loss_ave = 0.0;

  auto terminated = [&]() {
    if (iter >= terminate_criteria.iter_max) {
      return true;
    }
    if (loss_ave < terminate_criteria.loss_min) {
      return true;
    }
    if (iter > 1 &&
        std::abs(loss_ave - prev_loss_ave) < terminate_criteria.loss_eps) {
      return true;
    }

    return false;
  };

  std::vector<Eigen::Vector3<T>> current = src;
  std::vector<Eigen::Vector3<T>> target = dst;
  std::vector<KdTreeSearchResults> corresp(current.size());
  auto corresp_func = [&](size_t idx) {
    corresp[idx] = kdtree->SearchNn(current[idx]);
    target[idx] = dst[corresp[idx][0].index];
  };
  auto update_func = [&](size_t idx) {
    current[idx] = accum_transform * src[idx];
  };
  do {
    if (iter > 0) {
      // Solve rotation, translation and scale by Umeyama closed-form method
      Eigen::Transform<T, 3, Eigen::Affine> cur_transform;
      if (with_scale) {
        cur_transform =
            FindSimilarityTransformFrom3dCorrespondences(current, target)
                .cast<T>();
      } else {
        cur_transform =
            FindRigidTransformFrom3dCorrespondences(current, target).cast<T>();
      }

      accum_transform = cur_transform * accum_transform;

      // Update current points
      parallel_for(0u, current.size(), update_func);
    }

    // Find coressponding points
    parallel_for(0u, current.size(), corresp_func);

    // Compute loss
    prev_loss_ave = loss_ave;
    loss_ave = 0;
    for (const auto& r : corresp) {
      loss_ave += r[0].dist;
    }
    loss_ave /= static_cast<double>(corresp.size());

    // Update count
    iter++;

    output.transform_histry.push_back(accum_transform.cast<double>());
    output.loss_histroty.push_back(loss_ave);

    std::cout << iter << " " << loss_ave << std::endl;
    // std::cout << best_transform. << std::endl;
    std::cout << accum_transform.translation() << std::endl;
    std::cout << accum_transform.rotation().matrix() << std::endl << std::endl;

  } while (!terminated());

  return true;
}

}  // namespace

namespace ugu {

// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
  if (src.size() < 3 || src.size() != dst.size()) {
    return Eigen::Affine3d::Identity();
  }

  Eigen::Vector3d src_centroid;
  src_centroid.setZero();
  for (const auto& p : src) {
    src_centroid += p;
  }
  src_centroid /= static_cast<double>(src.size());

  Eigen::Vector3d dst_centroid;
  dst_centroid.setZero();
  for (const auto& p : dst) {
    dst_centroid += p;
  }
  dst_centroid /= static_cast<double>(dst.size());

  Eigen::MatrixXd normed_src(3, src.size());
  for (size_t i = 0; i < src.size(); i++) {
    normed_src.col(i) = src[i] - src_centroid;
  }

  Eigen::MatrixXd normed_dst(3, dst.size());
  for (size_t i = 0; i < dst.size(); i++) {
    normed_dst.col(i) = dst[i] - dst_centroid;
  }

  Eigen::MatrixXd normed_dst_T = normed_dst.transpose();
  Eigen::Matrix3d H = normed_src * normed_dst_T;

  // TODO: rank check

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  double det = R.determinant();

  constexpr double assert_eps = 0.001;
  assert(std::abs(std::abs(det) - 1.0) < assert_eps);

  // https://github.com/nghiaho12/rigid_transform_3D/blob/32c95173966488926ccabc67c5ea70c03f63df7a/rigid_transform_3D.py#L44
  if (det < 0) {
    Eigen::Matrix3d V = svd.matrixV();

    V.col(2) *= -1.0;

    R = V * svd.matrixU().transpose();
    det = R.determinant();
  }

  assert(std::abs(det - 1.0) < assert_eps);

  Eigen::Vector3d t = dst_centroid - R * src_centroid;

  Eigen::Affine3d T = Eigen::Translation3d(t) * R;

  return T;
}

Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst) {
  std::vector<Eigen::Vector3d> src_d, dst_d;
  auto to_double = [](const std::vector<Eigen::Vector3f>& fvec,
                      std::vector<Eigen::Vector3d>& dvec) {
    std::transform(fvec.begin(), fvec.end(), std::back_inserter(dvec),
                   [](const Eigen::Vector3f& f) { return f.cast<double>(); });
  };

  to_double(src, src_d);
  to_double(dst, dst_d);
  return FindRigidTransformFrom3dCorrespondences(src_d, dst_d);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
  Eigen::MatrixXd src_(src.size(), 3);
  for (size_t i = 0; i < src.size(); i++) {
    src_.row(i) = src[i];
  }
  Eigen::MatrixXd dst_(dst.size(), 3);
  for (size_t i = 0; i < dst.size(); i++) {
    dst_.row(i) = dst[i];
  }
  return FindSimilarityTransformFrom3dCorrespondences(src_, dst_);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst) {
  Eigen::MatrixXd src_(src.size(), 3);
  for (size_t i = 0; i < src.size(); i++) {
    src_.row(i) = src[i].cast<double>();
  }
  Eigen::MatrixXd dst_(dst.size(), 3);
  for (size_t i = 0; i < dst.size(); i++) {
    dst_.row(i) = dst[i].cast<double>();
  }
  return FindSimilarityTransformFrom3dCorrespondences(src_, dst_);
}

Eigen::Affine3d FindSimilarityTransformFrom3dCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst) {
  Eigen::MatrixXd R;
  Eigen::MatrixXd t;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd T;
  bool ret =
      FindSimilarityTransformFromPointCorrespondences(src, dst, R, t, scale, T);
  assert(R.rows() == 3 && R.cols() == 3);
  assert(t.rows() == 3 && t.cols() == 1);
  assert(T.rows() == 4 && T.cols() == 4);
  Eigen::Affine3d T_3d = Eigen::Affine3d::Identity();
  if (ret) {
    T_3d = Eigen::Translation3d(t) * R * Eigen::Scaling(scale.diagonal());
  }

  return T_3d;
}

bool FindSimilarityTransformFromPointCorrespondences(
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst, Eigen::MatrixXd& R,
    Eigen::MatrixXd& t, Eigen::MatrixXd& scale, Eigen::MatrixXd& T) {
  const size_t n_data = src.rows();
  const size_t n_dim = src.cols();
  if (n_data < 1 || n_dim < 1 || n_data < n_dim || src.rows() != dst.rows() ||
      src.cols() != dst.cols()) {
    return false;
  }

  Eigen::VectorXd src_mean = src.colwise().mean();
  Eigen::VectorXd dst_mean = dst.colwise().mean();

  Eigen::MatrixXd src_demean = src.rowwise() - src_mean.transpose();
  Eigen::MatrixXd dst_demean = dst.rowwise() - dst_mean.transpose();

  Eigen::MatrixXd A =
      dst_demean.transpose() * src_demean / static_cast<double>(n_data);

  Eigen::VectorXd d = Eigen::VectorXd::Ones(n_dim);
  double det_A = A.determinant();
  if (det_A < 0) {
    d.coeffRef(n_dim - 1, 0) = -1;
  }

  T = Eigen::MatrixXd::Identity(n_dim + 1, n_dim + 1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd S = svd.singularValues().asDiagonal();
  Eigen::MatrixXd V = svd.matrixV();

  double det_U = U.determinant();
  double det_V = V.determinant();
  double det_orgR = det_U * det_V;
  constexpr double assert_eps = 0.001;
  assert(std::abs(std::abs(det_orgR) - 1.0) < assert_eps);

  size_t rank_A = static_cast<size_t>(svd.rank());
  if (rank_A == 0) {
    // null matrix case
    return false;
    // R = U * d.asDiagonal() * V.transpose();
  } else if (rank_A == (n_dim - 1)) {
    if (det_orgR > 0) {
      // Valid rotation case
      R = U * V.transpose();
    } else {
      // Mirror (reflection) case
      double s = d.coeff(n_dim - 1, 0);
      d.coeffRef(n_dim - 1, 0) = -1;
      R = U * d.asDiagonal() * V.transpose();
      d.coeffRef(n_dim - 1, 0) = s;
    }
  } else {
    // degenerate case
    R = U * d.asDiagonal() * V.transpose();
  }
  assert(std::abs(R.determinant() - 1.0) < assert_eps);

  // Eigen::MatrixXd src_demean_cov =
  //    (src_demean.adjoint() * src_demean) / double(n_data);
  double src_var =
      src_demean.rowwise().squaredNorm().sum() / double(n_data) + 1e-30;
  double uniform_scale = 1.0 / src_var * (S * d.asDiagonal()).trace();
  // Question: Is it possible to estimate non-uniform scale?
  scale = Eigen::MatrixXd::Identity(R.rows(), R.cols());
  scale *= uniform_scale;

  t = dst_mean - scale * R * src_mean;

  T.block(0, 0, n_dim, n_dim) = scale * R;
  T.block(0, n_dim, n_dim, 1) = t;

  return true;
}

bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3f>& src,
                          const std::vector<Eigen::Vector3f>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          KdTreePtr<float, 3> kdtree) {
  return RigidIcpPointToPointImpl(src, dst, terminate_criteria, output,
                                  with_scale, kdtree);
}

bool RigidIcpPointToPoint(const std::vector<Eigen::Vector3d>& src,
                          const std::vector<Eigen::Vector3d>& dst,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          KdTreePtr<double, 3> kdtree) {
  return RigidIcpPointToPointImpl(src, dst, terminate_criteria, output,
                                  with_scale, kdtree);
}

bool RigidIcp(const Mesh& src, const Mesh& dst, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria, IcpOutput& output,
              bool with_scale, KdTreePtr<float, 3> kdtree) {
  if (loss_type == IcpLossType::kPointToPoint) {
    auto ret =
        RigidIcpPointToPoint(src.vertices(), dst.vertices(), terminate_criteria,
                             output, with_scale, kdtree);
    return ret;
  }
  throw std::runtime_error("Not implemented");
}

}  // namespace ugu