/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/util/math_util.h"

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

  if (det < 0) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(
        R, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d V = svd2.matrixV();

    V.coeffRef(0, 2) *= -1.0;
    V.coeffRef(1, 2) *= -1.0;
    V.coeffRef(2, 2) *= -1.0;

    R = V * svd2.matrixU().transpose();
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

  int rank_A = static_cast<int>(svd.rank());
  if (rank_A == 0) {
    // null matrix case
    return false;
  } else if (rank_A == n_dim - 1) {
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

}  // namespace ugu