/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/registration/registration.h"

// #define UGU_USE_OWN_UMEYAMA

namespace {
using namespace ugu;

template <typename T>
bool RigidIcpImpl(const std::vector<Eigen::Vector3<T>>& src_points,
                  const std::vector<Eigen::Vector3<T>>& dst_points,
                  const std::vector<Eigen::Vector3i>& dst_faces,
                  const IcpTerminateCriteria& terminate_criteria,
                  IcpOutput& output, bool with_scale, IcpLossType loss_type,
                  KdTreePtr<T, 3> kdtree = nullptr,
                  CorrespFinderPtr corresp_finder = nullptr) {
  if (loss_type == IcpLossType::kPointToPoint &&
      (kdtree == nullptr || !kdtree->IsInitialized())) {
    kdtree = GetDefaultKdTree<T, 3>();
    kdtree->SetData(dst_points);
    if (!kdtree->Build()) {
      return false;
    }
  }

  if (loss_type == IcpLossType::kPointToPlane && (corresp_finder == nullptr)) {
    corresp_finder = KDTreeCorrespFinder::Create();
    std::vector<Eigen::Vector3f> dst_points_(dst_points.size());
    for (size_t i = 0; i < dst_points_.size(); i++) {
      dst_points_[i] = dst_points[i].template cast<float>();
    }
    corresp_finder->Init(dst_points_, dst_faces);
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

  std::vector<Eigen::Vector3<T>> current = src_points;
  std::vector<Eigen::Vector3<T>> target = dst_points;
  std::vector<KdTreeSearchResults> corresp(current.size());
  auto point2point_corresp_func = [&](size_t idx) {
    corresp[idx] = kdtree->SearchNn(current[idx]);
    target[idx] = dst_points[corresp[idx][0].index];
  };
  auto point2plane_corresp_func = [&](size_t idx) {
    Corresp c = corresp_finder->Find(current[idx].template cast<float>(),
                                     Eigen::Vector3f::Zero());
    target[idx] = c.p.template cast<T>();
    corresp[idx].resize(1);
    corresp[idx][0].dist = c.abs_dist;
    corresp[idx][0].index = size_t(~0);
  };

  std::function<void(size_t)> corresp_func = nullptr;
  if (loss_type == IcpLossType::kPointToPoint) {
    corresp_func = point2point_corresp_func;
  } else if (loss_type == IcpLossType::kPointToPlane) {
    corresp_func = point2plane_corresp_func;
  }

  auto update_func = [&](size_t idx) {
    current[idx] = accum_transform * src_points[idx];
  };
  do {
    Eigen::Transform<T, 3, Eigen::Affine> cur_transform =
        Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    if (iter > 0) {
      // Solve rotation, translation and scale by Umeyama closed-form method
      if (with_scale) {
        cur_transform =
            FindSimilarityTransformFrom3dCorrespondences(current, target)
                .template cast<T>();
      } else {
        cur_transform = FindRigidTransformFrom3dCorrespondences(current, target)
                            .template cast<T>();
      }

      accum_transform = cur_transform * accum_transform;

      // Update current points
      parallel_for(0u, current.size(), update_func, 1);
    }

    // Find coressponding points
    parallel_for(0u, current.size(), corresp_func, 1);

    // Compute loss
    prev_loss_ave = loss_ave;
    loss_ave = 0;
    for (const auto& r : corresp) {
      loss_ave += r[0].dist;
    }
    loss_ave /= static_cast<double>(corresp.size());

    // Update count
    iter++;

    output.transform_histry.push_back(accum_transform.template cast<double>());
    output.loss_histroty.push_back(loss_ave);

#ifdef UGU_DEBUG_ICP
    std::cout << iter << " " << loss_ave << std::endl;

    std::cout << "cur_transform" << std::endl;
    std::cout << cur_transform.translation() << std::endl;
    std::cout << cur_transform.rotation().matrix() << std::endl;
    std::cout << cur_transform.matrix() << std::endl << std::endl;

    std::cout << "accum_transform" << std::endl;
    std::cout << accum_transform.translation() << std::endl;
    std::cout << accum_transform.rotation().matrix() << std::endl;
    std::cout << accum_transform.matrix() << std::endl << std::endl;
#endif

  } while (!terminated());

  return true;
}

template <typename T>
bool RigidIcpPointToPointImpl(const std::vector<Eigen::Vector3<T>>& src,
                              const std::vector<Eigen::Vector3<T>>& dst,
                              const IcpTerminateCriteria& terminate_criteria,
                              IcpOutput& output, bool with_scale,
                              KdTreePtr<T, 3> kdtree = nullptr) {
  return RigidIcpImpl(src, dst, {}, terminate_criteria, output, with_scale,
                      IcpLossType::kPointToPoint, kdtree, nullptr);
}

template <typename T>
bool RigidIcpPointToPlaneImpl(const std::vector<Eigen::Vector3<T>>& src_points,
                              const std::vector<Eigen::Vector3<T>>& dst_points,
                              const std::vector<Eigen::Vector3i>& dst_faces,
                              const IcpTerminateCriteria& terminate_criteria,
                              IcpOutput& output, bool with_scale,
                              CorrespFinderPtr corresp_finder = nullptr) {
  return RigidIcpImpl(src_points, dst_points, dst_faces, terminate_criteria,
                      output, with_scale, IcpLossType::kPointToPlane,
                      KdTreePtr<T, 3>(nullptr), corresp_finder);
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
#ifdef UGU_USE_OWN_UMEYAMA
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
#else
  T = Eigen::umeyama(src.transpose(), dst.transpose(), true);
  Eigen::Affine3d tmp;
  tmp.matrix() = T;
  R = tmp.rotation();
  t = tmp.translation();
  double Sx = T.block<3, 1>(0, 0).norm();
  double Sy = T.block<3, 1>(0, 1).norm();
  double Sz = T.block<3, 1>(0, 2).norm();
  scale = Eigen::MatrixXd::Identity(R.rows(), R.cols());
  scale(0, 0) = Sx;
  scale(1, 1) = Sy;
  scale(2, 2) = Sz;
  return true;
#endif
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

bool RigidIcpPointToPlane(const std::vector<Eigen::Vector3f>& src_points,
                          const std::vector<Eigen::Vector3f>& dst_points,
                          const std::vector<Eigen::Vector3i>& dst_faces,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          CorrespFinderPtr corresp_finder) {
  return RigidIcpPointToPlaneImpl(src_points, dst_points, dst_faces,
                                  terminate_criteria, output, with_scale,
                                  corresp_finder);
}

bool RigidIcpPointToPlane(const std::vector<Eigen::Vector3d>& src_points,
                          const std::vector<Eigen::Vector3d>& dst_points,
                          const std::vector<Eigen::Vector3i>& dst_faces,
                          const IcpTerminateCriteria& terminate_criteria,
                          IcpOutput& output, bool with_scale,
                          CorrespFinderPtr corresp_finder) {
  return RigidIcpPointToPlaneImpl(src_points, dst_points, dst_faces,
                                  terminate_criteria, output, with_scale,
                                  corresp_finder);
}

bool RigidIcp(const Mesh& src, const Mesh& dst, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria, IcpOutput& output,
              bool with_scale, KdTreePtr<float, 3> kdtree,
              CorrespFinderPtr corresp_finder) {
  if (loss_type == IcpLossType::kPointToPoint) {
    return RigidIcpPointToPoint(src.vertices(), dst.vertices(),
                                terminate_criteria, output, with_scale, kdtree);
  } else if (loss_type == IcpLossType::kPointToPlane) {
    return RigidIcpPointToPlane(src.vertices(), dst.vertices(),
                                dst.vertex_indices(), terminate_criteria,
                                output, with_scale, corresp_finder);
  }
  throw std::runtime_error("Not implemented");
}

}  // namespace ugu