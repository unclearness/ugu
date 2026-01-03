/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/registration/rigid.h"

namespace {
using namespace ugu;

template <typename Derived, typename OtherDerived>
void my_umeyama(const Eigen::MatrixBase<Derived>& src,
                const Eigen::MatrixBase<OtherDerived>& dst,
                typename Eigen::internal::umeyama_transform_matrix_type<
                    Derived, OtherDerived>::type& T_similarity,
                typename Eigen::internal::umeyama_transform_matrix_type<
                    Derived, OtherDerived>::type& T_rigid) {
  using namespace Eigen;
  typedef typename internal::umeyama_transform_matrix_type<
      Derived, OtherDerived>::type TransformationMatrixType;
  typedef typename internal::traits<TransformationMatrixType>::Scalar Scalar_;
  typedef typename NumTraits<Scalar_>::Real RealScalar_;

  EIGEN_STATIC_ASSERT(!NumTraits<Scalar_>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL)
  EIGEN_STATIC_ASSERT(
      (internal::is_same<
          Scalar_, typename internal::traits<OtherDerived>::Scalar>::value),
      YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  enum {
    Dimension = internal::min_size_prefer_dynamic(Derived::RowsAtCompileTime,
                                              OtherDerived::RowsAtCompileTime)
  };

  typedef Matrix<Scalar_, Dimension, 1> VectorType;
  typedef Matrix<Scalar_, Dimension, Dimension> MatrixType;
  typedef typename internal::plain_matrix_type_row_major<Derived>::type
      RowMajorMatrixType;

  const Index m = src.rows();  // dimension
  const Index n = src.cols();  // number of measurements

  // required for demeaning ...
  const RealScalar_ one_over_n = RealScalar_(1) / static_cast<RealScalar_>(n);

  // computation of mean
  const VectorType src_mean = src.rowwise().sum() * one_over_n;
  const VectorType dst_mean = dst.rowwise().sum() * one_over_n;

  // demeaning of src and dst points
  const RowMajorMatrixType src_demean = src.colwise() - src_mean;
  const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

  // Eq. (36)-(37)
  const Scalar_ src_var = src_demean.rowwise().squaredNorm().sum() * one_over_n;

  // Eq. (38)
  const MatrixType sigma = one_over_n * dst_demean * src_demean.transpose();

  JacobiSVD<MatrixType> svd(sigma, ComputeFullU | ComputeFullV);

  // Initialize the resulting transformation with an identity matrix...
  TransformationMatrixType Rt =
      TransformationMatrixType::Identity(m + 1, m + 1);

  // Eq. (39)
  VectorType S = VectorType::Ones(m);

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(m - 1) = -1;

  // Eq. (40) and (43)
  Rt.block(0, 0, m, m).noalias() =
      svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

  T_similarity = Rt;
  // Eq. (42)
  const Scalar_ c = Scalar_(1) / src_var * svd.singularValues().dot(S);

  // Eq. (41)
  T_similarity.col(m).head(m) = dst_mean;
  T_similarity.col(m).head(m).noalias() -=
      c * Rt.topLeftCorner(m, m) * src_mean;
  T_similarity.block(0, 0, m, m) *= c;

  T_rigid = Rt;
  T_rigid.col(m).head(m) = dst_mean;
  T_rigid.col(m).head(m).noalias() -= Rt.topLeftCorner(m, m) * src_mean;
}

struct IcpCorresp {
  double dist = -1.0;
  bool valid = false;
};

template <typename T>
bool RigidIcpImpl(const std::vector<Eigen::Vector3<T>>& src_points,
                  const std::vector<Eigen::Vector3<T>>& dst_points,
                  const std::vector<Eigen::Vector3<T>>& src_normals,
                  const std::vector<Eigen::Vector3<T>>& dst_normals,
                  const std::vector<Eigen::Vector3i>& dst_faces,
                  const IcpCorrespType& corresp_type,
                  const IcpLossType& loss_type,
                  const IcpTerminateCriteria& terminate_criteria,
                  const IcpCorrespCriteria& corresp_criateria,
                  IcpOutput& output, bool with_scale,
                  KdTreePtr<T, 3> kdtree = nullptr,
                  CorrespFinderPtr corresp_finder = nullptr,
                  int num_theads = -1, IcpCallbackFunc callback = nullptr,
                  uint32_t nn_num = 10) {
  if (corresp_type == IcpCorrespType::kPointToPoint &&
      (kdtree == nullptr || !kdtree->IsInitialized())) {
    kdtree = GetDefaultKdTree<T, 3>();
    kdtree->SetData(dst_points);
    if (!kdtree->Build()) {
      return false;
    }
  }

  if (corresp_type == IcpCorrespType::kPointToPlane &&
      (corresp_finder == nullptr)) {
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
  std::vector<Eigen::Vector3<T>> target(current.size());
  std::vector<Eigen::Vector3<T>> target_normals(target.size());
  std::vector<IcpCorresp> corresp(current.size());

  std::vector<Eigen::Vector3<T>> current_(current.size());
  std::vector<Eigen::Vector3<T>> target_(target.size());
  std::vector<Eigen::Vector3<T>> target_normals_(target.size());

  // Normal and distance threshold
  auto is_valid = [&](const Eigen::Vector3<T>& src_n,
                      const Eigen::Vector3<T>& dst_n, const double& dist) {
    if (0.f < corresp_criateria.normal_th) {
      double dot =
          static_cast<double>(std::clamp(src_n.dot(dst_n), T(-1), T(1)));
      if (corresp_criateria.normal_th < std::acos(dot)) {
        return false;
      }
    }
    if (0.f < corresp_criateria.dist_th) {
      if (corresp_criateria.dist_th < dist) {
        return false;
      }
    }
    return true;
  };

  auto point2point_corresp_func = [&](size_t idx) {
    if (corresp_criateria.test_nearest) {
      ugu::KdTreeSearchResults res = kdtree->SearchNn(current[idx]);
      corresp[idx].valid =
          is_valid(src_normals[idx], dst_normals[res[0].index], res[0].dist);
      corresp[idx].dist = res[0].dist;
      target[idx] = dst_points[res[0].index];
      target_normals[idx] = dst_normals[res[0].index];
    } else {
      ugu::KdTreeSearchResults res = kdtree->SearchKnn(current[idx], nn_num);
      for (const auto& r : res) {
        corresp[idx].valid =
            is_valid(src_normals[idx], dst_normals[r.index], r.dist);
        corresp[idx].dist = r.dist;
        target[idx] = dst_points[r.index];
        target_normals[idx] = dst_normals[r.index];
        if (corresp[idx].valid) {
          break;
        }
      }
    }
  };

  auto point2plane_corresp_func = [&](size_t idx) {
    if (corresp_criateria.test_nearest) {
      Corresp c = corresp_finder->Find(current[idx].template cast<float>());
      corresp[idx].valid =
          is_valid(src_normals[idx], c.n.cast<T>(), c.abs_dist);
      corresp[idx].dist = c.abs_dist;
      target[idx] = c.p.cast<T>();
      target_normals[idx] = c.n.cast<T>();
    } else {
      uint32_t nn_num_ = nn_num;
      std::vector<Corresp> cs =
          corresp_finder->FindKnn(current[idx].template cast<float>(), nn_num_);
      for (const auto& c : cs) {
        corresp[idx].valid =
            is_valid(src_normals[idx], c.n.cast<T>(), c.abs_dist);
        corresp[idx].dist = c.abs_dist;
        target[idx] = c.p.cast<T>();
        target_normals[idx] = c.n.cast<T>();
        if (corresp[idx].valid) {
          break;
        }
      }
    }
  };

  std::function<void(size_t)> corresp_func = nullptr;
  if (corresp_type == IcpCorrespType::kPointToPoint) {
    corresp_func = point2point_corresp_func;
  } else if (corresp_type == IcpCorrespType::kPointToPlane) {
    corresp_func = point2plane_corresp_func;
  }

  auto update_func = [&](size_t idx) {
    current[idx] = accum_transform * src_points[idx];
  };
  do {
    Eigen::Transform<T, 3, Eigen::Affine> cur_transform =
        Eigen::Transform<T, 3, Eigen::Affine>::Identity();
    if (iter > 0) {
      current_.clear();
      target_.clear();
      target_normals_.clear();
      for (size_t idx = 0; idx < current.size(); idx++) {
        const auto& c = corresp[idx];
        if (c.valid) {
          current_.push_back(current[idx]);
          target_.push_back(target[idx]);
          target_normals_.push_back(target_normals[idx]);
        }
      }
      float inlier_ratio = static_cast<float>(current_.size()) /
                           static_cast<float>(current.size());
      LOGD("Inliers : %d / %d, %f\n", current_.size(), current.size(),
           inlier_ratio);

      if (current_.size() < 3) {
        LOGW("The number of valid correspondences is less than 3. Stop...\n");
        break;
      }

      if (loss_type == IcpLossType::kPointToPoint) {
        // Solve rotation, translation and scale by Umeyama closed-form method
        if (with_scale) {
          cur_transform =
              FindSimilarityTransformFrom3dCorrespondences(current_, target_)
                  .template cast<T>();
        } else {
          cur_transform =
              FindRigidTransformFrom3dCorrespondences(current_, target_)
                  .template cast<T>();
        }
      } else if (loss_type == IcpLossType::kPointToPlane) {
        cur_transform = FindRigidTransformFrom3dCorrespondencesWithNormals(
            current_, target_, target_normals_);
      } else {
        throw std::runtime_error("Not supported");
      }

      accum_transform = cur_transform * accum_transform;

      // Update current points
      parallel_for(0u, current.size(), update_func, num_theads);
    }

    // Find coressponding points
    parallel_for(0u, current.size(), corresp_func, num_theads);

    // Compute loss
    prev_loss_ave = loss_ave;
    loss_ave = 0;
    for (const auto& r : corresp) {
      loss_ave += r.dist;
    }
    loss_ave /= static_cast<double>(corresp.size());

    // Update count
    iter++;

    output.transform_histry.push_back(accum_transform.template cast<double>());
    output.loss_histroty.push_back(loss_ave);

    if (callback != nullptr) {
      callback(terminate_criteria, output);
    }

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

template <typename Scalar>
void DecomposeRtsImpl(const Eigen::Transform<Scalar, 3, Eigen::Affine>& T,
                      Eigen::Matrix<Scalar, 3, 3>& R,
                      Eigen::Matrix<Scalar, 3, 1>& t,
                      Eigen::Matrix<Scalar, 3, 1>& s) {
  R = T.rotation();
  t = T.translation();
  auto mat = T.matrix();
  s[0] = mat.template block<3, 1>(0, 0).norm();
  s[1] = mat.template block<3, 1>(0, 1).norm();
  s[2] = mat.template block<3, 1>(0, 2).norm();
}

using namespace Eigen;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using MatrixXd6x = Eigen::MatrixXd;
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

struct CorrespondenceGIcp {
  Vector3d src;
  Matrix3d src_cov;
  Vector3d dst;
  Matrix3d dst_cov;
  CorrespondenceGIcp() {
    src = Vector3d::Constant(std::numeric_limits<double>::max());
    src_cov = Matrix3d::Constant(std::numeric_limits<double>::max());
    dst = Vector3d::Constant(std::numeric_limits<double>::max());
    dst_cov = Matrix3d::Constant(std::numeric_limits<double>::max());
  }
};

Eigen::Matrix<double, 3, 6> ComputeJacobian(const Eigen::Vector3d& p,
                                            const Eigen::Matrix3d& R) {
  Matrix3d skew_p;
  skew_p << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;

  Matrix<double, 3, 6> J;
  J.block<3, 3>(0, 0) = -R * skew_p;
#if 1
  // R is correct in this implementation
  J.block<3, 3>(0, 3) = R;
#else
  J.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
#endif
  return J;
}

TransformationGIcp MinimizeLossGuassNewton(
    const std::vector<CorrespondenceGIcp>& correspondences,
    const TransformationGIcp& initial_T, uint32_t max_iterations = 100,
    double tolerance = 1e-6) {
  TransformationGIcp T = initial_T;
  double total_error = 0.0;

  for (uint32_t iter = 0; iter < max_iterations; ++iter) {
    // Init
    MatrixXd H = MatrixXd::Zero(6, 6);
    Vector6d b = Vector6d::Zero();
    total_error = 0.0;

#if 0
    // Parallel version but slower...
    std::vector<MatrixXd> Hs(correspondences.size());
    std::vector<Vector6d> bs(correspondences.size());
    std::vector<double> errors(correspondences.size());
#pragma omp parallel for
    for (int64_t corr_idx = 0;
         corr_idx < static_cast<int64_t>(correspondences.size()); corr_idx++) {
      const auto& corr = correspondences[corr_idx];

      // Compute resigual: r_i = T*p_i - q_i
      Vector3d src_transed = T.Transform(corr.src);
      Vector3d r = src_transed - corr.dst;

      // Compute Jacobian: J_i
      Matrix<double, 3, 6> J_i = ComputeJacobian(corr.src, T.R);

      // Accumulate corresponding covariances ƒ°_i = R * ƒ°_src_i * R^T + ƒ°_dst_i
      Matrix3d sigma_src_transformed = T.R * corr.src_cov * T.R.transpose();
      Matrix3d sigma_i = sigma_src_transformed + corr.dst_cov;

#if 0
      // Add small value to avoid singular matrix
      double eplison = 1e-2;
      sigma_i += eplison * Eigen::Matrix3d::Identity();
#endif

      // Inverse of ƒ°_i
      Matrix3d W_i = sigma_i.inverse();

      // Accumulate H, approximated Hessian
      Hs[corr_idx] = J_i.transpose() * W_i * J_i;

      // Accumulate b, gradient vector
      bs[corr_idx] = J_i.transpose() * W_i * r;

      // Accumulated error
      errors[corr_idx] = r.transpose() * W_i * r;
    }

    for (int64_t corr_idx = 0;
         corr_idx < static_cast<int64_t>(correspondences.size()); corr_idx++) {
      H += Hs[corr_idx];
      b += bs[corr_idx];
      total_error += errors[corr_idx];
    }

#else
    for (const auto& corr : correspondences) {
      // Compute resigual: r_i = T*p_i - q_i
      Vector3d src_transed = T.Transform(corr.src);
      Vector3d r = src_transed - corr.dst;

      // Compute Jacobian: J_i
      Matrix<double, 3, 6> J_i = ComputeJacobian(corr.src, T.R);

      // Accumulate corresponding covariances ƒ°_i = R * ƒ°_src_i * R^T + ƒ°_dst_i
      Matrix3d sigma_src_transformed = T.R * corr.src_cov * T.R.transpose();
      Matrix3d sigma_i = sigma_src_transformed + corr.dst_cov;

#if 0
      // Add small value to avoid singular matrix
      double eplison = 1e-2;
      sigma_i += eplison * Eigen::Matrix3d::Identity();
#endif

      // Inverse of ƒ°_i
      Matrix3d W_i = sigma_i.inverse();

      // Accumulate H, approximated Hessian
      H += J_i.transpose() * W_i * J_i;

      // Accumulate b, gradient vector
      b += J_i.transpose() * W_i * r;

      // Accumulated error
      total_error += r.transpose() * W_i * r;
    }
#endif

    // The equation to be solved: H * delta = -b
    Vector6d delta = H.ldlt().solve(-b);

    // Update
    T.Update(delta);

    if (delta.norm() < tolerance) {
      // std::cout << "Converged at iteration " << iter + 1 << std::endl;
      break;
    }

    // std::cout << "Iteration " << iter + 1 << ": Error = " << total_error
    //           << ", Delta norm = " << delta.norm() << std::endl;
  }
  T.error = total_error;

  return T;
}

Eigen::Matrix3d ComputeCovariance(const std::vector<Eigen::Vector3d>& data) {
#if 1
  // Compute stride, considering padding for alignment
  // For example, if Eigen::Vector3d is aligned, the stride may be 4 instead of
  // 3
  const ptrdiff_t stride = sizeof(Eigen::Vector3d) / sizeof(double);

  // With strieded Eigen::Map, map vector onto Eigen::Matrix without copying
  Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 0,
             Eigen::OuterStride<>>
      mat(reinterpret_cast<const double*>(data.data()), 3, data.size(),
          Eigen::OuterStride<>(stride));

  // Compute the mean of each dimension (mean of each row)
  Eigen::Vector3d mean = mat.rowwise().mean();

  // Centering by subtracting the mean from each sample
  Eigen::Matrix<double, 3, Eigen::Dynamic> centered = mat.colwise() - mean;

  // unbiased estimation of covariance matrix
  Eigen::Matrix3d covariance =
      (centered * centered.transpose()) / static_cast<double>(data.size() - 1);

  return covariance;
#else

  size_t n = data.size();
  if (n == 0) {
    throw std::runtime_error("");
  }
  // Compute mean vector
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& vec : data) {
    mean += vec;
  }
  mean /= static_cast<double>(n);

  // Initialize covariance matrix
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  // Compute covariance matrix using deviation vectors
  for (const auto& vec : data) {
    Eigen::Vector3d deviation = vec - mean;
    covariance += deviation * deviation.transpose();
  }

  // Divide by n-1 for sample covariance matrix (unbiased variance)
  covariance /= static_cast<double>(n - 1);

  return covariance;
#endif
}

TransformationGIcpHistory GeneralizedIcpImpl(
    const std::vector<Eigen::Vector3d>& src_points,
    const std::vector<Eigen::Vector3d>& dst_points,
    const std::vector<Eigen::Matrix3d>& src_covs_,
    const std::vector<Eigen::Matrix3d>& dst_covs_,
    const IcpTerminateCriteria& terminate_criteria,
    const IcpCorrespCriteria& corresp_criateria,
    KdTreePtr<double, 3> kdtree = nullptr, int num_theads = -1,
    IcpCallbackFunc callback = nullptr, uint32_t nn_num = 20,
    uint32_t cov_nn_num = 20, uint32_t minimize_max_iterations = 100,
    double minimize_tolerance = 1e-6) {
  // TODO; Use num_theads
  (void)num_theads;

  TransformationGIcpHistory history;

  if (kdtree == nullptr || !kdtree->IsInitialized()) {
    kdtree = GetDefaultKdTree<double, 3>();
    kdtree->SetData(dst_points);
    if (!kdtree->Build()) {
      return history;
    }
  }

  // Compuate covariance matices
  std::vector<Eigen::Matrix3d> src_covs = src_covs_;
  if (src_covs.size() != src_points.size()) {
    KdTreePtr<double, 3> src_kdtree = GetDefaultKdTree<double, 3>();
    src_kdtree->SetData(src_points);
    src_kdtree->Build();
    src_covs.resize(src_points.size());
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(src_points.size()); i++) {
      KdTreeSearchResults res =
          src_kdtree->SearchKnn(src_points[i], cov_nn_num);
      std::vector<Eigen::Vector3d> nn_points;
      for (const auto& r : res) {
        if (static_cast<int64_t>(r.index) == i) {
          // Ignore itself
          continue;
        }
        nn_points.push_back(src_points[r.index]);
      }
      src_covs[i] = ComputeCovariance(nn_points);
    }
  }

  std::vector<Eigen::Matrix3d> dst_covs = dst_covs_;
  if (dst_covs.size() != dst_points.size()) {
    dst_covs.resize(dst_points.size());
#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(dst_points.size()); i++) {
      KdTreeSearchResults res = kdtree->SearchKnn(dst_points[i], cov_nn_num);
      std::vector<Eigen::Vector3d> nn_points;
      for (const auto& r : res) {
        if (static_cast<int64_t>(r.index) == i) {
          // Ignore itself
          continue;
        }
        nn_points.push_back(dst_points[r.index]);
      }
      dst_covs[i] = ComputeCovariance(nn_points);
    }
  }

  int iter = 1;
  while (true) {
    // 1. Correspondence Estimation
    std::vector<size_t> corresp_indices(src_points.size(),
                                        std::numeric_limits<size_t>::max());

#pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(src_points.size()); ++i) {
      Eigen::Vector3d src_transed =
          history.accumlated.R * src_points[i] + history.accumlated.t;
      KdTreeSearchResults res = kdtree->SearchKnn(src_transed, nn_num);

      for (size_t j = 0; j < res.size(); j++) {
        if (corresp_criateria.dist_th > 0 &&
            res[j].dist > corresp_criateria.dist_th) {
          continue;
        }
        corresp_indices[i] = res[j].index;
        break;
      }
    }

    std::vector<CorrespondenceGIcp> correspondences;
    correspondences.reserve(corresp_indices.size());
    for (size_t i = 0; i < corresp_indices.size(); ++i) {
      if (corresp_indices[i] != std::numeric_limits<size_t>::max()) {
        CorrespondenceGIcp c;
        c.src = src_points[i];
        c.src_cov = src_covs[i];
        c.dst = dst_points[corresp_indices[i]];
        c.dst_cov = dst_covs[corresp_indices[i]];
        correspondences.push_back(c);
      }
    }

    // 2. Gauss-Newton Optimization
    TransformationGIcp initial_T = history.accumlated;
    TransformationGIcp T =
        MinimizeLossGuassNewton(correspondences, initial_T,
                                minimize_max_iterations, minimize_tolerance);

    history.Update(T);

    // 3. Check convergence
    if (T.R.isIdentity() && T.t.isZero()) {
      break;
    }

    if (iter >= terminate_criteria.iter_max) {
      break;
    }

    if (history.history.back().error < terminate_criteria.loss_min) {
      break;
    }

    if (1 < history.history.size() &&
        std::abs(history.history.back().error -
                 history.history[history.history.size() - 2].error) <
            terminate_criteria.loss_eps) {
      break;
    }

    iter++;
  }

  return history;
}

}  // namespace

// #define UGU_USE_UMEYAMA_FOR_RIGID

namespace ugu {

// FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS
// http://nghiaho.com/?page_id=671
Eigen::Affine3d FindRigidTransformFrom3dCorrespondences(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst) {
#ifdef UGU_USE_UMEYAMA_FOR_RIGID

  Eigen::MatrixXd src_(src.size(), 3);
  for (size_t i = 0; i < src.size(); i++) {
    src_.row(i) = src[i];
  }
  Eigen::MatrixXd dst_(dst.size(), 3);
  for (size_t i = 0; i < dst.size(); i++) {
    dst_.row(i) = dst[i];
  }

  Eigen::MatrixXd R_sim, R_rigid;
  Eigen::MatrixXd t_sim, t_rigid;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd T_sim, T_rigid;
  bool ret = FindSimilarityTransformFromPointCorrespondences(
      src_, dst_, R_sim, t_sim, scale, T_sim, R_rigid, t_rigid, T_rigid);

  Eigen::Affine3d T = Eigen::Translation3d(t_rigid) * R_rigid;
  return T;
#else
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

  assert(std::abs(std::abs(det) - 1.0) < 0.001);

  // https://github.com/nghiaho12/rigid_transform_3D/blob/32c95173966488926ccabc67c5ea70c03f63df7a/rigid_transform_3D.py#L44
  if (det < 0) {
    Eigen::Matrix3d V = svd.matrixV();

    V.col(2) *= -1.0;

    R = V * svd.matrixU().transpose();
    det = R.determinant();
  }

  assert(std::abs(det - 1.0) < 0.001);

  Eigen::Vector3d t = dst_centroid - R * src_centroid;

  Eigen::Affine3d T = Eigen::Translation3d(t) * R;
  return T;

#endif
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
  Eigen::MatrixXd R, R_rigid;
  Eigen::MatrixXd t, t_rigid;
  Eigen::MatrixXd scale;
  Eigen::MatrixXd T, T_rigid;
  bool ret = FindSimilarityTransformFromPointCorrespondences(
      src, dst, R, t, scale, T, R_rigid, t_rigid, T_rigid);
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
    const Eigen::MatrixXd& src, const Eigen::MatrixXd& dst,
    Eigen::MatrixXd& R_similarity, Eigen::MatrixXd& t_similarity,
    Eigen::MatrixXd& scale, Eigen::MatrixXd& T_similarity,
    Eigen::MatrixXd& R_rigid, Eigen::MatrixXd& t_rigid,
    Eigen::MatrixXd& T_rigid) {
  Eigen::MatrixXd src_t = src.transpose();
  Eigen::MatrixXd dst_t = dst.transpose();
  my_umeyama(src_t, dst_t, T_similarity, T_rigid);

  Eigen::Affine3d tmp_similarity;
  tmp_similarity.matrix() = T_similarity;
  R_similarity = tmp_similarity.rotation();
  t_similarity = tmp_similarity.translation();
  scale = Eigen::MatrixXd::Identity(R_similarity.rows(), R_similarity.cols());
  for (Eigen::Index i = 0; i < R_similarity.cols(); i++) {
    scale(i, i) = T_similarity.block(0, i, R_similarity.rows(), 1).norm();
  }

  Eigen::Affine3d tmp_rigid;
  tmp_rigid.matrix() = T_rigid;
  R_rigid = tmp_rigid.rotation();
  t_rigid = tmp_rigid.translation();

  return true;
}

Eigen::Affine3f FindRigidTransformFrom3dCorrespondencesWithNormals(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst,
    const std::vector<Eigen::Vector3f>& dst_normals) {
  std::vector<Eigen::Vector3d> src_d, dst_d, dst_normals_d;
  auto to_double = [](const std::vector<Eigen::Vector3f>& fvec,
                      std::vector<Eigen::Vector3d>& dvec) {
    std::transform(fvec.begin(), fvec.end(), std::back_inserter(dvec),
                   [](const Eigen::Vector3f& f) { return f.cast<double>(); });
  };

  to_double(src, src_d);
  to_double(dst, dst_d);
  to_double(dst_normals, dst_normals_d);
  return FindRigidTransformFrom3dCorrespondencesWithNormals(src_d, dst_d,
                                                            dst_normals_d)
      .cast<float>();
}

Eigen::Affine3d FindRigidTransformFrom3dCorrespondencesWithNormals(
    const std::vector<Eigen::Vector3d>& src,
    const std::vector<Eigen::Vector3d>& dst,
    const std::vector<Eigen::Vector3d>& dst_normals) {
  // Linearize rotation matrix with small theta
  // https://tatsy.github.io/programming-for-beginners/cpp/point-to-plane-icp/

  if (src.size() < 3 || src.size() != dst.size() ||
      src.size() != dst_normals.size()) {
    throw std::runtime_error("Invalid input");
  }

  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Vector<double, 6>;
  Matrix6d A = Matrix6d::Zero(6, 6);
  Vector6d b = Vector6d::Zero(6);
  for (size_t i = 0; i < src.size(); i++) {
    Eigen::Vector<double, 6> a = Eigen::Vector<double, 6>::Zero();
    a.block(0, 0, 3, 1) = src[i].cross(dst_normals[i]);
    a.block(3, 0, 3, 1) = dst_normals[i];

    A += a * a.transpose();
    b += a * ((dst[i] - src[i]).dot(dst_normals[i]));
  }

  Eigen::ColPivHouseholderQR<Matrix6d> dec(A);
  Vector6d x = dec.solve(b);

  Eigen::Vector3d rodrigues_vec = x.head(3);
  Eigen::Vector3d t = x.tail(3);

  Eigen::Matrix3d R =
      Eigen::AngleAxisd(rodrigues_vec.norm(), rodrigues_vec.normalized())
          .toRotationMatrix();

  Eigen::Affine3d T = Eigen::Translation3d(t) * R;
  return T;
}

void DecomposeRts(const Eigen::Affine3f& T, Eigen::Matrix3f& R,
                  Eigen::Vector3f& t, Eigen::Vector3f& s) {
  DecomposeRtsImpl(T, R, t, s);
}

void DecomposeRts(const Eigen::Affine3d& T, Eigen::Matrix3d& R,
                  Eigen::Vector3d& t, Eigen::Vector3d& s) {
  DecomposeRtsImpl(T, R, t, s);
}

bool RigidIcp(const std::vector<Eigen::Vector3f>& src_points,
              const std::vector<Eigen::Vector3f>& dst_points,
              const std::vector<Eigen::Vector3f>& src_normals,
              const std::vector<Eigen::Vector3f>& dst_normals,
              const std::vector<Eigen::Vector3i>& dst_faces,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale, KdTreePtr<float, 3> kdtree,
              CorrespFinderPtr corresp_finder, int num_theads,
              IcpCallbackFunc callback, uint32_t approx_nn_num) {
  return RigidIcpImpl(src_points, dst_points, src_normals, dst_normals,
                      dst_faces, corresp_type, loss_type, terminate_criteria,
                      corresp_criteria, output, with_scale, kdtree,
                      corresp_finder, num_theads, callback, approx_nn_num);
}

bool RigidIcp(const std::vector<Eigen::Vector3d>& src_points,
              const std::vector<Eigen::Vector3d>& dst_points,
              const std::vector<Eigen::Vector3d>& src_normals,
              const std::vector<Eigen::Vector3d>& dst_normals,
              const std::vector<Eigen::Vector3i>& dst_faces,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale, KdTreePtr<double, 3> kdtree,
              CorrespFinderPtr corresp_finder, int num_theads,
              IcpCallbackFunc callback, uint32_t approx_nn_num) {
  return RigidIcpImpl(src_points, dst_points, src_normals, dst_normals,
                      dst_faces, corresp_type, loss_type, terminate_criteria,
                      corresp_criteria, output, with_scale, kdtree,
                      corresp_finder, num_theads, callback, approx_nn_num);
}

bool RigidIcp(const Mesh& src, const Mesh& dst,
              const IcpCorrespType& corresp_type, const IcpLossType& loss_type,
              const IcpTerminateCriteria& terminate_criteria,
              const IcpCorrespCriteria& corresp_criteria, IcpOutput& output,
              bool with_scale, KdTreePtr<float, 3> kdtree,
              CorrespFinderPtr corresp_finder, int num_theads,
              IcpCallbackFunc callback, uint32_t approx_nn_num) {
  return RigidIcpImpl(src.vertices(), dst.vertices(), src.normals(),
                      dst.normals(), dst.vertex_indices(), corresp_type,
                      loss_type, terminate_criteria, corresp_criteria, output,
                      with_scale, kdtree, corresp_finder, num_theads, callback,
                      approx_nn_num);
}

TransformationGIcp::TransformationGIcp() {
  R = Eigen::Matrix3d::Identity();
  t = Eigen::Vector3d::Zero();
  error = 0;
}

TransformationGIcp::~TransformationGIcp() {}

Eigen::Vector3d TransformationGIcp::Transform(const Eigen::Vector3d& p) const {
  return R * p + t;
}

void TransformationGIcp::Update(const Eigen::Matrix<double, 6, 1>& delta) {
  Eigen::Vector3d delta_rot = delta.head<3>();
  Eigen::Vector3d delta_trans = delta.tail<3>();

  double angle = delta_rot.norm();
  Eigen::Matrix3d delta_R;
  if (angle != 0.0) {
    Eigen::Vector3d axis = delta_rot.normalized();
    delta_R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
  } else {
    delta_R = Eigen::Matrix3d::Identity();
  }
  R = delta_R * R;
  t += delta_trans;
}

TransformationGIcpHistory::TransformationGIcpHistory() {
  accumlated = TransformationGIcp();
}
TransformationGIcpHistory::~TransformationGIcpHistory() {}
void TransformationGIcpHistory::Update(const TransformationGIcp& latest) {
  history.push_back(latest);
  accumlated = latest;
}

TransformationGIcpHistory GeneralizedIcp(
    const std::vector<Eigen::Vector3f>& src_points,
    const std::vector<Eigen::Vector3f>& dst_points,
    const std::vector<Eigen::Matrix3f>& src_covs,
    const std::vector<Eigen::Matrix3f>& dst_covs,
    const IcpTerminateCriteria& terminate_criteria,
    const IcpCorrespCriteria& corresp_criateria, KdTreePtr<double, 3> kdtree,
    int num_theads, IcpCallbackFunc callback, uint32_t nn_num,
    uint32_t cov_nn_num, uint32_t minimize_max_iterations,
    double minimize_tolerance) {
  std::vector<Eigen::Vector3d> src_points_d(src_points.size());
  std::vector<Eigen::Vector3d> dst_points_d(dst_points.size());
  std::vector<Eigen::Matrix3d> src_covs_d(src_covs.size());
  std::vector<Eigen::Matrix3d> dst_covs_d(dst_covs.size());

  for (size_t i = 0; i < src_points.size(); i++) {
    src_points_d[i] = src_points[i].cast<double>();
  }
  for (size_t i = 0; i < dst_points.size(); i++) {
    dst_points_d[i] = dst_points[i].cast<double>();
  }

  for (size_t i = 0; i < src_covs.size(); i++) {
    src_covs_d[i] = src_covs[i].cast<double>();
  }

  for (size_t i = 0; i < dst_covs.size(); i++) {
    dst_covs_d[i] = dst_covs[i].cast<double>();
  }

  return GeneralizedIcpImpl(src_points_d, dst_points_d, src_covs_d, dst_covs_d,
                            terminate_criteria, corresp_criateria, kdtree,
                            num_theads, callback, nn_num, cov_nn_num,
                            minimize_max_iterations, minimize_tolerance);
}

TransformationGIcpHistory GeneralizedIcp(
    const std::vector<Eigen::Vector3d>& src_points,
    const std::vector<Eigen::Vector3d>& dst_points,
    const std::vector<Eigen::Matrix3d>& src_covs,
    const std::vector<Eigen::Matrix3d>& dst_covs,
    const IcpTerminateCriteria& terminate_criteria,
    const IcpCorrespCriteria& corresp_criateria, KdTreePtr<double, 3> kdtree,
    int num_theads, IcpCallbackFunc callback, uint32_t nn_num,
    uint32_t cov_nn_num, uint32_t minimize_max_iterations,
    double minimize_tolerance) {
  return GeneralizedIcpImpl(src_points, dst_points, src_covs, dst_covs,
                            terminate_criteria, corresp_criateria, kdtree,
                            num_theads, callback, nn_num, cov_nn_num,
                            minimize_max_iterations, minimize_tolerance);
}

}  // namespace ugu