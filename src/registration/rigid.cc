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
  typedef typename internal::traits<TransformationMatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL)
  EIGEN_STATIC_ASSERT(
      (internal::is_same<
          Scalar, typename internal::traits<OtherDerived>::Scalar>::value),
      YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  enum {
    Dimension = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime,
                                              OtherDerived::RowsAtCompileTime)
  };

  typedef Matrix<Scalar, Dimension, 1> VectorType;
  typedef Matrix<Scalar, Dimension, Dimension> MatrixType;
  typedef typename internal::plain_matrix_type_row_major<Derived>::type
      RowMajorMatrixType;

  const Index m = src.rows();  // dimension
  const Index n = src.cols();  // number of measurements

  // required for demeaning ...
  const RealScalar one_over_n = RealScalar(1) / static_cast<RealScalar>(n);

  // computation of mean
  const VectorType src_mean = src.rowwise().sum() * one_over_n;
  const VectorType dst_mean = dst.rowwise().sum() * one_over_n;

  // demeaning of src and dst points
  const RowMajorMatrixType src_demean = src.colwise() - src_mean;
  const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

  // Eq. (36)-(37)
  const Scalar src_var = src_demean.rowwise().squaredNorm().sum() * one_over_n;

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
  const Scalar c = Scalar(1) / src_var * svd.singularValues().dot(S);

  // Eq. (41)
  T_similarity.col(m).head(m) = dst_mean;
  T_similarity.col(m).head(m).noalias() -=
      c * Rt.topLeftCorner(m, m) * src_mean;
  T_similarity.block(0, 0, m, m) *= c;

  T_rigid = Rt;
  T_rigid.col(m).head(m) = dst_mean;
  T_rigid.col(m).head(m).noalias() -= Rt.topLeftCorner(m, m) * src_mean;
}

template <typename T>
bool RigidIcpImpl(const std::vector<Eigen::Vector3<T>>& src_points,
                  const std::vector<Eigen::Vector3<T>>& dst_points,
                  const std::vector<Eigen::Vector3i>& dst_faces,
                  const IcpTerminateCriteria& terminate_criteria,
                  IcpOutput& output, bool with_scale, IcpLossType loss_type,
                  KdTreePtr<T, 3> kdtree = nullptr,
                  CorrespFinderPtr corresp_finder = nullptr,
                  int num_theads = -1) {
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
  std::vector<Eigen::Vector3<T>> target(current.size());
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
      parallel_for(0u, current.size(), update_func, num_theads);
    }

    // Find coressponding points
    parallel_for(0u, current.size(), corresp_func, num_theads);

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

void DecomposeRts(const Eigen::Affine3f& T, Eigen::Matrix3f& R,
                  Eigen::Vector3f& t, Eigen::Vector3f& s) {
  DecomposeRtsImpl(T, R, t, s);
}

void DecomposeRts(const Eigen::Affine3d& T, Eigen::Matrix3d& R,
                  Eigen::Vector3d& t, Eigen::Vector3d& s) {
  DecomposeRtsImpl(T, R, t, s);
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