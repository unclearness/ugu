/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "visibility_tester.h"

#include <cassert>
#include <iterator>

#include "ugu/timer.h"

namespace {
void PrepareRay(nanort::Ray<float>* ray, const Eigen::Vector3f& camera_pos_w,
                const Eigen::Vector3f& ray_w) {
  const float kFar = 1.0e+30f;
  ray->min_t = 0.0001f;
  ray->max_t = kFar;

  // camera position in world coordinate
  ray->org[0] = camera_pos_w[0];
  ray->org[1] = camera_pos_w[1];
  ray->org[2] = camera_pos_w[2];

  // ray in world coordinate
  ray->dir[0] = ray_w[0];
  ray->dir[1] = ray_w[1];
  ray->dir[2] = ray_w[2];
}

void BilinearInterpolation(float x, float y, const ugu::Image3b& image,
                           Eigen::Vector3f* color) {
  int tex_pos_min[2] = {0, 0};
  int tex_pos_max[2] = {0, 0};
  tex_pos_min[0] = static_cast<int>(std::floor(x));
  tex_pos_min[1] = static_cast<int>(std::floor(y));
  tex_pos_max[0] = tex_pos_min[0] + 1;
  tex_pos_max[1] = tex_pos_min[1] + 1;

  float local_u = x - tex_pos_min[0];
  float local_v = y - tex_pos_min[1];

  for (int k = 0; k < 3; k++) {
    // bilinear interpolation of pixel color
    (*color)[k] = (1.0f - local_u) * (1.0f - local_v) *
                      image.at<ugu::Vec3b>(tex_pos_min[1], tex_pos_min[0])[k] +
                  local_u * (1.0f - local_v) *
                      image.at<ugu::Vec3b>(tex_pos_max[1], tex_pos_min[0])[k] +
                  (1.0f - local_u) * local_v *
                      image.at<ugu::Vec3b>(tex_pos_min[1], tex_pos_max[0])[k] +
                  local_u * local_v *
                      image.at<ugu::Vec3b>(tex_pos_max[1], tex_pos_max[0])[k];

    // assert(0.0f <= (*color)[k] && (*color)[k] <= 255.0f);
    if (0.0f > (*color)[k]) {
      (*color)[k] = 0.0f;
    } else if (255.0f < (*color)[k]) {
      (*color)[k] = 255.0f;
    }
  }
}

template <typename T>
float Median(const std::vector<T>& data) {
  assert(data.size() > 0);
  if (data.size() == 1) {
    return data[0];
  } else if (data.size() == 2) {
    return (data[0] + data[1]) * 0.5f;
  }

  std::vector<T> data_tmp;
  std::copy(data.begin(), data.end(), std::back_inserter(data_tmp));

  size_t n = data_tmp.size() / 2;
  if (data_tmp.size() % 2 == 0) {
    std::nth_element(data_tmp.begin(), data_tmp.begin() + n, data_tmp.end());
    return data_tmp[n];
  }

  std::nth_element(data_tmp.begin(), data_tmp.begin() + n + 1, data_tmp.end());
  return (data_tmp[n] + data_tmp[n + 1]) * 0.5f;
}

Eigen::Vector3f MedianColor(const std::vector<Eigen::Vector3f>& colors) {
  Eigen::Vector3f median;
  std::vector<std::vector<float>> ith_channel_list(3);
  for (const auto& color : colors) {
    for (int i = 0; i < 3; i++) {
      ith_channel_list[i].push_back(color[i]);
    }
  }
  for (int i = 0; i < 3; i++) {
    median[i] = Median(ith_channel_list[i]);
  }
  return median;
}

void NormalizeWeights(const std::vector<float>& weights,
                      std::vector<float>* normalized_weights) {
  assert(!weights.empty());
  std::copy(weights.begin(), weights.end(),
            std::back_inserter(*normalized_weights));
  float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum > 0.000001) {
    std::for_each(normalized_weights->begin(), normalized_weights->end(),
                  [&](float& n) { n /= sum; });
  } else {
    // if sum is too small, just set even weights
    float val = 1.0f / static_cast<float>(normalized_weights->size());
    std::fill(normalized_weights->begin(), normalized_weights->end(), val);
  }
}

template <typename T>
Eigen::Matrix<T, 3, 1> WeightedAverage(
    const std::vector<Eigen::Matrix<T, 3, 1>>& data,
    const std::vector<float>& weights) {
  assert(data.size() > 0);
  assert(data.size() == weights.size());

  std::vector<float> normalized_weights;
  NormalizeWeights(weights, &normalized_weights);

  double weighted_average[3];
  for (size_t i = 0; i < data.size(); i++) {
    weighted_average[0] += (data[i][0] * normalized_weights[i]);
    weighted_average[1] += (data[i][1] * normalized_weights[i]);
    weighted_average[2] += (data[i][2] * normalized_weights[i]);
  }

  return Eigen::Matrix<T, 3, 1>(static_cast<float>(weighted_average[0]),
                                static_cast<float>(weighted_average[1]),
                                static_cast<float>(weighted_average[2]));
}

template <typename T>
T WeightedMedian(const std::vector<T>& data,
                 const std::vector<float>& weights) {
  assert(data.size() > 0);
  assert(data.size() == weights.size());

  std::vector<float> normalized_weights;
  NormalizeWeights(weights, &normalized_weights);

  std::vector<std::pair<float, T>> data_weights;
  for (size_t i = 0; i < data.size(); i++) {
    data_weights.push_back(std::make_pair(normalized_weights[i], data[i]));
  }
  std::sort(data_weights.begin(), data_weights.end(),
            [](const std::pair<float, T>& a, const std::pair<float, T>& b) {
              return a.first < b.first;
            });

  float weights_sum{0};
  size_t index{0};
  for (size_t i = 0; i < data_weights.size(); i++) {
    weights_sum += data_weights[i].first;
    if (weights_sum > 0.5f) {
      index = i;
      break;
    }
  }

  return data_weights[index].second;
}

inline float EdgeFunction(const Eigen::Vector2f& a, const Eigen::Vector2f& b,
                          const Eigen::Vector2f& c) {
  return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]);
}

}  // namespace

namespace ugu {

VisibilityTesterOption::VisibilityTesterOption() {}

VisibilityTesterOption::~VisibilityTesterOption() {}

void VisibilityTesterOption::CopyTo(VisibilityTesterOption* dst) const {
  // todo
  dst->backface_culling_angle_rad_th = backface_culling_angle_rad_th;
  dst->use_mask = use_mask;
  dst->use_depth = use_depth;
  dst->interp = interp;
  dst->calc_stat_vertex_info = calc_stat_vertex_info;
  dst->collect_face_info;
  dst->calc_stat_face_info = calc_stat_face_info;
}

VertexInfoPerKeyframe::VertexInfoPerKeyframe() {}
VertexInfoPerKeyframe::~VertexInfoPerKeyframe() {}

VertexInfo::VertexInfo() {}
VertexInfo::~VertexInfo() {}
void VertexInfo::Update(const VertexInfoPerKeyframe& info) {
  visible_keyframes.push_back(info);
}
void VertexInfo::CalcStat() {
  if (visible_keyframes.empty()) {
    return;
  }

  // mean
  mean_color = Eigen::Vector3f(0, 0, 0);
  mean_distance = 0;
  mean_viewing_angle = 0;
  for (auto& kf : visible_keyframes) {
    mean_color += kf.color;
    mean_distance += kf.distance;
    mean_viewing_angle += kf.viewing_angle;
  }
  mean_color /= static_cast<float>(visible_keyframes.size());
  mean_distance /= visible_keyframes.size();
  mean_viewing_angle /= visible_keyframes.size();

  // make vectors
  std::vector<Eigen::Vector3f> colors;
  std::vector<int> ids;
  std::vector<float> distances;
  std::vector<float> viewing_angles;
  std::vector<float> inv_distances;
  std::vector<float> inv_viewing_angles;
  for (auto& kf : visible_keyframes) {
    colors.push_back(kf.color);
    ids.push_back(kf.kf_id);
    distances.push_back(kf.distance);
    viewing_angles.push_back(kf.viewing_angle);
    inv_distances.push_back(1.0f / kf.distance);
    inv_viewing_angles.push_back(1.0f / kf.viewing_angle);
  }

  // min
  auto min_viewing_angle_it =
      std::min_element(viewing_angles.begin(), viewing_angles.end());
  min_viewing_angle_index = static_cast<int>(
      std::distance(viewing_angles.begin(), min_viewing_angle_it));
  min_viewing_angle_id = visible_keyframes[min_viewing_angle_index].kf_id;
  min_viewing_angle_color = visible_keyframes[min_viewing_angle_index].color;

  auto min_distance_it = std::min_element(distances.begin(), distances.end());
  min_distance_index =
      static_cast<int>(std::distance(distances.begin(), min_distance_it));
  min_distance_id = visible_keyframes[min_distance_index].kf_id;
  min_distance_color = visible_keyframes[min_distance_index].color;

  // median
  median_viewing_angle = Median(viewing_angles);
  median_distance = Median(distances);
  median_color = MedianColor(colors);

  // weighted average
  mean_viewing_angle_color = WeightedAverage(colors, inv_viewing_angles);
  mean_distance_color = WeightedAverage(colors, inv_distances);

  // weighted median
  median_viewing_angle_color = WeightedMedian(colors, inv_viewing_angles);
  median_distance_color = WeightedMedian(colors, inv_distances);
}

int VertexInfo::VisibleFrom(int kf_id) const {
  /*
  bool res = std::any_of(visible_keyframes.begin(), visible_keyframes.end(),
                         [kf_id](const VertexInfoPerKeyframe& infokf) {
                           return infokf.kf_id == kf_id;
                         });
  */

  int index = -1;
  for (int i = 0; i < static_cast<int>(visible_keyframes.size()); i++) {
    if (visible_keyframes[i].kf_id == kf_id) {
      index = i;
      break;
    }
  }

  return index;
}

FaceInfoPerKeyframe::FaceInfoPerKeyframe() {}
FaceInfoPerKeyframe::~FaceInfoPerKeyframe() {}

FaceInfo::FaceInfo() {}
FaceInfo::~FaceInfo() {}
void FaceInfo::Update(const FaceInfoPerKeyframe& info) {
  visible_keyframes.push_back(info);
}
void FaceInfo::CalcStat() {
  if (visible_keyframes.empty()) {
    return;
  }

  // make vectors
  std::vector<int> ids;
  std::vector<float> distances;
  std::vector<float> viewing_angles;
  std::vector<float> areas;
  for (auto& kf : visible_keyframes) {
    ids.push_back(kf.kf_id);
    distances.push_back(kf.distance);
    viewing_angles.push_back(kf.viewing_angle);
    areas.push_back(kf.area);
  }

  auto max_area_it = std::max_element(areas.begin(), areas.end());
  max_area_index = static_cast<int>(std::distance(areas.begin(), max_area_it));
  max_area_id = visible_keyframes[max_area_index].kf_id;

  auto min_viewing_angle_it =
      std::min_element(viewing_angles.begin(), viewing_angles.end());
  min_viewing_angle_index = static_cast<int>(
      std::distance(viewing_angles.begin(), min_viewing_angle_it));
  min_viewing_angle_id = visible_keyframes[min_viewing_angle_index].kf_id;

  auto min_distance_it = std::min_element(distances.begin(), distances.end());
  min_distance_index =
      static_cast<int>(std::distance(distances.begin(), min_distance_it));
  min_distance_id = visible_keyframes[min_distance_index].kf_id;

  // TODO: calculate stats for colors insidef of faces
}

VisibilityInfo::VisibilityInfo() {}
VisibilityInfo::VisibilityInfo(const Mesh& mesh) {
  vertex_info_list.resize(mesh.vertices().size());
  face_info_list.resize(mesh.vertex_indices().size());
}
VisibilityInfo::~VisibilityInfo() {}
void VisibilityInfo::CalcStatVertexInfo() {
  Timer<> timer;
  timer.Start();
  int vertex_info_list_num = static_cast<int>(vertex_info_list.size());
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_info_list_num; i++) {
    vertex_info_list[i].CalcStat();
  }
  has_vertex_stat = true;
  timer.End();
  LOGI("finalize statitics for vertices: %.1f msecs\n", timer.elapsed_msec());
}

void VisibilityInfo::CalcStatFaceInfo() {
  Timer<> timer;
  timer.Start();
  int face_info_list_num = static_cast<int>(face_info_list.size());
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < face_info_list_num; i++) {
    face_info_list[i].CalcStat();
  }
  has_face_stat = true;
  timer.End();
  LOGI("finalize statitics for faces: %.1f msecs\n", timer.elapsed_msec());
}

void VisibilityInfo::Update(int vertex_id, const VertexInfoPerKeyframe& info) {
  vertex_info_list[vertex_id].Update(info);
}
void VisibilityInfo::Update(int face_id, const FaceInfoPerKeyframe& info) {
  face_info_list[face_id].Update(info);
}
std::string VisibilityInfo::SerializeAsJson() const {
  // todo
  return "";
}

Keyframe::Keyframe() {}
Keyframe::~Keyframe() {}

VisibilityTester::VisibilityTester() {}

VisibilityTester::~VisibilityTester() {}

VisibilityTester::VisibilityTester(const VisibilityTesterOption& option) {
  set_option(option);
}

void VisibilityTester::set_option(const VisibilityTesterOption& option) {
  option.CopyTo(&option_);
}

void VisibilityTester::set_mesh(std::shared_ptr<const Mesh> mesh) {
  if (mesh == nullptr) {
    LOGE("mesh is nullptr\n");
    return;
  }

  mesh_initialized_ = false;
  mesh_ = mesh;

  if (mesh_->face_normals().empty()) {
    LOGW("face normal is empty. culling and shading may not work\n");
  }

  if (mesh_->normals().empty()) {
    LOGW("vertex normal is empty. shading may not work\n");
  }

  flatten_vertices_.clear();
  flatten_faces_.clear();

  const std::vector<Eigen::Vector3f>& vertices = mesh_->vertices();
  flatten_vertices_.resize(vertices.size() * 3);
  for (size_t i = 0; i < vertices.size(); i++) {
    flatten_vertices_[i * 3 + 0] = vertices[i][0];
    flatten_vertices_[i * 3 + 1] = vertices[i][1];
    flatten_vertices_[i * 3 + 2] = vertices[i][2];
  }

  const std::vector<Eigen::Vector3i>& vertex_indices = mesh_->vertex_indices();
  flatten_faces_.resize(vertex_indices.size() * 3);
  for (size_t i = 0; i < vertex_indices.size(); i++) {
    flatten_faces_[i * 3 + 0] = vertex_indices[i][0];
    flatten_faces_[i * 3 + 1] = vertex_indices[i][1];
    flatten_faces_[i * 3 + 2] = vertex_indices[i][2];
  }
}
bool VisibilityTester::PrepareMesh() {
  if (mesh_ == nullptr) {
    LOGE("mesh has not been set\n");
    return false;
  }

  if (flatten_vertices_.empty() || flatten_faces_.empty()) {
    LOGE("mesh is empty\n");
    return false;
  }

  bool ret = false;
  build_options_.cache_bbox = false;

  LOGI("  BVH build option:\n");
  LOGI("    # of leaf primitives: %d\n", build_options_.min_leaf_primitives);
  LOGI("    SAH binsize         : %d\n", build_options_.bin_size);

  Timer<> timer;
  timer.Start();

  triangle_mesh_.reset(new nanort::TriangleMesh<float>(
      &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3));

  triangle_pred_.reset(new nanort::TriangleSAHPred<float>(
      &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3));

  LOGI("num_triangles = %llu\n",
       static_cast<uint64_t>(mesh_->vertex_indices().size()));
  // LOGI("faces = %p\n", mesh_->vertex_indices().size());

  ret = accel_.Build(static_cast<unsigned int>(mesh_->vertex_indices().size()),
                     *triangle_mesh_, *triangle_pred_, build_options_);

  if (!ret) {
    LOGE("BVH building failed\n");
    return false;
  }

  timer.End();
  LOGI("  BVH build time: %.1f msecs\n", timer.elapsed_msec());

  stats_ = accel_.GetStatistics();

  LOGI("  BVH statistics:\n");
  LOGI("    # of leaf   nodes: %d\n", stats_.num_leaf_nodes);
  LOGI("    # of branch nodes: %d\n", stats_.num_branch_nodes);
  LOGI("  Max tree depth     : %d\n", stats_.max_tree_depth);

  accel_.BoundingBox(bmin_, bmax_);
  LOGI("  Bmin               : %f, %f, %f\n", bmin_[0], bmin_[1], bmin_[2]);
  LOGI("  Bmax               : %f, %f, %f\n", bmax_[0], bmax_[1], bmax_[2]);

  mesh_initialized_ = true;

  return true;
}

void VisibilityTester::set_keyframe(std::shared_ptr<Keyframe> keyframe) {
  keyframe_ = keyframe;
}

bool VisibilityTester::ValidateAndInitBeforeTest(VisibilityInfo* info) const {
  if (info == nullptr) {
    LOGE("info is nullptr\n");
    return false;
  }
  if (keyframe_ == nullptr) {
    LOGE("keyframe has not been set\n");
    return false;
  }
  if (!mesh_initialized_) {
    LOGE("mesh has not been initialized\n");
    return false;
  }

  if (keyframe_->color.empty()) {
    LOGE("keyframe color is empty\n");
    return false;
  }
  if (option_.use_mask && keyframe_->mask.empty()) {
    LOGE("use_mask is true but keyframe mask is empty\n");
    return false;
  }
  if (option_.use_depth && keyframe_->depth.empty()) {
    LOGE("use_depth is true but keyframe depth is empty\n");
    return false;
  }

  if (info->vertex_info_list.size() != mesh_->vertices().size()) {
    LOGE("info->vertex_info_list size %d is different from mesh %d\n",
         static_cast<int>(info->vertex_info_list.size()),
         static_cast<int>(mesh_->vertices().size()));
    return false;
  }
  if (info->face_info_list.size() != mesh_->vertex_indices().size()) {
    LOGE("info->face_info_list size %d is different from mesh %d\n",
         static_cast<int>(info->face_info_list.size()),
         static_cast<int>(mesh_->vertex_indices().size()));
    return false;
  }

  int vertex_info_list_num = static_cast<int>(info->vertex_info_list.size());
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_info_list_num; i++) {
    auto ii = info->vertex_info_list[i];
    ii.min_viewing_angle_color = option_.invalid_color;
    ii.min_distance_color = option_.invalid_color;
    ii.mean_color = option_.invalid_color;
    ii.median_color = option_.invalid_color;
    ii.mean_viewing_angle_color = option_.invalid_color;
    ii.median_viewing_angle_color = option_.invalid_color;
    ii.mean_distance_color = option_.invalid_color;
    ii.median_distance_color = option_.invalid_color;
  }

  return true;
}

bool VisibilityTester::TestVertices(VisibilityInfo* info) const {
  Timer<> timer;
  timer.Start();
  const auto& vertices = mesh_->vertices();
  const auto& normals = mesh_->normals();
  int vertex_num = static_cast<int>(vertices.size());

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_num; i++) {
    // convert vertex to camera space from world space
    const Eigen::Vector3f& world_p = vertices[i];
    Eigen::Vector3f camera_p = keyframe_->camera->w2c().cast<float>() * world_p;
    // project vertex to image space
    Eigen::Vector2f image_p;
    float d;
    keyframe_->camera->Project(camera_p, &image_p, &d);

    // check if projected point is inside of image space of the current camera
    if (d < 0 || image_p.x() < 0 || keyframe_->color.cols - 1 < image_p.x() ||
        image_p.y() < 0 || keyframe_->color.rows - 1 < image_p.y()) {
      continue;
    }

    // nearest integer pixel index
    int nn_x = static_cast<int>(std::round(image_p.x()));
    int nn_y = static_cast<int>(std::round(image_p.y()));

    // mask and depth value check
    // nearest mask pixel should be valid (255)
    if (option_.use_mask &&
        keyframe_->mask.at<unsigned char>(nn_y, nn_x) != 255) {
      continue;
    }
    // nearest depth pixel should be valid (larger than 0)
    if (option_.use_depth && keyframe_->depth.at<float>(nn_y, nn_x) <
                                 std::numeric_limits<float>::epsilon()) {
      continue;
    }

    // ray from projected point in image space
    Eigen::Vector3f ray_w, org_ray_w;
    keyframe_->camera->ray_w(image_p.x(), image_p.y(), &ray_w);
    keyframe_->camera->org_ray_w(image_p.x(), image_p.y(), &org_ray_w);
    nanort::Ray<float> ray;
    PrepareRay(&ray, org_ray_w, ray_w);

    // back-face culling
    float dot = -normals[i].dot(ray_w);
    float viewing_angle = std::acos(dot);
    // back-face if angle is larager than threshold
    if (option_.backface_culling_angle_rad_th < viewing_angle) {
      continue;
    }

    // shoot ray
    nanort::TriangleIntersector<> triangle_intersector(
        &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3);
    nanort::TriangleIntersection<> isect;
    bool hit = accel_.Traverse(ray, triangle_intersector, &isect);

    Eigen::Vector3f hit_pos_w;
    if (hit) {
      // if there is a large distance beween hit position and vertex position,
      // the vertex is occluded by another face and should be ignored
      hit_pos_w = org_ray_w + ray_w * isect.t;
      float dist = (hit_pos_w - vertices[i]).norm();
      if (dist > option_.eps) {
        continue;
      }
    } else {
      // theoritically must hit somewhere because ray are emitted from vertex
      // but sometimes no hit...maybe numerical reason
      // in the case use original vertex position for hit_pos_w hoping it was
      // not occluded
      hit_pos_w = vertices[i];
    }

    // check depth difference between hit position and input depth
    if (option_.use_depth) {
      // convert hit position to camera coordinate to get
      // depth value
      Eigen::Vector3f hit_pos_c =
          keyframe_->camera->w2c().cast<float>() * hit_pos_w;

      assert(0.0f <= hit_pos_c[2]);  // depth should be positive
      float diff =
          std::abs(hit_pos_c[2] - keyframe_->depth.at<float>(nn_y, nn_x));
      if (option_.max_depth_difference < diff) {
        continue;
      }
    }

    // update vertex info per keyframe
    VertexInfoPerKeyframe vertex_info;
    vertex_info.kf_id = keyframe_->id;
    vertex_info.projected_pos = image_p;
    vertex_info.viewing_angle = viewing_angle;
    vertex_info.distance = isect.t;

    if (vertex_info.distance > option_.max_distance_from_camera_to_vertex ||
        vertex_info.viewing_angle > option_.max_viewing_angle) {
      continue;
    }

    if (option_.interp == ColorInterpolation::kNn) {
      const ugu::Vec3b& color = keyframe_->color.at<ugu::Vec3b>(nn_y, nn_x);
      vertex_info.color[0] = color[0];
      vertex_info.color[1] = color[1];
      vertex_info.color[2] = color[2];
    } else if (option_.interp == ColorInterpolation::kBilinear) {
      ::BilinearInterpolation(image_p.x(), image_p.y(), keyframe_->color,
                            &vertex_info.color);
    }
    info->Update(i, vertex_info);
  }
  timer.End();
  LOGI("vertex information collection: %.1f msecs\n", timer.elapsed_msec());

  return true;
}

bool VisibilityTester::TestFaces(VisibilityInfo* info) const {
  Timer<> timer;
  timer.Start();
  const auto& faces = mesh_->vertex_indices();
  int face_num = static_cast<int>(faces.size());

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < face_num; i++) {
    const auto& face = faces[i];

    // update face info per keyframe
    FaceInfoPerKeyframe face_info;
    face_info.kf_id = keyframe_->id;
    face_info.face_id = i;
    face_info.viewing_angle = 0.0f;
    face_info.distance = 0.0f;
    bool face_visible = true;
    for (int j = 0; j < 3; j++) {
      int vid = face[j];
      const auto& vinfo = info->vertex_info_list[vid];
      int visible_index = vinfo.VisibleFrom(face_info.kf_id);
      // Face is visible only if all 3 vertices are visible
      if (visible_index < 0) {
        face_visible = false;
        break;
      }
      const auto& vinfo_kf = vinfo.visible_keyframes[visible_index];

      face_info.viewing_angle += vinfo_kf.viewing_angle;
      face_info.distance += vinfo_kf.distance;
      face_info.projected_tri[j] = vinfo_kf.projected_pos;
    }
    if (!face_visible) {
      continue;
    }

    // Take average
    face_info.viewing_angle /= 3;
    face_info.distance /= 3;

    // Calculate triangle area
    face_info.area = 0.5f * std::abs(EdgeFunction(face_info.projected_tri[0],
                                                  face_info.projected_tri[1],
                                                  face_info.projected_tri[2]));

    info->Update(i, face_info);
  }

  timer.End();
  LOGI("face information collection: %.1f msecs\n", timer.elapsed_msec());
  return true;
}

bool VisibilityTester::Test(VisibilityInfo* info) const {
  if (!ValidateAndInitBeforeTest(info)) {
    return false;
  }
  LOGI("Test for keyframe id %d\n", keyframe_->id);

  TestVertices(info);

  if (option_.calc_stat_vertex_info) {
    info->CalcStatVertexInfo();
  }

  if (option_.collect_face_info) {
    TestFaces(info);
    if (option_.calc_stat_face_info) {
      info->CalcStatFaceInfo();
    }
  }
  return true;
}

bool VisibilityTester::Test(std::vector<std::shared_ptr<Keyframe>> keyframes,
                            VisibilityInfo* info) {
  bool ret = true;
  VisibilityTesterOption org_option;
  option_.CopyTo(&org_option);

  // disable stat
  option_.calc_stat_face_info = false;
  option_.calc_stat_vertex_info = false;

  for (size_t i = 0; i < keyframes.size() - 1; i++) {
    set_keyframe(keyframes[i]);
    if (!Test(info)) {
      ret = false;
      break;
    }
  }

  // recover original option at the end
  set_option(org_option);
  set_keyframe(keyframes[keyframes.size() - 1]);
  if (!Test(info)) {
    ret = false;
  }

  return ret;
}

}  // namespace ugu
