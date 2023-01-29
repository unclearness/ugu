/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/texturing/visibility_tester.h"

#include <cassert>
#include <iterator>

#include "ugu/accel/bvh_nanort.h"
#include "ugu/timer.h"
#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace ugu {

std::ostream& operator<<(std::ostream& os, const VertexInfoPerKeyframe& vi) {
  os << "{ \"kf_id\": " << vi.kf_id << ", \"projected_pos\":["
     << vi.projected_pos[0] << "," << vi.projected_pos[1] << "],"
     << "\"viewing_angle\":" << vi.viewing_angle << ","
     << "\"distance\":" << vi.distance << "}";
  return os;
}

std::ostream& operator<<(std::ostream& os, const VertexInfo& vi) {
  os << "{\"visible_keyframes\":[";
  for (size_t i = 0; i < vi.visible_keyframes.size(); i++) {
    const auto& kf = vi.visible_keyframes[i];
    os << kf;
    if (i != vi.visible_keyframes.size() - 1) {
      os << ",";
    }
  }
  os << "]";
  os << "}";
  return os;
}

VisibilityTesterOption::VisibilityTesterOption() {}

VisibilityTesterOption::~VisibilityTesterOption() {}

void VisibilityTesterOption::CopyTo(VisibilityTesterOption* dst) const {
  // todo
  dst->backface_culling_angle_rad_th = backface_culling_angle_rad_th;
  dst->use_mask = use_mask;
  dst->use_depth = use_depth;
  dst->interp = interp;
  dst->calc_stat_vertex_info = calc_stat_vertex_info;
  dst->collect_face_info = collect_face_info;
  dst->calc_stat_face_info = calc_stat_face_info;
}

VertexInfoPerKeyframe::VertexInfoPerKeyframe() {}
VertexInfoPerKeyframe::~VertexInfoPerKeyframe() {}
VertexInfo::VertexInfo() {}
VertexInfo::VertexInfo(const VertexInfo& src) {
  (void)src;
  // TODO: implement
  // Disable default copy constructor for mutex, which is cannot be copied.
};
VertexInfo::~VertexInfo() {}
void VertexInfo::Update(const VertexInfoPerKeyframe& info) {
  std::lock_guard<std::mutex> lock(mtx_);
  visible_keyframes.push_back(info);
}
void VertexInfo::CalcStat(std::function<void(VertexInfo&)> vert_custom_func) {
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
  std::vector<float> intensities;
  std::vector<float> inv_distances;
  std::vector<float> inv_viewing_angles;
  for (auto& kf : visible_keyframes) {
    colors.push_back(kf.color);
    ids.push_back(kf.kf_id);
    distances.push_back(kf.distance);
    viewing_angles.push_back(kf.viewing_angle);
    intensities.push_back(kf.intensity);
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
  mean_viewing_angle_color = ugu::WeightedAverage(colors, inv_viewing_angles);
  mean_distance_color = ugu::WeightedAverage(colors, inv_distances);

  // weighted median
  median_viewing_angle_color = ugu::WeightedMedian(colors, inv_viewing_angles);
  median_distance_color = ugu::WeightedMedian(colors, inv_distances);

  // Intensity
  auto min_intensity_it =
      std::min_element(intensities.begin(), intensities.end());
  min_intensity = *min_intensity_it;
  min_intensity_color =
      colors[std::distance(intensities.begin(), min_intensity_it)];

  median_intensity = Median(intensities, true);
  auto median_viewing_it =
      std::find(intensities.begin(), intensities.end(), median_intensity);
  median_intensity_color =
      colors[std::distance(intensities.begin(), median_viewing_it)];

  // Mode
  Mode(colors.begin(), colors.end(), mode, mode_frequency, occurrence);
  Mode(colors.begin(), colors.end(), mode_viewing_angle_color,
       mode_frequency_viewing_angle_color, occurrence_viewing_angle,
       inv_viewing_angles);

  if (vert_custom_func != nullptr) {
    vert_custom_func(*this);
  }
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

FaceInfoPerKeyframe::FaceInfoPerKeyframe()
    : area(-1.f),
      viewing_angle(-1.f),
      distance(std::numeric_limits<float>::max()) {}
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
void VisibilityInfo::CalcStatVertexInfo(
    std::function<void(VertexInfo&)> vert_custom_func) {
  Timer<> timer;
  timer.Start();
  int vertex_info_list_num = static_cast<int>(vertex_info_list.size());
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_info_list_num; i++) {
    vertex_info_list[i].CalcStat(vert_custom_func);
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
  std::stringstream ss;
  ss << "{";
  ss << "\"vertex_info_list\":[";
  for (size_t i = 0; i < vertex_info_list.size(); i++) {
    const auto& vi = vertex_info_list[i];
    ss << vi;
    if (i != vertex_info_list.size() - 1) {
      ss << ",";
    }
  }
  ss << "]";
  ss << "}";
  return ss.str();
}

Keyframe::Keyframe() {}
Keyframe::~Keyframe() {}

void VisibilityTester::Init() {
#ifdef UGU_USE_NANORT
  bvh_ = std::make_unique<BvhNanort<Eigen::Vector3f, Eigen::Vector3i>>();
#else
  auto tmp = std::make_unique<BvhNaive<Eigen::Vector3f, Eigen::Vector3i>>();
  tmp->SetAxisNum(3);
  tmp->SetMinLeafPrimitives(5);
  bvh_ = std::move(tmp);
#endif
}

VisibilityTester::VisibilityTester() { Init(); }

VisibilityTester::~VisibilityTester() {}

VisibilityTester::VisibilityTester(const VisibilityTesterOption& option) {
  set_option(option);
  Init();
}

void VisibilityTester::set_option(const VisibilityTesterOption& option) {
  option.CopyTo(&option_);
}

void VisibilityTester::set_data(const std::vector<Eigen::Vector3f>& vertices,
                                const std::vector<Eigen::Vector3f>& normals,
                                const std::vector<Eigen::Vector3i>& indices) {
  mesh_initialized_ = false;
  vertices_ = vertices;
  normals_ = normals;
  indices_ = indices;
}

bool VisibilityTester::PrepareData() {
  if (vertices_.empty() || normals_.empty() || indices_.empty()) {
    LOGE("mesh is empty\n");
    return false;
  }

  bvh_->SetData(vertices_, indices_);
  bool ret = bvh_->Build();
  if (!ret) {
    return false;
  }

  face_centers_.resize(indices_.size());
  for (size_t fid = 0; fid < indices_.size(); fid++) {
    face_centers_[fid].setZero();
    for (int i = 0; i < 3; i++) {
      face_centers_[fid] += vertices_[indices_[fid][i]];
    }
    face_centers_[fid] /= 3;
  }

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

  if (info->vertex_info_list.size() != vertices_.size()) {
    LOGE("info->vertex_info_list size %d is different from mesh %d\n",
         static_cast<int>(info->vertex_info_list.size()),
         static_cast<int>(vertices_.size()));
    return false;
  }

  if (info->vertex_info_list.size() != normals_.size()) {
    LOGE("info->vertex_info_list size %d is different from mesh normal %d\n",
         static_cast<int>(info->vertex_info_list.size()),
         static_cast<int>(normals_.size()));
    return false;
  }

  if (info->face_info_list.size() != indices_.size()) {
    LOGE("info->face_info_list size %d is different from mesh %d\n",
         static_cast<int>(info->face_info_list.size()),
         static_cast<int>(indices_.size()));
    return false;
  }

  int vertex_info_list_num = static_cast<int>(info->vertex_info_list.size());
#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_info_list_num; i++) {
    auto& ii = info->vertex_info_list[i];
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

bool VisibilityTester::TestVertex(VisibilityInfo* info, int vid,
                                  VertexInfoPerKeyframe& vertex_info,
                                  const Eigen::Vector3f& v_offset) const {
  (void)info;
  // convert vertex to camera space from world space
  const Eigen::Vector3f& world_p = vertices_[vid] + v_offset;
  Eigen::Vector3f camera_p = keyframe_->camera->w2c().cast<float>() * world_p;
  // project vertex to image space
  Eigen::Vector2f image_p;
  float d;
  keyframe_->camera->Project(camera_p, &image_p, &d);

  // check if projected point is inside of image space of the current camera
  if (d < 0 || image_p.x() < 0 || keyframe_->color.cols - 1 < image_p.x() ||
      image_p.y() < 0 || keyframe_->color.rows - 1 < image_p.y()) {
    return false;
  }

  // nearest integer pixel index
  int nn_x = static_cast<int>(std::round(image_p.x()));
  int nn_y = static_cast<int>(std::round(image_p.y()));

  // mask and depth value check
  // nearest mask pixel should be valid (255)
  if (option_.use_mask &&
      keyframe_->mask.at<unsigned char>(nn_y, nn_x) != 255) {
    return false;
  }
  // nearest depth pixel should be valid (larger than 0)
  if (option_.use_depth && keyframe_->depth.at<float>(nn_y, nn_x) <
                               std::numeric_limits<float>::epsilon()) {
    return false;
  }

  // ray from projected point in image space
  Eigen::Vector3f ray_w, org_ray_w;
  keyframe_->camera->ray_w(image_p.x(), image_p.y(), &ray_w);
  keyframe_->camera->org_ray_w(image_p.x(), image_p.y(), &org_ray_w);
  Ray ray;
  ray.dir = ray_w;
  ray.org = org_ray_w;

  // back-face culling
  float dot = -normals_[vid].dot(ray_w);
  float viewing_angle = std::acos(dot);
  // back-face if angle is larager than threshold
  if (option_.backface_culling_angle_rad_th < viewing_angle) {
    return false;
  }

  // shoot ray
  auto results = bvh_->Intersect(ray, false);
  float distance = std::numeric_limits<float>::max();
  Eigen::Vector3f hit_pos_w;
  if (!results.empty()) {
    distance = results[0].t;
    // if there is a large distance beween hit position and vertex position,
    // the vertex is occluded by another face and should be ignored
    hit_pos_w = org_ray_w + ray_w * results[0].t;
    float dist = (hit_pos_w - vertices_[vid]).norm();
    if (dist > option_.eps) {
      return false;
    }
  } else {
    // theoritically must hit somewhere because ray are emitted from vertex
    // but sometimes no hit...maybe numerical reason
    // in the case use original vertex position for hit_pos_w hoping it was
    // not occluded
    hit_pos_w = vertices_[vid];
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
      return false;
    }
  }

  // update vertex info per keyframe;
  vertex_info.kf_id = keyframe_->id;
  vertex_info.projected_pos = image_p;
  vertex_info.viewing_angle = viewing_angle;
  vertex_info.distance = distance;

  if (vertex_info.distance > option_.max_distance_from_camera_to_vertex ||
      vertex_info.viewing_angle > option_.max_viewing_angle) {
    return false;
  }

  if (option_.interp == ColorInterpolation::kNn) {
    const ugu::Vec3b& color = keyframe_->color.at<ugu::Vec3b>(nn_y, nn_x);
    vertex_info.color[0] = color[0];
    vertex_info.color[1] = color[1];
    vertex_info.color[2] = color[2];
  } else if (option_.interp == ColorInterpolation::kBilinear) {
    const ugu::Vec3b color =
        BilinearInterpolation(image_p.x(), image_p.y(), keyframe_->color);
    vertex_info.color[0] = color[0];
    vertex_info.color[1] = color[1];
    vertex_info.color[2] = color[2];
  }

  vertex_info.intensity = Color2Gray(vertex_info.color);

  return true;
}

bool VisibilityTester::TestVertices(VisibilityInfo* info) const {
  Timer<> timer;
  timer.Start();
  int vertex_num = static_cast<int>(vertices_.size());

#if defined(_OPENMP) && defined(UGU_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int i = 0; i < vertex_num; i++) {
    VertexInfoPerKeyframe vertex_info;
    if (TestVertex(info, i, vertex_info)) {
      info->Update(i, vertex_info);
    }
  }
  timer.End();
  LOGI("vertex information collection: %.1f msecs\n", timer.elapsed_msec());

  return true;
}

bool VisibilityTester::TestFaces(VisibilityInfo* info) const {
  Timer<> timer;
  timer.Start();
  const auto& faces = indices_;
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

bool VisibilityTester::TestFacewiseVertices(VisibilityInfo* info) const {
  Timer<> timer;
  timer.Start();
  const auto& faces = indices_;
  int face_num = static_cast<int>(faces.size());
  std::vector<bool> v_succeeded(vertices_.size(), false);
  constexpr float kEpsilon = 1e-3f;

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
      VertexInfoPerKeyframe vertex_info;
      Eigen::Vector3f diff = (face_centers_[i] - vertices_[vid]);
      Eigen::Vector3f v_offset = diff * kEpsilon;
      face_visible = TestVertex(info, vid, vertex_info, v_offset);
      if (face_visible) {
        if (!v_succeeded[vid]) {
          // Update only once per vertex
          v_succeeded[vid] = true;
          // TODO: Handle multi-thread
          info->Update(vid, vertex_info);
        }
      }
      // Face is visible only if all 3 vertices are visible
      if (!face_visible) {
        break;
      }
      face_info.viewing_angle += vertex_info.viewing_angle;
      face_info.distance += vertex_info.distance;
      face_info.projected_tri[j] = vertex_info.projected_pos;
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
  LOGI("face-wise information collection: %.1f msecs\n", timer.elapsed_msec());
  return true;
}

bool VisibilityTester::Test(
    VisibilityInfo* info, bool facewise,
    std::function<void(VertexInfo&)> vert_custom_func) const {
  if (!ValidateAndInitBeforeTest(info)) {
    return false;
  }

  LOGI("Test for keyframe id %d\n", keyframe_->id);

  if (facewise) {
    TestFacewiseVertices(info);
    if (option_.calc_stat_vertex_info) {
      info->CalcStatVertexInfo(vert_custom_func);
    }
    if (option_.calc_stat_face_info) {
      info->CalcStatFaceInfo();
    }
  } else {
    TestVertices(info);

    if (option_.calc_stat_vertex_info) {
      info->CalcStatVertexInfo(vert_custom_func);
    }

    if (option_.collect_face_info) {
      TestFaces(info);
      if (option_.calc_stat_face_info) {
        info->CalcStatFaceInfo();
      }
    }
  }
  return true;
}

bool VisibilityTester::Test(std::vector<std::shared_ptr<Keyframe>> keyframes,
                            VisibilityInfo* info, bool facewise,
                            std::function<void(VertexInfo&)> vert_custom_func) {
  bool ret = true;
  VisibilityTesterOption org_option;
  option_.CopyTo(&org_option);

  // disable stat
  option_.calc_stat_face_info = false;
  option_.calc_stat_vertex_info = false;

  for (size_t i = 0; i < keyframes.size() - 1; i++) {
    set_keyframe(keyframes[i]);
    if (!Test(info, facewise, vert_custom_func)) {
      ret = false;
      break;
    }
  }

  // recover original option at the end
  set_option(org_option);
  set_keyframe(keyframes[keyframes.size() - 1]);
  if (!Test(info, facewise, vert_custom_func)) {
    ret = false;
  }

  return ret;
}

}  // namespace ugu
