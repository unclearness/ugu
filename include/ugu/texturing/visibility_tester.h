/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANORT

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ugu/camera.h"
#include "ugu/image.h"
#include "ugu/mesh.h"

#include "nanort.h"

namespace ugu {

enum class ViewSelectionCriteria {
  kMinViewingAngle = 0,
  kMinDistance = 1,
  kMeanViewingAngle = 2,
  kMedianViewingAngle = 3,
  kMeanDistance = 4,
  kMedianDistance = 5,
  kMaxArea = 6
};

struct VisibilityTesterOption {
  float backface_culling_angle_rad_th{ugu::radians(90.0f)};
  bool use_mask{true};
  bool use_depth{true};
  ColorInterpolation interp{ColorInterpolation::kBilinear};
  bool calc_stat_vertex_info{true};
  bool collect_face_info{true};
  bool calc_stat_face_info{true};

  float max_viewing_angle{ugu::radians(90.0f)};
  float max_distance_from_camera_to_vertex{std::numeric_limits<float>::max()};
  float max_depth_difference{std::numeric_limits<float>::max()};

  float eps = 0.001f;
  Eigen::Vector3f invalid_color{0.0f, 0.0f, 0.0f};

  VisibilityTesterOption();
  ~VisibilityTesterOption();
  void CopyTo(VisibilityTesterOption* dst) const;
};

struct VertexInfoPerKeyframe {
  int kf_id{-1};  // positive keyframe id
  Eigen::Vector3f color;
  Eigen::Vector2f projected_pos;  // projected 2d image space position
  float viewing_angle{999.9f};    // radian
  float distance{-999.9f};  // distance along with ray from camera (not depth)

  VertexInfoPerKeyframe();
  ~VertexInfoPerKeyframe();
};

struct VertexInfo {
  std::vector<VertexInfoPerKeyframe> visible_keyframes;

  /*** gotten by Finalize() **************/
  Eigen::Vector3f mean_color;
  Eigen::Vector3f median_color;
  int min_viewing_angle_index{-1};
  int min_viewing_angle_id{-1};
  Eigen::Vector3f min_viewing_angle_color;

  Eigen::Vector3f
      mean_viewing_angle_color;  // weighted average by inverse viewing angle
  Eigen::Vector3f
      median_viewing_angle_color;  // weighted median by inverse viewing angle

  int min_distance_index{-1};
  int min_distance_id{-1};
  Eigen::Vector3f min_distance_color;

  Eigen::Vector3f mean_distance_color;  // weighted average by inverse distance
  Eigen::Vector3f median_distance_color;  // weighted median by inverse distance

  float mean_viewing_angle{-1.0f};
  float median_viewing_angle{-1.0f};

  float mean_distance{-1.0f};
  float median_distance{-1.0f};
  /**************************************/

  VertexInfo();
  ~VertexInfo();
  void Update(const VertexInfoPerKeyframe& info);
  void CalcStat();
  int VisibleFrom(int kf_id) const;
};

struct FaceInfoPerKeyframe {
  int kf_id{-1};
  int face_id{-1};
  Eigen::Vector3f mean_color;    // mean inside projected triangle
  Eigen::Vector3f median_color;  // median inside projected triangle
  float area;  // are in projected image space. unit is pixel*pixel
  float viewing_angle;
  float distance;

  std::array<Eigen::Vector2f, 3> projected_tri;

  FaceInfoPerKeyframe();
  ~FaceInfoPerKeyframe();
};

struct FaceInfo {
  std::vector<FaceInfoPerKeyframe> visible_keyframes;

  /*** gotten by Finalize() **************/
  int max_area_index{-1};
  int max_area_id{-1};
  float max_area{-1.0f};

  int min_viewing_angle_index{-1};
  int min_viewing_angle_id{-1};
  float min_viewing_angle = std::numeric_limits<float>::max();

  int min_distance_index{-1};
  int min_distance_id{-1};
  float min_distance = std::numeric_limits<float>::max();

  // TODO
  /*
  Eigen::Vector3f mean_color;
  Eigen::Vector3f median_color;
  Eigen::Vector3f best_color_viewing_angle;
  Eigen::Vector3f mean_color_viewing_angle;
  Eigen::Vector3f median_color_viewing_angle;
  */
  /**************************************/

  FaceInfo();
  ~FaceInfo();
  void Update(const FaceInfoPerKeyframe& info);
  void CalcStat();
};

struct VisibilityInfo {
  bool has_vertex_stat{false};
  bool has_face_stat{false};
  std::vector<VertexInfo> vertex_info_list;  // correspond to input mesh vertex
  std::vector<FaceInfo> face_info_list;      // correspond to input face vertex

  VisibilityInfo();
  explicit VisibilityInfo(const Mesh& mesh);
  ~VisibilityInfo();
  void CalcStatVertexInfo();
  void CalcStatFaceInfo();
  void Update(int vertex_id, const VertexInfoPerKeyframe& info);
  void Update(int face_id, const FaceInfoPerKeyframe& info);
  std::string SerializeAsJson() const;
};

struct Keyframe {
  int id;

  std::string color_path;
  Image3b color;
  std::string depth_path;
  Image1f depth;
  std::string mask_path;
  Image1b mask;

  std::shared_ptr<const Camera> camera;

  Keyframe();
  ~Keyframe();
};

class VisibilityTester {
  bool mesh_initialized_{false};
  std::shared_ptr<const Keyframe> keyframe_{nullptr};
  std::shared_ptr<const Mesh> mesh_{nullptr};
  VisibilityTesterOption option_;

  std::vector<float> flatten_vertices_;
  std::vector<unsigned int> flatten_faces_;

  nanort::BVHBuildOptions<float> build_options_;
  std::unique_ptr<nanort::TriangleMesh<float>> triangle_mesh_;
  std::unique_ptr<nanort::TriangleSAHPred<float>> triangle_pred_;
  nanort::BVHAccel<float> accel_;
  nanort::BVHBuildStatistics stats_;
  float bmin_[3], bmax_[3];

  bool ValidateAndInitBeforeTest(VisibilityInfo* info) const;

  bool TestVertices(VisibilityInfo* info) const;
  bool TestFaces(VisibilityInfo* info) const;

 public:
  VisibilityTester();
  ~VisibilityTester();

  // Set option
  explicit VisibilityTester(const VisibilityTesterOption& option);
  void set_option(const VisibilityTesterOption& option);

  // Set mesh
  void set_mesh(std::shared_ptr<const Mesh> mesh);

  // Should call after set_mesh() and before Render()
  // Don't modify mesh outside after calling PrepareMesh()
  bool PrepareMesh();

  // Set keyframe
  void set_keyframe(std::shared_ptr<Keyframe> keyframe);

  // run visibility test
  bool Test(VisibilityInfo* info) const;
  bool Test(std::vector<std::shared_ptr<Keyframe>> keyframes,
            VisibilityInfo* info);
};

}  // namespace ugu

#endif