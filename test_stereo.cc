#include <stdio.h>
#include <fstream>

#include "ugu/camera.h"
#include "ugu/stereo/base.h"
#include "ugu/util.h"

// test by bunny data
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/bunny/";
  // data_dir = "../data/scenes2005/Art/";

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 0.5f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);

  ugu::StereoParam param;
  param.baseline = 50.0f;
  param.fx = focal_length.x();
  param.fy = focal_length.y();
  param.lcx = principal_point.x();
  param.rcx = principal_point.x();
  param.lcy = principal_point.y();
  param.rcy = principal_point.y();
  param.maxd = 1000.0f;

  ugu::Image3b left_c, right_c;
  left_c = ugu::imread<ugu::Image3b>(data_dir + "00000_color.png");
  right_c = ugu::imread<ugu::Image3b>(data_dir + "r_00000_color.png");

  // left_c = ugu::imread<ugu::Image3b>(data_dir + "view5.png");
  // right_c = ugu::imread<ugu::Image3b>(data_dir + "view1.png");

  ugu::Image1b left, right;
  Color2Gray(left_c, &left);
  Color2Gray(right_c, &right);

  ugu::Image1f disparity, cost, depth;

  ugu::Image1b vis_depth;

  const float kMaxConnectZDiff = 100.0f;
  std::shared_ptr<ugu::Camera> camera = std::make_shared<ugu::PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  ugu::ComputeStereoBruteForce(left, right, &disparity, &cost, &depth, param);

  Depth2Gray(depth, &vis_depth);
  ugu::imwrite(data_dir + "stereo_vis_depth.png", vis_depth);

  ugu::Mesh view_mesh, view_point_cloud;
  ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
  ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
  view_point_cloud.WritePly(data_dir + "stereo_mesh.ply");
  view_mesh.WriteObj(data_dir, "stereo_mesh");

  ugu::PatchMatchStereoParam pmparam;
  pmparam.base_param = param;
  ugu::Image1f rdisparity, rcost;
  ugu::ComputePatchMatchStereo(left_c, right_c, &disparity, &cost, &rdisparity,
                               &rcost, &depth, pmparam);
  Depth2Gray(depth, &vis_depth);
  ugu::imwrite(data_dir + "pmstereo_vis_depth.png", vis_depth);

  {
    ugu::Mesh view_mesh, view_point_cloud;
    ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
    ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
    view_point_cloud.WritePly(data_dir + "pmstereo_mesh.ply");
    view_mesh.WriteObj(data_dir, "pmstereo_mesh");
  }

  return 0;
}
