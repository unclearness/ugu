#include <stdio.h>

#include <fstream>

#include "example_utils.h"
#include "ugu/camera.h"
#include "ugu/stereo/base.h"
#include "ugu/timer.h"
#include "ugu/util/rgbd_util.h"
#include "ugu/util/image_util.h"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"

// test by bunny data
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  ugu::Timer<> timer;
  std::string rendered_dir = ugu_example::GetEx02RenderedBunnyDir();
  std::string out_dir = ugu_example::GetOutDir("ex10_stereo");

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 0.5f;  // scale to smaller size from VGA
  // int width = static_cast<int>(640 * r);
  // int height = static_cast<int>(480 * r);
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
  std::string left_path = rendered_dir + "00000_color.png";
  if (!ugu::FileExists(left_path)) {
    printf("Please run ex02_renderer first to generate %s\n",
           left_path.c_str());
    return -1;
  }
  left_c = ugu::Imread<ugu::Image3b>(left_path);
  right_c = ugu::Imread<ugu::Image3b>(rendered_dir + "r_00000_color.png");

#if 0
  std::string data_dir = ugu_example::GetDataDir("scenes2005/Art");
  left_c = ugu::Imread<ugu::Image3b>(data_dir + "view1.png");
  right_c = ugu::Imread<ugu::Image3b>(data_dir + "view5.png");
#endif

  ugu::Image1b left, right;
  ugu::Color2Gray(left_c, &left);
  ugu::Color2Gray(right_c, &right);

  ugu::Image1f disparity, cost, depth;

  ugu::Image1b vis_depth;

  const float kMaxConnectZDiff = 100.0f;
  std::shared_ptr<ugu::Camera> camera = std::make_shared<ugu::PinholeCamera>(
      left_c.cols, left_c.rows, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  ugu::Image1b lcensus, rcensus;
  ugu::CensusTransform8u(left, &lcensus);
  ugu::imwrite(out_dir + "lcensus.png", lcensus);
  ugu::CensusTransform8u(right, &rcensus);
  ugu::imwrite(out_dir + "rcensus.png", rcensus);

#if 1
  {
    param.census_transform = true;
    param.kernel = 3;
    param.cost = ugu::StereoCost::HAMMING;
    param.max_disparity = 50;
    timer.Start();
    ugu::ComputeStereoBruteForceCensus(lcensus, rcensus, &disparity, &cost,
                                       &depth, param);
    timer.End();
    ugu::LOGI("ComputeStereoBruteForceCensus: %f ms\n", timer.elapsed_msec());

    ugu::Depth2Gray(depth, &vis_depth);
    ugu::imwrite(out_dir + "naivecensus_vis_depth.png", vis_depth);

    ugu::Mesh view_mesh, view_point_cloud;
    ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
    ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
    view_point_cloud.WritePly(out_dir + "naivecensus_mesh.ply");
    view_mesh.WriteObj(out_dir, "naivecensus_mesh");
  }
#endif

#if 1
  {
    ugu::SgmParam sgm_param;

    sgm_param.base_param = param;
    timer.Start();
    ugu::ComputeStereoSgm(left, right, &disparity, &cost, &depth, sgm_param);
    timer.End();
    ugu::LOGI("ComputeStereoSgm: %f ms\n", timer.elapsed_msec());

    ugu::Depth2Gray(depth, &vis_depth);
    ugu::imwrite(out_dir + "sgm_vis_depth.png", vis_depth);

    ugu::Mesh view_mesh, view_point_cloud;
    ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
    ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
    view_point_cloud.WritePly(out_dir + "sgm_mesh.ply");
    view_mesh.WriteObj(out_dir, "sgm_mesh");
  }
#endif

#if 1
  {
    param.census_transform = false;
    param.kernel = 35;
    param.cost = ugu::StereoCost::SAD;
    param.max_disparity = -1.0f;
    timer.Start();
    ugu::ComputeStereoBruteForce(left, right, &disparity, &cost, &depth, param);
    timer.End();
    ugu::LOGI("ComputeStereoBruteForce: %f ms\n", timer.elapsed_msec());

    ugu::Depth2Gray(depth, &vis_depth);
    ugu::imwrite(out_dir + "naivesad_vis_depth.png", vis_depth);

    ugu::Mesh view_mesh, view_point_cloud;
    ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
    ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
    view_point_cloud.WritePly(out_dir + "naivesad_mesh.ply");
    view_mesh.WriteObj(out_dir, "naivesad_mesh");
  }
#endif

  ugu::PatchMatchStereoParam pmparam;
  pmparam.base_param = param;
  pmparam.patch_size = 35;
  // The library dumps debug images to the CWD when enabled
  pmparam.debug = false;
  ugu::Image1f rdisparity, rcost;

  timer.Start();

  ugu::ComputePatchMatchStereo(left_c, right_c, &disparity, &cost, &rdisparity,
                               &rcost, &depth, pmparam);
  timer.End();
  ugu::LOGI("ComputePatchMatchStereo: %f ms\n", timer.elapsed_msec());
  ugu::Depth2Gray(depth, &vis_depth);
  ugu::imwrite(out_dir + "pmstereo_vis_depth.png", vis_depth);

  {
    ugu::Mesh view_mesh, view_point_cloud;
    ugu::Depth2Mesh(depth, left_c, *camera, &view_mesh, kMaxConnectZDiff);
    ugu::Depth2PointCloud(depth, left_c, *camera, &view_point_cloud);
    view_point_cloud.WritePly(out_dir + "pmstereo_mesh.ply");
    view_mesh.WriteObj(out_dir, "pmstereo_mesh");
  }

  return 0;
}
