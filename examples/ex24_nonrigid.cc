/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include <filesystem>
#include <iostream>
#include <random>

#include "ugu/inpaint/inpaint.h"
#include "ugu/registration/nonrigid.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/io_util.h"
#include "ugu/util/path_util.h"
#include "ugu/util/string_util.h"

namespace {

void TestObject() {
  std::string src_dir = "../data/sphere/";
  std::string src_obj_path = src_dir + "icosphere5_smart_uv.obj";
  ugu::Mesh src_mesh;
  src_mesh.LoadObj(src_obj_path, src_dir);

  std::string dst_dir = "../data/spot/";
  std::string dst_obj_path = dst_dir + "spot_triangulated.obj";
  ugu::Mesh dst_mesh;
  dst_mesh.LoadObj(dst_obj_path, dst_dir);

  std::string out_dir = "../out/ex24/object";
  ugu::EnsureDirExists(out_dir);

  // Roughly align scale
  src_mesh.CalcStats();
  auto src_stats = src_mesh.stats();
  dst_mesh.CalcStats();
  auto dst_stats = dst_mesh.stats();
  // Eigen::Vector3f dst_center = (dst_stats.bb_max - dst_stats.bb_min) / 2.0;

  Eigen::Vector3f src_size = (src_stats.bb_max - src_stats.bb_min);
  Eigen::Vector3f dst_size = (dst_stats.bb_max - dst_stats.bb_min);
  Eigen::Vector3f src2dst_scale =
      dst_size.cwiseProduct(src_size.cwiseInverse());

  src2dst_scale.setOnes();

  src_mesh.Scale(src2dst_scale);
  src_mesh.WriteObj(out_dir, "0_scale");

  src_mesh.CalcStats();
  src_stats = src_mesh.stats();
  // Eigen::Vector3f src_center = (src_stats.bb_max - src_stats.bb_min) / 2.0;

  src_mesh.Translate(dst_stats.center - src_stats.center);

  src_mesh.WriteObj(out_dir, "0_trans");

#if 0
  // Apply Rigid ICP
  ugu::IcpOutput icp_output;
  ugu::RigidIcp(src_mesh, dst_mesh, ugu::IcpLossType::kPointToPlane,
                ugu::IcpTerminateCriteria(), icp_output);

  {
    ugu::Mesh src_similarity = src_mesh;
    src_similarity.Transform(icp_output.transform_histry.back().cast<float>());
    src_similarity.WriteObj(out_dir, "1_rigid");
  }
#endif

  ugu::NonRigidIcp nicp;
  // nicp.SetThreadNum(1);

  // nicp.SetSrc(src_mesh, icp_output.transform_histry.back().cast<float>());
  nicp.SetSrc(src_mesh);
  nicp.SetDst(dst_mesh);

  nicp.Init(false, 0.65f, false);

  double max_alpha = 10.0;
  double min_alpha = 0.1;
  double gamma = 1.0;
  int step = 100;
  // double decay_rate = 0.95;

  for (int i = 1; i <= step; ++i) {
    double alpha = max_alpha - i * (max_alpha - min_alpha) / step;
    // double alpha = max_alpha * std::pow(decay_rate, i) + min_alpha;

    ugu::LOGI("Iteration %d with alpha %f\n", i, alpha);

    nicp.Registrate(alpha, gamma);

    if (i % 10 == 0) {
      ugu::MeshPtr deformed = nicp.GetDeformedSrc();
      deformed->WriteObj(out_dir, "2_nonrigid_" + ugu::zfill(i, 3));
    }
  }
}

void TestFace() {
  std::string src_dir = "../data/face/";
  std::string src_obj_path = src_dir + "mediapipe_face.obj";
  ugu::Mesh src_mesh;
  src_mesh.LoadObj(src_obj_path, src_dir);
  std::vector<ugu::PointOnFace> src_landmarks =
      ugu::LoadPoints(src_dir + "mediapipe_face_landmarks.json",
                      ugu::PointOnFaceType::NAMED_POINT_ON_TRIANGLE);
  std::vector<Eigen::Vector3f> src_landmark_positions;
  for (const auto& pof : src_landmarks) {
    const auto& face = src_mesh.vertex_indices()[pof.fid];
    auto pos = pof.u * src_mesh.vertices()[face[0]] +
               pof.v * src_mesh.vertices()[face[1]] +
               (1.f - pof.u - pof.v) * src_mesh.vertices()[face[2]];
    src_landmark_positions.push_back(pos);
  }

  std::string dst_dir = "../data/face/lpshead/";
  std::string dst_obj_path = dst_dir + "head_triangulated.obj";
  ugu::Mesh dst_mesh;
  dst_mesh.LoadObj(dst_obj_path, dst_dir);
  std::vector<ugu::PointOnFace> dst_landmarks =
      ugu::LoadPoints(dst_dir + "head_triangulated_landmarks.json",
                      ugu::PointOnFaceType::NAMED_POINT_ON_TRIANGLE);
  std::vector<Eigen::Vector3f> dst_landmark_positions;
  for (const auto& pof : dst_landmarks) {
    const auto& face = dst_mesh.vertex_indices()[pof.fid];
    auto pos = pof.u * dst_mesh.vertices()[face[0]] +
               pof.v * dst_mesh.vertices()[face[1]] +
               (1.f - pof.u - pof.v) * dst_mesh.vertices()[face[2]];
    dst_landmark_positions.push_back(pos);
  }

  // Rigid alignment by landmarks
  Eigen::Affine3d transform = ugu::FindSimilarityTransformFrom3dCorrespondences(
      src_landmark_positions, dst_landmark_positions);

  std::string out_dir = "../out/ex24/face/";
  ugu::EnsureDirExists(out_dir);

  src_mesh.WriteObj(out_dir, "0_init_src");
  dst_mesh.WriteObj(out_dir, "0_init_dst");

  ugu::Mesh src_similarity = ugu::Mesh(src_mesh);
  src_similarity.Transform(transform.cast<float>());
  src_similarity.WriteObj(out_dir, "1_similarity");

  {
    ugu::Mesh tmp;
    tmp.set_vertices(src_landmark_positions);
    tmp.WritePly(out_dir + "0_init_src_landmarks.ply");
    tmp.set_vertices(dst_landmark_positions);
    tmp.WritePly(out_dir + "0_init_dst_landmarks.ply");
  }

  ugu::NonRigidIcp nicp;

  nicp.SetSrc(src_mesh, transform.cast<float>());
  double beta = 0.1;
  std::vector<double> betas(src_landmarks.size(), beta);
  nicp.SetSrcLandmarks(src_landmarks, betas);
  nicp.SetDst(dst_mesh);
  nicp.SetDstLandmarkPositions(dst_landmark_positions);

  bool keep_src_boundary_as_possible = true;
  nicp.Init(false, 0.65f, false, keep_src_boundary_as_possible);

  double max_alpha = 10.0;
  double min_alpha = 0.1;
  double gamma = 1.0;
  int step = 10;
  ugu::MeshPtr deformed;
  for (int i = 1; i <= step; ++i) {
    double alpha = max_alpha - i * (max_alpha - min_alpha) / step;

    ugu::LOGI("Iteration %d with alpha %f\n", i, alpha);

    nicp.Registrate(alpha, gamma);

    if (i % 1 == 0) {
      deformed = nicp.GetDeformedSrc();
      deformed->WriteObj(out_dir, "2_nonrigid_" + ugu::zfill(i, 3));
    }
  }

  // Texture transfer
  ugu::Image3f dst_tex;
  dst_mesh.materials()[0].diffuse_tex.convertTo(dst_tex, CV_32FC3);
  ugu::TexTransNoCorrespOutput textrans_out;
  ugu::TexTransNoCorresp(dst_tex, dst_mesh, *deformed, 512, 512, textrans_out);

  auto mats = deformed->materials();
  textrans_out.dst_tex.convertTo(mats[0].diffuse_tex, CV_8UC3);

  ugu::Image1b inpaint_mask;
  ugu::Not(textrans_out.dst_mask, &inpaint_mask);
  // ugu::Dilate(inpaint_mask.clone(), &inpaint_mask, 3);
  ugu::Image3b dst_tex_inpainted = mats[0].diffuse_tex.clone();
  ugu::Inpaint(inpaint_mask, dst_tex_inpainted, 3.f);

  mats[0].diffuse_tex = dst_tex_inpainted;

  deformed->set_materials(mats);

  deformed->WriteObj(out_dir, "3_textrans");
}

}  // namespace

int main() {
  ugu::EnsureDirExists("../out/");
  ugu::EnsureDirExists("../out/ex24");

  //TestObject();

  TestFace();

  return 0;
}
