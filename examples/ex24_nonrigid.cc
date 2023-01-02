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

  ugu::NonRigidIcp nicp;

  nicp.SetSrc(src_mesh);
  nicp.SetDst(dst_mesh);

  nicp.Init();

  double max_alpha = 20.0;
  double min_alpha = 2.0;
  double beta = 10.0;
  double gamma = 1.0;
  int step = 100;
  double decay_rate = 0.95;

  for (int i = 1; i <= step; ++i) {
    double alpha = max_alpha - i * (max_alpha - min_alpha) / step;
    // double alpha = max_alpha * std::pow(decay_rate, i) + min_alpha;

    ugu::LOGI("Iteration %d with alpha %f\n", i, alpha);

    nicp.Registrate(alpha, beta, gamma);

    if (i % 10 == 0) {
      ugu::MeshPtr deformed = nicp.GetDeformedSrc();
      deformed->WriteObj(out_dir, "deformed_" + ugu::zfill(i, 2));
    }
  }
}

void TestFace() {
  std::string src_dir = "../data/face/";
  std::string src_obj_path = src_dir + "mediapipe_face.obj";
  ugu::Mesh src_mesh;
  src_mesh.LoadObj(src_obj_path, src_dir);
  std::vector<int> src_landmark_vids =
      ugu::LoadTxtAsVector<int>(src_dir + "mediapipe_face_landmarks_8.txt");
  std::vector<Eigen::Vector3f> src_landmark_positions;
  for (const auto& idx : src_landmark_vids) {
    src_landmark_positions.push_back(src_mesh.vertices()[idx]);
  }

  std::string dst_dir = "../data/face/lpshead/";
  std::string dst_obj_path = dst_dir + "head_triangulated.obj";
  ugu::Mesh dst_mesh;
  dst_mesh.LoadObj(dst_obj_path, dst_dir);
  std::vector<int> dst_landmark_vids =
      ugu::LoadTxtAsVector<int>(dst_dir + "head_triangulated_landmarks_8.txt");
  std::vector<Eigen::Vector3f> dst_landmark_positions;
  for (const auto& idx : dst_landmark_vids) {
    dst_landmark_positions.push_back(dst_mesh.vertices()[idx]);
  }

  // Rigid alignment by landmarks
  Eigen::Affine3d transform = ugu::FindSimilarityTransformFrom3dCorrespondences(
      src_landmark_positions, dst_landmark_positions);

  std::string out_dir = "../out/ex24/face/";
  ugu::EnsureDirExists(out_dir);

  src_mesh.WriteObj(out_dir, "0_init_src");
  dst_mesh.WriteObj(out_dir, "0_init_dst");

  ugu::Mesh src_similarity = src_mesh;
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
  nicp.SetSrcLandmakrVertexIds(src_landmark_vids);
  nicp.SetDst(dst_mesh);
  nicp.SetDstLandmakrVertexIds(dst_landmark_vids);

  nicp.Init();

  double max_alpha = 50.0;
  double min_alpha = 1.0;
  double beta = 1.0;
  double gamma = 1.0;
  int step = 100;
  ugu::MeshPtr deformed;
  for (int i = 1; i <= step; ++i) {
    double alpha = max_alpha - i * (max_alpha - min_alpha) / step;

    ugu::LOGI("Iteration %d with alpha %f\n", i, alpha);

    nicp.Registrate(alpha, beta, gamma);

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

  TestObject();

  TestFace();

  return 0;
}
