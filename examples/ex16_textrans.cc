/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/inpaint/inpaint.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/find_correspondence/";
  std::string src_obj_path = data_dir + "spot_triangulated.obj";
  std::string dst_obj_path = data_dir + "spot_remesh.obj";
  ugu::Timer<> timer;
  ugu::Mesh src_mesh, dst_mesh;
  src_mesh.LoadObj(src_obj_path, data_dir);
  dst_mesh.LoadObj(dst_obj_path, data_dir);

  ugu::Image3f src_tex;
  ugu::ConvertTo(src_mesh.materials()[0].diffuse_tex, &src_tex);

  ugu::TexTransNoCorrespOutput output;

  timer.Start();
  ugu::TexTransNoCorresp(src_tex, src_mesh, dst_mesh, 1024, 1024, output);
  timer.End();
  ugu::LOGI("TexTransNoCorresp: %f ms", timer.elapsed_msec());

  std::string out_basename = data_dir + "spot_remesh_texture";
  ugu::Image3b dst_tex_vis;
  ugu::ConvertTo(output.dst_tex, &dst_tex_vis);
  ugu::imwrite(out_basename + "_org.png", dst_tex_vis);
  ugu::imwrite(out_basename + "_mask.png", output.dst_mask);

  ugu::Image3b srcpos_tex_vis =
      ugu::Image3b::zeros(output.srcpos_tex.rows, output.srcpos_tex.cols);
  for (int y = 0; y < srcpos_tex_vis.rows; y++) {
    for (int x = 0; x < srcpos_tex_vis.cols; x++) {
      auto& color = srcpos_tex_vis.at<ugu::Vec3b>(y, x);
      const auto& spos = output.srcpos_tex.at<ugu::Vec2f>(y, x);

      color[0] = static_cast<uint8_t>(spos[0] / src_tex.cols * 255);

      color[1] = static_cast<uint8_t>(spos[1] / src_tex.rows * 255);
    }
  }
  ugu::imwrite(out_basename + "_srcpos.png", srcpos_tex_vis);

  ugu::Image3f remaped;
  ugu::Remap(src_tex, output.srcpos_tex, output.dst_mask, remaped);
  ugu::Image3b remaped_vis;
  ugu::ConvertTo(remaped, &remaped_vis);
  ugu::imwrite(out_basename + "_remap.png", remaped_vis);

  ugu::Image1b inpaint_mask;
  ugu::Not(output.dst_mask, &inpaint_mask);
  // ugu::Dilate(inpaint_mask.clone(), &inpaint_mask, 3);
  ugu::Image3b dst_tex_vis_inpainted = dst_tex_vis.clone();
  ugu::Inpaint(inpaint_mask, dst_tex_vis_inpainted, 3.f);
  ugu::imwrite(out_basename + ".png", dst_tex_vis_inpainted);

  ugu::Image3b nn_fid_tex_vis;
  ugu::FaceId2Color(output.nn_fid_tex, &nn_fid_tex_vis);
  ugu::imwrite(out_basename + "_fid.png", nn_fid_tex_vis);

  std::array<float, 3> pos_min = {std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::max()};
  std::array<float, 3> pos_max = {std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::lowest()};
  ugu::Image3b nn_pos_tex_vis =
      ugu::Image3b::zeros(output.nn_pos_tex.rows, output.nn_pos_tex.cols);
  for (int y = 0; y < nn_pos_tex_vis.rows; y++) {
    for (int x = 0; x < nn_pos_tex_vis.cols; x++) {
      const auto& pos = output.nn_pos_tex.at<ugu::Vec3f>(y, x);
      for (int c = 0; c < 3; c++) {
        if (pos[c] < pos_min[c]) {
          pos_min[c] = pos[c];
        }
        if (pos_max[c] < pos[c]) {
          pos_max[c] = pos[c];
        }
      }
    }
  }
  for (int y = 0; y < nn_pos_tex_vis.rows; y++) {
    for (int x = 0; x < nn_pos_tex_vis.cols; x++) {
      auto& color = nn_pos_tex_vis.at<ugu::Vec3b>(y, x);
      const auto& pos = output.nn_pos_tex.at<ugu::Vec3f>(y, x);
      for (int c = 0; c < 3; c++) {
        color[c] = static_cast<uint8_t>((pos[c] - pos_min[c]) /
                                        (pos_max[c] - pos_min[c]) * 255);
      }
    }
  }

  ugu::imwrite(out_basename + "_pos.png", nn_pos_tex_vis);

  ugu::Image3b nn_bary_tex_vis =
      ugu::Image3b::zeros(output.nn_bary_tex.rows, output.nn_bary_tex.cols);
  ugu::ConvertTo(output.nn_bary_tex, &nn_bary_tex_vis, 255);
  for (int y = 0; y < nn_bary_tex_vis.rows; y++) {
    for (int x = 0; x < nn_bary_tex_vis.cols; x++) {
      nn_bary_tex_vis.at<ugu::Vec3b>(y, x)[0] = 0;
    }
  }

  ugu::imwrite(out_basename + "_bary.png", nn_bary_tex_vis);

  return 0;
}
