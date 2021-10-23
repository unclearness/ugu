/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "ugu/inpaint/inpaint.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/io_util.h"

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir = "../data/spot/";
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
      ugu::ColorizeImagePosMap(output.srcpos_tex, src_tex.cols, src_tex.rows);
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
  ugu::WriteFaceIdAsText(output.nn_fid_tex, out_basename + "_fid.txt");

  ugu::Image3b nn_pos_tex_vis = ugu::ColorizePosMap(output.nn_pos_tex);
  ugu::imwrite(out_basename + "_pos.png", nn_pos_tex_vis);

  ugu::Image3b nn_bary_tex_vis = ugu::ColorizeBarycentric(output.nn_bary_tex);
  ugu::imwrite(out_basename + "_bary.png", nn_bary_tex_vis);

  return 0;
}
