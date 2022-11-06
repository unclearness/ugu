/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <iostream>

#include "cxxopts.hpp"
#include "ugu/image_io.h"
#include "ugu/image_proc.h"
#include "ugu/inpaint/inpaint.h"
#include "ugu/textrans/texture_transfer.h"
#include "ugu/timer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/io_util.h"

namespace {

std::string Path2Dir(const std::string& path) {
  std::string::size_type pos = std::string::npos;
  const std::string::size_type unix_pos = path.find_last_of('/');
  const std::string::size_type windows_pos = path.find_last_of('\\');
  if (unix_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = unix_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  if (windows_pos != std::string::npos) {
    if (pos == std::string::npos) {
      pos = windows_pos;
    } else {
      pos = std::max(pos, unix_pos);
    }
  }
  return (pos == std::string::npos) ? "./" : path.substr(0, pos + 1);
}

std::string GetExt(const std::string& path) {
  size_t ext_i = path.find_last_of(".");
  std::string extname = path.substr(ext_i, path.size() - ext_i);
  return extname;
}

#ifdef UGU_USE_OPENCV
cv::Mat3f LoadTex(const std::string& path) {
  cv::Mat tmp = cv::imread(path, -1);

  if (tmp.type() == CV_32FC3) {
    return tmp;
  }

  if (tmp.type() == CV_8UC3 || tmp.type() == CV_8UC4) {
    tmp.clone().convertTo(tmp, CV_32FC3);
    return tmp;
  }

  throw std::runtime_error(
      "Input texture format is not supported.Please input 32bit float 3ch or "
      "8bit uint8 3ch/4ch");
}
#else
ugu::Image3f LoadTex(const std::string& path) {
  ugu::Image3b tmp = ugu::Imread<ugu::Image3b>(path, -1);
  ugu::Image3f tmp_f;

  ugu::ConvertTo(tmp, &tmp_f);

  return tmp_f;
}
#endif

template <typename T>
void SaveFloatAndUint8Image(const std::string& out_basename,
                            const std::string& ext, const T& org,
                            bool is_float_input, const ugu::Image3b& vis) {
  std::string out_path = out_basename + ".png";
  if (is_float_input) {
    ugu::imwrite(out_basename + ext, org);
    std::string out_path2 = out_basename + "_vis.png";
    ugu::imwrite(out_path2, vis);
  } else {
    ugu::imwrite(out_path, vis);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  cxxopts::Options options("textrans",
                           "Texture transfer for almost aligned meshes");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src mesh (.obj)",
                        cxxopts::value<std::string>())(
      "t,src_tex", "Src tex", cxxopts::value<std::string>()->default_value(""))(
      "d,dst", "Dst mesh (.obj)", cxxopts::value<std::string>())(
      "o,out", "Output directory",
      cxxopts::value<std::string>()->default_value("./"))(
      "b,base", "Output basename",
      cxxopts::value<std::string>()->default_value("out"))(
      "width", "Output texture width",
      cxxopts::value<int>()->default_value("512"))(
      "height", "Output texture height",
      cxxopts::value<int>()->default_value("512"))(
      "m,map", "Mapping table",
      cxxopts::value<std::string>()->default_value(""))(
      "i,inpaint", "Enable inpaint",
      cxxopts::value<bool>()->default_value("false"))(
      "v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "nn_sampling", "Enable NearestNeighbor sampling",
      cxxopts::value<bool>()->default_value("false"))(
      "nn_num", "NN parameter", cxxopts::value<int>()->default_value("10"))(
      "h,help", "Print usage");

  options.parse_positional({"src", "dst"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1 ||
      result.count("dst") < 1) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  const std::string& src_obj_path = result["src"].as<std::string>();
  const std::string& src_tex_path = result["src_tex"].as<std::string>();
  const std::string& dst_obj_path = result["dst"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  int width = result["width"].as<int>();
  int height = result["height"].as<int>();
  const std::string& srcpos = result["map"].as<std::string>();
  bool inpaint = result["inpaint"].as<bool>();
  bool verbose = result["verbose"].as<bool>();
  bool nn_sampling = result["nn_sampling"].as<bool>();
  int nn_num = result["nn_num"].as<int>();

  ugu::Timer<> timer;
  ugu::Mesh src_mesh, dst_mesh;

  std::string src_dir = Path2Dir(src_obj_path);
  std::string dst_dir = Path2Dir(dst_obj_path);

  src_mesh.LoadObj(src_obj_path, src_dir);
  dst_mesh.LoadObj(dst_obj_path, dst_dir);

  std::string src_ext;

  ugu::Image3f src_tex;
  if (src_tex_path != "") {
    src_tex = LoadTex(src_tex_path);
    src_ext = GetExt(src_tex_path);
  } else {
    ugu::ConvertTo(src_mesh.materials()[0].diffuse_tex, &src_tex);
    src_ext = GetExt(src_mesh.materials()[0].diffuse_texpath);
  }

  bool is_float_input = (src_ext == ".exr" || src_ext == ".hdr");

  ugu::TexTransNoCorrespOutput output;

  int interp = ugu::InterpolationFlags::INTER_LINEAR;
  if (nn_sampling) {
    interp = ugu::InterpolationFlags::INTER_NEAREST;
  }

  std::string out_basename = out_dir + "/" + basename;

  if (srcpos == "") {
    timer.Start();
    ugu::TexTransNoCorresp(src_tex, src_mesh, dst_mesh, height, width, output,
                           interp, nn_num);
    timer.End();
    if (verbose) {
      ugu::LOGI("TexTransNoCorresp: %f ms", timer.elapsed_msec());
    }
  } else {
    ugu::Image3f remaped;
    ugu::Remap(src_tex, output.srcpos_tex, output.dst_mask, remaped);
    ugu::Image3b remaped_vis;
    ugu::ConvertTo(remaped, &remaped_vis);
    ugu::imwrite(out_basename + "_remap.png", remaped_vis);
  }

  // ugu::Image3b dst_tex_vis;
  // ugu::ConvertTo(output.dst_tex, &dst_tex_vis);
  // ugu::imwrite(out_basename + src_ext, output.dst_tex);

  ugu::Image3b dst_tex_vis;
  if (is_float_input) {
    ugu::ConvertTo(output.dst_tex, &dst_tex_vis, 255.f);
  } else {
    ugu::ConvertTo(output.dst_tex, &dst_tex_vis, 1.f);
  }
  SaveFloatAndUint8Image(out_basename, src_ext, output.dst_tex, is_float_input,
                         dst_tex_vis);

  if (inpaint) {
    ugu::Image1b inpaint_mask;
    ugu::Not(output.dst_mask, &inpaint_mask);
    // ugu::Dilate(inpaint_mask.clone(), &inpaint_mask, 3);
    ugu::Image3f dst_tex_inpainted = output.dst_tex.clone();

    ugu::Inpaint(inpaint_mask, dst_tex_inpainted, 3.f,
                 ugu::InpaintMethod::NAIVE);

    ugu::Image3b dst_tex_inpainted_vis;
    if (is_float_input) {
      ugu::ConvertTo(dst_tex_inpainted, &dst_tex_inpainted_vis, 255.f);
    } else {
      ugu::ConvertTo(dst_tex_inpainted, &dst_tex_inpainted_vis, 1.f);
    }
    SaveFloatAndUint8Image(out_basename + "_inpainted", src_ext,
                           dst_tex_inpainted, is_float_input,
                           dst_tex_inpainted_vis);
  }

  if (verbose) {
    ugu::imwrite(out_basename + "_mask.png", output.dst_mask);

#ifdef UGU_USE_OPENCV
    std::string additional_info_raw_ext = ".exr";
#else
    std::string additional_info_raw_ext = ".bin";
#endif

    ugu::Image3b srcpos_tex_vis =
        ugu::ColorizeImagePosMap(output.srcpos_tex, src_tex.cols, src_tex.rows);
    SaveFloatAndUint8Image(out_basename + "_srcpos", additional_info_raw_ext,
                           output.srcpos_tex, true, srcpos_tex_vis);

    ugu::Image3b nn_fid_tex_vis;
    ugu::FaceId2Color(output.nn_fid_tex, &nn_fid_tex_vis);
    ugu::imwrite(out_basename + "_fid_vis.png", nn_fid_tex_vis);
    ugu::WriteFaceIdAsText(output.nn_fid_tex, out_basename + "_fid.txt");

    ugu::Image3b nn_pos_tex_vis = ugu::ColorizePosMap(output.nn_pos_tex);
    SaveFloatAndUint8Image(out_basename + "_pos", additional_info_raw_ext,
                           output.nn_pos_tex, true, nn_pos_tex_vis);

    ugu::Image3b nn_bary_tex_vis = ugu::ColorizeBarycentric(output.nn_bary_tex);
    SaveFloatAndUint8Image(out_basename + "_bary", additional_info_raw_ext,
                           output.nn_bary_tex, true, nn_bary_tex_vis);
  }

  return 0;
}