/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <algorithm>
#include <iostream>

#include "cxxopts.hpp"
#include "ugu/correspondence/correspondence_finder.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/io_util.h"
#include "ugu/util/path_util.h"
#include "ugu/util/string_util.h"
#include "ugu/util/thread_util.h"

#ifdef UGU_USE_JSON
#include "nlohmann/json.hpp"
#endif

#if 0
  // to_json() + 
 //  nlohmann::json j(correspondences);
 // didn't work...
void to_json(nlohmann::json& j, const ugu::Corresp& obj) {
  j = nlohmann::json{{"fid", obj.fid},
                     {"uv", {obj.uv[0], obj.uv[1]}},
                     {"p",
                      {
                          obj.p[0],
                          obj.p[1],
                          obj.p[2],
                      }},
                     {"singed_dist", obj.singed_dist}};
}
#endif

int main(int argc, char* argv[]) {
  cxxopts::Options options("textrans",
                           "Texture transfer for almost aligned meshes");
  options.positional_help("[optional args]").show_positional_help();

  options.add_options()("s,src", "Src points (.obj)",
                        cxxopts::value<std::string>())(
      "d,dst", "Dst mesh (.obj)", cxxopts::value<std::string>())(
      "o,out", "Output directory",
      cxxopts::value<std::string>()->default_value("./"))(
      "b,base", "Output basename",
      cxxopts::value<std::string>()->default_value("out"))
      ("v,verbose", "Verbose", cxxopts::value<bool>()->default_value("false"))(
      "nn_num", "NN parameter", cxxopts::value<int>()->default_value("10"))(
      "num_threads", "#threads", cxxopts::value<int>()->default_value("-1"))(
      "vis_max", "maximam value for visualization", cxxopts::value<float>()->default_value("-1.f"))(
      "vis_min", "minimum value for visualization", cxxopts::value<float>()->default_value("1.f"))(
      "h,help", "Print usage");

  options.parse_positional({"src", "dst"});

  auto result = options.parse(argc, argv);

  if (result.count("help") > 0 || result.count("src") < 1 ||
      result.count("dst") < 1) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  const std::string& src_obj_path = result["src"].as<std::string>();
  const std::string& dst_obj_path = result["dst"].as<std::string>();
  const std::string& basename = result["base"].as<std::string>();
  const std::string& out_dir = result["out"].as<std::string>();
  bool verbose = result["verbose"].as<bool>();
  int nn_num = result["nn_num"].as<int>();
  int num_threads = result["num_threads"].as<int>();
  float vis_max = result["vis_max"].as<float>();
  float vis_min = result["vis_min"].as<float>();

  if (ugu::ExtractExt(src_obj_path) != "obj" ||
      ugu::ExtractExt(dst_obj_path) != "obj") {
    std::cout << "src and dst must be .obj" << std::endl;
    exit(0);
  }

  ugu::EnsureDirExists(out_dir);

  ugu::Timer<> timer;
  ugu::Mesh src_mesh, dst_mesh;

  timer.Start();
  std::string src_dir = ugu::ExtractDir(src_obj_path);
  std::string dst_dir = ugu::ExtractDir(dst_obj_path);
  src_mesh.LoadObj(src_obj_path, src_dir);
  dst_mesh.LoadObj(dst_obj_path, dst_dir);
  timer.End();
  if (verbose) {
    ugu::LOGI("Load mesh: %f ms\n", timer.elapsed_msec());
  }

  timer.Start();
  ugu::CorrespFinderPtr coresp_finder =
      ugu::KDTreeCorrespFinder::Create(nn_num);
  coresp_finder->Init(dst_mesh.vertices(), dst_mesh.vertex_indices());
  timer.End();
  if (verbose) {
    ugu::LOGI("Init: %f ms\n", timer.elapsed_msec());
  }

  timer.Start();
  std::vector<ugu::Corresp> correspondences(src_mesh.vertices().size());
  const Eigen::Vector3f stub_normal(0.f, 0.f, 1.f);
  auto func = [&](size_t i) {
    correspondences[i] =
        coresp_finder->Find(src_mesh.vertices()[i], stub_normal);
  };
  ugu::parallel_for(size_t(0), src_mesh.vertices().size(), func, num_threads);
  timer.End();
  if (verbose) {
    ugu::LOGI("Get correspondences: %f ms\n", timer.elapsed_msec());
  }

  timer.Start();

  // signed distance -> .txt
  std::vector<float> signed_dists;
  std::transform(correspondences.begin(), correspondences.end(),
                 std::back_inserter(signed_dists),
                 [&](const ugu::Corresp& c) { return c.singed_dist; });
  ugu::WriteTxt(out_dir + "/" + basename + ".txt", signed_dists);

  // all info -> .json
  {
#ifdef UGU_USE_JSON
    std::ofstream ofs(out_dir + "/" + basename + ".json");
    nlohmann::json j;
    for (const auto& c : correspondences) {
      nlohmann::json jj = nlohmann::json{{"fid", c.fid},
                                         {"uv", {c.uv[0], c.uv[1]}},
                                         {"p",
                                          {
                                              c.p[0],
                                              c.p[1],
                                              c.p[2],
                                          }},
                                         {"singed_dist", c.singed_dist}};
      j.push_back(jj);
    }
    ofs << std::setw(4) << j.dump(-1, ' ', true) << std::endl;
#endif
  }
  // Visualization -> .ply/.obj
  {
    if (vis_max <= 0.f) {
      vis_max = std::max(
          *std::max_element(signed_dists.begin(), signed_dists.end()), 1e-10f);
    }
    if (vis_min >= 0.f) {
      vis_min = std::min(
          *std::min_element(signed_dists.begin(), signed_dists.end()), -1e-10f);
    }

    std::vector<Eigen::Vector3f> colors;
    for (size_t i = 0; i < correspondences.size(); i++) {
      float d = correspondences[i].singed_dist;
      Eigen::Vector3f c(255.f, 255.f, 255.f);
      if (d > 0.f) {
        c[0] = std::min(d, vis_max) / vis_max * 255.f;
        c[1] = 0.f;
        c[2] = 0.f;
      } else {
        c[0] = 0.f;
        c[1] = 0.f;
        c[2] = std::max(d, vis_min) / vis_min * 255.f;
      }
      colors.push_back(c);
    }

    src_mesh.set_vertex_colors(colors);
    src_mesh.WritePly(out_dir + "/" + basename + ".ply");
    src_mesh.WriteObj(out_dir, basename);
  }

  timer.End();
  if (verbose) {
    ugu::LOGI("File output: %f ms\n", timer.elapsed_msec());
  }

  return 0;
}