/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/inpaint/inpaint.h"
#include "ugu/parameterize/parameterize.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/raster_util.h"

namespace {}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;
  // std::string data_dir = "../data/sphere/";
  // std::string obj_path = data_dir + "icosphere3_smart_uv.obj";

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  // load mesh
  ugu::MeshPtr input_mesh = ugu::Mesh::Create();
  input_mesh->LoadObj(obj_path, data_dir);

  int tex_w = 512;
  int tex_h = 512;

  ugu::Timer<> timer;
  timer.Start();
  ugu::Parameterize(*input_mesh, tex_w, tex_h,
                    ugu::ParameterizeUvType::kSmartUv);
  timer.End();
  ugu::LOGI("Parameterize %f ms\n", timer.elapsed_msec());

  auto uv = ugu::DrawUv(input_mesh->uv(), input_mesh->uv_indices(),
                        {255, 255, 255}, {0, 0, 0});
  ugu::imwrite(data_dir + "uv.jpg", uv);

  auto [clusters, non_orphans, orphans, clusters_f] =
      ugu::ClusterByConnectivity(input_mesh->uv_indices(),
                                 static_cast<int32_t>(input_mesh->uv().size()));

  std::string diffuse_tex_name = "my_uv.png";
#if 1
  std::vector<Eigen::Vector3f> random_colors =
      ugu::GenRandomColors(clusters.size(), 0.f, 225.f);
  std::vector<int> material_ids(input_mesh->uv_indices().size());
  // std::transform(clusters.begin(), clusters.end(),
  //               std::back_inserter(material_ids),
  //               [&](uint32_t cid) { return int(cid); });
  for (size_t i = 0; i < clusters_f.size(); i++) {
    const auto& cf = clusters_f[i];
    for (const auto& f : cf) {
      material_ids[f] = i;
    }
  }
  std::vector<ugu::ObjMaterial> materials;
  for (size_t i = 0; i < clusters.size(); i++) {
    ugu::ObjMaterial mat;
    mat.name = "mat_" + std::to_string(i);
    mat.diffuse[0] = random_colors[i][0] / 255.f;
    mat.diffuse[1] = random_colors[i][1] / 255.f;
    mat.diffuse[2] = random_colors[i][2] / 255.f;
    mat.diffuse_texname = diffuse_tex_name;
    materials.push_back(mat);
  };

#endif

  std::vector<Eigen::Vector3f> face_colors(input_mesh->vertex_indices().size());
  for (size_t i = 0; i < face_colors.size(); i++) {
    face_colors[i][0] = materials[material_ids[i]].diffuse[0] * 255.f;
    face_colors[i][1] = materials[material_ids[i]].diffuse[1] * 255.f;
    face_colors[i][2] = materials[material_ids[i]].diffuse[2] * 255.f;
  }

  ugu::Image3b fc_rasterized;
  ugu::RasterizeFaceAttributeToTexture(face_colors, input_mesh->uv(),
                                       input_mesh->uv_indices(), fc_rasterized,
                                       tex_w, tex_h);
  for (auto& m : materials) {
    m.diffuse_tex = fc_rasterized;
  }

  input_mesh->set_material_ids(material_ids);
  input_mesh->set_materials(materials);

  input_mesh->WriteObj(data_dir, "bunny_my_uv_mat");

  {
    std::vector<Eigen::Vector2f> points_2d;
    ugu::OrthoProjectToXY(Eigen::Vector3f(0.f, 0.f, 1.f),
                          input_mesh->vertices(), points_2d, false, false,
                          false);

    std::vector<Eigen::Vector3f> points_3d;
    for (const auto& p2d : points_2d) {
      points_3d.push_back(Eigen::Vector3f(p2d[0], p2d[1], 0.f));
    }
    auto out_mesh = ugu::Mesh(*input_mesh);

    out_mesh.set_vertices(points_3d);
    out_mesh.WriteObj(data_dir, "bunny_projected_xy");
  }

  return 0;
}
