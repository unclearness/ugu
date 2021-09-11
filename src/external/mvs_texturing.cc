/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/external/external.h"

#ifdef UGU_USE_MVS_TEXTURING

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include <mve/image_io.h>
#include <mve/mesh_io_ply.h>
#include <util/file_system.h>
#include <util/system.h>
#include <util/timer.h>

#include "apps/texrecon/arguments.h"
#include "tex/debug.h"
#include "tex/progress_counter.h"
#include "tex/texturing.h"
#include "tex/timer.h"
#include "tex/util.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

#endif

namespace {

bool Keyframes2TextureViews(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    tex::TextureViews* texture_views) {
  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    std::shared_ptr<ugu::Keyframe> kf = keyframes[i];

    mve::CameraInfo cam_info;
    ugu::PinholeCamera pinhole_camera =
        *dynamic_cast<const ugu::PinholeCamera*>(&(*kf->camera));
    float ave_f =
        (pinhole_camera.focal_length()[0] + pinhole_camera.focal_length()[1]) /
        2;
    float longer_length = static_cast<float>(
        std::max(pinhole_camera.width(), pinhole_camera.height()));
    cam_info.flen = ave_f / longer_length;

    cam_info.ppoint[0] =
        pinhole_camera.principal_point()[0] / (pinhole_camera.width() - 1);
    cam_info.ppoint[1] =
        pinhole_camera.principal_point()[1] / (pinhole_camera.height() - 1);

    cam_info.dist[0] = 0.0f;
    cam_info.dist[1] = 0.0f;

    cam_info.paspect =
        pinhole_camera.focal_length()[0] / pinhole_camera.focal_length()[1];

    Eigen::Matrix3d w2c_R = pinhole_camera.w2c().rotation();
    for (int jj = 0; jj < 3; jj++) {
      for (int ii = 0; ii < 3; ii++) {
        cam_info.rot[jj * 3 + ii] = static_cast<float>(w2c_R(jj, ii));
      }
    }

    Eigen::Vector3d w2c_t = pinhole_camera.w2c().translation();
    cam_info.trans[0] = static_cast<float>(w2c_t[0]);
    cam_info.trans[1] = static_cast<float>(w2c_t[1]);
    cam_info.trans[2] = static_cast<float>(w2c_t[2]);

    mve::ByteImage::Ptr image =
        mve::ByteImage::create(kf->color.cols, kf->color.rows, 3);
    const size_t data_size =
        static_cast<size_t>(kf->color.cols * kf->color.rows * 3);
    std::memcpy(image->get_data_pointer(), kf->color.data, data_size);

    // mve::image::save_png_file(image, "mve_" + std::to_string(i) + ".png");

    texture_views->push_back(tex::TextureView(i, cam_info, image));
  }

  return true;
}

bool ConvertMesh(const ugu::Mesh& ugu_mesh, mve::TriangleMesh::Ptr mesh) {
  // covnert vertices
  std::vector<math::Vec3f>& vertices = mesh->get_vertices();
  vertices.clear();
  for (const auto& v : ugu_mesh.vertices()) {
    vertices.push_back(math::Vec3f(v.x(), v.y(), v.z()));
  }

  // convert faces
  std::vector<unsigned int>& faces = mesh->get_faces();
  faces.clear();
  for (const auto& f : ugu_mesh.vertex_indices()) {
    faces.push_back(f.x());
    faces.push_back(f.y());
    faces.push_back(f.z());
  }

  return true;
}

bool ConvertObjModel(tex::Model& objmodel, ugu::Mesh& ugu_mesh) {
  // ugu_mesh.Clear();

  // TODO: Vertex id and face id may change...

  auto vertices = objmodel.get_vertices();
  auto texcoords = objmodel.get_texcoords();
  auto normals = objmodel.get_normals();
  auto material_lib = objmodel.get_material_lib();
  auto groups = objmodel.get_groups();

  std::vector<Eigen::Vector2f> ugu_uv;
  std::vector<Eigen::Vector3f> ugu_vertices;
  std::vector<Eigen::Vector3f> ugu_normals;
  std::vector<ugu::ObjMaterial> ugu_materials;
  std::vector<Eigen::Vector3i> ugu_uv_indices, ugu_indices, ugu_normal_indices;
  //  std::vector<std::vector<int>> ugu_face_indices_per_material;
  std::vector<int> ugu_material_ids;

  std::transform(texcoords.begin(), texcoords.end(), std::back_inserter(ugu_uv),
                 [](auto& texcoord) {
                   return Eigen::Vector2f(texcoord[0], 1.f - texcoord[1]);
                 });

  std::transform(vertices.begin(), vertices.end(),
                 std::back_inserter(ugu_vertices), [](auto& vertex) {
                   return Eigen::Vector3f(vertex[0], vertex[1], vertex[2]);
                 });

  std::transform(normals.begin(), normals.end(),
                 std::back_inserter(ugu_normals), [](auto& normal) {
                   return Eigen::Vector3f(normal[0], normal[1], normal[2]);
                 });

  std::transform(material_lib.begin(), material_lib.end(),
                 std::back_inserter(ugu_materials), [](auto& material) {
                   ugu::ObjMaterial ugu_material;
                   ugu_material.name = material.name;
                   mve::ByteImage::ConstPtr diffuse_map = material.diffuse_map;
                   ugu_material.diffuse_tex = ugu::Image3b::zeros(
                       diffuse_map->height(), diffuse_map->width());
                   const size_t data_size = static_cast<size_t>(
                       diffuse_map->height() * diffuse_map->width() * 3);
                   std::memcpy(ugu_material.diffuse_tex.data,
                               diffuse_map->get_data_pointer(), data_size);
                   return ugu_material;
                 });

  // ugu_face_indices_per_material.resize(groups.size());
  for (const auto& group : groups) {
    int index = -1;
    for (size_t i = 0; i < ugu_materials.size(); i++) {
      if (group.material_name == ugu_materials[i].name) {
        index = static_cast<int>(i);
        break;
      }
    }
    if (index < 0) {
      ugu::LOGE("something wrong...\n");
      return false;
    }

    // auto& ugu_group = ugu_face_indices_per_material[index];

    for (size_t j = 0; j < group.faces.size(); j++) {
      const auto& face = group.faces[j];
      ugu_indices.emplace_back(static_cast<int>(face.vertex_ids[0]),
                               static_cast<int>(face.vertex_ids[1]),
                               static_cast<int>(face.vertex_ids[2]));

      ugu_normal_indices.emplace_back(static_cast<int>(face.normal_ids[0]),
                                      static_cast<int>(face.normal_ids[1]),
                                      static_cast<int>(face.normal_ids[2]));
      ugu_uv_indices.emplace_back(static_cast<int>(face.texcoord_ids[0]),
                                  static_cast<int>(face.texcoord_ids[1]),
                                  static_cast<int>(face.texcoord_ids[2]));

      // ugu_group.push_back(j + offset);
      ugu_material_ids.push_back(index);
    }
  }

  ugu_mesh.set_vertices(ugu_vertices);
  ugu_mesh.set_normals(ugu_normals);
  ugu_mesh.set_uv(ugu_uv);

  ugu_mesh.set_vertex_indices(ugu_indices);
  ugu_mesh.set_normal_indices(ugu_normal_indices);
  ugu_mesh.set_uv_indices(ugu_uv_indices);

  ugu_mesh.set_materials(ugu_materials);
  ugu_mesh.set_material_ids(ugu_material_ids);

  return true;
}

}  // namespace

namespace ugu {

bool MvsTexturing(const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
                  ugu::Mesh* ugu_mesh, ugu::Mesh* debug_mesh) {
#ifdef UGU_USE_MVS_TEXTURING

  Arguments conf;

  conf.num_threads = -1;

  conf.out_prefix = "hoge";

  conf.write_view_selection_model = debug_mesh != nullptr;
  conf.settings.keep_unseen_faces = true;

  conf.settings.data_term = tex::DataTerm::DATA_TERM_AREA;
  conf.settings.smoothness_term = tex::SmoothnessTerm::SMOOTHNESS_TERM_POTTS;
  conf.settings.outlier_removal = tex::OutlierRemoval::
      OUTLIER_REMOVAL_GAUSS_CLAMPING;  // OUTLIER_REMOVAL_NONE;

  if (conf.num_threads > 0) {
#ifdef UGU_USE_OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(conf.num_threads);
#endif
  }

  mve::TriangleMesh::Ptr mesh = mve::TriangleMesh::create();
  ConvertMesh(*ugu_mesh, mesh);

  mve::MeshInfo mesh_info(mesh);
  tex::prepare_mesh(&mesh_info, mesh);

  tex::TextureViews texture_views;
  Keyframes2TextureViews(keyframes, &texture_views);

  const uint32_t num_faces =
      static_cast<uint32_t>(mesh->get_faces().size() / 3);

  tex::Graph graph(num_faces);
  tex::build_adjacency_graph(mesh, mesh_info, &graph);

  if (conf.labeling_file.empty()) {
    tex::DataCosts data_costs(num_faces,
                              static_cast<uint16_t>(texture_views.size()));
    if (conf.data_cost_file.empty()) {
      tex::calculate_data_costs(mesh, &texture_views, conf.settings,
                                &data_costs);
#if 0
    
      if (conf.write_intermediate_results) {
        std::cout << "\tWriting data cost file... " << std::flush;
        tex::DataCosts::save_to_file(data_costs,
                                     conf.out_prefix + "_data_costs.spt");
        std::cout << "done." << std::endl;
      }
#endif  // 0

    } else {
      try {
        tex::DataCosts::load_from_file(conf.data_cost_file, &data_costs);
      } catch (util::FileException e) {
        std::cout << "failed!" << std::endl;
        std::cerr << e.what() << std::endl;
        return false;
      }
    }

    try {
      tex::view_selection(data_costs, &graph, conf.settings);
    } catch (std::runtime_error& e) {
      std::cerr << "\tOptimization failed: " << e.what() << std::endl;
      return false;
    }

#if 0
       /* Write labeling to file. */
    if (conf.write_intermediate_results) {
      std::vector<std::size_t> labeling(graph.num_nodes());
      for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        labeling[i] = graph.get_label(i);
      }
      vector_to_file(conf.out_prefix + "_labeling.vec", labeling);
    }
#endif  // 0

  } else {
    /* Load labeling from file. */
    std::vector<std::size_t> labeling =
        vector_from_file<std::size_t>(conf.labeling_file);
    if (labeling.size() != graph.num_nodes()) {
      std::cerr << "Wrong labeling file for this mesh/scene combination... "
                   "aborting!"
                << std::endl;
      return false;
    }

    /* Transfer labeling to graph. */
    for (std::size_t i = 0; i < labeling.size(); ++i) {
      const std::size_t label = labeling[i];
      if (label > texture_views.size()) {
        std::cerr << "Wrong labeling file for this mesh/scene combination... "
                     "aborting!"
                  << std::endl;
        return false;
      }
      graph.set_label(i, label);
    }
  }

  tex::TextureAtlases texture_atlases;
  {
    /* Create texture patches and adjust them. */
    tex::TexturePatches texture_patches;
    tex::VertexProjectionInfos vertex_projection_infos;
    // std::cout << "Generating texture patches:" << std::endl;
    tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                                  conf.settings, &vertex_projection_infos,
                                  &texture_patches);

    if (conf.settings.global_seam_leveling) {
      tex::global_seam_leveling(graph, mesh, mesh_info, vertex_projection_infos,
                                &texture_patches);
      // timer.measure("Running global seam leveling");
    } else {
      ProgressCounter texture_patch_counter(
          "Calculating validity masks for texture patches",
          texture_patches.size());
#pragma omp parallel for schedule(dynamic)
      for (std::int64_t i = 0;
           i < static_cast<std::int64_t>(texture_patches.size()); ++i) {
        texture_patch_counter.progress<SIMPLE>();
        TexturePatch::Ptr texture_patch = texture_patches[i];
        std::vector<math::Vec3f> patch_adjust_values(
            texture_patch->get_faces().size() * 3, math::Vec3f(0.0f));
        texture_patch->adjust_colors(patch_adjust_values);
        texture_patch_counter.inc();
      }
    }

    if (conf.settings.local_seam_leveling) {
      tex::local_seam_leveling(graph, mesh, vertex_projection_infos,
                               &texture_patches);
    }

    /* Generate texture atlases. */
    tex::generate_texture_atlases(&texture_patches, conf.settings,
                                  &texture_atlases);
  }

  /* Create and write out obj model. */

  tex::Model objmodel;
  tex::build_model(mesh, texture_atlases, &objmodel);

  ConvertObjModel(objmodel, *ugu_mesh);

  // tex::Model::save(model, conf.out_prefix);

#if 0
      if (conf.write_timings) {
    timer.write_to_file(conf.out_prefix + "_timings.csv");
  }
#endif  // 0

  if (conf.write_view_selection_model) {
    texture_atlases.clear();
    {
      tex::TexturePatches texture_patches;
      generate_debug_embeddings(&texture_views);
      tex::VertexProjectionInfos
          vertex_projection_infos;  // Will only be written
      tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                                    conf.settings, &vertex_projection_infos,
                                    &texture_patches);
      tex::generate_texture_atlases(&texture_patches, conf.settings,
                                    &texture_atlases);
    }

    tex::Model debugobjmodel;
    tex::build_model(mesh, texture_atlases, &debugobjmodel);
    ConvertObjModel(debugobjmodel, *debug_mesh);
    // tex::Model::save(debugobjmodel, conf.out_prefix + "_view_selection");
  }

  // convert to ugu_mesh

  return true;
#else
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

}  // namespace ugu
