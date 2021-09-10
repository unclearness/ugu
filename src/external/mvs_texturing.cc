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

    mve::image::save_png_file(image, "mve_" + std::to_string(i) + ".png");

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

}  // namespace

namespace ugu {

bool MvsTexturing(const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
                  ugu::Mesh* ugu_mesh, ugu::Mesh* debug_mesh) {
#ifdef UGU_USE_MVS_TEXTURING
  // util::system::print_build_timestamp(aergv[0]);
  // util::system::register_segfault_handlr();

  // Timer timer;
  // util::WallTimer wtimer;

  // Arguments conf;
  // try {
  //  conf = parse_args(argc, argv);
  //} catch (std::invalid_argument& ia) {
  //  std::cerr << ia.what() << std::endl;
  //  std::exit(EXIT_FAILURE);
  //}

  // std::string const out_dir = util::fs::dirname(conf.out_prefix);

#if 0
      if (!util::fs::dir_exists(out_dir.c_str())) {
    std::cerr << "Destination directory does not exist!" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string const tmp_dir = util::fs::join_path(out_dir, "tmp");
  if (!util::fs::dir_exists(tmp_dir.c_str())) {
    util::fs::mkdir(tmp_dir.c_str());
  } else {
    std::cerr
        << "Temporary directory \"tmp\" exists within the destination "
           "directory.\n"
        << "Cannot continue since this directory would be delete in the end.\n"
        << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif  // 0

  Arguments conf;

  conf.num_threads = -1;

  conf.out_prefix = "hoge";

  conf.write_view_selection_model = debug_mesh != nullptr;
  conf.settings.keep_unseen_faces = true;

  conf.settings.data_term = tex::DataTerm::DATA_TERM_AREA;
  conf.settings.smoothness_term = tex::SmoothnessTerm::SMOOTHNESS_TERM_POTTS;
  conf.settings.outlier_removal = tex::OutlierRemoval::
      OUTLIER_REMOVAL_GAUSS_CLAMPING;  // OUTLIER_REMOVAL_NONE;
  // ToneMapping tone_mapping = TONE_MAPPING_NONE;

  // Set the number of threads to use.
  // tbb::task_scheduler_init schedule(conf.num_threads > 0 ? conf.num_threads
  // : tbb::task_scheduler_init::automatic);
  if (conf.num_threads > 0) {
    omp_set_dynamic(0);
    omp_set_num_threads(conf.num_threads);
  }

  // std::cout << "Load and prepare mesh: " << std::endl;
  mve::TriangleMesh::Ptr mesh = mve::TriangleMesh::create();
  ConvertMesh(*ugu_mesh, mesh);

#if 0
      try {
    mesh = mve::geom::load_ply_mesh(conf.in_mesh);
  } catch (std::exception& e) {
    std::cerr << "\tCould not load mesh: " << e.what() << std::endl;
    // std::exit(EXIT_FAILURE);
    return false;
  }
#endif  // 0

  mve::MeshInfo mesh_info(mesh);
  tex::prepare_mesh(&mesh_info, mesh);

  // std::cout << "Generating texture views: " << std::endl;
  tex::TextureViews texture_views;
  Keyframes2TextureViews(keyframes, &texture_views);

  // timer.measure("Loading");

  const uint32_t num_faces =
      static_cast<uint32_t>(mesh->get_faces().size() / 3);

  // std::cout << "Building adjacency graph: " << std::endl;
  tex::Graph graph(num_faces);
  tex::build_adjacency_graph(mesh, mesh_info, &graph);

  if (conf.labeling_file.empty()) {
    // std::cout << "View selection:" << std::endl;
    // util::WallTimer rwtimer;

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
      // std::cout << "\tLoading data cost file... " << std::flush;
      try {
        tex::DataCosts::load_from_file(conf.data_cost_file, &data_costs);
      } catch (util::FileException e) {
        std::cout << "failed!" << std::endl;
        std::cerr << e.what() << std::endl;
        // std::exit(EXIT_FAILURE);
        return false;
      }
      // std::cout << "done." << std::endl;
    }
    // timer.measure("Calculating data costs");

    try {
      tex::view_selection(data_costs, &graph, conf.settings);
    } catch (std::runtime_error& e) {
      std::cerr << "\tOptimization failed: " << e.what() << std::endl;
      // std::exit(EXIT_FAILURE);
      return false;
    }
    // timer.measure("Running MRF optimization");
    // std::cout << "\tTook: " << rwtimer.get_elapsed_sec() << "s" << std::endl;

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
    // std::cout << "Loading labeling from file... " << std::flush;

    /* Load labeling from file. */
    std::vector<std::size_t> labeling =
        vector_from_file<std::size_t>(conf.labeling_file);
    if (labeling.size() != graph.num_nodes()) {
      std::cerr << "Wrong labeling file for this mesh/scene combination... "
                   "aborting!"
                << std::endl;
      // std::exit(EXIT_FAILURE);
      return false;
    }

    /* Transfer labeling to graph. */
    for (std::size_t i = 0; i < labeling.size(); ++i) {
      const std::size_t label = labeling[i];
      if (label > texture_views.size()) {
        std::cerr << "Wrong labeling file for this mesh/scene combination... "
                     "aborting!"
                  << std::endl;
        // std::exit(EXIT_FAILURE);
        return false;
      }
      graph.set_label(i, label);
    }

    // std::cout << "done." << std::endl;
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
      // std::cout << "Running global seam leveling:" << std::endl;
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
      // timer.measure("Calculating texture patch validity masks");
    }

    if (conf.settings.local_seam_leveling) {
      // std::cout << "Running local seam leveling:" << std::endl;
      tex::local_seam_leveling(graph, mesh, vertex_projection_infos,
                               &texture_patches);
    }
    // timer.measure("Running local seam leveling");

    /* Generate texture atlases. */
    // std::cout << "Generating texture atlases:" << std::endl;
    tex::generate_texture_atlases(&texture_patches, conf.settings,
                                  &texture_atlases);
  }

  /* Create and write out obj model. */
  {
    // std::cout << "Building objmodel:" << std::endl;
    tex::Model model;
    tex::build_model(mesh, texture_atlases, &model);
    // timer.measure("Building OBJ model");

    // std::cout << "\tSaving model... " << std::flush;
    tex::Model::save(model, conf.out_prefix);
    // std::cout << "done." << std::endl;
    // timer.measure("Saving");
  }

  // std::cout << "Whole texturing procedure took: " << wtimer.get_elapsed_sec()
  //         << "s" << std::endl;
  // timer.measure("Total");

#if 0
      if (conf.write_timings) {
    timer.write_to_file(conf.out_prefix + "_timings.csv");
  }
#endif  // 0

  if (conf.write_view_selection_model) {
    texture_atlases.clear();
    // std::cout << "Generating debug texture patches:" << std::endl;
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

    // std::cout << "Building debug objmodel:" << std::endl;
    {
      tex::Model model;
      tex::build_model(mesh, texture_atlases, &model);
      // std::cout << "\tSaving model... " << std::flush;
      tex::Model::save(model, conf.out_prefix + "_view_selection");
      // std::cout << "done." << std::endl;
    }
  }

  // convert to ugu_mesh

#if 0
   /* Remove temporary files. */
  for (util::fs::File const& file : util::fs::Directory(tmp_dir)) {
    util::fs::unlink(util::fs::join_path(file.path, file.name).c_str());
  }
  util::fs::rmdir(tmp_dir.c_str());
#endif  // 0

  return true;
#else
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

}  // namespace ugu
