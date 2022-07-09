/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/clustering/clustering.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"

namespace {
std::default_random_engine engine(0);

void SavePoints(const std::string& ply_path,
                const std::vector<Eigen::VectorXf>& points, int num_clusters,
                const std::vector<size_t>& labels,
                const std::vector<Eigen::VectorXf>& centroids) {
  ugu::Mesh mesh;
  std::vector<Eigen::Vector3f> points3d;
  // Points
  std::transform(
      points.begin(), points.end(), std::back_inserter(points3d),
      [](const auto& p) { return Eigen::Vector3f(p[0], p[1], p[2]); });
  // Centroids
  std::transform(
      centroids.begin(), centroids.end(), std::back_inserter(points3d),
      [](const auto& p) { return Eigen::Vector3f(p[0], p[1], p[2]); });
  mesh.set_vertices(points3d);

  std::vector<Eigen::Vector3f> point_colors;
  std::vector<Eigen::Vector3f> label_colors;
  std::uniform_real_distribution<float> color_dstr(0.f, 255.f);
  // Random color for each cluster
  for (int i = 0; i < num_clusters; i++) {
    label_colors.emplace_back(color_dstr(engine), color_dstr(engine),
                              color_dstr(engine));
  }
  for (size_t i = 0; i < points.size(); i++) {
    auto l = labels[i];
    if (l < num_clusters) {
      point_colors.push_back(label_colors[l]);
    } else {
      point_colors.emplace_back(0.f, 0.f, 0.f);
    }
  }
  for (int i = 0; i < num_clusters; i++) {
    // Red for cluster centroids
    point_colors.emplace_back(255.f, 0.f, 0.f);
  }
  mesh.set_vertex_colors(point_colors);
  mesh.WritePly(ply_path);
}

void KMeansTest() {
  std::vector<Eigen::VectorXf> points;
  float r = 1.f;
  std::normal_distribution<float> dstr(0.f, r);

#if 1
  size_t pc = 3000;
  size_t gt_clusters = 10;
  for (size_t i = 0; i < pc; i++) {
    auto gt_cluster = i % gt_clusters;
    float offset = static_cast<float>(gt_cluster * r);
    float x = dstr(engine) + offset * 3.f;
    float y = dstr(engine);
    float z = dstr(engine) + offset * 3.f;
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }
#else
  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";
  // load mesh
  std::shared_ptr<ugu::Mesh> input_mesh = std::make_shared<ugu::Mesh>();
  input_mesh->LoadObj(obj_path, data_dir);
  points.clear();
  std::transform(input_mesh->vertices().begin(), input_mesh->vertices().end(),
                 std::back_inserter(points), [](const auto& p) {
                   Eigen::VectorXf v(3);
                   v[0] = p[0];
                   v[1] = p[1];
                   v[2] = p[2];
                   return v;
                 });
#endif

  ugu::Timer<> timer;

  std::vector<size_t> labels;
  std::vector<Eigen::VectorXf> centroids;
  std::vector<float> dists;
  std::vector<std::vector<Eigen::VectorXf>> clustered_points;
  int num_clusters = 10;
  timer.Start();
  ugu::KMeans(points, num_clusters, labels, centroids, dists, clustered_points,
              100, 1.f, false, 0);
  timer.End();
  ugu::LOGI("KMeans naive %f ms\n", timer.elapsed_msec());
  SavePoints("kmeans_naive.ply", points, num_clusters, labels, centroids);

  timer.Start();
  ugu::KMeans(points, num_clusters, labels, centroids, dists, clustered_points,
              100, 1.f, true, 0);
  ugu::LOGI("KMeans++ %f ms\n", timer.elapsed_msec());
  timer.End();
  SavePoints("kmeans_plusplus.ply", points, num_clusters, labels, centroids);
}

void MeanShiftTest() {
  std::vector<Eigen::VectorXf> points;
  float r = 1.f;
  std::normal_distribution<float> dstr(0.f, r);

#if 0
  size_t pc = 1000;
  for (size_t i = 0; i < pc; i++) {
    float x = dstr(engine);
    float y = dstr(engine);
    float z = dstr(engine);
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }
#endif

  Eigen::VectorXf init(3);
  init[0] = r;
  init[1] = r;
  init[2] = r;

#if 1
  size_t pc = 3000;
  size_t gt_clusters = 10;
  ugu::LOGI("#Clusters GT %d\n", gt_clusters);
  for (size_t i = 0; i < pc; i++) {
    auto gt_cluster = i % gt_clusters;
    float offset = static_cast<float>(gt_cluster * r);
    float x = dstr(engine) + offset * 3.f;
    float y = dstr(engine);
    float z = dstr(engine) + offset * 3.f;
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }
#else

  // For bunny model, MeanShift does not work well. Maybe there is not extreme
  // points on watertight surface.
  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";
  // load mesh
  std::shared_ptr<ugu::Mesh> input_mesh = std::make_shared<ugu::Mesh>();
  input_mesh->LoadObj(obj_path, data_dir);
  points.clear();
  std::transform(input_mesh->vertices().begin(), input_mesh->vertices().end(),
                 std::back_inserter(points), [](const auto& p) {
                   Eigen::VectorXf v(3);
                   v[0] = p[0];
                   v[1] = p[1];
                   v[2] = p[2];
                   return v;
                 });

  points.resize(points.size());
  init = points[0];
  r = 100.f;
#endif

  ugu::Timer<> timer;

  Eigen::VectorXf node(3);
  ugu::LOGI("MeanShift init              (%f %f %f)\n", init[0], init[1],
            init[2]);
  timer.Start();
  ugu::MeanShift(points, init, r, 0.0001f, 100, node);
  timer.End();
  ugu::LOGI("MeanShift %f ms\n", timer.elapsed_msec());
  ugu::LOGI("MeanShift converged extrema (%f %f %f)\n", node[0], node[1],
            node[2]);

  std::vector<size_t> labels;
  std::vector<Eigen::VectorXf> nodes;
  std::vector<float> dists;
  std::vector<std::vector<Eigen::VectorXf>> clustered_points;
  int num_clusters = 0;

  timer.Start();
  ugu::MeanShiftClustering(points, num_clusters, labels, nodes,
                           clustered_points, r, 0.0001f, 100, r / 50);
  timer.End();
  ugu::LOGI("MeanShiftClustering %f ms\n", timer.elapsed_msec());
  ugu::LOGI("#Clusters MeanShiftClustering %d\n", num_clusters);
  std::vector<Eigen::VectorXf> centroids;
  ugu::CalcCentroids(points, labels, centroids, num_clusters);
  SavePoints("mean_shift.ply", points, num_clusters, labels, centroids);
}

void DBSCANTest() {
  std::vector<Eigen::VectorXf> points;
  float r = 1.f;
  std::normal_distribution<float> dstr(0.f, r);

#if 0
  size_t pc = 1000;
  for (size_t i = 0; i < pc; i++) {
    float x = dstr(engine);
    float y = dstr(engine);
    float z = dstr(engine);
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }
#endif

  Eigen::VectorXf init(3);
  init[0] = r;
  init[1] = r;
  init[2] = r;

#if 1
  size_t pc = 3000;
  size_t gt_clusters = 10;
  ugu::LOGI("#Clusters GT %d\n", gt_clusters);
  for (size_t i = 0; i < pc; i++) {
    auto gt_cluster = i % gt_clusters;
    float offset = static_cast<float>(gt_cluster * r);
    float x = dstr(engine) + offset * 3.f;
    float y = dstr(engine);
    float z = dstr(engine) + offset * 3.f;
    Eigen::VectorXf p(3);
    p[0] = x;
    p[1] = y;
    p[2] = z;
    points.emplace_back(p);
  }
#else

  // For bunny model, MeanShift does not work well. Maybe there is not extreme
  // points on watertight surface.
  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";
  // load mesh
  std::shared_ptr<ugu::Mesh> input_mesh = std::make_shared<ugu::Mesh>();
  input_mesh->LoadObj(obj_path, data_dir);
  points.clear();
  std::transform(input_mesh->vertices().begin(), input_mesh->vertices().end(),
                 std::back_inserter(points), [](const auto& p) {
                   Eigen::VectorXf v(3);
                   v[0] = p[0];
                   v[1] = p[1];
                   v[2] = p[2];
                   return v;
                 });

  points.resize(points.size());
  init = points[0];
  r = 100.f;
#endif

  std::vector<int32_t> labels;
  std::vector<std::vector<Eigen::VectorXf>> clustered_points;
  std::vector<Eigen::VectorXf> noise_points;
  int num_clusters = 0;
  ugu::Timer<> timer;
  timer.Start();
  ugu::DBSCAN(points, num_clusters, labels, clustered_points, noise_points, r,
              40);
  timer.End();
  ugu::LOGI("DBSCAN %f ms\n", timer.elapsed_msec());
  ugu::LOGI("#Clusters DBSCAN %d\n", num_clusters);
  std::vector<Eigen::VectorXf> centroids;
  std::vector<size_t> labels_;
  std::transform(labels.begin(), labels.end(), std::back_inserter(labels_),
                 [](const auto& l) { return static_cast<size_t>(l); });
  ugu::CalcCentroids(points, labels_, centroids, num_clusters);
  SavePoints("dbscan.ply", points, num_clusters, labels_, centroids);
}

void SegmentMeshTest() {
  std::vector<std::string> data_dirs{"../data/blendshape/", "../data/sphere/",
                                     "../data/sphere/", "../data/bunny/"};

  std::vector<std::string> names{"cube", "icosphere3_smart_uv",
                                 "icosphere5_smart_uv", "bunny"};

  for (size_t i = 0; i < data_dirs.size(); i++) {
    std::string data_dir = data_dirs[i];
    std::string name = names[i];
    std::string obj_path = data_dir + name + ".obj";

    // load mesh
    ugu::MeshPtr input_mesh = ugu::Mesh::Create();
    input_mesh->LoadObj(obj_path, data_dir);

    ugu::SegmentMeshResult res;
    ugu::Timer<> timer;
    timer.Start();
    ugu::SegmentMesh(input_mesh->vertices(), input_mesh->vertex_indices(),
                     input_mesh->face_normals(), res, 66.4f, 0.f, true, true);
    timer.End();
    ugu::LOGI("SegmentMesh %f ms\n", timer.elapsed_msec());
    ugu::LOGI("#Clusters %d\n", res.clusters.size());

    std::uniform_real_distribution<float> color_dstr(0.f, 255.f);
    std::vector<Eigen::Vector3f> random_colors;

    for (size_t i = 0; i < res.clusters.size(); i++) {
      ugu::MeshPtr output_mesh = ugu::Mesh::Create();

      // TODO: not to decompose
      std::vector<Eigen::Vector3f> vertices;
      std::vector<Eigen::Vector3i> faces;
      for (const auto& fid : res.cluster_fids[i]) {
        auto v0 = input_mesh->vertices()[input_mesh->vertex_indices()[fid][0]];
        auto v1 = input_mesh->vertices()[input_mesh->vertex_indices()[fid][1]];
        auto v2 = input_mesh->vertices()[input_mesh->vertex_indices()[fid][2]];
        vertices.push_back(v0);
        vertices.push_back(v1);
        vertices.push_back(v2);

        faces.push_back({static_cast<int>(vertices.size() - 3),
                         static_cast<int>(vertices.size() - 2),
                         static_cast<int>(vertices.size() - 1)});
      }

      output_mesh->set_vertex_indices(faces);
      output_mesh->set_vertices(vertices);

      Eigen::Vector3f random_color(color_dstr(engine), color_dstr(engine),
                                   color_dstr(engine));
      random_colors.push_back(random_color);
      std::vector<Eigen::Vector3f> vertex_colors(vertices.size(), random_color);
      output_mesh->set_vertex_colors(vertex_colors);
      output_mesh->WritePly(data_dir + name + "_segmented_" +
                            std::to_string(i) + ".ply");
    }

    ugu::MeshPtr output_mesh = ugu::Mesh::Create();
    for (size_t i = 0; i < res.cluster_ids.size(); i++) {
      auto id = res.cluster_ids[i];
      if (id > 1000) {
        ugu::LOGI("%d %d\n", i, id);
      }
    }

    std::vector<int> material_ids;
    std::transform(res.cluster_ids.begin(), res.cluster_ids.end(),
                   std::back_inserter(material_ids),
                   [&](uint32_t cid) { return int(cid); });
    std::vector<ugu::ObjMaterial> materials;
    for (size_t i = 0; i < res.clusters.size(); i++) {
      ugu::ObjMaterial mat;
      mat.name = "mat_" + std::to_string(i);
      mat.diffuse[0] = random_colors[i][0] / 255.f;
      mat.diffuse[1] = random_colors[i][1] / 255.f;
      mat.diffuse[2] = random_colors[i][2] / 255.f;
      materials.push_back(mat);
    };

    output_mesh->set_vertices(input_mesh->vertices());
    output_mesh->set_vertex_indices(input_mesh->vertex_indices());
    output_mesh->set_material_ids(material_ids);
    output_mesh->set_materials(materials);

    output_mesh->WriteObj(data_dir, name + "_segmented");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  SegmentMeshTest();

  KMeansTest();

  MeanShiftTest();

  DBSCANTest();

  return 0;
}
