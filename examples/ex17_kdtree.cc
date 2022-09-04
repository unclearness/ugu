/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <iostream>
#include <random>
#include <unordered_set>

#include "ugu/accel/kdtree.h"
#include "ugu/accel/kdtree_nanoflann.h"
#include "ugu/image.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/io_util.h"

namespace {

template <typename T>
ugu::Image3b DrawResult2d(const T& query2d, const std::vector<T>& points2d,
                          const ugu::KdTreeSearchResults& res, int r = -1) {
  ugu::Image3b out = ugu::Image3b::zeros(480, 480);
  for (const auto& p : points2d) {
    ugu::Point center{static_cast<int>(p.x() * out.cols),
                      static_cast<int>(p.y() * out.rows)};
    ugu::circle(out, center, 2, {255, 255, 255}, -1);
  }

  {
    ugu::Point center{static_cast<int>(query2d.x() * out.cols),
                      static_cast<int>(query2d.y() * out.rows)};
    ugu::circle(out, center, 3, {0, 255, 0}, -1);
    if (0 < r) {
      ugu::circle(out, center, r, {255, 0, 0}, 2);
    }
  }
  for (const auto& r_ : res) {
    ugu::Point center{static_cast<int>(points2d[r_.index].x() * out.cols),
                      static_cast<int>(points2d[r_.index].y() * out.rows)};
    ugu::circle(out, center, 3, {255, 0, 0}, -1);
  }

  return out;
}

template <typename T>
ugu::MeshPtr VisualizeResult3d(const T& query3d, const std::vector<T>& points3d,
                               const ugu::KdTreeSearchResults& res,
                               double r = -1.0) {
  ugu::MeshPtr mesh = ugu::Mesh::Create();
  std::vector<Eigen::Vector3f> vertices, vertex_colors;

  std::unordered_set<size_t> to_ignore;
  for (const auto& r_ : res) {
    to_ignore.insert(r_.index);
  }
  for (size_t i = 0; i < points3d.size(); i++) {
    if (to_ignore.count(i) != 0) {
      continue;
    }
    vertices.push_back(points3d[i].template cast<float>());
  }
  vertex_colors.resize(vertices.size(), {0.f, 0.f, 0.f});

  vertices.push_back(query3d.template cast<float>());
  vertex_colors.push_back({0.f, 255.f, 0.f});

  for (const auto& rr : res) {
    vertices.push_back(points3d[rr.index].template cast<float>());
    vertex_colors.push_back({255.f, 0.f, 0.f});
  }

  mesh->set_vertices(vertices);
  mesh->set_vertex_colors(vertex_colors);

  if (0 < r) {
    auto sphere = ugu::MakeUvSphere(query3d.template cast<float>(),
                                    static_cast<float>(r));
    sphere->set_uv({});
    sphere->set_uv_indices({});
    std::vector<Eigen::Vector3f> vc;
    vc.resize(sphere->vertices().size(), {255.f, 255.f, 255.f});
    sphere->set_vertex_colors(vc);
    ugu::MeshPtr merged = ugu::Mesh::Create();
    ugu::MergeMeshes(*mesh, *sphere, merged.get());
    mesh = merged;
  }

  return mesh;
}

void Test2D() {
  std::default_random_engine engine;
  // 2D case
  std::uniform_real_distribution<double> dist2d(0.0, 1.0);
  std::vector<Eigen::Vector2d> points2d;
  for (int i = 0; i < 500; i++) {
    points2d.push_back({dist2d(engine), dist2d(engine)});
  }
  ugu::KdTreeNaive<double, 2> kdtree;
  kdtree.SetData(points2d);
  kdtree.SetMaxLeafDataNum(10);

  ugu::Timer timer;

  timer.Start();
  kdtree.Build();
  timer.End();
  ugu::LOGI("KdTree.Build(): %f msec\n", timer.elapsed_msec());

  ugu::KdTreeSearchResults res;

  Eigen::Vector2d query2d{0.4f, 0.6f};
  int k = 20;
  timer.Start();
  res = kdtree.SearchKnn(query2d, k);
  timer.End();
  ugu::LOGI("KdTree.SearchKnn(): %f msec\n", timer.elapsed_msec());
  ugu::imwrite("kdtree2d_knn.png", DrawResult2d(query2d, points2d, res));

  timer.Start();
  {
    res.clear();
    for (size_t i = 0; i < points2d.size(); i++) {
      const auto& p = points2d[i];
      const double dist = (p - query2d).norm();
      res.push_back({i, dist});
      std::sort(res.begin(), res.end(),
                [](const ugu::KdTreeSearchResult& lfs,
                   const ugu::KdTreeSearchResult& rfs) {
                  return lfs.dist < rfs.dist;
                });
      if (res.size() <= k) {
        continue;
      }
      res.resize(k);
    }
  }
  timer.End();
  ugu::LOGI("BruteForce: %f msec\n", timer.elapsed_msec());

  timer.Start();
  double r = 0.1;
  res = kdtree.SearchRadius(query2d, r);
  timer.End();
  ugu::LOGI("KdTree.SearchRadius(): %f msec\n", timer.elapsed_msec());
  ugu::imwrite("kdtree2d_radius.png",
               DrawResult2d(query2d, points2d, res, static_cast<int>(r * 480)));
}

void Test3D() {
#if 1
  std::default_random_engine engine;
  // 3D case
  std::uniform_real_distribution<double> dist3d(0.0, 1.0);
  std::vector<Eigen::Vector3d> points3d;
  for (int i = 0; i < 5000; i++) {
    points3d.push_back({dist3d(engine), dist3d(engine), dist3d(engine)});
  }

  Eigen::IOFormat my_format(Eigen::StreamPrecision, Eigen::DontAlignCols, " ",
                            " ", "", "", "", "");
  std::ofstream ofs("tmp.txt");
  for (const auto& p : points3d) {
    ofs << p.format(my_format) << std::endl;
  }
#else
  std::string path = "tmp.txt";
  std::vector<Eigen::Vector3f> points3d =
      ugu::LoadTxtAsEigenVec<Eigen::Vector3f>(path);
#endif

  ugu::KdTreeNaive<double, 3> kdtree;
  kdtree.SetData(points3d);
  kdtree.SetMaxLeafDataNum(10);

  ugu::Timer timer;

  timer.Start();
  kdtree.Build();
  timer.End();
  ugu::LOGI("KdTree.Build(): %f msec\n", timer.elapsed_msec());

  ugu::KdTreeSearchResults res;

  Eigen::Vector3d query3d{0.4f, 0.6f, 0.5f};
  int k = 20;
  timer.Start();
  res = kdtree.SearchKnn(query3d, k);
  timer.End();
  ugu::LOGI("KdTree.SearchKnn(): %f msec\n", timer.elapsed_msec());

  timer.Start();
  {
    ugu::KdTreeSearchResults res;
    for (size_t i = 0; i < points3d.size(); i++) {
      const auto& p = points3d[i];
      const double dist = (p - query3d).norm();
      res.push_back({i, dist});
      std::sort(res.begin(), res.end(),
                [](const ugu::KdTreeSearchResult& lfs,
                   const ugu::KdTreeSearchResult& rfs) {
                  return lfs.dist < rfs.dist;
                });
      if (res.size() <= k) {
        continue;
      }
      res.resize(k);
    }
  }
  timer.End();
  ugu::LOGI("BruteForce: %f msec\n", timer.elapsed_msec());

  {
    auto pc = VisualizeResult3d(query3d, points3d, res);
    pc->WritePly("kdtree3d_knn.ply");
  }

  timer.Start();
  double r = 0.1;
  res = kdtree.SearchRadius(query3d, r);
  timer.End();
  ugu::LOGI("KdTree.SearchRadius(): %f msec\n", timer.elapsed_msec());
  {
    auto pc = VisualizeResult3d(query3d, points3d, res, r);
    pc->WritePly("kdtree3d_radius.ply");
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  Test2D();

  Test3D();

  return 0;
}
