/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <random>

#include "ugu/line.h"
#include "ugu/mesh.h"

namespace {

const uint32_t seed = 0;
static std::default_random_engine engine(seed);

void SavePoints(const std::vector<ugu::Line3d>& lines,
                const std::string& path) {
  ugu::MeshPtr mesh = ugu::Mesh::Create();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3f> colors;

  for (const auto& l : lines) {
    vertices.push_back(l.a.cast<float>());
    normals.push_back(l.d.cast<float>());
    colors.push_back(
        (Eigen::Vector3f(l.d.cast<float>()) + Eigen::Vector3f::Constant(1.f)) *
        0.5f * 255.f);
  }

  mesh->set_vertices(vertices);
  mesh->set_normals(normals);
  mesh->set_vertex_colors(colors);

  mesh->WritePly(path);
}

}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  int num = 100;
  // mm scale
  Eigen::Vector3d d(1.0, 0.0, 0.0);
  Eigen::Vector3d offset(0.1, 0.2, 0.2);
  double step = 0.1;
  double p_mu = 0.0;
  double p_sigma = 0.02;
  double d_mu = 0.0;
  double d_sigma = ugu::pi / 20.0;
  std::normal_distribution<double> p_dist(p_mu, p_sigma);
  std::normal_distribution<double> d_dist(d_mu, d_sigma);

  std::uniform_real_distribution<double> a_dist;

  std::vector<ugu::Line3d> clean;
  for (int i = 0; i < num / 2; i++) {
    ugu::Line3d l;
    l.a = d * step * i;
    l.d = d;
    clean.push_back(l);

    ugu::Line3d l2;
    l2.a = d * step * i + offset;
    l2.d = d;
    clean.push_back(l2);
  }
  SavePoints(clean, "clean.ply");

  std::vector<ugu::Line3d> unclean;
  for (int i = 0; i < num; i++) {
    ugu::Line3d l;
    l.a = clean[i].a + Eigen::Vector3d::Constant(p_dist(engine));
    Eigen::Vector3d axis(a_dist(engine), a_dist(engine), a_dist(engine));
    axis.normalize();
    Eigen::Matrix3d R = Eigen::AngleAxisd(d_dist(engine), axis).matrix();
    l.d = R * clean[i].d;
    unclean.push_back(l);
  }
  SavePoints(unclean, "unclean.ply");

  std::vector<ugu::Line3d> fused;
  ugu::LineClustering(unclean, fused, 0.002, 2.0, 0.1, ugu::pi / 6);
  SavePoints(fused, "fused.ply");

  return 0;
}
