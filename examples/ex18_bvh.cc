/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include <iostream>
#include <random>
#include <unordered_set>

#include "ugu/accel/bvh.h"
#include "ugu/accel/bvh_nanort.h"
#include "ugu/image.h"
#include "ugu/mesh.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"

namespace {}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  ugu::MeshPtr mesh = ugu::Mesh::Create();
  mesh->LoadObj("../data/bunny/bunny.obj", "../data/bunny/");

  ugu::Timer timer;
  Eigen::Vector3f origin(0.f, 0.f, 700.f);
  Eigen::Vector3f dir(0.f, 0.f, -1.f);

  {
    timer.Start();
    auto results = ugu::Intersect(origin, dir, mesh->vertices(),
                                  mesh->vertex_indices(), 1);
    timer.End();
    ugu::LOGI("ugu::Intersect took %f ms\n", timer.elapsed_msec());
    std::vector<Eigen::Vector3f> intersected_points;
    for (auto result : results) {
      auto pos_t = origin + dir * result.t;
      const auto& face = mesh->vertex_indices()[result.fid];
      auto pos_uv = (1.f - (result.u + result.v)) * mesh->vertices()[face[0]] +
                    result.u * mesh->vertices()[face[1]] +
                    result.v * mesh->vertices()[face[2]];
      ugu::LOGI("%d (%f, %f), %f, (%f, %f, %f) (%f, %f, %f)\n", result.fid,
                result.u, result.v, result.t, pos_t.x(), pos_t.y(), pos_t.z(),
                pos_uv.x(), pos_uv.y(), pos_uv.z());
      intersected_points.push_back(pos_t);
    }

    auto tmp = ugu::Mesh::Create();
    tmp->set_vertices(intersected_points);
    tmp->WritePly("intersected_raw.ply");
  }

  {
    ugu::BvhNaive<Eigen::Vector3f, Eigen::Vector3i> bvh;
    bvh.SetAxisNum(3);
    bvh.SetMinLeafPrimitives(5);
    bvh.SetData(mesh->vertices(), mesh->vertex_indices());

    timer.Start();
    bvh.Build();
    timer.End();
    ugu::LOGI("ugu::Bvh::Build() took %f ms\n", timer.elapsed_msec());

    auto meshes = bvh.Visualize(3);
    ugu::Mesh merged;
    ugu::MergeMeshes(meshes, &merged);
    merged.WriteObj("../data/bunny/", "bunny_bvh");
    timer.Start();
    ugu::Ray ray;
    ray.org = origin;
    ray.dir = dir;
    std::vector<ugu::IntersectResult> results = bvh.Intersect(ray);
    timer.End();
    ugu::LOGI("bvhNaive.Intersect took %f ms\n", timer.elapsed_msec());
    std::vector<Eigen::Vector3f> intersected_points;
    for (auto result : results) {
      auto pos_t = origin + dir * result.t;
      const auto& face = mesh->vertex_indices()[result.fid];
      auto pos_uv = (1.f - (result.u + result.v)) * mesh->vertices()[face[0]] +
                    result.u * mesh->vertices()[face[1]] +
                    result.v * mesh->vertices()[face[2]];
      ugu::LOGI("%d (%f, %f), %f, (%f, %f, %f) (%f, %f, %f)\n", result.fid,
                result.u, result.v, result.t, pos_t.x(), pos_t.y(), pos_t.z(),
                pos_uv.x(), pos_uv.y(), pos_uv.z());
      intersected_points.push_back(pos_t);
    }

    auto tmp = ugu::Mesh::Create();
    tmp->set_vertices(intersected_points);
    tmp->WritePly("intersected_bvh.ply");
  }

  {
    ugu::BvhNanort<Eigen::Vector3f, Eigen::Vector3i> bvh;
    bvh.SetData(mesh->vertices(), mesh->vertex_indices());

    timer.Start();
    bvh.Build();
    timer.End();
    ugu::LOGI("ugu::Bvh::Build() took %f ms\n", timer.elapsed_msec());

#if 0
    auto meshes = bvh.Visualize(3);
    ugu::Mesh merged;
    ugu::MergeMeshes(meshes, &merged);
    merged.WriteObj("../data/bunny/", "bunny_bvh");
#endif

    timer.Start();
    ugu::Ray ray;
    ray.org = origin;
    ray.dir = dir;
    std::vector<ugu::IntersectResult> results = bvh.Intersect(ray);
    timer.End();
    ugu::LOGI("BvhNanort.Intersect took %f ms\n", timer.elapsed_msec());
    std::vector<Eigen::Vector3f> intersected_points;
    for (auto result : results) {
      auto pos_t = origin + dir * result.t;
      const auto& face = mesh->vertex_indices()[result.fid];
      auto pos_uv = (1.f - (result.u + result.v)) * mesh->vertices()[face[0]] +
                    result.u * mesh->vertices()[face[1]] +
                    result.v * mesh->vertices()[face[2]];
      ugu::LOGI("%d (%f, %f), %f, (%f, %f, %f) (%f, %f, %f)\n", result.fid,
                result.u, result.v, result.t, pos_t.x(), pos_t.y(), pos_t.z(),
                pos_uv.x(), pos_uv.y(), pos_uv.z());
      intersected_points.push_back(pos_t);
    }

    auto tmp = ugu::Mesh::Create();
    tmp->set_vertices(intersected_points);
    tmp->WritePly("intersected_bvh.ply");
  }

  return 0;
}
