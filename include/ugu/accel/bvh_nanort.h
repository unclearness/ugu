/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once
#ifdef UGU_USE_NANORT

#include <memory>

#include "nanort.h"
#include "ugu/accel/bvh.h"

namespace ugu {

template <typename T, typename TT>
class BvhNanort : public Bvh<T, TT> {
 public:
  BvhNanort() {}
  ~BvhNanort() {}

  void SetData(const std::vector<T>& vertices,
               const std::vector<TT>& indices) override {
    m_initialized = false;
    m_vertices.clear();
    m_indices.clear();

    m_vertices = vertices;
    std::transform(
        indices.begin(), indices.end(), std::back_inserter(m_indices),
        [&](const TT& i) { return i.template cast<unsigned int>(); });
  }

  bool Build() override {
    m_initialized = false;

    bool ret = false;
    m_build_options.cache_bbox = false;

    // LOGI("  BVH build option:\n");
    // LOGI("    # of leaf primitives: %d\n",
    // m_build_options.min_leaf_primitives); LOGI("    SAH binsize         :
    // %d\n", m_build_options.bin_size);

    // Timer<> timer;
    // timer.Start();

    m_triangle_mesh =
        std::make_unique<nanort::TriangleMesh<typename T::Scalar>>(
            m_vertices[0].data(), m_indices[0].data(),
            sizeof(typename T::Scalar) * 3);

    m_triangle_pred =
        std::make_unique<nanort::TriangleSAHPred<typename T::Scalar>>(
            m_vertices[0].data(), m_indices[0].data(),
            sizeof(typename T::Scalar) * 3);

    // LOGI("num_triangles = %llu\n",
    //     static_cast<uint64_t>(mesh_->vertex_indices().size()));
    // LOGI("faces = %p\n", mesh_->vertex_indices().size());

    ret = m_accel.Build(static_cast<unsigned int>(m_indices.size()),
                        *m_triangle_mesh, *m_triangle_pred, m_build_options);

    if (!ret) {
      LOGE("BVH building failed\n");
      return false;
    }

    // timer.End();
    // LOGI("  BVH build time: %.1f msecs\n", timer.elapsed_msec());

    m_stats = m_accel.GetStatistics();
    m_accel.BoundingBox(m_bmin.data(), m_bmax.data());

    // m_triangle_intersector = nanort::TriangleIntersector<typename T::Scalar>(
    //    m_vertices[0].data(), m_indices[0].data(), sizeof(T::Scalar) * 3);

    m_initialized = true;

    return true;
  }

  const BvhBuildStatistics& GetBuildStatistics() const override {
    BvhBuildStatistics stats;
    stats.build_secs = m_stats.build_secs;
    stats.max_tree_depth = m_stats.max_tree_depth;
    stats.num_branch_nodes = m_stats.num_branch_nodes;
    stats.num_leaf_nodes = m_stats.num_leaf_nodes;
    Eigen::Vector3d bmax, bmin;
    bmax[0] = static_cast<double>(m_bmax[0]);
    bmax[1] = static_cast<double>(m_bmax[1]);
    bmax[2] = static_cast<double>(m_bmax[2]);
    bmin[0] = static_cast<double>(m_bmin[0]);
    bmin[1] = static_cast<double>(m_bmin[1]);
    bmin[2] = static_cast<double>(m_bmin[2]);
    stats.bb = Aabb<Eigen::Vector3d>(bmax, bmin);

    return stats;
  }

  std::vector<IntersectResult> Intersect(const Ray& ray,
                                         bool test_all = true) const override {
    if (!m_initialized) {
      LOGE("Not initialized yet");
      return {};
    }

    std::vector<IntersectResult> results;
    nanort::Ray<typename T::Scalar> ray_nanort;

    const float kFar = 1.0e+30f;
    ray_nanort.min_t = 0.0001f;
    ray_nanort.max_t = kFar;
    for (int i = 0; i < 3; i++) {
      ray_nanort.org[i] = ray.org[i];
      ray_nanort.dir[i] = ray.dir[i];
    }

    // shoot ray
    float t_offset = 0.f;
    constexpr float epsilon = 1e-7f;
    nanort::TriangleIntersection<typename T::Scalar> isect;
    nanort::TriangleIntersector<typename T::Scalar> m_triangle_intersector(
        m_vertices[0].data(), m_indices[0].data(),
        sizeof(typename T::Scalar) * 3);
    bool hit = m_accel.Traverse(ray_nanort, m_triangle_intersector, &isect);
    while (hit) {
      IntersectResult res;
      res.fid = isect.prim_id;
      res.t = isect.t + t_offset;
      res.u = isect.u;
      res.v = isect.v;
      results.emplace_back(res);

      t_offset += res.t;

      if (!test_all) {
        break;
      }

      for (int i = 0; i < 3; i++) {
        typename T::Scalar hit_pos_i =
            ray_nanort.org[i] + ray_nanort.dir[i] * isect.t;
        ray_nanort.org[i] = hit_pos_i + epsilon * ray_nanort.dir[i];
      }
      hit = m_accel.Traverse(ray_nanort, m_triangle_intersector, &isect);
    }

    return results;
  }

 private:
  bool m_initialized{false};
  std::vector<T> m_vertices;
  std::vector<Eigen::Vector<unsigned int, 3>>
      m_indices;  // Nanort expects unsinged int
  // std::vector<T::Scalar> m_flatten_vertices;
  // std::vector<TT::Scalar> m_flatten_faces;
  // nanort::TriangleIntersector<typename T::Scalar> m_triangle_intersector;

  nanort::BVHBuildOptions<typename T::Scalar> m_build_options;
  std::unique_ptr<nanort::TriangleMesh<typename T::Scalar>> m_triangle_mesh;
  std::unique_ptr<nanort::TriangleSAHPred<typename T::Scalar>> m_triangle_pred;
  nanort::BVHAccel<typename T::Scalar> m_accel;
  nanort::BVHBuildStatistics m_stats;
  T m_bmin, m_bmax;
};

}  // namespace ugu

#endif