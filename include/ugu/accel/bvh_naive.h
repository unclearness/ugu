/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include "ugu/accel/bvh_base.h"

namespace ugu {

template <typename T, typename TT>
class BvhNaive : public Bvh<T, TT> {
 public:
  BvhNaive(){};
  ~BvhNaive(){};

  void SetData(const std::vector<T>& vertices,
               const std::vector<TT>& indices) override {
    m_vertices = vertices;
    m_indices = indices;
  }

  void SetAxisNum(int axis_num) { m_axis_num = axis_num; }

  void SetMinLeafPrimitives(int min_leaf_primitives) {
    m_min_leaf_primitives = min_leaf_primitives;
  }

  void SetCostType(kBvhCostType cost_type) { m_cost_type = cost_type; }

  bool Build() override {
    if (m_axis_num < 0 || m_vertices.empty() || m_indices.empty()) {
      return false;
    }
    m_face_ids.clear();
    m_face_ids.resize(m_indices.size());
    std::iota(m_face_ids.begin(), m_face_ids.end(), 0);
    m_root = BuildImpl(m_face_ids, 0);
    return true;
  }

  const BvhBuildStatistics& GetBuildStatistics() const override {
    LOGW("Maybe not implemented...");
    return m_stats;
  }

  std::vector<MeshPtr> Visualize(int depth_limit = 10) const {
    std::vector<NodePtr> q{m_root};

    std::vector<MeshPtr> meshes;

    auto bbox_to_mesh = [&](const Aabb<T>& bbox) -> ugu::MeshPtr {
#if 0
      Eigen::Vector3f length = bbox.length.cast<float>();
      ugu::MeshPtr mesh = ugu::MakeCube(length);
      mesh->Translate(bbox.center.cast<float>());
#else
      ugu::MeshPtr mesh = ugu::Mesh::Create();
      mesh->set_vertices(bbox.vertices);
      mesh->set_vertex_indices(bbox.indices);

      std::vector<ugu::ObjMaterial> materials(1);
      mesh->set_materials(materials);
      std::vector<int> material_ids(bbox.indices.size(), 0);
      mesh->set_material_ids(material_ids);
      mesh->CalcNormal();
#endif
      return mesh;
    };

    int loop_cnt = 0;
    while (!q.empty()) {
      auto b = q.back();
      q.pop_back();

      auto mesh = bbox_to_mesh(b->bbox);
      SetRandomUniformVertexColor(mesh, loop_cnt);

      for (const auto& c : b->children) {
        if (c != nullptr && c->depth <= depth_limit) {
          q.push_back(c);
        }
      }

      meshes.push_back(mesh);

      loop_cnt++;
    }
    return meshes;
  }

  std::vector<IntersectResult> Intersect(const Ray& ray,
                                         bool test_all = true) const override {
    // TODO: immediately return only 1 result if test_all = false
    std::vector<IntersectResult> results = IntersectImpl(ray, m_root);
    std::sort(results.begin(), results.end(),
              [&](const IntersectResult& lhs, const IntersectResult& rhs) {
                return lhs.t < rhs.t;
              });
    // TODO:
    if (!test_all && !results.empty()) {
      results.resize(1);
    }

    return results;
  }

 private:
  std::vector<T> m_vertices;
  std::vector<TT> m_indices;
  std::vector<size_t> m_face_ids;
  int m_axis_num = -1;
  unsigned int m_min_leaf_primitives = 4;
  unsigned int m_max_tree_depth = 256;
  kBvhCostType m_cost_type = kBvhCostType::NAIVE;
  BvhBuildStatistics m_stats;

  int m_num_threads = 1;
  float m_epsilon = 1e-8f;

  struct Node;
  using NodePtr = std::shared_ptr<Node>;
  struct Node {
    Node(){};
    ~Node(){};
    Aabb<T> bbox;
    std::array<NodePtr, 2> children{nullptr, nullptr};  // TODO
    std::vector<TT> indices;
    std::vector<size_t> face_ids;
    int depth = -1;
    int axis = -1;
    bool isLeaf() const {
      if (children[0] == nullptr || children[1] == nullptr) {
        return true;
      }
      return false;
    }
  };
  NodePtr m_root = nullptr;

  NodePtr BuildImpl(const std::vector<size_t>& face_ids, int depth) {
    const size_t poly_num = face_ids.size();

    if (static_cast<uint32_t>(poly_num) <= m_min_leaf_primitives ||
        m_max_tree_depth <= static_cast<uint32_t>(depth)) {
      NodePtr node = std::make_shared<Node>();
      node->face_ids = face_ids;
      node->depth = depth;
      for (const auto& f : face_ids) {
        node->indices.push_back(m_indices[f]);
      }
      return node;
    }

    auto [left_fids, right_fids, axis] = Split(face_ids, depth);

    NodePtr node = std::make_shared<Node>();
    node->axis = axis;
    node->depth = depth;
    std::vector<T> points;
    for (const auto& f : face_ids) {
      for (int i = 0; i < m_indices[f].rows(); i++) {
        points.push_back(m_vertices[m_indices[f][i]]);
      }
    }
    node->bbox = Aabb(points);
    node->children[0] = BuildImpl(left_fids, depth + 1);
    node->children[1] = BuildImpl(right_fids, depth + 1);

    return node;
  }

  std::tuple<std::vector<size_t>, std::vector<size_t>, int> Split(
      const std::vector<size_t>& face_ids, int depth) {
    std::vector<size_t> left_fids, right_fids;
    int axis = -1;
    if (m_cost_type == kBvhCostType::NAIVE) {
      auto calc_center = [&](const TT& index) {
#if 0
        // Not smart...
        std::vector<T> tmp;
        std::transform(index.data(), index.data() + index.rows(),
                       std::back_inserter(tmp),
                       [&](const TT::Scalar& x) { return m_vertices[x]; });
        T center = std::accumulate(tmp.begin(), tmp.end(), T::Zero().eval()) /
                   tmp.size();
#else
        T center = T::Zero();
        for (int i = 0; i < index.rows(); i++) {
          center += m_vertices[index[i]];
        }
        center /= static_cast<float>(index.rows());
#endif
        return center;
      };

      const size_t poly_num = face_ids.size();
      axis = depth % m_axis_num;

      const size_t mid_index = poly_num / 2;

      left_fids = face_ids;

      std::sort(left_fids.begin(), left_fids.end(),
                [&](const size_t& lhs, const size_t& rhs) {
                  auto center_lhs = calc_center(m_indices[lhs]);
                  auto center_rhs = calc_center(m_indices[rhs]);
                  return center_lhs[axis] < center_rhs[axis];
                });

      std::copy(left_fids.begin() + mid_index, left_fids.end(),
                std::back_inserter(right_fids));
      left_fids.resize(left_fids.size() - mid_index);

    } else {
      throw std::exception();
    }
    return {left_fids, right_fids, axis};
  }

  std::vector<IntersectResult> IntersectImpl(const Ray& ray,
                                             NodePtr node) const {
    if (node->isLeaf()) {
      auto cur = ugu::Intersect(ray.org, ray.dir, m_vertices, node->indices,
                                m_num_threads, m_epsilon);
      // Convert face id from node to original
      for (auto& c : cur) {
        c.fid = static_cast<uint32_t>(node->face_ids[c.fid]);
      }
      return cur;
    }

    auto cur_bbox_res =
        ugu::Intersect(ray.org, ray.dir, node->bbox.vertices,
                       node->bbox.indices, m_num_threads, m_epsilon);
    if (cur_bbox_res.empty()) {
      return {};
    }

    auto child_res0 = IntersectImpl(ray, node->children[0]);
    std::vector<IntersectResult> results = child_res0;
    auto child_res1 = IntersectImpl(ray, node->children[1]);
    std::copy(child_res1.begin(), child_res1.end(),
              std::back_inserter(results));
    return results;
  }
};

}  // namespace ugu
