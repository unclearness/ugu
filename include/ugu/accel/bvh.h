/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "Eigen/Core"
#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"

namespace ugu {

template <typename T>
struct Aabb {
  T max_v;
  T min_v;
  T length;
  T center;
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> indices;

  void Init(const T& max_v, const T& min_v) {
    this->max_v = max_v;
    this->min_v = min_v;
    length = max_v - min_v;
    center = (max_v + min_v) * 0.5;

    vertices.resize(24);
    indices.resize(12);
    vertices[0] = Eigen::Vector3f(min_v[0], max_v[1], min_v[2]);
    vertices[1] = Eigen::Vector3f(max_v[0], max_v[1], min_v[2]);
    vertices[2] = Eigen::Vector3f(max_v[0], max_v[1], max_v[2]);
    vertices[3] = Eigen::Vector3f(min_v[0], max_v[1], max_v[2]);
    indices[0] = Eigen::Vector3i(0, 2, 1);
    indices[1] = Eigen::Vector3i(0, 3, 2);

    vertices[4] = Eigen::Vector3f(min_v[0], min_v[1], min_v[2]);
    vertices[5] = Eigen::Vector3f(max_v[0], min_v[1], min_v[2]);
    vertices[6] = Eigen::Vector3f(max_v[0], min_v[1], max_v[2]);
    vertices[7] = Eigen::Vector3f(min_v[0], min_v[1], max_v[2]);
    indices[2] = Eigen::Vector3i(4, 5, 6);
    indices[3] = Eigen::Vector3i(4, 6, 7);

    vertices[8] = vertices[1];
    vertices[9] = vertices[2];
    vertices[10] = vertices[6];
    vertices[11] = vertices[5];
    indices[4] = Eigen::Vector3i(8, 9, 10);
    indices[5] = Eigen::Vector3i(8, 10, 11);

    vertices[12] = vertices[0];
    vertices[13] = vertices[3];
    vertices[14] = vertices[7];
    vertices[15] = vertices[4];
    indices[6] = Eigen::Vector3i(12, 14, 13);
    indices[7] = Eigen::Vector3i(12, 15, 14);

    vertices[16] = vertices[0];
    vertices[17] = vertices[1];
    vertices[18] = vertices[5];
    vertices[19] = vertices[4];
    indices[8] = Eigen::Vector3i(16, 17, 18);
    indices[9] = Eigen::Vector3i(16, 18, 19);

    vertices[20] = vertices[3];
    vertices[21] = vertices[2];
    vertices[22] = vertices[6];
    vertices[23] = vertices[7];
    indices[10] = Eigen::Vector3i(20, 22, 21);
    indices[11] = Eigen::Vector3i(20, 23, 22);
  };

  Aabb(){};
  ~Aabb(){};
  Aabb(const T& max_v, const T& min_v) { Init(max_v, min_v); }
  Aabb(const std::vector<T>& points) {
    max_v = ComputeMaxBound(points);
    min_v = ComputeMinBound(points);
    Init(max_v, min_v);
  }

#if 0
  template <typename TT>
  Aabb(const std::vector<T>& vertices, const std::vector<TT>& indices) {
    std::vector<T> points;
    for (const auto& i : indices) {
      for (int j = 0; j < i.rows(); j++) {
        points.push_back(vertices[i[j]]);
      }
    }
    max_v = ComputeMaxBound(points);
    min_v = ComputeMinBound(points);
    Init(max_v, min_v);
  }
#endif

  static double SurfaceArea(const Aabb& x) {
    double sum = 0.0;
    for (int i = 0; i < x.rows(); i++) {
      for (int j = i + 1; j < x.rows(); j++) {
        sum = static_cast<double>(x.length[i] * x.length[j]);
      }
    }
    return 2.0 * sum;
  }
  double SurfaceArea() const { return SurfaceArea(*this); }
};

using Aabb3f = Aabb<Eigen::Vector3f>;

#if 0
template <typename T = float>
class Ray {
 public:
  Ray() : min_t(static_cast<T>(0.0)), max_t(std::numeric_limits<T>::max()) {
    org[0] = static_cast<T>(0.0);
    org[1] = static_cast<T>(0.0);
    org[2] = static_cast<T>(0.0);
    dir[0] = static_cast<T>(0.0);
    dir[1] = static_cast<T>(0.0);
    dir[2] = static_cast<T>(-1.0);
  }

  T org[3];  // must set
  T dir[3];  // must set
  T min_t;   // minimum ray hit distance.
  T max_t;   // maximum ray hit distance.
};
#endif

struct Ray {
  Eigen::Vector3f org, dir;
};

template <typename T, typename TT>
class Bvh {
 public:
  Bvh(){};
  ~Bvh(){};

  enum kCostType { NAIVE, SAH };  // Surface Area Heuristics

  void SetData(const std::vector<T>& vertices, const std::vector<TT>& indices) {
    m_vertices = vertices;
    m_indices = indices;
  }

  void SetAxisNum(int axis_num) { m_axis_num = axis_num; }

  void SetMaxLeafDataNum(int max_leaf_data_num) {
    m_max_leaf_data_num = max_leaf_data_num;
  }

  void SetCostType(kCostType cost_type) { m_cost_type = cost_type; }

  bool Build() {
    if (m_axis_num < 0 || m_vertices.empty() || m_indices.empty()) {
      return false;
    }
    m_face_ids.clear();
    m_face_ids.resize(m_indices.size());
    std::iota(m_face_ids.begin(), m_face_ids.end(), 0);
    m_root = BuildImpl(m_face_ids, 0);
    return true;
  }

  std::vector<MeshPtr> Visualize(int depth_limit = 10) const {
    std::vector<NodePtr> q{m_root};

    std::vector<MeshPtr> meshes;

    auto bbox_to_mesh = [&](const Aabb<T>& bbox) -> ugu::MeshPtr {
      // Eigen::Vector3f length = bbox.length.cast<float>();
      // ugu::MeshPtr mesh = ugu::MakeCube(length);
      // mesh->Translate(bbox.center.cast<float>());
      ugu::MeshPtr mesh = ugu::Mesh::Create();
      mesh->set_vertices(bbox.vertices);
      mesh->set_vertex_indices(bbox.indices);

      std::vector<ugu::ObjMaterial> materials(1);
      mesh->set_materials(materials);
      std::vector<int> material_ids(bbox.indices.size(), 0);
      mesh->set_material_ids(material_ids);
      mesh->CalcNormal();

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

  bool Intersect(const Ray& ray, std::vector<IntersectResult>& results) const {
    results = IntersectImpl(ray, m_root);
    std::sort(results.begin(), results.end(),
              [&](const IntersectResult& lhs, const IntersectResult& rhs) {
                return lhs.t < rhs.t;
              });
    return !results.empty();
  }

 private:
  std::vector<T> m_vertices;
  std::vector<TT> m_indices;
  std::vector<size_t> m_face_ids;
  int m_axis_num = -1;
  int m_max_leaf_data_num = -1;
  kCostType m_cost_type = kCostType::NAIVE;

  int m_num_threads = 1;
  float m_epsilon = 1e-6f;

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
    if (m_max_leaf_data_num <= 0) {
      if (poly_num <= 0) {
        return nullptr;
      }
    } else {
      if (static_cast<int>(poly_num) <= m_max_leaf_data_num) {
        NodePtr node = std::make_shared<Node>();
        node->face_ids = face_ids;
        node->depth = depth;
        for (const auto& f : face_ids) {
          node->indices.push_back(m_indices[f]);
        }
        return node;
      }
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
    if (m_cost_type == kCostType::NAIVE) {
      auto calc_center = [&](const TT& index) {
#if 0
        auto center = std::accumulate(
                          index.data(), index.data() + index.rows(), T::Zero(),
                          [&](const TT::Scalar& lhs, const TT::Scalar& rhs) {
                            return m_vertices[lhs] + m_vertices[rhs];
                          }) /
                      index.rows();
#endif
        T center = T::Zero();
        for (int i = 0; i < index.rows(); i++) {
          center += m_vertices[index[i]];
        }
        center /= index.rows();

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
        c.fid = node->face_ids[c.fid];
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
