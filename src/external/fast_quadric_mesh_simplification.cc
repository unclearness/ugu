/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/external/external.h"

#ifdef UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION

#ifdef _WIN32
#pragma warning(push, 0)
#endif

#include "Simplify.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

#endif

namespace ugu {

bool FastQuadricMeshSimplification(const Mesh& src, int target_face_num,
                                   Mesh* decimated) {
#ifdef UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION
  if (target_face_num > src.vertex_indices().size()) {
    return false;
  }

  // convert vertices and faces
  Simplify::vertices.clear();
  Simplify::triangles.clear();
  Simplify::refs.clear();
  Simplify::materials.clear();

  Simplify::vertices.resize(src.vertices().size());
  for (int i = 0; i < static_cast<int>(src.vertices().size()); i++) {
    Simplify::Vertex& v = Simplify::vertices[i];
    v.p.x = src.vertices()[i].x();
    v.p.y = src.vertices()[i].y();
    v.p.z = src.vertices()[i].z();
  }

  Simplify::triangles.resize(src.vertex_indices().size());
  for (int i = 0; i < static_cast<int>(src.vertex_indices().size()); i++) {
    Simplify::Triangle& t = Simplify::triangles[i];
    for (int j = 0; j < 3; j++) {
      t.v[j] = src.vertex_indices()[i][j];
    }
  }

  std::array<double, 8> agressiveness_list = {
      {1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0, 100.0}};

  bool ret = true;
  const int max_iter = static_cast<int>(agressiveness_list.size());

  for (int i = 0; i < max_iter; i++) {
    double agressiveness = agressiveness_list[i];
    Simplify::simplify_mesh(target_face_num, agressiveness, false);
    ugu::LOGI(
        "FastQuadricMeshSimplification: iter %d, agressiveness %f, current "
        "%d\n",
        i, agressiveness, Simplify::triangles.size());
    if (Simplify::triangles.size() <= target_face_num) {
      break;
    }
  }

  if (Simplify::triangles.size() > target_face_num) {
    ret = false;
    ugu::LOGE("FastQuadricMeshSimplification failed: target %d, result: %d\n",
              target_face_num, Simplify::triangles.size());
  } else {
    ugu::LOGI(
        "FastQuadricMeshSimplification succeeded: target %d, result: %d\n",
        target_face_num, Simplify::triangles.size());
  }

  // convert back vertices and faces
  decimated->Clear();

  std::vector<Eigen::Vector3f> vertices(Simplify::vertices.size());
  for (int i = 0; i < static_cast<int>(vertices.size()); i++) {
    Simplify::Vertex& v = Simplify::vertices[i];
    vertices[i].x() = static_cast<float>(v.p.x);
    vertices[i].y() = static_cast<float>(v.p.y);
    vertices[i].z() = static_cast<float>(v.p.z);
  }

  std::vector<Eigen::Vector3i> vertex_indices(Simplify::triangles.size());
  for (int i = 0; i < static_cast<int>(vertex_indices.size()); i++) {
    Simplify::Triangle& t = Simplify::triangles[i];
    for (int j = 0; j < 3; j++) {
      vertex_indices[i][j] = t.v[j];
    }
  }

  // TODO: multiple uv for a single vertex (obj format)
  decimated->set_materials(src.materials());
  std::vector<int> material_ids(vertex_indices.size(), 0);
  decimated->set_material_ids(material_ids);

  decimated->set_vertices(vertices);
  decimated->set_vertex_indices(vertex_indices);

  decimated->RemoveUnreferencedVertices();
  decimated->CalcNormal();

  return ret;
#else
  (void)src;
  (void)target_face_num;
  (void)decimated;
  ugu::LOGE("Not available in current configuration\n");
  return false;
#endif
}

}  // namespace ugu
