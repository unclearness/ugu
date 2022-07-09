/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <filesystem>
#include <iostream>
#include <random>

#include "ugu/decimation/decimation.h"
#include "ugu/external/external.h"
#include "ugu/inpaint/inpaint.h"
#include "ugu/mesh.h"
#include "ugu/parameterize/parameterize.h"
#include "ugu/timer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"
#include "ugu/util/rgbd_util.h"

namespace {

inline std::vector<std::string> Split(const std::string& s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      elems.push_back(item);
    }
  }
  return elems;
}

inline bool GetFileNames(std::string folderPath,
                         std::vector<std::string>& file_names) {
  using namespace std::filesystem;
  directory_iterator iter(folderPath), end;
  std::error_code err;

  for (; iter != end && !err; iter.increment(err)) {
    const directory_entry entry = *iter;
    std::string name = *(Split(entry.path().string(), '/').end() - 1);
    file_names.push_back(name);
  }

  if (err) {
    std::cout << err.value() << std::endl;
    std::cout << err.message() << std::endl;
    return false;
  }
  return true;
}

inline std::string ExtractPathExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return "";
  }
  return fn.substr(pos + 1, fn.size());
}

inline std::string ExtractPathWithoutExt(const std::string& fn) {
  std::string::size_type pos;
  if ((pos = fn.find_last_of(".")) == std::string::npos) {
    return fn;
  }

  return fn.substr(0, pos);
}

void TestBlendshapes() {
  std::string data_dir = "../data/blendshape/";
  std::vector<std::string> file_names;
  GetFileNames(data_dir, file_names);
  std::string base_name = "cube.obj";
  ugu::Mesh base_mesh;
  base_mesh.LoadObj(data_dir + base_name, data_dir);
  base_mesh.SplitMultipleUvVertices();

  std::vector<std::string> obj_names;
  for (auto& name : file_names) {
    auto ext = ExtractPathExt(name);
    if (name == base_name || ext != "obj") {
      continue;
    }
    obj_names.push_back(name);
  }

  std::vector<ugu::Mesh> blendshape_meshes;
  for (auto& name : obj_names) {
    ugu::Mesh mesh;
    mesh.LoadObj(data_dir + name, data_dir);
    mesh.SplitMultipleUvVertices();
    blendshape_meshes.push_back(mesh);
  }

  std::vector<ugu::Blendshape> blendshapes;
  for (int i = 0; i < blendshape_meshes.size(); i++) {
    auto& blendshape_mesh = blendshape_meshes[i];
    auto name = ExtractPathWithoutExt(obj_names[i]);

    ugu::Blendshape blendshape;
    std::vector<Eigen::Vector3f> displacement = base_mesh.vertices();
    for (size_t j = 0; j < displacement.size(); j++) {
      displacement[j] = blendshape_mesh.vertices()[j] - displacement[j];
    }
    blendshape.vertices = displacement;
    blendshape.normals = blendshape_mesh.normals();
    blendshape.name = name;

    blendshapes.push_back(blendshape);
  }
  base_mesh.set_blendshapes(blendshapes);

  base_mesh.WriteGltfSeparate(data_dir, "blendshape");
  base_mesh.WriteGlb(data_dir, "blendshape.glb");
}

void TestIO() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";

  ugu::Mesh src, dst;

  src.LoadObj(in_obj_path, data_dir);

  src.SplitMultipleUvVertices();
  src.WriteGltfSeparate(data_dir, "bunny");

  src.WriteGlb(data_dir, "bunny.glb");

  dst = ugu::Mesh(src);

  dst.FlipFaces();

  dst.WriteObj(data_dir, "bunny2");
}

void TestMerge() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny, bunny_moved, dst;
  bunny.LoadObj(in_obj_path1, data1_dir);
  bunny_moved = ugu::Mesh(bunny);
  bunny_moved.Translate(bunny.stats().bb_max);
  bunny_moved.FlipFaces();  // Flip face for moved bunny

  std::string data2_dir = "../data/buddha/";
  std::string in_obj_path2 = data2_dir + "buddha.obj";
  ugu::Mesh buddha;
  buddha.LoadObj(in_obj_path2, data2_dir);

  buddha.SplitMultipleUvVertices();
  buddha.WriteGlb(data2_dir, "buddha.glb");

  ugu::MergeMeshes(bunny, buddha, &dst);
  dst.WriteObj(data1_dir, "bunny_and_buddha");

  ugu::MergeMeshes(bunny, bunny_moved, &dst, true);
  dst.WriteObj(data1_dir, "bunny_twin");

  ugu::MergeMeshes(bunny, bunny_moved, &dst);
  dst.WriteObj(data1_dir, "bunny_twin_2materials");
}

void TestRemove() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny;
  bunny.LoadObj(in_obj_path1, data1_dir);

  std::vector<bool> valid_face_table;

  Eigen::Vector3f direc(0.0, 0.0, 1.0f);
  for (const auto& fn : bunny.face_normals()) {
    if (direc.dot(fn) > 0.0) {
      valid_face_table.push_back(true);
    } else {
      valid_face_table.push_back(false);
    }
  }

  bunny.RemoveFaces(valid_face_table);

  bunny.WriteObj(data1_dir, "bunny_removed_back");
}

void TestAlignment() {
  std::string data1_dir = "../data/bunny/";
  std::string in_obj_path1 = data1_dir + "bunny.obj";
  ugu::Mesh bunny;
  bunny.LoadObj(in_obj_path1, data1_dir);

  // Add noise
  ugu::Mesh noised_bunny = ugu::Mesh(bunny);
  bunny.CalcStats();
  const auto& stats = bunny.stats();
  std::mt19937 engine(0);
  float bb_mean = (stats.bb_max - stats.bb_min).mean();
  float sigma = bb_mean * 0.01f;
  std::normal_distribution<float> gauss(0.0f, sigma);
  auto org_vertices = bunny.vertices();
  auto noised_vertices = org_vertices;
  for (size_t i = 0; i < noised_vertices.size(); i++) {
    auto& v = noised_vertices[i];
    auto& n = bunny.normals()[i];
    v += gauss(engine) * n;
  }
  noised_bunny.set_vertices(noised_vertices);

  Eigen::Vector3f noise_t{bb_mean, bb_mean * 2, bb_mean * -3};
  Eigen::Vector3f axis(5, 2, 1);
  axis.normalize();
  Eigen::AngleAxisf noise_R(ugu::radians(30.f), axis);
  float noise_s = 0.45f;
  Eigen::Affine3f T_Rt_gt = Eigen::Translation3f(noise_t) * noise_R.matrix();
  ugu::Mesh noised_bunny_Rt = ugu::Mesh(noised_bunny);
  noised_bunny_Rt.Transform(T_Rt_gt.rotation(), T_Rt_gt.translation());
  noised_bunny_Rt.WriteObj(data1_dir, "noised_Rt_gt");
  std::cout << "GT Rt" << std::endl;
  std::cout << T_Rt_gt.matrix() << std::endl;

  Eigen::Affine3f T_Rt_estimated = ugu::FindRigidTransformFrom3dCorrespondences(
                                       org_vertices, noised_bunny_Rt.vertices())
                                       .cast<float>();
  std::cout << "Estimated Rt" << std::endl;
  std::cout << T_Rt_estimated.matrix() << std::endl;
  bunny.Transform(T_Rt_estimated.rotation(), T_Rt_estimated.translation());
  bunny.WriteObj(data1_dir, "noised_Rt_estimated");
  bunny.Transform(T_Rt_estimated.inverse().rotation(),
                  T_Rt_estimated.inverse().translation());

  ugu::Mesh noised_bunny_Rts = ugu::Mesh(noised_bunny);
  Eigen::Affine3f T_Rts_gt = Eigen::Translation3f(noise_t) * noise_R.matrix() *
                             Eigen::Scaling(noise_s);
  noised_bunny_Rts.Scale(noise_s);
  noised_bunny_Rts.Transform(T_Rts_gt.rotation(), T_Rts_gt.translation());
  noised_bunny_Rts.WriteObj(data1_dir, "noised_Rts_gt");
  std::cout << "GT Rts" << std::endl;
  std::cout << T_Rts_gt.matrix() << std::endl;

  Eigen::Affine3f T_Rts_estimated =
      ugu::FindSimilarityTransformFrom3dCorrespondences(
          org_vertices, noised_bunny_Rts.vertices())
          .cast<float>();
  std::cout << "Estimated Rts" << std::endl;
  std::cout << T_Rts_estimated.matrix() << std::endl;
  float estimated_s_x =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(0).norm();
  float estimated_s_y =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(1).norm();
  float estimated_s_z =
      T_Rts_estimated.matrix().block(0, 0, 3, 3).row(2).norm();
  bunny.Scale(estimated_s_x, estimated_s_y, estimated_s_z);
  bunny.Transform(T_Rts_estimated.rotation(), T_Rts_estimated.translation());
  bunny.WriteObj(data1_dir, "noised_Rts_estimated");
}

void TestTexture() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";
  ugu::Mesh bunny;
  bunny.LoadObj(in_obj_path, data_dir);

  std::vector<Eigen::Vector3f> vertex_colors;
  std::vector<Eigen::Vector2f> vertex_uv(bunny.vertices().size());
  // Convert per-face uv to per-vertex uv
  for (size_t i = 0; i < bunny.uv_indices().size(); i++) {
    const auto pos_f = bunny.vertex_indices()[i];
    const auto uv_f = bunny.uv_indices()[i];
    for (size_t j = 0; j < 3; j++) {
      const auto& vid = pos_f[j];
      vertex_uv[vid] = bunny.uv()[uv_f[j]];
    }
  }
  ugu::FetchVertexAttributeFromTexture(bunny.materials()[0].diffuse_tex,
                                       vertex_uv, vertex_colors);

  bunny.set_vertex_colors(vertex_colors);

  bunny.WritePly(data_dir + "fetched_vertex_color.ply");

  ugu::Parameterize(bunny, 1024, 0124,
                    ugu::ParameterizeUvType::kSimpleTriangles);

  ugu::Image1b mask = ugu::Image1b::zeros(1024, 1024);
  auto rerasterized = bunny.materials()[0].diffuse_tex.clone();
  ugu::RasterizeVertexAttributeToTexture(vertex_colors, bunny.vertex_indices(),
                                         bunny.uv(), bunny.uv_indices(),
                                         rerasterized, 1024, 1024, &mask);
  ugu::Image1b inv_mask = mask.clone();
  ugu::Not(mask, &inv_mask);
  ugu::Inpaint(inv_mask, rerasterized, 3.f, ugu::InpaintMethod::TELEA);
  auto mat = bunny.materials();
  mat[0].diffuse_tex = rerasterized;
  mat[0].diffuse_texname = "rerasterized.jpg";

  bunny.set_materials(mat);

  bunny.WriteObj(data_dir, "rerasterized");
}

void TestCut() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";
  auto bunny = ugu::Mesh::Create();
  bunny->LoadObj(in_obj_path, data_dir);
  Eigen::Vector3f n(1.f, 0.f, 1.f);
  n.normalize();

  ugu::Planef plane(n, 50.f);
  ugu::CutByPlane(bunny, plane);

  bunny->WritePly(data_dir + "cut_by_plane.ply");
  bunny->WriteObj(data_dir, "cut_by_plane");
}

void TestDecimation() {
  {
    std::string data_dir = "../data/plane/";
    std::string in_obj_path = data_dir + "plane.obj";
    ugu::MeshPtr src = ugu::Mesh::Create();
    ugu::Mesh dst;
    src->LoadObj(in_obj_path, data_dir);
    ugu::QSlim(src, ugu::QSlimType::XYZ_UV,
               static_cast<int32_t>(src->vertex_indices().size() * 0.1), -1);

    src->WriteObj(data_dir, "plane_qslim");
  }

  {
    std::string data_dir = "../data/spot/";
    std::string in_obj_path = data_dir + "spot_triangulated.obj";
    ugu::MeshPtr src = ugu::Mesh::Create();
    ugu::Mesh dst;
    src->LoadObj(in_obj_path, data_dir);

    ugu::QSlim(src, ugu::QSlimType::XYZ_UV,
               static_cast<int32_t>(src->vertex_indices().size() * 0.1), -1);

    src->WriteObj(data_dir, "spot_qslim");
  }

  {
    std::string data_dir = "../data/bunny/";
    std::string in_obj_path = data_dir + "bunny.obj";
    ugu::MeshPtr src = ugu::Mesh::Create();
    ugu::Mesh dst;
    src->LoadObj(in_obj_path, data_dir);

    auto targe_face_num = static_cast<int>(src->vertex_indices().size() * 0.02);

    ugu::FastQuadricMeshSimplification(*src, targe_face_num, &dst);
    dst.WritePly(data_dir + "bunny_fast_decimated.ply");

    ugu::QSlim(src, ugu::QSlimType::XYZ_UV, targe_face_num, -1);

    src->WriteObj(data_dir, "bunny_qslim");
  }
}

void TestRayInteresection() {
  std::string data_dir = "../data/bunny/";
  std::string in_obj_path = data_dir + "bunny.obj";
  auto bunny = ugu::Mesh::Create();
  bunny->LoadObj(in_obj_path, data_dir);

  Eigen::Vector3f origin(0.f, 0.f, 700.f);
  Eigen::Vector3f ray(0.f, 0.f, -1.f);
  ugu::Timer timer;
  timer.Start();
  auto results = ugu::Intersect(origin, ray, bunny->vertices(),
                                bunny->vertex_indices(), 1);
  timer.End();
  ugu::LOGI("ugu::Intersect took %f ms\n", timer.elapsed_msec());
  std::vector<Eigen::Vector3f> intersected_points;
  for (auto result : results) {
    auto pos_t = origin + ray * result.t;
    const auto& face = bunny->vertex_indices()[result.fid];
    auto pos_uv = (1.f - (result.u + result.v)) * bunny->vertices()[face[0]] +
                  result.u * bunny->vertices()[face[1]] +
                  result.v * bunny->vertices()[face[2]];
    ugu::LOGI("%d (%f, %f), %f, (%f, %f, %f) (%f, %f, %f)\n", result.fid,
              result.u, result.v, result.t, pos_t.x(), pos_t.y(), pos_t.z(),
              pos_uv.x(), pos_uv.y(), pos_uv.z());
    intersected_points.push_back(pos_t);
  }

  auto tmp = ugu::Mesh::Create();
  tmp->set_vertices(intersected_points);
  tmp->WritePly("intersected.ply");
}

void TestMakeGeom() {
  std::string data_dir = "../data/";
  auto cone = ugu::MakeCone(0.5f, 1.f);
  cone->WriteObj(data_dir, "cone");

  auto cylinder = ugu::MakeCylinder(0.5f, 1.f);
  cylinder->WriteObj(data_dir, "cylinder");

  ugu::ObjMaterial cylinder_mat, cone_mat;
  cylinder_mat.diffuse = {0.f, 1.f, 0.f};
  cone_mat.diffuse = {1.f, 0.f, 1.f};
  auto arrow =
      ugu::MakeArrow(0.1f, 1.f, 0.2f, 0.2f, 30, 30, cylinder_mat, cone_mat);
  arrow->WriteObj(data_dir, "arrow");

  auto origin = ugu::MakeOrigin(1.f);
  origin->WriteObj(data_dir, "origin");

  std::vector<Eigen::Affine3d> poses;
  ugu::LoadTumFormat("../data/bunny/tumpose.txt", &poses);
  auto trajectory = ugu::MakeTrajectoryGeom(poses, 100.f);
  trajectory->WriteObj(data_dir, "tumpose");

  auto frustum = ugu::MakeFrustum(1.f, 0.8f, 0.5f, 0.4f, 1.f);
  frustum->WriteObj(data_dir, "frustum");

  ugu::Image3b view0_image =
      ugu::imread<ugu::Image3b>("../data/bunny/00000_color.png");
  auto view_frustum =
      ugu::MakeViewFrustum(ugu::radians(30.f), poses[0].cast<float>(), 200.f,
                           view0_image, ugu::CoordinateType::OpenCV, 10.f);
  view_frustum->WriteObj(data_dir, "view_frustum");
}

}  // namespace

int main() {
  TestMakeGeom();

  TestRayInteresection();

  TestDecimation();

  TestCut();

  TestTexture();

  TestAlignment();

  TestBlendshapes();

  TestIO();

  TestMerge();

  TestRemove();

  return 0;
}
