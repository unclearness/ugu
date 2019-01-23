/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "include/renderer.h"

#include <cassert>
#include <array>
#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#include <windows.h>
#ifdef __cplusplus
}
#endif
#pragma comment(lib, "winmm.lib")
#else
#if defined(__unix__) || defined(__APPLE__)
#include <sys/time.h>
#else
#include <ctime>
#endif
#endif

#include "include/pixel_shader.h"

namespace {
// This class is NOT thread-safe timer!
class timerutil {
 public:
#ifdef _WIN32
  typedef DWORD time_t;

  timerutil() { ::timeBeginPeriod(1); }
  ~timerutil() { ::timeEndPeriod(1); }

  void start() { t_[0] = ::timeGetTime(); }
  void end() { t_[1] = ::timeGetTime(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / 1000); }
  time_t msec() { return (time_t)((t_[1] - t_[0])); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000); }
  time_t current() { return ::timeGetTime(); }

#else
#if defined(__unix__) || defined(__APPLE__)
  typedef unsigned long int time_t;

  void start() { gettimeofday(tv + 0, &tz); }
  void end() { gettimeofday(tv + 1, &tz); }

  time_t sec() { return (time_t)(tv[1].tv_sec - tv[0].tv_sec); }
  time_t msec() {
    return this->sec() * 1000 +
           (time_t)((tv[1].tv_usec - tv[0].tv_usec) / 1000);
  }
  time_t usec() {
    return this->sec() * 1000000 + (time_t)(tv[1].tv_usec - tv[0].tv_usec);
  }
  time_t current() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (time_t)(t.tv_sec * 1000 + t.tv_usec);
  }

#else  // C timer
  // using namespace std;
  typedef clock_t time_t;

  void start() { t_[0] = clock(); }
  void end() { t_[1] = clock(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / CLOCKS_PER_SEC); }
  time_t msec() { return (time_t)((t_[1] - t_[0]) * 1000 / CLOCKS_PER_SEC); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000000 / CLOCKS_PER_SEC); }
  time_t current() { return (time_t)clock(); }

#endif
#endif

 private:
#ifdef _WIN32
  DWORD t_[2];
#else
#if defined(__unix__) || defined(__APPLE__)
  struct timeval tv[2];
  struct timezone tz;
#else
  time_t t_[2];
#endif
#endif
};
}  // namespace

namespace crender {

RendererOption::RendererOption() {}

RendererOption::~RendererOption() {}

void RendererOption::copy_to(RendererOption& dst) const {
  dst.use_vertex_color = use_vertex_color;
  dst.depth_scale = depth_scale;
  dst.interp = interp;
  dst.backface_culling = backface_culling;
}

Renderer::Renderer() {}

Renderer::~Renderer() {}

Renderer::Renderer(const RendererOption& option) {
  set_option(option);
}

void Renderer::set_option(const RendererOption& option) {
  option.copy_to(option_);
}

void Renderer::set_mesh(std::shared_ptr<Mesh> mesh) {
  mesh_initialized_ = false;
  mesh_ = mesh;

  flatten_vertices.clear();
  flatten_faces.clear();

  const std::vector<glm::vec3>& vertices = mesh_->vertices();
  flatten_vertices.resize(vertices.size() * 3);
  for (size_t i = 0; i < vertices.size(); i++) {
    flatten_vertices[i * 3 + 0] = vertices[i][0];
    flatten_vertices[i * 3 + 1] = vertices[i][1];
    flatten_vertices[i * 3 + 2] = vertices[i][2];
  }

  const std::vector<glm::ivec3>& vertex_indices = mesh_->vertex_indices();
  flatten_faces.resize(vertex_indices.size() * 3);
  for (size_t i = 0; i < vertex_indices.size(); i++) {
    flatten_faces[i * 3 + 0] = vertex_indices[i][0];
    flatten_faces[i * 3 + 1] = vertex_indices[i][1];
    flatten_faces[i * 3 + 2] = vertex_indices[i][2];
  }
}
bool Renderer::prepare_mesh() {
  if (mesh_ == nullptr) {
    LOGE("mesh has not been set\n");
    return false;
  }

  bool ret = false;
  build_options.cache_bbox = false;

  LOGI("  BVH build option:\n");
  LOGI("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  LOGI("    SAH binsize         : %d\n", build_options.bin_size);

  timerutil t;
  t.start();

  triangle_mesh.reset(new nanort::TriangleMesh<float>(
      &flatten_vertices[0], &flatten_faces[0], sizeof(float) * 3));

  triangle_pred.reset(new nanort::TriangleSAHPred<float>(
      &flatten_vertices[0], &flatten_faces[0], sizeof(float) * 3));

  LOGI("num_triangles = %lu\n",
       static_cast<unsigned long>(mesh_->vertex_indices().size()));
  // LOGI("faces = %p\n", mesh_->vertex_indices().size());

  ret = accel.Build(static_cast<unsigned int>(mesh_->vertex_indices().size()),
                    *triangle_mesh, *triangle_pred, build_options);

  if (!ret) {
    LOGE("BVH building failed\n");
    return false;
  }

  t.end();
  LOGI("  BVH build time: %f secs\n", t.msec() / 1000.0);

  stats = accel.GetStatistics();

  LOGI("  BVH statistics:\n");
  LOGI("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  LOGI("    # of branch nodes: %d\n", stats.num_branch_nodes);
  LOGI("  Max tree depth     : %d\n", stats.max_tree_depth);

  accel.BoundingBox(bmin, bmax);
  LOGI("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  LOGI("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  mesh_initialized_ = true;

  return true;
}
void Renderer::set_camera(std::shared_ptr<Camera> camera) {
  camera_ = camera;
}
bool Renderer::render(Image3b& color, Image1w& depth, Image1b& mask) {
  if (camera_ == nullptr) {
    LOGE("camera has not been set\n");
    return false;
  }
  if (!mesh_initialized_) {
    LOGE("mesh has not been initialized\n");
    return false;
  }

  color.init(camera_->width(), camera_->height());
  depth.init(camera_->width(), camera_->height());
  mask.init(camera_->width(), camera_->height());

  const glm::vec3& t = camera_->c2w().t();
  const Pose& w2c = camera_->w2c();
  const std::vector<glm::vec3>& vertices = mesh_->vertices();
  const std::vector<glm::ivec3>& faces = mesh_->vertex_indices();
  const std::vector<glm::ivec3>& uv_indices = mesh_->uv_indices();
  const std::vector<glm::vec2>& uv = mesh_->uv();
  const std::vector<glm::vec3>& vertex_colors = mesh_->vertex_colors();
  const Image3b& diffuse_texture = mesh_->diffuse_tex();

  int width = camera_->width();
  int height = camera_->height();

  // set pixel shader
  auto pixel_shader = default_shader;
  if (option_.use_vertex_color && !vertex_colors.empty()) {
    pixel_shader = vertex_color_shader;
  } else if (!uv.empty()) {
    if (option_.interp == RendererOption::ColorInterpolation::NN) {
      pixel_shader = diffuse_nn_shader;
    } else if (option_.interp ==
               RendererOption::ColorInterpolation::BILINEAR) {
      pixel_shader = diffuse_bilinear_shader;
    } else {
      LOGE("Specified color interpolation is not implemented\n");
      return false;
    }
  } else {
    LOGW("No color on geometry\n");
  }

  timerutil time;
  time.start();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      nanort::Ray<float> ray;
      float kFar = 1.0e+30f;
      ray.min_t = 0.0001f;
      ray.max_t = kFar;

      // camera position in world coordinate
      ray.org[0] = t[0];
      ray.org[1] = t[1];
      ray.org[2] = t[2];

      // ray in world coordinate
      glm::vec3 dir;
      camera_->ray_w(static_cast<float>(x), static_cast<float>(y), dir);
      ray.dir[0] = dir[0];
      ray.dir[1] = dir[1];
      ray.dir[2] = dir[2];

      // shoot ray
      nanort::TriangleIntersector<> triangle_intersector(
          &flatten_vertices[0], &flatten_faces[0], sizeof(float) * 3);
      nanort::TriangleIntersection<> isect;
      bool hit = accel.Traverse(ray, triangle_intersector, &isect);

      if (!hit) {
        continue;
      }

      // back-face culling
      if (option_.backface_culling) {
        glm::vec3 vec1 = vertices[faces[isect.prim_id][1]] -
                         vertices[faces[isect.prim_id][0]];
        vec1 = glm::normalize(vec1);
        glm::vec3 vec2 = vertices[faces[isect.prim_id][2]] -
                         vertices[faces[isect.prim_id][0]];
        vec2 = glm::normalize(vec2);

        glm::vec3 face_normal = glm::cross(vec1, vec2);

        // back-face if face normal has same direction to ray
        if (glm::dot(face_normal, dir) > 0) {
          continue;
        }
      }

      // fill mask
      mask.at(x, y, 0) = 255;

      // convert hit position to camera coordinate to get depth value
      glm::vec3 hit_pos_w = t + dir * isect.t;
      glm::vec3 hit_pos_c = hit_pos_w;
      w2c.transform(hit_pos_c);
      assert(0.0f <= hit_pos_c[2]);  // depth should be positive
      depth.at(x, y, 0) =
          static_cast<unsigned short>(hit_pos_c[2] * option_.depth_scale);

      // delegate color calculation to pixel_shader
      pixel_shader(color, x, y, isect, faces, uv_indices, uv, vertex_colors,
                   diffuse_texture);
    }
  }

  time.end();
  LOGI("  Rendering main loop time: %f secs\n", time.msec() / 1000.0);

  return true;
}

}  // namespace crender
