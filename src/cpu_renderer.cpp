#include <array>
#include <cassert>
#include <iterator>

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

#include "cpu_renderer.h"

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

namespace unclearness {
CpuRenderer::CpuRenderer() {}
CpuRenderer::~CpuRenderer() {}
void CpuRenderer::set_mesh(std::shared_ptr<Mesh> mesh) {
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
bool CpuRenderer::prepare_mesh() {
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
void CpuRenderer::set_camera(std::shared_ptr<Camera> camera) {
  camera_ = camera;
}
bool CpuRenderer::render(Image3b& color, Image1w& depth, Image1b& mask) {
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
  const std::vector<glm::vec2>& uv = mesh_->uv();
  const Pose& w2c = camera_->w2c();
  const std::vector<glm::ivec3>& faces = mesh_->vertex_indices();
  const std::vector<glm::ivec3>& uv_indices = mesh_->uv_indices();
  const std::vector<glm::vec3>& vertex_colors = mesh_->vertex_colors();

  int width = camera_->width();
  int height = camera_->height();

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

      ray.org[0] = t[0];
      ray.org[1] = t[1];
      ray.org[2] = t[2];

      glm::vec3 dir;
      // ray direction is flipped for y axis
      camera_->ray_w(static_cast<float>(x),
                     static_cast<float>(y), dir);

      ray.dir[0] = dir[0];
      ray.dir[1] = dir[1];
      ray.dir[2] = dir[2];

      nanort::TriangleIntersector<> triangle_intersector(
          &flatten_vertices[0], &flatten_faces[0], sizeof(float) * 3);
      nanort::TriangleIntersection<> isect;
      bool hit = accel.Traverse(ray, triangle_intersector, &isect);

      if (!hit) {
        continue;
      }

      glm::vec3 hit_pos_w = t + dir * isect.t;
      glm::vec3 hit_pos_c = hit_pos_w;
      w2c.transform(hit_pos_c);
      assert(0.0f <= hit_pos_c[2]);  // depth should be positive
      mask.at(x, y, 0) = 255;
      depth.at(x, y, 0) =
          static_cast<unsigned short>(hit_pos_c[2] * option_.depth_scale);

      unsigned int fid = isect.prim_id;
      float u = isect.u;
      float v = isect.v;
      glm::vec3 interp_color;
      if (option_.use_vertex_color && !vertex_colors.empty()) {
        // barycentric interpolation of vertex color
        interp_color = (1.0f - u - v) * vertex_colors[faces[fid][0]] +
                       u * vertex_colors[faces[fid][1]] +
                       v * vertex_colors[faces[fid][2]];
      } else if (!uv.empty()) {
        // barycentric interpolation of uv
        glm::vec2 interp_uv = (1.0f - u - v) * uv[uv_indices[fid][0]] +
                              u * uv[uv_indices[fid][1]] +
                              v * uv[uv_indices[fid][2]];
        float f_tex_pos[2];
        f_tex_pos[0] = interp_uv[0] * (mesh_->diffuse_tex().width() - 1);
        f_tex_pos[1] =
            (1.0f - interp_uv[1]) * (mesh_->diffuse_tex().height() - 1);

        if (option_.interp == CpuRendererOption::ColorInterpolation::NN) {
          int tex_pos[2] = {0, 0};
          tex_pos[0] = static_cast<int>(std::round(f_tex_pos[0]));
          tex_pos[1] = static_cast<int>(std::round(f_tex_pos[1]));
          for (int k = 0; k < 3; k++) {
            interp_color[k] =
                mesh_->diffuse_tex().at(tex_pos[0], tex_pos[1], k);
          }

        } else if (option_.interp ==
                   CpuRendererOption::ColorInterpolation::BILINEAR) {
          int tex_pos_min[2] = {0, 0};
          int tex_pos_max[2] = {0, 0};
          tex_pos_min[0] = static_cast<int>(std::floor(f_tex_pos[0]));
          tex_pos_min[1] = static_cast<int>(std::floor(f_tex_pos[1]));
          tex_pos_max[0] = tex_pos_min[0] + 1;
          tex_pos_max[1] = tex_pos_min[1] + 1;

          float local_u = f_tex_pos[0] - tex_pos_min[0];
          float local_v = f_tex_pos[1] - tex_pos_min[1];

          for (int k = 0; k < 3; k++) {
            // bilinear interpolation of pixel color
            interp_color[k] =
                (1.0f - local_u) * (1.0f - local_v) *
                    mesh_->diffuse_tex().at(tex_pos_min[0], tex_pos_min[1], k) +
                local_u * (1.0f - local_v) *
                    mesh_->diffuse_tex().at(tex_pos_max[0], tex_pos_min[1], k) +
                (1.0f - local_u) * local_v *
                    mesh_->diffuse_tex().at(tex_pos_min[0], tex_pos_max[1], k) +
                local_u * local_v *
                    mesh_->diffuse_tex().at(tex_pos_max[0], tex_pos_max[1], k);

            assert(0.0f <= interp_color[k] && interp_color[k] <= 255.0f);
          }

        } else {
          LOGE("Specified color interpolation is not implemented\n");
          break;
        }
        for (int k = 0; k < 3; k++) {
          color.at(x, y, k) = static_cast<unsigned char>(interp_color[k]);
        }

      } else {
        // no color on gemetry
        color.at(x, y, 1) = static_cast<unsigned char>(255);
      }
    }
  }

  time.end();
  LOGI("  Rendering main loop time: %f secs\n", time.msec() / 1000.0);

  return true;
}

}  // namespace unclearness