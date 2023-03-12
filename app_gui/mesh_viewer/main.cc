// Dear ImGui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline (GLFW is a cross-platform general purpose library for
// handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation,
// etc.) If you are new to Dear ImGui, read documentation from the docs/ folder
// + read the top of imgui.cpp. Read online:
// https://github.com/ocornut/imgui/tree/master/docs

#include <limits>
#include <mutex>

#include "glad/gl.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ugu/camera.h"
#include "ugu/image_io.h"
#include "ugu/renderable_mesh.h"
#include "ugu/renderer/gl/renderer.h"
#include "ugu/util/image_util.h"
#include "ugu/util/string_util.h"
// #define GL_SILENCE_DEPRECATION
// #if defined(IMGUI_IMPL_OPENGL_ES2)
// #include <GLES2/gl2.h>
// #endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to
// maximize ease of testing and compatibility with old VS compilers. To link
// with VS2010-era libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma. Your own project
// should not be affected, as you are likely to link with a newer binary of GLFW
// that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && \
    !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#if 0
#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <windows.h>  // so APIENTRY gets defined and GLFW doesn't define it
#undef far
#undef near
#endif
#endif

using namespace ugu;

namespace {

Eigen::Vector2d g_prev_cursor_pos;
Eigen::Vector2d g_cursor_pos;
Eigen::Vector2d g_mouse_l_pressed_pos;
Eigen::Vector2d g_mouse_l_released_pos;
Eigen::Vector2d g_mouse_m_pressed_pos;
Eigen::Vector2d g_mouse_m_released_pos;
Eigen::Vector2d g_mouse_r_pressed_pos;
Eigen::Vector2d g_mouse_r_released_pos;

bool g_to_process_drag_l = false;
bool g_to_process_drag_m = false;
bool g_to_process_drag_r = false;

uint32_t g_subwindow_id = ~0u;
uint32_t g_prev_subwindow_id = ~0u;

bool g_mouse_l_pressed = false;
bool g_mouse_m_pressed = false;
bool g_mouse_r_pressed = false;
const double drag_th = 0.0;
const double g_drag_point_pix_dist_th = 20.0;
uint32_t g_selecting_point_id = ~0u;

double g_mouse_wheel_yoffset = 0.0;
bool g_to_process_wheel = false;

const uint32_t MAX_N_SPLIT_WIDTH = 2;
// uint32_t g_n_split_views = 2;

int g_width = 1280;
int g_height = 720;

struct SplitViewInfo;
std::mutex views_mtx;
std::vector<SplitViewInfo> g_views;

std::mutex mouse_mtx;

Eigen::Vector3f default_clear_color = {0.45f, 0.55f, 0.60f};
Eigen::Vector3f default_wire_color = {0.1f, 0.1f, 0.1f};

void Draw(GLFWwindow *window);

auto GetWidthHeightForView() {
  return std::make_pair(g_width / g_views.size(), g_height);
}

bool IsCursorOnView(uint32_t vidx) {
  uint32_t x = static_cast<uint32_t>(g_cursor_pos.x());
  uint32_t unit_w = g_width / g_views.size();

  if (unit_w * vidx <= x && x < unit_w * (vidx + 1)) {
    return true;
  }

  return false;
}

// Global geometry info
std::vector<RenderableMeshPtr> g_meshes;
std::vector<std::string> g_mesh_names;
std::vector<std::string> g_mesh_paths;
std::vector<BvhPtr<Eigen::Vector3f, Eigen::Vector3i>> g_bvhs;

std::unordered_map<RenderableMeshPtr, std::vector<Eigen::Vector3f>>
    g_selected_positions;

struct CastRayResult {
  size_t min_geoid = ~0u;
  float min_geo_dist = std::numeric_limits<float>::max();
  Eigen::Vector3f min_geo_dist_pos =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
};

struct SplitViewInfo {
  RendererGlPtr renderer;
  PinholeCameraPtr camera;
  Eigen::Vector2i offset = {0, 0};

  void Init(uint32_t vidx) {
    std::lock_guard<std::mutex> lock(views_mtx);

    auto [w, h] = GetWidthHeightForView();

    offset.x() = vidx * w;
    offset.y() = 0;

    camera = std::make_shared<PinholeCamera>(w, h, 45.f);
    renderer = std::make_shared<RendererGl>();
    renderer->SetSize(w, h);
    renderer->SetCamera(camera);
    renderer->Init();

    renderer->SetBackgroundColor(default_clear_color);
    renderer->SetWireColor(default_wire_color);
  }

  void Reset() {
    std::lock_guard<std::mutex> lock(views_mtx);

    auto [w, h] = GetWidthHeightForView();

    camera->set_size(w, h);
    camera->set_fov_y(45.0f);
    camera->set_principal_point({w / 2.f, h / 2.f});

    renderer->SetSize(w, h);

    // renderer->SetSize(w, h);

    ResetGl();
  }

  void ResetGl() {
    renderer->ClearGlState();
    for (const auto &mesh : g_meshes) {
      renderer->SetMesh(mesh);
      renderer->AddSelectedPositions(mesh, g_selected_positions[mesh]);
    }

    renderer->Init();
  }

  CastRayResult CastRay() {
    // Cast ray
    Eigen::Vector3f dir_c_cv;
    // Substract offset to align cursor pos on window to corresponding positon
    // on framebuffer
    camera->ray_c(static_cast<float>(g_cursor_pos[0] - offset.x()),
                  static_cast<float>(g_cursor_pos[1] - offset.y()), &dir_c_cv);

    const Eigen::Affine3d offset =
        Eigen::Affine3d(Eigen::AngleAxisd(pi, Eigen::Vector3d::UnitX()))
            .inverse();
    Eigen::Vector3f dir_c_gl =
        (dir_c_cv.transpose() * offset.rotation().cast<float>());
    Eigen::Vector3f dir_w_gl =
        camera->c2w().rotation().cast<float>() * dir_c_gl;

    size_t min_geoid = ~0u;
    float min_geo_dist = std::numeric_limits<float>::max();
    Eigen::Vector3f min_geo_dist_pos =
        Eigen::Vector3f::Constant(std::numeric_limits<float>::max());

    Ray ray;
    ray.dir = dir_w_gl;
    ray.org = camera->c2w().translation().cast<float>();
    auto results_all = renderer->Intersect(ray);

    for (size_t geoid = 0; geoid < g_meshes.size(); geoid++) {
      if (!renderer->GetVisibility(g_meshes[geoid])) {
        continue;
      }
      const std::vector<IntersectResult> &results = results_all[geoid];
      if (!results.empty()) {
        // std::cout << geoid << ": " << results[0].t << " " << results[0].fid
        //           << " " << results[0].u << ", " << results[0].v <<
        //           std::endl;
        if (results[0].t < min_geo_dist) {
          min_geoid = geoid;
          min_geo_dist = results[0].t;
          min_geo_dist_pos = results[0].t * ray.dir + ray.org;
        }
      }
    }

    if (min_geoid != ~0u) {
      // std::cout << "closest geo: " << min_geoid << std::endl;
    }

    CastRayResult result;
    result.min_geoid = min_geoid;
    result.min_geo_dist = min_geo_dist;
    result.min_geo_dist_pos = min_geo_dist_pos;

    return result;
  }

  auto FindClosestSelectedPoint(const Eigen::Vector2d &cursor_pos) {
    bool not_close = true;
    size_t closest_selected_id = ~0u;
    double min_dist = std::numeric_limits<double>::max();
    // RenderableMeshPtr closest_mesh = nullptr;
    float near_z, far_z;
    renderer->GetNearFar(near_z, far_z);
    Eigen::Matrix4f view_mat = camera->c2w().inverse().matrix().cast<float>();
    Eigen::Matrix4f prj_mat = camera->ProjectionMatrixOpenGl(near_z, far_z);
    for (const auto &mesh : g_meshes) {
      if (g_selected_positions.find(mesh) == g_selected_positions.end()) {
        continue;
      }
      if (!renderer->GetVisibility(mesh)) {
        continue;
      }
      for (size_t i = 0; i < g_selected_positions[mesh].size(); i++) {
        const auto &p_wld = g_selected_positions[mesh][i];
        auto [is_visible, results_all] = renderer->TestVisibility(p_wld);
        if (is_visible) {
          // std::cout << "selected " << i << ": visibile" << std::endl;
        } else {
          //  std::cout << "selected " << i << ": not visibile" << std::endl;
        }
        if (is_visible) {
          Eigen::Vector4f p_cam =
              view_mat * Eigen::Vector4f(p_wld.x(), p_wld.y(), p_wld.z(), 1.f);
          Eigen::Vector4f p_ndc = prj_mat * p_cam;
          p_ndc /= p_ndc.w();  // NDC [-1:1]

          // [-1:1],[-1:1] -> [0:w], [0:h]
          Eigen::Vector2d p_gl_frag =
              Eigen::Vector2d(((p_ndc.x() + 1.f) / 2.f) * camera->width(),
                              ((p_ndc.y() + 1.f) / 2.f) * camera->height());

          p_gl_frag.y() = camera->height() - p_gl_frag.y();

          // Add offset
          p_gl_frag += offset.cast<double>();

          double dist = (p_gl_frag - cursor_pos).norm();
          std::cout << i << " " << dist << std::endl;
          if (dist < g_drag_point_pix_dist_th && dist < min_dist) {
            not_close = false;
            min_dist = dist;
            closest_selected_id = i;
            // closest_mesh = mesh;
            //  break;
          }
        }
      }
    }

    return std::make_tuple(!not_close, closest_selected_id, min_dist);
  }
};

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}
#if 0
static void mouse_callback(GLFWwindow *window, int button, int action,
                           int mods) {
  bool lbutton_down = false;
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (GLFW_PRESS == action)
      lbutton_down = true;
    else if (GLFW_RELEASE == action)
      lbutton_down = false;
  }

  if (lbutton_down) {
    // do your drag here
    std::cout << "press" << std::endl;
  } else {
    std::cout << "release" << std::endl;
  }
}
#endif

void Clear() {
  g_meshes.clear();
  g_mesh_names.clear();
  g_mesh_paths.clear();
  g_selected_positions.clear();
  for (auto &view : g_views) {
    view.ResetGl();
  }
}

void key_callback(GLFWwindow *pwin, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_UP && action == GLFW_PRESS) {
    printf("key up\n");
  }
  if (key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
    printf("key down\n");
  }
  if (key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
    printf("key left\n");
  }
  if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
    printf("key right\n");
  }

  if (key >= GLFW_KEY_A && key <= GLFW_KEY_Z) {
    if (action == GLFW_PRESS) {
      const char *key_name = glfwGetKeyName(key, 0);
      printf("key - %s\n", key_name);
    }
  }
  if (key == GLFW_KEY_R) {
    if (action == GLFW_PRESS) {
      Clear();
    }
  }
}

void mouse_button_callback(GLFWwindow *pwin, int button, int action, int mods) {
  // std::lock_guard<std::mutex> lock(mouse_mtx);

  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    // printf("L - down\n");
    // g_prev_mouse_l_pressed = g_mouse_l_pressed;
    g_mouse_l_pressed = action == GLFW_PRESS;
    if (g_mouse_l_pressed) {
      g_mouse_l_pressed_pos = g_cursor_pos;
    } else {
      g_mouse_l_released_pos = g_cursor_pos;

      if ((g_mouse_l_pressed_pos - g_mouse_l_released_pos).norm() > drag_th) {
        // std::cout << "drag finish " << g_mouse_l_pressed_pos << " -> "
        //           << g_mouse_l_released_pos << std::endl;
      }
    }
  }

  if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    g_mouse_r_pressed = action == GLFW_PRESS;
    if (g_mouse_r_pressed) {
      g_mouse_r_pressed_pos = g_cursor_pos;
    } else {
      g_mouse_r_released_pos = g_cursor_pos;
    }

    if (g_mouse_r_pressed) {
      bool not_close = false;
      CastRayResult result;
      double min_dist;
      // for (size_t vidx = 0; vidx < g_views.size(); vidx++) {
      auto &view = g_views[g_subwindow_id];
      result = view.CastRay();
      if (result.min_geoid != ~0u) {
        auto [is_close, id, min_dist_] =
            view.FindClosestSelectedPoint(g_mouse_r_pressed_pos);
        not_close = !is_close;
        // break;
        //}
        min_dist = min_dist_;
      }

      if (not_close) {
        g_selected_positions[g_meshes[result.min_geoid]].push_back(
            result.min_geo_dist_pos);
        for (size_t vidx = 0; vidx < g_views.size(); vidx++) {
          auto &view = g_views[vidx];
          view.renderer->AddSelectedPositions(
              g_meshes[result.min_geoid],
              g_selected_positions[g_meshes[result.min_geoid]]);
        }

        std::cout << "added " << min_dist << std::endl;
      } else {
        std::cout << "ignored " << min_dist << std::endl;
      }
    }
  }

  if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    // g_prev_mouse_m_pressed = g_mouse_m_pressed;
    g_mouse_m_pressed = action == GLFW_PRESS;
    if (g_mouse_m_pressed) {
      g_mouse_m_pressed_pos = g_cursor_pos;
    } else {
      g_mouse_m_released_pos = g_cursor_pos;

      if ((g_mouse_m_pressed_pos - g_mouse_m_released_pos).norm() > drag_th) {
        // std::cout << "drag finish " << g_mouse_m_pressed_pos << " -> "
        //           << g_mouse_m_released_pos << std::endl;
      }
    }
  }
}

void mouse_wheel_callback(GLFWwindow *window, double xoffset, double yoffset) {
  if (yoffset < 0) {
    // printf("wheel down \n");
  }
  if (yoffset > 0) {
    // printf("wheel up \n");
  }

  g_mouse_wheel_yoffset = yoffset;
  g_to_process_wheel = true;
}

void cursor_pos_callback(GLFWwindow *window, double xoffset, double yoffset) {
  // std::cout << "pos: " << xoffset << ", " << yoffset << std::endl;
  g_prev_cursor_pos = g_cursor_pos;

  g_cursor_pos[0] = xoffset;
  g_cursor_pos[1] = yoffset;
  if (g_mouse_l_pressed) {
    g_to_process_drag_l = true;
  }
  if (g_mouse_m_pressed) {
    g_to_process_drag_m = true;
  }
  if (g_mouse_r_pressed) {
    g_to_process_drag_r = true;
  }

  g_prev_subwindow_id = g_subwindow_id;
  for (uint32_t vidx = 0; vidx < static_cast<uint32_t>(g_views.size());
       vidx++) {
    if (IsCursorOnView(vidx)) {
      g_subwindow_id = vidx;
      break;
    }
  }
}

void LoadMesh(const std::string &path) {
  auto ext = ugu::ExtractExt(path);
  auto mesh = ugu::RenderableMesh::Create();
  if (ext == "obj") {
    std::string obj_path = path;
    std::string obj_dir = ExtractDir(obj_path);
    if (!mesh->LoadObj(obj_path, obj_dir)) {
      return;
    }
    g_mesh_names.push_back(ugu::ExtractFilename(obj_path, true));
    g_mesh_paths.push_back(obj_path);
    g_meshes.push_back(mesh);

  } else {
    return;
  }

  for (auto &view : g_views) {
    view.ResetGl();

    Eigen::Vector3f bb_max, bb_min;
    view.renderer->GetMergedBoundingBox(bb_max, bb_min);
    float z_trans = (bb_max - bb_min).maxCoeff() * 2.0f;
    view.renderer->SetNearFar(static_cast<float>(z_trans * 0.5f / 10),
                              static_cast<float>(z_trans * 2.f * 10));

    Eigen::Affine3d c2w = Eigen::Affine3d::Identity();
    c2w.translation() = Eigen::Vector3d(0, 0, z_trans);
    view.camera->set_c2w(c2w);
  }
}

void drop_callback(GLFWwindow *window, int count, const char **paths) {
  for (int i = 0; i < count; i++) {
    std::cout << "Dropped: " << i << "/" << count << " " << paths[i]
              << std::endl;
  }
  LoadMesh(paths[0]);
}

void window_size_callback(GLFWwindow *window, int width, int height) {
  // std::cout << width << " " << height << std::endl;

  if (width < 1 && height < 1) {
    return;
  }

  g_width = width;
  g_height = height;

  for (auto &view : g_views) {
    view.Reset();
  }

  // Draw(window);

  // glViewport(0, 0, width, height);
  // std::cout << "window size " << width << " " << height << std::endl;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
// glViewport(0, 0, width, height)
#if 0
  g_width = width;
  g_height = height;

  g_camera->set_size(g_width, g_height);
  g_camera->set_fov_y(45.0f);
  g_camera->set_principal_point({g_width / 2.f, g_height / 2.f});

  g_renderer->SetSize(g_width, g_height);
#endif
  // ResetGl();

  // glViewport(0, 0, width/2, height);

  Draw(window);

  // std::cout << "frame buffer size " << width << " " << height << std::endl;
}

void cursor_enter_callback(GLFWwindow *window, int entered) {
  g_to_process_drag_l = false;
  g_to_process_drag_r = false;
  g_to_process_drag_m = false;

  g_subwindow_id = ~0u;
}

void SetupWindow(GLFWwindow *window) {
  if (window == NULL) return;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  glfwSetCursorPosCallback(window, cursor_pos_callback);

  glfwSetKeyCallback(window, key_callback);

  glfwSetMouseButtonCallback(window, mouse_button_callback);

  glfwSetScrollCallback(window, mouse_wheel_callback);

  glfwSetDropCallback(window, drop_callback);

  glfwSetCursorEnterCallback(window, cursor_enter_callback);

  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
}

void DrawViews() {
  glViewport(0, 0, g_width, g_height);
  for (size_t i = 0; i < g_views.size(); i++) {
    const auto &view = g_views[i];
    const auto offset_w = g_width / g_views.size();
    view.renderer->SetViewport(offset_w * i, 0, offset_w, g_height);
    view.renderer->Draw();
  }
}

void ProcessDrags() {
  // std::lock_guard<std::mutex> lock(mouse_mtx);

  if (!ImGui::IsAnyItemActive()) {
    Eigen::Vector3f bb_max, bb_min;

    if (g_subwindow_id != ~0u && g_subwindow_id == g_prev_subwindow_id) {
      uint32_t vidx = g_subwindow_id;
      auto &view = g_views[vidx];

      view.renderer->GetMergedBoundingBox(bb_max, bb_min);

      if (g_to_process_drag_l) {
        g_to_process_drag_l = false;
        Eigen::Vector2d diff = g_cursor_pos - g_prev_cursor_pos;
        if (diff.norm() > drag_th) {
          const double rotate_speed = ugu::pi / 180 * 10;

          Eigen::Affine3d cam_pose_cur = view.camera->c2w();
          Eigen::Matrix3d R_cur = cam_pose_cur.rotation();

          Eigen::Vector3d right_axis = -R_cur.col(0);
          Eigen::Vector3d up_axis = -R_cur.col(1);

          Eigen::Quaterniond R_offset =
              Eigen::AngleAxisd(2 * ugu::pi * diff[0] / g_height * rotate_speed,
                                up_axis) *
              Eigen::AngleAxisd(2 * ugu::pi * diff[1] / g_height * rotate_speed,
                                right_axis);

          Eigen::Affine3d cam_pose_new = R_offset * cam_pose_cur;

          view.camera->set_c2w(cam_pose_new);
        }

        //  g_prev_to_process_drag_l_id = vidx;
      }

      if (g_to_process_drag_m) {
        g_to_process_drag_m = false;
        Eigen::Vector2d diff = g_cursor_pos - g_prev_cursor_pos;
        if (diff.norm() > drag_th) {
          const double trans_speed = (bb_max - bb_min).maxCoeff() / g_height;

          Eigen::Affine3d cam_pose_cur = view.camera->c2w();
          Eigen::Matrix3d R_cur = cam_pose_cur.rotation();

          Eigen::Vector3d right_axis = -R_cur.col(0);
          Eigen::Vector3d up_axis = R_cur.col(1);

          Eigen::Vector3d t_offset = right_axis * diff[0] * trans_speed +
                                     up_axis * diff[1] * trans_speed;

          Eigen::Affine3d cam_pose_new =
              Eigen::Translation3d(t_offset + cam_pose_cur.translation()) *
              cam_pose_cur.rotation();
          view.camera->set_c2w(cam_pose_new);
        }
      }

      if (g_to_process_drag_r) {
        g_to_process_drag_r = false;
        CastRayResult result;
        bool is_close = false;
        double min_dist;
        size_t id;
        result = view.CastRay();
        if (result.min_geoid != ~0u) {
          std::tie(is_close, id, min_dist) =
              view.FindClosestSelectedPoint(g_cursor_pos);

          if (is_close) {
            g_selected_positions[g_meshes[result.min_geoid]][id] =
                result.min_geo_dist_pos;
            for (size_t vidx = 0; vidx < g_views.size(); vidx++) {
              auto &view = g_views[vidx];
              view.renderer->AddSelectedPositions(
                  g_meshes[result.min_geoid],
                  g_selected_positions[g_meshes[result.min_geoid]]);
            }
          } else {
            std::cout << "Failed " << min_dist << std::endl;
          }
        }
      }
      if (g_to_process_wheel) {
        g_to_process_wheel = false;
        const double wheel_speed = (bb_max - bb_min).maxCoeff() / 20;

        Eigen::Affine3d cam_pose_cur = view.camera->c2w();
        Eigen::Vector3d t_offset = cam_pose_cur.rotation().col(2) *
                                   -g_mouse_wheel_yoffset * wheel_speed;
        Eigen::Affine3d cam_pose_new =
            Eigen::Translation3d(t_offset + cam_pose_cur.translation()) *
            cam_pose_cur.rotation();
        view.camera->set_c2w(cam_pose_new);
      }
    }
  }

  // Get visibile selected points
  {
    for (uint32_t vidx = 0; vidx < static_cast<uint32_t>(g_views.size());
         vidx++) {
      auto &view = g_views[vidx];
      const auto &camera = view.renderer->GetCamera();
      Eigen::Matrix4f view_mat = camera->c2w().inverse().matrix().cast<float>();
      float near_z, far_z;
      view.renderer->GetNearFar(near_z, far_z);
      Eigen::Matrix4f prj_mat = camera->ProjectionMatrixOpenGl(near_z, far_z);

      std::vector<TextRendererGl::Text> texts;
      for (const auto &geo : g_meshes) {
        if (!view.renderer->GetVisibility(geo)) {
          continue;
        }

        for (size_t i = 0; i < g_selected_positions[geo].size(); i++) {
          const auto &p = g_selected_positions[geo][i];
          auto [is_visibile, results] = view.renderer->TestVisibility(p);
          if (!is_visibile) {
            continue;
          }

          Eigen::Vector4f p_cam =
              view_mat * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.f);
          Eigen::Vector4f p_ndc = prj_mat * p_cam;
          p_ndc /= p_ndc.w();  // NDC [-1:1]

          // [-1:1],[-1:1] -> [0:w], [0:h]
          TextRendererGl::Text text;
          text.body = std::to_string(i);
          text.x = ((p_ndc.x() + 1.f) / 2.f) * camera->width();
          text.y = ((p_ndc.y() + 1.f) / 2.f) * camera->height();
          text.y = camera->height() - text.y;
          text.scale = 1.f;
          text.color = Eigen::Vector3f(0.f, 0.f, 0.f);
          texts.push_back(text);
        }
      }
      view.renderer->SetTexts(texts);
    }
  }
}

void DrawImgui(GLFWwindow *window) {
  std::lock_guard<std::mutex> lock(views_mtx);
  // 1. Show the big demo window (Most of the sample code is in
  // ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear
  // ImGui!).
  // if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

#if 0
  // 2. Show a simple window that we create ourselves. We use a Begin/End pair
  // to create a named window.
  {
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Hello, world!");  // Create a window called "Hello, world!"
                                    // and append into it.

    ImGui::Text("This is some useful text.");  // Display some text (you can
                                               // use a format strings too)
    // ImGui::Checkbox(
    //     "Demo Window",
    //     &show_demo_window);  // Edit bools storing our window open/close
    //     state
    // ImGui::Checkbox("Another Window", &show_another_window);

    ImGui::SliderFloat("float", &f, 0.0f,
                       1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
    ImGui::ColorEdit3("clear color",
                      (float *)&g_views[0]
                          .clear_color);  // Edit 3 floats representing a color

    if (ImGui::Button("Button"))  // Buttons return true when clicked (most
                                  // widgets return true when edited/activated)
      counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();
  }
#endif
  // 3. Show another simple window.
  // if (show_another_window) {
  //  ImGui::Begin(
  //      "Another Window",
  //      &show_another_window);  // Pass a pointer to our bool variable (the
  //                              // window will have a closing button that will
  //                              // clear the bool when clicked)
  //  ImGui::Text("Hello from another window!");
  //  if (ImGui::Button("Close Me")) show_another_window = false;
  //  ImGui::End();
  //}

  for (size_t i = 0; i < g_views.size(); i++) {
    auto &view = g_views[i];
    const std::string title = std::string("View ") + std::to_string(i);

    ImGui::Begin(title.c_str());

    if (ImGui::TreeNodeEx("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
      for (size_t i = 0; i < g_meshes.size(); i++) {
        bool v = view.renderer->GetVisibility(g_meshes[i]);
        std::string label =
            std::to_string(i) + " " + g_mesh_names[i] + ": " + g_mesh_paths[i];
        if (ImGui::Checkbox(label.c_str(), &v)) {
          view.renderer->SetVisibility(g_meshes[i], v);
        }
        Eigen::Vector3f pos_col =
            view.renderer->GetSelectedPositionColor(g_meshes[i]);
        if (ImGui::ColorEdit3((label + "select color").c_str(), pos_col.data(),
                              ImGuiColorEditFlags_NoLabel)) {
          view.renderer->AddSelectedPositionColor(g_meshes[i], pos_col);
        }
      }

      ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
      bool update_pose = false;
      Eigen::Affine3f c2w_gl = view.camera->c2w().cast<float>();
      Eigen::Vector3f pos = c2w_gl.translation();
      Eigen::Matrix3f R_gl = c2w_gl.rotation();
      Eigen::Matrix3f R_cv = R_gl;
      R_cv.col(1) *= -1.f;
      R_cv.col(2) *= -1.f;

      Eigen::Vector2f nearfar;
      view.renderer->GetNearFar(nearfar[0], nearfar[1]);
      if (ImGui::InputFloat2("near far", nearfar.data())) {
        view.renderer->SetNearFar(nearfar[0], nearfar[1]);
      }

      Eigen::Vector2i size(view.camera->width(), view.camera->height());
      if (ImGui::InputInt2("width height", size.data())) {
        // view.camera->set_size(size[0], size[1]);
        // TODO: Window resize
      }

      Eigen::Vector2f fov(view.camera->fov_x(), view.camera->fov_y());
      if (ImGui::InputFloat2("FoV-X FoV-Y", nearfar.data())) {
        view.camera->set_fov_x(fov[0]);
        view.camera->set_fov_y(fov[1]);
      }

      if (ImGui::TreeNodeEx("OpenCV Style", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::InputFloat3("Position", pos.data())) {
          update_pose = true;
        }

        if (ImGui::InputFloat3("Rotation", R_cv.data()) ||
            ImGui::InputFloat3("        ", R_cv.data() + 3) ||
            ImGui::InputFloat3("        ", R_cv.data() + 6)) {
          update_pose = true;
          R_gl = R_cv;
          R_gl.col(1) *= -1.f;
          R_gl.col(2) *= -1.f;
        }

        Eigen::Vector2f fxfy = view.camera->focal_length();
        if (ImGui::InputFloat2("fx fy", fxfy.data())) {
          view.camera->set_focal_length(fxfy);
        }

        Eigen::Vector2f cxcy = view.camera->principal_point();
        if (ImGui::InputFloat2("cx cy", cxcy.data())) {
          view.camera->set_principal_point(cxcy);
        }

        Eigen::Vector4f distortion(0.f, 0.f, 0.f, 0.f);
        if (ImGui::InputFloat4("k1 k2 p1 p2 (TODO)", distortion.data())) {
          // TODO
        }

        ImGui::TreePop();
      }

      if (ImGui::TreeNodeEx("OpenGL Style", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::InputFloat3("Position", pos.data())) {
          update_pose = true;
        }

        if (ImGui::InputFloat3("Rotation", R_gl.data()) ||
            ImGui::InputFloat3("        ", R_gl.data() + 3) ||
            ImGui::InputFloat3("        ", R_gl.data() + 6)) {
          update_pose = true;
        }

        Eigen::Matrix4f prj_gl =
            view.camera->ProjectionMatrixOpenGl(nearfar[0], nearfar[1]);
        if (ImGui::InputFloat4("Projection", prj_gl.data()) ||
            ImGui::InputFloat4("        ", prj_gl.data() + 4) ||
            ImGui::InputFloat4("        ", prj_gl.data() + 8) ||
            ImGui::InputFloat4("        ", prj_gl.data() + 12)) {
          ugu::LOGI("No update for GL projection matrix\n");
        }

        if (update_pose) {
          c2w_gl = R_gl * Eigen::Translation3f(pos);
          view.camera->set_c2w(c2w_gl.cast<double>());
        }

        ImGui::TreePop();
      }

      bool show_wire = view.renderer->GetShowWire();
      if (ImGui::Checkbox("show wire", &show_wire)) {
        view.renderer->SetShowWire(show_wire);
      }

      Eigen::Vector3f wire_col = view.renderer->GetWireColor();
      if (ImGui::ColorEdit3("wire color", wire_col.data())) {
        view.renderer->SetWireColor(wire_col);
      }

      Eigen::Vector3f bkg_col = view.renderer->GetBackgroundColor();
      if (ImGui::ColorEdit3("background color", bkg_col.data())) {
        view.renderer->SetBackgroundColor(bkg_col);
      }

      static int save_counter = 0;
      static char gbuf_save_path[1024] = "./";
      ImGui::InputText("GBuffer Save Dir.", gbuf_save_path, 1024u);
      ImGui::Text("Prefix %d", save_counter);
      ImGui::SameLine();
      if (ImGui::Button("Save")) {
        std::string prefix = std::to_string(save_counter) + "_";

        view.renderer->ReadGbuf();
        GBuffer gbuf;
        view.renderer->GetGbuf(gbuf);

        imwrite(prefix + "pos_wld.bin", gbuf.pos_wld);
        imwrite(prefix + "pos_cam.bin", gbuf.pos_cam);
        Image3b vis_pos_wld, vis_pos_cam;
        vis_pos_wld = ColorizePosMap(gbuf.pos_wld);
        imwrite(prefix + "pos_wld.jpg", vis_pos_wld);
        vis_pos_cam = ColorizePosMap(gbuf.pos_cam);
        imwrite(prefix + "pos_cam.jpg", vis_pos_cam);

        imwrite(prefix + "normal_wld.bin", gbuf.normal_wld);
        imwrite(prefix + "normal_cam.bin", gbuf.normal_cam);
        Image3b vis_normal_wld, vis_normal_cam;
        Normal2Color(gbuf.normal_wld, &vis_normal_wld, true);
        imwrite(prefix + "normal_wld.jpg", vis_normal_wld);
        Normal2Color(gbuf.normal_cam, &vis_normal_cam, true);
        imwrite(prefix + "normal_cam.jpg", vis_normal_cam);

        imwrite(prefix + "depth01.bin", gbuf.depth_01);
        Image3b vis_depth;
        Depth2Color(gbuf.depth_01, &vis_depth, 0.f, 1.f);
        imwrite(prefix + "depth01.jpg", vis_depth);

        Image1b geoid_1b;
        gbuf.geo_id.convertTo(geoid_1b, CV_8UC1);
        imwrite(prefix + "geoid.png", geoid_1b);
        Image3b vis_geoid;
        FaceId2RandomColor(gbuf.geo_id, &vis_geoid);
        imwrite(prefix + "geoid.jpg", vis_geoid);

        imwrite(prefix + "faceid.bin", gbuf.face_id);
        Image3b vis_faceid;
        FaceId2RandomColor(gbuf.face_id, &vis_faceid);
        imwrite(prefix + "faceid.jpg", vis_faceid);

        imwrite(prefix + "bary.bin", gbuf.bary);
        Image3b vis_bary = ColorizeBarycentric(gbuf.bary);
        imwrite(prefix + "bary.jpg", vis_bary);

        imwrite(prefix + "uv.bin", gbuf.uv);
        Image3b vis_uv = ColorizeBarycentric(gbuf.uv);
        imwrite(prefix + "uv.jpg", vis_uv);

        imwrite(prefix + "color.png", gbuf.color);

        save_counter++;
      }

      ImGui::TreePop();
    }

    ImGui::End();
  }

  {
    ImGui::Begin("General");

    static char mesh_path[1024] = "../data/bunny/bunny.obj";
    ImGui::InputText("Mesh path", mesh_path, 1024u);
    if (ImGui::Button("Load mesh")) {
      LoadMesh(mesh_path);
    }

    ImGui::End();
  }

  ImGui::Render();
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Draw(GLFWwindow *window) {
  glClear(GL_COLOR_BUFFER_BIT);
  // glClearColor(g_views[0].clear_color.x(), g_views[0].clear_color.y(),
  //              g_views[0].clear_color.z(), 1.f);

  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  DrawViews();

  ProcessDrags();

  glClear(GL_DEPTH_BUFFER_BIT);

  DrawImgui(window);

  glfwSwapBuffers(window);
}

}  // namespace

int main(int, char **) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char *glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else

#if 0
  const char *glsl_version = "#version 450";

  // Upgrade WSL's OpenGL to 4.5
  // https://zenn.dev/suudai/articles/a25e3ed0a944c7

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else

  const char *glsl_version = "#version 330";

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

#endif

  // Create window with graphics context
  GLFWwindow *window =
      glfwCreateWindow(g_width, g_height, "UGU Mesh Viewer", NULL, NULL);

  SetupWindow(window);

  // GLFWwindow *window2 =
  //     glfwCreateWindow(g_width, g_height, "UGU Mesh Viewer2", NULL, window);
  //  SetupWindow(window2);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable
  // Keyboard Controls io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; //
  // Enable Gamepad Controls

  const int version = gladLoadGL(glfwGetProcAddress);
  if (version == 0) {
    fprintf(stderr, "Failed to load OpenGL 3.x/4.x libraries!\n");
    return 1;
  }

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  // ImGui_ImplGlfw_InitForOpenGL(window2, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Load Fonts
  // - If no fonts are loaded, dear imgui will use the default font. You can
  // also load multiple fonts and use ImGui::PushFont()/PopFont() to select
  // them.
  // - AddFontFromFileTTF() will return the ImFont* so you can store it if you
  // need to select the font among multiple.
  // - If the file cannot be loaded, the function will return NULL. Please
  // handle those errors in your application (e.g. use an assertion, or display
  // an error and quit).
  // - The fonts will be rasterized at a given size (w/ oversampling) and stored
  // into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which
  // ImGui_ImplXXXX_NewFrame below will call.
  // - Use '#define IMGUI_ENABLE_FREETYPE' in your imconfig file to use Freetype
  // for higher quality font rendering.
  // - Read 'docs/FONTS.md' for more instructions and details.
  // - Remember that in C/C++ if you want to include a backslash \ in a string
  // literal you need to write a double backslash \\ !
  // io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf", 18.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
  // io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
  // ImFont* font =
  // io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f,
  // NULL, io.Fonts->GetGlyphRangesJapanese()); IM_ASSERT(font != NULL);

  // Our state
  // bool show_demo_window = true;
  // bool show_another_window = false;

  glEnable(GL_DEPTH_TEST);

  g_views.resize(MAX_N_SPLIT_WIDTH);

  for (size_t vidx = 0; vidx < g_views.size(); vidx++) {
    auto &view = g_views[vidx];
    view.Init(vidx);
  }

  //  Main loop
  while (!glfwWindowShouldClose(window)) {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
    // tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to
    // your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input
    // data to your main application, or clear/overwrite your copy of the
    // keyboard data. Generally you may always pass all inputs to dear imgui,
    // and hide them from your application based on those two flags.
    glfwPollEvents();

    glfwMakeContextCurrent(window);
    Draw(window);
    // glfwSwapBuffers(window);

    // glfwMakeContextCurrent(window2);
    // Draw(window2);
    // glfwSwapBuffers(window2);

    glViewport(0, 0, g_width, g_height);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
