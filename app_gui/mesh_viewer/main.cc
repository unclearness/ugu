// Dear ImGui: standalone example application for GLFW + OpenGL 3, using
// programmable pipeline (GLFW is a cross-platform general purpose library for
// handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation,
// etc.) If you are new to Dear ImGui, read documentation from the docs/ folder
// + read the top of imgui.cpp. Read online:
// https://github.com/ocornut/imgui/tree/master/docs

#include <limits>

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

RendererGlPtr g_renderer;

int g_width = 1280;
int g_height = 720;

Eigen::Vector2d g_prev_cursor_pos;
Eigen::Vector2d g_cursor_pos;
Eigen::Vector2d g_mouse_l_pressed_pos;
Eigen::Vector2d g_mouse_l_released_pos;
Eigen::Vector2d g_mouse_m_pressed_pos;
Eigen::Vector2d g_mouse_m_released_pos;
// Eigen::Vector3d g_center;
// ugu::MeshStats g_stats;
// Eigen::Vector3f g_bb_max, g_bb_min;
PinholeCameraPtr g_camera;
bool g_to_process_drag_l = false;
bool g_to_process_drag_m = false;

bool g_prev_mouse_l_pressed = false;
bool g_prev_mouse_m_pressed = false;
bool g_mouse_l_pressed = false;
bool g_mouse_m_pressed = false;
const double drag_th = 2.0;

double g_mouse_wheel_yoffset = 0.0;
bool g_to_process_wheel = false;

std::vector<RenderableMeshPtr> g_meshes;
// std::vector<BvhPtr<Eigen::Vector3f, Eigen::Vector3i>> g_bvhs;

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

void ResetGl() {
  g_renderer->ClearGlState();
  for (const auto mesh : g_meshes) {
    g_renderer->SetMesh(mesh);
  }
  g_renderer->Init();
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
      g_meshes.clear();
      ResetGl();
    }
  }
}

void mouse_button_callback(GLFWwindow *pwin, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    // printf("L - down\n");
    g_prev_mouse_l_pressed = g_mouse_l_pressed;
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

  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
    // Cast ray
    Eigen::Vector3f dir_c_cv;
    g_camera->ray_c(static_cast<float>(g_cursor_pos[0]),
                    static_cast<float>(g_cursor_pos[1]), &dir_c_cv);

    const Eigen::Affine3d offset =
        Eigen::Affine3d(Eigen::AngleAxisd(pi, Eigen::Vector3d::UnitX()))
            .inverse();
    Eigen::Vector3f dir_c_gl =
        (dir_c_cv.transpose() * offset.rotation().cast<float>());
    Eigen::Vector3f dir_w_gl =
        g_camera->c2w().rotation().cast<float>() * dir_c_gl;

    size_t min_geoid = ~0u;
    float min_geo_dist = std::numeric_limits<float>::max();
    Eigen::Vector3f min_geo_dist_pos =
        Eigen::Vector3f::Constant(std::numeric_limits<float>::max());

    Ray ray;
    ray.dir = dir_w_gl;
    ray.org = g_camera->c2w().translation().cast<float>();
    auto results_all = g_renderer->Intersect(ray);

    for (size_t geoid = 0; geoid < g_meshes.size(); geoid++) {
      const std::vector<IntersectResult> &results = results_all[geoid];
      if (!results.empty()) {
        std::cout << geoid << ": " << results[0].t << " " << results[0].fid
                  << " " << results[0].u << ", " << results[0].v << std::endl;
        if (results[0].t < min_geo_dist) {
          min_geoid = geoid;
          min_geo_dist = results[0].t;
          min_geo_dist_pos = results[0].t * ray.dir + ray.org;
        }
      }
    }

    if (min_geoid != ~0u) {
      std::cout << "closest geo: " << min_geoid << std::endl;
      g_renderer->AddSelectedPos(min_geo_dist_pos);
    }
  }

  if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    g_prev_mouse_m_pressed = g_mouse_m_pressed;
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
    g_meshes.push_back(mesh);

  } else {
    return;
  }

  ResetGl();

  Eigen::Vector3f bb_max, bb_min;
  g_renderer->GetMergedBoundingBox(bb_max, bb_min);
  float z_trans = (bb_max - bb_min).maxCoeff() * 2.0f;
  g_renderer->SetNearFar(static_cast<float>(z_trans * 0.5f / 10),
                         static_cast<float>(z_trans * 2.f * 10));

  Eigen::Affine3d c2w = Eigen::Affine3d::Identity();
  c2w.translation() = Eigen::Vector3d(0, 0, z_trans);
  g_camera->set_c2w(c2w);
}

void drop_callback(GLFWwindow *window, int count, const char **paths) {
  for (int i = 0; i < count; i++) {
    std::cout << "Dropped: " << i << "/" << count << " " << paths[i]
              << std::endl;
  }
  LoadMesh(paths[0]);
}

void window_size_callback(GLFWwindow *window, int width, int height) {
  // Assume window size equals to framebuffer size
  // So, do nothing here. Everything will be handled by
  // framebuffer_size_callback
  // TODO: Deal with it more appopriately
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // glViewport(0, 0, width, height)
  g_width = width;
  g_height = height;

  g_camera->set_size(g_width, g_height);
  g_camera->set_fov_y(45.0f);
  g_camera->set_principal_point({g_width / 2.f, g_height / 2.f});

  g_renderer->SetSize(g_width, g_height);
  ResetGl();
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
  if (window == NULL) return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  glfwSetCursorPosCallback(window, cursor_pos_callback);

  glfwSetKeyCallback(window, key_callback);

  glfwSetMouseButtonCallback(window, mouse_button_callback);

  glfwSetScrollCallback(window, mouse_wheel_callback);

  glfwSetDropCallback(window, drop_callback);

  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

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
  bool show_demo_window = true;
  bool show_another_window = false;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  glEnable(GL_DEPTH_TEST);

#if 0
  RenderableMeshPtr mesh = RenderableMesh::Create();
  mesh->LoadObj("../data/bunny/bunny.obj", "../data/bunny/");

  RenderableMeshPtr mesh2 = RenderableMesh::Create();
  mesh2->LoadObj("../data/spot/spot_triangulated.obj", "../data/spot/");

  MeshStats stats = mesh->stats();
  Eigen::Vector3f bb_len = stats.bb_max - stats.bb_min;

  MeshStats stats2 = mesh2->stats();
  Eigen::Vector3f bb_len_2 = stats2.bb_max - stats2.bb_min;

  float scale = bb_len_2.maxCoeff() / bb_len.maxCoeff();

  mesh->Scale(scale);
  mesh->CalcStats();
  stats = mesh->stats();
  bb_len = stats.bb_max - stats.bb_min;

  // mesh.BindTextures();
  // mesh.SetupMesh();

  // Shader shader;
  // shader.SetFragType(FragShaderType::UNLIT);
  // shader.Prepare();

  Eigen::Matrix4f model_mat = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f model_mat_2 = Eigen::Matrix4f::Identity();

  // int modelLoc = glGetUniformLocation(shader.ID, "model");
  // glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model_mat.data());
#endif
  g_camera = std::make_shared<PinholeCamera>(g_width, g_height, 45.f);
  // Eigen::Affine3d c2w = Eigen::Affine3d::Identity();
  //  double z_trans = bb_len.maxCoeff() * 3;
  //  c2w.translation() = Eigen::Vector3d(0, 0, z_trans);
  //  camera->set_c2w(c2w);

#if 0
  Eigen::Matrix4f view_mat = camera.c2w().inverse().matrix().cast<float>();

  int viewLoc = glGetUniformLocation(shader.ID, "view");

  glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view_mat.data());
  Eigen::Matrix4f prj_mat = camera.ProjectionMatrixOpenGl(0.1f, 1000.f);
  int prjLoc = glGetUniformLocation(shader.ID, "projection");
  glUniformMatrix4fv(prjLoc, 1, GL_FALSE, prj_mat.data());
#endif

  g_renderer = std::make_shared<RendererGl>();

  g_renderer->SetSize(g_width, g_height);

  g_renderer->SetCamera(g_camera);
  // renderer->SetMesh(mesh);

  // renderer->SetMesh(mesh2);

  // renderer->SetNearFar(static_cast<float>(z_trans * 0.5f / 10),
  //                      static_cast<float>(z_trans * 2.f * 10));

  g_renderer->Init();

  Eigen::Vector3f default_color(0.45f, 0.55f, 0.60f);
  g_renderer->SetBackgroundColor(default_color);
  g_renderer->SetWireColor(default_color);
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // g_stats = mesh->stats();

  int count = 0;
  // Main loop
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

    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if 1
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in
    // ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear
    // ImGui!).
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair
    // to create a named window.
    {
      static float f = 0.0f;
      static int counter = 0;

      ImGui::Begin("Hello, world!");  // Create a window called "Hello, world!"
                                      // and append into it.

      ImGui::Text("This is some useful text.");  // Display some text (you can
                                                 // use a format strings too)
      ImGui::Checkbox(
          "Demo Window",
          &show_demo_window);  // Edit bools storing our window open/close state
      ImGui::Checkbox("Another Window", &show_another_window);

      ImGui::SliderFloat(
          "float", &f, 0.0f,
          1.0f);  // Edit 1 float using a slider from 0.0f to 1.0f
      ImGui::ColorEdit3(
          "clear color",
          (float *)&clear_color);  // Edit 3 floats representing a color

      if (ImGui::Button(
              "Button"))  // Buttons return true when clicked (most widgets
                          // return true when edited/activated)
        counter++;
      ImGui::SameLine();
      ImGui::Text("counter = %d", counter);

      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }
    // 3. Show another simple window.
    if (show_another_window) {
      ImGui::Begin(
          "Another Window",
          &show_another_window);  // Pass a pointer to our bool variable (the
                                  // window will have a closing button that will
                                  // clear the bool when clicked)
      ImGui::Text("Hello from another window!");
      if (ImGui::Button("Close Me")) show_another_window = false;
      ImGui::End();
    }

    {
      ImGui::Begin("Mesh Tool");

      static char mesh_path[1024] = "../data/bunny/bunny.obj";
      ImGui::InputText("Mesh path", mesh_path, 1024u);
      if (ImGui::Button("Load mesh")) {
        LoadMesh(mesh_path);
      }

      bool show_wire = g_renderer->GetShowWire();
      if (ImGui::Checkbox("show wire", &show_wire)) {
        g_renderer->SetShowWire(show_wire);
      }

      Eigen::Vector3f wire_col = g_renderer->GetWireColor();
      if (ImGui::ColorEdit3("wire color", wire_col.data())) {
        g_renderer->SetWireColor(Eigen::Vector3f(wire_col));
      }

      if (ImGui::Button("Save GBuffer")) {
        g_renderer->ReadGbuf();
        GBuffer gbuf;
        g_renderer->GetGbuf(gbuf);
#if 1
        Image3b vis_pos_wld, vis_pos_cam;
        vis_pos_wld = ColorizePosMap(gbuf.pos_wld);
        imwrite("pos_wld.png", vis_pos_wld);
        vis_pos_cam = ColorizePosMap(gbuf.pos_cam);
        imwrite("pos_cam.png", vis_pos_cam);

        Image3b vis_normal_wld, vis_normal_cam;
        Normal2Color(gbuf.normal_wld, &vis_normal_wld, true);
        imwrite("normal_wld.png", vis_normal_wld);
        Normal2Color(gbuf.normal_cam, &vis_normal_cam, true);
        imwrite("normal_cam.png", vis_normal_cam);
#endif
        Image3b vis_depth;
        Depth2Color(gbuf.depth_01, &vis_depth, 0.f, 1.f);
        imwrite("depth01.png", vis_depth);

        Image3b vis_geoid;
        FaceId2RandomColor(gbuf.geo_id, &vis_geoid);
        imwrite("geoid.png", vis_geoid);

        Image3b vis_faceid;
        FaceId2RandomColor(gbuf.face_id, &vis_faceid);
        imwrite("faceid.png", vis_faceid);

        Image3b vis_bary = ColorizeBarycentric(gbuf.bary);
        imwrite("bary.png", vis_bary);

        Image3b vis_uv = ColorizeBarycentric(gbuf.uv);
        imwrite("uv.png", vis_uv);

        imwrite("color.png", gbuf.color);
      }
      ImGui::End();
    }

#endif

    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);

#if 0
    {
      shader.Use();

      model_mat.block(0, 0, 3, 3) =
          Eigen::AngleAxisf(0.03 * count, Eigen::Vector3f(0, 1, 0)).matrix();
      count++;

      glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model_mat.data());
      glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view_mat.data());
      glUniformMatrix4fv(prjLoc, 1, GL_FALSE, prj_mat.data());

      mesh.Draw(shader);
    }
#endif

#if 0
    model_mat.block(0, 0, 3, 3) =
        Eigen::AngleAxisf(0.03f * count, Eigen::Vector3f(0, 1, 0)).matrix();

    model_mat_2.block(0, 0, 3, 3) =
        Eigen::AngleAxisf(0.05f * count, Eigen::Vector3f(1, 0, 0)).matrix();



    renderer->SetMesh(mesh, Eigen::Affine3f(model_mat));

    renderer->SetMesh(mesh2, Eigen::Affine3f(model_mat_2));
#endif

    count++;

    g_renderer->Draw();

    if (!ImGui::IsAnyItemActive()) {
      Eigen::Vector3f bb_max, bb_min;
      g_renderer->GetMergedBoundingBox(bb_max, bb_min);

      if (g_to_process_drag_l) {
        g_to_process_drag_l = false;
        Eigen::Vector2d diff = g_cursor_pos - g_prev_cursor_pos;
        if (diff.norm() > drag_th) {
          const double rotate_speed = ugu::pi / 180 * 10;

          Eigen::Affine3d cam_pose_cur = g_camera->c2w();
          Eigen::Matrix3d R_cur = cam_pose_cur.rotation();

          Eigen::Vector3d right_axis = -R_cur.col(0);
          Eigen::Vector3d up_axis = -R_cur.col(1);

          Eigen::Quaterniond R_offset =
              Eigen::AngleAxisd(2 * ugu::pi * diff[0] / g_height * rotate_speed,
                                up_axis) *
              Eigen::AngleAxisd(2 * ugu::pi * diff[1] / g_height * rotate_speed,
                                right_axis);

          Eigen::Affine3d cam_pose_new = R_offset * cam_pose_cur;

          g_camera->set_c2w(cam_pose_new);
        }
      }

      if (g_to_process_drag_m) {
        g_to_process_drag_m = false;
        Eigen::Vector2d diff = g_cursor_pos - g_prev_cursor_pos;
        if (diff.norm() > drag_th) {
          const double trans_speed = (bb_max - bb_min).maxCoeff() / g_height;

          Eigen::Affine3d cam_pose_cur = g_camera->c2w();
          Eigen::Matrix3d R_cur = cam_pose_cur.rotation();

          Eigen::Vector3d right_axis = -R_cur.col(0);
          Eigen::Vector3d up_axis = R_cur.col(1);

          Eigen::Vector3d t_offset = right_axis * diff[0] * trans_speed +
                                     up_axis * diff[1] * trans_speed;

          Eigen::Affine3d cam_pose_new =
              Eigen::Translation3d(t_offset + cam_pose_cur.translation()) *
              cam_pose_cur.rotation();
          g_camera->set_c2w(cam_pose_new);
        }
      }

      if (g_to_process_wheel) {
        g_to_process_wheel = false;
        const double wheel_speed = (bb_max - bb_min).maxCoeff() / 20;

        Eigen::Affine3d cam_pose_cur = g_camera->c2w();
        Eigen::Vector3d t_offset = cam_pose_cur.rotation().col(2) *
                                   -g_mouse_wheel_yoffset * wheel_speed;
        Eigen::Affine3d cam_pose_new =
            Eigen::Translation3d(t_offset + cam_pose_cur.translation()) *
            cam_pose_cur.rotation();
        g_camera->set_c2w(cam_pose_new);
      }
    }
    glClear(GL_DEPTH_BUFFER_BIT);

#if 1
    // glClear(GL_COLOR_BUFFER_BIT);
    //  Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
#endif
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
