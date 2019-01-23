#include <fstream>

#include "glm/ext/matrix_transform.hpp"
#include "include/renderer.h"

using namespace crender;

glm::mat4 make_c2w(const glm::vec3& eye, const glm::vec3& center,
                   const glm::vec3& up) {
  // glm::lookAtRH returns view matrix (world -> camera) with z:backward, y:up,
  // x:right, like OpenGL
  glm::mat4 w2c_gl = glm::lookAtRH(eye, center, up);
  glm::mat4 c2w_gl = glm::inverse(w2c_gl);

  // rotate 180 deg.around x_axis to align z:forward, y:down, x:right,
  glm::vec3 x_axis(1, 0, 0);
  glm::mat4 c2w = glm::rotate(c2w_gl, glm::radians<float>(180), x_axis);

  return c2w;
}

void visualize_depth(const Image1w& depth, Image1b& vis_depth,
                     float min_d = 200.0f, float max_d = 1500.0f) {
  vis_depth.init(depth.width(), depth.height());

  for (int y = 0; y < vis_depth.height(); y++) {
    for (int x = 0; x < vis_depth.width(); x++) {
      unsigned short d = depth.at(x, y, 0);
      if (d < 1) {
        continue;
      }

      int color = static_cast<int>((d - min_d) / (max_d - min_d) * 255.0);

      if (color < 0) {
        color = 0;
      }
      if (255 < color) {
        color = 255;
      }

      vis_depth.at(x, y, 0) = static_cast<unsigned char>(color);
    }
  }
}

void test(const std::string& out_dir, std::shared_ptr<Mesh> mesh,
          std::shared_ptr<Camera> camera, Renderer& renderer) {
  // images
  Image3b color;
  Image1w depth;
  Image1b mask;
  Image1b vis_depth;

  MeshStats stats = mesh->stats();
  glm::vec3 center = stats.center;
  glm::vec3 eye;
  glm::mat4 c2w;

  // translation offset is the largest edge length of bounding box * 2
  glm::vec3 diff = stats.bb_max - stats.bb_min;
  float offset = std::max(diff[0], std::max(diff[1], diff[2])) * 2;

  // from front
  eye = center;
  eye[2] -= offset;
  c2w = make_c2w(eye, center, glm::vec3(0, -1, 0));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "front_color.png");
  mask.write_png(out_dir + "front_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "front_vis_depth.png");

  // from back
  eye = center;
  eye[2] += offset;
  c2w = make_c2w(eye, center, glm::vec3(0, -1, 0));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "back_color.png");
  mask.write_png(out_dir + "back_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "back_vis_depth.png");

  // from right
  eye = center;
  eye[0] += offset;
  c2w = make_c2w(eye, center, glm::vec3(0, -1, 0));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "right_color.png");
  mask.write_png(out_dir + "right_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "right_vis_depth.png");

  // from left
  eye = center;
  eye[0] -= offset;
  c2w = make_c2w(eye, center, glm::vec3(0, -1, 0));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "left_color.png");
  mask.write_png(out_dir + "left_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "left_vis_depth.png");

  // from top
  eye = center;
  eye[1] -= offset;
  c2w = make_c2w(eye, center, glm::vec3(0, 0, 1));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "top_color.png");
  mask.write_png(out_dir + "top_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "top_vis_depth.png");

  // from bottom
  eye = center;
  eye[1] += offset;
  c2w = make_c2w(eye, center, glm::vec3(0, 0, -1));
  camera->set_c2w(Pose(c2w));
  renderer.render(color, depth, mask);
  color.write_png(out_dir + "bottom_color.png");
  mask.write_png(out_dir + "bottom_mask.png");
  visualize_depth(depth, vis_depth);
  vis_depth.write_png(out_dir + "bottom_vis_depth.png");
}

void align_mesh(std::shared_ptr<Mesh> mesh) {
  // move center as origin
  MeshStats stats = mesh->stats();
  mesh->translate(-stats.center);

  // rotate 180 deg. around x_axis to align z:forward, y:down, x:right,
  glm::vec3 x_axis(1, 0, 0);
  glm::mat4 R = glm::rotate(glm::mat4(1), glm::radians<float>(180), x_axis);
  mesh->rotate(R);

  // recover original translation
  mesh->translate(stats.center);
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // CMake downloads and unzip this data automatically
  // Please do manually if it got wrong
  // http://www.kunzhou.net/tex-models/bunny.zip
  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  std::ifstream ifs(obj_path);
  if (!ifs.is_open()) {
    LOGE("Please put %s\n", obj_path.c_str());
    return -1;
  }

  // load mesh
  std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
  mesh->load_obj(obj_path, data_dir);

  // original mesh with z:backward, y:up, x:right, like OpenGL
  // align z:forward, y:down, x:right
  align_mesh(mesh);

  // initialize renderer with default option
  RendererOption option;
  Renderer renderer(option);

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.prepare_mesh();

  Pose pose;
  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  int width = 640;
  int height = 480;
  glm::vec2 principal_point(318.6f, 255.3f);
  glm::vec2 focal_length(517.3f, 516.5f);
  std::shared_ptr<Camera> camera = std::make_shared<PinholeCamera>(
      width, height, pose, principal_point, focal_length);

  // set camera
  renderer.set_camera(camera);

  // test
  test(data_dir, mesh, camera, renderer);

  return 0;
}
