#include "cpu_renderer.h"

int main(int argc, char *argv[]) {
  
  using namespace unclearness;

  std::string data_dir = "../data/bunny/";
  std::string obj_path = data_dir + "bunny.obj";

  // load mesh
  std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
  mesh->load_obj(obj_path, data_dir);

  //mesh->diffuse_tex().write_png(data_dir + "out_texture.png");

  CpuRenderer renderer;

  // set mesh
  renderer.set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer.prepare_mesh();

  // camera pose
  glm::mat3 R{0.997859f,  0.0555307f, -0.0345624f, -0.0595537f, 0.989853f,
              -0.129011f, 0.0270477f, 0.130793f,   0.991041f};
  glm::vec3 t{-26.3404f, -127.373f, -965.128f};

  Pose pose(R, t);

  // Make PinholeCamera
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  // Freiburg 1 RGB
  int width = 640;
  int height = 480;
  glm::vec2 principal_point(318.6f, 255.3f);
  glm::vec2 focal_length(517.3f, 516.5f);
  std::shared_ptr<Camera> camera = std::make_shared<PinholeCamera>(
      width, height, pose, principal_point, focal_length);

  // set camera
  renderer.set_camera(camera);

  // render images
  Image3b color;
  Image1w depth;
  Image1b mask;
  renderer.render(color, depth, mask);

  color.write_png(data_dir + "render_color.png");
  mask.write_png(data_dir + "render_mask.png");

  Image3b test(width, height);
  test.at(100, 200, 0) = 255;

  test.at(300, 400, 1) = 255;
  test.write_png(data_dir + "test.png");

  return 0;
}
