/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#if defined(UGU_USE_OPENCV) && __has_include("opencv2/ximgproc.hpp")
#include "ugu/image_io.h"
#include "ugu/superpixel/superpixel.h"
#include "ugu/util/image_util.h"
#include "ugu/util/path_util.h"

using namespace ugu;

namespace {
void MeanColorPerLabel(const Image3b& img, const Image1i& labels, int label_num,
                       Image3b& mean_color) {
  mean_color = Image3b::zeros(img.rows, img.cols);

  for (int i = 0; i < label_num; i++) {
    Vec3d mean, stddev;
    Image1b mask = labels == i;
    meanStdDev(img, mean, stddev, mask);

    Vec3b mean3b(static_cast<uint8_t>(mean[0]), static_cast<uint8_t>(mean[1]),
                 static_cast<uint8_t>(mean[2]));
    mean_color.setTo(mean3b, mask);
  }
}

}  // namespace

int main() {
  std::string out_dir = "../out/ex25/";
  EnsureDirExists(out_dir);

  ImageBase img, bgr;
  Image1i labels;
  Image1b contour_mask;
  Image3b img_vis;
  Image3b contour_mask3b;
  Image3b mean_color;
  int sp_num;

  std::string data_path = "../data/inpaint/fruits.jpg";
  // data_path = "../data/spot/spot_texture.png";
  img = imread(data_path);
  bgr = img.clone();

  int region_size = 20;
  float ruler = 30.f;
  int min_element_size_percent = 10;
  int num_iterations = 10;
  size_t min_clusters = 2;
  double max_color_diff = 40.0;
  double max_boundary_strengh_for_merge = 80.0;
  double max_boundary_strengh_for_terminate = 120.0;
  SimilarColorClusteringMode mode = SimilarColorClusteringMode::MEAN;

  auto slic_proc = [&](const ImageBase& img) {
    Slic(img, labels, contour_mask, sp_num, region_size, ruler,
         min_element_size_percent, num_iterations);
    FaceId2RandomColor(labels, &img_vis);
    contour_mask3b = Merge(contour_mask, contour_mask, contour_mask);
    contour_mask3b.copyTo(img_vis, contour_mask3b);
    addWeighted(img_vis.clone(), 0.3, bgr, 0.7, 0, img_vis);
    MeanColorPerLabel(bgr, labels, sp_num, mean_color);
  };

  slic_proc(img);
  imwrite(out_dir + "rgb_slic.png", img_vis);
  imwrite(out_dir + "rgb_slic_mean.png", mean_color);
  std::cout << "#Cluster: " << sp_num << std::endl;

  cvtColor(img.clone(), img, cv::COLOR_BGR2HSV);
  slic_proc(img);
  imwrite(out_dir + "hsv_slic.png", img_vis);
  imwrite(out_dir + "hsv_slic_mean.png", mean_color);
  std::cout << "#Cluster: " << sp_num << std::endl;

  SimilarColorClustering(bgr, labels, sp_num, region_size, ruler,
                         min_element_size_percent, num_iterations, min_clusters,
                         max_color_diff, max_boundary_strengh_for_merge,
                         max_boundary_strengh_for_terminate,
                         SimilarColorClusteringMode::MEAN);
  FaceId2RandomColor(labels, &img_vis);
  addWeighted(img_vis.clone(), 0.3, bgr, 0.7, 0, img_vis);
  MeanColorPerLabel(bgr, labels, sp_num, mean_color);
  imwrite(out_dir + "rgb_cluster_mean.png", img_vis);
  imwrite(out_dir + "rgb_cluster_mean_mean.png", mean_color);
  std::cout << "#Cluster: " << sp_num << std::endl;

  SimilarColorClustering(bgr, labels, sp_num, region_size, ruler,
                         min_element_size_percent, num_iterations, min_clusters,
                         max_color_diff, max_boundary_strengh_for_merge,
                         max_boundary_strengh_for_terminate,
                         SimilarColorClusteringMode::MEDIAN);
  FaceId2RandomColor(labels, &img_vis);
  addWeighted(img_vis.clone(), 0.3, bgr, 0.7, 0, img_vis);
  MeanColorPerLabel(bgr, labels, sp_num, mean_color);
  imwrite(out_dir + "rgb_cluster_median.png", img_vis);
  imwrite(out_dir + "rgb_cluster_median_mean.png", mean_color);
  std::cout << "#Cluster: " << sp_num << std::endl;

  return 0;
}
#else
int main() { return 0; }
#endif