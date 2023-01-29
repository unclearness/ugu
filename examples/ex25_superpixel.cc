/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

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
#if defined(UGU_USE_OPENCV) && __has_include("opencv2/ximgproc.hpp")
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

  auto slic_proc = [&](const ImageBase& img) {
    Slic(img, labels, contour_mask, sp_num);
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

  SimilarColorClustering(bgr, labels, sp_num);
  FaceId2RandomColor(labels, &img_vis);
  addWeighted(img_vis.clone(), 0.3, bgr, 0.7, 0, img_vis);
  MeanColorPerLabel(bgr, labels, sp_num, mean_color);
  imwrite(out_dir + "rgb_cluster.png", img_vis);
  imwrite(out_dir + "rgb_cluster_mean.png", mean_color);
  std::cout << "#Cluster: " << sp_num << std::endl;

#endif
  return 0;
}
