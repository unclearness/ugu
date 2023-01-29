/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/image_io.h"
#include "ugu/superpixel/superpixel.h"
#include "ugu/util/image_util.h"
#include "ugu/util/path_util.h"

using namespace ugu;

namespace {}  // namespace

int main() {
#ifdef UGU_USE_OPENCV
  std::string out_dir = "../out/ex25/";
  EnsureDirExists(out_dir);

  ImageBase img, bgr;
  Image1i labels;
  Image1b contour_mask;
  Image3b img_vis;
  Image3b contour_mask3b;

  img = imread("../data/inpaint/fruits.jpg");
  bgr = img.clone();

  auto slic_proc = [&](const ImageBase& img) {
    Slic(img, labels, contour_mask);
    FaceId2RandomColor(labels, &img_vis);
    contour_mask3b = Merge(contour_mask, contour_mask, contour_mask);
    contour_mask3b.copyTo(img_vis, contour_mask3b);
    addWeighted(img_vis.clone(), 0.2, bgr, 0.8, 0, img_vis);
  };

  slic_proc(img);
  imwrite(out_dir + "rgb_slic.png", img_vis);

  cvtColor(img.clone(), img, cv::COLOR_BGR2HSV);
  slic_proc(img);
  imwrite(out_dir + "hsv_slic.png", img_vis);
#endif
  return 0;
}
