/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/discrete/bin_packer_2d.h"

#include <deque>

#include "ugu/util/image_util.h"

namespace {

bool FindBestRect(const std::vector<ugu::Rect>& available_rects,
                  const ugu::Rect& target, int* index, ugu::Rect* best_rect) {
  *index = -1;
  int best_area_diff = std::numeric_limits<int>::max();

  for (int i = 0; i < static_cast<int>(available_rects.size()); i++) {
    const ugu::Rect& rect = available_rects[i];
    if (target.height <= rect.height && target.width <= rect.width &&
        (rect.area() - target.area()) < best_area_diff) {
      best_area_diff = rect.area() - target.area();
      *best_rect = rect;
      *index = i;
    }
  }

  return true;
}

bool CutRect(const ugu::Rect& src, const ugu::Rect& target, ugu::Rect* cut1,
             ugu::Rect* cut2) {
  int h_diff = src.height - target.height;
  int w_diff = src.width - target.width;

  if (h_diff > w_diff) {
    cut1->x = src.x;
    cut1->y = src.y;
    cut1->width = src.width;
    cut1->height = target.height;

    cut2->x = src.x;
    cut2->y = src.y + target.height;
    cut2->width = src.width;
    cut2->height = src.height - target.height;

  } else {
    cut1->x = src.x;
    cut1->y = src.y;
    cut1->width = target.width;
    cut1->height = src.height;

    cut2->x = src.x + target.width;
    cut2->y = src.y;
    cut2->width = src.width - target.width;
    cut2->height = src.height;
  }

  return true;
}

}  // namespace

namespace ugu {

bool BinPacking2D(const std::vector<Rect>& rects, std::vector<Rect>* packed_pos,
                  std::vector<Rect>* available_rects, int w, int h) {
  std::deque<Rect> rects_;
  std::copy(rects.begin(), rects.end(), std::back_inserter(rects_));

  available_rects->clear();
  available_rects->push_back(Rect(0, 0, w, h));
  packed_pos->clear();

  while (true) {
    if (rects_.empty()) {
      break;
    }
    if (available_rects->empty()) {
      return false;
    }

    Rect rect = rects_.front();
    rects_.pop_front();

    while (true) {
      int best_index{-1};
      Rect best_rect;
      FindBestRect(*available_rects, rect, &best_index, &best_rect);
      if (best_index < 0 || best_rect.area() < rect.area()) {
        return false;
      } else if (best_rect.height == rect.height &&
                 best_rect.width == rect.width) {
        packed_pos->push_back(best_rect);
        available_rects->erase(available_rects->begin() + best_index);
        break;
      } else {
        Rect cut1, cut2;
        CutRect(best_rect, rect, &cut1, &cut2);
        available_rects->erase(available_rects->begin() + best_index);
        available_rects->push_back(cut2);
        available_rects->push_back(cut1);
      }
    }
  }

  return true;
}

Image3b DrawPackesRects(const std::vector<Rect>& packed_rects, int w, int h) {
  Image3b res = Image3b::zeros(h, w);

  std::vector<Eigen::Vector3f> random_colors =
      GenRandomColors(static_cast<int32_t>(packed_rects.size()), 0.f, 255.f, 0);
  for (size_t i = 0; i < packed_rects.size(); i++) {
    Vec3b c{random_colors[i][0], random_colors[i][1], random_colors[i][2]};
    const auto& r = packed_rects[i];
    for (int y = r.y; y < r.y + r.height; y++) {
      for (int x = r.x; x < r.x + r.width; x++) {
        res.at<Vec3b>(y, x) = c;
      }
    }
  }
  return res;
}

}  // namespace ugu
