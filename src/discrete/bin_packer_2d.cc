/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/discrete/bin_packer_2d.h"

#include <deque>

#include "ugu/util/image_util.h"

namespace {

template <typename T>
bool FindBestRect(const std::vector<ugu::Rect_<T>>& available_rects,
                  const ugu::Rect_<T>& target, int* index,
                  ugu::Rect_<T>* best_rect) {
  *index = -1;
  T best_area_diff = std::numeric_limits<T>::max();

  for (int i = 0; i < static_cast<int>(available_rects.size()); i++) {
    const ugu::Rect_<T>& rect = available_rects[i];
    if (target.height <= rect.height && target.width <= rect.width &&
        (rect.area() - target.area()) < best_area_diff) {
      best_area_diff = rect.area() - target.area();
      *best_rect = rect;
      *index = i;
    }
  }

  return true;
}

template <typename T>
bool CutRect(const ugu::Rect_<T>& src, const ugu::Rect_<T>& target,
             ugu::Rect_<T>* cut1, ugu::Rect_<T>* cut2) {
  T h_diff = src.height - target.height;
  T w_diff = src.width - target.width;

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

template <typename T>
bool BinPacking2DImpl(const std::vector<ugu::Rect_<T>>& rects,
                      std::vector<ugu::Rect_<T>>* packed_pos,
                      std::vector<ugu::Rect_<T>>* available_rects, T w_min,
                      T w_max, T h_min, T h_max) {
  std::deque<ugu::Rect_<T>> rects_;
  std::copy(rects.begin(), rects.end(), std::back_inserter(rects_));

  available_rects->clear();
  available_rects->push_back(ugu::Rect_<T>(w_min, h_min, w_max, h_max));
  packed_pos->clear();

  while (true) {
    if (rects_.empty()) {
      break;
    }
    if (available_rects->empty()) {
      return false;
    }

    ugu::Rect_<T> rect = rects_.front();
    rects_.pop_front();

    while (true) {
      int best_index{-1};
      ugu::Rect_<T> best_rect;
      FindBestRect(*available_rects, rect, &best_index, &best_rect);
      if (best_index < 0 || best_rect.area() < rect.area()) {
        return false;
      } else if (best_rect.height == rect.height &&
                 best_rect.width == rect.width) {
        packed_pos->push_back(best_rect);
        available_rects->erase(available_rects->begin() + best_index);
        break;
      } else {
        ugu::Rect_<T> cut1, cut2;
        CutRect(best_rect, rect, &cut1, &cut2);
        available_rects->erase(available_rects->begin() + best_index);
        available_rects->push_back(cut2);
        available_rects->push_back(cut1);
      }
    }
  }

  return true;
}

}  // namespace

namespace ugu {

bool BinPacking2D(const std::vector<Rect>& rects, std::vector<Rect>* packed_pos,
                  std::vector<Rect>* available_rects, int x_min, int x_max,
                  int y_min, int y_max) {
  return BinPacking2DImpl(rects, packed_pos, available_rects, x_min, x_max,
                          y_min, y_max);
}

bool BinPacking2D(const std::vector<Rect2f>& rects,
                  std::vector<Rect2f>* packed_pos,
                  std::vector<Rect2f>* available_rects, float x_min,
                  float x_max, float y_min, float y_max) {
  return BinPacking2DImpl(rects, packed_pos, available_rects, x_min, x_max,
                          y_min, y_max);
}

Image3b DrawPackedRects(const std::vector<Rect>& packed_rects, int w, int h) {
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

Image3b DrawPackedRects(const std::vector<Rect2f>& packed_rects, int w, int h) {
  std::vector<Rect> packed_rects_2i;
  std::transform(packed_rects.begin(), packed_rects.end(),
                 std::back_inserter(packed_rects_2i), [&](const Rect2f& r) {
                   return Rect(static_cast<int>(r.x * w),
                               static_cast<int>(r.y * h),
                               static_cast<int>(r.width * w),
                               static_cast<int>(r.height * w));
                 });
  return DrawPackedRects(packed_rects_2i, w, h);
}

}  // namespace ugu
