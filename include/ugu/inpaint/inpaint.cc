/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "inpaint.h"

#include <queue>

namespace {

void InpaintNaive(const ugu::Image1b& mask, ugu::Image3b& color,
                  int inpaint_kernel_size) {
  using namespace ugu;

  int ks = std::max(3, inpaint_kernel_size);
  int hks = ks / 2;
  // int erode_kernel_size = hks * 2 + 1;

  auto blend_color = [&](Vec3b& blended_pix,
                         const std::vector<Vec3b>& sources) {
    if (sources.empty()) {
      return;
    }
    for (int c = 0; c < 3; c++) {
      double ave_color = 0.0;
      for (const auto& s : sources) {
        ave_color += s[c];
      }
      ave_color /= sources.size();
      ave_color = std::clamp(ave_color, 0.0, 255.0);
      blended_pix[c] = static_cast<unsigned char>(ave_color);
    }
    return;
  };

  // Initialize not_visited pixels within inside mask and not on org_pix_mask
  // std::set<std::pair<int, int>> not_visited;
  ugu::Image1b not_visited_mask = ugu::Image1b::zeros(mask.rows, mask.cols);
  for (int j = hks; j < mask.rows - hks; j++) {
    for (int i = hks; i < mask.cols - hks; i++) {
      if (mask.at<unsigned char>(j, i) != 0) {
        not_visited_mask.at<unsigned char>(j, i) = 255;
      }
    }
  }

  // Distance transform
  ugu::Image1f dist;
  ugu::DistanceTransformL1(mask, &dist);
  double minDist, maxDist;
  ugu::minMaxLoc(dist, &minDist, &maxDist);
  std::vector<std::vector<std::pair<int, int>>> dist_pix_table(
      static_cast<size_t>(std::ceil(maxDist)) + 1);
  for (int j = hks; j < mask.rows - hks; j++) {
    for (int i = hks; i < mask.cols - hks; i++) {
      if (mask.at<unsigned char>(j, i) != 0) {
        auto d = static_cast<int>(std::ceil(dist.at<float>(j, i)));
        dist_pix_table[d].push_back({i, j});
      }
    }
  }

  int cur_dist = 0;
  while (true) {
    for (auto d = 0; d < cur_dist; d++) {
      auto& dist_pix = dist_pix_table[d];
      for (auto it = dist_pix.begin(); it != dist_pix.end();) {
        auto [i, j] = *it;

        auto& m = not_visited_mask.at<unsigned char>(j, i);
        if (m != 0) {
          // Check at least one inpainted pixel
          std::vector<Vec3b> sources;
          for (int jj = -hks; jj <= hks; jj++) {
            for (int ii = -hks; ii <= hks; ii++) {
              if (ii == 0 && jj == 0) {
                continue;
              }
              if (not_visited_mask.at<unsigned char>(j + jj, i + ii) == 0) {
                sources.push_back(color.at<Vec3b>(j + jj, i + ii));
              }
            }
          }

          if (sources.empty()) {
            ++it;
            continue;
          }

          // Make inpainted color
          auto& inpainted_pix = color.at<Vec3b>(j, i);
          blend_color(inpainted_pix, sources);

          // Update not_visited flags
          m = 0;
          // not_visited.erase({i, j});
          it = dist_pix.erase(it);
        } else {
          it = dist_pix.erase(it);
        }
      }
    }

    if (cur_dist < dist_pix_table.size()) {
      cur_dist++;
    }

    bool finished = true;
    for (const auto& dp : dist_pix_table) {
      finished &= dp.empty();
    }
    if (finished) {
      break;
    }
  }
}

}  // namespace

namespace ugu {

void FastMarchingMethod(const Image1b& mask, Image1f* dist,
                        const float illegal_val) {
  // Reference of FMM for inpainting
  // OpenCV implementation:
  // https://github.com/opencv/opencv/blob/master/modules/photo/src/inpaint.cpp
  // Author's implementation:
  // https://github.com/erich666/jgt-code/blob/master/Volume_09/Number_1/Telea2004/AFMM_Inpainting/fmm.cpp
  // A python implementation: https://github.com/olvb/pyheal

  constexpr unsigned char UNKNOWN = 0;
  constexpr unsigned char BAND = 1;
  constexpr unsigned char KNOWN = 2;
  // constexpr unsigned char INSIDE = 3; // Used for inpainting with
  // modification

  Image1b flags = Image1b::zeros(mask.rows, mask.cols);

  if (mask.cols != dist->cols || mask.rows != dist->rows) {
    *dist = Image1f::zeros(mask.rows, mask.cols);
  } else {
    dist->setTo(0.f);
  }

  struct FrontPix {
    float T = std::numeric_limits<float>::max();  // distanceto - boundary
    int x = 0;
    int y = 0;
    FrontPix(){};
    ~FrontPix(){};
    FrontPix(float T_, int x_, int y_) {
      T = T_;
      x = x_;
      y = y_;
    }
  };

  auto compare = [](const FrontPix& l, const FrontPix& r) { return l.T > r.T; };
  std::priority_queue<FrontPix, std::vector<FrontPix>, decltype(compare)>
      narrow_band{compare};

  // Initialization
  for (int j = 0; j < mask.rows; j++) {
    for (int i = 0; i < mask.cols; i++) {
      if (mask.at<unsigned char>(j, i) == 0) {
        flags.at<unsigned char>(j, i) = KNOWN;  // outside of mask is known as 0
        continue;
      }

      // Check 4-neightbors if it contancts non-zero pixel
      std::vector<std::pair<int, int>> neighbors_4;
      if (0 < i) {
        neighbors_4.push_back({i - 1, j});
      }
      if (i < mask.cols - 1) {
        neighbors_4.push_back({i + 1, j});
      }
      if (0 < j) {
        neighbors_4.push_back({i, j - 1});
      }
      if (j < mask.rows - 1) {
        neighbors_4.push_back({i, j + 1});
      }
      for (auto& neighbor : neighbors_4) {
        auto [x, y] = neighbor;

        auto& f = flags.at<unsigned char>(y, x);
        if (f == BAND) {
          continue;
        }

        if (mask.at<unsigned char>(y, x) == 0) {
          narrow_band.push(FrontPix(0.f, x, y));
          dist->at<float>(y, x) = 0.f;
          f = BAND;
        }
      }
    }
  }

  auto solve_eikonal = [&](int i1, int j1, int i2, int j2) -> float {
    if (flags.cols - 1 < i1 || flags.cols - 1 < i2 || flags.rows - 1 < j1 ||
        flags.rows - 1 < j2) {
      return std::numeric_limits<float>::max();
    }

    if (flags.at<unsigned char>(j1, i1) != UNKNOWN &&
        flags.at<unsigned char>(j2, i2) != UNKNOWN) {
      const auto& d1 = dist->at<float>(j1, i1);
      const auto& d2 = dist->at<float>(j2, i2);
      const auto d_diff = std::abs(d1 - d2);

      if (d_diff >= 1.f) {
        return 1.f + std::min(d1, d2);
      }

      float d = 2.f - d_diff * d_diff;
      if (d > 0.f) {
        float r = std::sqrt(d);
        float s = (d1 + d2 - r) * 0.5f;
        if (d1 <= s && d2 <= s) {
          return s;
        }
        s += r;
        if (d1 <= s && d2 <= s) {
          return s;
        }
        // Failure case
        return std::numeric_limits<float>::max();
      }
    }

    if (flags.at<unsigned char>(j1, i1) != UNKNOWN) {
      return 1.f + dist->at<float>(j1, i1);
    }

    if (flags.at<unsigned char>(j2, i2) != UNKNOWN) {
      return 1.f + dist->at<float>(j2, i2);
    }

    return std::numeric_limits<float>::max();
  };

  while (!narrow_band.empty()) {
    const auto head = narrow_band.top();
    narrow_band.pop();
    flags.at<unsigned char>(head.y, head.x) = KNOWN;

    int i = head.x;
    int j = head.y;
    std::vector<std::pair<int, int>> neighbors_4;
    if (0 < i) {
      neighbors_4.push_back({i - 1, j});
    }
    if (i < mask.cols - 1) {
      neighbors_4.push_back({i + 1, j});
    }
    if (0 < j) {
      neighbors_4.push_back({i, j - 1});
    }
    if (j < mask.rows - 1) {
      neighbors_4.push_back({i, j + 1});
    }
    for (auto& neighbor : neighbors_4) {
      auto [x, y] = neighbor;

      auto& f = flags.at<unsigned char>(y, x);
      if (f != UNKNOWN) {
        continue;
      }

      auto& d = dist->at<float>(y, x);
      d = std::min({solve_eikonal(x - 1, y, x, y - 1),
                    solve_eikonal(x + 1, y, x, y - 1),
                    solve_eikonal(x - 1, y, x, y + 1),
                    solve_eikonal(x + 1, y, x, y + 1)});
      if (std::numeric_limits<float>::max() * 0.1f < d) {
        d = illegal_val;
      }

      f = BAND;
      narrow_band.push(FrontPix(d, x, y));
    }
  }
}

void Inpaint(const Image1b& mask, Image3b& color, float inpaint_radius,
             InpaintMethod method) {
  if (method == InpaintMethod::NAIVE) {
    return InpaintNaive(mask, color, static_cast<int>(inpaint_radius));
  } else if (method == InpaintMethod::TELEA) {
  }
}

}  // namespace ugu
