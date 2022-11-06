/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 *
 */

#include "ugu/inpaint/inpaint.h"

#include <queue>

#include "ugu/util/image_util.h"
#include "ugu/image_proc.h"

namespace {

template <typename T>
void InpaintNaive(const ugu::Image1b& mask, ugu::Image<T>& color,
                  int inpaint_kernel_size) {
  using namespace ugu;

  int ks = std::max(3, inpaint_kernel_size);
  int hks = ks / 2;
  // int erode_kernel_size = hks * 2 + 1;

  auto blend_color = [&](T& blended_pix, const std::vector<T>& sources) {
    if (sources.empty()) {
      return;
    }
    for (int c = 0; c < 3; c++) {
      double ave_color = 0.0;
      for (const auto& s : sources) {
        ave_color += s[c];
      }
      ave_color /= sources.size();
      blended_pix[c] = ugu::saturate_cast<typename T::value_type>(ave_color);
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
  double minDist = std::numeric_limits<double>::max(),
         maxDist = std::numeric_limits<double>::lowest();
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
          std::vector<T> sources;
          for (int jj = -hks; jj <= hks; jj++) {
            for (int ii = -hks; ii <= hks; ii++) {
              if (ii == 0 && jj == 0) {
                continue;
              }
              if (not_visited_mask.at<unsigned char>(j + jj, i + ii) == 0) {
                sources.push_back(color.template at<T>(j + jj, i + ii));
              }
            }
          }

          if (sources.empty()) {
            ++it;
            continue;
          }

          // Make inpainted color
          auto& inpainted_pix = color.template at<T>(j, i);
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

    if (static_cast<size_t>(cur_dist) < dist_pix_table.size()) {
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

constexpr unsigned char UNKNOWN = 0;
constexpr unsigned char BAND = 1;
constexpr unsigned char KNOWN = 2;
// constexpr unsigned char INSIDE = 3; // Used for inpainting with
// modification

struct BandPix {
  float T = std::numeric_limits<float>::max();  // distanceto - boundary
  int x = 0;
  int y = 0;
  BandPix(){};
  ~BandPix(){};
  BandPix(float T_, int x_, int y_) {
    T = T_;
    x = x_;
    y = y_;
  }
};

auto band_pix_compare = [](const BandPix& l, const BandPix& r) {
  return l.T > r.T;
};
using InpaintHeap = std::priority_queue<BandPix, std::vector<BandPix>,
                                        decltype(band_pix_compare)>;

float SolveEikonal(int i1, int j1, int i2, int j2, const ugu::Image1b& flags,
                   const ugu::Image1f& dist) {
  if (flags.cols - 1 < i1 || flags.cols - 1 < i2 || flags.rows - 1 < j1 ||
      flags.rows - 1 < j2 || i1 < 0 || i2 < 0 || j1 < 0 || j2 < 0) {
    return std::numeric_limits<float>::max();
  }

  if (flags.at<unsigned char>(j1, i1) != UNKNOWN &&
      flags.at<unsigned char>(j2, i2) != UNKNOWN) {
    const auto& d1 = dist.at<float>(j1, i1);
    const auto& d2 = dist.at<float>(j2, i2);
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
    return 1.f + dist.at<float>(j1, i1);
  }

  if (flags.at<unsigned char>(j2, i2) != UNKNOWN) {
    return 1.f + dist.at<float>(j2, i2);
  }

  return std::numeric_limits<float>::max();
};

std::vector<std::pair<int, int>> GenNeighbors4(int i, int j,
                                               const ugu::Image1b& mask) {
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

  return neighbors_4;
}

void FmmInitCommon(const ugu::Image1b& mask, ugu::Image1b& flags,
                   ugu::Image1f& dist, InpaintHeap& narrow_band) {
  // Initialization
  for (int j = 0; j < mask.rows; j++) {
    for (int i = 0; i < mask.cols; i++) {
      if (mask.at<unsigned char>(j, i) == 0) {
        if (flags.at<unsigned char>(j, i) == UNKNOWN) {
          flags.at<unsigned char>(j, i) =
              KNOWN;  // outside of mask is known as 0
        }
        continue;
      }

      // Check 4-neightbors if it contancts non-zero pixel
      std::vector<std::pair<int, int>> neighbors_4 = GenNeighbors4(i, j, mask);
      for (auto& neighbor : neighbors_4) {
        auto [x, y] = neighbor;

        auto& f = flags.at<unsigned char>(y, x);
        if (f == BAND) {
          continue;
        }

        if (mask.at<unsigned char>(y, x) == 0) {
          narrow_band.push(BandPix(0.f, x, y));
          dist.at<float>(y, x) = 0.f;
          f = BAND;
        }
      }
    }
  }
}

template <typename T>
void InpaintTeleaPixel(int i, int j, const ugu::Image1b& mask,
                       ugu::Image<T>& color, const ugu::Image1b& flags,
                       const ugu::Image1f& dist, int inpaint_range) {
  (void)mask;
  ugu::Vec2f grad_T{0.f, 0.f};
  if (0 <= i - 1 && flags.at<unsigned char>(j, i - 1) == KNOWN &&
      i + 1 < flags.cols && flags.at<unsigned char>(j, i + 1) == KNOWN) {
    grad_T[0] = (dist.at<float>(j, i + 1) - dist.at<float>(j, i - 1)) * 0.5f;
  } else if (i + 1 < flags.cols && flags.at<unsigned char>(j, i + 1) == KNOWN) {
    grad_T[0] = dist.at<float>(j, i + 1) - dist.at<float>(j, i);
  } else if (0 <= i - 1 && flags.at<unsigned char>(j, i - 1) == KNOWN) {
    grad_T[0] = dist.at<float>(j, i) - dist.at<float>(j, i - 1);
  }

  if (0 <= j - 1 && flags.at<unsigned char>(j - 1, i) == KNOWN &&
      j + 1 < flags.rows && flags.at<unsigned char>(j + 1, i) == KNOWN) {
    grad_T[1] = (dist.at<float>(j + 1, i) - dist.at<float>(j - 1, i)) * 0.5f;
  } else if (j + 1 < flags.rows && flags.at<unsigned char>(j + 1, i) == KNOWN) {
    grad_T[1] = dist.at<float>(j + 1, i) - dist.at<float>(j, i);
  } else if (0 <= j - 1 && flags.at<unsigned char>(j - 1, i) == KNOWN) {
    grad_T[1] = dist.at<float>(j, i) - dist.at<float>(j - 1, i);
  }

  float grad_T_norm = std::sqrt(grad_T[0] * grad_T[0] + grad_T[1] * grad_T[1]);
  if (0.000001f < grad_T_norm) {
    grad_T[0] /= grad_T_norm;
    grad_T[1] /= grad_T_norm;
  }

  int min_x = std::max(0, i - inpaint_range);
  int max_x = std::min(color.cols - 1, i + inpaint_range);
  int min_y = std::max(0, j - inpaint_range);
  int max_y = std::min(color.rows - 1, j + inpaint_range);

  float inpaint_range_sq = static_cast<float>(inpaint_range * inpaint_range);

  constexpr double eps = 1.0e-20f;
  constexpr double d0_sq = 1.0;
  constexpr double T0 = 1.0;

  for (int c = 0; c < 3; c++) {
    double Jx = 0, Jy = 0;
    double Ia = 0;
    double w_sum = eps;
    for (int jj = min_y; jj <= max_y; jj++) {
      for (int ii = min_x; ii <= max_x; ii++) {
        if (flags.at<unsigned char>(jj, ii) != KNOWN) {
          continue;
        }
        float actual_r_sq =
            static_cast<float>((jj - j) * (jj - j) + (ii - i) * (ii - i));
        if (inpaint_range_sq < actual_r_sq) {
          continue;
        }

        ugu::Vec2f vec{0.f, 0.f};
        vec[0] = static_cast<float>(i) - static_cast<float>(ii);
        vec[1] = static_cast<float>(j) - static_cast<float>(jj);

        double pix_dist_sq =
            static_cast<double>(vec[0]) * static_cast<double>(vec[0]) +
            static_cast<double>(vec[1]) * static_cast<double>(vec[1]);
        double pix_dist = std::sqrt(pix_dist_sq);

        // The directional component  (dot product of relative pixel position
        // vector and normal direction)
        double direc_w =
            static_cast<double>(vec[0]) * static_cast<double>(grad_T[0]) +
            static_cast<double>(vec[1]) * static_cast<double>(grad_T[1]);
        direc_w /= (pix_dist + eps);

        direc_w = std::abs(direc_w) < eps ? eps : direc_w;

        // The geometric distance component (pixel position difference)
        double distance_w = d0_sq / (pix_dist_sq + eps);

        // The level set distance component (marched distance difference)
        double levelset_w =
            T0 /
            (1.0 + std::abs(dist.at<float>(j, i) - dist.at<float>(jj, ii)));

        double weight = std::abs(direc_w * distance_w * levelset_w);

        ugu::Vec2f grad_I{0.f, 0.f};
        if (0 <= ii - 1 && flags.at<unsigned char>(jj, ii - 1) == KNOWN &&
            ii + 1 < flags.cols &&
            flags.at<unsigned char>(jj, ii + 1) == KNOWN) {
          grad_I[0] = (color.template at<T>(jj, ii + 1)[c] -
                       color.template at<T>(jj, ii - 1)[c]) *
                      0.5f;
        } else if (ii + 1 < flags.cols &&
                   flags.at<unsigned char>(jj, ii + 1) == KNOWN) {
          grad_I[0] = static_cast<float>(color.template at<T>(jj, ii + 1)[c]) -
                      static_cast<float>(color.template at<T>(jj, ii)[c]);
        } else if (0 <= ii - 1 &&
                   flags.at<unsigned char>(jj, ii - 1) == KNOWN) {
          grad_I[0] = static_cast<float>(color.template at<T>(jj, ii)[c]) -
                      static_cast<float>(color.template at<T>(jj, ii - 1)[c]);
        }

        if (0 <= jj - 1 && flags.at<unsigned char>(jj - 1, ii) == KNOWN &&
            jj + 1 < flags.rows &&
            flags.at<unsigned char>(jj + 1, ii) == KNOWN) {
          grad_I[1] = (color.template at<T>(jj + 1, ii)[c] -
                       color.template at<T>(jj - 1, ii)[c]) *
                      0.5f;
        } else if (jj + 1 < flags.rows &&
                   flags.at<unsigned char>(jj + 1, ii) == KNOWN) {
          grad_I[1] = static_cast<float>(color.template at<T>(jj + 1, ii)[c]) -
                      static_cast<float>(color.template at<T>(jj, ii)[c]);
        } else if (0 <= jj - 1 &&
                   flags.at<unsigned char>(jj - 1, ii) == KNOWN) {
          grad_I[1] = static_cast<float>(color.template at<T>(jj, ii)[c]) -
                      static_cast<float>(color.template at<T>(jj - 1, ii)[c]);
        }

        Ia += weight * color.template at<T>(jj, ii)[c];
        Jx -= weight *
              (static_cast<double>(grad_I[0]) * static_cast<double>(vec[0]));
        Jy -= weight *
              (static_cast<double>(grad_I[1]) * static_cast<double>(vec[1]));
        w_sum += weight;
      }
    }

    double weighted_val =
        ((Ia / w_sum + (Jx + Jy) / (std::sqrt(Jx * Jx + Jy * Jy) + eps) + 0.5));
    color.template at<T>(j, i)[c] =
        ugu::saturate_cast<typename T::value_type>(weighted_val);
  }
}

template <typename T>
void InpaintTelea(const ugu::Image1b& mask, ugu::Image<T>& color,
                  float inpaint_radius) {
  inpaint_radius = std::max(inpaint_radius, 1.f);

  ugu::Image1b flags = ugu::Image1b::zeros(mask.rows, mask.cols);
  ugu::Image1f dist = ugu::Image1f::zeros(mask.rows, mask.cols);

  InpaintHeap narrow_band{band_pix_compare};

  // Initialization
  FmmInitCommon(mask, flags, dist, narrow_band);

  // Make inverted mask
  ugu::Image1b inv_mask;
  ugu::Not(mask, &inv_mask);
  // "
  // We first run the FMM outside the initial inpainting boundary ∂Ω and obtain
  // the distance field Tout.Since we use only those points closer to ∂Ω than ε,
  // we run the FMM outside ∂Ω only until we reach T > ε.This restricts the FMM
  // computations to a band of thickness ε around ∂Ω, thus speeding up the
  // process.
  // "
  ugu::Image1f out_dist;
  ugu::FastMarchingMethod(inv_mask, out_dist, 0.0f, 2 * inpaint_radius);
  for (int j = 0; j < mask.rows; j++) {
    for (int i = 0; i < mask.cols; i++) {
      const auto& out_d = out_dist.at<float>(j, i);
      if (out_d > 0.f) {
        dist.at<float>(j, i) = -out_d;
        flags.at<unsigned char>(j, i) = KNOWN;
      }
    }
  }

  constexpr float illegal_val = std::numeric_limits<float>::max();

  int inpaint_range = static_cast<int>(std::round(inpaint_radius));

  while (!narrow_band.empty()) {
    const auto head = narrow_band.top();
    narrow_band.pop();
    flags.at<unsigned char>(head.y, head.x) = KNOWN;

    int i = head.x;
    int j = head.y;
    std::vector<std::pair<int, int>> neighbors_4 = GenNeighbors4(i, j, mask);
    for (auto& neighbor : neighbors_4) {
      auto [x, y] = neighbor;

      auto& f = flags.at<unsigned char>(y, x);
      if (f != UNKNOWN) {
        continue;
      }

      auto& d = dist.at<float>(y, x);
      d = std::min({SolveEikonal(x - 1, y, x, y - 1, flags, dist),
                    SolveEikonal(x + 1, y, x, y - 1, flags, dist),
                    SolveEikonal(x - 1, y, x, y + 1, flags, dist),
                    SolveEikonal(x + 1, y, x, y + 1, flags, dist)});
      if (std::numeric_limits<float>::max() * 0.1f < d) {
        d = illegal_val;
      }

      // Inpaint process
      InpaintTeleaPixel(x, y, mask, color, flags, dist, inpaint_range);

      f = BAND;
      narrow_band.push(BandPix(d, x, y));
    }
  }
}

}  // namespace

namespace ugu {

void FastMarchingMethod(const Image1b& mask, Image1f& dist, float illegal_val,
                        float terminate_dist) {
  // Reference of FMM for inpainting
  // OpenCV implementation:
  // https://github.com/opencv/opencv/blob/master/modules/photo/src/inpaint.cpp
  // Author's implementation:
  // https://github.com/erich666/jgt-code/blob/master/Volume_09/Number_1/Telea2004/AFMM_Inpainting/fmm.cpp
  // A python implementation: https://github.com/olvb/pyheal

  Image1b flags = Image1b::zeros(mask.rows, mask.cols);

  if (mask.cols != dist.cols || mask.rows != dist.rows) {
    dist = Image1f::zeros(mask.rows, mask.cols);
  } else {
    dist.setTo(0.f);
  }

  InpaintHeap narrow_band{band_pix_compare};

  // Initialization
  FmmInitCommon(mask, flags, dist, narrow_band);

  float lastest_dist = 0.f;
  while (!narrow_band.empty()) {
    const auto head = narrow_band.top();
    narrow_band.pop();
    flags.at<unsigned char>(head.y, head.x) = KNOWN;

    int i = head.x;
    int j = head.y;
    std::vector<std::pair<int, int>> neighbors_4 = GenNeighbors4(i, j, mask);
    for (auto& neighbor : neighbors_4) {
      auto [x, y] = neighbor;

      auto& f = flags.at<unsigned char>(y, x);
      if (f != UNKNOWN) {
        continue;
      }

      auto& d = dist.at<float>(y, x);
      d = std::min({SolveEikonal(x - 1, y, x, y - 1, flags, dist),
                    SolveEikonal(x + 1, y, x, y - 1, flags, dist),
                    SolveEikonal(x - 1, y, x, y + 1, flags, dist),
                    SolveEikonal(x + 1, y, x, y + 1, flags, dist)});
      if (std::numeric_limits<float>::max() * 0.1f < d) {
        d = illegal_val;
      } else {
        lastest_dist = std::max(d, lastest_dist);
      }

      f = BAND;
      narrow_band.push(BandPix(d, x, y));
    }

    if (terminate_dist > 0.f && terminate_dist <= lastest_dist) {
      break;
    }
  }
}

void Inpaint(const Image1b& mask, Image3b& color, float inpaint_radius,
             InpaintMethod method) {
  if (method == InpaintMethod::NAIVE) {
    return InpaintNaive(mask, color, static_cast<int>(inpaint_radius));
  } else if (method == InpaintMethod::TELEA) {
    return InpaintTelea(mask, color, inpaint_radius);
  }
}

void Inpaint(const Image1b& mask, Image3f& color, float inpaint_radius,
             InpaintMethod method) {
  if (method == InpaintMethod::NAIVE) {
    return InpaintNaive(mask, color, static_cast<int>(inpaint_radius));
  } else if (method == InpaintMethod::TELEA) {
    ugu::LOGW(
        "InpaintMethod::TELEA is buggy for float input. Consider to use "
        "InpaintMethod::NAIVE.");
    return InpaintTelea(mask, color, inpaint_radius);
  }
}

}  // namespace ugu
