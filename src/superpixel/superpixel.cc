/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/superpixel/superpixel.h"

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "ugu/image_proc.h"
#include "ugu/rect.h"
#include "ugu/util/math_util.h"

#if defined(UGU_USE_OPENCV) && __has_include("opencv2/ximgproc.hpp")
#ifdef _WIN32
#pragma warning(push, UGU_OPENCV_WARNING_LEVEL)
#endif
#include "opencv2/ximgproc.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

using namespace ugu;

namespace {
struct ClusterNode {
  int id = -1;
  Vec3d mean;  // TODO: Remove outlier colors
  Vec3d stddev;
  Image1b mask;
  Rect bb;
  std::vector<std::shared_ptr<ClusterNode>> ancestors;
};

struct ClusterEdge {
  std::pair<int, int> ids;
  double color_diff = -1.0;
  double boundary_strength = -1.0;
};

#if 0
bool operator<(const ClusterEdge& l, const ClusterEdge& r) {
  return l.color_diff < r.color_diff;
};
bool operator>(const ClusterEdge& l, const ClusterEdge& r) {
  return l.color_diff > r.color_diff;
};
#endif

auto edge_greater_compare = [&](const ClusterEdge& l, const ClusterEdge& r) {
  return l.color_diff > r.color_diff;
};

// using ClusterHeap = std::priority_queue<ClusterEdge,
// std::vector<ClusterEdge>>;

auto MakeEdgePair(const int& a, const int& b) {
  if (a < b) {
    return std::make_pair(a, b);
  }
  return std::make_pair(b, a);
}

void InitializeGraph(
    const Image3b& img, const Image1i& labels, const int sp_num,
    std::unordered_map<int, std::shared_ptr<ClusterNode>>& nodes,
    std::vector<ClusterEdge>& edges,
    std::unordered_map<int, std::unordered_set<int>>& node_connections) {
  // Initialize graph
  for (int i = 0; i < sp_num; i++) {
    auto node = std::make_shared<ClusterNode>();
    node->id = i;
    node->mask = (labels == i);
    // cv::imshow("mask", node->mask);
    // cv::waitKey(1);
    node->bb = boundingRect(node->mask);
    meanStdDev(img, node->mean, node->stddev, node->mask);
    nodes.insert({i, node});
  }

  // edge_pair -> boundary strength
  std::map<std::pair<int, int>, std::vector<double>> edges_;

  auto test_pixels = [&](int x0, int y0, int x1, int y1) {
    const int& a = labels.at<int>(y0, x0);
    const int& b = labels.at<int>(y1, x1);
    if (a != b) {
      auto edge_candidate = MakeEdgePair(a, b);
      const Vec3b& a_col = img.at<Vec3b>(y0, x0);
      const Vec3b& b_col = img.at<Vec3b>(y1, x1);
      double strength = ugu::norm(a_col - b_col);
      if (edges_.find(edge_candidate) != edges_.end()) {
        edges_[edge_candidate].push_back(strength);
      } else {
        edges_[edge_candidate] = {strength};
      }
    }
  };

  // Check x-axis
  for (int y = 0; y < labels.rows; y++) {
    for (int x = 0; x < labels.cols - 1; x++) {
      test_pixels(x, y, x + 1, y);
    }
  }
  // Check y-axis
  for (int y = 0; y < labels.rows - 1; y++) {
    for (int x = 0; x < labels.cols; x++) {
      test_pixels(x, y, x, y + 1);
    }
  }

  for (const auto& edge_ : edges_) {
    ClusterEdge edge;
    edge.ids = edge_.first;
    const auto& strenghs = edge_.second;
    assert(!strenghs.empty());
    edge.boundary_strength =
        std::accumulate(strenghs.begin(), strenghs.end(), 0.0) /
        static_cast<double>(strenghs.size());
    edge.color_diff =
        norm(nodes[edge.ids.first]->mean - nodes[edge.ids.second]->mean);

    edges.push_back(edge);

#if 0
    if (node_connections.find(edge.ids.first) == node_connections.end()) {
      node_connections[edge.ids.first] = {edge.ids.second};
    } else {
      node_connections[edge.ids.first].insert(edge.ids.second);
    }
#endif
    node_connections[edge.ids.first].insert(edge.ids.second);
    node_connections[edge.ids.second].insert(edge.ids.first);
  }

  std::sort(edges.begin(), edges.end(), edge_greater_compare);
}

}  // namespace

namespace ugu {

void Slic(const ImageBase& img, Image1i& labels, Image1b& contour_mask,
          int& sp_num, int region_size, float ruler,
          int min_element_size_percent, int num_iterations) {
#if defined(UGU_USE_OPENCV) && __has_include("opencv2/ximgproc.hpp")
  auto slic = cv::ximgproc::createSuperpixelSLIC(img, cv::ximgproc::SLIC,
                                                 region_size, ruler);

  slic->iterate(num_iterations);

  slic->enforceLabelConnectivity(min_element_size_percent);

  slic->getLabels(labels);

  slic->getLabelContourMask(contour_mask, false);

  sp_num = slic->getNumberOfSuperpixels();
#else
  (void)img, (void)labels, (void)contour_mask, (void)region_size, (void)ruler,
      (void)min_element_size_percent, (void)num_iterations;

  LOGE("Not avairable with this configuration\n");
#endif
}

void SimilarColorClustering(const ImageBase& img, Image1i& labels,
                            int& labels_num, int region_size, float ruler,
                            int min_element_size_percent, int num_iterations,
                            size_t min_clusters, double max_color_diff,
                            double max_boundary_strengh) {
  /* H.J.Ku, H.Hat, J.H.Lee, D.Kang, J.Tompkin and M.H.Kim,
   * "Differentiable Appearance Acquisition from a Flash/No-flash RGB-D Pair,"
   * 2022 IEEE International Conference on Computational Photography (ICCP),
   * Pasadena, CA, USA, 2022, pp. 1-12, doi: 10.1109/ICCP54855.2022.9887646.
   * https://vclab.kaist.ac.kr/iccp2022/iccp2022-paper.pdf
   *
   * 5 METHOD: IMPLEMENTATION DETAILS
   * Material Clustering.
   *
   */

#if defined(UGU_USE_OPENCV) && __has_include("opencv2/ximgproc.hpp")
  // SLIC
  Image1b contour_mask;
  int sp_num;
  Slic(img, labels, contour_mask, sp_num, region_size, ruler,
       min_element_size_percent, num_iterations);

  // cv::imshow("hoge", img);
  // cv::waitKey(1);
  //  Initialize graph with SLIC
  std::unordered_map<int, std::shared_ptr<ClusterNode>> nodes;
  std::vector<ClusterEdge> edges;
  std::unordered_map<int, std::unordered_set<int>> node_connections;
  InitializeGraph(img, labels, sp_num, nodes, edges, node_connections);

  auto terminate_criteria = [&]() {
    if (nodes.size() < 2 || nodes.size() <= min_clusters) {
      return true;
    }

    const auto& e = edges.back();

    if (max_color_diff > 0.0 && e.color_diff >= max_color_diff) {
#if 0
      cv::imshow("m1", nodes[edges.back().ids.first]->mask);
      cv::imshow("m2", nodes[edges.back().ids.second]->mask);
      cv::Mat tmp1, tmp2;
      img.copyTo(tmp1, nodes[edges.back().ids.first]->mask);
      img.copyTo(tmp2, nodes[edges.back().ids.second]->mask);
      std::cout << nodes[edges.back().ids.first]->mean << std::endl;
      std::cout << nodes[edges.back().ids.second]->mean << std::endl;
      std::cout << norm(nodes[edges.back().ids.first]->mean -
                        nodes[edges.back().ids.second]->mean)
                << std::endl;
      cv::imshow("c1", tmp1);
      cv::imshow("c2", tmp2);
      cv::waitKey(1);
#endif
      return true;
    }

    if (max_boundary_strengh > 0.0 &&
        e.boundary_strength >= max_boundary_strengh) {
      return true;
    }

    return false;
  };

  size_t id_counter = nodes.size();  // Current max id = nodes.size() - 1
  while (!terminate_criteria()) {
    const auto& e = edges.back();
    edges.pop_back();

    const auto n0 = nodes.at(e.ids.first);
    const auto n1 = nodes.at(e.ids.second);

    auto merged_node = std::make_shared<ClusterNode>();
    // Update nodes
    {
      // Make merged node

      merged_node->id = static_cast<int>(id_counter++);
      merged_node->mask = Image1b::zeros(n0->mask.rows, n0->mask.cols);
      cv::bitwise_or(n0->mask, n1->mask, merged_node->mask);
      meanStdDev(img, merged_node->mean, merged_node->stddev,
                 merged_node->mask);
      merged_node->bb = n0->bb | n1->bb;  // Take union
      merged_node->ancestors.push_back(n0);
      merged_node->ancestors.push_back(n1);

      // Update labels
      labels.setTo(merged_node->id, merged_node->mask);

      // Add the merged node
      nodes[merged_node->id] = merged_node;
    }

    // std::cout << merged_node->id << std::endl;

    // Update edges
    {
      auto connected0 = node_connections.at(n0->id);
      auto connected1 = node_connections.at(n1->id);

      connected0.erase(n1->id);
      connected1.erase(n0->id);

      std::unordered_set<int> all_connected = connected0;
      for (const auto& c : connected1) {
        all_connected.insert(c);
      }

      auto test_pixel = [&](const Vec3b& now_col, int target_label, int x1,
                            int y1, std::vector<double>& boundary_strengths) {
        const int& l = labels.at<int>(y1, x1);
        if (l != target_label) {
          return;
        }
        const Vec3b& col = img.at<Vec3b>(y1, x1);
        boundary_strengths.push_back(norm(now_col - col));
      };

      const int offset = 2;
      Rect merged_enlarged_rect = merged_node->bb;
      merged_enlarged_rect.x -= offset;
      merged_enlarged_rect.y -= offset;
      merged_enlarged_rect.width += offset;
      merged_enlarged_rect.height += offset;

      auto add_edge = [&](const std::unordered_set<int> connected) {
        // Add new edge
        for (const auto& c : connected) {
          Rect enlarged_rect = nodes.at(c)->bb;
          enlarged_rect.x -= offset;
          enlarged_rect.y -= offset;
          enlarged_rect.width += offset;
          enlarged_rect.height += offset;

          Rect intersect_bb = merged_enlarged_rect & enlarged_rect;

          int y_min = std::max(0, intersect_bb.y - 1);
          int y_max =
              std::min(labels.rows - 1, intersect_bb.y + intersect_bb.height);
          int x_min = std::max(0, intersect_bb.x - 1);
          int x_max =
              std::min(labels.cols - 1, intersect_bb.x + intersect_bb.width);

          ClusterEdge merged_edge;
          merged_edge.ids = MakeEdgePair(merged_node->id, c);

          merged_edge.boundary_strength = 0.0;

          std::vector<double> boundary_strengths;
          for (int y = y_min; y < y_max - 1; y++) {
            for (int x = x_min; x < x_max - 1; x++) {
              const int& now_l = labels.at<int>(y, x);
              const Vec3b& now_col = img.at<Vec3b>(y, x);
              if (now_l == merged_node->id) {
                test_pixel(now_col, nodes[c]->id, x + 1, y, boundary_strengths);
                test_pixel(now_col, nodes[c]->id, x, y + 1, boundary_strengths);
              } else if (now_l == nodes[c]->id) {
                test_pixel(now_col, merged_node->id, x + 1, y,
                           boundary_strengths);
                test_pixel(now_col, merged_node->id, x, y + 1,
                           boundary_strengths);
              }
            }
          }

          if (!boundary_strengths.empty()) {
            merged_edge.boundary_strength =
                std::accumulate(boundary_strengths.begin(),
                                boundary_strengths.end(), 0.0) /
                static_cast<double>(boundary_strengths.size());
          }

          merged_edge.color_diff = norm(merged_node->mean - nodes[c]->mean);

          edges.push_back(merged_edge);

          node_connections[merged_edge.ids.first].insert(
              merged_edge.ids.second);
          node_connections[merged_edge.ids.second].insert(
              merged_edge.ids.first);
        }
      };

      add_edge(all_connected);

      // Remove edges
      std::set<std::pair<int, int>> to_remove_edges;
      for (const auto& c0 : connected0) {
        to_remove_edges.insert(MakeEdgePair(n0->id, c0));
        node_connections.at(c0).erase(n0->id);
      }
      for (const auto& c1 : connected1) {
        to_remove_edges.insert(MakeEdgePair(n1->id, c1));
        node_connections.at(c1).erase(n1->id);
      }

      to_remove_edges.insert(MakeEdgePair(n0->id, n1->id));
      node_connections.erase(n0->id);
      node_connections.erase(n1->id);

#if 0
      edges.erase(std::remove_if(edges.begin(), edges.end(),
                                 [&](const ClusterEdge& edge_) {
                                   for (const auto& e : to_remove_edges) {
                                     if (e.first == edge_.ids.first &&
                                         e.second == edge_.ids.second) {
                                       return true;
                                     }
                                   }
                                   return false;
                                 }),
                  edges.end());
#endif

      edges.erase(std::remove_if(edges.begin(), edges.end(),
                                 [&](const ClusterEdge& edge_) {
                                   // for (const auto& e : to_remove_edges) {
                                   if (edge_.ids.first == n0->id ||
                                       edge_.ids.first == n1->id ||
                                       edge_.ids.second == n0->id ||
                                       edge_.ids.second == n1->id) {
                                     return true;
                                   }
                                   //}
                                   return false;
                                 }),
                  edges.end());

      // for (const auto& r : to_remove_edges) {
      // std::cout << r.first << " -> " << r.second << std::endl;
      //}
    }

    // Erase ancestors
    // But still alive in memory because they are kept as ancestors
    nodes.erase(n0->id);
    nodes.erase(n1->id);

    // std::cout << n0->id << std::endl;
    // std::cout << n1->id << std::endl;

    // Release masks to reduce memory usage
    // n0->mask.release();
    // n1->mask.release();

    std::sort(edges.begin(), edges.end(), edge_greater_compare);
  }

  labels_num = static_cast<int>(nodes.size());
  int count = 0;
  ImageBase old_labels = labels.clone();
  labels.setTo(0.0);
  for (const auto& n : nodes) {
    int old_id = n.first;
    int new_id = count++;
    labels.setTo(new_id, old_labels == old_id);
  }

#else
  (void)img, (void)labels, (void)contour_mask, (void)region_size, (void)ruler,
      (void)min_element_size_percent, (void)num_iterations;

  LOGE("Not avairable with this configuration\n");
#endif
}

}  // namespace ugu
