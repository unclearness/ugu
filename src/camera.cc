/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "ugu/camera.h"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
std::vector<std::string> Split(const std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}
}  // namespace

namespace ugu {

void WriteTumFormat(const std::vector<Eigen::Affine3d>& poses,
                    const std::string& path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out);

  ofs << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < poses.size(); i++) {
    const Eigen::Quaterniond q(poses[i].rotation());
    const Eigen::Vector3d t = poses[i].translation();
    ofs << zfill(i) << " " << t[0] << " " << t[1] << " " << t[2] << " " << q.x()
        << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
}

bool LoadTumFormat(const std::string& path,
                   std::vector<std::pair<int, Eigen::Affine3d>>* poses) {
  poses->clear();

  std::ifstream ifs(path);

  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = Split(line, ' ');
    if (splited.size() != 8) {
      LOGE("wrong tum format\n");
      return false;
    }

    std::pair<int, Eigen::Affine3d> pose;
    pose.first = std::atoi(splited[0].c_str());

    Eigen::Translation3d t;
    t.x() = std::atof(splited[1].c_str());
    t.y() = std::atof(splited[2].c_str());
    t.z() = std::atof(splited[3].c_str());

    Eigen::Quaterniond q;
    q.x() = std::atof(splited[4].c_str());
    q.y() = std::atof(splited[5].c_str());
    q.z() = std::atof(splited[6].c_str());
    q.w() = std::atof(splited[7].c_str());

    pose.second = t * q;

    poses->push_back(pose);
  }

  return true;
}

bool LoadTumFormat(const std::string& path,
                   std::vector<Eigen::Affine3d>* poses) {
  std::vector<std::pair<int, Eigen::Affine3d>> tmp_poses;
  bool ret = LoadTumFormat(path, &tmp_poses);
  if (!ret) {
    return false;
  }

  poses->clear();
  for (const auto& pose_pair : tmp_poses) {
    poses->push_back(pose_pair.second);
  }

  return true;
}

std::tuple<Eigen::Vector3f, std::vector<double>> FindLineCrossingPoint(
    const std::vector<Line3d>& lines) {
  // http://tercel-sakuragaoka.blogspot.com/2012/01/vs.html
  std::vector<double> errors;
  Eigen::Vector3f p;

  Eigen::MatrixXd lhs = lines.size() * Eigen::MatrixXd::Identity(
                                           lines.size() + 3, lines.size() + 3);
  for (auto i = 0; i < lines.size(); i++) {
    lhs(i, i) = lines[i].d.dot(lines[i].d);
  }

  for (auto i = 0; i < lines.size(); i++) {
    for (auto j = lines.size(); j < lines.size() + 3; j++) {
      lhs(i, j) = -lines[i].d[j - lines.size()];
    }
  }

  for (auto j = 0; j < lines.size(); j++) {
    for (auto i = lines.size(); i < lines.size() + 3; i++) {
      lhs(i, j) = -lines[j].d[i - lines.size()];
    }
  }

  Eigen::VectorXd rhs = Eigen::VectorXd(lines.size() + 3);
  for (auto i = 0; i < lines.size(); i++) {
    rhs(i) = -lines[i].d.dot(lines[i].a);
  }
  for (auto i = lines.size(); i < lines.size() + 3; i++) {
    double sum = 0.0;
    for (auto j = 0; j < lines.size(); j++) {
      sum += lines[j].a[i - lines.size()];
    }
    rhs(i) = sum;
  }

  Eigen::VectorXd ans =
      lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);

  p.x() = ans.cast<float>()[lines.size()];
  p.y() = ans.cast<float>()[lines.size() + 1];
  p.z() = ans.cast<float>()[lines.size() + 2];

  for (auto i = 0; i < lines.size(); i++) {
    auto a = (p.cast<double>() - lines[i].a);
    auto perpendiculer = lines[i].a + a.dot(lines[i].d) * lines[i].d;
    auto error = (p.cast<double>() - perpendiculer).norm();

    errors.push_back(error);
  }

  return {p, errors};
}

}  // namespace ugu
