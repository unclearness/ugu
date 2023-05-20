/*
 * Copyright (C) 2022, unclearness
 * All rights reserved.
 */

#include "ugu/point.h"

#ifdef UGU_USE_JSON

#ifdef _WIN32
#pragma warning(push, 0)
#endif
#include "nlohmann/json.hpp"
#ifdef _WIN32
#pragma warning(pop)
#endif

using namespace nlohmann;

#endif

namespace {
using namespace ugu;

#ifdef UGU_USE_JSON
std::vector<PointOnFace> LoadPointsPointOnTriangle(
    const std::string& json_path) {
  nlohmann::json j;
  std::ifstream ifs(json_path);
  ifs >> j;
  std::vector<PointOnFace> pofs;
  for (const auto& fid_uv : j) {
    int fid = fid_uv[0].get<int>();
    float u = fid_uv[1].get<float>();
    float v = fid_uv[2].get<float>();
    PointOnFace pof;
    pof.fid = fid;
    pof.u = u;
    pof.v = v;
    pofs.push_back(pof);
  }
  return pofs;
}
std::vector<PointOnFace> LoadPointsNamedPointOnTriangle(
    const std::string& json_path) {
  nlohmann::json j;
  std::ifstream ifs(json_path);
  ifs >> j;
  std::vector<PointOnFace> pofs;

  for (const auto& kv : j.items()) {
    auto name = kv.key();
    auto fid_uv = kv.value();
    int fid = fid_uv[0].get<int>();
    float u = fid_uv[1].get<float>();
    float v = fid_uv[2].get<float>();
    PointOnFace pof;
    pof.name = name;
    pof.fid = fid;
    pof.u = u;
    pof.v = v;
    pofs.push_back(pof);
  }

  return pofs;
}

std::vector<PointOnFace> LoadPointsTheedPoint(const std::string& json_path) {
  nlohmann::json j;
  std::ifstream ifs(json_path);
  ifs >> j;
  std::vector<PointOnFace> pofs;
  for (const auto& xyz : j) {
    PointOnFace pof;
    pof.pos.x() = xyz["x"].get<float>();
    pof.pos.y() = xyz["y"].get<float>();
    pof.pos.z() = xyz["z"].get<float>();
    pofs.push_back(pof);
  }
  return pofs;
}

void WritePointsPointOnTriangle(const std::string& json_path,
                                const std::vector<PointOnFace>& points) {
  json j;

  j = json::array();
  for (const auto& p : points) {
    json jj;
    jj.push_back(p.fid);
    jj.push_back(p.u);
    jj.push_back(p.v);
    j.push_back(jj);
  }
  std::ofstream ofs(json_path);
  ofs << j;
}

void WritePointsNamedPointOnTriangle(const std::string& json_path,
                                     const std::vector<PointOnFace>& points) {
  json j;

  j = json::object();
  for (const auto& p : points) {
    json jj;
    jj.push_back(p.fid);
    jj.push_back(p.u);
    jj.push_back(p.v);
    j[p.name] = jj;
  }
  std::ofstream ofs(json_path);
  ofs << j;
}

void WritePointsTheedPoint(const std::string& json_path,
                           const std::vector<PointOnFace>& points) {
  json j;
  j = json::array();
  for (const auto& p : points) {
    json jj;
    jj["x"] = p.pos.x();
    jj["y"] = p.pos.y();
    jj["z"] = p.pos.z();
    j.push_back(jj);
  }
  std::ofstream ofs(json_path);
  ofs << j;
}

#else
std::vector<PointOnFace> LoadPointsPointOnTriangle(
    const std::string& json_path) {
  (void)json_path;
  LOGE("Not implemented with this configuration\n");
  return {};
}
std::vector<PointOnFace> LoadPointsNamedPointOnTriangle(
    const std::string& json_path) {
  (void)json_path;
  LOGE("Not implemented with this configuration\n");
  return {};
}
std::vector<PointOnFace> LoadPointsTheedPoint(const std::string& json_path) {
  (void)json_path;
  LOGE("Not implemented with this configuration\n");
  return {};
}

void WritePointsPointOnTriangle(const std::string& json_path,
                                const std::vector<PointOnFace>& points) {
  (void)json_path, points;
  LOGE("Not implemented with this configuration\n");
}

void WritePointsNamedPointOnTriangle(const std::string& json_path,
                                     const std::vector<PointOnFace>& points) {
  (void)json_path, points;
  LOGE("Not implemented with this configuration\n");
}

void WritePointsTheedPoint(const std::string& json_path,
                           const std::vector<PointOnFace>& points) {
  (void)json_path, points;
  LOGE("Not implemented with this configuration\n");
}

#endif
}  // namespace

namespace ugu {

std::vector<PointOnFace> LoadPoints(const std::string& json_path,
                                    const PointOnFaceType& type) {
  if (type == PointOnFaceType::POINT_ON_TRIANGLE) {
    return LoadPointsPointOnTriangle(json_path);
  } else if (type == PointOnFaceType::NAMED_POINT_ON_TRIANGLE) {
    return LoadPointsNamedPointOnTriangle(json_path);
  } else if (type == PointOnFaceType::THREED_POINT) {
    return LoadPointsTheedPoint(json_path);
  }

  return {};
}

void WritePoints(const std::string& json_path,
                 const std::vector<PointOnFace>& points,
                 const PointOnFaceType& type) {
  if (type == PointOnFaceType::POINT_ON_TRIANGLE) {
    return WritePointsPointOnTriangle(json_path, points);
  } else if (type == PointOnFaceType::NAMED_POINT_ON_TRIANGLE) {
    return WritePointsNamedPointOnTriangle(json_path, points);
  } else if (type == PointOnFaceType::THREED_POINT) {
    return WritePointsTheedPoint(json_path, points);
  }
}

}  // namespace ugu
