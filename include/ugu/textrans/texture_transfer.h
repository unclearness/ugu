/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANOFLANN

#include "ugu/image.h"
#include "ugu/mesh.h"

namespace ugu {

struct Corresp {
  int32_t fid = -1;
  Eigen::Vector3f uv = Eigen::Vector3f::Zero();
  Eigen::Vector3f p =
      Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
  float singed_dist = std::numeric_limits<float>::lowest();
  float abs_dist = std::numeric_limits<float>::max();
};

class CorrespFinder {
 public:
  virtual ~CorrespFinder() {}
  virtual bool Init(const std::vector<Eigen::Vector3f>& verts,
                    const std::vector<Eigen::Vector3i>& verts_faces) = 0;
  virtual Corresp Find(const Eigen::Vector3f& src_p,
                       const Eigen::Vector3f& src_n) const = 0;
};
using CorrespFinderPtr = std::shared_ptr<CorrespFinder>;

struct TexTransNoCorrespOutput {
  Image3f dst_tex;
  Image1b dst_mask;
  Image1i nn_fid_tex;
  Image3f nn_pos_tex;
  Image3f nn_bary_tex;
  Image3f srcpos_tex;
};

bool TexTransNoCorresp(const Image3f& src_tex,
                       const std::vector<Eigen::Vector2f>& src_uvs,
                       const std::vector<Eigen::Vector3i>& src_uv_faces,
                       const std::vector<Eigen::Vector3f>& src_verts,
                       const std::vector<Eigen::Vector3i>& src_verts_faces,
                       const std::vector<Eigen::Vector2f>& dst_uvs,
                       const std::vector<Eigen::Vector3i>& dst_uv_faces,
                       const std::vector<Eigen::Vector3f>& dst_verts,
                       const std::vector<Eigen::Vector3i>& dst_vert_faces,
                       int32_t dst_tex_h, int32_t dst_tex_w,
                       TexTransNoCorrespOutput& output,
                       int32_t interp = InterpolationFlags::INTER_LINEAR,
                       int32_t nn_num = 10);

bool TexTransNoCorresp(const Image3f& src_tex, const Mesh& src_mesh,
                       const Mesh& dst_mesh, int32_t dst_tex_h,
                       int32_t dst_tex_w, TexTransNoCorrespOutput& output,
                       int32_t interp = InterpolationFlags::INTER_LINEAR,
                       int32_t nn_num = 10);

}  // namespace ugu

#endif