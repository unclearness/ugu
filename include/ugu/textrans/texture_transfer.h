/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#pragma once

#ifdef UGU_USE_NANOFLANN

#include "ugu/image.h"
#include "ugu/mesh.h"

namespace ugu {

struct TexTransNoCorrespOutput {
  ugu::Image3f dst_tex;
  ugu::Image1b dst_mask;
  ugu::Image1i nn_fid_tex;
  ugu::Image3f nn_pos_tex;
  ugu::Image3f nn_bary_tex;
};

bool TexTransNoCorresp(const ugu::Image3f& src_tex,
                       const std::vector<Eigen::Vector2f>& src_uvs,
                       const std::vector<Eigen::Vector3i>& src_uv_faces,
                       const std::vector<Eigen::Vector3f>& src_verts,
                       const std::vector<Eigen::Vector3i>& src_verts_faces,
                       const std::vector<Eigen::Vector2f>& dst_uvs,
                       const std::vector<Eigen::Vector3i>& dst_uv_faces,
                       const std::vector<Eigen::Vector3f>& dst_verts,
                       const std::vector<Eigen::Vector3i>& dst_vert_faces,
                       int32_t dst_tex_h, int32_t dst_tex_w,
                       TexTransNoCorrespOutput& output, int32_t nn_num = 10);

bool TexTransNoCorresp(const ugu::Image3f& src_tex, const ugu::Mesh& src_mesh,
                       const ugu::Mesh& dst_mesh, int32_t dst_tex_h,
                       int32_t dst_tex_w, TexTransNoCorrespOutput& output,
                       int32_t nn_num = 10);

}  // namespace ugu

#endif