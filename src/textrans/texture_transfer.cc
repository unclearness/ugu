/*
 * Copyright (C) 2021, unclearness
 * All rights reserved.
 */

#include "ugu/textrans/texture_transfer.h"

#include "ugu/correspondence/correspondence_finder.h"
#include "ugu/util/math_util.h"
#include "ugu/util/raster_util.h"

namespace ugu {

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
                       TexTransNoCorrespOutput& output, int32_t interp,
                       int32_t nn_num) {
  if (interp != InterpolationFlags::INTER_LINEAR &&
      interp != InterpolationFlags::INTER_NEAREST) {
    ugu::LOGE("interp is not supported\n");
    return false;
  }

  if (nn_num < 1) {
    ugu::LOGE("nn_num must be larger than 1\n");
    return false;
  }

  output.dst_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.dst_mask = ugu::Image1b::zeros(dst_tex_h, dst_tex_w);
  output.nn_pos_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.nn_bary_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);
  output.nn_fid_tex = ugu::Image1i::zeros(dst_tex_h, dst_tex_w);
  output.srcpos_tex = ugu::Image3f::zeros(dst_tex_h, dst_tex_w);

  float src_w = static_cast<float>(src_tex.cols);
  float src_h = static_cast<float>(src_tex.rows);

  CorrespFinderPtr corresp_finder = KDTreeCorrespFinder::Create(nn_num);
  corresp_finder->Init(src_verts, src_verts_faces);

  assert(dst_uv_faces.size() == dst_vert_faces.size());
#pragma omp parallel for
  for (int64_t df_idx = 0; df_idx < static_cast<int64_t>(dst_uv_faces.size());
       df_idx++) {
    const auto& duvface = dst_uv_faces[df_idx];
    const auto& dvface = dst_vert_faces[df_idx];

    // Get bounding box in dst tex
    const auto& duv0 = dst_uvs[duvface[0]];
    const auto& duv1 = dst_uvs[duvface[1]];
    const auto& duv2 = dst_uvs[duvface[2]];
    const auto& dv0 = dst_verts[dvface[0]];
    const auto& dv1 = dst_verts[dvface[1]];
    const auto& dv2 = dst_verts[dvface[2]];
    const auto bb_min_x =
        U2X(std::max({0.f, std::min({duv0[0], duv1[0], duv2[0]})}), dst_tex_w);
    const auto bb_max_x = U2X(std::min({static_cast<float>(dst_tex_w - 1),
                                        std::max({duv0[0], duv1[0], duv2[0]})}),
                              dst_tex_w);

    // Be careful how to get min/max of y. It is an inversion of v
    const auto bb_min_v = std::min({duv0[1], duv1[1], duv2[1]});
    const auto bb_max_v = std::max({duv0[1], duv1[1], duv2[1]});
    const auto bb_min_y = std::max(0.f, V2Y(bb_max_v, dst_tex_h));
    const auto bb_max_y =
        std::min(static_cast<float>(dst_tex_h - 1), V2Y(bb_min_v, dst_tex_h));
    // pixel-wise loop for the bb in dst tex
    float area = ugu::EdgeFunction(duv0, duv1, duv2);
    float inv_area = 1.f / area;

    for (int32_t bb_y = static_cast<int32_t>(bb_min_y);
         bb_y <= static_cast<int32_t>(std::ceil(bb_max_y)); bb_y++) {
      for (int32_t bb_x = static_cast<int32_t>(bb_min_x);
           bb_x <= static_cast<int32_t>(std::ceil(bb_max_x)); bb_x++) {
        Eigen::Vector2f pix_uv(X2U(static_cast<float>(bb_x), dst_tex_w),
                               Y2V(static_cast<float>(bb_y), dst_tex_h));

        float w0 = ugu::EdgeFunction(duv1, duv2, pix_uv) * inv_area;
        float w1 = ugu::EdgeFunction(duv2, duv0, pix_uv) * inv_area;
        float w2 = ugu::EdgeFunction(duv0, duv1, pix_uv) * inv_area;
        // Check if this pixel is on the dst triangle
        if (w0 < 0 || w1 < 0 || w2 < 0) {
          continue;
        }

        // Get corresponding position on the dst face
        Eigen::Vector3f dpos = w0 * dv0 + w1 * dv1 + w2 * dv2;

        // Get the closest src face info
        // auto [foot, min_signed_dist, min_dist, min_index, bary] =
        //   CalcClosestSurfaceInfo(tree, dpos, src_verts, src_verts_faces,
        //                          src_face_planes, nn_num);
        Corresp corresp = corresp_finder->Find(dpos, Eigen::Vector3f::Ones(),
                                               CorrespFinderMode::kMinDist);

        if (corresp.fid < 0) {
          ugu::LOGE("min_index is None %d %d\n", bb_y, bb_x);
          continue;
        }
        output.nn_fid_tex.at<int>(bb_y, bb_x) = corresp.fid;
        output.nn_pos_tex.at<ugu::Vec3f>(bb_y, bb_x) =
            ugu::Vec3f({corresp.p[0], corresp.p[1], corresp.p[2]});
        output.nn_bary_tex.at<ugu::Vec3f>(bb_y, bb_x) =
            ugu::Vec3f({corresp.uv[0], corresp.uv[1],
                        (1.f - corresp.uv[0] - corresp.uv[1])});
        const auto& suvface = src_uv_faces[corresp.fid];
        const auto& suv0 = src_uvs[suvface[0]];
        const auto& suv1 = src_uvs[suvface[1]];
        const auto& suv2 = src_uvs[suvface[2]];

        Eigen::Vector2f suv = corresp.uv[0] * suv0 + corresp.uv[1] * suv1 +
                              (1.f - corresp.uv[0] - corresp.uv[1]) * suv2;

        // Calc pixel pos in src tex
        float sx = std::clamp(U2X(suv[0], static_cast<int32_t>(src_w)), 0.f,
                              src_w - 1.f - 0.001f);
        float sy = std::clamp(V2Y(suv[1], static_cast<int32_t>(src_h)), 0.f,
                              src_h - 1.f - 0.001f);

        ugu::Vec3f& srcpos = output.srcpos_tex.at<ugu::Vec3f>(bb_y, bb_x);
        srcpos[0] = sx;
        srcpos[1] = sy;

        // Fetch and copy to dst tex
        ugu::Vec3f& src_color = output.dst_tex.at<ugu::Vec3f>(bb_y, bb_x);
        if (interp == InterpolationFlags::INTER_LINEAR) {
          src_color = ugu::BilinearInterpolation(sx, sy, src_tex);
        } else if (interp == InterpolationFlags::INTER_NEAREST) {
          src_color =
              src_tex.at<ugu::Vec3f>(static_cast<int32_t>(std::round(sy)),
                                     static_cast<int32_t>(std::round(sx)));
        } else {
          ugu::LOGE("interp is not supported\n");
        }

        output.dst_mask.at<uint8_t>(bb_y, bb_x) = 255;
      }
    }
  }

  return true;
}

bool TexTransNoCorresp(const ugu::Image3f& src_tex, const ugu::Mesh& src_mesh,
                       const ugu::Mesh& dst_mesh, int32_t dst_tex_h,
                       int32_t dst_tex_w, TexTransNoCorrespOutput& output,
                       int32_t interp, int32_t nn_num) {
  return TexTransNoCorresp(src_tex, src_mesh.uv(), src_mesh.uv_indices(),
                           src_mesh.vertices(), src_mesh.vertex_indices(),
                           dst_mesh.uv(), dst_mesh.uv_indices(),
                           dst_mesh.vertices(), dst_mesh.vertex_indices(),
                           dst_tex_h, dst_tex_w, output, interp, nn_num);
}

}  // namespace ugu
