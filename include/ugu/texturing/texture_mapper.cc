/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "texture_mapper.h"

namespace {

bool MakeTiledImage(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    ugu::Image3b* tiled, int x_tile_num, int y_tile_num) {
  // TODO:: different image size

  int org_w = keyframes[0]->color.cols;
  int org_h = keyframes[0]->color.rows;
  int tex_w = x_tile_num * keyframes[0]->color.cols;
  int tex_h = y_tile_num * keyframes[0]->color.rows;

  *tiled = ugu::Image3b::zeros(tex_h, tex_w);

  std::vector<std::array<int, 2>> kf_tile_pos_list(keyframes.size());
  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    auto& pos = kf_tile_pos_list[i];
    pos[0] = i % x_tile_num;
    pos[1] = i / x_tile_num;
    int tex_pos_x = pos[0] * org_w;
    int tex_pos_y = pos[1] * org_h;

    unsigned char* base_adr = tiled->data + (tex_pos_y * tex_w + tex_pos_x) * 3;

    // copy per line
    for (int j = 0; j < keyframes[0]->color.rows; j++) {
      std::memcpy(base_adr + tex_w * 3 * (j),
                  keyframes[i]->color.data + org_w * 3 * j, org_w * 3);
    }
  }

  return true;
}

bool GenerateSimpleTileUv(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option,
    const std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>&
        bestkfid2faceid,
    const std::vector<ugu::FaceInfoPerKeyframe>& faceid2bestkf) {
  // Make tiled image and get tile xy
  ugu::Image3b texture;

  // Make tiled image and get tile xy
  int x_tile_num =
      keyframes.size() < 5 ? static_cast<int>(keyframes.size()) : 5;
  int y_tile_num = static_cast<int>(keyframes.size() / x_tile_num) + 1;

  MakeTiledImage(keyframes, &texture, x_tile_num, y_tile_num);
  // ugu::imwrite("tile.png", texture);

  // Convert projected_tri to UV by tile xy
  std::vector<Eigen::Vector2f> uv;
  std::vector<Eigen::Vector3i> uv_indices;
  std::unordered_map<int, int> id2index;

  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    id2index.emplace(keyframes[i]->id, i);
  }

  int org_w = keyframes[0]->color.cols;
  int org_h = keyframes[0]->color.rows;
  int tex_w = x_tile_num * keyframes[0]->color.cols;
  int tex_h = y_tile_num * keyframes[0]->color.rows;

  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    // Get corresponding kf_id, index and projected_tri
    const auto& bestkf = faceid2bestkf[i];
    std::array<Eigen::Vector2f, 3> texture_tri, uv_tri;
    if (bestkf.kf_id < 0) {
      // face is not visible, set invalid value
      std::fill(uv_tri.begin(), uv_tri.end(), Eigen::Vector2f::Zero());

    } else {
      // Calc image position on tiled image
      int tile_pos_x = id2index[bestkf.kf_id] % x_tile_num;
      int tile_pos_y = id2index[bestkf.kf_id] / x_tile_num;
      int tex_pos_x = tile_pos_x * org_w;
      int tex_pos_y = tile_pos_y * org_h;
      for (int j = 0; j < 3; j++) {
        texture_tri[j].x() = bestkf.projected_tri[j].x() + tex_pos_x;
        texture_tri[j].y() = bestkf.projected_tri[j].y() + tex_pos_y;

        // Convert to 0-1 UV
        uv_tri[j].x() = texture_tri[j].x() / tex_w;
        uv_tri[j].y() = 1.0f - (texture_tri[j].y() / tex_h);
      }
    }
    uv.push_back(uv_tri[0]);
    uv.push_back(uv_tri[1]);
    uv.push_back(uv_tri[2]);

    int uv_size = static_cast<int>(uv.size());
    uv_indices.push_back(
        Eigen::Vector3i(uv_size - 3, uv_size - 2, uv_size - 1));
  }

  mesh->set_uv(uv);
  mesh->set_uv_indices(uv_indices);

  std::vector<ugu::ObjMaterial> materials(1);
  materials[0].name = "GenerateSimpleTileUv";
  materials[0].diffuse_tex = texture;
  // materials[0].diffuse_texname = "GenerateSimpleTileUv";
  // materials[0].diffuse_texpath = "material000.png";
  mesh->set_materials(materials);

  std::vector<int> material_ids(mesh->vertex_indices().size(), 0);
  mesh->set_material_ids(material_ids);

  return true;
}

bool SimpleTextureMapping(
    const std::vector<std::shared_ptr<ugu::Keyframe>>& keyframes,
    const ugu::VisibilityInfo& info, ugu::Mesh* mesh,
    const ugu::TextureMappingOption& option) {
  // Init
  std::unordered_map<int, std::vector<ugu::FaceInfoPerKeyframe>>
      bestkfid2faceid;
  for (int i = 0; i < static_cast<int>(keyframes.size()); i++) {
    bestkfid2faceid.emplace(keyframes[i]->id,
                            std::vector<ugu::FaceInfoPerKeyframe>());
  }
  std::vector<ugu::FaceInfoPerKeyframe> faceid2bestkf(
      info.face_info_list.size());

  std::function<ugu::FaceInfoPerKeyframe(const ugu::FaceInfo&)> get_best_kfid;
  if (option.criteria == ugu::ViewSelectionCriteria::kMinViewingAngle) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.min_viewing_angle_index];
    };
  } else if (option.criteria == ugu::ViewSelectionCriteria::kMinDistance) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.min_distance_index];
    };
  } else if (option.criteria == ugu::ViewSelectionCriteria::kMaxArea) {
    get_best_kfid = [](const ugu::FaceInfo& info) -> ugu::FaceInfoPerKeyframe {
      return info.visible_keyframes[info.max_area_index];
    };
  } else {
    LOGE("ViewSelectionCriteria %d is not implemented\n", option.criteria);
    return false;
  }

  // Select the best kf_id for each face in iterms of criteria
  for (int i = 0; i < static_cast<int>(info.face_info_list.size()); i++) {
    const auto& face_info = info.face_info_list[i];

    if (face_info.visible_keyframes.empty()) {
      faceid2bestkf[i].kf_id = -1;
      faceid2bestkf[i].face_id = i;
      continue;
    }

    // Get the best kf id and projected_tri
    ugu::FaceInfoPerKeyframe bestkf = get_best_kfid(face_info);
    bestkfid2faceid[bestkf.kf_id].push_back(bestkf);
    faceid2bestkf[i] = bestkf;
  }

  bool ret_uv_gen = false;
  if (option.uv_type == ugu::OutputUvType::kGenerateSimpleTile) {
    ret_uv_gen = GenerateSimpleTileUv(keyframes, info, mesh, option,
                                      bestkfid2faceid, faceid2bestkf);
  } else {
    LOGE("OutputUvType %d is not implemented\n", option.uv_type);
    return false;
  }

  return ret_uv_gen;
}

}  // namespace

namespace ugu {

bool TextureMapping(const std::vector<std::shared_ptr<Keyframe>>& keyframes,
                    const VisibilityInfo& info, Mesh* mesh,
                    const TextureMappingOption& option) {
  if (option.type == TextureMappingType::kSimpleProjection) {
    return SimpleTextureMapping(keyframes, info, mesh, option);
  }
  LOGE("TextureMappingType %d is not implemented", option.type);
  return false;
}

}  // namespace ugu
