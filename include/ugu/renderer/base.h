/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

namespace ugu {

// Diffuse color
enum class DiffuseColor {
  kNone = 0,     // Default white color
  kTexture = 1,  // From diffuse uv texture
  kVertex = 2    // From vertex color
};

// Normal used for shading
// Also returned as output normal
enum class ShadingNormal {
  kFace = 0,   // Face normal
  kVertex = 1  // Vertex normal. Maybe average of face normals
};

// Diffuse shading
// Light ray same to viewing ray is used for shading
enum class DiffuseShading {
  kNone = 0,        // No shading
  kLambertian = 1,  // Lambertian reflectance model
  kOrenNayar =
      2  // Simplified Oren-Nayar reflectatnce model described in wikipedia
         // https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
};

struct GBuffer {
  Image3b color;
  Image3f normal_cam;
  Image3f normal_wld;
  Image3f pos_cam;
  Image3f pos_wld;
  Image1f depth_01;
  Image1b stencil;
  Image1i face_id;
  Image3f uv;
  Image3f bary;
  Image1i geo_id;
  Image3b shaded;

  void Init(int w, int h) {
    color = Image3b::zeros(h, w);
    normal_cam = Image3f::zeros(h, w);
    normal_wld = Image3f::zeros(h, w);
    pos_cam = Image3f::zeros(h, w);
    pos_wld = Image3f::zeros(h, w);
    depth_01 = Image1f::zeros(h, w);
    stencil = Image1b::zeros(h, w);
    face_id = Image1i::zeros(h, w);
    uv = Image3f::zeros(h, w);
    bary = Image3f::zeros(h, w);
    geo_id = Image1i::zeros(h, w);
    shaded = Image3b::zeros(h, w);
  }

  void Reset() {
    color.setTo(0.0);
    normal_cam.setTo(0.0);
    normal_wld.setTo(0.0);
    pos_cam.setTo(0.0);
    pos_wld.setTo(0.0);
    depth_01.setTo(0.0);
    stencil.setTo(0.0);
    face_id.setTo(0.0);
    uv.setTo(0.0);
    bary.setTo(0.0);
    geo_id.setTo(0.0);
    shaded.setTo(0.0);
  }
};

}  // namespace ugu
