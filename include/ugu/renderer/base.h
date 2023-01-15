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
  //Image1f depth_cam;
  Image3b shaded;
};

}  // namespace ugu
