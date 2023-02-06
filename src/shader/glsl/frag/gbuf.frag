#version 330
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
layout(location = 3) out vec4 gId;
layout(location = 4) out vec4 gFace;

in vec3 fragPos;
in vec3 viewPos;
in vec2 texCoords;
in vec3 normal;
in vec3 wldNormal;
in vec3 vertexColor;
in vec3 vertexId;
in vec2 bary;
in vec3 dist;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular1;

void main() {
  // store the fragment position vector in the first gbuffer texture
  gPosition = fragPos;
  // also store the per-fragment normals into the gbuffer
  gNormal = normalize(normal);
  // and the diffuse per-fragment color
  gAlbedoSpec.rgb = texture(texture_diffuse1, texCoords).rgb;
  // store specular intensity in gAlbedoSpec's alpha component
  gAlbedoSpec.a = texture(texture_specular1, texCoords).r;

  gId.x = float(gl_PrimitiveID + 1);  // vertedId.x;
  gId.y = vertexId.y;

  gFace.xy = bary;
  gFace.zw = texCoords;

  vec3 dist_vec = dist;
  float d = min(dist_vec[0], min(dist_vec[1], dist_vec[2]));
  float I = exp2(-2.0 * d * d);
  //  Use specular for wire intensity
  gAlbedoSpec.a = clamp(I, 0.0, 1.0);
}