#version 330
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;
layout(location = 3) out vec4 gFace;
layout(location = 4) out vec4 gGeo;


in vec3 fragPos;
in vec3 viewPos;
in vec2 texCoords;
in vec3 normal;
in vec3 wldNormal;
in vec3 vertexColor;
//flat in int fid;

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

  gFace.x = gl_PrimitiveID;
  gFace.yz = texCoords;
  gFace.w = 0.0;

  gGeo = vec4(0.5);

}