#version 330
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;
layout(location = 3) in vec3 aVertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// out vec3 fragPos;
// out vec3 viewPos;
// out vec2 texCoords;
// out vec3 normal;
// out vec3 wldNormal;
// out vec3 vertexColor;

// void main() {
//   vec4 worldPos = model * vec4(aPos, 1.0);
//   vec4 view_Pos = view * worldPos;
//   fragPos = worldPos.xyz;
//   viewPos = view_Pos.xyz;
//   texCoords = aTexCoords;
//   vertexColor = aVertexColor;

//   mat3 normalMatrix = transpose(inverse(mat3(model)));
//   normal = normalMatrix * aNormal;
//   wldNormal = aNormal;

//   gl_Position = projection * view_Pos;
// }

out vec3 vFragPos;
out vec3 vViewPos;
out vec2 vTexCoords;
out vec3 vNormal;
out vec3 vWldNormal;
out vec3 vVertexColor;

void main() {
  vec4 worldPos = model * vec4(aPos, 1.0);
  vec4 view_Pos = view * worldPos;
  vFragPos = worldPos.xyz;
  vViewPos = view_Pos.xyz;
  vTexCoords = aTexCoords;
  vVertexColor = aVertexColor;

  mat3 normalMatrix = transpose(inverse(mat3(model)));
  vNormal = normalMatrix * aNormal;
  vWldNormal = aNormal;

  gl_Position = projection * view_Pos;
}