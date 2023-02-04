#version 330
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 vFragPos[];
in vec3 vViewPos[];
in vec2 vTexCoords[];
in vec3 vNormal[];
in vec3 vWldNormal[];
in vec3 vVertexColor[];
in vec3 vVertexId[];

out vec3 fragPos;
out vec3 viewPos;
out vec2 texCoords;
out vec3 normal;
out vec3 wldNormal;
out vec3 vertexColor;
out vec3 vertexId;
out vec2 bary;

void main() {
  for (int i = 0; i < gl_in.length(); ++i) {
    gl_Position = gl_in[i].gl_Position;
    gl_PrimitiveID = gl_PrimitiveIDIn;
    // fid = gl_PrimitiveIDIn;
    fragPos = vFragPos[i];
    viewPos = vViewPos[i];
    texCoords = vTexCoords[i];
    normal = vNormal[i];
    wldNormal = vWldNormal[i];
    vertexColor = vVertexColor[i];
    vertexId = vVertexId[i];
    if (i == 0) {
      bary = vec2(0.0, 0.0);
    } else if (i == 1) {
      bary = vec2(1.0, 0.0);
    } else {
      bary = vec2(0.0, 1.0);
    }
    EmitVertex();
  }
  EndPrimitive();
}
