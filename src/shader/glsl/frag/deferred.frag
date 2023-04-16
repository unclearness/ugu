#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gId;
uniform sampler2D gFace;

uniform bool showWire;
uniform vec3 wireCol;
uniform float nearZ;
uniform float farZ;
uniform vec3 bkgCol;

const int N_GEOMS = 4;
const int N_POSITIONS = 128;
// uniform vec3 selectedPositions[N_POSITIONS];
uniform vec3 selectedPositions[N_POSITIONS * N_GEOMS];
uniform float selectedPosDepthTh;
uniform vec2 viewportOffset;
uniform vec3 selectedPosColors[N_GEOMS];

struct Light {
  vec3 Position;
  vec3 Color;

  float Linear;
  float Quadratic;
};
const int NR_LIGHTS = 32;
uniform Light lights[NR_LIGHTS];
uniform vec3 viewPos;

void main() {
  // retrieve data from gbuffer
  vec3 FragPos = texture(gPosition, TexCoords).rgb;
  vec3 Normal = texture(gNormal, TexCoords).rgb;
  vec3 Diffuse = texture(gAlbedoSpec, TexCoords).rgb;
  float Specular = texture(gAlbedoSpec, TexCoords).a;
  vec4 Face = texture(gFace, TexCoords);
  vec4 Id = texture(gId, TexCoords);

  // then calculate lighting as usual
  vec3 lighting = Diffuse * 0.1;  // hard-coded ambient component
  vec3 viewDir = normalize(viewPos - FragPos);
  for (int i = 0; i < NR_LIGHTS; ++i) {
    // diffuse
    vec3 lightDir = normalize(lights[i].Position - FragPos);
    vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Diffuse * lights[i].Color;
    // specular
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(Normal, halfwayDir), 0.0), 16.0);
    vec3 specular = lights[i].Color * spec * Specular;
    // attenuation
    float distance = length(lights[i].Position - FragPos);
    float attenuation = 1.0 / (1.0 + lights[i].Linear * distance +
                               lights[i].Quadratic * distance * distance);
    diffuse *= attenuation;
    specular *= attenuation;
    lighting += diffuse + specular;
  }
  // FragColor = vec4((Id.y * 3) * 0.2, 0.5, 0.6, 1.0);
  vec4 wireCol4 = vec4(wireCol, 1.0);
  float wire = mix(0.0, Specular, showWire);
  float depth = Id.z;
  // FragColor = vec4(Specular, Specular, Specular, 1.0);

  vec3 surface_col = vec3(1.0, 1.0, 1.0);
  float ratio = 0.7;
  float scale = dot(viewDir, Normal);
  if (scale > 0.0) {
    scale = scale * ratio + (1.0 - ratio);
  } else {
    scale = (scale + 1.0) * (1.0 - ratio);
  }
  //(dot(viewDir, Normal) + 1.0) * 0.5, * ratio + (1.0 - ratio);
  Diffuse = Diffuse * surface_col * scale;

  FragColor = vec4(Diffuse, 1.0) * (1.0 - wire) + wire * wireCol4;
  bool is_frg = nearZ < depth && depth < farZ;
  FragColor = mix(vec4(bkgCol, 1.0), FragColor, vec4(is_frg));

  int geoid = int(round(Id.y - 1));
  vec3 selectPosColor = selectedPosColors[geoid];
  const float SELECT_COLOR_RADIUS = 10;
  for (int i = 0; i < N_POSITIONS; ++i) {
    // Ignore defualt [0, 0]
    vec3 selected_pos = selectedPositions[geoid * N_POSITIONS + i];
    if (selected_pos.x < 1.0 || selected_pos.y < 1.0) {
      continue;
    }
    // Handle occulsion by depth check
    if (is_frg &&
        selected_pos.z - depth > selectedPosDepthTh) {
      continue;
    }
    vec2 posInBuf = gl_FragCoord.xy - viewportOffset;
    float dist = distance(posInBuf, selected_pos.xy);
    if (dist <= SELECT_COLOR_RADIUS) {
      FragColor = vec4(selectPosColor, 1.0);
    }
  }
}