#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedoSpec;
uniform sampler2D gId;
uniform sampler2D gFace;

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
  FragColor = vec4((Id.y * 3) * 0.2, 0.5, 0.6, 1.0);
  //FragColor = vec4(lighting, 1.0);
}