/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/renderer/gl/renderer.h"

#ifdef UGU_USE_GLFW
#include "glad/gl.h"
#endif

namespace ugu {

RendererGl::RendererGl() {}

RendererGl::~RendererGl() {}

bool RendererGl::Init() {
  if (m_cam == nullptr) {
    LOGE("camera has not been set\n");
    return false;
  }

  m_gbuf_shader.SetVertType(VertShaderType::GBUF);
  m_gbuf_shader.SetFragType(FragShaderType::GBUF);

  m_deferred_shader.SetVertType(VertShaderType::DEFERRED);
  m_deferred_shader.SetFragType(FragShaderType::DEFERRED);

  if (!m_gbuf_shader.Prepare() || !m_deferred_shader.Prepare()) {
    return false;
  }

  for (auto [mesh, trans] : m_nodes) {
    int model_loc = glGetUniformLocation(m_gbuf_shader.ID, "model");
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    m_node_locs[mesh] = model_loc;

    mesh->BindTextures();
    mesh->SetupMesh();
  }

  // configure g-buffer framebuffer
  // ------------------------------
  // unsigned int gBuffer;
  glGenFramebuffers(1, &gBuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
  // unsigned int gPosition, gNormal, gAlbedoSpec;
  //  position color buffer
  glGenTextures(1, &gPosition);
  glBindTexture(GL_TEXTURE_2D, gPosition);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_width, m_height, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         gPosition, 0);
  // normal color buffer
  glGenTextures(1, &gNormal);
  glBindTexture(GL_TEXTURE_2D, gNormal);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, m_width, m_height, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                         gNormal, 0);
  // color + specular color buffer
  glGenTextures(1, &gAlbedoSpec);
  glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D,
                         gAlbedoSpec, 0);
  // tell OpenGL which color attachments we'll use (of this framebuffer) for
  // rendering
  attachments[0] = GL_COLOR_ATTACHMENT0;
  attachments[1] = GL_COLOR_ATTACHMENT1;
  attachments[2] = GL_COLOR_ATTACHMENT2;

  glDrawBuffers(3, attachments);
  // create and attach depth buffer (renderbuffer)

  glGenRenderbuffers(1, &rboDepth);
  glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, m_width, m_height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, rboDepth);
  // finally check if framebuffer is complete
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "Framebuffer not complete!" << std::endl;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  Eigen::Matrix4f view_mat = m_cam->c2w().inverse().matrix().cast<float>();
  m_view_loc = glGetUniformLocation(m_gbuf_shader.ID, "view");
  glUniformMatrix4fv(m_view_loc, 1, GL_FALSE, view_mat.data());

  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  m_prj_loc = glGetUniformLocation(m_gbuf_shader.ID, "projection");
  glUniformMatrix4fv(m_prj_loc, 1, GL_FALSE, prj_mat.data());

  m_deferred_shader.Use();
  m_deferred_shader.SetInt("gPosition", 0);
  m_deferred_shader.SetInt("gNormal", 1);
  m_deferred_shader.SetInt("gAlbedoSpec", 2);

  return true;
}

bool RendererGl::Draw(double tic) {
  (void)tic;

  // GBuf


  
  glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_gbuf_shader.Use();

  Eigen::Matrix4f view_mat = m_cam->c2w().inverse().matrix().cast<float>();
  glUniformMatrix4fv(m_view_loc, 1, GL_FALSE, view_mat.data());
  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  glUniformMatrix4fv(m_prj_loc, 1, GL_FALSE, prj_mat.data());

  for (auto [mesh, trans] : m_nodes) {
    int model_loc = m_node_locs[mesh];
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    mesh->Draw(m_gbuf_shader);
  }
 glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Deferred

  #if 1
  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  m_deferred_shader.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gPosition);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, gNormal);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
#if 0
  // send light relevant uniforms
  for (unsigned int i = 0; i < lightPositions.size(); i++) {
    shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].Position",
                               lightPositions[i]);
    shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].Color",
                               lightColors[i]);
    // update attenuation parameters and calculate radius
    const float linear = 0.7f;
    const float quadratic = 1.8f;
    shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].Linear",
                                linear);
    shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].Quadratic",
                                quadratic);
  }
  shaderLightingPass.setVec3("viewPos", camera.Position);
#endif

#if 0
  for (auto [mesh, trans] : m_nodes) {
    int model_loc = m_node_locs[mesh];
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    mesh->Draw(m_gbuf_shader);
  }
#endif

      if (quadVAO == 0) {
    float quadVertices[] = {
        // positions        // texture Coords
        -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
        1.0f,  1.0f, 0.0f, 1.0f, 1.0f, 1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
    };
    // setup plane VAO
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          (void*)(3 * sizeof(float)));
  }
  glBindVertexArray(quadVAO);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
  #endif


  return true;
}

bool RendererGl::ReadGbuffer(GBuffer& buf) { return true; }

void RendererGl::SetCamera(const CameraPtr cam) { m_cam = cam; }
void RendererGl::SetMesh(RenderableMeshPtr mesh, const Eigen::Affine3f& trans) {
  m_nodes[mesh] = trans;
}
void RendererGl::ClearMesh() {
  m_nodes.clear();
  m_node_locs.clear();
}

void RendererGl::SetFragType(const FragShaderType& frag_type) {
  m_deferred_shader.frag_type = frag_type;
}

void RendererGl::SetNearFar(float near_z, float far_z) {
  m_near_z = near_z;
  m_far_z = far_z;
}

void RendererGl::SetSize(uint32_t width, uint32_t height) {
  m_width = width;
  m_height = height;
}

}  // namespace ugu
