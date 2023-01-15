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

  if (!m_shader.Prepare()) {
    return false;
  }

  for (auto [mesh, trans] : m_nodes) {
    int model_loc = glGetUniformLocation(m_shader.ID, "model");
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    m_node_locs[mesh] = model_loc;

    mesh->BindTextures();
    mesh->SetupMesh();
  }

  Eigen::Matrix4f view_mat = m_cam->c2w().inverse().matrix().cast<float>();
  m_view_loc = glGetUniformLocation(m_shader.ID, "view");
  glUniformMatrix4fv(m_view_loc, 1, GL_FALSE, view_mat.data());

  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  m_prj_loc = glGetUniformLocation(m_shader.ID, "projection");
  glUniformMatrix4fv(m_prj_loc, 1, GL_FALSE, prj_mat.data());

  return true;
}

bool RendererGl::Draw(double tic) {
  (void)tic;

  m_shader.Use();

  Eigen::Matrix4f view_mat = m_cam->c2w().inverse().matrix().cast<float>();
  glUniformMatrix4fv(m_view_loc, 1, GL_FALSE, view_mat.data());
  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  glUniformMatrix4fv(m_prj_loc, 1, GL_FALSE, prj_mat.data());

  for (auto [mesh, trans] : m_nodes) {
    int model_loc = m_node_locs[mesh];
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    mesh->Draw(m_shader);
  }

  return true;
}

void RendererGl::SetCamera(const CameraPtr cam) { m_cam = cam; }
void RendererGl::SetMesh(RenderableMeshPtr mesh, const Eigen::Affine3f& trans) {
  m_nodes[mesh] = trans;
}
void RendererGl::ClearMesh() {
  m_nodes.clear();
  m_node_locs.clear();
}

void RendererGl::SetFragType(const FragShaderType& frag_type) {
  m_shader.frag_type = frag_type;
}

}  // namespace ugu
