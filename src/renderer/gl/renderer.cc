/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#include "ugu/renderer/gl/renderer.h"

#include "ugu/util/image_util.h"

#ifdef UGU_USE_GLFW
#include "glad/gl.h"
#endif

#include "ft2build.h"
#include FT_FREETYPE_H

#include "font/Open_Sans/static_OpenSans_OpenSans_Regular_ttf.h"

namespace {

const uint8_t* DEFAULT_FONT_DATA = ugu::static_OpenSans_OpenSans_Regular_ttf;

uint32_t DEFAULT_FONT_DATA_LEN = ugu::static_OpenSans_OpenSans_Regular_ttf_len;

Eigen::Vector3f GetDefaultSelectedPositionColor(uint32_t geomid) {
  Eigen::Vector3f table[3] = {
      {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
  if (geomid < 3) {
    return table[geomid];
  }

  return (Eigen::Vector3f::Random() + Eigen::Vector3f::Ones()) * 0.5f;
}

}  // namespace

namespace ugu {

TextRendererGl::TextRendererGl(unsigned int width, unsigned int height) {
  // load and configure shader
  this->TextShader.SetFragType(FragShaderType::TEXT);
  this->TextShader.SetVertType(VertShaderType::TEXT);

  this->TextShader.Prepare();

  this->TextShader.Use();

  this->TextShader.SetMat4("projection",
                           GetProjectionMatrixOpenGlForOrtho(
                               0.f, static_cast<float>(width),
                               static_cast<float>(height), 0.f, -1.f, 1.f));
  this->TextShader.SetInt("text", 0);
  // configure VAO/VBO for texture quads
  glGenVertexArrays(1, &this->VAO);
  glGenBuffers(1, &this->VBO);
  glBindVertexArray(this->VAO);
  glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void TextRendererGl::Load(std::string font, unsigned int fontSize) {
  // first clear the previously loaded Characters
  this->Characters.clear();
  // then initialize and load the FreeType library
  FT_Library ft;
  if (FT_Init_FreeType(&ft))  // all functions return a value different than 0
                              // whenever an error occurred
    std::cout << "ERROR::FREETYPE: Could not init FreeType Library"
              << std::endl;
  // load font as face
  FT_Face face;
  FT_Error err;

  if (font == "default") {
    err = FT_New_Memory_Face(ft, DEFAULT_FONT_DATA, DEFAULT_FONT_DATA_LEN, 0,
                             &face);
  } else {
    err = FT_New_Face(ft, font.c_str(), 0, &face);
  }
  if (err) std::cout << "ERROR::FREETYPE: Failed to load font" << std::endl;
  // set size to load glyphs as
  FT_Set_Pixel_Sizes(face, 0, fontSize);
  // disable byte-alignment restriction
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  // then for the first 128 ASCII characters, pre-load/compile their characters
  // and store them
  for (GLubyte c = 0; c < 128; c++)  // lol see what I did there
  {
    // load character glyph
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      std::cout << "ERROR::FREETYTPE: Failed to load Glyph" << std::endl;
      continue;
    }
    // generate texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width,
                 face->glyph->bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE,
                 face->glyph->bitmap.buffer);
    // set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // now store character for later use
    RendererCharacter character = {
        texture,
        Eigen::Vector2i(face->glyph->bitmap.width, face->glyph->bitmap.rows),
        Eigen::Vector2i(face->glyph->bitmap_left, face->glyph->bitmap_top),
        static_cast<uint32_t>(face->glyph->advance.x)};
    Characters.insert(std::pair<char, RendererCharacter>(c, character));
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  // destroy FreeType once we're finished
  FT_Done_Face(face);
  FT_Done_FreeType(ft);
}

void TextRendererGl::RenderText(const std::string& text, float x, float y,
                                float scale, const Eigen::Vector3f& color) {
  // activate corresponding render state
  this->TextShader.Use();
  this->TextShader.SetVec3("textColor", color);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glActiveTexture(GL_TEXTURE0);
  glBindVertexArray(this->VAO);

  // iterate through all characters
  std::string::const_iterator c;
  for (c = text.begin(); c != text.end(); c++) {
    RendererCharacter ch = Characters.at(*c);

    // this->TextShader.SetInt("text", ch.TextureID);

    float xpos = x + ch.Bearing.x() * scale;
    float ypos =
        y + (this->Characters['H'].Bearing.y() - ch.Bearing.y()) * scale;

    float w = ch.Size.x() * scale;
    float h = ch.Size.y() * scale;
    // update VBO for each character
    float vertices[6][4] = {
        {xpos, ypos + h, 0.0f, 1.0f}, {xpos + w, ypos, 1.0f, 0.0f},
        {xpos, ypos, 0.0f, 0.0f},

        {xpos, ypos + h, 0.0f, 1.0f}, {xpos + w, ypos + h, 1.0f, 1.0f},
        {xpos + w, ypos, 1.0f, 0.0f}};
    // render glyph texture over quad
    glBindTexture(GL_TEXTURE_2D, ch.TextureID);
    // update content of VBO memory
    glBindBuffer(GL_ARRAY_BUFFER, this->VBO);
    glBufferSubData(
        GL_ARRAY_BUFFER, 0, sizeof(vertices),
        vertices);  // be sure to use glBufferSubData and not glBufferData
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // render quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // now advance cursors for next glyph
    x += (ch.Advance >> 6) *
         scale;  // bitshift by 6 to get value in pixels (1/64th times 2^6 = 64)
  }
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glDisable(GL_BLEND);
  // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void TextRendererGl::RenderText(const Text& text) {
  RenderText(text.body, text.x, text.y, text.scale, text.color);
}

RendererGl::RendererGl() {}

RendererGl::~RendererGl() {}

bool RendererGl::ClearGlState() {
  if (!m_initialized) {
    return false;
  }

  // Clear mesh vertices and textures
  for (size_t i = 0; i < m_geoms.size(); i++) {
    auto mesh = m_geoms[i];
    mesh->ClearGlState();
  }
  ClearMesh();

  ClearSelectedPositions();

  // Delete buffers
  const GLuint texture_ids[5] = {gPosition, gNormal, gAlbedoSpec, gId, gFace};
  glDeleteTextures(5, texture_ids);
  glDeleteFramebuffers(1, &gBuffer);
  glDeleteRenderbuffers(1, &rboDepth);
  glDeleteVertexArrays(1, &quadVAO);
  return true;
}

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

  for (size_t i = 0; i < m_geoms.size(); i++) {
    auto mesh = m_geoms[i];
    auto trans = m_node_trans.at(mesh);
    int model_loc = glGetUniformLocation(m_gbuf_shader.ID, "model");
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    m_node_locs[mesh] = model_loc;

    mesh->BindTextures();
    mesh->SetupMesh(static_cast<int>(i + 1));
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
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA,
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

// Face id & geometry id
#if 1
  glGenTextures(1, &gId);
  glBindTexture(GL_TEXTURE_2D, gId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D,
                         gId, 0);
#endif

  // bary centric & uv
  glGenTextures(1, &gFace);
  glBindTexture(GL_TEXTURE_2D, gFace);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D,
                         gFace, 0);

  // tell OpenGL which color attachments we'll use (of this framebuffer) for
  // rendering
  attachments[0] = GL_COLOR_ATTACHMENT0;
  attachments[1] = GL_COLOR_ATTACHMENT1;
  attachments[2] = GL_COLOR_ATTACHMENT2;
  attachments[3] = GL_COLOR_ATTACHMENT3;
  attachments[4] = GL_COLOR_ATTACHMENT4;

  glDrawBuffers(static_cast<GLsizei>(attachments.size()), attachments.data());
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
  m_gbuf_shader.SetMat4("view", view_mat);

  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  m_gbuf_shader.SetMat4("projection", prj_mat);

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
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void*)(3 * sizeof(float)));

  m_deferred_shader.Use();
  m_deferred_shader.SetInt("gPosition", 0);
  m_deferred_shader.SetInt("gNormal", 1);
  m_deferred_shader.SetInt("gAlbedoSpec", 2);
  m_deferred_shader.SetInt("gId", 3);
  m_deferred_shader.SetInt("gFace", 4);

  m_initialized = true;

  m_text_renderer = std::make_shared<TextRendererGl>(m_width, m_height);
  m_text_renderer->Load(m_font, m_font_size);

  return true;
}

bool RendererGl::Draw(double tic) {
  (void)tic;

  // GBuf
  glViewport(0, 0, m_width, m_height);
  glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_gbuf_shader.Use();
  m_gbuf_shader.SetVec2(
      "WIN_SCALE", {static_cast<float>(m_width), static_cast<float>(m_height)});
  Eigen::Matrix4f view_mat = m_cam->c2w().inverse().matrix().cast<float>();
  m_gbuf_shader.SetMat4("view", view_mat);
  Eigen::Matrix4f prj_mat = m_cam->ProjectionMatrixOpenGl(m_near_z, m_far_z);
  m_gbuf_shader.SetMat4("projection", prj_mat);

  for (const auto& mesh : m_geoms) {
    int model_loc = m_node_locs[mesh];
    auto trans = m_node_trans[mesh];
    if (!m_visibility.at(mesh)) {
      // If invisible, move it far away
      trans.translation().setConstant(std::numeric_limits<float>::max());
    }
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, trans.data());
    mesh->Draw(m_gbuf_shader);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Deferred

#if 1
  // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  m_deferred_shader.Use();

  m_deferred_shader.SetBool("showWire", m_show_wire);
  m_deferred_shader.SetVec3("wireCol", m_wire_col);
  m_deferred_shader.SetFloat("nearZ", m_near_z);
  m_deferred_shader.SetFloat("farZ", m_far_z);
  m_deferred_shader.SetVec3("bkgCol", m_bkg_col);

  std::vector<Eigen::Vector3f> selected_positions_prj(
      MAX_SELECTED_POS * MAX_GEOM, Eigen::Vector3f::Constant(-1.f));  // Flatten
  std::vector<Eigen::Vector3f> selected_position_colors(
      m_geoms.size(), Eigen::Vector3f::Zero());

  for (size_t j = 0; j < m_geoms.size(); j++) {
    const auto& mesh = m_geoms[j];

    // std::cout << m_selected_positions[mesh].size() << std::endl;
    if (!m_selected_positions[mesh].empty()) {
      // Eigen::Matrix4f prj_view_mat = prj_mat * view_mat;
      for (size_t i = 0; i < m_selected_positions.at(mesh).size(); i++) {
        Eigen::Vector3f p_gl_frag_camz(0.f, 0.f, 0.f);
        if (m_visibility.at(mesh)) {
          Eigen::Vector3f p_wld = m_selected_positions.at(mesh)[i];
          Eigen::Vector4f p_cam =
              view_mat * Eigen::Vector4f(p_wld.x(), p_wld.y(), p_wld.z(), 1.f);
          Eigen::Vector4f p_ndc = prj_mat * p_cam;
          p_ndc /= p_ndc.w();  // NDC [-1:1]

          float cam_depth = -p_cam.z();

          // [-1:1],[-1:1] -> [0:w], [0:h]
          p_gl_frag_camz = Eigen::Vector3f(
              ((p_ndc.x() + 1.f) / 2.f) * m_cam->width(),
              ((p_ndc.y() + 1.f) / 2.f) * m_cam->height(), cam_depth);
        }
        selected_positions_prj[j * MAX_SELECTED_POS + i] = p_gl_frag_camz;
      }

      selected_position_colors[j] = m_selected_position_colors[mesh];
    }
  }
  if (!m_geoms.empty()) {
    m_deferred_shader.SetVec2("viewportOffset",
                              Eigen::Vector2f(m_viewport_x, m_viewport_y));
    m_deferred_shader.SetVec3Array("selectedPositions", selected_positions_prj);
    m_deferred_shader.SetVec3Array("selectedPosColors",
                                   selected_position_colors);
  }

  m_deferred_shader.SetVec3("viewPos", m_cam->c2w().translation().cast<float>());

  m_deferred_shader.SetFloat("selectedPosDepthTh", GetDepthThreshold());

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, gPosition);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, gNormal);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);

  glActiveTexture(GL_TEXTURE3);
  glBindTexture(GL_TEXTURE_2D, gId);
  glActiveTexture(GL_TEXTURE4);
  glBindTexture(GL_TEXTURE_2D, gFace);

  glBindVertexArray(quadVAO);
  glViewport(m_viewport_x, m_viewport_y, m_viewport_width, m_viewport_height);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
#endif

  // To write text on meshes
  glClear(GL_DEPTH_BUFFER_BIT);

  // Texts
  for (const auto& text : m_texts) {
    m_text_renderer->RenderText(text);
  }

  return true;
}

bool RendererGl::ReadGbuf() {
  const bool flip_y = true;

  // To read out of [0, 1] range
  // glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
  //

  Image4f tmp4f(m_height, m_width);
  Image1f tmp1f(m_height, m_width);
  Image4i tmp4i(m_height, m_width);

  if (static_cast<uint32_t>(m_gbuf.color.cols) != m_width ||
      static_cast<uint32_t>(m_gbuf.color.rows) != m_height) {
    m_gbuf.Init(m_width, m_height);
  }

  m_gbuf.Reset();

  glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

  // Depth 01
  {
    glReadBuffer(GL_NONE);
    // glRead"n"Pixels() is valid only if OpenGL4.5 (2017) or later
    // glReadPixels() has been supported from long years ago
    glReadPixels(0, 0, m_width, m_height, GL_DEPTH_COMPONENT, GL_FLOAT,
                 tmp1f.data);
    tmp1f.forEach([&](float& d, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];

      if (d <= 0.f || 1.f <= d) {
        m_gbuf.stencil.at<uint8_t>(y, x) = 0;
      } else {
        m_gbuf.stencil.at<uint8_t>(y, x) = 255;
      }

      m_gbuf.depth_01.at<float>(y, x) = std::clamp(d, 0.f, 1.f);
    });
  }

  Eigen::Matrix3f w2c_R = m_cam->w2c().rotation().matrix().cast<float>();
  Eigen::Vector3f w2c_t = m_cam->w2c().translation().cast<float>();

  // Pos
  {
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, tmp4f.data);
    tmp4f.forEach([&](Vec4f& n, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];
      if (m_gbuf.stencil.at<uint8_t>(y, x) != 255) {
        return;
      }

      auto& wld = m_gbuf.pos_wld.at<Vec3f>(y, x);
      wld[0] = n[0];
      wld[1] = n[1];
      wld[2] = n[2];
      auto& cam = m_gbuf.pos_cam.at<Vec3f>(y, x);
      cam = w2c_t + (w2c_R * wld);
    });
  }

  // Normal
  {
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, tmp4f.data);
    tmp4f.forEach([&](Vec4f& n, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];
      if (m_gbuf.stencil.at<uint8_t>(y, x) != 255) {
        return;
      }
      auto& wld = m_gbuf.normal_wld.at<Vec3f>(y, x);
      wld[0] = n[0];
      wld[1] = n[1];
      wld[2] = n[2];
      auto& cam = m_gbuf.normal_cam.at<Vec3f>(y, x);
      cam = w2c_R * wld;
    });
  }

  // Color
  {
    glReadBuffer(GL_COLOR_ATTACHMENT2);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, tmp4f.data);
    tmp4f.forEach([&](Vec4f& n, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];
      if (m_gbuf.stencil.at<uint8_t>(y, x) != 255) {
        return;
      }
      auto& col = m_gbuf.color.at<Vec3b>(y, x);
      col[0] = saturate_cast<uint8_t>(n[0] * 255);
      col[1] = saturate_cast<uint8_t>(n[1] * 255);
      col[2] = saturate_cast<uint8_t>(n[2] * 255);
    });

    // Ensure default channel order
    m_gbuf.color = RGB2Default(m_gbuf.color);
  }

  tmp4f.setTo(0.f);
  // Face id & geo id
  {
    glReadBuffer(GL_COLOR_ATTACHMENT3);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, tmp4f.data);
    tmp4f.forEach([&](Vec4f& val, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];
      if (m_gbuf.stencil.at<uint8_t>(y, x) != 255) {
        return;
      }
      int geo_id = static_cast<int>(
          std::round(val[1]));  // static_cast<int>(val[1] * m_geoms.size());
      if (geo_id > 0) {
        //  std::cout << geo_id << std::endl;
      }
      m_gbuf.geo_id.at<int>(y, x) = geo_id;
      m_gbuf.face_id.at<int>(y, x) = static_cast<int>(std::round(val[0]));
    });
  }

  tmp4f.setTo(0.f);
  // Face id & barycenteric
  {
    glReadBuffer(GL_COLOR_ATTACHMENT4);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_FLOAT, tmp4f.data);
    tmp4f.forEach([&](Vec4f& val, const int yx[2]) {
      int y = flip_y ? (m_height - 1 - yx[0]) : yx[0];
      int x = yx[1];
      if (m_gbuf.stencil.at<uint8_t>(y, x) != 255) {
        return;
      }
      auto& bary = m_gbuf.bary.at<Vec3f>(y, x);
      bary[0] = val[0];
      bary[1] = val[1];
      bary[2] = 1.f - bary[0] - bary[1];
      auto& uv = m_gbuf.uv.at<Vec3f>(y, x);
      uv[0] = val[2];
      uv[1] = val[3];
      uv[2] = 1.f - uv[0] - uv[1];
    });
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  return true;
}

void RendererGl::SetCamera(const CameraPtr cam) { m_cam = cam; }
CameraPtr RendererGl::GetCamera() const { return m_cam; }
void RendererGl::SetMesh(RenderableMeshPtr mesh, const Eigen::Affine3f& trans,
                         bool update_bvh) {
  bool first_set = m_node_trans.find(mesh) == m_node_trans.end();
  if (first_set) {
    mesh->CalcStats();
    m_geoms.push_back(mesh);

    m_visibility[mesh] = true;
  }

  if (first_set || update_bvh) {
    // Init BVH
    auto bvh = GetDefaultBvh<Eigen::Vector3f, Eigen::Vector3i>();
    auto scaled_verts = mesh->vertices();
    for (auto& v : scaled_verts) {
      v = trans * v;
    }
    bvh->SetData(scaled_verts, mesh->vertex_indices());
    bvh->Build();
    m_bvhs[mesh] = bvh;
  }

  m_node_trans[mesh] = trans;

  m_bb_max.setConstant(std::numeric_limits<float>::lowest());
  m_bb_min.setConstant(std::numeric_limits<float>::max());
  for (const auto& geom : m_geoms) {
    auto stats = geom->stats();
    m_bb_max = ComputeMaxBound(std::vector{m_bb_max, trans * stats.bb_max});
    m_bb_min = ComputeMinBound(std::vector{m_bb_min, trans * stats.bb_min});
  }
}

void RendererGl::ClearMesh() {
  m_geoms.clear();
  m_node_trans.clear();
  m_node_locs.clear();
  m_bvhs.clear();
  m_visibility.clear();
}

void RendererGl::SetFragType(const FragShaderType& frag_type) {
  m_deferred_shader.frag_type = frag_type;
}

void RendererGl::SetNearFar(float near_z, float far_z) {
  m_near_z = near_z;
  m_far_z = far_z;
}

void RendererGl::GetNearFar(float& near_z, float& far_z) const {
  near_z = m_near_z;
  far_z = m_far_z;
}

void RendererGl::SetSize(uint32_t width, uint32_t height) {
  m_width = width;
  m_height = height;
}

void RendererGl::SetViewport(uint32_t x, uint32_t y, uint32_t width,
                             uint32_t height) {
  m_viewport_x = x;
  m_viewport_y = y;
  m_viewport_width = width;
  m_viewport_height = height;
}

void RendererGl::GetGbuf(GBuffer& gbuf) const { gbuf = m_gbuf; }

bool RendererGl::GetShowWire() const { return m_show_wire; }
void RendererGl::SetShowWire(bool show_wire) { m_show_wire = show_wire; }

void RendererGl::SetWireColor(const Eigen::Vector3f& wire_col) {
  m_wire_col = wire_col;
}

const Eigen::Vector3f& RendererGl::GetWireColor() const { return m_wire_col; }

void RendererGl::SetBackgroundColor(const Eigen::Vector3f& bkg_col) {
  m_bkg_col = bkg_col;
}

const Eigen::Vector3f& RendererGl::GetBackgroundColor() const {
  return m_bkg_col;
}

bool RendererGl::AddSelectedPosition(const RenderableMeshPtr& geom,
                                     const Eigen::Vector3f& pos) {
  auto mesh_id = GetMeshId(geom);
  if (MAX_SELECTED_POS <= m_selected_positions.size() || mesh_id == ~0u) {
    return false;
  }

  if (m_selected_position_colors.find(geom) ==
      m_selected_position_colors.end()) {
    m_selected_position_colors[geom] = GetDefaultSelectedPositionColor(mesh_id);
  }

  m_selected_positions[geom].push_back(pos);

  return true;
}

bool RendererGl::AddSelectedPositions(
    const RenderableMeshPtr& geom,
    const std::vector<Eigen::Vector3f>& pos_list) {
  auto mesh_id = GetMeshId(geom);
  if (MAX_SELECTED_POS <= pos_list.size() || mesh_id == ~0u) {
    return false;
  }

  if (m_selected_position_colors.find(geom) ==
      m_selected_position_colors.end()) {
    m_selected_position_colors[geom] = GetDefaultSelectedPositionColor(mesh_id);
  }

  m_selected_positions[geom] = pos_list;

  return true;
}

bool RendererGl::AddSelectedPositionColor(const RenderableMeshPtr& geom,
                                          const Eigen::Vector3f& color) {
  m_selected_position_colors[geom] = color;
  return true;
}

const Eigen::Vector3f& RendererGl::GetSelectedPositionColor(
    const RenderableMeshPtr& geom) const {
  return m_selected_position_colors.at(geom);
}

void RendererGl::ClearSelectedPositions() {
  m_selected_positions.clear();
  m_selected_position_colors.clear();
}

void RendererGl::SetVisibility(const RenderableMeshPtr& geom, bool is_visible) {
  m_visibility.at(geom) = is_visible;
}

bool RendererGl::GetVisibility(const RenderableMeshPtr& geom) const {
  return m_visibility.at(geom);
}

void RendererGl::GetMergedBoundingBox(Eigen::Vector3f& bb_max,
                                      Eigen::Vector3f& bb_min) const {
  bb_max = m_bb_max;
  bb_min = m_bb_min;
}

float RendererGl::GetDepthThreshold() const {
  return (m_bb_max - m_bb_min).maxCoeff() * 0.01f;
}

std::vector<std::vector<IntersectResult>> RendererGl::Intersect(
    const Ray& ray) const {
  std::vector<std::vector<IntersectResult>> results_all;

  for (size_t geoid = 0; geoid < m_geoms.size(); geoid++) {
    std::vector<IntersectResult> results;
    if (m_visibility.at(m_geoms[geoid])) {
      results = m_bvhs.at(m_geoms[geoid])->Intersect(ray, true);
    }
    results_all.push_back(results);
  }

  return results_all;
}

std::pair<uint32_t, std::vector<std::vector<IntersectResult>>>
RendererGl::TestVisibility(const Eigen::Vector3f& point) const {
  Eigen::Vector3f wld_campos = m_cam->c2w().translation().cast<float>();

  Ray ray;
  ray.dir = (point - wld_campos).normalized();
  ray.org = wld_campos;

  auto results_all = Intersect(ray);

  uint32_t front_id = ~0u;
  float min_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < results_all.size(); i++) {
    if (!results_all[i].empty()) {
      if (results_all[i][0].t < min_dist) {
        front_id = static_cast<uint32_t>(i);
        min_dist = results_all[i][0].t;
      }
    }
  }

  return std::make_pair(front_id, results_all);
}

uint32_t RendererGl::GetMeshId(const RenderableMeshPtr& mesh) const {
  auto pos = std::find(m_geoms.begin(), m_geoms.end(), mesh);

  if (pos == m_geoms.end()) {
    return ~0u;
  }

  return static_cast<uint32_t>(std::distance(m_geoms.begin(), pos));
}

void RendererGl::SetText(const TextRendererGl::Text& text) {
  m_texts.push_back(text);
}

void RendererGl::SetTexts(const std::vector<TextRendererGl::Text>& texts) {
  m_texts = texts;
}

const std::vector<TextRendererGl::Text>& RendererGl::GetTexts() const {
  return m_texts;
}

std::vector<TextRendererGl::Text>& RendererGl::GetTexts() { return m_texts; }

}  // namespace ugu
