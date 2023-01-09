/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

namespace ugu {

static std::string vert_default_code =
    "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in vec3 aNormal;\n"
    "layout(location = 2) in vec3 aVertexColor;\n"
    "layout(location = 3) in vec2 aTexCoords;\n"

    "out vec3 Pos;\n"
    "out vec3 Normal;\n"
    "out vec3 VertexColor;\n"
    "out vec2 TexCoords;\n"

    "uniform mat4 model;\n"
    "uniform mat4 view;\n"
    "uniform mat4 projection;\n"

    "void main() {\n"
    "  Pos = aPos;\n"
    "  Normal = aNormal;\n"
    "  VertexColor = aVertexColor;\n"
    "  TexCoords = aTexCoords;\n"
    "  gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
    "}\n";

}