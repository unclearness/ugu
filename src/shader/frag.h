/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */

#pragma once

#include <string>

namespace ugu {

static std::string frag_unlit_code =
    "#version 330 core\n"
    "out vec4 FragColor;\n"

    "in vec3 Pos;\n"
    "in vec3 Normal;\n"
    "in vec3 VertexColor;\n"
    "in vec2 TexCoords;\n"

    "uniform sampler2D texture_diffuse1;\n"
    "void main() { FragColor = texture(texture_diffuse1, TexCoords); }\n";

}