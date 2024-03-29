import os
import subprocess

root_dir = "../src/shader"
glsl_dir = "../src/shader/glsl"

shader_types = ["vert", "frag", "geom"]

cpp_header = """/*
 * Automatically generated by script/glsl2header.py
 */
/*
 * Copyright (C) 2023, unclearness
 * All rights reserved.
 */



#pragma once

#include <string>

namespace ugu {
"""

cpp_footer = "}\n"

for shader_type in shader_types:
    shader_header = cpp_header
    shader_dir = os.path.join(glsl_dir, shader_type)
    for shader_name in os.listdir(shader_dir):
        if not shader_name.endswith(shader_type):
            continue
        shader_path = os.path.join(shader_dir, shader_name)
        with open(shader_path, 'r') as fp:
            shader_body = ['R"(']
            for line in fp:
                line = line.rstrip()
                shader_body.append(line)
            shader_body = "\n".join(shader_body)
        shader_cpp_var_name = shader_type + "_" + \
            shader_name.split('.')[0] + "_code"
        shader_header += '\n'
        shader_header += f"static inline std::string {shader_cpp_var_name} = \n"
        shader_header += shader_body + ')";'
    shader_header += "\n"
    shader_header += cpp_footer
    cpp_header_path = os.path.join(root_dir, shader_type + ".h")
    with open(cpp_header_path, 'w') as fp:
        fp.write(shader_header)
    subprocess.run(["clang-format", cpp_header_path, "-i"])
