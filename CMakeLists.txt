cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME cpu_renderer)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "cpu renderer")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

# .exe
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

if (WIN32)
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "-std=c++14 -W4 -EHsc")
endif()

# Search package example: OpenGL
# find_package(OpenGL REQUIRED)

# global include setting
#include_directories(
#    include
#    src/include
#    third_party)

# naort
# add_subdirectory(third_party/nanort)
# include_directories(third_party/nanort/examples/common)


set(PUBLIC_LIB_NAME cpu_renderer)
# sample to make library from source
# SHARED: .dll and .lib
# STATIC: .lib only
add_library(${PUBLIC_LIB_NAME}
    STATIC
    src/cpu_renderer.cpp
    src/camera.cpp
    src/pose.cpp
    src/mesh.cpp
    src/image.cpp
	third_party/nanort/nanort.cc
	#third_party/tinyobjloader/tiny_obj_loader.cc
)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE include)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/nanort)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/glm)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/tinyobjloader)
target_include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/stb)

set_target_properties(${PUBLIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set(EXE_NAME ${PUBLIC_LIB_NAME}_run)
add_executable(${EXE_NAME}
    pc/main.cpp)

target_include_directories(${EXE_NAME} PRIVATE include)
target_include_directories(${EXE_NAME} PRIVATE third_party/nanort)
target_include_directories(${EXE_NAME} PRIVATE third_party/glm)
target_include_directories(${EXE_NAME} PRIVATE third_party/tinyobjloader)
target_include_directories(${EXE_NAME} PRIVATE third_party/stb)

target_link_libraries(${EXE_NAME}
    ${PUBLIC_LIB_NAME}
    )