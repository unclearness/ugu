cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME currender)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "cpu renderer")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

if (WIN32)
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "-std=c++14 -W4 -EHsc")
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(PUBLIC_LIB_NAME ${PROJECT_NAME})
add_library(${PUBLIC_LIB_NAME}
    STATIC
    include/renderer.h
    include/camera.h
    include/pose.h
    include/mesh.h
    include/image.h

    src/renderer.cc
    src/camera.cc
    src/pose.cc
    src/mesh.cc
    src/image.cc
    src/pixel_shader.h
    src/pixel_shader.cc
)

# set folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

SOURCE_GROUP("Public header files" FILES
    include/renderer.h
    include/camera.h
    include/pose.h
    include/mesh.h
    include/image.h
)

SOURCE_GROUP("Private header files" FILES
    src/pixel_shader.h
)

SOURCE_GROUP("Source files" FILES
    src/renderer.cc
    src/camera.cc
    src/pose.cc
    src/mesh.cc
    src/image.cc
    src/pixel_shader.cc
)

include_directories(${PUBLIC_LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party)
include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/nanort)
include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/glm)
include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/tinyobjloader)
include_directories(${PUBLIC_LIB_NAME} PRIVATE third_party/stb)

set_target_properties(${PUBLIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set(EXE_NAME ${PUBLIC_LIB_NAME}_run)
add_executable(${EXE_NAME}
    pc/main.cc)


target_link_libraries(${EXE_NAME}
    ${PUBLIC_LIB_NAME}
    )

# test data preparation
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny/bunny.obj)

# download test data
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
file(DOWNLOAD http://www.kunzhou.net/tex-models/bunny.zip ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny)
add_custom_target(makeDirectory ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory  ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny)
endif()

# unzip test data
if (WIN32)
add_custom_target( unZip ALL)
add_custom_command(TARGET unZip PRE_BUILD
   COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny
DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
COMMENT "Unpacking bunny.zip"
VERBATIM)
endif()

endif()