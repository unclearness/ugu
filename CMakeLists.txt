cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME currender)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "A CPU renderer for computer vision")

set(CMAKE_VERBOSE_MAKEFILE TRUE)

# .lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# .dll and .exe
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin) 

if (WIN32)
# option for Visual Studio
# -EHsc (enable proper Exxeption Handling) needs to avoid C4530
# -Wall is too noisy so that set -W4. 
# https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level?view=vs-2017
# "However, for a new project, it may be best to use /W4 in all compilations;
# this will ensure the fewest possible hard-to-find code defects."
set(CMAKE_CXX_FLAGS "-std=c++14 -W4 -EHsc")

else()
# g++ option for *nix
set(CMAKE_CXX_FLAGS  "-std=c++14 -Wall -O2")

endif()

# third_party directries
if(NOT DEFINED NANORT_INSTALL_DIR)
  set(NANORT_INSTALL_DIR "third_party/nanort" CACHE PATH "NanoRT installed directory")
endif()
message("NANORT_INSTALL_DIR: ${NANORT_INSTALL_DIR}") 
if(NOT DEFINED STB_INSTALL_DIR)
  set(STB_INSTALL_DIR "third_party/stb" CACHE PATH "stb installed directory")
endif()
message("STB_INSTALL_DIR: ${STB_INSTALL_DIR}") 
if(NOT DEFINED TINYOBJLOADER_INSTALL_DIR)
  set(TINYOBJLOADER_INSTALL_DIR "third_party/tinyobjloader" CACHE PATH "tinyobjloader installed directory")
endif()
message("TINYOBJLOADER_INSTALL_DIR: ${TINYOBJLOADER_INSTALL_DIR}")
if(NOT DEFINED EIGEN_INSTALL_DIR)
  set(EIGEN_INSTALL_DIR "third_party/eigen" CACHE PATH "eigen installed directory")
message("EIGEN_INSTALL_DIR: ${EIGEN_INSTALL_DIR}") 
endif()
if(NOT DEFINED LODEPNG_INSTALL_DIR)
  set(LODEPNG_INSTALL_DIR "third_party/lodepng" CACHE PATH "LodePNG installed directory")
endif()
if(NOT DEFINED TINYCOLORMAP_INSTALL_DIR)
  set(TINYCOLORMAP_INSTALL_DIR "third_party/tinycolormap" CACHE PATH "tinycolormap installed directory")
endif()

# switch for optional libraries
option(CURRENDER_USE_STB "Use stb to enable image i/o" ON)
message("CURRENDER_USE_STB: ${CURRENDER_USE_STB}")
if(CURRENDER_USE_STB)
  add_definitions(-DCURRENDER_USE_STB)
  option(CURRENDER_USE_STB_AS_STATIC_LIB "Use stb as static lib" OFF)
  message("CURRENDER_USE_STB_AS_STATIC_LIB: ${CURRENDER_USE_STB_AS_STATIC_LIB}")
  if(CURRENDER_USE_STB_AS_STATIC_LIB)
    add_library(stb
      STATIC
      src/stb.cc)
    set(STB_LIB_NAME stb)
  else()
    set(CURRENDER_STB_IMPLEMENTATION_CC src/stb.cc)
  endif()
endif()

option(CURRENDER_USE_OPENCV "Use OpenCV as Image class" OFF)
message("CURRENDER_USE_OPENCV: ${CURRENDER_USE_OPENCV}")
if(CURRENDER_USE_OPENCV)
  find_package(OpenCV REQUIRED)
  add_definitions(-DCURRENDER_USE_OPENCV)
endif()

option(CURRENDER_USE_TINYOBJLOADER "Use tinyobjloader to enable .obj input" ON)
message("CURRENDER_USE_TINYOBJLOADER: ${CURRENDER_USE_TINYOBJLOADER}")
if(CURRENDER_USE_TINYOBJLOADER)
  add_definitions(-DCURRENDER_USE_TINYOBJLOADER)
  option(CURRENDER_USE_TINYOBJLOADER_AS_STATIC_LIB "Use tinyobjloader as static lib" OFF)
  message("CURRENDER_USE_TINYOBJLOADER_AS_STATIC_LIB: ${CURRENDER_USE_TINYOBJLOADER_AS_STATIC_LIB}")
  if(CURRENDER_USE_TINYOBJLOADER_AS_STATIC_LIB)
    add_subdirectory(${TINYOBJLOADER_INSTALL_DIR})
    set(TINYOBJLOADER_LIB_NAME tinyobjloader)
  else()
    set(CURRENDER_TINYOBJLOADER_IMPLEMENTATION_CC ${TINYOBJLOADER_INSTALL_DIR}/tiny_obj_loader.cc)
  endif()
endif()

option(CURRENDER_USE_LODEPNG "Use LodePNG to enable image i/o, especially for 16bit" ON)
message("CURRENDER_USE_LODEPNG: ${CURRENDER_USE_LODEPNG}")
if(CURRENDER_USE_LODEPNG)
  add_definitions(-DCURRENDER_USE_LODEPNG)
  option(CURRENDER_USE_LODEPNG_AS_STATIC_LIB "Use LodePNG as static lib" OFF)
  if(CURRENDER_USE_LODEPNG_AS_STATIC_LIB)
    add_library(lodepng
      STATIC
      ${LODEPNG_INSTALL_DIR}/lodepng.cpp
      ${LODEPNG_INSTALL_DIR}/lodepng.h)
      set(LODEPNG_LIB_NAME lodepng)
  else()
   set(CURRENDER_LODEPNG_SOURCES
      ${LODEPNG_INSTALL_DIR}/lodepng.cpp
      ${LODEPNG_INSTALL_DIR}/lodepng.h)
  endif()
endif()

option(CURRENDER_USE_TINYCOLORMAP "Use tinycolormap for visualization" ON)
message("CURRENDER_USE_TINYCOLORMAP: ${CURRENDER_USE_TINYCOLORMAP}")
if(CURRENDER_USE_TINYCOLORMAP)
  add_definitions(-DCURRENDER_USE_TINYCOLORMAP)
endif()

option(CURRENDER_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("CURRENDER_USE_OPENMP: ${CURRENDER_USE_OPENMP}")
if(CURRENDER_USE_OPENMP)
  add_definitions(-DCURRENDER_USE_OPENMP)
endif()

option(CURRENDER_USE_NANORT "Use NanoRT for Raytracer" ON)
message("CURRENDER_USE_NANORT: ${CURRENDER_USE_NANORT}")
if(CURRENDER_USE_NANORT)
  add_definitions(-DCURRENDER_USE_NANORT)
endif()

# For OpenMP
if(CURRENDER_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

set(PUBLIC_LIB_NAME ${PROJECT_NAME})
add_library(${PUBLIC_LIB_NAME}
  STATIC
  include/currender/common.h
  include/currender/renderer.h
  include/currender/raytracer.h
  include/currender/rasterizer.h
  include/currender/camera.h
  include/currender/mesh.h
  include/currender/image.h
  include/currender/log.h
  include/currender/util.h

  src/raytracer.cc
  src/rasterizer.cc
  src/camera.cc
  src/mesh.cc
  src/image.cc
  src/pixel_shader.h
  src/timer.h
  src/log.cc
  src/util.cc
  src/util_private.h
  src/util_private.cc

  # implementations of header-only library
  ${CURRENDER_STB_IMPLEMENTATION_CC}
  ${CURRENDER_TINYOBJLOADER_IMPLEMENTATION_CC}

  # lodepng
  ${CURRENDER_LODEPNG_SOURCES}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${NANORT_INSTALL_DIR})
include_directories(${EIGEN_INSTALL_DIR})
include_directories(${TINYOBJLOADER_INSTALL_DIR})
include_directories(${STB_INSTALL_DIR}/..)
include_directories(${LODEPNG_INSTALL_DIR})
include_directories(${TINYCOLORMAP_INSTALL_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

set_target_properties(${PUBLIC_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})

set(MINIMUM_EXAMPLE_EXE currender_minimum_example)
add_executable(${MINIMUM_EXAMPLE_EXE}
  minimum_example.cc)
target_link_libraries(${MINIMUM_EXAMPLE_EXE}
  ${PUBLIC_LIB_NAME}
  ${STB_LIB_NAME}
  ${TINYOBJLOADER_LIB_NAME}
  ${LODEPNG_LIB_NAME}
  ${OpenCV_LIBS}
  )

set(EXAMPLES_EXE currender_examples)
add_executable(${EXAMPLES_EXE}
  examples.cc)
target_link_libraries(${EXAMPLES_EXE}
  ${PUBLIC_LIB_NAME}
  ${STB_LIB_NAME}
  ${TINYOBJLOADER_LIB_NAME}
  ${LODEPNG_LIB_NAME}
  ${OpenCV_LIBS}
  )
if (WIN32)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXAMPLES_EXE})
  # suppress C2338 for eigen
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

# make test data directory
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)
endif()
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny)
endif()
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/minimum_example)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/minimum_example)
endif()

# test data preparation
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny/bunny.obj)

# download test data
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
file(DOWNLOAD http://www.kunzhou.net/tex-models/bunny.zip ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
endif()

# unzip test data
add_custom_target( unZip ALL)
add_custom_command(TARGET unZip PRE_BUILD
   COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny
DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
COMMENT "Unpacking bunny.zip"
VERBATIM)

endif()