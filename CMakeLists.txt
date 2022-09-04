cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME ugu)
project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "UGU: Unclearness Geometry Utility")

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
set(CMAKE_CXX_FLAGS "/std:c++17 /W4 /EHsc /MP")


else()
# g++ option for *nix

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #Set this to ON if you want to use GNU++
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O2")


endif()

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  message(STATUS "Has a parent scope.")
else()
  message(STATUS "Doesn't have a parent scope.")
endif()

function(set_with_parent ARG_NAME ARG_PATH ARG_TEXT)
    set(${ARG_NAME} ${ARG_PATH} CACHE PATH ${ARG_TEXT} FORCE)
    if (hasParent)
      set(${ARG_NAME} ${ARG_PATH} PARENT_SCOPE)
    endif()
endfunction(set_with_parent)

set(Ugu_LIBS)
set(Ugu_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)

# third_party directries
# must
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
  set_with_parent(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" "eigen installed directory" PARENT_SCOPE)
message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}") 
endif()
set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

# optional
if(NOT DEFINED STB_INSTALL_DIR)
  set_with_parent(STB_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb" "stb installed directory")
endif()
message("STB_INSTALL_DIR: ${STB_INSTALL_DIR}") 
if(NOT DEFINED TINYOBJLOADER_INSTALL_DIR)
  set_with_parent(TINYOBJLOADER_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyobjloader" "tinyobjloader installed directory")
endif()
message("TINYOBJLOADER_INSTALL_DIR: ${TINYOBJLOADER_INSTALL_DIR}")
if(NOT DEFINED LODEPNG_INSTALL_DIR)
  set_with_parent(LODEPNG_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/lodepng" "LodePNG installed directory")
endif()
message("LODEPNG_INSTALL_DIR: ${LODEPNG_INSTALL_DIR}")
if(NOT DEFINED TINYCOLORMAP_INSTALL_DIR)
  set_with_parent(TINYCOLORMAP_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinycolormap/include" "tinycolormap installed directory")
endif()
message("TINYCOLORMAP_INSTALL_DIR: ${TINYCOLORMAP_INSTALL_DIR}")
if(NOT DEFINED NANORT_INSTALL_DIR)
  set_with_parent(NANORT_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanort" "nanort installed directory")
endif()
message("NANORT_INSTALL_DIR: ${NANORT_INSTALL_DIR}") 
if(NOT DEFINED JSON_INSTALL_DIR)
  set_with_parent(JSON_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include" "json installed directory")
endif()
message("JSON_INSTALL_DIR: ${JSON_INSTALL_DIR}") 
if(NOT DEFINED NANOFLANN_INSTALL_DIR)
  set_with_parent(NANOFLANN_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann/include" "nanoflann installed directory")
endif()
message("NANOFLANN_INSTALL_DIR: ${NANOFLANN_INSTALL_DIR}")

if(NOT DEFINED FAST_QUADRIC_MESH_SIMPLIFICATION_INSTALL_DIR)
  set_with_parent(FAST_QUADRIC_MESH_SIMPLIFICATION_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Fast-Quadric-Mesh-Simplification/src.cmd" "Fast-Quadric-Mesh-Simplification installed directory")
endif()
message("FAST_QUADRIC_MESH_SIMPLIFICATION_INSTALL_DIR: ${FAST_QUADRIC_MESH_SIMPLIFICATION_INSTALL_DIR}") 

if(NOT DEFINED CXXOPTS_INSTALL_DIR)
  set_with_parent(CXXOPTS_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cxxopts/include" "cxxopts installed directory")
endif()
message("CXXOPTS_INSTALL_DIR: ${CXXOPTS_INSTALL_DIR}")

if(NOT DEFINED MVS_TEXTURING_INSTALL_DIR)
  set_with_parent(MVS_TEXTURING_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mvs-texturing" "mvs-texturing installed directory")
endif()
message("MVS_TEXTURING_INSTALL_DIR: ${MVS_TEXTURING_INSTALL_DIR}")


if(NOT DEFINED LIBIGL_INSTALL_DIR)
  set_with_parent(LIBIGL_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libigl" "libigl installed directory")
endif()
message("LIBIGL_INSTALL_DIR: ${LIBIGL_INSTALL_DIR}")

# switch for optional libraries
option(UGU_USE_STB "Use stb to enable image i/o" ON)
message("UGU_USE_STB: ${UGU_USE_STB}")
if(UGU_USE_STB)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${STB_INSTALL_DIR})
  add_definitions(-DUGU_USE_STB)
  option(UGU_USE_STB_BUT_NO_IMPLEMENTATION "Include stb but implement somewhere else" OFF)
  if (NOT UGU_USE_STB_BUT_NO_IMPLEMENTATION)
    option(UGU_USE_STB_AS_STATIC_LIB "Use stb as static lib" OFF)
    message("UGU_USE_STB_AS_STATIC_LIB: ${UGU_USE_STB_AS_STATIC_LIB}")
    if(UGU_USE_STB_AS_STATIC_LIB)
      add_library(stb
        STATIC
        src/stb.cc)
      set(Ugu_LIBS ${Ugu_LIBS} stb)
    else()
      set(UGU_STB_IMPLEMENTATION_CC src/stb.cc)
    endif()
  endif()
endif()

option(UGU_USE_OPENCV "Use OpenCV as Image class" OFF)
message("UGU_USE_OPENCV: ${UGU_USE_OPENCV}")
if(UGU_USE_OPENCV)
  find_package(OpenCV REQUIRED)
  add_definitions(-DUGU_USE_OPENCV)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
  set(Ugu_LIBS ${Ugu_LIBS} ${OpenCV_LIBS})
endif()

option(UGU_USE_TINYOBJLOADER "Use tinyobjloader to enable .obj input" ON)
message("UGU_USE_TINYOBJLOADER: ${UGU_USE_TINYOBJLOADER}")
if(UGU_USE_TINYOBJLOADER)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${TINYOBJLOADER_INSTALL_DIR})
  add_definitions(-DUGU_USE_TINYOBJLOADER)
  option(UGU_USE_TINYOBJLOADER_BUT_NO_IMPLEMENTATION "Include tinyobjloader but implement somewhere else" OFF)
  if (NOT UGU_USE_TINYOBJLOADER_BUT_NO_IMPLEMENTATION)
    option(UGU_USE_TINYOBJLOADER_AS_STATIC_LIB "Use tinyobjloader as static lib" OFF)
    message("UGU_USE_TINYOBJLOADER_AS_STATIC_LIB: ${UGU_USE_TINYOBJLOADER_AS_STATIC_LIB}")
    if(UGU_USE_TINYOBJLOADER_AS_STATIC_LIB)
      add_subdirectory(${TINYOBJLOADER_INSTALL_DIR})
      set(Ugu_LIBS ${Ugu_LIBS} tinyobjloader)
    else()
      set(UGU_TINYOBJLOADER_IMPLEMENTATION_CC ${TINYOBJLOADER_INSTALL_DIR}/tiny_obj_loader.cc)
    endif()
  endif()
endif()

option(UGU_USE_LODEPNG "Use LodePNG to enable image i/o, especially for 16bit" ON)
message("UGU_USE_LODEPNG: ${UGU_USE_LODEPNG}")
if(UGU_USE_LODEPNG)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${LODEPNG_INSTALL_DIR})
  add_definitions(-DUGU_USE_LODEPNG)
  option(UGU_USE_LODEPNG_AS_STATIC_LIB "Use LodePNG as static lib" OFF)
  if(UGU_USE_LODEPNG_AS_STATIC_LIB)
    add_library(lodepng
      STATIC
      ${LODEPNG_INSTALL_DIR}/lodepng.cpp
      ${LODEPNG_INSTALL_DIR}/lodepng.h)
      set(Ugu_LIBS ${Ugu_LIBS} lodepng)
  else()
   set(UGU_LODEPNG_SOURCES
      ${LODEPNG_INSTALL_DIR}/lodepng.cpp
      ${LODEPNG_INSTALL_DIR}/lodepng.h)
  endif()
endif()

option(UGU_USE_TINYCOLORMAP "Use tinycolormap for visualization" ON)
message("UGU_USE_TINYCOLORMAP: ${UGU_USE_TINYCOLORMAP}")
if(UGU_USE_TINYCOLORMAP)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${TINYCOLORMAP_INSTALL_DIR})
  add_definitions(-DUGU_USE_TINYCOLORMAP)
endif()

option(UGU_USE_NANORT "Use nanort for raytrace" ON)
message("UGU_USE_NANORT: ${UGU_USE_NANORT}")
if(UGU_USE_NANORT)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${NANORT_INSTALL_DIR})
  add_definitions(-DUGU_USE_NANORT)
endif()

option(UGU_USE_JSON "Use nlohmann/json for json" ON)
message("UGU_USE_JSON: ${UGU_USE_JSON}")
if(UGU_USE_JSON)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${JSON_INSTALL_DIR})
  add_definitions(-DUGU_USE_JSON)
endif()

option(UGU_USE_NANOFLANN "Use nanoflann" ON)
message("UGU_USE_NANOFLANN: ${UGU_USE_NANOFLANN}")
if(UGU_USE_NANOFLANN)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${NANOFLANN_INSTALL_DIR})
  add_definitions(-DUGU_USE_NANOFLANN)
endif()

option(UGU_USE_CXXOPTS "Use cxxopts" ON)
message("UGU_USE_CXXOPTS: ${UGU_USE_CXXOPTS}")
if(UGU_USE_CXXOPTS)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${CXXOPTS_INSTALL_DIR})
  add_definitions(-DUGU_USE_CXXOPTS)
endif()


option(UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION "Use Fast-Quadric-Mesh-Simplification" OFF)
message("UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION: ${UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION}")
if(UGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${FAST_QUADRIC_MESH_SIMPLIFICATION_INSTALL_DIR})
  add_definitions(-DUGU_USE_FAST_QUADRIC_MESH_SIMPLIFICATION)
endif()

option(UGU_USE_MVS_TEXTURING "Use mvs-texturing" OFF)
message("UGU_USE_MVS_TEXTURING: ${UGU_USE_MVS_TEXTURING}")
if(UGU_USE_MVS_TEXTURING)
  add_subdirectory(${MVS_TEXTURING_INSTALL_DIR})
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS}
        ${MVS_TEXTURING_INSTALL_DIR}
        ${MVS_TEXTURING_INSTALL_DIR}/libs
        ${MVS_TEXTURING_INSTALL_DIR}/third_party/mve/libs
        ${MVS_TEXTURING_INSTALL_DIR}/third_party/rayint/libs)
  set(MvsTexturing_LIBS tex mve mve_util mrf)
  add_definitions(-DUGU_USE_MVS_TEXTURING)
  set_property(TARGET texrecon tex mve mve_sfm mve_math mve_fssr mve_dmrecon mve_util mrf PROPERTY FOLDER "mvs-texturing") 
endif()


option(UGU_USE_LIBIGL "Use libigl" ON)
message("UGU_USE_LIBIGL: ${UGU_USE_LIBIGL}")
if(UGU_USE_LIBIGL)
  set(Ugu_INCLUDE_DIRS ${Ugu_INCLUDE_DIRS} ${LIBIGL_INSTALL_DIR}/include)
  add_definitions(-DUGU_USE_LIBIGL)
endif()

option(UGU_USE_OPENMP "Use OpenMP to enable parallelization" ON)
if(APPLE)
set(UGU_USE_OPENMP OFF)
endif()
message("UGU_USE_OPENMP: ${UGU_USE_OPENMP}")
if(UGU_USE_OPENMP)
  add_definitions(-DUGU_USE_OPENMP)
endif()

# For OpenMP
if(UGU_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()


option(UGU_STATIC_LIBRARY "Build UGU as static library" ON)
message("UGU_STATIC_LIBRARY: ${UGU_STATIC_LIBRARY}")
set(
  UGU_SOURCE

  include/ugu/eigen_util.h
  include/ugu/common.h
  include/ugu/camera.h
  include/ugu/mesh.h
  include/ugu/face_adjacency.h
  include/ugu/image.h
  include/ugu/line.h
  include/ugu/plane.h
  include/ugu/rect.h
  include/ugu/log.h
  include/ugu/util/camera_util.h
  include/ugu/util/io_util.h
  include/ugu/util/math_util.h
  include/ugu/util/raster_util.h
  include/ugu/util/rgbd_util.h
  include/ugu/util/thread_util.h
  include/ugu/util/geom_util.h
  include/ugu/util/image_util.h
  include/ugu/util/string_util.h
  include/ugu/util/path_util.h
  include/ugu/timer.h

  src/common.cc
  src/camera.cc
  src/line.cc
  src/plane.cc
  src/mesh.cc
  src/gltf.h
  src/ugu_stb.h
  src/log.cc
  src/util/camera_util.cc
  src/util/io_util.cc
  src/util/math_util.cc
  src/util/rgbd_util.cc
  src/util/geom_util.cc
  src/util/image_util.cc
  src/util/string_util.cc
  src/util/raster_util.cc
  src/util/path_util.cc

  include/ugu/renderer/base.h
  include/ugu/renderer/raytracer.h
  src/renderer/raytracer.cc
  include/ugu/renderer/rasterizer.h
  src/renderer/rasterizer.cc
  include/ugu/renderer/pixel_shader.h
  include/ugu/renderer/util_private.h
  src/renderer/util_private.cc

  
  include/ugu/voxel/voxel.h
  src/voxel/voxel.cc
  include/ugu/voxel/extract_voxel.h
  src/voxel/extract_voxel.cc
  include/ugu/voxel/marching_cubes.h
  src/voxel/marching_cubes.cc
  include/ugu/voxel/marching_cubes_lut.h
  src/voxel/marching_cubes_lut.cc

  include/ugu/sfs/voxel_carver.h
  src/sfs/voxel_carver.cc

  include/ugu/texturing/visibility_tester.h
  src/texturing/visibility_tester.cc
  include/ugu/texturing/vertex_colorizer.h
  src/texturing/vertex_colorizer.cc
  include/ugu/texturing/texture_mapper.h
  src/texturing/texture_mapper.cc

  include/ugu/stereo/base.h
  src/stereo/base.cc

  include/ugu/synthesis/bdsim.h
  src/synthesis/bdsim.cc

  include/ugu/inflation/inflation.h
  src/inflation/inflation.cc

  include/ugu/inpaint/inpaint.h
  src/inpaint/inpaint.cc

  include/ugu/geodesic/geodesic.h
  src/geodesic/geodesic.cc

  include/ugu/clustering/clustering.h
  src/clustering/clustering.cc

  include/ugu/optimizer/optimizer.h
  src/optimizer/optimizer.cc

  include/ugu/decimation/decimation.h
  src/decimation/decimation.cc

  include/ugu/correspondence/correspondence_finder.h
  src/correspondence/correspondence_finder.cc

  include/ugu/textrans/texture_transfer.h
  src/textrans/texture_transfer.cc

  include/ugu/accel/kdtree.h
  include/ugu/accel/kdtree_base.h
  include/ugu/accel/kdtree_naive.h
  include/ugu/accel/kdtree_nanoflann.h
  include/ugu/accel/bvh.h
  include/ugu/accel/bvh_nanort.h

  include/ugu/discrete/bin_packer_2d.h
  src/discrete/bin_packer_2d.cc

  include/ugu/parameterize/parameterize.h
  src/parameterize/parameterize.cc

  include/ugu/external/external.h
  src/external/fast_quadric_mesh_simplification.cc
  src/external/mvs_texturing.cc
  src/external/libigl.cc

  # implementations of header-only library
  ${UGU_STB_IMPLEMENTATION_CC}
  ${UGU_TINYOBJLOADER_IMPLEMENTATION_CC}

  # lodepng
  ${UGU_LODEPNG_SOURCES}

)
if(UGU_STATIC_LIBRARY)
  set(Ugu_LIB ugu)
  add_definitions(-DUGU_STATIC_LIBRARY)
  add_library(${Ugu_LIB}
    STATIC
    ${UGU_SOURCE}
  )

  set(Ugu_LIBS ${Ugu_LIBS} ${Ugu_LIB} ${MvsTexturing_LIBS}) # This order is important in Ubuntu...
endif()

set(SOURCE_LIST ${SOURCE_LIST} ${UGU_SOURCE})

set_with_parent(Ugu_LIBS "${Ugu_LIBS}" "Ugu_LIBS")
set_with_parent(Ugu_INCLUDE_DIRS "${Ugu_INCLUDE_DIRS}" "Ugu_INCLUDE_DIRS")
message("Ugu_LIBS: ${Ugu_LIBS}") 
message("Ugu_INCLUDE_DIRS: ${Ugu_INCLUDE_DIRS}") 

target_include_directories(${Ugu_LIB} PUBLIC ${Ugu_INCLUDE_DIRS})
set_target_properties(${Ugu_LIB} PROPERTIES VERSION ${PROJECT_VERSION})

set_property( GLOBAL PROPERTY USE_FOLDERS ON)

function(setup_exe EXE_NAME EXE_SRC)
  set(SETUP_EXE_FOLDER_NAME "ugu_examples")
  if(ARGN)
    cmake_parse_arguments(SETUP_EXE "" "FOLDER_NAME" "" ${ARGN})
  endif()
  #SOURCE_GROUP("Source files\\examples\\" FILES ${EXE_SRC})
  add_executable(${EXE_NAME} ${EXE_SRC})
  target_include_directories(${EXE_NAME} PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(${EXE_NAME} ${Ugu_LIBS})
  #set(SOURCE_LIST ${SOURCE_LIST} ${EXE_SRC})

  set_property(TARGET ${EXE_NAME} PROPERTY FOLDER ${SETUP_EXE_FOLDER_NAME}) 
endfunction(setup_exe)

if (NOT hasParent)
  option(UGU_BUILD_EXAMPLES "Switch to build exampls" ON)
  option(UGU_BUILD_APP "Switch to build app" ON)
else()
  option(UGU_BUILD_EXAMPLES "Switch to build exampls" OFF)
  option(UGU_BUILD_APP "Switch to build app" OFF)
  message("hasParent is ON. Not build examples and app by default") 
endif()
message("UGU_BUILD_EXAMPLES: ${UGU_BUILD_EXAMPLES}") 
message("UGU_BUILD_APP: ${UGU_BUILD_APP}") 

if (UGU_BUILD_EXAMPLES)
  setup_exe(ex01_mesh examples/ex01_mesh.cc)
  setup_exe(ex02_renderer examples/ex02_renderer.cc)
  setup_exe(ex03_sfs examples/ex03_sfs.cc)
  setup_exe(ex04_adjacency examples/ex04_adjacency.cc)
  setup_exe(ex05_geodesic examples/ex05_geodesic.cc)
  setup_exe(ex06_image examples/ex06_image.cc)
  setup_exe(ex07_inflation examples/ex07_inflation.cc)
  setup_exe(ex08_inpaint examples/ex08_inpaint.cc)
  setup_exe(ex09_renderer_realtime examples/ex09_renderer_realtime.cc)
  setup_exe(ex10_stereo examples/ex10_stereo.cc)
  setup_exe(ex11_synthesis examples/ex11_synthesis.cc)
  setup_exe(ex12_texturing examples/ex12_texturing.cc)
  setup_exe(ex13_clustering examples/ex13_clustering.cc)
  setup_exe(ex14_thread examples/ex14_thread.cc)
  setup_exe(ex15_optimizer examples/ex15_optimizer.cc)
  setup_exe(ex16_textrans examples/ex16_textrans.cc)
  setup_exe(ex17_kdtree examples/ex17_kdtree.cc)
  setup_exe(ex18_bvh examples/ex18_bvh.cc)
  setup_exe(ex19_parameterize examples/ex19_parameterize.cc)
  setup_exe(ex20_manifold examples/ex20_manifold.cc)
  setup_exe(ex21_fusion examples/ex21_fusion.cc)
  setup_exe(ex22_line examples/ex22_line.cc)
endif()

if (UGU_BUILD_APP)
  setup_exe(textrans_cmd app/textrans.cc FOLDER_NAME "ugu_app")
  setup_exe(vc2tex_cmd app/vc2tex.cc FOLDER_NAME "ugu_app")
  setup_exe(image3d app/image3d.cc FOLDER_NAME "ugu_app")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

if (WIN32)
  #set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXAMPLES_EXE})
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
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha)
endif()

# test data preparation
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny/bunny.obj)

# download test data
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
file(DOWNLOAD http://www.kunzhou.net/tex-models/bunny.zip ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip)
endif()

if (NOT hasParent)
  # unzip test data
  add_custom_target( Ugu_UNZip_bunny ALL)
  add_custom_command(TARGET Ugu_UNZip_bunny PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/bunny.zip
  COMMENT "Unpacking bunny.zip"
  VERBATIM)
endif()

endif()

# test data preparation
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha/buddha.obj)

# download test data
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha.zip)
file(DOWNLOAD http://www.kunzhou.net/tex-models/buddha.zip ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha.zip)
endif()

if (NOT hasParent)
  # unzip test data
  add_custom_target( Ugu_UNZip_buddha ALL)
  add_custom_command(TARGET Ugu_UNZip_buddha PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha.zip
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/buddha.zip
  COMMENT "Unpacking buddha.zip"
  VERBATIM)
endif()

endif()
