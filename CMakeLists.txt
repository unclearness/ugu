cmake_minimum_required(VERSION 3.10)

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
set(CMAKE_CXX_FLAGS "/std:c++17 -W4 -EHsc")

else()
# g++ option for *nix
set(CMAKE_CXX_FLAGS  "-std=c++17 -Wall -O2")

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
  set_with_parent(JSON_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/single_include" "nanort installed directory")
endif()
message("JSON_INSTALL_DIR: ${JSON_INSTALL_DIR}") 
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

option(UGU_USE_OPENMP "Use OpenMP to enable parallelization" ON)
message("UGU_USE_OPENMP: ${UGU_USE_OPENMP}")
if(UGU_USE_OPENMP)
  add_definitions(-DUGU_USE_OPENMP)
endif()

# For OpenMP
if(UGU_USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()


option(UGU_STATIC_LIBRARY "Build UGU as static library" ON)
message("UGU_STATIC_LIBRARY: ${UGU_STATIC_LIBRARY}")
if(UGU_STATIC_LIBRARY)
  set(Ugu_LIB ugu)
  add_definitions(-DUGU_STATIC_LIBRARY)
  add_library(${Ugu_LIB}
    STATIC
    include/ugu/common.h
    include/ugu/camera.h
    include/ugu/mesh.h
    include/ugu/face_adjacency.h
    include/ugu/image.h
    include/ugu/line.h
    include/ugu/log.h
    include/ugu/util.h
    include/ugu/timer.h

    src/camera.cc
    src/mesh.cc
    src/gltf.h
    src/image.cc
    src/log.cc
    src/util.cc

    include/ugu/renderer/base.h
    include/ugu/renderer/raytracer.h
    include/ugu/renderer/raytracer.cc
    include/ugu/renderer/rasterizer.h
    include/ugu/renderer/rasterizer.cc
    include/ugu/renderer/pixel_shader.h
    include/ugu/renderer/util_private.h
    include/ugu/renderer/util_private.cc

    include/ugu/sfs/voxel_carver.h
    include/ugu/sfs/voxel_carver.cc
    include/ugu/sfs/extract_voxel.h
    include/ugu/sfs/extract_voxel.cc
    include/ugu/sfs/marching_cubes.h
    include/ugu/sfs/marching_cubes.cc
    include/ugu/sfs/marching_cubes_lut.h
    include/ugu/sfs/marching_cubes_lut.cc

    include/ugu/texturing/visibility_tester.h
    include/ugu/texturing/visibility_tester.cc
    include/ugu/texturing/vertex_colorizer.h
    include/ugu/texturing/vertex_colorizer.cc
    include/ugu/texturing/texture_mapper.h
    include/ugu/texturing/texture_mapper.cc
    include/ugu/texturing/bin_packer_2d.h
    include/ugu/texturing/bin_packer_2d.cc

    include/ugu/stereo/base.h
    include/ugu/stereo/base.cc

    include/ugu/synthesis/bdsim.h
    include/ugu/synthesis/bdsim.cc

    include/ugu/inflation/inflation.h
    include/ugu/inflation/inflation.cc

    # implementations of header-only library
    ${UGU_STB_IMPLEMENTATION_CC}
    ${UGU_TINYOBJLOADER_IMPLEMENTATION_CC}

    # lodepng
    ${UGU_LODEPNG_SOURCES}
  )

  set(Ugu_LIBS ${Ugu_LIBS} ${Ugu_LIB})
endif()

set_with_parent(Ugu_LIBS "${Ugu_LIBS}" "Ugu_LIBS")
set_with_parent(Ugu_INCLUDE_DIRS "${Ugu_INCLUDE_DIRS}" "Ugu_INCLUDE_DIRS")
message("Ugu_LIBS: ${Ugu_LIBS}") 
message("Ugu_INCLUDE_DIRS: ${Ugu_INCLUDE_DIRS}") 

target_include_directories(${Ugu_LIB} PUBLIC ${Ugu_INCLUDE_DIRS})
set_target_properties(${Ugu_LIB} PROPERTIES VERSION ${PROJECT_VERSION})

if (NOT hasParent)
  set(MESH_TEST_EXE ugu_mesh_test)
  add_executable(${MESH_TEST_EXE}
    mesh_test.cc)
  target_include_directories(${MESH_TEST_EXE} PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(${MESH_TEST_EXE}
    ${Ugu_LIBS}
    )

  add_executable(ugu_image_test
    test_image.cc)
    target_include_directories(ugu_image_test PRIVATE ${Ugu_INCLUDE_DIRS})
    target_link_libraries(ugu_image_test
    ${Ugu_LIBS}
  )

  add_executable(ugu_renderer_test
    renderer_test.cc)
  target_include_directories(ugu_renderer_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_renderer_test
    ${Ugu_LIBS}
  )

  add_executable(ugu_sfs_test
    sfs_test.cc)
  target_include_directories(ugu_sfs_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_sfs_test
    ${Ugu_LIBS}
  )

  add_executable(ugu_texturing_test
  test_texturing.cc)
  target_include_directories(ugu_texturing_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_texturing_test
  ${Ugu_LIBS}
  )

  add_executable(ugu_stereo_test
  test_stereo.cc)
  target_include_directories(ugu_stereo_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_stereo_test
  ${Ugu_LIBS}
  )

  add_executable(ugu_synthesis_test
  test_synthesis.cc)
  target_include_directories(ugu_synthesis_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_synthesis_test
  ${Ugu_LIBS}
  )

  add_executable(ugu_adjacency_test
  test_adjacency.cc)
  target_include_directories(ugu_adjacency_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_adjacency_test
  ${Ugu_LIBS}
  )

  add_executable(ugu_inflation_test
  test_inflation.cc)
  target_include_directories(ugu_inflation_test PRIVATE ${Ugu_INCLUDE_DIRS})
  target_link_libraries(ugu_inflation_test
  ${Ugu_LIBS}
  )

endif()

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