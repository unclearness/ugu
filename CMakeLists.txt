cmake_minimum_required(VERSION 3.0)

project(cmake_sample LANGUAGES CXX VERSION 0.0.1 DESCRIPTION "cmake sample")

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
find_package(OpenGL REQUIRED)

# global include setting
#include_directories(
#    include
#    src/include
#    third_party)


# sample to make library from source
# SHARED: .dll and .lib
# STATIC: .lib only
add_library(lib_public
    STATIC
    src/public_lib.cpp
    src/private_lib.cpp
)

target_include_directories(lib_public PRIVATE include)
target_include_directories(lib_public PRIVATE src/include)
target_include_directories(lib_public PRIVATE third_party)

set_target_properties(lib_public PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(cmake_sample_run
    pc/main.cpp)

target_include_directories(cmake_sample_run PRIVATE include)

target_link_libraries(cmake_sample_run
    lib_public
    )