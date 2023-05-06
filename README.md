# **UGU**: **U**nclearness **G**eometry **U**tility

**UGU** handles the intersection of computer vision and graphics from the point of view of geometries and images.

# Design concept

## Minimal dependency but scalable

Mandatory dependency is only [Eigen](https://gitlab.com/libeigen/eigen).

All other dependencies in `third_party/` are optional. The use of the dependencies can be handled with CMake configuration.

## Easy build on any platform

Most dependencies are kept as `git submodule`.

Build with CMake should pass on Linux (clang/gcc) and Windows (MSVC).

## Compatibility to cv::Mat

For easy integration into OpenCV-based assets, ugu::ImageBase can be replaced with cv::Mat.

# Build

- `git submodule update --init --recursive`
  - To pull dependencies registered as git submodule.
- Use CMake with `CMakeLists.txt`.
  - `reconfigure.bat` and `rebuild.bat` are command line CMake utilities for Windows 10/11 and Visual Studio 2017-2022.
