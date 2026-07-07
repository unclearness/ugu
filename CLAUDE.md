# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

UGU (Unclearness Geometry Utility) is a C++17 static library for 3D geometry and image processing at the intersection of computer vision and graphics. The only mandatory dependency is Eigen; everything else in `third_party/` (git submodules) is optional and toggled through CMake.

## Build

```
git submodule update --init --recursive   # required once, pulls all third_party deps
cmake -B win_build                        # configure (VS generator on Windows)
cmake --build win_build --config Release  # build
```

- `reconfigure.bat` / `rebuild.bat` are the Windows convenience wrappers (they use the `win_build/` directory; `reconfigure.bat` ends with an interactive `pause`, so prefer direct cmake commands when automating).
- Static lib outputs to `lib/`, executables to `bin/` (set in CMakeLists, not the build dir).
- CI (`.github/workflows/cmake.yml`) builds Debug on Ubuntu with both clang and gcc; keep both MSVC and clang/gcc compiling. MSVC builds with `/W4`.
- Test data (bunny, buddha) is auto-downloaded and unzipped into `data/` during CMake configure.

There is no unit test suite. The executables in `examples/` (ex01–ex29, one per module) are the de facto smoke tests; run the relevant `bin/exNN_*` binary from the repo root so it can find `data/`.

## CMake options and the feature-flag pattern

Every optional dependency has a `UGU_USE_<NAME>` CMake option (e.g. `UGU_USE_STB`, `UGU_USE_GLFW`, `UGU_USE_CUDA`, `UGU_USE_OPENCV`, `UGU_USE_TBB`). Each enabled option becomes a public compile definition of the same name, and source code guards optional functionality with `#ifdef UGU_USE_<NAME>`. New code touching optional deps must compile with the flag off as well.

Key defaults: stb, tinyobjloader, lodepng, glfw, freetype, tinycolormap, nanort, json, nanoflann, poisson_reconstruction are ON; OpenCV, CUDA, TBB (auto-disabled if not found), mvs-texturing, libigl, cxxopts are OFF. `UGU_BUILD_PYTHON` (nanobind binding in `python/ugu_py.cc`) is OFF.

When UGU is added as a subdirectory of a parent project, examples/apps/GUI apps are not built by default.

## Architecture

- **Image abstraction (`include/ugu/image.h`)**: `ugu::ImageBase` is `cv::Mat` when `UGU_USE_OPENCV` is on; otherwise UGU provides its own cv::Mat-compatible implementation (`Matx`, `Vec*`, `Image1f`/`Image3b`/etc.). All image code must work in both modes — use the `ugu::` type aliases and the cv-style API subset, never OpenCV-only features directly.
- **Module layout**: public headers in `include/ugu/<module>/`, implementations mirrored in `src/<module>/`. Modules are largely independent: `renderer` (CPU raytracer/rasterizer and OpenGL renderer), `voxel` (TSDF fusion, marching cubes), `sfs` (voxel carving), `texturing`, `registration` (rigid/nonrigid ICP), `parameterize`, `decimation`, `geodesic`, `inpaint`, `clustering`, `accel` (kd-tree/BVH, each with a naive backend plus optional nanoflann/nanort backends), etc. New source files must be added to the explicit `UGU_SOURCE` list in `CMakeLists.txt`.
- **External wrappers (`src/external/`, `include/ugu/external/external.h`)**: heavyweight third-party algorithms (Fast-Quadric-Mesh-Simplification, mvs-texturing, libigl, PoissonRecon) are exposed only through thin wrapper functions here, guarded by their `UGU_USE_*` flags.
- **CUDA (`src/cuda/`)**: public API in `include/ugu/cuda/*.h`, host-side wrappers in `src/cuda/*.cc` (compiled with and without CUDA — they `#ifdef UGU_USE_CUDA` between GPU path and CPU fallback), kernels in `*.cu` with declarations in `*.cuh`. Without CUDA only the `.cc` files build, so keep `.cuh`/`.cu` includes inside the guard.
- **GL shaders**: GLSL sources live in `src/shader/glsl/`; `script/glsl2header.py` converts them into the C++ string headers `src/shader/vert.h`, `frag.h`, `geom.h`. Edit the GLSL and regenerate — do not hand-edit those generated headers.
- **Executables**: `examples/` (per-module demos), `app/` (CLI tools like textrans, image3d), `app_gui/` (imgui+glfw+glad mesh viewers). All are wired up via the `setup_exe()` helper in CMakeLists.

## Conventions

- Source files start with a copyright header: `Copyright (C) 20XX, unclearness`.
- Third-party includes are wrapped in `#pragma warning(push, 0)` / `#pragma warning(pop)` on Windows to keep `/W4` clean.
