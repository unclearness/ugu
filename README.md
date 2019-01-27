# Currender: A computer vision friendly CPU rendering library
**Currender** is a CPU raytracing based rendering library written in C++.
With triangular mesh and camera parameters, you can easily render 

- Color image
- Depth image
- Normal image
- Mask image

Pros and cons against popular OpenGL based rendering are listed below.
## Pros
- **Simple API, set mesh, set camera and render**. You do not need to care about complex OpenGL settings.
- **Standard coordinate system in computer vision community identical to OpenCV** (right-handed, z:forward, y:down, x:right). You are not annoyed with coordinate conversion for OpenGL.
- **Pixel-scale intrinsic parameters (principal point and focal length) with pinhole camera model**, which are popular camera projection representation in computer vision. You are not annoyed with converting the intrinsics to perspective projection matrix for OpenGL.
- **Rendering depth, normal and mask image** besides color image is enabled as default. They are frequently used in computer vision algorithm.
- **Fast for lower resolution**. Enough speed with less than VGA (640 * 480). Such small image size is commonly used in computer vison algorithm.
- **Rendered images are directly stored in RAM**. Easy to pass them to other CPU based programs.

## Cons
- Slow for higher resolution due to the nature of raytracing.
- Showing images on window is not supported. You should use external libraries for visualization.
- Not desgined to render beautiful and realistic color images. Only simple diffuse shading is implemented. 



# Use case
Expected use cases are the following but not limited to
- Embedded in computer vision algortihm with rendering. 
    - Especially in the case that OpenGL based visualization is running on the main thread and on another thread computer vision algorithm should render images from explicit 3D model without blocking the visualization.
- Data augumentation for machine learning.

# Dependencies
## Mandatory
- GLM
    https://github.com/g-truc/glm
    - Math
- NanoRT
    https://github.com/lighttransport/nanort
    - Ray intersection with BVH
## Optional
- stb
    https://github.com/nothings/stb
    - Image I/O
- tinyobjloader
    https://github.com/syoyo/tinyobjloader
    - Load .obj
- OpenMP
    (if supported by your compiler)
    - Multi-thread accelaration

# Build
- git submodule update --init --recursive
  - To pull dependencies registered as git submodule. 
- Use CMake with CMakeLists.txt.
  -  reconfigure.bat and rebuild.bat are command line CMake utilities for Windows 10 and Visual Studio 2017.

# Platforms
Tested on Windows 10 and Visual Studio 2017.
Porting to the other platforms (Linux, Android, Mac and iOS) is under planning.
Minor modifitation of code and CMakeLists.txt would be required.

# To do
- Porting to other platforms.
- Real-time rendering visualization sample with external library (maybe OpenGL).
- Support point cloud rendering.
- Introduce ambient and specular.

