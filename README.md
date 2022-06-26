# **UGU**: **U**nclearness **G**eometry **U**tility

**UGU** is a small geometry library which implements camera, image, mesh, etc.

# Modules

<details>
<summary>common</summary>

- Camera
- Image
- Mesh
- Line
- Log
- Timer
- Face adjacency

</details>

<details>
<summary>accel</summary>

## Acceleration data structure

- BVH
- KD Tree

</details>

<details>
<summary>clustering</summary>

## 3D point cloud (and higher order points) clustering

- k-means
- Mean Shift
- DBSCAN

</details>

<details>
<summary>decimation</summary>

## Mesh Simplification

- QSlim like Quadratic Error Metric based method (WIP)

</details>

<details>
<summary>external</summary>

## Wrappers for external modules

- Fast Quadric Mesh Simplification
- mvs-texturing

</details>

<details>
<summary> geodesic </summary>

## Geodesic Distance

- Dijkstra
- Fast Marching Method (FMM)

</details>

<details>
<summary>inflation</summary>

## Inflation from silhouette

"Notes on Inflating Curves" [Baran and Lehtinen 2009].

</details>

<details>
<summary> inpaint </summary>

## Image inpainting

Telea, Alexandru. "An image inpainting technique based on the fast marching method." Journal of graphics tools 9.1 (2004): 23-34.

</details>

<details>
<summary>optimizer</summary>

## Gradient based optimization

- Numerical Garadient vector / Hessian computation
- Gradient Descent
- Newton
- QuasiNewton (LBFGS)

</details>

<details>
<summary>renderer</summary>

## CPU Rendering

- in : mesh, camera parameters
- out: color, depth, normal, mask and face id

<img src="https://raw.githubusercontent.com/wiki/unclearness/ugu/images/renderer/bunny_realtime.gif" width="640">

- Original code: https://github.com/unclearness/currender
</details>

<details>
<summary>sfs</summary>

## Shape-from-Silhouette (SfS)

- in : silhouettes, camera parameters
- out: mesh

<img src="https://raw.githubusercontent.com/wiki/unclearness/vacancy/images/how_it_works.gif" width="640">

- Original code: https://github.com/unclearness/vacancy
</details>

<details>
<summary>stereo</summary>

## Dense Stereo

- PatchMatch Stereo
  - in : rectified left and right image pair
  - out: disparity, cost, plane image, left right consistency mask, and depth (if baseline and intrinsics are avairable)
  - Speed: approx. 90 sec. with post-process for middlebury dataset third size (463 \* 370)
  - Reference: Bleyer, Michael, Christoph Rhemann, and Carsten Rother. "PatchMatch Stereo-Stereo Matching with Slanted Support Windows." Bmvc. Vol. 11. 2011.

<img src="https://raw.githubusercontent.com/wiki/unclearness/ugu/images/patchmatch_stereo.gif" width="640">
</details>

<details>
<summary>synthesis</summary>

## Patch-based image synthesis

- Reference: Simakov, Denis, et al. "Summarizing visual data using bidirectional similarity." 2008 IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2008.

<img src="https://raw.githubusercontent.com/wiki/unclearness/ugu/images/bdsim_knit.gif" width="320">
</details>

<details>
<summary>textrans</summary>

## Texture transfer

UV texture transfer between almost aligned meshes

</details>

<details>
<summary>texturing</summary>

## Texture Mapping

- in : texture-less mesh, camera parameters, (predefined UV on the mesh, optional)
- out: textured mesh

<img src="https://raw.githubusercontent.com/wiki/unclearness/ugu/images/texture_mapping.png" width="640">
PP
- Comparizon table

|                      | Vertex Color | Projective UV | Tile UV | Triangle UV | Predefined UV |
| -------------------- | :----------: | :-----------: | :-----: | :---------: | :-----------: |
| Runtime Speed        |      ✅      |      ❌       |         |             |               |
| Runtime Memory       |              |               |         |             |               |
| UV Space Efficiency  |              |               |   ❌    |             |      ❓       |
| Quality at Rendering |      ❌      |      ✅       |   ✅    |     ❌      |      ❓       |

</details>

<details>
<summary>util</summary>

## Utilities

- geom
- image
- io
- math
- path
- raster
- rgbd
- string
- thread

</details>

# Dependencies

## Mandatory

- Eigen
  https://github.com/eigenteam/eigen-git-mirror
  - Math

## Optional

- OpenCV
  - cv::Mat\_ as Image class. Image I/O
- stb
  https://github.com/nothings/stb
  - Image I/O
- LodePNG
  https://github.com/lvandeve/lodepng
  - .png I/O particularly for 16bit writing that is not supported by stb
- tinyobjloader
  https://github.com/syoyo/tinyobjloader
  - Load .obj
- tinycolormap
  https://github.com/yuki-koyama/tinycolormap
  - Colorization of depth, face id, etc.
- OpenMP
  (if supported by your compiler)
  - Multi-thread accelaration

# Build

- `git submodule update --init --recursive`
  - To pull dependencies registered as git submodule.
- Use CMake with `CMakeLists.txt`.
  - `reconfigure.bat` and `rebuild.bat` are command line CMake utilities for Windows 10 and Visual Studio 2017.

# Data

Borrowed .obj from [Zhou, Kun, et al. "TextureMontage." ACM Transactions on Graphics (TOG) 24.3 (2005): 1148-1155.](http://www.kunzhou.net/tex-models.htm) for testing purposes.
