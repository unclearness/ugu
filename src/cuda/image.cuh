
namespace ugu {

void BoxFilterCuda3b(int width, int height, void* data, int k);

void ComputeNormalsCudaImpl(int width, int height, float* h_depths,
                            int num_images, const float* h_fx,
                            const float* h_fy, const float* h_cx,
                            const float* h_cy, float* h_normals,
                            float max_connect_z_diff, int step, bool gl_coord);

}  // namespace ugu