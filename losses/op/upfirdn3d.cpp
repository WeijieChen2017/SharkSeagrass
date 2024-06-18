#include <ATen/ATen.h>
#include <torch/extension.h>

torch::Tensor upfirdn3d_op(const torch::Tensor &input,
                           const torch::Tensor &kernel, int up_x, int up_y, int up_z,
                           int down_x, int down_y, int down_z, int pad_x0, int pad_x1,
                           int pad_y0, int pad_y1, int pad_z0, int pad_z1);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor upfirdn3d(const torch::Tensor &input, const torch::Tensor &kernel,
                        int up_x, int up_y, int up_z, int down_x, int down_y, int down_z,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1, int pad_z0, int pad_z1) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);

  at::DeviceGuard guard(input.device());

  return upfirdn3d_op(input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1,
                      pad_y0, pad_y1, pad_z0, pad_z1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("upfirdn3d", &upfirdn3d, "upfirdn3d (CUDA)");
}
