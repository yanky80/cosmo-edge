#pragma once

#include <cstdint>
#include <string>

#include "rknn_api.h"

namespace cosmo::nn {

// Shared description of the RKNN input tensor memory. Created by RknnNetNode
// during BindInputBlobs and handed to RknnImageToTensorNode through the
// network input blob handle, so RGA can write the RGB letterbox straight into
// the tensor DMA buffer (zero-copy, no host RGB buffer).
struct RknnTensorBinding {
    rknn_context context = 0;
    rknn_tensor_mem* memory = nullptr;
    int width_stride = 0;
    int height_stride = 0;
    int model_width = 0;
    int model_height = 0;
    std::string model_path;
};

}  // namespace cosmo::nn
