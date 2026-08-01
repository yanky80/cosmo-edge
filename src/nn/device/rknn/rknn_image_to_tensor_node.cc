#include "nn/device/rknn/rknn_image_to_tensor_node.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "nn/node/node_type_utils.h"

namespace cosmo::nn {

namespace {

constexpr rga_buffer_handle_t kRgaImportFailed = 0;

std::string SurfaceToString(const media::FrameSurface& surface) {
    std::string text = "planes=" + std::to_string(surface.planes.size());
    for (size_t i = 0; i < surface.planes.size(); ++i) {
        const auto& plane = surface.planes[i];
        text += " plane" + std::to_string(i) + "(fd=" + std::to_string(plane.fd) +
                ",offset=" + std::to_string(plane.offset) + ",pitch=" + std::to_string(plane.pitch) +
                ",vertical_stride=" + std::to_string(plane.vertical_stride) +
                ",size=" + std::to_string(plane.size) + ")";
    }
    return text;
}

std::string RgaStatusText(IM_STATUS status) {
    return std::to_string(static_cast<int>(status)) + " (" + imStrError_t(status) + ")";
}

}  // namespace

RknnImageToTensorNode::RknnImageToTensorNode() : Node() {
    node_type     = NodeType::NODE_IMAGE_TO_TENSOR;
    name          = NodeTypeUtils::NodeTypeToStr(NodeType::NODE_IMAGE_TO_TENSOR).append("_0");
    one_blob_only = true;
}

RknnImageToTensorNode::~RknnImageToTensorNode() {}

void RknnImageToTensorNode::LoadParam(Op* op) {
    auto* image_to_tensor = dynamic_cast<ImageToTensor*>(op);
    if (!image_to_tensor)
        return;
    input_width_  = image_to_tensor->input_width;
    input_height_ = image_to_tensor->input_height;
    if (!image_to_tensor->padding_color.empty())
        // The padding is initialized with a byte memset over the tensor buffer,
        // so only the first channel value is representable; non-uniform colors
        // are normalized here instead of silently mis-filling channels.
        padding_color_ = {image_to_tensor->padding_color.front()};
}

Status RknnImageToTensorNode::InferTopShapes() {
    if (input_width_ <= 0 || input_height_ <= 0)
        return Status(COSMO_NN_ERR_PARAM, "image_to_tensor requires a positive input_size");
    top_blob_shapes     = {{1, input_height_, input_width_, 3}};
    top_blob_data_types = {DATA_TYPE_UINT8};
    return COSMO_NN_OK;
}

DeviceType RknnImageToTensorNode::GetTopBlobDeviceType() {
    return DeviceType::DEVICE_RKNN;
}

size_t RknnImageToTensorNode::GetBottomCount() {
    return 1;
}

size_t RknnImageToTensorNode::GetTopCount() {
    return 1;
}

Status RknnImageToTensorNode::ValidateSurface(const media::FrameSurface& surface, int frame_width,
                                              int frame_height) const {
    if (surface.memory_type != media::FrameSurfaceMemoryType::DmaBuf)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor input is not a DMA-BUF surface");
    if (!surface.IsValid())
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor surface is invalid " +
                                                     SurfaceToString(surface));
    if (surface.planes.size() != 2)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor expects an NV12 surface with two planes " +
                                                     SurfaceToString(surface));
    if (frame_width <= 0 || frame_height <= 0 || (frame_width & 1) != 0 || (frame_height & 1) != 0)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 frame dimensions must be positive and even");

    const auto& luma   = surface.planes[0];
    const auto& chroma = surface.planes[1];
    if (luma.fd != chroma.fd || luma.fd < 0)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 planes must share one DMA-BUF fd " +
                                                     SurfaceToString(surface));
    if (luma.size != chroma.size || luma.size > static_cast<size_t>(std::numeric_limits<int>::max()))
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 planes must share one importable DMA-BUF allocation " +
                                                     SurfaceToString(surface));
    if (luma.offset != 0)
        return Status(COSMO_NN_ERR_RKNN_SURFACE,
                      "NV12 luma offset must be zero for RGA import " + SurfaceToString(surface));
    if (luma.pitch < static_cast<size_t>(frame_width) || luma.vertical_stride < static_cast<size_t>(frame_height))
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 pitch/vertical stride is shorter than the frame " +
                                                     SurfaceToString(surface));
    if (chroma.pitch != luma.pitch)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 chroma pitch must match the luma pitch " +
                                                     SurfaceToString(surface));
    if (chroma.offset != luma.pitch * luma.vertical_stride)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 chroma must start right after the luma plane " +
                                                     SurfaceToString(surface));
    if (chroma.vertical_stride * 2 != luma.vertical_stride)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 chroma vertical stride must be half the luma stride " +
                                                     SurfaceToString(surface));
    if (luma.vertical_stride > (luma.size - luma.offset) / luma.pitch ||
        chroma.vertical_stride > (chroma.size - chroma.offset) / chroma.pitch)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "NV12 plane extent exceeds the DMA-BUF allocation " +
                                                     SurfaceToString(surface));
    return COSMO_NN_OK;
}

RknnImageToTensorNode::Letterbox RknnImageToTensorNode::ComputeLetterbox(
    int frame_width, int frame_height, const RknnTensorBinding& binding) const {
    const int image_size = binding.model_width;
    Letterbox letterbox;
    letterbox.scale = std::min(static_cast<float>(image_size) / static_cast<float>(frame_width),
                               static_cast<float>(image_size) / static_cast<float>(frame_height));
    letterbox.resized_width  = std::max(2, static_cast<int>(std::round(frame_width * letterbox.scale)) & ~1);
    letterbox.resized_height = std::max(2, static_cast<int>(std::round(frame_height * letterbox.scale)) & ~1);
    letterbox.resized_width  = std::min(letterbox.resized_width, image_size);
    letterbox.resized_height = std::min(letterbox.resized_height, image_size);
    letterbox.pad_x          = ((image_size - letterbox.resized_width) / 2) & ~1;
    letterbox.pad_y          = ((image_size - letterbox.resized_height) / 2) & ~1;
    return letterbox;
}

Status RknnImageToTensorNode::RunRga(const media::FrameSurface& surface, int frame_width, int frame_height,
                                     const RknnTensorBinding& binding, const Letterbox& letterbox) {
    const int image_size = binding.model_width;
    const auto& luma     = surface.planes[0];

    // Destination: the RKNN input tensor DMA buffer (zero-copy, no host RGB).
    if (binding.memory->size > static_cast<uint32_t>(std::numeric_limits<int>::max()))
        return Status(COSMO_NN_ERR_RKNN_RGA, "RKNN tensor is too large for RGA import model=" +
                                                 binding.model_path + " fd=" +
                                                 std::to_string(binding.memory->fd) + " size=" +
                                                 std::to_string(binding.memory->size));
    const rga_buffer_handle_t output_handle =
        importbuffer_fd(binding.memory->fd, static_cast<int>(binding.memory->size));
    if (output_handle == kRgaImportFailed) {
        return Status(COSMO_NN_ERR_RKNN_RGA, "RGA could not import RKNN tensor fd model=" + binding.model_path +
                                                 " fd=" + std::to_string(binding.memory->fd) +
                                                 " size=" + std::to_string(binding.memory->size) +
                                                 " rga_handle=" + std::to_string(output_handle));
    }
    struct OutputGuard {
        rga_buffer_handle_t value;
        ~OutputGuard() {
            if (value != kRgaImportFailed)
                releasebuffer_handle(value);
        }
    } output_guard{output_handle};

    const rga_buffer_handle_t input_handle = importbuffer_fd(luma.fd, static_cast<int>(luma.size));
    if (input_handle == kRgaImportFailed) {
        return Status(COSMO_NN_ERR_RKNN_RGA, "RGA could not import MPP DMA fd model=" + binding.model_path +
                                                 " fd=" + std::to_string(luma.fd) +
                                                 " size=" + std::to_string(luma.size) +
                                                 " rga_handle=" + std::to_string(input_handle));
    }
    struct InputGuard {
        rga_buffer_handle_t value;
        ~InputGuard() {
            if (value != kRgaImportFailed)
                releasebuffer_handle(value);
        }
    } input_guard{input_handle};

    const rga_buffer_t input = wrapbuffer_handle(input_handle, frame_width, frame_height,
                                                 RK_FORMAT_YCbCr_420_SP,
                                                 static_cast<int>(luma.pitch),
                                                 static_cast<int>(luma.vertical_stride));
    const rga_buffer_t output = wrapbuffer_handle(output_handle, image_size, image_size, RK_FORMAT_RGB_888,
                                                  binding.width_stride, binding.height_stride);

    const im_rect destination{letterbox.pad_x, letterbox.pad_y, letterbox.resized_width,
                              letterbox.resized_height};
    if (destination.x != previous_rect_.x || destination.y != previous_rect_.y ||
        destination.width != previous_rect_.width || destination.height != previous_rect_.height) {
        if (binding.memory->virt_addr == nullptr) {
            return Status(COSMO_NN_ERR_RKNN_RGA, "RKNN input memory is not CPU accessible model=" +
                                                     binding.model_path);
        }
        std::memset(binding.memory->virt_addr, static_cast<unsigned char>(padding_color_.front()),
                    binding.memory->size);
        const int sync_ret = rknn_mem_sync(binding.context, binding.memory, RKNN_MEMORY_SYNC_TO_DEVICE);
        if (sync_ret != RKNN_SUCC) {
            return Status(COSMO_NN_ERR_RKNN_SYNC, "rknn_mem_sync failed model=" + binding.model_path +
                                                      " fd=" + std::to_string(binding.memory->fd) +
                                                      " size=" + std::to_string(binding.memory->size) +
                                                      " rknn_ret=" + std::to_string(sync_ret));
        }
        previous_rect_ = destination;
    }

    const IM_STATUS check_status = imcheck_t(input, output, {}, {}, destination, {}, 0);
    if (check_status != IM_STATUS_NOERROR) {
        return Status(COSMO_NN_ERR_RKNN_RGA, "RGA format check failed model=" + binding.model_path +
                                                 " status=" + RgaStatusText(check_status) +
                                                 " src_fd=" + std::to_string(luma.fd) +
                                                 " dst_fd=" + std::to_string(binding.memory->fd));
    }
    const IM_STATUS process_status = improcess(input, output, {}, {}, destination, {}, IM_SYNC);
    if (process_status != IM_STATUS_SUCCESS) {
        return Status(COSMO_NN_ERR_RKNN_RGA, "RGA convert failed model=" + binding.model_path +
                                                 " status=" + RgaStatusText(process_status) +
                                                 " src_fd=" + std::to_string(luma.fd) +
                                                 " dst_fd=" + std::to_string(binding.memory->fd));
    }

    rga_stats_.last_src_fd = luma.fd;
    rga_stats_.last_dst_fd = binding.memory->fd;
    last_letterbox_        = {letterbox.scale, letterbox.pad_x, letterbox.pad_y, letterbox.resized_width,
                              letterbox.resized_height};
    return COSMO_NN_OK;
}

Status RknnImageToTensorNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                                      std::vector<std::shared_ptr<Blob>>& top_blobs) {
    timer.Start();

    if (bottom_blobs.size() != 1 || !bottom_blobs[0] || !bottom_blobs[0]->GetHandle().base)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor input blob is missing");
    if (top_blobs.size() != 1 || !top_blobs[0] || !top_blobs[0]->GetHandle().base)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor output blob is not bound");

    auto* surface = static_cast<media::FrameSurface*>(bottom_blobs[0]->GetHandle().base);
    if (surface == nullptr)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor input does not carry a FrameSurface");

    const auto& input_desc = bottom_blobs[0]->GetBlobDesc();
    if (input_desc.dims.size() != 4)
        return Status(COSMO_NN_ERR_RKNN_SURFACE, "image_to_tensor input dims must be NHWC {1,H,W,3}");
    const int frame_width  = input_desc.dims[2];
    const int frame_height = input_desc.dims[1];

    auto* binding = static_cast<RknnTensorBinding*>(top_blobs[0]->GetHandle().base);
    if (binding == nullptr || binding->memory == nullptr || binding->context == 0)
        return Status(COSMO_NN_ERR_RKNN_RGA, "image_to_tensor output is not bound to an RKNN tensor model=" +
                                                 (binding ? binding->model_path : std::string()));
    if (binding->model_width <= 0 || binding->model_height <= 0 || binding->model_width != binding->model_height)
        return Status(COSMO_NN_ERR_RKNN_RGA, "image_to_tensor requires a square RKNN input model=" +
                                                 binding->model_path + " w=" +
                                                 std::to_string(binding->model_width) + " h=" +
                                                 std::to_string(binding->model_height));
    if (input_width_ != binding->model_width || input_height_ != binding->model_height) {
        return Status(COSMO_NN_ERR_RKNN_RGA,
                      "image_to_tensor input_size does not match the RKNN model model=" + binding->model_path +
                          " op=" + std::to_string(input_width_) + "x" + std::to_string(input_height_) +
                          " model=" + std::to_string(binding->model_width) + "x" +
                          std::to_string(binding->model_height));
    }

    RETURN_ON_FAIL(ValidateSurface(*surface, frame_width, frame_height));

    const Letterbox letterbox = ComputeLetterbox(frame_width, frame_height, *binding);
    RETURN_ON_FAIL(RunRga(*surface, frame_width, frame_height, *binding, letterbox));

    timer.Stop();
    return COSMO_NN_OK;
}

}  // namespace cosmo::nn
