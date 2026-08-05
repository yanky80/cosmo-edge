#include "nn/device/ascend/ascend_image_to_tensor_node.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "acl/dvpp/hi_dvpp.h"
#include "nn/device/ascend/ascend_acl.h"
#include "nn/node/node_type_utils.h"
#include "nn/utils/data_type_utils.h"
#include "util/Log.h"

namespace cosmo::nn {

namespace {

    // The DVPP VPC subsystem needs hi_mpi_sys_init() before the first channel is
    // created (HI_ERR_VPC_SYS_NOT_READY otherwise). The Ascend media decoder
    // initializes the same subsystem, but the node must not depend on that.
    std::once_flag g_dvpp_sys_init_flag;

    size_t Align32(size_t size) {
        return (size + 31U) & ~size_t{31U};
    }

    std::string SurfaceToString(const media::FrameSurface& surface) {
        std::string text = "planes=" + std::to_string(surface.planes.size());
        for (size_t i = 0; i < surface.planes.size(); ++i) {
            const auto& plane = surface.planes[i];
            text += " plane" + std::to_string(i) +
                    "(virt=" + std::to_string(reinterpret_cast<uintptr_t>(plane.virt_addr)) +
                    ",offset=" + std::to_string(plane.offset) + ",pitch=" + std::to_string(plane.pitch) +
                    ",vertical_stride=" + std::to_string(plane.vertical_stride) +
                    ",size=" + std::to_string(plane.size) + ")";
        }
        return text;
    }

}  // namespace

AscendImageToTensorNode::AscendImageToTensorNode() : Node() {
    node_type     = NodeType::NODE_IMAGE_TO_TENSOR;
    name          = NodeTypeUtils::NodeTypeToStr(NodeType::NODE_IMAGE_TO_TENSOR).append("_0");
    one_blob_only = true;
}

AscendImageToTensorNode::~AscendImageToTensorNode() {
    Destroy();
}

void AscendImageToTensorNode::LoadParam(Op* op) {
    auto* image_to_tensor = dynamic_cast<ImageToTensor*>(op);
    if (!image_to_tensor)
        return;
    input_width_  = image_to_tensor->input_width;
    input_height_ = image_to_tensor->input_height;
    if (!image_to_tensor->padding_color.empty())
        padding_color_ = image_to_tensor->padding_color;
}

Status AscendImageToTensorNode::InferTopShapes() {
    if (input_width_ <= 0 || input_height_ <= 0)
        return Status(COSMO_NN_ERR_PARAM, "image_to_tensor requires a positive input_size");
    // NCHW {1,3,H,W}: the Ascend OM contract (image_to_tensor owns the
    // letterbox, /255 normalization and NV12->RGB888, so no static AIPP).
    top_blob_shapes     = {{1, 3, input_height_, input_width_}};
    top_blob_data_types = {DATA_TYPE_HALF};
    return COSMO_NN_OK;
}

DeviceType AscendImageToTensorNode::GetTopBlobDeviceType() {
    return DeviceType::DEVICE_NAIVE;
}

size_t AscendImageToTensorNode::GetBottomCount() {
    return 1;
}

size_t AscendImageToTensorNode::GetTopCount() {
    return 1;
}

Status AscendImageToTensorNode::ValidateSurface(const media::FrameSurface& surface, int frame_width,
                                                int frame_height) const {
    if (surface.memory_type != media::FrameSurfaceMemoryType::Host)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT, "ascend image_to_tensor input is not a host surface");
    if (!surface.IsValid())
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor surface is invalid " + SurfaceToString(surface));
    if (surface.planes.size() != 2)
        return Status(
            COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
            "ascend image_to_tensor expects an NV12 surface with two planes " + SurfaceToString(surface));
    if (frame_width <= 0 || frame_height <= 0 || (frame_width & 1) != 0 || (frame_height & 1) != 0)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor NV12 frame dimensions must be positive and even");

    const auto& luma   = surface.planes[0];
    const auto& chroma = surface.planes[1];
    if (luma.pitch < static_cast<size_t>(frame_width) || luma.pitch < static_cast<size_t>(frame_height) / 2 ||
        luma.pitch % 16 != 0)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor NV12 luma pitch must be >= width and 16-byte aligned for "
                      "DVPP VPC " +
                          SurfaceToString(surface));
    if (chroma.pitch != luma.pitch)
        return Status(
            COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
            "ascend image_to_tensor NV12 chroma pitch must match the luma pitch " + SurfaceToString(surface));
    if (luma.vertical_stride < static_cast<size_t>(frame_height) ||
        chroma.vertical_stride < static_cast<size_t>(frame_height) / 2)
        return Status(
            COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
            "ascend image_to_tensor NV12 plane stride is shorter than the frame " + SurfaceToString(surface));
    if (luma.pitch * static_cast<size_t>(frame_height) > luma.size ||
        luma.pitch * static_cast<size_t>(frame_height) / 2 > chroma.size)
        return Status(
            COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
            "ascend image_to_tensor NV12 plane span exceeds its backing buffer " + SurfaceToString(surface));
    return COSMO_NN_OK;
}

bool AscendImageToTensorNode::ComputeLetterbox(int frame_width, int frame_height, int target_width,
                                               int target_height, int& fit_width, int& fit_height, int& pad_x,
                                               int& pad_y) {
    if (frame_width <= 0 || frame_height <= 0 || target_width <= 0 || target_height <= 0)
        return false;
    const double scale = std::min(static_cast<double>(target_width) / frame_width,
                                  static_cast<double>(target_height) / frame_height);
    fit_width          = std::min(target_width, static_cast<int>(std::lround(frame_width * scale)));
    fit_height         = std::min(target_height, static_cast<int>(std::lround(frame_height * scale)));
    fit_width &= ~1;
    fit_height &= ~1;
    if (fit_width < 2)
        fit_width = 2;
    if (fit_height < 2)
        fit_height = 2;
    pad_x = (target_width - fit_width) / 2;
    pad_y = (target_height - fit_height) / 2;
    return fit_width <= target_width && fit_height <= target_height;
}

void AscendImageToTensorNode::NormalizeRgbToFp16Nchw(const uint8_t* rgb, size_t stride, int width, int height,
                                                     uint16_t* dst) {
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgb + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) {
            const uint8_t* pixel              = row + static_cast<size_t>(x) * 3;
            dst[(0 * height + y) * width + x] = FloatToFp16(pixel[0] / 255.0F);
            dst[(1 * height + y) * width + x] = FloatToFp16(pixel[1] / 255.0F);
            dst[(2 * height + y) * width + x] = FloatToFp16(pixel[2] / 255.0F);
        }
    }
}

Status AscendImageToTensorNode::MakeDvppStatus(int code, const std::string& stage, const std::string& detail,
                                               int32_t hi_ret) const {
    std::string message = "stage=ascend-dvpp-" + stage + " device=" + std::to_string(device_id_);
    if (!detail.empty())
        message += " " + detail;
    message += " hi_ret=" + std::to_string(hi_ret);
    return Status(code, message);
}

Status AscendImageToTensorNode::EnsureDvpp(const media::FrameSurface& surface, int frame_width,
                                           int frame_height) {
    const size_t src_bytes = Align32(surface.planes[0].pitch * static_cast<size_t>(frame_height) * 3 / 2);
    const size_t rgb_bytes = Align32(static_cast<size_t>(input_width_) * input_height_ * 3);

    if (!initialized_) {
        const aclError init_ret = ascend::EnsureAclInitialized();
        if (init_ret != ACL_SUCCESS)
            return MakeDvppStatus(
                COSMO_NN_ERR_ASCEND_INIT, "init",
                "aclInit src=" + std::to_string(frame_width) + "x" + std::to_string(frame_height), init_ret);
        const aclError dev_ret = ascend::AcquireDevice(device_id_);
        if (dev_ret != ACL_SUCCESS)
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DEVICE, "device", "", dev_ret);
        device_set_ = true;

        aclError ret = aclrtCreateContext(&context_, device_id_);
        if (ret != ACL_SUCCESS) {
            Destroy();
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "context", "", ret);
        }
        ret = aclrtSetCurrentContext(context_);
        if (ret != ACL_SUCCESS) {
            Destroy();
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "context-set", "", ret);
        }
        initialized_ = true;
    }

    const aclError ctx_ret = aclrtSetCurrentContext(context_);
    if (ctx_ret != ACL_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_CONTEXT, "context-set", "", ctx_ret);

    // One channel for the whole stream, created once. Every Forward pops its
    // completed task via hi_mpi_vpc_get_process_result(); without that pop the
    // SDK send ring fills after 16 tasks and the next submission wedges.
    if (vpc_chn_ < 0) {
        std::call_once(g_dvpp_sys_init_flag, [] {
            if (hi_mpi_sys_init() != HI_SUCCESS)
                LOG_WARN("ascend dvpp hi_mpi_sys_init failed");
        });
        hi_vpc_chn_attr attr{};
        attr.pic_width          = static_cast<hi_u32>(std::max(frame_width, input_width_));
        attr.pic_height         = static_cast<hi_u32>(std::max(frame_height, input_height_));
        hi_vpc_chn chn          = -1;
        const hi_s32 create_ret = hi_mpi_vpc_sys_create_chn(&chn, &attr);
        if (create_ret != HI_SUCCESS)
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DVPP_INIT, "channel-create",
                                  "max=" + std::to_string(std::max(frame_width, input_width_)) + "x" +
                                      std::to_string(std::max(frame_height, input_height_)),
                                  create_ret);
        vpc_chn_ = static_cast<int>(chn);
    }

    if (src_dev_size_ < src_bytes) {
        if (src_dev_ != nullptr) {
            (void)hi_mpi_dvpp_free(src_dev_);
            src_dev_ = nullptr;
        }
        const hi_s32 alloc_ret = hi_mpi_dvpp_malloc(static_cast<hi_u32>(device_id_), &src_dev_, src_bytes);
        if (alloc_ret != HI_SUCCESS)
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DVPP_MEM, "malloc",
                                  "src bytes=" + std::to_string(src_bytes), alloc_ret);
        src_dev_size_ = src_bytes;
    }
    if (rgb_dev_ == nullptr) {
        const hi_s32 alloc_ret = hi_mpi_dvpp_malloc(static_cast<hi_u32>(device_id_), &rgb_dev_, rgb_bytes);
        if (alloc_ret != HI_SUCCESS)
            return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DVPP_MEM, "malloc",
                                  "rgb bytes=" + std::to_string(rgb_bytes), alloc_ret);
        rgb_dev_size_ = rgb_bytes;
    }
    try {
        rgb_scratch_.resize(static_cast<size_t>(input_width_) * input_height_ * 3);
    } catch (const std::bad_alloc&) {
        return Status(COSMO_NN_ERR_OUT_OF_MEMORY, "stage=ascend-dvpp device=" + std::to_string(device_id_) +
                                                      " rgb scratch allocation failed");
    }
    return COSMO_NN_OK;
}

Status AscendImageToTensorNode::Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                                        std::vector<std::shared_ptr<Blob>>& top_blobs) {
    using clock = std::chrono::steady_clock;
    timer.Start();
    const auto frame_start = clock::now();

    if (bottom_blobs.size() != 1 || !bottom_blobs[0] || !bottom_blobs[0]->GetHandle().base)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT, "ascend image_to_tensor input blob is missing");
    if (top_blobs.size() != 1 || !top_blobs[0] || !top_blobs[0]->GetHandle().base)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT, "ascend image_to_tensor output blob is not bound");

    auto* surface = static_cast<media::FrameSurface*>(bottom_blobs[0]->GetHandle().base);
    if (surface == nullptr)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor input does not carry a FrameSurface");

    const auto& input_desc = bottom_blobs[0]->GetBlobDesc();
    if (input_desc.dims.size() != 4)
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor input dims must be NHWC {1,H,W,3}");
    const int frame_width  = input_desc.dims[2];
    const int frame_height = input_desc.dims[1];

    RETURN_ON_FAIL(ValidateSurface(*surface, frame_width, frame_height));
    RETURN_ON_FAIL(EnsureDvpp(*surface, frame_width, frame_height));

    const auto& luma        = surface->planes[0];
    const auto& chroma      = surface->planes[1];
    const size_t pitch      = luma.pitch;
    const size_t luma_row   = static_cast<size_t>(frame_height);
    const size_t chroma_row = static_cast<size_t>(frame_height) / 2;

    // Upload NV12 (luma + interleaved chroma) into one contiguous DVPP buffer.
    aclError ret =
        aclrtMemcpy(src_dev_, pitch * luma_row, luma.virt_addr, pitch * luma_row, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_MEMCPY, "upload-y",
                              "pitch=" + std::to_string(pitch) + " rows=" + std::to_string(luma_row), ret);
    ret = aclrtMemcpy(static_cast<uint8_t*>(src_dev_) + pitch * luma_row, pitch * chroma_row,
                      chroma.virt_addr, pitch * chroma_row, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_MEMCPY, "upload-uv",
                              "pitch=" + std::to_string(pitch) + " rows=" + std::to_string(chroma_row), ret);
    const auto upload_done = clock::now();

    // DVPP: centered letterbox + NV12->RGB888 in a single crop+resize+make-border
    // task. The RGB888 destination width_stride is in bytes (width*3), not pixels.
    int fit_width  = 0;
    int fit_height = 0;
    int pad_x      = 0;
    int pad_y      = 0;
    if (!ComputeLetterbox(frame_width, frame_height, input_width_, input_height_, fit_width, fit_height,
                          pad_x, pad_y))
        return Status(COSMO_NN_ERR_ASCEND_DVPP_FORMAT,
                      "ascend image_to_tensor letterbox geometry failed frame=" +
                          std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                          " target=" + std::to_string(input_width_) + "x" + std::to_string(input_height_));

    hi_vpc_pic_info src_pic{};
    src_pic.picture_address       = src_dev_;
    src_pic.picture_buffer_size   = static_cast<hi_u32>(src_dev_size_);
    src_pic.picture_width         = static_cast<hi_u32>(frame_width);
    src_pic.picture_height        = static_cast<hi_u32>(frame_height);
    src_pic.picture_width_stride  = static_cast<hi_u32>(pitch);
    src_pic.picture_height_stride = static_cast<hi_u32>(frame_height);
    src_pic.picture_format        = HI_PIXEL_FORMAT_YUV_SEMIPLANAR_420;

    hi_vpc_pic_info rgb_pic{};
    rgb_pic.picture_address       = rgb_dev_;
    rgb_pic.picture_buffer_size   = static_cast<hi_u32>(rgb_dev_size_);
    rgb_pic.picture_width         = static_cast<hi_u32>(input_width_);
    rgb_pic.picture_height        = static_cast<hi_u32>(input_height_);
    rgb_pic.picture_width_stride  = static_cast<hi_u32>(input_width_) * 3;
    rgb_pic.picture_height_stride = static_cast<hi_u32>(input_height_);
    rgb_pic.picture_format        = HI_PIXEL_FORMAT_RGB_888;

    hi_vpc_crop_resize_border_region border_region{};
    border_region.dest_pic_info             = rgb_pic;
    border_region.crop_region.top_offset    = 0;
    border_region.crop_region.left_offset   = 0;
    border_region.crop_region.crop_width    = static_cast<hi_u32>(frame_width);
    border_region.crop_region.crop_height   = static_cast<hi_u32>(frame_height);
    border_region.resize_info.resize_width  = static_cast<hi_u32>(fit_width);
    border_region.resize_info.resize_height = static_cast<hi_u32>(fit_height);
    border_region.resize_info.interpolation = 0;  // bilinear
    border_region.dest_top_offset           = static_cast<hi_u32>(pad_y);
    border_region.dest_left_offset          = static_cast<hi_u32>(pad_x);
    border_region.border_type               = HI_BORDER_CONSTANT;
    border_region.scalar_value.val[0] = padding_color_.size() > 0 ? padding_color_[0] : kDefaultPaddingColor;
    border_region.scalar_value.val[1] = padding_color_.size() > 1 ? padding_color_[1] : kDefaultPaddingColor;
    border_region.scalar_value.val[2] = padding_color_.size() > 2 ? padding_color_[2] : kDefaultPaddingColor;

    hi_u32 task_id          = 0;
    const hi_s32 border_ret = hi_mpi_vpc_crop_resize_make_border(static_cast<hi_vpc_chn>(vpc_chn_), &src_pic,
                                                                 &border_region, 1, &task_id, -1);
    if (border_ret != HI_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DVPP_EXECUTE, "letterbox",
                              "frame=" + std::to_string(frame_width) + "x" + std::to_string(frame_height) +
                                  " fit=" + std::to_string(fit_width) + "x" + std::to_string(fit_height) +
                                  " pad=" + std::to_string(pad_x) + "," + std::to_string(pad_y),
                              border_ret);
    // Block until the VPC task has actually finished. Without this the D2H
    // copy below races the VPC engine and captures a half-written buffer (on
    // the 310P3 the bottom half of the frame is all zeros). Awaiting the task
    // also pops it from the channel's send ring, so the ring cannot fill up
    // and block later submissions.
    const hi_s32 wait_ret = hi_mpi_vpc_get_process_result(static_cast<hi_vpc_chn>(vpc_chn_), task_id, -1);
    if (wait_ret != HI_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_DVPP_EXECUTE, "letterbox-wait",
                              "task_id=" + std::to_string(task_id), wait_ret);
    const auto dvpp_done = clock::now();

    // Download RGB888 for the host-side /255 + FP16 conversion.
    const size_t rgb_bytes = static_cast<size_t>(input_width_) * input_height_ * 3;
    ret = aclrtMemcpy(rgb_scratch_.data(), rgb_bytes, rgb_dev_, rgb_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
        return MakeDvppStatus(COSMO_NN_ERR_ASCEND_MEMCPY, "download-rgb",
                              "bytes=" + std::to_string(rgb_bytes), ret);
    const auto download_done = clock::now();

    NormalizeRgbToFp16Nchw(rgb_scratch_.data(), static_cast<size_t>(input_width_) * 3, input_width_,
                           input_height_, static_cast<uint16_t*>(top_blobs[0]->GetHandle().base));
    const auto normalize_done = clock::now();

    const auto us = [](clock::time_point begin, clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    };
    LOG_INFO(
        "ascend image_to_tensor frame={}x{} target={}x{} upload={}us dvpp={}us download={}us "
        "normalize={}us total={}us",
        frame_width, frame_height, input_width_, input_height_, us(frame_start, upload_done),
        us(upload_done, dvpp_done), us(dvpp_done, download_done), us(download_done, normalize_done),
        us(frame_start, normalize_done));

    timer.Stop();
    return COSMO_NN_OK;
}

void AscendImageToTensorNode::Destroy() {
    rgb_scratch_.clear();
    if (rgb_dev_ != nullptr) {
        (void)hi_mpi_dvpp_free(rgb_dev_);
        rgb_dev_      = nullptr;
        rgb_dev_size_ = 0;
    }
    if (src_dev_ != nullptr) {
        (void)hi_mpi_dvpp_free(src_dev_);
        src_dev_      = nullptr;
        src_dev_size_ = 0;
    }
    if (vpc_chn_ >= 0) {
        (void)hi_mpi_vpc_destroy_chn(static_cast<hi_vpc_chn>(vpc_chn_));
        vpc_chn_ = -1;
    }
    if (context_ != nullptr) {
        (void)aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    if (device_set_) {
        (void)ascend::ReleaseDevice(device_id_);
        device_set_ = false;
    }
    initialized_ = false;
}

}  // namespace cosmo::nn
