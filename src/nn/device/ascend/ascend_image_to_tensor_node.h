#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "media/FrameSurface.h"
#include "nn/node/node.h"

namespace cosmo::nn {

// YOLO letterbox padding gray (114, 114, 114); the model config may override.
inline constexpr int kDefaultPaddingColor = 114;

// Ascend DVPP image_to_tensor node. Consumes an NV12 FrameSurface and
// produces the fixed OM input contract: NCHW FP16 [1,3,H,W] normalized to
// 0..1 RGB, centered-letterboxed with the configured padding color. DVPP runs
// crop/resize/make-border letterbox and NV12->RGB888 color conversion; only
// the /255 + FP16 host conversion happens on the CPU. Input surfaces:
//   - Device (AV_PIX_FMT_ASCEND frames from the custom FFmpeg decoder): DVPP
//     consumes the decoder's DVPP buffer directly, no H2D upload and no
//     intermediate host image (issue #32).
//   - Host NV12 (stage-one fallback for sources without device export): the
//     surface is uploaded to a DVPP buffer first.
// Every host/device transfer and DVPP stage is timed and logged separately.
class AscendImageToTensorNode : public Node {
public:
    AscendImageToTensorNode();
    ~AscendImageToTensorNode() override;

    void LoadParam(Op* op) override;
    Status InferTopShapes() override;
    DeviceType GetTopBlobDeviceType() override;
    size_t GetBottomCount() override;
    size_t GetTopCount() override;

    Status Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                   std::vector<std::shared_ptr<Blob>>& top_blobs) override;

    // Pure letterbox geometry: scale the frame into [target_w x target_h]
    // preserving aspect ratio, center it, and emit even fit dims + offsets.
    static bool ComputeLetterbox(int frame_width, int frame_height, int target_width, int target_height,
                                 int& fit_width, int& fit_height, int& pad_x, int& pad_y);

    // Packed RGB888 (stride bytes per row) -> FP16 NCHW {1,3,H,W} with /255.
    static void NormalizeRgbToFp16Nchw(const uint8_t* rgb, size_t stride, int width, int height,
                                       uint16_t* dst);

private:
    Status EnsureDvpp(const media::FrameSurface& surface, int frame_width, int frame_height);
    Status ValidateSurface(const media::FrameSurface& surface, int frame_width, int frame_height) const;
    Status MakeDvppStatus(int code, const std::string& stage, const std::string& detail,
                          int32_t hi_ret) const;
    void Destroy();

    int input_width_                = 0;
    int input_height_               = 0;
    std::vector<int> padding_color_ = {kDefaultPaddingColor, kDefaultPaddingColor, kDefaultPaddingColor};

    int device_id_        = 0;
    bool device_set_      = false;
    aclrtContext context_ = nullptr;
    int vpc_chn_          = -1;
    bool initialized_     = false;

    void* src_dev_       = nullptr;
    size_t src_dev_size_ = 0;
    void* rgb_dev_       = nullptr;
    size_t rgb_dev_size_ = 0;
    std::vector<uint8_t> rgb_scratch_;
};

}  // namespace cosmo::nn
