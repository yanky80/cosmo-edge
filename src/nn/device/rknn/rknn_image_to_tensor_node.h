#pragma once

#include <cstddef>
#include <vector>

#include "im2d.hpp"
#include "media/FrameSurface.h"
#include "nn/device/rknn/rknn_tensor_binding.h"
#include "nn/node/node.h"

namespace cosmo::nn {

// RK3588 image_to_tensor node. Validates the NV12 DRM PRIME FrameSurface on
// the input blob, then lets RGA convert it into the RGB letterbox written
// directly into the RKNN input tensor DMA buffer exposed by RknnNetNode.
// No intermediate RGB host buffer is created.
class RknnImageToTensorNode : public Node {
public:
    RknnImageToTensorNode();
    ~RknnImageToTensorNode() override;

    void LoadParam(Op* op) override;
    Status InferTopShapes() override;
    DeviceType GetTopBlobDeviceType() override;
    size_t GetBottomCount() override;
    size_t GetTopCount() override;

    Status Forward(std::vector<std::shared_ptr<Blob>>& bottom_blobs,
                   std::vector<std::shared_ptr<Blob>>& top_blobs) override;

    // RGA file descriptors used by the last successful conversion. Lets board
    // checks prove that the RGA destination is the RKNN tensor DMA fd.
    struct RgaStats {
        int last_src_fd = -1;
        int last_dst_fd = -1;
    };
    const RgaStats& GetRgaStats() const {
        return rga_stats_;
    }

    // Centered letterbox geometry used by the last successful conversion, so
    // board checks can map model-space detections back to source coordinates.
    struct LetterboxInfo {
        float scale = 1.0F;
        int pad_x = 0;
        int pad_y = 0;
        int resized_width = 0;
        int resized_height = 0;
    };
    const LetterboxInfo& GetLastLetterbox() const {
        return last_letterbox_;
    }

private:
    struct Letterbox {
        float scale = 1.0F;
        int pad_x = 0;
        int pad_y = 0;
        int resized_width = 0;
        int resized_height = 0;
    };

    Status ValidateSurface(const media::FrameSurface& surface, int frame_width, int frame_height) const;
    Letterbox ComputeLetterbox(int frame_width, int frame_height, const RknnTensorBinding& binding) const;
    Status RunRga(const media::FrameSurface& surface, int frame_width, int frame_height,
                  const RknnTensorBinding& binding, const Letterbox& letterbox);

    int input_width_ = 0;
    int input_height_ = 0;
    std::vector<int> padding_color_ = {114, 114, 114};
    im_rect previous_rect_{-1, -1, -1, -1};
    RgaStats rga_stats_{};
    LetterboxInfo last_letterbox_{};
};

}  // namespace cosmo::nn
