#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cosmo::media {

enum class FrameSurfaceMemoryType {
    Host,
    Device,
    DmaBuf,
};

struct FramePlane {
    int fd               = -1;
    uint8_t* virt_addr   = nullptr;
    size_t offset        = 0;
    size_t pitch         = 0;
    size_t vertical_stride = 0;
    size_t size          = 0;

    [[nodiscard]] bool IsValid(FrameSurfaceMemoryType memory_type) const {
        if (fd < -1 || size == 0 || pitch == 0 || offset >= size || pitch > size - offset) {
            return false;
        }
        if (memory_type == FrameSurfaceMemoryType::Host && virt_addr == nullptr) {
            return false;
        }
        if (memory_type == FrameSurfaceMemoryType::DmaBuf && fd < 0) {
            return false;
        }
        return memory_type == FrameSurfaceMemoryType::Device || fd >= 0 || virt_addr != nullptr;
    }
};

struct FrameSurface {
    FrameSurfaceMemoryType memory_type = FrameSurfaceMemoryType::Host;
    std::vector<FramePlane> planes;
    std::shared_ptr<void> lifetime;

    [[nodiscard]] bool IsValid() const {
        if (planes.empty()) {
            return false;
        }
        for (const auto& plane : planes) {
            if (!plane.IsValid(memory_type)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] size_t TotalSize() const {
        size_t total = 0;
        for (const auto& plane : planes) {
            total += plane.size;
        }
        return total;
    }

    [[nodiscard]] uint8_t* GetPlaneData(size_t index) const {
        if (index >= planes.size() || planes[index].virt_addr == nullptr) {
            return nullptr;
        }
        return planes[index].virt_addr + planes[index].offset;
    }

    [[nodiscard]] uint8_t* GetContiguousData() const {
        if (memory_type != FrameSurfaceMemoryType::Host || planes.empty() || planes[0].virt_addr == nullptr) {
            return nullptr;
        }
        auto* base = planes[0].virt_addr;
        for (const auto& plane : planes) {
            if (plane.virt_addr != nullptr && plane.virt_addr != base) {
                return nullptr;
            }
        }
        return base + planes[0].offset;
    }
};

using FrameSurfacePtr = std::shared_ptr<FrameSurface>;

}  // namespace cosmo::media
