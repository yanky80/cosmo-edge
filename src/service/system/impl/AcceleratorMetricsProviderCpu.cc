#if defined(COSMO_NN_USE_CPU_BACKEND) || defined(COSMO_NN_USE_RKNN_BACKEND) || \
    defined(COSMO_NN_USE_ASCEND_BACKEND)

#include "service/system/impl/AcceleratorMetricsProvider.h"

namespace cosmo::service::detail {
namespace {

    class CpuAcceleratorMetricsProvider final : public AcceleratorMetricsProvider {
    public:
        cosmo::MsgGpuInfo QueryUtilization() override {
            return {};
        }

        int64_t QueryAvailableMemoryMB() override {
            return 0;
        }
    };

}  // namespace

std::unique_ptr<AcceleratorMetricsProvider> CreateAcceleratorMetricsProvider() {
    return std::make_unique<CpuAcceleratorMetricsProvider>();
}

}  // namespace cosmo::service::detail

#endif  // CPU/RKNN/Ascend backends use the host metrics fallback.
