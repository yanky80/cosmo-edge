#pragma once

#include <cstdlib>
#include <map>
#include <mutex>

#include "acl/acl.h"

namespace cosmo::nn::ascend {

// AscendCL initializes exactly once per process: CANN 8.0 documents that
// aclInit may be called only once, and a second call returns
// ACL_ERROR_REPEAT_INITIALIZE (100002). Every AscendCL entry point in the
// engine routes through this helper; aclFinalize runs once at process exit.
inline aclError EnsureAclInitialized() {
    static std::once_flag init_flag;
    static aclError init_result = ACL_SUCCESS;
    std::call_once(init_flag, []() {
        init_result = aclInit(nullptr);
        if (init_result == ACL_SUCCESS)
            std::atexit([]() { (void)aclFinalize(); });
    });
    return init_result;
}

namespace {
    struct AscendDeviceRefs {
        std::mutex mutex;
        std::map<int, int> counts;
    };

    AscendDeviceRefs& AscendDeviceRefsInstance() {
        static AscendDeviceRefs instance;
        return instance;
    }
}  // namespace

// aclrtSetDevice/aclrtResetDevice govern process/device-scoped resources that
// every graph on the device shares. CANN rejects aclrtResetDevice while
// another graph's context is still live, so the first graph acquires the
// device and the last graph releases it.
inline aclError AcquireDevice(int device_id) {
    AscendDeviceRefs& refs = AscendDeviceRefsInstance();
    std::lock_guard<std::mutex> guard(refs.mutex);
    if (refs.counts[device_id]++ == 0)
        return aclrtSetDevice(device_id);
    return ACL_SUCCESS;
}

inline aclError ReleaseDevice(int device_id) {
    AscendDeviceRefs& refs = AscendDeviceRefsInstance();
    std::lock_guard<std::mutex> guard(refs.mutex);
    auto it = refs.counts.find(device_id);
    if (it == refs.counts.end() || it->second == 0)
        return ACL_SUCCESS;
    if (--it->second == 0) {
        refs.counts.erase(it);
        return aclrtResetDevice(device_id);
    }
    return ACL_SUCCESS;
}

}  // namespace cosmo::nn::ascend
