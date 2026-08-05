#pragma once

#include <cstdlib>
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

}  // namespace cosmo::nn::ascend
