#include "catch_amalgamated.hpp"
#include "nn/core/blob.h"
#include "nn/core/common.h"

#ifdef COSMO_NN_USE_HOST_BACKEND
#include "nn/device/host/host_node_factory.h"
#include "nn/node/node_type.h"
#endif

TEST_CASE("backend capabilities identify host memory boundaries", "[nn][backend]") {
    using namespace cosmo::nn;

    CHECK(UsesHostMemory(DEVICE_NAIVE));
    CHECK(UsesHostMemory(DEVICE_CPU));
    CHECK_FALSE(UsesHostMemory(DEVICE_SOPHON_TPU));
}

TEST_CASE("backend capabilities derive copy insertion from memory boundaries", "[nn][backend]") {
    using namespace cosmo::nn;

    CHECK_FALSE(NeedsCopyNode(DEVICE_NAIVE, DEVICE_CPU));
    CHECK_FALSE(NeedsCopyNode(DEVICE_CPU, DEVICE_NAIVE));
    CHECK_FALSE(NeedsCopyNode(DEVICE_SOPHON_TPU, DEVICE_SOPHON_TPU));
    CHECK(NeedsCopyNode(DEVICE_NAIVE, DEVICE_SOPHON_TPU));
    CHECK(NeedsCopyNode(DEVICE_CPU, DEVICE_SOPHON_TPU));
}

TEST_CASE("blob descriptors carry affine quantization metadata", "[nn][backend]") {
    using namespace cosmo::nn;

    BlobDesc desc;
    desc.is_affine_quantized = true;
    desc.affine_scale        = 0.25f;
    desc.affine_zero_point   = 17;

    Blob blob(desc);
    const auto roundtrip = blob.GetBlobDesc();

    CHECK(roundtrip.is_affine_quantized);
    CHECK(roundtrip.affine_scale == Catch::Approx(0.25f));
    CHECK(roundtrip.affine_zero_point == 17);
}

#ifdef COSMO_NN_USE_HOST_BACKEND
TEST_CASE("host node factory excludes network runtime ownership", "[nn][backend]") {
    using namespace cosmo::nn;

    CHECK(CreateHostNode(NODE_RESIZE) != nullptr);
    CHECK(CreateHostNode(NODE_NORMALIZE) != nullptr);
    CHECK(CreateHostNode(NODE_NET) == nullptr);
    CHECK(CreateHostNode(NODE_UNKNOWN) == nullptr);
}
#endif
