#include "catch_amalgamated.hpp"
#include "nn/core/blob.h"
#include "nn/core/common.h"
#define private public
#include "nn/core/graph.h"
#undef private
#include "nn/node/input_node.h"
#include "nn/node/net_node.h"
#include "nn/node/node_type_utils.h"

#ifdef COSMO_NN_USE_HOST_BACKEND
#include "nn/device/host/host_node_factory.h"
#include "nn/node/node_type.h"
#endif

namespace {
class BindingNetNode final : public cosmo::nn::NetNode {
public:
    explicit BindingNetNode(void* external_base) : external_base_(external_base) {
        SetNodeName("binding_net");
        SetBottomBlobNames({"bound_input"});
        SetNetworkInputNames({"bound_input"});
    }

    cosmo::nn::Status LoadWeight(const char*, size_t) override {
        return cosmo::nn::COSMO_NN_OK;
    }

    cosmo::nn::Status BindInputBlobs(std::vector<std::shared_ptr<cosmo::nn::Blob>>& bottom_blobs) override {
        ++bind_calls_;
        auto handle       = bottom_blobs.at(0)->GetHandle();
        handle.base       = external_base_;
        handle.ownership  = cosmo::nn::BLOB_HANDLE_EXTERNAL_OWNED;
        bottom_blobs.at(0)->SetHandle(handle);
        return cosmo::nn::COSMO_NN_OK;
    }

    int bind_calls() const {
        return bind_calls_;
    }

private:
    void* external_base_ = nullptr;
    int bind_calls_      = 0;
};
}  // namespace

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

TEST_CASE("graph binds backend-owned inputs before blob allocation", "[nn][backend]") {
    using namespace cosmo::nn;

    Graph graph;
    graph.blob_store = std::make_unique<BlobStore>(DEVICE_NAIVE, 0);

    BlobDesc desc;
    desc.name        = "bound_input";
    desc.device_type = DEVICE_NAIVE;
    desc.data_type   = DATA_TYPE_FLOAT;
    desc.dims        = {1, 4};

    auto blob = std::make_shared<Blob>(desc);
    graph.blob_store->AddBlob(blob);

    float backend_buffer[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto net_node           = std::make_unique<BindingNetNode>(backend_buffer);
    auto* raw_net_node      = net_node.get();
    graph.nodes.emplace_back(std::move(net_node));

    REQUIRE(bool(graph.BindNetInputs()));
    CHECK(raw_net_node->bind_calls() == 1);
    CHECK(blob->GetHandle().base == backend_buffer);
    CHECK(blob->GetHandle().ownership == BLOB_HANDLE_EXTERNAL_OWNED);

    REQUIRE(bool(graph.blob_store->AllocaAllBlobs()));
    CHECK(blob->GetHandle().base == backend_buffer);

    REQUIRE(bool(graph.blob_store->FreeBlob(blob)));
    CHECK(blob->GetHandle().base == backend_buffer);
    CHECK(blob->GetHandle().ownership == BLOB_HANDLE_EXTERNAL_OWNED);
}

#ifdef COSMO_NN_USE_CPU_BACKEND
TEST_CASE("graph skips host-output copy nodes for host net backends", "[nn][backend]") {
    using namespace cosmo::nn;

    Graph graph;
    graph.device_type_   = DEVICE_CPU;
    graph.max_batch_size = 1;
    graph.blob_store     = std::make_unique<BlobStore>(DEVICE_NAIVE, 0);

    auto input = std::make_unique<InputNode>();
    input->SetNodeName("input_0");
    input->SetTopBlobNames({"input_0/0"});
    graph.nodes.emplace_back(std::move(input));

    ModelInfo model;
    model.input_node_infos.push_back(InputNodeInfo{.name = "image", .shape = {1, 3, 8, 8}, .data_type = 0});
    model.output_node_infos.push_back(OutputNodeInfo{.name = "scores", .shape = {1, 4}, .data_type = 0});

    std::set<size_t> used_prev_outputs;
    std::vector<std::string> current_model_output_blobs;
    REQUIRE(bool(graph.WireNetNode(model, {"input_0/0"}, {}, {}, used_prev_outputs, {"image"}, {"scores"},
                                   current_model_output_blobs)));

    CHECK(NodeTypeUtils::TypedNodeCount(graph.nodes, NODE_NET) == 1);
    CHECK(NodeTypeUtils::TypedNodeCount(graph.nodes, NODE_COPY) == 0);
    REQUIRE(current_model_output_blobs.size() == 1);
    CHECK(current_model_output_blobs.front() == "net_0/0");
}
#endif

#ifdef COSMO_NN_USE_HOST_BACKEND
TEST_CASE("host node factory excludes network runtime ownership", "[nn][backend]") {
    using namespace cosmo::nn;

    CHECK(CreateHostNode(NODE_RESIZE) != nullptr);
    CHECK(CreateHostNode(NODE_NORMALIZE) != nullptr);
    CHECK(CreateHostNode(NODE_NET) == nullptr);
    CHECK(CreateHostNode(NODE_UNKNOWN) == nullptr);
}
#endif
