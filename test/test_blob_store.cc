#include "catch_amalgamated.hpp"
/*
 * test_blob_store.cc - BlobStore regression tests
 *
 * Guards the fix in BlobStore::FreeBlob(shared_ptr&) that clears the blob's
 * BlobHandle after a successful device->Free(). Without the clear, a repeat
 * FreeBlob (or ~BlobStore iterating blobs, or a blob that escaped via
 * GetBlob/GetBlobs and outlives the BlobStore) would free a stale pointer,
 * causing a double-free / use-after-free.
 *
 * Uses the real NaiveDevice (auto-registered for DEVICE_NAIVE, host malloc/free)
 * so the test runs on any backend without TPU access.
 */
#include <memory>
#include <string>
#include <vector>

#include "nn/core/blob.h"
#include "nn/core/blob_store.h"

using namespace cosmo::nn;

namespace {
// Build a float32 blob descriptor with dims {1, 4} on the naive device.
BlobDesc MakeNaiveFloatDesc(const std::string& name) {
    BlobDesc desc;
    desc.device_type = DEVICE_NAIVE;
    desc.data_type   = DATA_TYPE_FLOAT;
    desc.dims        = {1, 4};
    desc.name        = name;
    return desc;
}
}  // namespace

TEST_CASE("BlobStore: AllocaBlob sets a non-null handle", "[BlobStore]") {
    BlobStore store(DEVICE_NAIVE, 0);

    auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("blob0"));
    store.AddBlob(blob);

    REQUIRE(bool(store.AllocaBlob(blob)));
    REQUIRE(blob->GetHandle().base != nullptr);
}

TEST_CASE("BlobStore: FreeBlob clears the handle", "[BlobStore]") {
    BlobStore store(DEVICE_NAIVE, 0);

    auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("blob0"));
    store.AddBlob(blob);
    REQUIRE(bool(store.AllocaBlob(blob)));
    REQUIRE(blob->GetHandle().base != nullptr);

    REQUIRE(bool(store.FreeBlob(blob)));
    REQUIRE(blob->GetHandle().base == nullptr);
}

TEST_CASE("BlobStore: FreeBlob(shared_ptr) is idempotent", "[BlobStore]") {
    BlobStore store(DEVICE_NAIVE, 0);

    auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("blob0"));
    store.AddBlob(blob);
    REQUIRE(bool(store.AllocaBlob(blob)));

    // First free succeeds and nulls the handle.
    REQUIRE(bool(store.FreeBlob(blob)));
    REQUIRE(blob->GetHandle().base == nullptr);

    // Second free must be a no-op (returns OK) and must not crash.
    // Before the fix, the handle was left dangling and this would double-free.
    REQUIRE(bool(store.FreeBlob(blob)));
    REQUIRE(blob->GetHandle().base == nullptr);
}

TEST_CASE("BlobStore: FreeBlob(name) is idempotent", "[BlobStore]") {
    BlobStore store(DEVICE_NAIVE, 0);

    auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("named_blob"));
    store.AddBlob(blob);
    REQUIRE(bool(store.AllocaBlob(blob)));
    REQUIRE(blob->GetHandle().base != nullptr);

    // First free by name succeeds and clears the handle.
    REQUIRE(bool(store.FreeBlob(std::string("named_blob"))));
    REQUIRE(blob->GetHandle().base == nullptr);

    // The blob remains in the store's vector with a null handle, so the name
    // lookup still finds it and the shared_ptr overload early-returns OK.
    // Must not return COSMO_NN_ERR_INVALID_INPUT and must not double-free.
    REQUIRE(bool(store.FreeBlob(std::string("named_blob"))));
    REQUIRE(blob->GetHandle().base == nullptr);
}

TEST_CASE("BlobStore: escaped blob sees null handle after store destruction", "[BlobStore]") {
    // The blob outlives the BlobStore via shared ownership retrieved through
    // GetBlob. After the store is destroyed, its destructor frees and clears
    // the handle, so the external holder must observe null (not dangling).
    std::shared_ptr<Blob> external;
    {
        BlobStore store(DEVICE_NAIVE, 0);

        auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("escaped"));
        store.AddBlob(blob);
        REQUIRE(bool(store.AllocaBlob(blob)));
        REQUIRE(blob->GetHandle().base != nullptr);

        store.GetBlob(std::string("escaped"), external);
        REQUIRE(external != nullptr);
        REQUIRE(external->GetHandle().base != nullptr);
    }  // ~BlobStore frees and clears the handle on every tracked blob.

    // The Blob object is still alive (shared ownership via `external`).
    REQUIRE(external != nullptr);
    // Before the fix, the handle was left dangling here, so a later FreeBlob
    // or another owner freeing it would be a use-after-free.
    REQUIRE(external->GetHandle().base == nullptr);
}

TEST_CASE("BlobStore: FreeBlob on a self-owning blob does not double-free", "[BlobStore]") {
    // A self-owning blob (Blob(desc, true)) allocates its own memory. Clearing
    // the handle via SetHandle would trigger its free-on-replace path and
    // double-free (device->Free already freed it). ClearHandle skips that path,
    // so FreeBlob is safe even for a self-owning blob. In practice BlobStore
    // only holds AllocaBlob'd blobs, but this guards the clearing mechanism.
    BlobStore store(DEVICE_NAIVE, 0);

    auto blob = std::make_shared<Blob>(MakeNaiveFloatDesc("self_owned"), true);
    REQUIRE(blob->GetHandle().base != nullptr);

    store.AddBlob(blob);
    REQUIRE(bool(store.FreeBlob(blob)));
    REQUIRE(blob->GetHandle().base == nullptr);

    // Repeat free is a no-op; blob destruction (~BlobImpl) must not re-free.
    REQUIRE(bool(store.FreeBlob(blob)));
}

TEST_CASE("BlobStore: external-owned blobs are not allocated by the store", "[BlobStore]") {
    BlobStore store(DEVICE_NAIVE, 0);

    auto desc              = MakeNaiveFloatDesc("external_input");
    auto external_backing  = std::make_unique<float[]>(4);
    BlobHandle ext_handle{};
    ext_handle.base      = external_backing.get();
    ext_handle.ownership = BLOB_HANDLE_EXTERNAL_OWNED;

    auto blob = std::make_shared<Blob>(desc, ext_handle);
    store.AddBlob(blob);

    REQUIRE(bool(store.AllocaBlob(blob)));
    CHECK(blob->GetHandle().base == external_backing.get());
    CHECK(blob->GetHandle().ownership == BLOB_HANDLE_EXTERNAL_OWNED);
}

TEST_CASE("BlobStore: external-owned blobs are not freed by the store", "[BlobStore]") {
    auto external_backing = std::make_unique<float[]>(4);

    std::shared_ptr<Blob> blob;
    {
        BlobStore store(DEVICE_NAIVE, 0);

        auto desc              = MakeNaiveFloatDesc("external_output");
        BlobHandle ext_handle{};
        ext_handle.base      = external_backing.get();
        ext_handle.ownership = BLOB_HANDLE_EXTERNAL_OWNED;

        blob = std::make_shared<Blob>(desc, ext_handle);
        store.AddBlob(blob);

        REQUIRE(bool(store.FreeBlob(blob)));
        CHECK(blob->GetHandle().base == external_backing.get());
    }

    REQUIRE(blob != nullptr);
    CHECK(blob->GetHandle().base == external_backing.get());
    CHECK(blob->GetHandle().ownership == BLOB_HANDLE_EXTERNAL_OWNED);
}
