#pragma once

#include <map>
#include <memory>
#include <string>

#include "nn/core/common.h"
#include "nn/core/macros.h"
#include "nn/utils/image_format_utils.h"

namespace cosmo::nn {

struct PUBLIC BlobDesc {
    DeviceType device_type = DEVICE_NAIVE;

    DataType data_type = DATA_TYPE_FLOAT;

    DataFormat data_format   = DATA_FORMAT_NCHW;
    ImageFormat image_format = IMAGE_UNKNOWN;

    DimsVector dims;

    std::string name = "";

    bool is_affine_quantized = false;
    float affine_scale       = 1.0f;
    int affine_zero_point    = 0;

    std::string description();
};

typedef enum {
    BLOB_HANDLE_STORE_OWNED = 0,
    BLOB_HANDLE_EXTERNAL_OWNED,
} BlobHandleOwnership;

struct PUBLIC BlobHandle {
    void* base        = nullptr;
    unsigned long phy = 0;
    BlobHandleOwnership ownership = BLOB_HANDLE_STORE_OWNED;
};

class BlobImpl;

class PUBLIC Blob {
public:
    explicit Blob(BlobDesc desc);

    Blob(BlobDesc desc, bool alloc_memory);

    Blob(BlobDesc desc, BlobHandle handle);

    ~Blob();

    BlobDesc& GetBlobDesc();

    void SetBlobDesc(BlobDesc desc);

    BlobHandle GetHandle();

    void SetHandle(BlobHandle handle);

    void ClearHandle();

private:
    std::unique_ptr<BlobImpl> impl;
};

using BlobMap = std::map<std::string, std::shared_ptr<Blob>>;

}  // namespace cosmo::nn
