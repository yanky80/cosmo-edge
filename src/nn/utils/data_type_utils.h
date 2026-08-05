#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "nn/core/common.h"
#include "nn/core/macros.h"

namespace cosmo::nn {

class PUBLIC DataTypeUtils {
public:
    static int GetBytesSize(DataType);

    static std::string GetDataTypeString(DataType);
};

/**
 * Convert InputNodeInfo raw data_type integer to DataType enum.
 * InputNodeInfo encoding: 0=fp32, 1=fp16, 2=bfp16, 3=int8
 */
inline DataType DataTypeFromInputInfo(int raw) {
    switch (raw) {
        case 1:
            return DATA_TYPE_HALF;
        case 2:
            return DATA_TYPE_BFP16;
        case 3:
            return DATA_TYPE_INT8;
        default:
            return DATA_TYPE_FLOAT;
    }
}

/**
 * IEEE 754 half -> float. Shared by backends that receive FP16 device
 * outputs (Ascend OM, Sophon bmodel).
 */
inline float Fp16ToFloat(uint16_t value) {
    const uint32_t sign     = static_cast<uint32_t>(value & 0x8000U) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1fU;
    const uint32_t mantissa = value & 0x03ffU;
    uint32_t bits           = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t normalized_mantissa = mantissa;
            int shift                    = 0;
            while ((normalized_mantissa & 0x0400U) == 0U) {
                normalized_mantissa <<= 1U;
                ++shift;
            }
            normalized_mantissa &= 0x03ffU;
            const uint32_t normalized_exponent = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (normalized_exponent << 23U) | (normalized_mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + (127U - 15U)) << 23U) | (mantissa << 13U);
    }

    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

}  // namespace cosmo::nn
