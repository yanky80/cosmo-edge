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

/**
 * IEEE 754 float -> half (round to nearest even). Used to build FP16 host
 * tensors, e.g. the Ascend image_to_tensor normalization.
 */
inline uint16_t FloatToFp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign     = (bits >> 16U) & 0x8000U;
    const uint32_t exponent = (bits >> 23U) & 0xffU;
    const uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        // Inf/NaN stay Inf/NaN in half precision.
        return static_cast<uint16_t>(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
    }
    if (exponent == 0U && mantissa == 0U) {
        return static_cast<uint16_t>(sign);
    }
    // Normalize the biased FP32 exponent into the half range.
    int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00U);  // overflow -> Inf
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);  // underflow -> zero
        }
        // Subnormal half: shift the FP32 mantissa into half subnormal range,
        // rounding the dropped bits to nearest even like the normal path.
        const int shift        = 14 - half_exponent;
        const uint32_t full    = mantissa | 0x800000U;
        uint32_t shifted       = full >> shift;
        const uint32_t dropped = full & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1);
        if (dropped > halfway || (dropped == halfway && (shifted & 1U) != 0U))
            ++shifted;
        return static_cast<uint16_t>(sign | shifted);
    }
    const uint16_t half_mantissa = static_cast<uint16_t>(mantissa >> 13U);
    const uint32_t remainder     = mantissa & 0x1fffU;
    const uint32_t halfway       = 0x1000U;
    uint16_t result =
        static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exponent) << 10U) | half_mantissa);
    // Round to nearest even on the 13 dropped mantissa bits.
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0U)) {
        ++result;
    }
    return result;
}

}  // namespace cosmo::nn
