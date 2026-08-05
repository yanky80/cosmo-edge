// NnBackendConstants.h — Compile-time NN backend selection constants.
//
// Provides directory naming prefixes, engine type identifiers, and model file
// extensions that differ between Sophon (BM1688/CV186X) and CPU (x86) backends.
// All values are constexpr so they resolve at compile time with zero runtime cost.
//
// Folder names are platform-agnostic in the chip/platform token slot:
//   prod_<TOKEN>_<alg_code>_<name>_<version>   (TOKEN may be BM1688, CV186X, SOPHGO, X86, ...)
// The token is a readable label only; model identity is the alg_code, and the
// actual chip is stored in config.json "chip_type".
//
// Usage:
//   #include "util/NnBackendConstants.h"
//   std::string dir = cosmo::util::kNewDirPrefix + modelCode + "_" + name;

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace cosmo::util {

#ifdef COSMO_NN_USE_SOPHON_BACKEND

/// Legacy directory prefix for Sophon backend model directories: "prod_BM1688_".
/// Kept for backward compatibility; new directories are generated with kNewDirPrefix.
static constexpr const char* kPlatformDirPrefix = "prod_BM1688_";

/// Prefix used when GENERATING new model directories via web add: "prod_SOPHGO_".
/// Vendor-level on purpose — the folder name no longer encodes a specific chip
/// (BM1688 / CV186X / ...). The chip itself lives in config.json "chip_type".
static constexpr const char* kNewDirPrefix = "prod_SOPHGO_";

/// Regex pattern to extract algorithm code from model directory names.
/// Chip-agnostic: matches prod_BM1688_, prod_CV186X_, prod_SOPHGO_, ... and captures
/// the numeric algorithm code segment regardless of the platform token.
static constexpr const char* kPlatformDirRegex = "prod_[A-Z0-9]+_([0-9]+)_.*";

/// Engine type identifier reported to frontend / device info API.
static constexpr const char* kEngineType = "BM1688";

/// Model binary file extension for Sophon backend (.nn wraps .bmodel).
static constexpr const char* kModelFileExt = ".nn";

/// Compatibility extensions accepted when resolving legacy Sophon model packages
/// that predate config.json models[].file_name.
static constexpr const char* kCompatibleModelFileExts[] = {".nn", ".bmodel"};

/// Supported Sophon chip types, as written to config.json "chip_type".
/// Add a new chip here to support it across the model pipeline.
static constexpr const char* kSupportedChips[] = {"BM1688", "CV186X"};

#elif defined(COSMO_NN_USE_RKNN_BACKEND)

/// Directory prefix for RK3588 model directories: "prod_RK3588_".
static constexpr const char* kPlatformDirPrefix = "prod_RK3588_";

/// Prefix used when generating new RK3588 model directories.
static constexpr const char* kNewDirPrefix = "prod_RK3588_";

/// Regex pattern to extract algorithm code from RK3588 model directory names.
static constexpr const char* kPlatformDirRegex = "prod_[A-Z0-9]+_([0-9]+)_.*";

/// Engine type identifier reported to frontend / device info API.
static constexpr const char* kEngineType = "RK3588";

/// RKNN model binary file extension.
static constexpr const char* kModelFileExt = ".rknn";

static constexpr const char* kCompatibleModelFileExts[] = {".rknn"};
static constexpr const char* kSupportedChips[]          = {"RK3588"};

#elif defined(COSMO_NN_USE_ASCEND_BACKEND)

/// Directory prefix for Ascend 310P3 model directories: "prod_ASCEND310P3_".
static constexpr const char* kPlatformDirPrefix = "prod_ASCEND310P3_";

/// Prefix used when generating new Ascend 310P3 model directories.
static constexpr const char* kNewDirPrefix = "prod_ASCEND310P3_";

/// Regex pattern to extract algorithm code from Ascend 310P3 model directory names.
static constexpr const char* kPlatformDirRegex = "prod_[A-Z0-9]+_([0-9]+)_.*";

/// Engine type identifier reported to frontend / device info API.
static constexpr const char* kEngineType = "ASCEND310P3";

/// Model binary file extension for the Ascend backend (.om from ATC).
static constexpr const char* kModelFileExt = ".om";

/// Compatibility extensions accepted for Ascend model packages.
static constexpr const char* kCompatibleModelFileExts[] = {".om"};

/// Supported Ascend chip types, as written to config.json "chip_type".
static constexpr const char* kSupportedChips[] = {"ASCEND310P3"};

#elif defined(COSMO_NN_USE_CPU_BACKEND)

/// Directory prefix for CPU/x86 backend model directories: "prod_X86_".
static constexpr const char* kPlatformDirPrefix = "prod_X86_";

/// Prefix used when generating new model directories via web add. For the CPU
/// backend there is a single platform, so it coincides with kPlatformDirPrefix.
static constexpr const char* kNewDirPrefix = "prod_X86_";

/// Regex pattern to extract algorithm code from x86 model directory names.
static constexpr const char* kPlatformDirRegex = "prod_[A-Z0-9]+_([0-9]+)_.*";

/// Engine type identifier reported to frontend / device info API.
static constexpr const char* kEngineType = "X86";

/// Model binary file extension for CPU backend (.onnx used directly).
static constexpr const char* kModelFileExt = ".onnx";

/// Compatibility extensions accepted when resolving legacy x86 model packages
/// that predate config.json models[].file_name.
static constexpr const char* kCompatibleModelFileExts[] = {".onnx"};

/// Supported platform identifier for the CPU backend.
static constexpr const char* kSupportedChips[] = {"X86"};

#else
#error                                                                                                       \
    "One of COSMO_NN_USE_SOPHON_BACKEND, COSMO_NN_USE_RKNN_BACKEND, COSMO_NN_USE_ASCEND_BACKEND, or COSMO_NN_USE_CPU_BACKEND must be defined"
#endif

/// Case-insensitive check whether `chip` is a supported chip/platform type for the
/// compiled backend. Used to validate config.json "chip_type" during model import.
inline bool IsSupportedChip(const std::string& chip) {
    auto iequal = [](char a, char b) {
        return std::toupper(static_cast<unsigned char>(a)) == std::toupper(static_cast<unsigned char>(b));
    };
    for (const char* supported : kSupportedChips) {
        std::string s(supported);
        if (s.size() == chip.size() && std::equal(s.begin(), s.end(), chip.begin(), iequal))
            return true;
    }
    return false;
}

/// Returns true when `extension` belongs to the compiled platform's model-artifact
/// profile. Used by import and managed-path lookup to reject cross-platform files.
inline bool IsSupportedModelFileExtension(const std::string& extension) {
    for (const char* supported : kCompatibleModelFileExts) {
        if (extension == supported)
            return true;
    }
    return false;
}

}  // namespace cosmo::util
