// ModelArtifactTool — Utility for inspecting backend model artifacts and installing nn payloads.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace cosmo {

// Input/output node info
struct BmodelNodeInfo {
    std::string name;        // Node name
    std::vector<int> shape;  // Shape
    int data_type{0};        // Data type (0=float32, 1=float16, etc.)
};

// Single network info
struct BmodelNetworkInfo {
    std::string name;  // Network name
    int max_batch{1};  // Max batch size
    std::vector<BmodelNodeInfo> inputs;
    std::vector<BmodelNodeInfo> outputs;
};

// Complete model artifact info
struct BmodelInfo {
    std::string file_path;
    std::vector<BmodelNetworkInfo> networks;
    bool valid{false};
    std::string error_msg;
};

class ModelArtifactTool {
public:
    ModelArtifactTool()  = default;
    ~ModelArtifactTool() = default;

    // Get backend model artifact info.
    static BmodelInfo GetModelArtifactInfo(const std::string& modelArtifactPath);

    // Install single or multiple model artifacts to the target nn payload.
    // Returns empty string on success; error message on failure.
    static std::string InstallModelArtifacts(const std::vector<std::string>& modelArtifactPaths,
                                             const std::string& outputPath);

    // Clean up temporary files.
    static void CleanupTempFiles(const std::vector<std::string>& filePaths);

    // Log model info for debugging.
    static void LogModelArtifactInfo(const BmodelInfo& info,
                                     const std::string& logPrefix = "[ModelArtifactTool]");

private:
    static int ConvertDataType(int bmDataType);
};


}  // namespace cosmo
