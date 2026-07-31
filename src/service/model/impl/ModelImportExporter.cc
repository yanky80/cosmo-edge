// clang-format off
#include "service/detail/ServiceRegistry.h"
#include "service/model/impl/ModelImportExporter.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include "infer/ModelArtifactTool.h"
#include "nlohmann/json.hpp"
#include "util/ErrorCode.h"
#include "util/Exception.h"
#include "util/Exec.h"
#include "util/PathUtil.h"
#include "util/UuidUtil.h"

namespace cosmo::service {

// Anonymous namespace helpers (IsDetectionModel, InferClassCount, etc.) — moved to ModelAddModel.cc

void ModelImportExporter::SetHelpers(
    std::function<std::string()> getModelPath, std::function<std::string()> getModelTemplatePath,
    std::function<std::string(const std::string&)> findModelDir,
    std::function<std::string()> generateUniqueModelCode,
    std::function<void(const nlohmann::json&)> validateModelOutputFormat,
    std::function<void(const std::string&, const std::string&)> setModelPathMapping) {
    get_model_path_               = std::move(getModelPath);
    get_model_template_path_      = std::move(getModelTemplatePath);
    find_model_dir_               = std::move(findModelDir);
    generate_unique_model_code_   = std::move(generateUniqueModelCode);
    validate_model_output_format_ = std::move(validateModelOutputFormat);
    set_model_path_mapping_       = std::move(setModelPathMapping);
}

util::ErrorEnum ModelImportExporter::ExportModelConfig(const std::string& modelCode,
                                                       const std::string& modelName, std::string& outFilePath,
                                                       std::string& outFileName) {
    if (!cosmo::path::IsSafePathComponent(modelCode, 128)) {
        LOG_WARN("{}", "modelCode is unsafe in exportConfig request");
        return util::ErrorEnum::InvalidParam;
    }

    // Find model directory (need dirName for tar)
    namespace fs               = std::filesystem;
    std::string model_dir_path = find_model_dir_(modelCode);

    if (model_dir_path.empty()) {
        LOG_WARN("Model directory not found for modelCode: {}", modelCode);
        return util::ErrorEnum::FileNotExist;
    }
    std::string resolved_model_dir;
    for (const auto& root : cosmo::path::GetModelSearchPaths()) {
        if (cosmo::path::ResolveExistingPathWithinRoot(
                root, model_dir_path, cosmo::path::PathEntryType::kDirectory, resolved_model_dir)) {
            break;
        }
    }
    if (resolved_model_dir.empty()) {
        LOG_WARN("Reject unmanaged model export directory for modelCode: {}", modelCode);
        return util::ErrorEnum::InvalidParam;
    }
    model_dir_path                   = resolved_model_dir;
    const std::string model_dir_name = fs::path(model_dir_path).filename().string();
    if (!cosmo::path::IsSafePathComponent(model_dir_name, 200)) {
        return util::ErrorEnum::InvalidParam;
    }

    // Preset (built-in) models are encrypted and device-bound. Check the
    // canonical managed path so aliases and ancestor symlinks cannot bypass
    // the export restriction.
    if (cosmo::path::IsWithinRoot(cosmo::path::GetPresetModelPath(), model_dir_path)) {
        LOG_WARN("Reject export of preset (built-in) model {}: device-bound encryption", modelCode);
        return util::ErrorEnum::DefaultCantBeExport;
    }

    const std::string models_dir = fs::path(model_dir_path).parent_path().string();

    // Clean model name for filename
    std::string clean_model_name = modelName;
    std::replace_if(
        clean_model_name.begin(), clean_model_name.end(),
        [](char c) {
            return c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' ||
                   c == '>' || c == '|';
        },
        '_');
    if (clean_model_name.empty())
        clean_model_name = "model";
    if (!cosmo::path::IsSafePathComponent(clean_model_name, 120)) {
        return util::ErrorEnum::InvalidParam;
    }

    // Create tar.gz
    const auto temporary_root = cosmo::path::GetTemporaryDirPath();
    const auto archive_name   = "model_export_" + cosmo::util::GenerateUUID() + ".tar.gz";
    std::string archive_file_path;
    if (!cosmo::path::ResolvePathWithinRoot(
            temporary_root, (fs::path(temporary_root) / archive_name).string(), archive_file_path)) {
        return util::ErrorEnum::SysErr;
    }
    std::error_code archive_status_error;
    const auto archive_status = fs::symlink_status(archive_file_path, archive_status_error);
    if ((!archive_status_error && fs::exists(archive_status)) ||
        (archive_status_error && archive_status_error != std::errc::no_such_file_or_directory)) {
        return util::ErrorEnum::SysErr;
    }

    LOG_INFO("Creating tar.gz: {} from {}/{}", archive_file_path, models_dir, model_dir_name);

    std::string out_str;
    int ret = util::Exec({"tar", "-czf", archive_file_path, "-C", models_dir, model_dir_name}, out_str);
    if (ret != 0) {
        LOG_WARN("Failed to create tar.gz file, command return code: {}, error: {}", ret, out_str);
        std::error_code cleanup_error;
        fs::remove(archive_file_path, cleanup_error);
        return util::ErrorEnum::SysErr;
    }

    std::string resolved_archive;
    if (!cosmo::path::ResolveExistingPathWithinRoot(
            temporary_root, archive_file_path, cosmo::path::PathEntryType::kRegularFile, resolved_archive)) {
        LOG_WARN("Tar.gz file was not created successfully: {}", archive_file_path);
        std::error_code cleanup_error;
        fs::remove(archive_file_path, cleanup_error);
        return util::ErrorEnum::SysErr;
    }
    archive_file_path = resolved_archive;
    std::error_code permission_error;
    fs::permissions(archive_file_path, fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace, permission_error);
    if (permission_error) {
        LOG_WARN("Failed to protect model export archive: {}", permission_error.message());
        std::error_code cleanup_error;
        fs::remove(archive_file_path, cleanup_error);
        return util::ErrorEnum::SysErr;
    }

    std::error_code file_size_error;
    const auto file_size = fs::file_size(archive_file_path, file_size_error);
    if (file_size_error || file_size == 0) {
        LOG_WARN("Failed to read model export archive size: {}", file_size_error.message());
        std::error_code cleanup_error;
        fs::remove(archive_file_path, cleanup_error);
        return util::ErrorEnum::SysErr;
    }
    LOG_INFO("Successfully created tar.gz file for modelCode: {}, path: {}, size: {} bytes", modelCode,
             archive_file_path, file_size);

    outFileName = clean_model_name + "_" + modelCode + ".tar.gz";
    outFilePath = archive_file_path;

    return util::ErrorEnum::Success;
}

// ValidateAddModelInputs through AddAtomicModel — moved to ModelAddModel.cc
// ImportFlatArchive, ImportDirectoryArchive, ImportModel — moved to ModelImporter.cc

}  // namespace cosmo::service
