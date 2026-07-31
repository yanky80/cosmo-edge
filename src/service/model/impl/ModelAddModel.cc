// ModelAddModel.cc — AddAtomicModel pipeline and helpers for ModelImportExporter.
// Split from ModelImportExporter.cc to reduce file size (DEBT-007).

// clang-format off
#include "service/detail/ServiceRegistry.h"
#include "service/model/impl/ModelImportExporter.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include "infer/ModelArtifactTool.h"
#include "nlohmann/json.hpp"
#include "util/ErrorCode.h"
#include "util/Exception.h"
#include "util/FileUtil.h"
#include "util/NnBackendConstants.h"
#include "util/PathUtil.h"

namespace cosmo::service {

namespace {

    bool IsTextFile(const std::string& path) {
        std::string extension = std::filesystem::path(path).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return extension == ".txt";
    }

    bool CountCharacterTableEntries(const std::string& path, size_t& entry_count) {
        entry_count = 0;
        std::ifstream stream(path);
        if (!stream.is_open())
            return false;

        std::string line;
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            entry_count++;
        }
        return entry_count > 0;
    }

    bool HasLeadingBlankToken(const std::string& path) {
        std::ifstream stream(path);
        std::string token;
        if (!stream.is_open() || !std::getline(stream, token))
            return false;

        auto trim = [](std::string& value) {
            const auto first = value.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                value.clear();
                return;
            }
            const auto last = value.find_last_not_of(" \t\r\n");
            value           = value.substr(first, last - first + 1);
        };
        trim(token);
        if (!token.empty() && token.back() == ',') {
            token.pop_back();
            trim(token);
        }
        if (token.size() >= 2 && token.front() == '"' && token.back() == '"')
            token = token.substr(1, token.size() - 2);
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        return token.empty() || token == "blank";
    }

    int GetOcrClassCount(const std::vector<cosmo::BmodelInfo>& bmodel_infos) {
        if (bmodel_infos.size() != 1 || !bmodel_infos[0].valid || bmodel_infos[0].networks.empty())
            return 0;

        const auto& outputs = bmodel_infos[0].networks[0].outputs;
        if (outputs.size() != 1 || outputs[0].shape.empty())
            return 0;

        const int class_count = outputs[0].shape.back();
        return class_count > 0 ? class_count : 0;
    }

    bool ResolveManagedUploadFile(const std::string& path, std::string& resolved) {
        return cosmo::path::ResolveExistingPathWithinRoot(cosmo::path::GetModelUploadTmpDir(), path,
                                                          cosmo::path::PathEntryType::kRegularFile,
                                                          resolved) ||
               cosmo::path::ResolveExistingPathWithinRoot(cosmo::path::GetUploadPath(), path,
                                                          cosmo::path::PathEntryType::kRegularFile, resolved);
    }

    void CleanupManagedUploadFiles(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            std::string resolved;
            if (!ResolveManagedUploadFile(path, resolved)) {
                LOG_WARN("[AddModel] Refusing to clean unmanaged upload path: {}", path);
                continue;
            }
            std::error_code ec;
            std::filesystem::remove(resolved, ec);
            if (ec) {
                LOG_WARN("[AddModel] Failed to clean upload file {}: {}", resolved, ec.message());
            }
        }
    }

}  // namespace

util::ErrorEnum ModelImportExporter::ConfigureOcrCharacterTable(
    nlohmann::json& config, const std::vector<cosmo::BmodelInfo>& bmodel_infos,
    const std::string& character_table_path) {
    size_t entry_count = 0;
    if (!CountCharacterTableEntries(character_table_path, entry_count)) {
        LOG_WARN("{}", "[AddModel] OCR character table is empty or unreadable");
        return util::ErrorEnum::InvalidParam;
    }

    const int class_count = GetOcrClassCount(bmodel_infos);
    if (class_count == 0) {
        LOG_WARN("{}", "[AddModel] OCR model requires one output with a static class dimension");
        return util::ErrorEnum::InvalidParam;
    }

    nlohmann::json prepended_tokens = nlohmann::json::array();
    nlohmann::json appended_tokens  = nlohmann::json::array();
    const bool has_leading_blank    = HasLeadingBlankToken(character_table_path);
    if (entry_count == static_cast<size_t>(class_count)) {
        // The uploaded table already contains every CTC class.
    } else if (entry_count + 1 == static_cast<size_t>(class_count)) {
        if (has_leading_blank) {
            // Legacy license-plate table: blank is present and the final class is a space.
            appended_tokens.push_back(" ");
        } else {
            // Standard CTC table: the model reserves class zero for blank.
            prepended_tokens.push_back("");
        }
    } else if (entry_count + 2 == static_cast<size_t>(class_count) && !has_leading_blank) {
        // PP-OCR Chinese table: CTC prepends blank and use_space_char appends ASCII space.
        prepended_tokens.push_back("");
        appended_tokens.push_back(" ");
    } else if (entry_count != static_cast<size_t>(class_count)) {
        LOG_WARN("[AddModel] OCR character table entries {} do not match output classes {}", entry_count,
                 class_count);
        return util::ErrorEnum::InvalidParam;
    }

    if (!config.contains("models") || !config["models"].is_array() || config["models"].empty()) {
        LOG_WARN("{}", "[AddModel] OCR template has no model configuration");
        return util::ErrorEnum::InvalidParam;
    }

    auto& params = config["models"][0]["params"];
    if (!params.is_object())
        params = nlohmann::json::object();
    params["character_table_file"] = "character_table.txt";
    params["ctc_blank_index"]      = 0;
    params["ctc_prepend_tokens"]   = std::move(prepended_tokens);
    params["ctc_append_tokens"]    = std::move(appended_tokens);
    params["ctc_class_count"]      = class_count;
    return util::ErrorEnum::Success;
}

// JSON helpers moved to ModelAddModel_Json.cc

// ============================================================
// Extracted helpers for AddAtomicModel (insert before AddAtomicModel)
// ============================================================

util::ErrorEnum ModelImportExporter::ValidateAddModelInputs(
    const std::string& modelCode, const std::string& modelName, const std::string& modelType,
    const std::vector<cosmo::Model::ModelArtifactInfo>& model_files, const std::string& vocabFilePath,
    const std::string& tokenizerFilePath, const std::string& characterTableFilePath,
    std::string& resolved_model_code, std::vector<std::string>& model_artifact_paths) {
    namespace fs = std::filesystem;

    resolved_model_code = generate_unique_model_code_();
    LOG_INFO("[AddModel] Using resolved model code: {} (request code: {})", resolved_model_code, modelCode);
    if (resolved_model_code.empty())
        return util::ErrorEnum::SysErr;

    // 1. Validate model code format (7-digit number)
    if (resolved_model_code.length() != 7 ||
        !std::all_of(resolved_model_code.begin(), resolved_model_code.end(), ::isdigit)) {
        LOG_WARN("[AddModel] Generated model code is invalid: {}", resolved_model_code);
        return util::ErrorEnum::InvalidParam;
    }

    if (!cosmo::path::IsSafePathComponent(modelName, 64) || modelName.find('_') != std::string::npos) {
        LOG_WARN("{}", "[AddModel] Model name must not contain underscores");
        return util::ErrorEnum::InvalidParam;
    }

    // 2. Validate model artifacts
    if (model_files.empty()) {
        LOG_WARN("{}", "[AddModel] No model artifacts provided");
        return util::ErrorEnum::InvalidParam;
    }

    bool is_sam2 = (modelType == "sam2");
    if (is_sam2 && model_files.size() != 2) {
        LOG_WARN("{}", "[AddModel] SAM2 model requires exactly 2 model artifacts (encoder and decoder)");
        return util::ErrorEnum::InvalidParam;
    }
    if (!is_sam2 && model_files.size() != 1) {
        LOG_WARN("{}", "[AddModel] Non-SAM2 model must have exactly 1 model artifact");
        return util::ErrorEnum::InvalidParam;
    }

    std::string resolved_auxiliary_path;
    if (modelType == "dino" &&
        (vocabFilePath.empty() || !ResolveManagedUploadFile(vocabFilePath, resolved_auxiliary_path))) {
        LOG_WARN("{}", "[AddModel] DINO model requires a managed vocab.txt upload");
        return util::ErrorEnum::InvalidParam;
    }
    if ((modelType == "qwen3vl" || modelType == "qwen3_5") &&
        (tokenizerFilePath.empty() ||
         !ResolveManagedUploadFile(tokenizerFilePath, resolved_auxiliary_path))) {
        LOG_WARN("{}", "[AddModel] This model type requires a managed tokenizer.json upload");
        return util::ErrorEnum::InvalidParam;
    }
    if (modelType == "ocr") {
        if (characterTableFilePath.empty() ||
            !ResolveManagedUploadFile(characterTableFilePath, resolved_auxiliary_path)) {
            LOG_WARN("{}", "[AddModel] OCR model requires a managed character table upload");
            return util::ErrorEnum::InvalidParam;
        }
        if (!IsTextFile(resolved_auxiliary_path)) {
            LOG_WARN("[AddModel] OCR character table must be a .txt file: {}", characterTableFilePath);
            return util::ErrorEnum::InvalidParam;
        }
        if (fs::file_size(resolved_auxiliary_path) == 0) {
            LOG_WARN("[AddModel] OCR character table does not exist or is empty: {}", characterTableFilePath);
            return util::ErrorEnum::FileNotExist;
        }
    } else if (!characterTableFilePath.empty()) {
        LOG_WARN("[AddModel] Only OCR models may include a character table: {}", modelType);
        return util::ErrorEnum::InvalidParam;
    }

    // Check all model artifacts exist
    for (const auto& model_file : model_files) {
        std::string resolved_path;
        if (!ResolveManagedUploadFile(model_file.filePath, resolved_path)) {
            LOG_WARN("[AddModel] model artifact is not a managed upload: {}", model_file.filePath);
            return util::ErrorEnum::FileNotExist;
        }
        model_artifact_paths.push_back(std::move(resolved_path));
        LOG_INFO("[AddModel] model artifact: role={}, path={}", model_file.role, model_artifact_paths.back());
    }

    return util::ErrorEnum::Success;
}

util::ErrorEnum ModelImportExporter::CollectModelArtifactInfo(
    const std::string& modelType, const std::vector<std::string>& model_artifact_paths,
    std::vector<cosmo::BmodelInfo>& artifact_infos, bool& use_template_defaults) {
    use_template_defaults = (modelType == "qwen3vl" || modelType == "qwen3_5");

    if (!use_template_defaults) {
        for (const auto& model_artifact_path : model_artifact_paths) {
            auto info = cosmo::ModelArtifactTool::GetModelArtifactInfo(model_artifact_path);
            if (!info.valid && info.error_msg != "SDK_NOT_AVAILABLE") {
                LOG_WARN("[AddModel] Failed to inspect model artifact: {}", info.error_msg);
                return util::ErrorEnum::InvalidParam;
            }
            artifact_infos.push_back(info);
        }

        if (!artifact_infos.empty() && !artifact_infos[0].valid &&
            artifact_infos[0].error_msg == "SDK_NOT_AVAILABLE") {
            use_template_defaults = true;
            LOG_INFO("{}", "[AddModel] SDK not available, using template defaults");
        }

        for (size_t index = 0; index < artifact_infos.size(); index++) {
            std::string prefix = "[AddModel] modelArtifact[" + std::to_string(index) + "]";
            cosmo::ModelArtifactTool::LogModelArtifactInfo(artifact_infos[index], prefix);
        }
    }

    return util::ErrorEnum::Success;
}

std::string ModelImportExporter::CalculateNextVersion(const std::string& models_dir,
                                                      const std::string& modelCode) {
    namespace fs = std::filesystem;

    int max_version = 0;
    {
        // Chip-agnostic: match prod_<TOKEN>_<code>_..._<version>, where TOKEN may be
        // BM1688, CV186X, SOPHGO, ... so legacy and new directories both count.
        std::regex pattern("^prod_[A-Z0-9]+_" + modelCode + "_.*_V(\\d+)\\.0\\.(\\d+)$");
        for (const auto& dirEntry : fs::directory_iterator(models_dir)) {
            if (!dirEntry.is_directory())
                continue;
            std::string dir_name = dirEntry.path().filename().string();
            std::smatch matches;
            if (std::regex_match(dir_name, matches, pattern) && matches.size() > 2) {
                try {
                    int major   = std::stoi(matches[1].str());
                    int minor   = std::stoi(matches[2].str());
                    int version = major * 1000 + minor;
                    if (version > max_version)
                        max_version = version;
                } catch (const std::exception&) {
                }
            }
        }
    }

    int new_version = max_version + 1;
    int major       = (new_version - 1) / 1000 + 1;
    int minor       = (new_version - 1) % 1000;
    return "V" + std::to_string(major) + ".0." + std::to_string(minor);
}

util::ErrorEnum ModelImportExporter::InstallModelArtifacts(
    const std::string& modelType, const std::vector<std::string>& model_artifact_paths,
    const std::string& model_dir) {
    namespace fs = std::filesystem;

#ifdef COSMO_NN_USE_ONNX_BACKEND
    // CPU/x86: copy .onnx file directly; no Sophon wrapper needed.
    std::string convert_error;
    if (modelType == "sam2") {
        const std::vector<std::string> dest_names = {"sam2_encoder.onnx", "sam2_decoder.onnx"};
        for (size_t i = 0; i < dest_names.size() && i < model_artifact_paths.size(); i++) {
            std::error_code ec;
            std::string dest_path = (fs::path(model_dir) / dest_names[i]).string();
            fs::copy_file(model_artifact_paths[i], dest_path, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                convert_error = "Failed to copy SAM2 ONNX file to " + dest_path + ": " + ec.message();
                break;
            }
            LOG_INFO("[AddModel] x86 SAM2: copied ONNX file to {}", dest_path);
        }
    } else {
        std::string model_file_path = model_dir + "/model.onnx";
        std::error_code ec;
        fs::copy_file(model_artifact_paths[0], model_file_path, fs::copy_options::overwrite_existing, ec);
        if (ec)
            convert_error = "Failed to copy model file to " + model_file_path + ": " + ec.message();
        else
            LOG_INFO("[AddModel] x86: copied model file to {}", model_file_path);
    }
#else
    // Sophon: wrap backend artifacts into .nn format
    std::string nn_path = model_dir + "/model.nn";

    std::string convert_error;
    if (modelType == "qwen3vl" || modelType == "qwen3_5") {
        std::error_code ec;
        fs::copy_file(model_artifact_paths[0], nn_path, fs::copy_options::overwrite_existing, ec);
        if (ec)
            convert_error = "Failed to copy model artifact to model.nn: " + ec.message();
        else
            LOG_INFO("[AddModel] qwen3vl: copied raw model artifact to {} (no nn wrapper header)", nn_path);
    } else {
        convert_error = cosmo::ModelArtifactTool::InstallModelArtifacts(model_artifact_paths, nn_path);
    }
#endif

    if (!convert_error.empty()) {
        try {
            fs::remove_all(model_dir);
        } catch (const std::exception&) {
        }
        LOG_WARN("[AddModel] model file conversion/copy failed: {}", convert_error);
        return util::ErrorEnum::SysErr;
    }
    return util::ErrorEnum::Success;
}

// UpdateTemplateConfig moved to ModelAddModel_Json.cc

util::ErrorEnum ModelImportExporter::CopyAuxiliaryFiles(const std::string& modelType,
                                                        const std::string& vocabFilePath,
                                                        const std::string& tokenizerFilePath,
                                                        const std::string& characterTableFilePath,
                                                        const std::string& model_dir) {
    namespace fs = std::filesystem;

    if (modelType == "dino" && !vocabFilePath.empty()) {
        if (fs::exists(vocabFilePath)) {
            std::string dest_vocab = model_dir + "/vocab.txt";
            try {
                fs::copy_file(vocabFilePath, dest_vocab, fs::copy_options::overwrite_existing);
                LOG_INFO("[AddModel] Copied vocab.txt to {}", dest_vocab);
            } catch (const std::exception& e) {
                try {
                    fs::remove_all(model_dir);
                } catch (const std::exception&) {
                }
                LOG_WARN("[AddModel] Failed to copy vocab.txt: {}", e.what());
                return util::ErrorEnum::SysErr;
            }
        } else {
            try {
                fs::remove_all(model_dir);
            } catch (const std::exception&) {
            }
            LOG_WARN("[AddModel] vocab.txt temp file does not exist: {}", vocabFilePath);
            return util::ErrorEnum::FileNotExist;
        }
    }

    if ((modelType == "qwen3vl" || modelType == "qwen3_5") && !tokenizerFilePath.empty()) {
        if (fs::exists(tokenizerFilePath)) {
            std::string dest_tokenizer = model_dir + "/tokenizer.json";
            try {
                fs::copy_file(tokenizerFilePath, dest_tokenizer, fs::copy_options::overwrite_existing);
                LOG_INFO("[AddModel] Copied tokenizer.json to {}", dest_tokenizer);
            } catch (const std::exception& e) {
                try {
                    fs::remove_all(model_dir);
                } catch (const std::exception&) {
                }
                LOG_WARN("[AddModel] Failed to copy tokenizer.json: {}", e.what());
                return util::ErrorEnum::SysErr;
            }
        } else {
            try {
                fs::remove_all(model_dir);
            } catch (const std::exception&) {
            }
            LOG_WARN("[AddModel] tokenizer.json temp file does not exist: {}", tokenizerFilePath);
            return util::ErrorEnum::FileNotExist;
        }
    }

    if (modelType == "ocr") {
        std::string dest_table = model_dir + "/character_table.txt";
        try {
            fs::copy_file(characterTableFilePath, dest_table, fs::copy_options::overwrite_existing);
            LOG_INFO("[AddModel] Copied OCR character table to {}", dest_table);
        } catch (const std::exception& e) {
            try {
                fs::remove_all(model_dir);
            } catch (const std::exception&) {
            }
            LOG_WARN("[AddModel] Failed to copy OCR character table: {}", e.what());
            return util::ErrorEnum::SysErr;
        }
    }

    return util::ErrorEnum::Success;
}

// ============================================================
// Slim AddAtomicModel orchestrator
// ============================================================
util::ErrorEnum ModelImportExporter::AddAtomicModel(
    const std::string& modelCode, const std::string& modelName, const std::string& modelType,
    const std::string& description, const std::vector<cosmo::Model::ModelArtifactInfo>& model_files,
    const std::string& vocabFilePath, const std::string& tokenizerFilePath,
    const std::string& characterTableFilePath, const std::string& normalizationMode,
    const std::string& colorChannel) {
    namespace fs = std::filesystem;

    const std::string models_dir   = get_model_path_();
    const std::string template_dir = get_model_template_path_();

    // Collect temp files for cleanup
    std::vector<std::string> temp_files_to_cleanup;
    for (const auto& model_file : model_files) {
        if (!model_file.filePath.empty())
            temp_files_to_cleanup.push_back(model_file.filePath);
    }
    if (!vocabFilePath.empty())
        temp_files_to_cleanup.push_back(vocabFilePath);
    if (!tokenizerFilePath.empty())
        temp_files_to_cleanup.push_back(tokenizerFilePath);
    if (!characterTableFilePath.empty())
        temp_files_to_cleanup.push_back(characterTableFilePath);

    auto cleanup_and_return = [&](util::ErrorEnum err) -> util::ErrorEnum {
        CleanupManagedUploadFiles(temp_files_to_cleanup);
        return err;
    };

    // 1-2. Validate inputs
    std::string resolved_model_code;
    std::vector<std::string> model_artifact_paths;
    auto err =
        ValidateAddModelInputs(modelCode, modelName, modelType, model_files, vocabFilePath,
                               tokenizerFilePath, characterTableFilePath, resolved_model_code,
                               model_artifact_paths);
    if (err != util::ErrorEnum::Success)
        return cleanup_and_return(err);

    std::string resolved_vocab_path;
    std::string resolved_tokenizer_path;
    std::string resolved_character_table_path;
    if ((!vocabFilePath.empty() && !ResolveManagedUploadFile(vocabFilePath, resolved_vocab_path)) ||
        (!tokenizerFilePath.empty() &&
         !ResolveManagedUploadFile(tokenizerFilePath, resolved_tokenizer_path)) ||
        (!characterTableFilePath.empty() &&
         !ResolveManagedUploadFile(characterTableFilePath, resolved_character_table_path))) {
        LOG_WARN("{}", "[AddModel] Auxiliary file is not a managed upload");
        return cleanup_and_return(util::ErrorEnum::FileNotExist);
    }

    // 3. Read template file
    std::string template_path = (fs::path(template_dir) / (modelType + ".json")).string();
    if (!fs::exists(template_path)) {
        LOG_WARN("[AddModel] Model type template not found: {}", template_path);
        return cleanup_and_return(util::ErrorEnum::FileNotExist);
    }

    std::ifstream templateFile(template_path);
    if (!templateFile.is_open()) {
        LOG_WARN("[AddModel] Failed to open template file: {}", template_path);
        return cleanup_and_return(util::ErrorEnum::SysErr);
    }

    std::stringstream templateBuffer;
    templateBuffer << templateFile.rdbuf();
    templateFile.close();

    nlohmann::json templateDoc;
    try {
        templateDoc = nlohmann::json::parse(templateBuffer.str());
    } catch (const std::exception& e) {
        LOG_WARN("[AddModel] Template file JSON parse error: {} ({})", template_path, e.what());
        return cleanup_and_return(util::ErrorEnum::InvalidParam);
    }

    // 4. Get model artifact info
    bool use_template_defaults = false;
    std::vector<cosmo::BmodelInfo> artifact_infos;
    err = CollectModelArtifactInfo(modelType, model_artifact_paths, artifact_infos, use_template_defaults);
    if (err != util::ErrorEnum::Success)
        return cleanup_and_return(err);

    if (modelType == "ocr") {
        err = ConfigureOcrCharacterTable(templateDoc, artifact_infos, resolved_character_table_path);
        if (err != util::ErrorEnum::Success)
            return cleanup_and_return(err);
    }

    // 5. Calculate version number
    std::string version_str = CalculateNextVersion(models_dir, resolved_model_code);

    // 6. Create model directory
    std::string clean_model_name = modelName;
    std::replace(clean_model_name.begin(), clean_model_name.end(), ' ', '_');
    std::replace(clean_model_name.begin(), clean_model_name.end(), '/', '_');
    std::replace(clean_model_name.begin(), clean_model_name.end(), '\\', '_');

    std::string folder_name = std::string(cosmo::util::kNewDirPrefix) + resolved_model_code + "_" +
                              clean_model_name + "_" + version_str;
    std::string model_dir;
    if (!cosmo::path::IsSafePathComponent(folder_name, 200) ||
        !cosmo::path::ResolvePathWithinRoot(models_dir, (fs::path(models_dir) / folder_name).string(),
                                            model_dir)) {
        LOG_WARN("[AddModel] Refusing unsafe model directory component: {}", folder_name);
        return cleanup_and_return(util::ErrorEnum::InvalidParam);
    }

    LOG_INFO("[AddModel] Creating model directory: {}", model_dir);
    if (!util::CreateDir(model_dir)) {
        LOG_WARN("[AddModel] Failed to create model directory: {}", model_dir);
        return cleanup_and_return(util::ErrorEnum::SysErr);
    }

    // 7. Install backend model artifacts
    err = InstallModelArtifacts(modelType, model_artifact_paths, model_dir);
    if (err != util::ErrorEnum::Success)
        return cleanup_and_return(err);

    // 8. Update template config
    UpdateTemplateConfig(templateDoc, resolved_model_code, version_str, modelName, modelType, description,
                         artifact_infos, use_template_defaults, normalizationMode, colorChannel);

    // 8.1 Validate model output format
    {
        try {
            validate_model_output_format_(templateDoc);
        } catch (const std::exception&) {
            try {
                fs::remove_all(model_dir);
            } catch (const std::exception&) {
            }
            CleanupManagedUploadFiles(temp_files_to_cleanup);
            throw;  // re-throw to preserve original exception behavior
        }
    }

    // 9. Save config.json
    std::string config_path  = model_dir + "/config.json";
    std::string json_content = templateDoc.dump(2);

    if (!util::WriteFile(config_path, json_content)) {
        try {
            fs::remove_all(model_dir);
        } catch (const std::exception&) {
        }
        LOG_WARN("[AddModel] Failed to save config.json: {}", config_path);
        return cleanup_and_return(util::ErrorEnum::SysErr);
    }

    // 9.1 Copy auxiliary files
    err = CopyAuxiliaryFiles(modelType, resolved_vocab_path, resolved_tokenizer_path,
                             resolved_character_table_path, model_dir);
    if (err != util::ErrorEnum::Success)
        return cleanup_and_return(err);

    // 10. Clean up temp files
    CleanupManagedUploadFiles(temp_files_to_cleanup);

    // 11. Update memory mapping so service can find it without restart
    if (set_model_path_mapping_) {
        set_model_path_mapping_(resolved_model_code, model_dir);
    }

    LOG_INFO("[AddModel] Successfully added model: code={}, name={}, version={}, directory={}",
             resolved_model_code, modelName, version_str, model_dir);

    return util::ErrorEnum::Success;
}

// ImportFlatArchive, ImportDirectoryArchive, ImportModel — moved to ModelImporter.cc

}  // namespace cosmo::service
