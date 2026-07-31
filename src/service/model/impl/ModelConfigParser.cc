// ModelConfigParser — implementation.
// Centralizes the config.json parsing logic that was duplicated across
// QueryModels() and QueryAtomicModels().
#include "service/model/impl/ModelConfigParser.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <regex>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "util/ErrorCode.h"
#include "util/JsonFileUtil.h"
#include "util/Log.h"
#include "util/NnBackendConstants.h"
#include "util/PathUtil.h"

namespace cosmo::service::detail {

namespace fs = std::filesystem;

namespace {

    bool ResolveArtifactPath(const std::string& model_dir, const std::string& file_name,
                             std::string& resolved_path) {
        const fs::path artifact_name(file_name);
        if (file_name.empty() || artifact_name.has_parent_path() || artifact_name.filename() != artifact_name ||
            !cosmo::path::IsSafePathComponent(file_name, 200)) {
            return false;
        }
        return cosmo::path::ResolveExistingPathWithinRoot(
            model_dir, (fs::path(model_dir) / artifact_name).string(), cosmo::path::PathEntryType::kRegularFile,
            resolved_path);
    }

    std::vector<std::string> ScanLegacyArtifacts(const std::string& model_dir) {
        std::vector<std::string> artifacts;
        std::error_code ec;
        for (fs::directory_iterator it(model_dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file())
                continue;
            const auto extension = it->path().extension().string();
            if (!cosmo::util::IsSupportedModelFileExtension(extension))
                continue;
            artifacts.push_back(it->path().string());
        }
        std::sort(artifacts.begin(), artifacts.end());
        return artifacts;
    }

}  // namespace

// ─────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────

ParsedModelConfig ModelConfigParser::Parse(const std::string& config_path) {
    ParsedModelConfig result;

    nlohmann::json doc;
    cosmo::util::ErrorEnum ret = cosmo::util::JsonFileUtil::ReadJsonFile(config_path, doc);
    if (ret != cosmo::util::ErrorEnum::Success) {
        LOG_WARN("Failed to read config.json: {}, error: {}", config_path, static_cast<int>(ret));
        return result;
    }

    // Parse algorithm_code (string or int)
    if (doc.contains("algorithm_code") && doc["algorithm_code"].is_string()) {
        result.algorithm_code = doc["algorithm_code"].get<std::string>();
    } else if (doc.contains("algorithm_code") && doc["algorithm_code"].is_number_integer()) {
        result.algorithm_code = std::to_string(doc["algorithm_code"].get<int>());
    } else {
        LOG_WARN("config.json missing algorithm_code: {}", config_path);
        return result;
    }

    // Parse model name from models[0].name
    if (doc.contains("models") && doc["models"].is_array() && !doc["models"].empty()) {
        const auto& first_model = doc["models"][0];
        if (first_model.contains("name") && first_model["name"].is_string()) {
            result.model_name = first_model["name"].get<std::string>();
        } else {
            LOG_WARN("config.json models[0] missing name: {}", config_path);
        }
    } else {
        LOG_WARN("config.json missing or empty models array: {}", config_path);
    }

    // Parse model_type
    if (doc.contains("model_type") && doc["model_type"].is_string()) {
        result.model_type = doc["model_type"].get<std::string>();
    }

    // Parse chip_type (Sophon chip identifier, e.g. BM1688 / CV186X)
    if (doc.contains("chip_type") && doc["chip_type"].is_string()) {
        result.chip_type = doc["chip_type"].get<std::string>();
    }

    // Parse config.json "version" as fallback
    if (doc.contains("version") && doc["version"].is_string()) {
        result.version = doc["version"].get<std::string>();
    }

    // Parse I/O shapes
    result.input  = ParseIoShape(doc, "inputs", "1");
    result.output = ParseIoShape(doc, "outputs", "5");

    // Parse labels
    result.labels = ParseLabels(doc);

    result.valid = true;
    return result;
}

std::string ModelConfigParser::ParseVersionFromDirName(const std::string& dir_name,
                                                       const std::string& fallback_version) {
    // New format: ..._V1.0.3
    std::regex new_pattern("_V(\\d+)\\.0\\.(\\d+)$");
    std::smatch new_matches;
    if (std::regex_search(dir_name, new_matches, new_pattern) && new_matches.size() > 2) {
        return "V" + new_matches[1].str() + ".0." + new_matches[2].str();
    }

    // Legacy format: ..._v1003
    std::regex old_pattern("_v(\\d+)$");
    std::smatch old_matches;
    if (std::regex_search(dir_name, old_matches, old_pattern) && old_matches.size() > 1) {
        int old_version = std::stoi(old_matches[1].str());
        int major       = (old_version - 1) / 1000 + 1;
        int minor       = (old_version - 1) % 1000;
        return "V" + std::to_string(major) + ".0." + std::to_string(minor);
    }

    return fallback_version.empty() ? "V1.0.0" : fallback_version;
}

std::string ModelConfigParser::JoinShape(const std::vector<int>& shape) {
    std::string result;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            result += ",";
        result += std::to_string(shape[i]);
    }
    return result;
}

bool ModelConfigParser::ResolveModelArtifacts(const std::string& config_path, const std::string& model_dir,
                                              ResolvedModelArtifacts& artifacts, std::string& error) {
    artifacts = {};
    error.clear();

    nlohmann::json doc;
    if (cosmo::util::JsonFileUtil::ReadJsonFile(config_path, doc) != cosmo::util::ErrorEnum::Success) {
        error = "cannot read config.json";
        return false;
    }
    if (!doc.contains("models") || !doc["models"].is_array() || doc["models"].empty()) {
        error = "config.json missing models[]";
        return false;
    }

    std::vector<std::string> listed_files;
    listed_files.reserve(doc["models"].size());
    bool saw_listed_file = false;
    bool saw_empty_file  = false;
    for (const auto& model : doc["models"]) {
        if (!model.is_object()) {
            error = "config.json models[] must contain objects";
            return false;
        }
        const std::string file_name =
            (model.contains("file_name") && model["file_name"].is_string()) ? model["file_name"].get<std::string>()
                                                                            : std::string();
        if (file_name.empty()) {
            saw_empty_file = true;
            continue;
        }
        saw_listed_file = true;
        listed_files.push_back(file_name);
    }

    if (saw_listed_file) {
        if (saw_empty_file || listed_files.size() != doc["models"].size()) {
            error = "every model must declare file_name once any file_name is set";
            return false;
        }

        std::set<std::string> seen_files;
        for (const auto& file_name : listed_files) {
            std::string resolved_path;
            if (!ResolveArtifactPath(model_dir, file_name, resolved_path)) {
                error = "model file_name is missing or escapes package root: " + file_name;
                return false;
            }
            if (!cosmo::util::IsSupportedModelFileExtension(fs::path(resolved_path).extension().string())) {
                error = "model file_name does not match the compiled platform profile: " + file_name;
                return false;
            }
            if (!seen_files.insert(file_name).second) {
                error = "duplicate model file_name: " + file_name;
                return false;
            }
            artifacts.paths.push_back(std::move(resolved_path));
        }
        return true;
    }

    artifacts.paths = ScanLegacyArtifacts(model_dir);
    if (artifacts.paths.empty()) {
        error = "no compatible model artifact found";
        return false;
    }
    artifacts.used_compatibility_fallback = true;
    return true;
}

// ─────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────

std::vector<ParsedLabel> ModelConfigParser::ParseLabels(const nlohmann::json& doc) {
    std::vector<ParsedLabel> labels;
    if (!doc.contains("labels") || !doc["labels"].is_array())
        return labels;

    for (const auto& entry : doc["labels"]) {
        ParsedLabel label;

        // Parse label id (string or int)
        if (entry.contains("id") && entry["id"].is_string()) {
            label.id = entry["id"].get<std::string>();
        } else if (entry.contains("id") && entry["id"].is_number_integer()) {
            label.id = std::to_string(entry["id"].get<int>());
        }

        // Parse class_name
        if (entry.contains("name") && entry["name"].is_string()) {
            label.class_name = entry["name"].get<std::string>();
        }

        // Parse thresholds
        if (entry.contains("threshold") && entry["threshold"].is_array()) {
            for (const auto& th : entry["threshold"]) {
                if (th.is_number()) {
                    label.thresholds.push_back(th.get<float>());
                }
            }
        }

        // Only include labels with both id and class_name
        if (!label.id.empty() && !label.class_name.empty()) {
            labels.push_back(std::move(label));
        }
    }
    return labels;
}

ParsedIoShape ModelConfigParser::ParseIoShape(const nlohmann::json& doc, const std::string& array_key,
                                              const std::string& default_dim) {
    ParsedIoShape io;
    io.dim = default_dim;

    if (!doc.contains("models") || !doc["models"].is_array() || doc["models"].empty())
        return io;

    const auto& first_model = doc["models"][0];
    if (!first_model.contains(array_key) || !first_model[array_key].is_array())
        return io;

    const auto& io_array = first_model[array_key];
    io.count             = static_cast<int>(io_array.size());

    if (io.count > 0) {
        const auto& first_io = io_array[0];
        if (first_io.contains("shape") && first_io["shape"].is_array()) {
            std::vector<int> shape;
            for (const auto& s : first_io["shape"]) {
                if (s.is_number()) {
                    shape.push_back(s.get<int>());
                }
            }
            if (!shape.empty()) {
                io.dim = JoinShape(shape);
            }
        }
    }

    // Clamp defaults
    if (io.count == 0)
        io.count = 1;
    if (io.dim.empty())
        io.dim = default_dim;

    return io;
}

}  // namespace cosmo::service::detail
