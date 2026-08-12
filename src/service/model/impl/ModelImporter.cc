// ModelImporter.cc — Import-related operations for ModelImportExporter.
// Split from ModelImportExporter.cc to reduce file size (DEBT-007).

// clang-format off
#include "service/detail/ServiceRegistry.h"
#include "service/model/impl/ModelImportExporter.h"
// clang-format on

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <system_error>
#include <vector>

#include "nlohmann/json.hpp"
#include "service/model/impl/ModelConfigParser.h"
#include "util/ArchiveListingValidator.h"
#include "util/ErrorCode.h"
#include "util/Exception.h"
#include "util/Exec.h"
#include "util/JsonFileUtil.h"
#include "util/NnBackendConstants.h"
#include "util/PathUtil.h"
#include "util/ResourceBudget.h"
#include "util/UuidUtil.h"

#ifdef COSMO_NN_USE_RKNN_BACKEND
#include "rknn_api.h"
#endif

#ifdef COSMO_NN_USE_ASCEND_BACKEND
#include "acl/acl.h"
#include "nn/device/ascend/ascend_acl.h"
#endif

namespace cosmo::service {

namespace {

    constexpr size_t kMaxArchiveEntries = 10000;

    enum class ArchiveKind {
        kUnknown,
        kZip,
        kTarGzip,
    };

    ArchiveKind DetectArchiveKind(const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        std::array<unsigned char, 4> header{};
        if (!stream.read(reinterpret_cast<char*>(header.data()), header.size())) {
            return ArchiveKind::kUnknown;
        }
        if (header[0] == 'P' && header[1] == 'K' &&
            ((header[2] == 3 && header[3] == 4) || (header[2] == 5 && header[3] == 6) ||
             (header[2] == 7 && header[3] == 8))) {
            return ArchiveKind::kZip;
        }
        if (header[0] == 0x1f && header[1] == 0x8b) {
            return ArchiveKind::kTarGzip;
        }
        return ArchiveKind::kUnknown;
    }

    bool InspectArchiveListing(const std::string& archive_path, bool is_zip,
                               const std::string& extraction_root,
                               util::ArchiveListingInspection& inspection) {
        const auto format =
            is_zip ? util::ArchiveListingFormat::kZipVerbose : util::ArchiveListingFormat::kTarVerbose;
        if (!util::InspectArchiveListingFile(archive_path, format, kMaxArchiveEntries, inspection) ||
            inspection.total_bytes == 0) {
            LOG_WARN("[ImportModel] Failed to inspect archive member list: {}", archive_path);
            return false;
        }
        const auto budget = util::InspectStorageResourceBudget(extraction_root);
        if (!budget.valid) {
            throw util::ErrorMessage(util::ErrorEnum::SysErr, "Cannot inspect model extraction storage");
        }
        if (inspection.total_bytes > budget.usable_bytes) {
            throw util::ResourceLimitError("Insufficient safe disk space to extract the model archive",
                                           "archive-extraction", "model-archive", inspection.total_bytes,
                                           budget.usable_bytes, budget.reserve_bytes);
        }
        return true;
    }

    bool ValidateExtractedTree(const std::string& root, const util::ArchiveListingInspection& inspection) {
        namespace fs              = std::filesystem;
        size_t entry_count        = 0;
        std::uintmax_t total_size = 0;
        std::error_code ec;
        for (fs::recursive_directory_iterator it(root, fs::directory_options::none, ec), end;
             !ec && it != end; it.increment(ec)) {
            if (++entry_count > kMaxArchiveEntries) {
                return false;
            }

            const auto link_status = it->symlink_status(ec);
            if (ec || fs::is_symlink(link_status) ||
                (!fs::is_directory(link_status) && !fs::is_regular_file(link_status))) {
                return false;
            }

            std::string resolved;
            if (!cosmo::path::ResolveExistingPathWithinRoot(root, it->path().string(),
                                                            cosmo::path::PathEntryType::kAny, resolved)) {
                return false;
            }
            const auto relative = fs::relative(it->path(), root, ec);
            if (ec || relative.empty()) {
                return false;
            }
            for (const auto& component : relative) {
                if (!cosmo::path::IsSafePathComponent(component.string())) {
                    return false;
                }
            }
            if (fs::is_regular_file(link_status)) {
                const auto size = fs::file_size(it->path(), ec);
                if (ec || size > inspection.largest_file_bytes || total_size > inspection.total_bytes ||
                    size > inspection.total_bytes - total_size) {
                    return false;
                }
                total_size += size;
            }
        }
        return !ec;
    }

    bool ResolveModelDestination(const std::string& models_dir, const std::string& component,
                                 std::string& destination) {
        return cosmo::path::IsSafePathComponent(component, 200) &&
               cosmo::path::ResolvePathWithinRoot(
                   models_dir, (std::filesystem::path(models_dir) / component).string(), destination);
    }

    bool ResolveManagedModelUpload(const std::string& path, std::string& resolved) {
        return cosmo::path::ResolveExistingPathWithinRoot(cosmo::path::GetModelUploadTmpDir(), path,
                                                          cosmo::path::PathEntryType::kRegularFile,
                                                          resolved) ||
               cosmo::path::ResolveExistingPathWithinRoot(cosmo::path::GetUploadPath(), path,
                                                          cosmo::path::PathEntryType::kRegularFile, resolved);
    }

    bool RemoveExistingModelDirectory(const std::string& models_dir, const std::string& destination) {
        namespace fs = std::filesystem;
        std::error_code ec;
        const auto link_status = fs::symlink_status(destination, ec);
        if (ec) {
            if (ec == std::errc::no_such_file_or_directory) {
                return true;
            }
            return false;
        }
        if (!fs::exists(link_status)) {
            return true;
        }
        if (fs::is_symlink(link_status)) {
            return false;
        }
        std::string resolved;
        if (!cosmo::path::ResolveExistingPathWithinRoot(models_dir, destination,
                                                        cosmo::path::PathEntryType::kDirectory, resolved)) {
            return false;
        }
        fs::remove_all(resolved, ec);
        return !ec;
    }

    bool IsChipType(const std::string& chip_type, const char* expected) {
        const size_t expected_length = std::char_traits<char>::length(expected);
        return chip_type.size() == expected_length &&
               std::equal(chip_type.begin(), chip_type.end(), expected, expected + expected_length,
                          [](char lhs, char rhs) {
                              return std::toupper(static_cast<unsigned char>(lhs)) ==
                                     std::toupper(static_cast<unsigned char>(rhs));
                          });
    }

    bool IsKnownModelArtifact(const std::filesystem::path& path) {
        const auto extension = path.extension().string();
        return extension == ".onnx" || extension == ".nn" || extension == ".bmodel" || extension == ".rknn" ||
               extension == ".om";
    }

    std::vector<std::string> ScanPackageArtifacts(const std::string& model_dir) {
        namespace fs = std::filesystem;
        std::vector<std::string> artifacts;
        std::error_code ec;
        for (fs::directory_iterator it(model_dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_regular_file() || !IsKnownModelArtifact(it->path()))
                continue;
            artifacts.push_back(it->path().string());
        }
        std::sort(artifacts.begin(), artifacts.end());
        return artifacts;
    }

    bool ResolvePackageArtifactPath(const std::string& model_dir, const std::string& file_name,
                                    std::string& resolved_path) {
        const std::filesystem::path artifact_name(file_name);
        if (file_name.empty() || artifact_name.has_parent_path() ||
            artifact_name.filename() != artifact_name || !cosmo::path::IsSafePathComponent(file_name, 200)) {
            return false;
        }
        return cosmo::path::ResolveExistingPathWithinRoot(
            model_dir, (std::filesystem::path(model_dir) / artifact_name).string(),
            cosmo::path::PathEntryType::kRegularFile, resolved_path);
    }

    std::vector<int> ReadShape(const nlohmann::json& tensor) {
        std::vector<int> shape;
        if (!tensor.contains("shape") || !tensor["shape"].is_array())
            return shape;
        for (const auto& value : tensor["shape"]) {
            if (!value.is_number_integer())
                return {};
            shape.push_back(value.get<int>());
        }
        return shape;
    }

    std::string JoinShape(const std::vector<int>& shape) {
        std::ostringstream stream;
        stream << '[';
        for (std::size_t index = 0; index < shape.size(); ++index) {
            if (index != 0)
                stream << ',';
            stream << shape[index];
        }
        stream << ']';
        return stream.str();
    }

    std::string DescribeConfigTensor(const nlohmann::json& tensor) {
        std::ostringstream stream;
        stream << "name=" << tensor.value("name", "?") << " shape=" << JoinShape(ReadShape(tensor))
               << " data_type=" << tensor.value("data_type", -1);
        if (tensor.contains("scale"))
            stream << " scale=" << tensor.value("scale", 0.0F);
        if (tensor.contains("zero_point"))
            stream << " zp=" << tensor.value("zero_point", 0);
        return stream.str();
    }

    std::string DescribeRuntimeTensor(const ModelImportExporter::TensorMetadata& tensor) {
        std::ostringstream stream;
        stream << "name=" << tensor.name << " shape=" << JoinShape(tensor.dims) << " fmt=" << tensor.format
               << " type=" << tensor.type << " quant=" << tensor.quant_type << " scale=" << tensor.scale
               << " zp=" << tensor.zero_point;
        return stream.str();
    }

    std::string MakeValidationError(const std::string& stage, const std::string& config_path,
                                    const std::string& model_dir, const std::string& detail) {
        return "stage=" + stage + " config=" + config_path + " model_dir=" + model_dir + " " + detail;
    }

    bool NearlyEqual(float lhs, float rhs) {
        return std::fabs(lhs - rhs) <= 1.0e-6F;
    }

    // config.json tensor "data_type" codes follow cosmo::nn::DataType
    // (4=UINT8, 5=INT8); 2 is HALF, used for the Ascend FP16 contract.
    constexpr int kConfigDataTypeFp16 = 2;

    // Phase-1 Ascend contract runs on device 0 (docs/development/ascend310p3-adaptation-plan.md).
    constexpr int32_t kAscendDeviceId = 0;

    bool LoadRknnMetadataFromRuntime(const std::string& artifact_path,
                                     ModelImportExporter::RknnModelMetadata& metadata, std::string& error) {
#ifdef COSMO_NN_USE_RKNN_BACKEND
        const auto to_tensor_metadata = [](const std::string& name, const std::vector<int>& dims,
                                           const std::string& format, const std::string& type,
                                           const std::string& quant_type, int zero_point, float scale) {
            ModelImportExporter::TensorMetadata tensor;
            tensor.name       = name;
            tensor.dims       = dims;
            tensor.format     = format;
            tensor.type       = type;
            tensor.quant_type = quant_type;
            tensor.zero_point = zero_point;
            tensor.scale      = scale;
            return tensor;
        };
        std::ifstream stream(artifact_path, std::ios::binary);
        if (!stream.is_open()) {
            error = "stage=rknn-open artifact=" + artifact_path + " cannot open .rknn file";
            return false;
        }
        const std::vector<unsigned char> model((std::istreambuf_iterator<char>(stream)),
                                               std::istreambuf_iterator<char>());
        if (model.empty()) {
            error = "stage=rknn-open artifact=" + artifact_path + " empty .rknn file";
            return false;
        }

        rknn_context ctx   = 0;
        const int init_ret = rknn_init(&ctx, const_cast<unsigned char*>(model.data()),
                                       static_cast<uint32_t>(model.size()), 0, nullptr);
        if (init_ret != RKNN_SUCC) {
            error =
                "stage=rknn-open artifact=" + artifact_path + " rknn_init ret=" + std::to_string(init_ret);
            return false;
        }

        struct ContextGuard {
            rknn_context ctx;
            ~ContextGuard() {
                if (ctx != 0)
                    (void)rknn_destroy(ctx);
            }
        } guard{ctx};

        rknn_input_output_num io_num{};
        const int io_ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (io_ret != RKNN_SUCC) {
            error = "stage=rknn-query artifact=" + artifact_path +
                    " RKNN_QUERY_IN_OUT_NUM ret=" + std::to_string(io_ret);
            return false;
        }

        metadata.inputs.clear();
        metadata.outputs.clear();
        for (uint32_t index = 0; index < io_num.n_input; ++index) {
            rknn_tensor_attr attr{};
            attr.index      = index;
            const int query = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
            if (query != RKNN_SUCC) {
                error = "stage=rknn-query artifact=" + artifact_path + " RKNN_QUERY_INPUT_ATTR[" +
                        std::to_string(index) + "] ret=" + std::to_string(query);
                return false;
            }
            metadata.inputs.push_back(to_tensor_metadata(
                attr.name, std::vector<int>(attr.dims, attr.dims + attr.n_dims), get_format_string(attr.fmt),
                get_type_string(attr.type), get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale));
        }
        for (uint32_t index = 0; index < io_num.n_output; ++index) {
            rknn_tensor_attr attr{};
            attr.index      = index;
            const int query = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
            if (query != RKNN_SUCC) {
                error = "stage=rknn-query artifact=" + artifact_path + " RKNN_QUERY_OUTPUT_ATTR[" +
                        std::to_string(index) + "] ret=" + std::to_string(query);
                return false;
            }
            metadata.outputs.push_back(to_tensor_metadata(
                attr.name, std::vector<int>(attr.dims, attr.dims + attr.n_dims), get_format_string(attr.fmt),
                get_type_string(attr.type), get_qnt_type_string(attr.qnt_type), attr.zp, attr.scale));
        }
        return true;
#else
        (void)artifact_path;
        metadata = {};
        error    = "stage=rknn-open RKNN metadata loader unavailable in this build";
        return false;
#endif
    }

#ifdef COSMO_NN_USE_ASCEND_BACKEND
    std::string AclDataTypeName(aclDataType type) {
        switch (type) {
            case ACL_FLOAT:
                return "FP32";
            case ACL_FLOAT16:
                return "FP16";
            case ACL_INT8:
                return "INT8";
            case ACL_INT32:
                return "INT32";
            case ACL_UINT8:
                return "UINT8";
            case ACL_INT64:
                return "INT64";
            case ACL_DOUBLE:
                return "FP64";
            case ACL_BF16:
                return "BF16";
            default:
                return "UNKNOWN";
        }
    }

    std::string AclFormatName(aclFormat format) {
        switch (format) {
            case ACL_FORMAT_NCHW:
                return "NCHW";
            case ACL_FORMAT_NHWC:
                return "NHWC";
            case ACL_FORMAT_ND:
                return "ND";
            case ACL_FORMAT_NC1HWC0:
                return "NC1HWC0";
            default:
                return "UNDEFINED";
        }
    }

    bool AclIoDimsToVector(const aclmdlIODims& io_dims, std::vector<int>& dims) {
        dims.clear();
        if (io_dims.dimCount == 0 || io_dims.dimCount > ACL_MAX_DIM_CNT)
            return false;
        dims.reserve(io_dims.dimCount);
        for (size_t index = 0; index < io_dims.dimCount; ++index) {
            const int64_t value = io_dims.dims[index];
            if (value <= 0 || value > std::numeric_limits<int>::max())
                return false;
            dims.push_back(static_cast<int>(value));
        }
        return true;
    }
#endif

    bool LoadAscendOmMetadataFromRuntime(const std::string& artifact_path,
                                         ModelImportExporter::AscendModelMetadata& metadata,
                                         std::string& error) {
#ifdef COSMO_NN_USE_ASCEND_BACKEND
        const auto to_tensor_metadata = [](const std::string& name, const std::vector<int>& dims,
                                           const std::string& format, const std::string& type) {
            ModelImportExporter::TensorMetadata tensor;
            tensor.name       = name;
            tensor.dims       = dims;
            tensor.format     = format;
            tensor.type       = type;
            tensor.quant_type = "NONE";
            return tensor;
        };

        if (const aclError init_ret = cosmo::nn::ascend::EnsureAclInitialized();
            init_ret != ACL_SUCCESS) {
            error =
                "stage=ascend-open artifact=" + artifact_path +
                " aclInit ret=" + std::to_string(init_ret);
            return false;
        }

        if (const aclError set_ret = aclrtSetDevice(kAscendDeviceId); set_ret != ACL_SUCCESS) {
            error = "stage=ascend-open artifact=" + artifact_path +
                    " aclrtSetDevice ret=" + std::to_string(set_ret);
            return false;
        }
        struct DeviceGuard {
            ~DeviceGuard() {
                (void)aclrtResetDevice(kAscendDeviceId);
            }
        } device_guard;

        uint32_t model_id = 0;
        if (const aclError load_ret = aclmdlLoadFromFile(artifact_path.c_str(), &model_id);
            load_ret != ACL_SUCCESS) {
            error = "stage=ascend-open artifact=" + artifact_path +
                    " aclmdlLoadFromFile ret=" + std::to_string(load_ret);
            return false;
        }
        struct ModelGuard {
            uint32_t model_id;
            ~ModelGuard() {
                (void)aclmdlUnload(model_id);
            }
        } model_guard{model_id};

        aclmdlDesc* desc = aclmdlCreateDesc();
        if (desc == nullptr) {
            error = "stage=ascend-open artifact=" + artifact_path + " aclmdlCreateDesc failed";
            return false;
        }
        struct DescGuard {
            aclmdlDesc* desc;
            ~DescGuard() {
                (void)aclmdlDestroyDesc(desc);
            }
        } desc_guard{desc};

        if (const aclError desc_ret = aclmdlGetDesc(desc, model_id); desc_ret != ACL_SUCCESS) {
            error = "stage=ascend-query artifact=" + artifact_path +
                    " aclmdlGetDesc ret=" + std::to_string(desc_ret);
            return false;
        }

        metadata.inputs.clear();
        metadata.outputs.clear();
        const size_t num_inputs  = aclmdlGetNumInputs(desc);
        const size_t num_outputs = aclmdlGetNumOutputs(desc);
        for (size_t index = 0; index < num_inputs; ++index) {
            aclmdlIODims io_dims{};
            if (const aclError dims_ret = aclmdlGetInputDims(desc, index, &io_dims);
                dims_ret != ACL_SUCCESS) {
                error = "stage=ascend-query artifact=" + artifact_path + " aclmdlGetInputDims[" +
                        std::to_string(index) + "] ret=" + std::to_string(dims_ret);
                return false;
            }
            std::vector<int> dims;
            if (!AclIoDimsToVector(io_dims, dims)) {
                error = "stage=ascend-query artifact=" + artifact_path + " aclmdlGetInputDims[" +
                        std::to_string(index) + "] invalid dims";
                return false;
            }
            metadata.inputs.push_back(
                to_tensor_metadata(io_dims.name, dims, AclFormatName(aclmdlGetInputFormat(desc, index)),
                                   AclDataTypeName(aclmdlGetInputDataType(desc, index))));
        }
        for (size_t index = 0; index < num_outputs; ++index) {
            aclmdlIODims io_dims{};
            if (const aclError dims_ret = aclmdlGetOutputDims(desc, index, &io_dims);
                dims_ret != ACL_SUCCESS) {
                error = "stage=ascend-query artifact=" + artifact_path + " aclmdlGetOutputDims[" +
                        std::to_string(index) + "] ret=" + std::to_string(dims_ret);
                return false;
            }
            std::vector<int> dims;
            if (!AclIoDimsToVector(io_dims, dims)) {
                error = "stage=ascend-query artifact=" + artifact_path + " aclmdlGetOutputDims[" +
                        std::to_string(index) + "] invalid dims";
                return false;
            }
            metadata.outputs.push_back(
                to_tensor_metadata(io_dims.name, dims, AclFormatName(aclmdlGetOutputFormat(desc, index)),
                                   AclDataTypeName(aclmdlGetOutputDataType(desc, index))));
        }
        return true;
#else
        (void)artifact_path;
        metadata = {};
        error    = "stage=ascend-open AscendCL metadata loader unavailable in this build";
        return false;
#endif
    }

}  // namespace

bool ModelImportExporter::ValidateImportedModelPackage(const std::string& model_dir, std::string& alg_code,
                                                       std::string& error) {
    const std::string config_path = (std::filesystem::path(model_dir) / "config.json").string();
    const auto parsed             = detail::ModelConfigParser::Parse(config_path);
    if (!parsed.valid) {
        error = MakeValidationError("config", config_path, model_dir, "invalid config.json");
        return false;
    }
    if (!cosmo::util::IsSupportedChip(parsed.chip_type)) {
        error = MakeValidationError("platform", config_path, model_dir,
                                    "unsupported chip_type=" + parsed.chip_type);
        return false;
    }
    detail::ResolvedModelArtifacts artifacts;
    std::string artifact_error;
    if (!detail::ModelConfigParser::ResolveModelArtifacts(config_path, model_dir, artifacts,
                                                          artifact_error)) {
        error = MakeValidationError("artifact", config_path, model_dir, artifact_error);
        return false;
    }
    if (!ValidateModelPackageContract(config_path, model_dir, error))
        return false;

    alg_code = parsed.algorithm_code;
    return true;
}

bool ModelImportExporter::ValidateModelPackageContract(const std::string& config_path,
                                                       const std::string& model_dir, std::string& error) {
    error.clear();

    nlohmann::json doc;
    if (cosmo::util::JsonFileUtil::ReadJsonFile(config_path, doc) != cosmo::util::ErrorEnum::Success) {
        error = MakeValidationError("config", config_path, model_dir, "cannot read config.json");
        return false;
    }

    const std::string chip_type  = doc.value("chip_type", std::string());
    const std::string model_type = doc.value("model_type", std::string());

    if (IsChipType(chip_type, "ASCEND310P3")) {
        if (model_type != "yolo26_det") {
            error = MakeValidationError(
                "config", config_path, model_dir,
                "ASCEND310P3 package only supports model_type=yolo26_det, got=" + model_type);
            return false;
        }

        if (!doc.contains("models") || !doc["models"].is_array() || doc["models"].size() != 1 ||
            !doc["models"][0].is_object()) {
            error = MakeValidationError("config", config_path, model_dir,
                                        "ASCEND310P3 YOLO26 package must declare exactly one models[] entry");
            return false;
        }

        const auto& model           = doc["models"][0];
        const std::string file_name = model.value("file_name", std::string());
        std::string artifact_path;
        if (!ResolvePackageArtifactPath(model_dir, file_name, artifact_path)) {
            error = MakeValidationError("artifact", config_path, model_dir,
                                        "ASCEND310P3 package must declare one explicit .om file_name");
            return false;
        }
        if (std::filesystem::path(artifact_path).extension() != ".om") {
            error = MakeValidationError("artifact", config_path, model_dir,
                                        "ASCEND310P3 package artifact must be .om: " + artifact_path);
            return false;
        }

        const auto package_artifacts = ScanPackageArtifacts(model_dir);
        if (package_artifacts.size() != 1 || package_artifacts.front() != artifact_path) {
            std::ostringstream detail;
            detail << "single-artifact ASCEND310P3 package expected exactly one .om artifact, found "
                   << package_artifacts.size();
            for (const auto& path : package_artifacts)
                detail << " [" << std::filesystem::path(path).filename().string() << "]";
            error = MakeValidationError("artifact", config_path, model_dir, detail.str());
            return false;
        }

        const auto& params = model.contains("params") && model["params"].is_object()
                                 ? model["params"]
                                 : nlohmann::json::object();
        if (params.value("preprocess_mode", std::string()) != "image_to_tensor") {
            error = MakeValidationError(
                "config", config_path, model_dir,
                "ASCEND310P3 YOLO26 package requires params.preprocess_mode=image_to_tensor");
            return false;
        }
        if (params.value("output_format", std::string()) != "yolo_e2e") {
            error = MakeValidationError("config", config_path, model_dir,
                                        "ASCEND310P3 YOLO26 package requires params.output_format=yolo_e2e");
            return false;
        }
        if (!params.contains("input_size") || !params["input_size"].is_array() ||
            params["input_size"].size() != 2) {
            error = MakeValidationError("config", config_path, model_dir,
                                        "ASCEND310P3 YOLO26 package requires params.input_size=[w,h]");
            return false;
        }

        if (!model.contains("inputs") || !model["inputs"].is_array() || model["inputs"].size() != 1) {
            error = MakeValidationError("config", config_path, model_dir,
                                        "ASCEND310P3 YOLO26 package must declare one NCHW FP16 input tensor");
            return false;
        }
        if (!model.contains("outputs") || !model["outputs"].is_array() || model["outputs"].size() != 1) {
            error = MakeValidationError("config", config_path, model_dir,
                                        "ASCEND310P3 YOLO26 package must declare one FP16 output tensor");
            return false;
        }

        const auto input_size  = ReadShape(nlohmann::json{{"shape", params["input_size"]}});
        const auto input_w     = input_size.at(0);
        const auto input_h     = input_size.at(1);
        const auto input_cfg   = model["inputs"][0];
        const auto input_shape = ReadShape(input_cfg);
        if (input_shape != std::vector<int>{1, 3, input_h, input_w} ||
            input_cfg.value("data_type", -1) != kConfigDataTypeFp16) {
            error = MakeValidationError(
                "config", config_path, model_dir,
                "ASCEND310P3 YOLO26 input must be NCHW FP16 [1,3,H,W]: " + DescribeConfigTensor(input_cfg));
            return false;
        }

        const auto& output_cfg  = model["outputs"][0];
        const auto output_shape = ReadShape(output_cfg);
        // Fixed-shape end2end contract (docs/development/ascend310p3-yolo26-om-atc.md):
        // rows are x1,y1,x2,y2,score,class_id, capped at max_det=300 by the NMS head.
        const bool output_ok = output_cfg.value("data_type", -1) == kConfigDataTypeFp16 &&
                               output_shape == std::vector<int>{1, 300, 6};
        if (!output_ok) {
            error = MakeValidationError(
                "config", config_path, model_dir,
                "ASCEND310P3 YOLO26 output must be FP16 [1,300,6] end2end tensor: " +
                    DescribeConfigTensor(output_cfg));
            return false;
        }

        auto metadata_loader = ascend_metadata_loader_;
        if (!metadata_loader)
            metadata_loader = LoadAscendOmMetadataFromRuntime;

        AscendModelMetadata metadata;
        std::string metadata_error;
        if (!metadata_loader(artifact_path, metadata, metadata_error)) {
            error = MakeValidationError("ascend", config_path, model_dir,
                                        metadata_error + " artifact=" + artifact_path);
            return false;
        }
        if (metadata.inputs.size() != 1 || metadata.outputs.size() != 1) {
            std::ostringstream detail;
            detail << "expected 1 input and 1 output from AscendCL metadata, got inputs="
                   << metadata.inputs.size() << " outputs=" << metadata.outputs.size();
            error = MakeValidationError("ascend", config_path, model_dir, detail.str());
            return false;
        }

        const auto& runtime_input = metadata.inputs.front();
        if (runtime_input.format != "NCHW" || runtime_input.type != "FP16" ||
            runtime_input.dims != input_shape) {
            error = MakeValidationError("input", config_path, model_dir,
                                        "expected NCHW FP16 input " + DescribeConfigTensor(input_cfg) +
                                            ", got " + DescribeRuntimeTensor(runtime_input));
            return false;
        }

        const auto& runtime_output = metadata.outputs.front();
        if (runtime_output.format != "ND" || runtime_output.type != "FP16" ||
            runtime_output.dims != output_shape) {
            error = MakeValidationError("output[0]", config_path, model_dir,
                                        "config=" + DescribeConfigTensor(output_cfg) +
                                            " runtime=" + DescribeRuntimeTensor(runtime_output));
            return false;
        }

        return true;
    }

    if (!IsChipType(chip_type, "RK3588"))
        return true;

    if (model_type != "yolo26_det") {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 package only supports model_type=yolo26_det, got=" + model_type);
        return false;
    }

    if (!doc.contains("models") || !doc["models"].is_array() || doc["models"].size() != 1 ||
        !doc["models"][0].is_object()) {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package must declare exactly one models[] entry");
        return false;
    }

    const auto& model           = doc["models"][0];
    const std::string file_name = model.value("file_name", std::string());
    std::string artifact_path;
    if (!ResolvePackageArtifactPath(model_dir, file_name, artifact_path)) {
        error = MakeValidationError("artifact", config_path, model_dir,
                                    "RK3588 package must declare one explicit .rknn file_name");
        return false;
    }
    if (std::filesystem::path(artifact_path).extension() != ".rknn") {
        error = MakeValidationError("artifact", config_path, model_dir,
                                    "RK3588 package artifact must be .rknn: " + artifact_path);
        return false;
    }

    const auto package_artifacts = ScanPackageArtifacts(model_dir);
    if (package_artifacts.size() != 1 || package_artifacts.front() != artifact_path) {
        std::ostringstream detail;
        detail << "single-artifact RK3588 package expected exactly one .rknn artifact, found "
               << package_artifacts.size();
        for (const auto& path : package_artifacts)
            detail << " [" << std::filesystem::path(path).filename().string() << "]";
        error = MakeValidationError("artifact", config_path, model_dir, detail.str());
        return false;
    }

    const auto& params =
        model.contains("params") && model["params"].is_object() ? model["params"] : nlohmann::json::object();
    if (params.value("preprocess_mode", std::string()) != "image_to_tensor") {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package requires params.preprocess_mode=image_to_tensor");
        return false;
    }
    if (params.value("output_format", std::string()) != "yolo26_raw") {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package requires params.output_format=yolo26_raw");
        return false;
    }
    if (params.value("reg_max", 1) != 1) {
        error =
            MakeValidationError("config", config_path, model_dir, "RK3588 YOLO26 package requires reg_max=1");
        return false;
    }
    if (!params.contains("input_size") || !params["input_size"].is_array() ||
        params["input_size"].size() != 2) {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package requires params.input_size=[w,h]");
        return false;
    }

    if (!model.contains("inputs") || !model["inputs"].is_array() || model["inputs"].size() != 1) {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package must declare one NHWC input tensor");
        return false;
    }
    if (!model.contains("outputs") || !model["outputs"].is_array() || model["outputs"].size() != 6) {
        error = MakeValidationError("config", config_path, model_dir,
                                    "RK3588 YOLO26 package must declare six NCHW output tensors");
        return false;
    }

    const auto input_size  = ReadShape(nlohmann::json{{"shape", params["input_size"]}});
    const auto input_w     = input_size.at(0);
    const auto input_h     = input_size.at(1);
    const auto input_cfg   = model["inputs"][0];
    const auto input_shape = ReadShape(input_cfg);
    if (input_shape != std::vector<int>{1, input_h, input_w, 3} || input_cfg.value("data_type", -1) != 4) {
        error = MakeValidationError(
            "config", config_path, model_dir,
            "RK3588 YOLO26 input must be NHWC UINT8 [1,H,W,3]: " + DescribeConfigTensor(input_cfg));
        return false;
    }

    int class_count = -1;
    for (std::size_t index = 0; index < model["outputs"].size(); index += 2) {
        const auto& reg_cfg  = model["outputs"][index];
        const auto& cls_cfg  = model["outputs"][index + 1];
        const auto reg_shape = ReadShape(reg_cfg);
        const auto cls_shape = ReadShape(cls_cfg);
        const bool reg_ok    = reg_cfg.value("data_type", -1) == 5 && reg_shape.size() == 4 &&
                            reg_shape[0] == 1 && reg_shape[1] == 4 && reg_shape[2] > 0 && reg_shape[3] > 0 &&
                            reg_cfg.contains("scale") && reg_cfg.contains("zero_point") &&
                            reg_cfg.value("scale", 0.0F) > 0.0F;
        const bool cls_ok = cls_cfg.value("data_type", -1) == 5 && cls_shape.size() == 4 &&
                            cls_shape[0] == 1 && cls_shape[1] > 0 && cls_shape[2] == reg_shape[2] &&
                            cls_shape[3] == reg_shape[3] && cls_cfg.contains("scale") &&
                            cls_cfg.contains("zero_point") && cls_cfg.value("scale", 0.0F) > 0.0F;
        if (!reg_ok || !cls_ok) {
            error =
                MakeValidationError("config", config_path, model_dir,
                                    "invalid RK3588 YOLO26 output pair: " + DescribeConfigTensor(reg_cfg) +
                                        " | " + DescribeConfigTensor(cls_cfg));
            return false;
        }
        if (input_h % reg_shape[2] != 0 || input_w % reg_shape[3] != 0 ||
            input_h / reg_shape[2] != input_w / reg_shape[3]) {
            error = MakeValidationError(
                "config", config_path, model_dir,
                "output feature map does not divide input_size cleanly: " + DescribeConfigTensor(reg_cfg));
            return false;
        }
        if (class_count == -1)
            class_count = cls_shape[1];
        else if (class_count != cls_shape[1]) {
            error = MakeValidationError("config", config_path, model_dir,
                                        "output class channels must match across scales");
            return false;
        }
    }

    auto metadata_loader = rknn_metadata_loader_;
    if (!metadata_loader)
        metadata_loader = LoadRknnMetadataFromRuntime;

    RknnModelMetadata metadata;
    std::string metadata_error;
    if (!metadata_loader(artifact_path, metadata, metadata_error)) {
        error = MakeValidationError("rknn", config_path, model_dir,
                                    metadata_error + " artifact=" + artifact_path);
        return false;
    }
    if (metadata.inputs.size() != 1 || metadata.outputs.size() != 6) {
        std::ostringstream detail;
        detail << "expected 1 input and 6 outputs from RKNN metadata, got inputs=" << metadata.inputs.size()
               << " outputs=" << metadata.outputs.size();
        error = MakeValidationError("rknn", config_path, model_dir, detail.str());
        return false;
    }

    const auto& runtime_input = metadata.inputs.front();
    if (runtime_input.format != "NHWC" || runtime_input.type != "UINT8" ||
        runtime_input.dims != std::vector<int>{1, input_h, input_w, 3}) {
        error = MakeValidationError("input", config_path, model_dir,
                                    "expected NHWC UINT8 input " + DescribeConfigTensor(input_cfg) +
                                        ", got " + DescribeRuntimeTensor(runtime_input));
        return false;
    }

    for (std::size_t index = 0; index < metadata.outputs.size(); ++index) {
        const auto& runtime_output = metadata.outputs[index];
        const auto& config_output  = model["outputs"][index];
        const auto config_shape    = ReadShape(config_output);
        const bool matches         = runtime_output.format == "NCHW" && runtime_output.type == "INT8" &&
                             runtime_output.quant_type == "AFFINE" && runtime_output.dims == config_shape &&
                             runtime_output.zero_point == config_output.value("zero_point", 0) &&
                             NearlyEqual(runtime_output.scale, config_output.value("scale", 0.0F));
        if (!matches) {
            error = MakeValidationError("output[" + std::to_string(index) + "]", config_path, model_dir,
                                        "config=" + DescribeConfigTensor(config_output) +
                                            " runtime=" + DescribeRuntimeTensor(runtime_output));
            return false;
        }
    }

    return true;
}

util::ErrorEnum ModelImportExporter::ImportFlatArchive(const std::string& temp_dir,
                                                       const std::string& models_dir) {
    namespace fs = std::filesystem;
    std::error_code ec;

    std::string config_path = temp_dir + "/config.json";
    std::ifstream cfgFile(config_path);
    if (!cfgFile.is_open()) {
        LOG_WARN("{}", "[ImportModel] Cannot open config.json in flat archive");
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::InvalidParam;
    }
    std::stringstream cfgBuf;
    cfgBuf << cfgFile.rdbuf();
    cfgFile.close();

    std::string alg_code  = "0000000";
    std::string modelName = "imported";
    std::string version   = "V1.0.0";
    try {
        auto doc = nlohmann::json::parse(cfgBuf.str());
        if (doc.is_object()) {
            if (doc.contains("algorithm_code") && doc["algorithm_code"].is_string())
                alg_code = doc["algorithm_code"].get<std::string>();
            if (doc.contains("version") && doc["version"].is_string())
                version = doc["version"].get<std::string>();
            if (doc.contains("models") && doc["models"].is_array() && !doc["models"].empty()) {
                const auto& m = doc["models"][0];
                if (m.contains("name") && m["name"].is_string())
                    modelName = m["name"].get<std::string>();
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN("[ImportModel] Failed to parse config.json: {}", e.what());
    }

    static const std::regex kAlgorithmCodePattern("^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$");
    static const std::regex kVersionPattern("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$");
    if (!std::regex_match(alg_code, kAlgorithmCodePattern) || !std::regex_match(version, kVersionPattern) ||
        !cosmo::path::IsSafePathComponent(modelName, 64)) {
        LOG_WARN("{}", "[ImportModel] Reject unsafe model metadata components");
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::InvalidParam;
    }

    std::replace_if(
        modelName.begin(), modelName.end(),
        [](char c) {
            return c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' ||
                   c == '>' || c == '|' || c == ' ';
        },
        '_');

    std::string dir_name =
        std::string(cosmo::util::kNewDirPrefix) + alg_code + "_" + modelName + "_" + version;
    std::string dest_dir;
    if (!ResolveModelDestination(models_dir, dir_name, dest_dir)) {
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::InvalidParam;
    }

    // Remove existing if present
    if (!RemoveExistingModelDirectory(models_dir, dest_dir)) {
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::InvalidParam;
    }

    fs::rename(temp_dir, dest_dir, ec);
    if (ec) {
        // rename may fail across filesystems, fall back to copy
        ec.clear();
        fs::create_directories(dest_dir, ec);
        if (ec) {
            fs::remove_all(temp_dir, ec);
            return util::ErrorEnum::SysErr;
        }
        for (const auto& entry : fs::directory_iterator(temp_dir)) {
            ec.clear();
            fs::copy(entry.path(), fs::path(dest_dir) / entry.path().filename(),
                     fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
            if (ec) {
                std::error_code cleanup_ec;
                fs::remove_all(dest_dir, cleanup_ec);
                fs::remove_all(temp_dir, cleanup_ec);
                return util::ErrorEnum::SysErr;
            }
        }
        fs::remove_all(temp_dir, ec);
    }

    if (set_model_path_mapping_) {
        set_model_path_mapping_(alg_code, dest_dir);
    }

    LOG_INFO("[ImportModel] Imported flat archive as: {}", dir_name);
    return util::ErrorEnum::Success;
}

util::ErrorEnum ModelImportExporter::ImportDirectoryArchive(const std::string& temp_dir,
                                                            const std::string& models_dir,
                                                            int& imported_count) {
    namespace fs = std::filesystem;
    std::error_code ec;

    for (const auto& dirEntry : fs::directory_iterator(temp_dir)) {
        std::error_code status_ec;
        if (dirEntry.is_symlink(status_ec) || !dirEntry.is_directory(status_ec))
            continue;
        std::string sub_dir      = dirEntry.path().string();
        std::string sub_dir_name = dirEntry.path().filename().string();
        std::string resolved_sub_dir;
        if (!cosmo::path::IsSafePathComponent(sub_dir_name, 200) ||
            !cosmo::path::ResolveExistingPathWithinRoot(
                temp_dir, sub_dir, cosmo::path::PathEntryType::kDirectory, resolved_sub_dir)) {
            LOG_WARN("[ImportModel] Skipping unsafe model directory: {}", sub_dir_name);
            continue;
        }
        sub_dir = std::move(resolved_sub_dir);

        // Validate against the package config and compiled platform profile.
        std::string sub_alg_code;
        std::string validation_error;
        if (!fs::exists(sub_dir + "/config.json")) {
            LOG_WARN("[ImportModel] Skipping directory without config.json: {}", sub_dir_name);
            continue;
        }
        if (!ValidateImportedModelPackage(sub_dir, sub_alg_code, validation_error)) {
            LOG_WARN("[ImportModel] Skipping invalid model package {}: {}", sub_dir_name, validation_error);
            continue;
        }

        std::string dest_dir;
        if (!ResolveModelDestination(models_dir, sub_dir_name, dest_dir)) {
            LOG_WARN("[ImportModel] Skipping unsafe destination directory: {}", sub_dir_name);
            continue;
        }

        // Remove existing if present
        if (!RemoveExistingModelDirectory(models_dir, dest_dir)) {
            LOG_WARN("[ImportModel] Refusing to replace unsafe model destination: {}", dest_dir);
            continue;
        }

        fs::rename(sub_dir, dest_dir, ec);
        if (ec) {
            // Fall back to recursive copy
            fs::create_directories(dest_dir, ec);
            fs::copy(sub_dir, dest_dir, fs::copy_options::recursive | fs::copy_options::overwrite_existing,
                     ec);
            if (ec) {
                LOG_WARN("[ImportModel] Failed to copy model dir {}: {}", sub_dir_name, ec.message());
                continue;
            }
        }

        if (!sub_alg_code.empty() && set_model_path_mapping_) {
            set_model_path_mapping_(sub_alg_code, dest_dir);
        }

        imported_count++;
        LOG_INFO("[ImportModel] Imported model directory: {}", sub_dir_name);
    }

    return util::ErrorEnum::Success;
}

util::ErrorEnum ModelImportExporter::ImportModel(const std::string& archivePath) {
    namespace fs = std::filesystem;

    std::string managed_archive_path;
    if (!ResolveManagedModelUpload(archivePath, managed_archive_path)) {
        LOG_WARN("[ImportModel] Archive is not a managed upload: {}", archivePath);
        return util::ErrorEnum::FileNotExist;
    }

    size_t archive_size = 0;
    try {
        archive_size = fs::file_size(managed_archive_path);
    } catch (const std::exception& e) {
        LOG_WARN("[ImportModel] Failed to get archive size: {}", e.what());
    }
    if (archive_size == 0) {
        LOG_WARN("[ImportModel] Archive size is invalid: {}", archive_size);
        return util::ErrorEnum::InvalidParam;
    }
    LOG_INFO("[ImportModel] Starting managed import, size: {} bytes", archive_size);

    const auto temporary_root = cosmo::path::GetTemporaryDirPath();
    const auto archive_kind   = DetectArchiveKind(managed_archive_path);
    const bool is_zip         = archive_kind == ArchiveKind::kZip;
    util::ArchiveListingInspection inspection;
    if (archive_kind == ArchiveKind::kUnknown ||
        !InspectArchiveListing(managed_archive_path, is_zip, temporary_root, inspection)) {
        return util::ErrorEnum::InvalidParam;
    }

    // 1. Create temp extraction directory
    std::string temp_dir = (fs::path(temporary_root) / ("model_import_" + util::GenerateUUID())).string();
    std::error_code ec;
    fs::create_directories(temp_dir, ec);
    if (ec) {
        LOG_WARN("[ImportModel] Failed to create temp directory: {}", temp_dir);
        return util::ErrorEnum::SysErr;
    }

    // 2. Extract archive (detect format: tar.gz or zip)
    std::string out_str;
    std::vector<std::string> extract_argv;
    if (is_zip) {
        LOG_INFO("{}", "[ImportModel] Extracting managed ZIP archive");
        extract_argv = {"unzip", "-o", managed_archive_path, "-d", temp_dir};
    } else {
        LOG_INFO("{}", "[ImportModel] Extracting managed tar.gz archive");
        extract_argv = {"tar", "-xzf", managed_archive_path, "-C", temp_dir};
    }

    int extract_ret = util::Exec(extract_argv, out_str);
    if (extract_ret != 0) {
        LOG_WARN("[ImportModel] Extract failed (ret={}): {}", extract_ret, out_str);
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::SysErr;
    }
    if (!ValidateExtractedTree(temp_dir, inspection)) {
        LOG_WARN("{}", "[ImportModel] Extracted archive violates path, type, count, or size limits");
        fs::remove_all(temp_dir, ec);
        return util::ErrorEnum::InvalidParam;
    }

    // Log extracted contents for debugging
    {
        std::string listing = std::accumulate(
            fs::recursive_directory_iterator(temp_dir), fs::recursive_directory_iterator(), std::string{},
            [](std::string s, const auto& entry) { return std::move(s) + "\n  " + entry.path().string(); });
        LOG_INFO("[ImportModel] Extracted contents:{}", listing);
    }

    // 3. Find model directory(ies)
    const std::string models_dir = get_model_path_();
    int imported_count           = 0;

    // Check if the extracted content is a flat structure.
    bool flat_structure = false;
    std::string validation_error;
    if (fs::exists(temp_dir + "/config.json")) {
        std::string flat_alg_code;
        flat_structure = ValidateImportedModelPackage(temp_dir, flat_alg_code, validation_error);
        if (!flat_structure) {
            LOG_WARN("[ImportModel] Flat archive validation failed: {}", validation_error);
        }
    }

    if (flat_structure) {
        auto err = ImportFlatArchive(temp_dir, models_dir);
        if (err != util::ErrorEnum::Success)
            return err;
        imported_count = 1;
    } else {
        ImportDirectoryArchive(temp_dir, models_dir, imported_count);
    }

    // Cleanup
    fs::remove_all(temp_dir, ec);

    // Cleanup uploaded archive
    try {
        fs::remove(managed_archive_path);
    } catch (const std::exception& e) {
        LOG_WARN("[ImportModel] Failed to remove managed temp archive: {}", e.what());
    }

    if (imported_count == 0) {
        LOG_WARN("{}", "[ImportModel] No valid model directories found in archive");
        return util::ErrorEnum::InvalidParam;
    }

    LOG_INFO("[ImportModel] Successfully imported {} model(s)", imported_count);
    return util::ErrorEnum::Success;
}

}  // namespace cosmo::service
