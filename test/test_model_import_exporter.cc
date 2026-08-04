#include <filesystem>
#include <fstream>
#include <functional>
#include <vector>

#include "util/NnBackendConstants.h"
#include "util/PathUtil.h"

// clang-format off
#include "catch_amalgamated.hpp"
#include "catch2/trompeloeil.hpp"
// clang-format on
#include "nlohmann/json.hpp"
#ifdef COSMO_NN_USE_CPU_BACKEND
#include "nn/core/shared_resource.h"
#include "nn/device/cpu/cpu_crop_resize_node.h"
#include "nn/device/cpu/cpu_normalize_node.h"
#include "nn/utils/op.h"
#endif
#define private public
#include "service/model/impl/ModelImportExporter.h"
#undef private
#include "mock/MockServiceRegistry.h"
#include "util/Exec.h"
#include "util/JsonFileUtil.h"

using namespace cosmo::service;
using namespace cosmo::test;
namespace fs = std::filesystem;

namespace {

nlohmann::json MakeRk3588Yolo26Config() {
    return {{"algorithm_code", "8888001"},
            {"chip_type", "RK3588"},
            {"model_type", "yolo26_det"},
            {"version", "V1.0.0"},
            {"models",
             {{{"name", "rk_yolo26"},
               {"file_name", "model.rknn"},
               {"max_batch", 1},
               {"inputs", {{{"name", "images"}, {"shape", {1, 640, 640, 3}}, {"data_type", 4}}}},
               {"outputs",
                {{{"name", "reg0"},
                  {"shape", {1, 4, 80, 80}},
                  {"data_type", 5},
                  {"scale", 0.5},
                  {"zero_point", 0}},
                 {{"name", "cls0"},
                  {"shape", {1, 1, 80, 80}},
                  {"data_type", 5},
                  {"scale", 0.1},
                  {"zero_point", 0}},
                 {{"name", "reg1"},
                  {"shape", {1, 4, 40, 40}},
                  {"data_type", 5},
                  {"scale", 0.5},
                  {"zero_point", 0}},
                 {{"name", "cls1"},
                  {"shape", {1, 1, 40, 40}},
                  {"data_type", 5},
                  {"scale", 0.1},
                  {"zero_point", 0}},
                 {{"name", "reg2"},
                  {"shape", {1, 4, 20, 20}},
                  {"data_type", 5},
                  {"scale", 0.5},
                  {"zero_point", 0}},
                 {{"name", "cls2"},
                  {"shape", {1, 1, 20, 20}},
                  {"data_type", 5},
                  {"scale", 0.1},
                  {"zero_point", 0}}}},
               {"params",
                {{"preprocess_mode", "image_to_tensor"},
                 {"output_format", "yolo26_raw"},
                 {"input_size", {640, 640}},
                 {"padding_color", {114, 114, 114}},
                 {"confidence_threshold", 0.25},
                 {"nms_threshold", 0.45},
                 {"top_k", 300},
                 {"reg_max", 1}}}}}}};
}

ModelImportExporter::RknnModelMetadata MakeRknnYolo26Metadata() {
    using Tensor = ModelImportExporter::TensorMetadata;
    ModelImportExporter::RknnModelMetadata metadata;
    metadata.inputs.push_back({"images", {1, 640, 640, 3}, "NHWC", "UINT8", "NONE", 0, 0.0F});
    metadata.outputs = {
        Tensor{"reg0", {1, 4, 80, 80}, "NCHW", "INT8", "AFFINE", 0, 0.5F},
        Tensor{"cls0", {1, 1, 80, 80}, "NCHW", "INT8", "AFFINE", 0, 0.1F},
        Tensor{"reg1", {1, 4, 40, 40}, "NCHW", "INT8", "AFFINE", 0, 0.5F},
        Tensor{"cls1", {1, 1, 40, 40}, "NCHW", "INT8", "AFFINE", 0, 0.1F},
        Tensor{"reg2", {1, 4, 20, 20}, "NCHW", "INT8", "AFFINE", 0, 0.5F},
        Tensor{"cls2", {1, 1, 20, 20}, "NCHW", "INT8", "AFFINE", 0, 0.1F},
    };
    return metadata;
}

nlohmann::json MakeAscend310P3Yolo26Config() {
    return {{"algorithm_code", "8888002"},
            {"chip_type", "ASCEND310P3"},
            {"model_type", "yolo26_det"},
            {"version", "V1.0.0"},
            {"models",
             {{{"name", "ascend_yolo26"},
               {"file_name", "model.om"},
               {"max_batch", 1},
               {"inputs", {{{"name", "images"}, {"shape", {1, 3, 640, 640}}, {"data_type", 2}}}},
               {"outputs",
                {{{"name", "reg0"}, {"shape", {1, 4, 80, 80}}, {"data_type", 2}},
                 {{"name", "cls0"}, {"shape", {1, 1, 80, 80}}, {"data_type", 2}},
                 {{"name", "reg1"}, {"shape", {1, 4, 40, 40}}, {"data_type", 2}},
                 {{"name", "cls1"}, {"shape", {1, 1, 40, 40}}, {"data_type", 2}},
                 {{"name", "reg2"}, {"shape", {1, 4, 20, 20}}, {"data_type", 2}},
                 {{"name", "cls2"}, {"shape", {1, 1, 20, 20}}, {"data_type", 2}}}},
               {"params",
                {{"preprocess_mode", "image_to_tensor"},
                 {"output_format", "yolo26_raw"},
                 {"input_size", {640, 640}},
                 {"padding_color", {114, 114, 114}},
                 {"confidence_threshold", 0.25},
                 {"nms_threshold", 0.45},
                 {"top_k", 300},
                 {"reg_max", 1}}}}}}};
}

ModelImportExporter::AscendModelMetadata MakeAscendYolo26Metadata() {
    using Tensor = ModelImportExporter::TensorMetadata;
    ModelImportExporter::AscendModelMetadata metadata;
    metadata.inputs.push_back({"images", {1, 3, 640, 640}, "NCHW", "FP16", "NONE", 0, 0.0F});
    metadata.outputs = {
        Tensor{"reg0", {1, 4, 80, 80}, "NCHW", "FP16", "NONE", 0, 0.0F},
        Tensor{"cls0", {1, 1, 80, 80}, "NCHW", "FP16", "NONE", 0, 0.0F},
        Tensor{"reg1", {1, 4, 40, 40}, "NCHW", "FP16", "NONE", 0, 0.0F},
        Tensor{"cls1", {1, 1, 40, 40}, "NCHW", "FP16", "NONE", 0, 0.0F},
        Tensor{"reg2", {1, 4, 20, 20}, "NCHW", "FP16", "NONE", 0, 0.0F},
        Tensor{"cls2", {1, 1, 20, 20}, "NCHW", "FP16", "NONE", 0, 0.0F},
    };
    return metadata;
}

}  // namespace

TEST_CASE("ModelImportExporter Tests", "[model]") {
    MockServiceRegistry mocks;
    ModelImportExporter importExporter;

    std::string testRoot = "/tmp/cosmo_test_models";
    fs::remove_all(testRoot);
    cosmo::path::OverrideRootPathForTest(testRoot + "/udata", testRoot + "/adata");

    std::string testModelDir             = cosmo::path::GetModelPath();
    std::string testPresetDir            = cosmo::path::GetPresetModelPath();
    std::string testTemplateDir          = cosmo::path::GetModelTemplatePath();
    std::string testUploadDir            = cosmo::path::GetModelUploadTmpDir();
    const std::string longExportModelDir = testModelDir + "/" + std::string(129, 'm');
    fs::create_directories(testModelDir);
    fs::create_directories(testPresetDir);
    fs::create_directories(testTemplateDir);
    fs::create_directories(testUploadDir);

    std::ofstream outTemp(testTemplateDir + "/DET.json");
    outTemp
        << "{\"version\":\"1.0\",\"algorithm_code\":\"\",\"models\":[{\"name\":\"\",\"description\":\"\"}]}";
    outTemp.close();

    std::ofstream outTemp2(testTemplateDir + "/qwen3vl.json");
    outTemp2
        << "{\"version\":\"1.0\",\"algorithm_code\":\"\",\"models\":[{\"name\":\"\",\"description\":\"\"}]}";
    outTemp2.close();

    // Setup helper methods
    importExporter.SetHelpers([&]() { return testModelDir; }, [&]() { return testTemplateDir; },
                              [&](const std::string& code) {
                                  if (code == "1234567")
                                      return testModelDir + "/1234567";
                                  if (code == "test_export_model")
                                      return testModelDir + "/test_export_model";
                                  if (code == "preset_export_model")
                                      return testPresetDir + "/preset_export_model";
                                  if (code == "long_export_model")
                                      return longExportModelDir;
                                  return std::string("");
                              },
                              [&]() { return std::string("1234567"); }, [&](const nlohmann::json&) {},
                              [&](const std::string&, const std::string&) {});

    SECTION("2.1 AddAtomicModel：验证目录创建、config.json 生成、model.nn 创建") {
        std::string bmodelSrc = testUploadDir + "/source.bmodel";
        std::ofstream out(bmodelSrc);
        out << "fake bmodel";
        out.close();

        std::string tokenizerSrc = testUploadDir + "/tokenizer.json";
        std::ofstream outTok(tokenizerSrc);
        outTok << "{}";
        outTok.close();

        std::vector<cosmo::Model::ModelArtifactInfo> modelFiles;
        cosmo::Model::ModelArtifactInfo info;
        info.filePath = bmodelSrc;
        info.role     = "BM1684X";
        modelFiles.push_back(info);

        auto res = importExporter.AddAtomicModel("new_atomic_model", "AtomicName", "qwen3vl", "desc",
                                                 modelFiles, "", tokenizerSrc, "", "", "");
        REQUIRE(res == cosmo::util::ErrorEnum::Success);

        std::string expectedDir =
            testModelDir + "/" + std::string(cosmo::util::kNewDirPrefix) + "1234567_AtomicName_V1.0.0";
        REQUIRE(fs::exists(expectedDir));
        REQUIRE(fs::exists(expectedDir + "/config.json"));
        REQUIRE(fs::exists(expectedDir + "/model" + std::string(cosmo::util::kModelFileExt)));

        nlohmann::json doc;
        REQUIRE(cosmo::util::JsonFileUtil::ReadJsonFile(expectedDir + "/config.json", doc) ==
                cosmo::util::ErrorEnum::Success);
        REQUIRE(doc.contains("algorithm_code"));
        REQUIRE(doc["algorithm_code"].get<std::string>() == "1234567");

        fs::remove_all(expectedDir);
        fs::remove(bmodelSrc);
        fs::remove(tokenizerSrc);
    }

    SECTION("OCR character table is required, validated, and copied with a fixed name") {
        std::string bmodelSrc = testUploadDir + "/ocr_source.bmodel";
        std::ofstream(bmodelSrc).close();
        std::vector<cosmo::Model::ModelArtifactInfo> modelFiles = {{"main", bmodelSrc}};
        std::string resolvedCode;
        std::vector<std::string> modelArtifactPaths;

        auto result = importExporter.ValidateAddModelInputs("", "OcrModel", "ocr", modelFiles, "", "", "",
                                                            resolvedCode, modelArtifactPaths);
        REQUIRE(result == cosmo::util::ErrorEnum::InvalidParam);

        std::string invalidTable = testUploadDir + "/character_table.json";
        std::ofstream(invalidTable) << "[]";
        result = importExporter.ValidateAddModelInputs("", "OcrModel", "ocr", modelFiles, "", "",
                                                       invalidTable, resolvedCode, modelArtifactPaths);
        REQUIRE(result == cosmo::util::ErrorEnum::InvalidParam);

        std::string characterTable = testUploadDir + "/character_table.txt";
        std::ofstream(characterTable) << "blank\n京\n";
        result = importExporter.ValidateAddModelInputs("", "OcrModel", "ocr", modelFiles, "", "",
                                                       characterTable, resolvedCode, modelArtifactPaths);
        REQUIRE(result == cosmo::util::ErrorEnum::Success);

        std::string modelDir = testModelDir + "/ocr_copy_target";
        fs::create_directories(modelDir);
        result = importExporter.CopyAuxiliaryFiles("ocr", "", "", characterTable, modelDir);
        REQUIRE(result == cosmo::util::ErrorEnum::Success);
        REQUIRE(fs::exists(modelDir + "/character_table.txt"));

        fs::remove_all(modelDir);
        fs::remove(bmodelSrc);
        fs::remove(invalidTable);
    }

    SECTION("OCR character table config binds the legacy 6624-entry table to 6625 CTC classes") {
        const std::string characterTable = testUploadDir + "/legacy_character_table.txt";
        {
            std::ofstream table(characterTable);
            table << "blank\n";
            for (int index = 1; index < 6624; ++index)
                table << "token_" << index << '\n';
        }

        cosmo::BmodelInfo info;
        info.valid = true;
        cosmo::BmodelNetworkInfo network;
        network.outputs.push_back({"logits", {1, 40, 6625}, 0});
        info.networks.push_back(network);
        nlohmann::json config = {{"models", {{{"params", nlohmann::json::object()}}}}};

        auto result = importExporter.ConfigureOcrCharacterTable(config, {info}, characterTable);
        REQUIRE(result == cosmo::util::ErrorEnum::Success);
        const auto& params = config["models"][0]["params"];
        REQUIRE(params["character_table_file"].get<std::string>() == "character_table.txt");
        REQUIRE(params["ctc_blank_index"].get<int>() == 0);
        REQUIRE(params["ctc_class_count"].get<int>() == 6625);
        REQUIRE(params["ctc_prepend_tokens"].empty());
        REQUIRE(params["ctc_append_tokens"] == nlohmann::json::array({" "}));

        fs::remove(characterTable);
    }

    SECTION("PP-OCR character table prepends blank and appends space for 6625 classes") {
        const std::string characterTable = testUploadDir + "/ppocr_character_table.txt";
        {
            std::ofstream table(characterTable);
            for (int index = 0; index < 6623; ++index)
                table << "token_" << index << '\n';
        }

        cosmo::BmodelInfo info;
        info.valid = true;
        cosmo::BmodelNetworkInfo network;
        network.outputs.push_back({"logits", {1, 40, 6625}, 0});
        info.networks.push_back(network);
        nlohmann::json config = {{"models", {{{"params", nlohmann::json::object()}}}}};

        auto result = importExporter.ConfigureOcrCharacterTable(config, {info}, characterTable);
        REQUIRE(result == cosmo::util::ErrorEnum::Success);
        const auto& params = config["models"][0]["params"];
        REQUIRE(params["ctc_prepend_tokens"] == nlohmann::json::array({""}));
        REQUIRE(params["ctc_append_tokens"] == nlohmann::json::array({" "}));
        REQUIRE(params["ctc_class_count"].get<int>() == 6625);

        fs::remove(characterTable);
    }

    SECTION("OCR character table config rejects mismatched tables and dynamic class dimensions") {
        const std::string characterTable = testUploadDir + "/character_table_exact.txt";
        std::ofstream(characterTable) << "blank\nA\nB\n";
        nlohmann::json config = {{"models", {{{"params", nlohmann::json::object()}}}}};

        cosmo::BmodelInfo matchingInfo;
        matchingInfo.valid = true;
        cosmo::BmodelNetworkInfo matchingNetwork;
        matchingNetwork.outputs.push_back({"logits", {1, 40, 3}, 0});
        matchingInfo.networks.push_back(matchingNetwork);
        auto result = importExporter.ConfigureOcrCharacterTable(config, {matchingInfo}, characterTable);
        REQUIRE(result == cosmo::util::ErrorEnum::Success);
        REQUIRE(config["models"][0]["params"]["ctc_append_tokens"].empty());

        cosmo::BmodelInfo dynamicInfo;
        dynamicInfo.valid = true;
        cosmo::BmodelNetworkInfo dynamicNetwork;
        dynamicNetwork.outputs.push_back({"logits", {1, 40, -1}, 0});
        dynamicInfo.networks.push_back(dynamicNetwork);
        result = importExporter.ConfigureOcrCharacterTable(config, {dynamicInfo}, characterTable);
        REQUIRE(result == cosmo::util::ErrorEnum::InvalidParam);

        std::ofstream(characterTable, std::ios::trunc) << "blank\n";
        result = importExporter.ConfigureOcrCharacterTable(config, {matchingInfo}, characterTable);
        REQUIRE(result == cosmo::util::ErrorEnum::InvalidParam);

        fs::remove(characterTable);
    }

    SECTION("classify ONNX dynamic batch keeps concrete class count") {
        nlohmann::json doc = {
            {"algorithm_code", ""},
            {"version", ""},
            {"model_type", "classify"},
            {"models",
             {{{"name", ""},
               {"description", ""},
               {"max_batch", 1},
               {"inputs", {{{"name", "input"}, {"data_type", 0}, {"shape", {1, 3, 224, 224}}}}},
               {"outputs", {{{"name", "output"}, {"data_type", 0}, {"shape", {1, 1000}}}}}}}}};

        cosmo::BmodelInfo bmodelInfo;
        bmodelInfo.valid = true;
        cosmo::BmodelNetworkInfo network;
        network.name      = "onnx_network";
        network.max_batch = 1;
        network.inputs.push_back({"input", {-1, 3, 224, 224}, 0});
        network.outputs.push_back({"output", {-1, 2}, 0});
        bmodelInfo.networks.push_back(network);

        importExporter.UpdateTemplateConfig(doc, "4208523", "V1.0.0", "falldown", "classify", "desc",
                                            {bmodelInfo}, false, "0-1", "bgr");

        REQUIRE(doc["models"][0]["outputs"][0]["shape"].size() == 2);
        REQUIRE(doc["models"][0]["outputs"][0]["shape"][0].get<int>() == 1);
        REQUIRE(doc["models"][0]["outputs"][0]["shape"][1].get<int>() == 2);
        REQUIRE(doc["labels"].size() == 2);
        REQUIRE(doc["labels"][0]["name"].get<std::string>() == "category0");
        REQUIRE(doc["labels"][1]["name"].get<std::string>() == "category1");
    }

    SECTION("yolov5 ONNX raw head outputs expand single-output template") {
        nlohmann::json doc = {
            {"algorithm_code", ""},
            {"version", ""},
            {"model_type", "yolov5_det"},
            {"models",
             {{{"name", ""},
               {"description", ""},
               {"max_batch", 1},
               {"inputs", {{{"name", "images"}, {"data_type", 0}, {"shape", {1, 3, 640, 640}}}}},
               {"outputs", {{{"name", "output0"}, {"data_type", 0}, {"shape", {1, 25200, 85}}}}},
               {"params",
                {{"input_size", {640, 640}},
                 {"confidence_threshold", 0.1},
                 {"nms_threshold", 0.35},
                 {"top_k", 1000}}}}}}};

        cosmo::BmodelInfo bmodelInfo;
        bmodelInfo.valid = true;
        cosmo::BmodelNetworkInfo network;
        network.name      = "onnx_network";
        network.max_batch = 1;
        network.inputs.push_back({"images", {1, 3, 640, 640}, 0});
        network.outputs.push_back({"head_small", {1, 3, 80, 80, 6}, 0});
        network.outputs.push_back({"head_medium", {1, 3, 40, 40, 6}, 0});
        network.outputs.push_back({"head_large", {1, 3, 20, 20, 6}, 0});
        bmodelInfo.networks.push_back(network);

        importExporter.UpdateTemplateConfig(doc, "4208523", "V1.0.0", "pedestrian", "yolov5_det", "desc",
                                            {bmodelInfo}, false, "0-1", "rgb");

        const auto& model = doc["models"][0];
        REQUIRE(model["outputs"].size() == 3);
        REQUIRE(model["outputs"][0]["name"].get<std::string>() == "head_small");
        REQUIRE(model["outputs"][1]["shape"][2].get<int>() == 40);
        REQUIRE(model["outputs"][2]["shape"][3].get<int>() == 20);
        REQUIRE(model["params"]["use_npu_postprocess"].get<bool>());
        REQUIRE(model["params"]["anchors"].size() == 3);
        REQUIRE(model["params"]["stride"].size() == 3);
        REQUIRE(doc["labels"][0]["threshold"][0].get<double>() == 0.25);
        REQUIRE(doc["labels"][0]["threshold"][1].get<double>() == 0.25);
    }

    SECTION("yolov5 ONNX dynamic single output does not inherit template output shape") {
        nlohmann::json doc = {
            {"algorithm_code", ""},
            {"version", ""},
            {"model_type", "yolov5_det"},
            {"models",
             {{{"name", ""},
               {"description", ""},
               {"max_batch", 1},
               {"inputs", {{{"name", "images"}, {"data_type", 0}, {"shape", {1, 3, 640, 640}}}}},
               {"outputs", {{{"name", "output0"}, {"data_type", 0}, {"shape", {1, 25200, 85}}}}},
               {"params",
                {{"input_size", {640, 640}},
                 {"confidence_threshold", 0.1},
                 {"nms_threshold", 0.35},
                 {"top_k", 1000}}}}}}};

        cosmo::BmodelInfo bmodelInfo;
        bmodelInfo.valid = true;
        cosmo::BmodelNetworkInfo network;
        network.name      = "onnx_network";
        network.max_batch = 1;
        network.inputs.push_back({"images", {-1, 3, 512, 512}, 0});
        network.outputs.push_back({"output", {-1, -1, -1}, 0});
        bmodelInfo.networks.push_back(network);

        importExporter.UpdateTemplateConfig(doc, "4208523", "V1.0.0", "face", "yolov5_det", "desc",
                                            {bmodelInfo}, false, "0-1", "rgb");

        const auto& model = doc["models"][0];
        REQUIRE(model["inputs"][0]["shape"][0].get<int>() == 1);
        REQUIRE(model["inputs"][0]["shape"][2].get<int>() == 512);
        REQUIRE(model["params"]["input_size"][0].get<int>() == 512);
        REQUIRE(model["outputs"][0]["shape"][0].get<int>() == 1);
        REQUIRE(model["outputs"][0]["shape"][1].get<int>() == -1);
        REQUIRE(model["outputs"][0]["shape"][2].get<int>() == -1);
    }

    SECTION("2.2 ImportModel（tar.gz）：验证解压、目录搬迁") {
        // We will create a fake tar.gz. Wait, without a real tar.gz we can't test tar.gz extraction easily
        // unless we use (void)!system("tar -czf ...").
        // Let's create a real tar.gz using system.
        std::string tempArchiveDir = "/tmp/cosmo_test_archive_dir";
        fs::create_directories(tempArchiveDir + "/fake_model");
        std::string modelFile = "model" + std::string(cosmo::util::kModelFileExt);
        std::ofstream outNn(tempArchiveDir + "/fake_model/" + modelFile);
        outNn << "fake";
        outNn.close();
        std::ofstream outLegacy(tempArchiveDir + "/fake_model/ignored.legacy");
        outLegacy << "legacy";
        outLegacy.close();
        {
            std::ofstream outCfg(tempArchiveDir + "/fake_model/config.json", std::ios::trunc);
            outCfg << "{\"algorithm_code\": \"1234567\", \"chip_type\": \"" << cosmo::util::kEngineType
                   << "\", \"version\": \"V1.0.0\", \"models\": [{\"name\": \"fake_model\", \"file_name\": \""
                   << modelFile << "\"}]}";
        }
        std::string tarFile = testUploadDir + "/test_import.tar.gz";
        std::string cmd     = "cd " + tempArchiveDir + " && tar -czf " + tarFile + " fake_model";
        (void)!system(cmd.c_str());

        auto res = importExporter.ImportModel(tarFile);
        REQUIRE(res == cosmo::util::ErrorEnum::Success);
        REQUIRE(fs::exists(testModelDir + "/fake_model/config.json"));

        fs::remove_all(tempArchiveDir);
        fs::remove_all(testModelDir + "/fake_model");
        fs::remove(tarFile);
    }

    SECTION("2.3 ImportModel（flat 结构）：验证自动包装为标准目录") {
        // Create a flat tar.gz where config.json is at the root of the archive
        std::string tempArchiveDir = "/tmp/cosmo_test_flat_archive_dir";
        fs::create_directories(tempArchiveDir);
        std::string modelFile = "model" + std::string(cosmo::util::kModelFileExt);
        std::ofstream outNn(tempArchiveDir + "/" + modelFile);
        outNn << "fake";
        outNn.close();
        {
            std::ofstream outCfg(tempArchiveDir + "/config.json", std::ios::trunc);
            outCfg << "{\"algorithm_code\": \"7654321\", \"chip_type\": \"" << cosmo::util::kEngineType
                   << "\", \"version\": \"V1.0.0\", \"models\": [{\"name\": \"FlatModel\", \"file_name\": \""
                   << modelFile << "\"}]}";
        }
        std::string tarFile = testUploadDir + "/flat_import.tar.gz";
        std::string cmd = "cd " + tempArchiveDir + " && tar -czf " + tarFile + " config.json " + modelFile;
        (void)!system(cmd.c_str());

        auto res = importExporter.ImportModel(tarFile);
        REQUIRE(res == cosmo::util::ErrorEnum::Success);

        REQUIRE(fs::exists(testModelDir + "/" + std::string(cosmo::util::kNewDirPrefix) +
                           "7654321_FlatModel_V1.0.0/config.json"));

        fs::remove_all(tempArchiveDir);
        fs::remove_all(testModelDir + "/" + std::string(cosmo::util::kNewDirPrefix) +
                       "7654321_FlatModel_V1.0.0");
        fs::remove(tarFile);
    }

    SECTION("ImportModel rejects invalid model file_name mappings") {
        struct CaseSpec {
            std::string label;
            nlohmann::json config;
            std::vector<std::pair<std::string, std::string>> files;
        };

        const std::string mismatchFile =
#ifdef COSMO_NN_USE_CPU_BACKEND
            "model.bmodel";
#else
            "model.onnx";
#endif

        std::vector<CaseSpec> cases = {
            {"missing file_name target",
             {{"algorithm_code", "1111111"},
              {"chip_type", cosmo::util::kEngineType},
              {"version", "V1.0.0"},
              {"models",
               {{{"name", "Broken"}, {"file_name", "missing" + std::string(cosmo::util::kModelFileExt)}}}}},
             {}},
            {"escaping file_name",
             {{"algorithm_code", "1111112"},
              {"chip_type", cosmo::util::kEngineType},
              {"version", "V1.0.0"},
              {"models",
               {{{"name", "Broken"}, {"file_name", "../escape" + std::string(cosmo::util::kModelFileExt)}}}}},
             {{"model" + std::string(cosmo::util::kModelFileExt), "fake"}}},
            {"duplicate file_name",
             {{"algorithm_code", "1111113"},
              {"chip_type", cosmo::util::kEngineType},
              {"version", "V1.0.0"},
              {"models",
               {{{"name", "Net0"}, {"file_name", "shared" + std::string(cosmo::util::kModelFileExt)}},
                {{"name", "Net1"}, {"file_name", "shared" + std::string(cosmo::util::kModelFileExt)}}}}},
             {{"shared" + std::string(cosmo::util::kModelFileExt), "fake"}}},
            {"mismatched platform artifact",
             {{"algorithm_code", "1111114"},
              {"chip_type", cosmo::util::kEngineType},
              {"version", "V1.0.0"},
              {"models", {{{"name", "Broken"}, {"file_name", mismatchFile}}}}},
             {{mismatchFile, "fake"}}},
        };

        for (size_t index = 0; index < cases.size(); ++index) {
            const auto& spec = cases[index];
            INFO(spec.label);
            const std::string tempArchiveDir = "/tmp/cosmo_test_invalid_import_" + std::to_string(index);
            fs::remove_all(tempArchiveDir);
            fs::create_directories(tempArchiveDir + "/broken_model");
            {
                std::ofstream outCfg(tempArchiveDir + "/broken_model/config.json");
                outCfg << spec.config.dump();
            }
            for (const auto& file : spec.files) {
                std::ofstream out(tempArchiveDir + "/broken_model/" + file.first);
                out << file.second;
            }

            const std::string tarFile = testUploadDir + "/invalid_" + std::to_string(index) + ".tar.gz";
            const std::string cmd     = "cd " + tempArchiveDir + " && tar -czf " + tarFile + " broken_model";
            (void)!system(cmd.c_str());

            REQUIRE(importExporter.ImportModel(tarFile) == cosmo::util::ErrorEnum::InvalidParam);
            REQUIRE_FALSE(fs::exists(testModelDir + "/broken_model"));

            fs::remove_all(tempArchiveDir);
            fs::remove(tarFile);
        }
    }

    SECTION("RK3588 YOLO26 package validation accepts the supported contract and rejects mismatches") {
        const fs::path package_dir = fs::path(testRoot) / "rk3588_yolo26_package";
        fs::remove_all(package_dir);
        fs::create_directories(package_dir);
        std::ofstream(package_dir / "model.rknn") << "fake-rknn";

        struct CaseSpec {
            std::string label;
            std::function<void(nlohmann::json&, ModelImportExporter::RknnModelMetadata&, std::string&)>
                mutate;
            std::string expected_stage;
        };

        const std::vector<CaseSpec> cases = {
            {"valid metadata fixture",
             [](nlohmann::json&, ModelImportExporter::RknnModelMetadata&, std::string&) {}, ""},
            {"unsupported model type",
             [](nlohmann::json& config, ModelImportExporter::RknnModelMetadata&, std::string&) {
                 config["model_type"] = "yolov8_det";
             },
             "stage=config"},
            {"missing explicit artifact",
             [](nlohmann::json& config, ModelImportExporter::RknnModelMetadata&, std::string&) {
                 config["models"][0]["file_name"] = "";
             },
             "stage=artifact"},
            {"mixed platform artifacts",
             [&package_dir](nlohmann::json&, ModelImportExporter::RknnModelMetadata&, std::string&) {
                 std::ofstream(package_dir / "extra.onnx") << "wrong-platform";
             },
             "stage=artifact"},
            {"NHWC input mismatch",
             [](nlohmann::json&, ModelImportExporter::RknnModelMetadata& metadata, std::string&) {
                 metadata.inputs[0].format = "NCHW";
             },
             "stage=input"},
            {"output tensor mismatch",
             [](nlohmann::json&, ModelImportExporter::RknnModelMetadata& metadata, std::string&) {
                 metadata.outputs[3].scale = 0.2F;
             },
             "stage=output[3]"},
            {"reg_max mismatch",
             [](nlohmann::json& config, ModelImportExporter::RknnModelMetadata&, std::string&) {
                 config["models"][0]["params"]["reg_max"] = 2;
             },
             "stage=config"},
            {"runtime loader failure includes return code",
             [](nlohmann::json&, ModelImportExporter::RknnModelMetadata&, std::string& loader_error) {
                 loader_error = "rknn_query ret=-7";
             },
             "stage=rknn"},
        };

        for (const auto& spec : cases) {
            INFO(spec.label);
            fs::remove(package_dir / "extra.onnx");

            auto config   = MakeRk3588Yolo26Config();
            auto metadata = MakeRknnYolo26Metadata();
            std::string loader_error;
            spec.mutate(config, metadata, loader_error);

            std::ofstream(package_dir / "config.json") << config.dump();
            importExporter.SetRknnMetadataLoaderForTest(
                [metadata, loader_error](const std::string&, ModelImportExporter::RknnModelMetadata& out,
                                         std::string& error) mutable {
                    if (!loader_error.empty()) {
                        error = loader_error;
                        return false;
                    }
                    out = metadata;
                    return true;
                });

            std::string error;
            const bool valid = importExporter.ValidateModelPackageContract(
                (package_dir / "config.json").string(), package_dir.string(), error);
            if (spec.expected_stage.empty()) {
                REQUIRE(valid);
                REQUIRE(error.empty());
            } else {
                REQUIRE_FALSE(valid);
                REQUIRE(error.find(spec.expected_stage) != std::string::npos);
            }
        }

        fs::remove_all(package_dir);
    }

    SECTION(
        "ASCEND310P3 YOLO26 package validation accepts the fixed-shape FP16 contract and rejects "
        "mismatches") {
        const fs::path package_dir = fs::path(testRoot) / "ascend310p3_yolo26_package";
        fs::remove_all(package_dir);
        fs::create_directories(package_dir);
        std::ofstream(package_dir / "model.om") << "fake-om";

        struct CaseSpec {
            std::string label;
            std::function<void(nlohmann::json&, ModelImportExporter::AscendModelMetadata&, std::string&)>
                mutate;
            std::string expected_stage;
        };

        const std::vector<CaseSpec> cases = {
            {"valid metadata fixture",
             [](nlohmann::json&, ModelImportExporter::AscendModelMetadata&, std::string&) {}, ""},
            {"unsupported model type",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["model_type"] = "yolov8_det";
             },
             "stage=config"},
            {"missing explicit artifact",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["file_name"] = "";
             },
             "stage=artifact"},
            {"wrong artifact extension",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["file_name"] = "model.rknn";
             },
             "stage=artifact"},
            {"mixed platform artifacts",
             [&package_dir](nlohmann::json&, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 std::ofstream(package_dir / "extra.onnx") << "wrong-platform";
             },
             "stage=artifact"},
            {"bad input dtype",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["inputs"][0]["data_type"] = 5;
             },
             "stage=config"},
            {"bad input shape",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["inputs"][0]["shape"] = {1, 640, 640, 3};
             },
             "stage=config"},
            {"bad output count",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["outputs"].erase(config["models"][0]["outputs"].begin() + 5);
             },
             "stage=config"},
            {"misordered output names",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["outputs"][1]["name"] = "reg1";
             },
             "stage=config"},
            {"bad output shape",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["outputs"][2]["shape"] = {1, 4, 80, 80};
             },
             "stage=config"},
            {"bad output dtype",
             [](nlohmann::json& config, ModelImportExporter::AscendModelMetadata&, std::string&) {
                 config["models"][0]["outputs"][0]["data_type"] = 5;
             },
             "stage=config"},
            {"runtime input mismatch",
             [](nlohmann::json&, ModelImportExporter::AscendModelMetadata& metadata, std::string&) {
                 metadata.inputs[0].type = "INT8";
             },
             "stage=input"},
            {"runtime output mismatch",
             [](nlohmann::json&, ModelImportExporter::AscendModelMetadata& metadata, std::string&) {
                 metadata.outputs[4].dims = {1, 4, 80, 80};
             },
             "stage=output[4]"},
            {"runtime output name mismatch",
             [](nlohmann::json&, ModelImportExporter::AscendModelMetadata& metadata, std::string&) {
                 metadata.outputs[1].name = "reg1";
             },
             "stage=output[1]"},
            {"runtime loader failure includes stage",
             [](nlohmann::json&, ModelImportExporter::AscendModelMetadata&, std::string& loader_error) {
                 loader_error = "aclmdlLoadFromFile ret=-1";
             },
             "stage=ascend"},
        };

        for (const auto& spec : cases) {
            INFO(spec.label);
            fs::remove(package_dir / "extra.onnx");

            auto config   = MakeAscend310P3Yolo26Config();
            auto metadata = MakeAscendYolo26Metadata();
            std::string loader_error;
            spec.mutate(config, metadata, loader_error);

            std::ofstream(package_dir / "config.json") << config.dump();
            importExporter.SetAscendMetadataLoaderForTest(
                [metadata, loader_error](const std::string&, ModelImportExporter::AscendModelMetadata& out,
                                         std::string& error) mutable {
                    if (!loader_error.empty()) {
                        error = loader_error;
                        return false;
                    }
                    out = metadata;
                    return true;
                });

            std::string error;
            const bool valid = importExporter.ValidateModelPackageContract(
                (package_dir / "config.json").string(), package_dir.string(), error);
            if (spec.expected_stage.empty()) {
                REQUIRE(valid);
                REQUIRE(error.empty());
            } else {
                REQUIRE_FALSE(valid);
                REQUIRE(error.find(spec.expected_stage) != std::string::npos);
            }
        }

        fs::remove_all(package_dir);
    }

    SECTION("import validation rejects chip types outside the compiled platform profile") {
        const fs::path package_dir = fs::path(testRoot) / "wrong_chip_package";
        fs::remove_all(package_dir);
        fs::create_directories(package_dir);
        std::ofstream(package_dir / "model.om") << "fake-om";

        auto config         = MakeAscend310P3Yolo26Config();
        config["chip_type"] = "UNKNOWN_CHIP";
        std::ofstream(package_dir / "config.json") << config.dump();

        std::string alg_code;
        std::string error;
        REQUIRE_FALSE(importExporter.ValidateImportedModelPackage(package_dir.string(), alg_code, error));
        REQUIRE(error.find("stage=platform") != std::string::npos);
        REQUIRE(error.find("unsupported chip_type=UNKNOWN_CHIP") != std::string::npos);

        fs::remove_all(package_dir);
    }

    SECTION("2.4 ExportModelConfig：验证 tar.gz 创建") {
        std::string exportModelDir = testModelDir + "/test_export_model";
        fs::create_directories(exportModelDir);
        std::ofstream outCfg(exportModelDir + "/config.json");
        outCfg << "{\"modelCode\": \"test_export_model\", \"algorithmVersion\": \"1.0.0\"}";
        outCfg.close();

        std::string outPath;
        std::string outName;
        auto res = importExporter.ExportModelConfig("test_export_model", "export_name", outPath, outName);

        REQUIRE(res == cosmo::util::ErrorEnum::Success);
        REQUIRE(fs::exists(outPath));
        REQUIRE(outName.find("test_export_model") != std::string::npos);
        const auto permissions = fs::status(outPath).permissions();
        REQUIRE((permissions & (fs::perms::group_all | fs::perms::others_all)) == fs::perms::none);

        fs::remove_all(exportModelDir);
        fs::remove(outPath);
    }

    SECTION("ExportModelConfig rejects preset model directories") {
        const std::string presetModelDir = testPresetDir + "/preset_export_model";
        fs::create_directories(presetModelDir);
        std::ofstream(presetModelDir + "/config.json") << "{}";

        std::string outPath;
        std::string outName;
        REQUIRE(importExporter.ExportModelConfig("preset_export_model", "preset", outPath, outName) ==
                cosmo::util::ErrorEnum::DefaultCantBeExport);
        REQUIRE(outPath.empty());
        REQUIRE(outName.empty());
        REQUIRE(fs::is_regular_file(presetModelDir + "/config.json"));
    }

    SECTION("ExportModelConfig accepts managed model directory names up to the importer limit") {
        fs::create_directories(longExportModelDir);
        std::ofstream(longExportModelDir + "/config.json") << "{}";

        std::string outPath;
        std::string outName;
        REQUIRE(importExporter.ExportModelConfig("long_export_model", "long", outPath, outName) ==
                cosmo::util::ErrorEnum::Success);
        REQUIRE(fs::is_regular_file(outPath));
        fs::remove(outPath);
    }

    SECTION("ExportModelConfig rejects unsafe codes and symlink model directories") {
        std::string outPath;
        std::string outName;
        REQUIRE(importExporter.ExportModelConfig("../escape", "export_name", outPath, outName) ==
                cosmo::util::ErrorEnum::InvalidParam);
        REQUIRE(outPath.empty());

        const std::string outsideDir = testRoot + "/outside_model";
        fs::create_directories(outsideDir);
        std::ofstream(outsideDir + "/config.json") << "{}";
        const std::string linkedDir = testModelDir + "/test_export_model";
        fs::create_directory_symlink(outsideDir, linkedDir);

        REQUIRE(importExporter.ExportModelConfig("test_export_model", "export_name", outPath, outName) ==
                cosmo::util::ErrorEnum::InvalidParam);
        REQUIRE(outPath.empty());
        fs::remove(linkedDir);
        fs::remove_all(outsideDir);
    }

    SECTION("managed upload boundary never deletes an external model component") {
        const std::string outsideFile = testRoot + "/outside.bmodel";
        std::ofstream(outsideFile) << "must survive";
        std::vector<cosmo::Model::ModelArtifactInfo> files = {{"main", outsideFile}};

        auto result = importExporter.AddAtomicModel("", "OutsideModel", "DET", "", files, "", "", "", "", "");
        REQUIRE(result == cosmo::util::ErrorEnum::FileNotExist);
        REQUIRE(fs::exists(outsideFile));

        result = importExporter.ImportModel(outsideFile);
        REQUIRE(result == cosmo::util::ErrorEnum::FileNotExist);
        REQUIRE(fs::exists(outsideFile));
        fs::remove(outsideFile);
    }

    SECTION("flat model import rejects path components from config") {
        const std::string archiveDir = testRoot + "/unsafe_flat_archive";
        fs::create_directories(archiveDir);
        std::ofstream(archiveDir + "/config.json")
            << "{\"algorithm_code\":\"../../escape\",\"chip_type\":\"" << cosmo::util::kEngineType
            << "\",\"version\":\"V1.0.0\",\"models\":[{\"name\":\"safe\",\"file_name\":\"model"
            << std::string(cosmo::util::kModelFileExt) << "\"}]}";
        std::ofstream(archiveDir + "/model" + std::string(cosmo::util::kModelFileExt)) << "fake";

        const std::string archivePath = testUploadDir + "/unsafe_flat.tar.gz";
        const std::string command = "tar -czf " + archivePath + " -C " + archiveDir + " config.json model" +
                                    std::string(cosmo::util::kModelFileExt);
        REQUIRE(system(command.c_str()) == 0);

        REQUIRE(importExporter.ImportModel(archivePath) == cosmo::util::ErrorEnum::InvalidParam);
        REQUIRE_FALSE(fs::exists(testRoot + "/escape"));
        REQUIRE(fs::is_empty(testModelDir));
        REQUIRE(fs::is_regular_file(archivePath));

        fs::remove_all(archiveDir);
        fs::remove(archivePath);
    }

    SECTION("model import rejects duplicate names after dot-prefix normalization") {
        const fs::path archive_dir = fs::path(testRoot) / "duplicate_member_archive";
        fs::create_directories(archive_dir);
        std::ofstream(archive_dir / "config.json") << R"({"algorithm_code":"1234567"})";
        std::ofstream(archive_dir / "model.bmodel") << "fake";

        const fs::path archive_path = fs::path(testUploadDir) / "duplicate_member.tar.gz";
        std::string output;
        REQUIRE(cosmo::util::Exec({"tar", "--hard-dereference", "-czf", archive_path.string(), "-C",
                                   archive_dir.string(), "./config.json", "config.json", "model.bmodel"},
                                  output) == 0);

        REQUIRE(importExporter.ImportModel(archive_path.string()) == cosmo::util::ErrorEnum::InvalidParam);
        REQUIRE(fs::is_regular_file(archive_path));

        fs::remove_all(archive_dir);
        fs::remove(archive_path);
    }

    fs::remove_all(testRoot);
}

#ifdef COSMO_NN_USE_CPU_BACKEND
TEST_CASE("CPU classify crop preprocess keeps normalize shape non-zero", "[cwnn][cpu]") {
    cosmo::nn::SharedResource shared;

    cosmo::nn::CropResize crop_op("crop_resize");
    crop_op.type          = "crop";
    crop_op.h_top_crop    = {0.0f};
    crop_op.h_bottom_crop = {0.0f};
    crop_op.w_left_crop   = {0.0f};
    crop_op.w_right_crop  = {0.0f};
    crop_op.dsize         = {224, 224};
    crop_op.color         = {114, 114, 114};

    cosmo::nn::CpuCropResizeNode crop;
    crop.SetMaxBatch(1);
    crop.SetSharedResource(&shared);
    crop.LoadParam(&crop_op);
    REQUIRE(bool(crop.InferTopShapes()));
    REQUIRE(shared.net_input_h == 224);
    REQUIRE(shared.net_input_w == 224);

    cosmo::nn::Normalize normalize_op("normalize");
    normalize_op.mean  = {0.0f, 0.0f, 0.0f};
    normalize_op.scale = 0.00392157f;

    cosmo::nn::CpuNormalizeNode normalize;
    normalize.SetMaxBatch(1);
    normalize.SetSharedResource(&shared);
    normalize.LoadParam(&normalize_op);
    REQUIRE(bool(normalize.InferTopShapesWithBottoms({{1, 224, 224, 3}}, {cosmo::nn::DATA_TYPE_UINT8})));

    auto top_shapes = normalize.GetTopBlobShapes();
    const cosmo::nn::DimsVector expected_shape{1, 3, 224, 224};
    REQUIRE(top_shapes.size() == 1);
    REQUIRE(top_shapes[0] == expected_shape);
}
#endif
