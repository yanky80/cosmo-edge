---
title: Ascend310P3 原生安装包与运维记录
description: 310P3 原生安装包的内容、构建、安装、环境初始化、设备检查、模型导入、启停与故障诊断记录。
---

# Ascend310P3 原生安装包与运维记录

本记录对应 `docs/development/ascend310p3-adaptation-plan.md` 的「发布方式：测试机原生安装包」。
首期发布包只包含应用与 Ascend 资源集，**依赖而非打包**已批准的 CANN、驱动、固件与定制 FFmpeg
安装（`docs/development/ascend310p3-test-host-baseline.md` 记录基线版本）。

## 安装包内容

`COSMO_TARGET_PLATFORM=ascend310p3` 的 CMake 配置选择
`data/resource/aiboxresource_ascend310p3` 作为资源目录并安装为 `resource/`，其余目录与
x86/sophon 包一致：

| 目录 | 内容 | 来源 |
| --- | --- | --- |
| `bin/` | `cosmo-engine`、`version.txt`、`nginx_conf/` | CMake install |
| `web/` | 前端静态资源 | `web_frontend` |
| `files/` | `data/`（接口文档、模型包等） | CMake install |
| `font/` | 字体 | CMake install |
| `scripts/` | 启停/升级脚本 | CMake install |
| `resource/` | `aiboxresource_ascend310p3`（含 `model_template/yolo26_det.json`） | `-DRESOURCE_DIR=…` |
| `lib/` | 模型 guard 动态库（空目录时自动创建） | CMake install |

包内**不包含**：CANN toolkit、驱动、固件、定制 FFmpeg 二进制。`cmake/ascend_sdk.cmake`
与 `cmake/ffmpeg.cmake` 从外部路径定位 CANN 与 `/opt/ffmpeg-4.4.1/ascend`，ELF 架构校验
在配置阶段失败（库架构与目标不符时）。运行期依赖由测试机已安装的 CANN/FFmpeg 提供。

## 构建与打包

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 导出 ASCEND_TOOLKIT_HOME 等
cmake -S . -B build-ascend \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PWD}/build-ascend/install" \
  -DCOSMO_TARGET_PLATFORM=ascend310p3 \
  -DRESOURCE_DIR="${PWD}/data/resource/aiboxresource_ascend310p3" \
  -DBUILD_TESTS=OFF
cmake --build build-ascend --target install -j"$(nproc)"
cmake --build build-ascend --target package_all    # cpack TGZ + MD5 重命名
```

产物：`build-ascend/packages/cosmo-<version>-<md5>.tar.gz`（`package_md5_rename.sh` 重命名）。
验证包内容与依赖：

```bash
tar -tzf build-ascend/packages/cosmo-*.tar.gz | head
ldd build-ascend/install/bin/cosmo-engine | grep -E "ascend|dvpp|ffmpeg|avcodec"   # 应指向外部 CANN/FFmpeg，不在包内
```

## 安装

```bash
tar -xzf cosmo-<version>-<md5>.tar.gz
cd cosmo-<version>
COSMO_INSTALL_DIR=/opt/cosmo/ascend310p3 ./scripts/install.sh   # 默认 /appfs/cosmo_wander/cwai_data
```

`install.sh` 会先 `stop.sh`，再安装 `bin lib scripts web font files resource` 目录。

## 环境初始化与设备检查

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
uname -a
npu-smi info                     # Health OK、设备 0、驱动 24.1.1.1、固件 7.5.0.5.220
cat /usr/local/Ascend/driver/version.info
/opt/ffmpeg-4.4.1/ascend/bin/ffmpeg -hide_banner -decoders | grep ascend   # h264_ascend/h265_ascend
```

## 模型转换与导入

模型转换（ATC）记录见 `docs/development/ascend310p3-yolo26-om-atc.md`（ONNX→OM，FP16，
`Ascend310P3`，`images:1,3,960,960`，输出 `[1,300,6]`）。导入校验使用 AscendCL metadata
（`src/service/model/impl/ModelImporter.cc`）：一个固定 batch-1 输入、一个输出、
固定 shape `[1,300,6]`；不匹配即拒绝。

模型包结构（模板 `data/resource/aiboxresource_ascend310p3/model_template/yolo26_det.json`）：

```text
model-package/
├── config.json      # chip_type: ASCEND310P3, model_type: yolo26_det, models[0].file_name: model.om
└── model.om
```

通过 Web 界面「添加模型」或导入 API 上传 zip/tar.gz 模型包（与 x86/sophon 相同的
`modelFiles`/`models[].file_name` 语义）。`file_md5` 可留空；导入时按契约校验。

## 启动与关闭

```bash
cd <install>/scripts
./start.sh              # 升级检查后调用 run_start.sh start <log>，拉起 nginx/SRS/cosmo-engine
./stop.sh               # 先 SIGTERM 后 SIGKILL，cosmo-engine/srs/nginx
```

## 故障诊断

- 设备不可用：`npu-smi info` 非 OK、`cat /usr/local/Ascend/driver/version.info`、`dmesg | grep -i ascend`。
- 库缺失/架构错误：配置阶段 `cmake/ascend_sdk.cmake` 报
  `Ascend SDK header not found` / `Ascend library is not an x86_64 ELF`；
  运行期 `ldd <install>/bin/cosmo-engine | grep ascend` 查未解析符号。
- 解码器不可用：`/opt/ffmpeg-4.4.1/ascend/bin/ffmpeg -decoders | grep ascend` 必须列出
  `h264_ascend`/`h265_ascend`；引擎在缺失时明确失败，不静默回退软件解码。
- 模型导入失败：日志中的失败阶段、device id、模型路径、tensor 描述与 ACL 错误码；
  坏 OM / 错误契约在 `Graph::Init` 抛错（无 ONNX Runtime 回退）。
- 解码/推理挂死（310P3 已知约束）：VDEC 通道在 VPC 通道存活时不可 Close/Open，
  换码流需重启任务；见 `docs/development/ascend310p3-adaptation-plan.md`「VDEC/VPC 楔死」。

## 首期边界

多卡、动态 batch/shape、量化、NPU 后处理、Docker 发布与多 CANN/FFmpeg 版本兼容层不在首期
范围；构建与发布基线以测试机实际版本为准。
