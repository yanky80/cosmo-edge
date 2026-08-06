---
title: Ascend 310P3 Preview 运维
description: 310P3 原生包、部署、环境、诊断与基准流程。
prev:
  text: 构建指南
  link: /guide/build
next:
  text: 故障排查
  link: /guide/troubleshooting
---

# Ascend 310P3 Preview 运维

## 包与外部依赖

用 `-DCOSMO_TARGET_PLATFORM=ascend310p3` 配置，Profile 选择
`data/resource/aiboxresource_ascend310p3`、安装为 `resource/`，只接受 `.om` 模型。
CMake 从外部 CANN 检查 `acl/acl.h`、`acl/dvpp/hi_dvpp.h`、`libascendcl.so`、
`libacl_dvpp*.so`，并从 `/opt/ffmpeg-4.4.1/ascend` 定位定制 FFmpeg（h264_ascend/h265_ascend）；
库 ELF 架构与目标不符时配置阶段失败。CANN、驱动、固件与 FFmpeg 均为外部输入，
任何厂商二进制都不进入仓库或发布包。

## 运行契约

支持路径：H.264/H.265 本地文件或 RTSP → `h264_ascend`/`h265_ascend` 硬解码 → DVPP
`image_to_tensor`（居中 letterbox 到 960x960、NV12→RGB、host 侧 /255+FP16）→ AscendCL
推理（FP16 OM，输入 `[1,3,960,960]`，输出 `[1,300,6]` end2end）→ host `yolo_e2e` 坐标恢复。
解码器默认输出设备面（`AV_PIX_FMT_ASCEND`），推理直接消费设备帧，无 H2D 上传；预览/抓图按需
D2H 下载。无软件解码、无 CPU resize、无 ONNX Runtime 回退；失败即上抛。

模型包为资源目录加 `.om` 条目（`model_template/yolo26_det.json` 模板，
`chip_type: ASCEND310P3`）；导入时用 AscendCL metadata 校验一个 batch-1 输入与
`[1,300,6]` 输出。

## 诊断与排障

- 缺 `h264_ascend`/`h265_ascend`：检查 `/opt/ffmpeg-4.4.1/ascend` 定制 FFmpeg，
  `ffmpeg -decoders | grep ascend`；引擎必须失败，不得选软件解码器。
- 设备检查：`npu-smi info` Health OK、驱动/固件版本、`/usr/local/Ascend/driver/version.info`。
- 库/ELF 问题：配置期报错路径见 `docs/development/ascend310p3-native-package.md`。
- 模型契约：坏 OM / 错误 shape 在 `Graph::Init` 抛错；日志含失败阶段、device id、
  模型路径、tensor 描述与 ACL 错误码。
- 换码流/挂死：VDEC 通道不可在 VPC 存活时 Close/Open，换码流需重启任务。

## 验证与基准

完整验证、100 帧 ONNX FP32 对比、1/3 实例 30 分钟长稳与分阶段基准见
[`docs/benchmarks/ascend310p3-preview/README.md`](../benchmarks/ascend310p3-preview/README)。
所有硬件命令在共享 `flock` 锁下执行，命令与结果记录在 PR 中。报告不设 FPS 门槛。

发布前同时运行不变的 x86/Sophon Profile 配置路径、模型导入、Graph 契约测试与现有单测套件；
见 [`docs/guide/build.md`](build) 与 [`docs/guide/test-cases.md`](test-cases)。
