---
title: 多算力平台架构调整与 RK3588 适配计划
description: 面向 x86、Sophon 和 RK3588 的推理、媒体、模型制品与构建架构调整方案。
---

# 多算力平台架构调整与 RK3588 适配计划

## 目标

采用“同一源码、按目标平台单独编译”的静态装配架构，支持以下平台：

| 目标平台 | 推理后端 | 模型制品 | 媒体后端 |
| --- | --- | --- | --- |
| x86 | ONNX Runtime | `.onnx` | FFmpeg 软件实现 |
| Sophon | BMRuntime | `.bmodel` / `.nn` | Sophon VPU / BMCV |
| RK3588 | RKNN Runtime | `.rknn` | MPP / DRM PRIME / RGA |

不建设运行时动态插件系统，也不要求同一进程同时使用不同厂商的算力设备。新增第四种后端时，改动应限定在平台构建配置、模型制品实现、`nn/device/<backend>` 和可选媒体实现中，不再修改业务服务或 Graph 主流程。

RK3588 首期只支持 YOLO26 检测，覆盖 H.264/H.265 本地文件和 RTSP：

```text
H.264/H.265 packet
    -> MPP DRM PRIME / NV12
    -> RGA RGB letterbox
    -> RKNN input tensor DMA
    -> RKNN inference
    -> INT8 raw-head decode / class-aware NMS on CPU
```

零拷贝保证范围是 MPP 解码帧到 RKNN 输入。RKNN 输出继续使用 `rknn_outputs_get`，随后在 CPU 执行 YOLO26 后处理。

## 领域边界

以下概念必须分开，避免继续使用“算力卡”同时指代硬件、运行时和模型格式：

| 术语 | 定义 |
| --- | --- |
| 目标平台 | 发布包运行的硬件和操作系统组合，例如 RK3588、Sophon 或 x86 |
| 推理后端 | 执行模型的运行时，例如 RKNN Runtime、BMRuntime 或 ONNX Runtime |
| 媒体后端 | 提供解码、图像处理和编码能力的实现 |
| 模型制品 | 推理后端可以直接加载的模型文件 |
| 模型包 | `config.json`、一个目标平台的模型制品和辅助文件组成的导入单元 |
| Frame Surface | 带内存类型、plane、fd、offset、pitch 和生命周期的图像帧存储描述 |

每个模型包只包含一个目标平台的制品。不同平台可以使用相同的算法编码，但不能在一个模型包中混装 `.onnx`、`.nn` 和 `.rknn`。

## 构建架构

新增单一构建参数：

```cmake
COSMO_TARGET_PLATFORM=x86|sophon|rk3588
```

平台 Profile 统一派生以下配置：

- 目标架构和 toolchain。
- 推理与媒体后端源码。
- 编译宏和链接依赖。
- 模型扩展名、目录 token、引擎类型和支持的 `chip_type`。
- 默认资源目录和发布包内容。

旧的 `COSMO_NN_USE_CPU_BACKEND` 和 `COSMO_NN_USE_SOPHON_BACKEND` 暂时作为兼容输入，使用时输出弃用警告。新旧参数冲突或形成非法组合时，CMake 必须立即失败。

RK3588 的 RKNN Runtime、ffmpeg-rockchip、librga 和 libdrm 由外部 SDK/sysroot 提供。CMake 必须检查头文件、aarch64 共享库和必要的 `pkg-config` 模块，不向仓库提交厂商二进制。

## 模型制品与导入

将业务层中的 Sophon 领域名称泛化：

- `BmodelFileInfo` 改为 `ModelArtifactInfo`。
- 内部参数 `bmodelFiles` 改为 `modelFiles`。
- `BmodelTool` 的模型检查职责改为平台后端实现的通用模型制品检查接口。
- `WriteNnFile` 改为平台后端实现的制品安装操作。

上传 API 使用 `modelFiles`。为兼容现有前端和调用方，请求仍接受旧字段 `bmodelFiles`；两个字段同时出现时返回参数错误。新前端和序列化输出只使用 `modelFiles`。

模型定位不再依赖目录扩展名扫描。新导入模型必须在 `models[].file_name` 中写入文件名；旧 x86/Sophon 模型中 `file_name` 为空时，保留现有扩展名扫描作为兼容回退。

RK3588 YOLO26 配置至少包含：

```json
{
  "chip_type": "RK3588",
  "model_type": "yolo26_det",
  "models": [
    {
      "file_name": "model.rknn",
      "max_batch": 1,
      "params": {
        "preprocess_mode": "image_to_tensor",
        "output_format": "yolo26_raw",
        "input_size": [640, 640],
        "padding_color": [114, 114, 114],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "top_k": 300
      }
    }
  ]
}
```

配置还必须声明 RKNN 模型实际的 6 个 INT8 affine raw outputs。导入时检查一个 NHWC RGB 输入、6 个 NCHW 输出、`reg_max=1`、量化类型、shape、scale 和 zero-point。

新增 `data/resource/aiboxresource_rk3588`，首期只提供 YOLO26 模板和必要的资源。

## 帧与内存模型

当前 `VideoFrame` 主要表达可通过指针访问的内存，不能安全描述 DRM PRIME。新增通用 `FrameSurface`：

- 内存类型为 host、device 或 DMA-BUF。
- 每个 plane 包含 fd、虚拟地址、offset、pitch 和 size。
- 通过共享生命周期对象保证 FFmpeg `AVFrame` 和 DMA fd 在所有消费者结束前有效。
- `VideoFrame` 增加包装外部 surface 的构造路径，现有内存池路径保持不变。

`BlobHandle` 增加 store-owned 和 external-owned 所有权标记。Graph 遇到后端已绑定的输入 tensor 时不得重复分配；销毁时也不得通过通用设备 allocator 释放 RKNN context 拥有的内存。

`BlobDesc` 增加 affine quantization 描述，包括 scale 和 zero-point，使通用后处理节点不依赖 RKNN SDK 类型。

## Graph 与推理后端

为 `NetNode` 增加两个稳定能力：

- 模型加载和 shape 推导完成后，绑定网络输入 blob。
- 声明网络输出位于主机还是设备内存。

Graph 初始化顺序调整为：

```text
构建节点
    -> 加载模型
    -> 推导 tensor 描述
    -> NetNode 绑定输入 tensor
    -> 分配剩余 blob
```

复制节点的插入由生产节点和 NetNode 的内存能力决定，不再通过 CPU/Sophon 条件分支判断。

新增通用 `image_to_tensor` op。RK 实现负责：

- 校验输入是 NV12 DRM PRIME surface。
- 从 DRM descriptor 读取 fd、offset、pitch 和垂直 stride。
- 使用 RKNN `size_with_stride` 创建并绑定输入 tensor memory。
- 将 RKNN tensor fd 作为 RGA 输出。
- 完成 NV12 到 RGB、等比例缩放、居中 padding。
- padding 几何变化时原地初始化，并执行必要的 RKNN memory sync。

`RknnNetNode` 每个实例只创建一个 RKNN context，使用 `RKNN_NPU_CORE_AUTO`。它负责模型加载、输入 tensor 生命周期、`rknn_run`、6 个输出读取和量化描述传播。

新增通用 `yolo26_raw` 主机后处理节点，复用 RK3588 参考项目中的：

- INT8 confidence 阈值量化。
- 三尺度 `reg/cls` 解码。
- `reg_max=1` 校验。
- class-aware NMS。
- top-k 截断。

后处理节点输出统一的 `[batch, top_k, 6]` 浮点检测结果，继续复用现有检测结果解析和原图坐标恢复逻辑。现有单输出 E2E YOLO26 通过默认 `output_format` 保持不变。

## RK 媒体后端

RK decoder 适配现有 packet-based `VideoDecoder`，不复制参考项目中自行打开文件/RTSP 的 demux 主循环：

- 现有 `VideoDemuxer` 继续负责文件、RTSP、BSF 和 packet 时序。
- RK decoder 使用 ffmpeg-rockchip 的 `h264_rkmpp` 和 `hevc_rkmpp`。
- decoder 强制选择 `AV_PIX_FMT_DRM_PRIME`，返回 NV12 `FrameSurface`。
- MPP、DRM PRIME 或格式校验失败时明确返回错误，不回退软件解码。

预览、抓图、OSD 和录像继续可用，但不属于推理零拷贝契约。DMA 帧允许按需转换到主机并复用通用实现；后续只有在 profiling 证明必要时，才增加 RGA 绘制或 MPP 编码优化。

USB/MJPEG 保留现有通用路径，不纳入 RK3588 首期硬件链路验收。

## 并发与资源管理

继续使用现有外层 `InstancePool`，避免和 RKNN 内部 context pool 形成双层并发：

- RK DetectorPool 的 `inst_per_tasks` 设为 `1`。
- RK DetectorPool 的 `max_inst_count` 设为 `3`。
- 每个 Detector/Graph/RknnNetNode 实例拥有一个 RKNN context 和输入 tensor。
- 不调用 `rknn_dup_context`，不实现全设备绑核分配器。
- RKNN Runtime 使用 `RKNN_NPU_CORE_AUTO` 调度三个 NPU core。

该方案会为每个实例重复占用模型内存。首期只有 YOLO26，接受该限制；只有板端内存数据证明不可接受时，才改为每模型共享 context worker。

## 故障策略

以下情况必须使任务初始化或运行明确失败：

- 找不到 RKNN Runtime、RGA、libdrm 或 ffmpeg-rockchip decoder。
- MPP 未返回 DRM PRIME/NV12。
- DRM plane、offset、pitch 或 stride 不符合要求。
- RKNN 输入不是 NHWC RGB，或无法创建、绑定 tensor memory。
- 输出数量、shape、类型、量化方式或 `reg_max` 不匹配。
- RGA 无法导入 MPP/RKNN fd 或执行转换。

错误日志必须包含失败阶段、模型路径、输入/输出 tensor 描述和底层返回码。H.264/H.265 的 RK 硬件链路不得静默回退到主机推理。

## 测试与验收

### 通用回归

- 平台 Profile 解析、旧构建参数兼容和非法组合测试。
- `modelFiles`、`bmodelFiles` 兼容及冲突测试。
- 新旧模型目录和 `models[].file_name` 路径解析测试。
- external-owned Blob 不被重复分配或释放。
- NetNode 输入绑定和主机输出不会插入错误复制节点。
- x86 与 Sophon 构建、模型导入、Graph contract 和现有测试全部通过。

### YOLO26 后处理

移植参考项目的 self-test，为 `yolo26_raw` 建立固定量化 tensor 黄金数据，覆盖：

- confidence 阈值边界。
- scale/zero-point 反量化。
- 三尺度输出。
- class-aware NMS。
- letterbox 坐标恢复。
- 非法输出数量、shape 和量化信息。

### RK3588 板端验证

- H.264 与 H.265 本地文件 smoke test。
- RTSP H.264/H.265 断流、重连和持续运行测试。
- 验证 MPP 输出 DRM PRIME，RGA 目标 fd 来自 RKNN input tensor。
- 验证推理路径不调用 `rknn_inputs_set`，不生成中间 RGB 主机副本。
- 验证预览、抓图、OSD 和录像旁路。
- 模拟缺失 decoder、RGA 设备和 RKNN tensor 绑定失败，确认无软件推理回退。

使用同一块板、同一模型和固定视频，对比独立参考程序前 100 个有效帧：

- 检测类别一致。
- 置信度绝对误差不超过 `1e-3`。
- 匹配框 IoU 不低于 `0.99`。

分别运行 1 个和 3 个 Detector 实例，记录 decode、RGA、RKNN、postprocess、端到端吞吐和内存占用。首期生成性能报告，但不设置 FPS 阻塞门槛。

## 文档与决策记录

实施时新增根目录 `CONTEXT.md`，记录本方案的领域术语，不写实现细节。

新增 ADR，记录以下难以回退的决策：

- 按目标平台静态编译，不使用运行时动态插件。
- 媒体后端和推理后端保持独立。
- 一个模型包只携带一个目标平台的制品。

同步更新架构、构建和部署文档，说明 RK sysroot、运行库、设备节点、模型格式、零拷贝边界和板端诊断方式。

## 首期不做

- 同一进程同时调度多厂商算力设备。
- 运行时动态加载第三方后端。
- 一个模型包携带多后端制品。
- RKNN 输出 tensor memory 绑定。
- NPU 内执行 YOLO26 后处理。
- RKNN 分类、关键点、OCR、SAM2、DINO 或大模型适配。
- USB/MJPEG 硬件加速。
- 预览、OSD 和录像的 RK 专用硬件优化。
