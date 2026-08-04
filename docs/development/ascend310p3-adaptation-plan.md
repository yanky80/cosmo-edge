---
title: 华为昇腾 310P3 适配方案
description: 在 RK3588 适配分支基础上增加 AscendCL、DVPP、OM 模型和硬解码链路的实施方案。
---

# 华为昇腾 310P3 适配方案

## 目标与首期边界

在现有“同一源码、按目标平台单独编译”的架构上增加第四个平台：

| 项目 | 首期选择 |
| --- | --- |
| 目标平台 | x86_64 宿主 + 华为昇腾 310P3 PCIe 算力卡 |
| 构建 Profile | `COSMO_TARGET_PLATFORM=ascend310p3` |
| 推理后端 | CANN AscendCL |
| 模型制品 | 单个 `.om` |
| 媒体后端 | 测试机已有的定制 FFmpeg 硬解码 |
| 图像处理 | DVPP resize、letterbox 和颜色处理 |
| 首期算法 | YOLO26 检测 |
| 模型输出 | 六路 NCHW FP16 raw heads |
| 后处理 | CPU 解码和 class-aware NMS |
| 并发 | device 0，验证 1 个和 3 个 detector 实例 |
| 发布方式 | 测试机原生安装包 |

不建设运行时插件系统，不要求同一进程同时使用多个厂商后端，不在首期支持多卡、
动态 batch、量化、NPU 后处理、Docker 或其他算法。

## 现有能力复用

RK3588 适配已经完成了后端扩展所需的公共改造，310P3 应直接复用：

- `COSMO_TARGET_PLATFORM` 静态平台 Profile。
- 模型包、`modelFiles` 和 `models[].file_name` 的通用模型制品语义。
- Graph 的 `NetNode` 输入绑定、Blob 所有权和 host/device 复制边界。
- `FrameSurface` 的 host、device、DMA 帧及共享生命周期表达。
- YOLO26 六路 raw-head CPU 后处理、坐标恢复和任务层逻辑。
- 现有 demux、RTSP 重连、任务调度、预览、告警和性能统计。

310P3 不复用 RKNN、MPP、DRM PRIME 或 RGA 实现。新增代码限定在平台构建配置、
Ascend 推理后端、Ascend 媒体工厂、模型校验和必要的通用 FP16 后处理扩展中。

## 目标数据链路

首期按两个阶段交付，两个阶段使用同一模型和任务配置。

### 阶段一：正确性基线

```text
H.264/H.265 packet
    -> 定制 FFmpeg 硬解码
    -> host NV12 或可下载的硬件帧
    -> H2D（如需要）
    -> DVPP resize / centered letterbox / color processing
    -> OM input buffer
    -> AscendCL inference
    -> six FP16 raw heads copied to host
    -> CPU YOLO26 decode / class-aware NMS
```

该阶段允许一个明确记录的 host/device 搬运，但不允许静默回退到 FFmpeg 软件解码或
CPU resize。先建立可复现的正确性和性能基线。

### 阶段二：设备帧直通

确认定制 FFmpeg 的硬件像素格式、`AVFrame` buffer 和设备地址契约后：

- 用 `FrameSurface::Device` 包装硬件帧地址、plane、pitch、size 和生命周期。
- 通过持有 `AVFrame` 引用保证设备帧在 DVPP 完成前有效。
- DVPP 直接消费硬件解码帧，并将结果写入模型输入 buffer。
- 只有预览、抓图、OSD 等 host 消费者按需下载帧。

若测试机 FFmpeg 不能导出可由 DVPP 消费的设备地址，阶段一保持为可用回退路径；
不得根据猜测解释 `AVFrame::data[]` 中的私有句柄。

## 环境 Gate 0

实施前在 `AGENTS.md` 记录的 310P3 测试机上采集并写入验收记录：

- `uname -a`、CPU 架构和发行版。
- `npu-smi info` 输出的设备型号、驱动和固件版本。
- CANN、AscendCL、DVPP、ATC 的安装路径和版本。
- 编译器、glibc 和 C++ ABI。
- `ffmpeg -version`、硬件解码器名称、硬件像素格式和共享库来源。
- H.264/H.265 解码后 `AVFrame` 的 format、linesize、buffer 类型和地址所有权。

构建和发布基线以测试机的实际版本为准，不首期维护多个 CANN/FFmpeg 兼容分支。

## 构建与平台 Profile

在平台 Profile 中增加 `ascend310p3`，派生以下固定配置：

- 架构为 `x86_64`。
- NN 后端为 Ascend，媒体后端为 Ascend。
- 模型扩展名和主扩展名为 `.om`。
- 模型目录 token、引擎类型和 `chip_type` 为 `ASCEND310P3`。
- 默认资源目录为 `data/resource/aiboxresource_ascend310p3`。
- FFmpeg 使用测试机系统安装，不复制仓库中的 x86 预编译版本。

新增 CMake SDK 检查，从测试机 CANN 环境查找 AscendCL、DVPP 头文件和共享库。
缺少 SDK、库架构错误或平台后端组合非法时在配置阶段失败。仓库不提交 CANN、
驱动、固件或定制 FFmpeg 二进制。

构建脚本只负责选择平台 Profile 和资源目录；CANN 环境初始化继续使用测试机安装包
提供的环境脚本，不在项目中复制一份厂商环境管理逻辑。

## AscendCL 推理后端

新增 `src/nn/device/ascend/`，实现现有抽象所需的最小集合。

### 设备和上下文

- 进程级执行一次 `aclInit`，进程退出时执行一次 `aclFinalize`。
- 每个 Graph 实例拥有独立 ACL context、stream、模型、dataset 和 tensor buffer。
- 使用 `aclrtMalloc`、`aclrtFree`、`aclrtMemcpy` 和 stream 同步实现设备内存边界。
- 资源严格按创建顺序的逆序释放；部分初始化失败也必须安全清理。

首期每个 detector 实例独立加载 OM，以现有 `InstancePool` 控制最多三个实例。
不增加第二层模型 worker pool，也不实现多卡调度。

### `AscendNetNode`

模型加载时：

1. 加载 OM 并取得 `aclmdlDesc`。
2. 校验一个固定 batch 输入和六个 FP16 输出。
3. 读取输入输出名称、shape、dtype 和 buffer size。
4. 一次性创建 dataset 和输入输出 buffer，后续推理复用。

推理时：

1. 将预处理结果绑定或复制到模型输入。
2. 在实例 stream 上执行模型。
3. 同步 stream。
4. 将六路输出复制到 host Blob。
5. 交给通用 YOLO26 raw 后处理节点。

错误日志必须包含失败阶段、device id、模型路径、tensor 描述和底层 ACL 错误码。
设备不可用、模型不匹配或执行失败时任务明确失败，不切换到 ONNX Runtime。

### Node 和 Graph 装配

- 新增 `DEVICE_ASCEND` 和相应字符串表示。
- Ascend Node Creator 为 `NODE_NET` 创建 `AscendNetNode`。
- `image_to_tensor` 创建 Ascend/DVPP 预处理节点。
- 后处理节点继续使用 host 内存。
- 复制节点仍由生产者和消费者的内存能力决定，不在 Graph 中增加平台特判。

## 硬解码和 DVPP

现有 demux 继续负责文件、RTSP、BSF、时间戳和重连。Ascend decoder 只负责：

- 按 Gate 0 结果显式选择测试机 H.264/H.265 硬件解码器。
- 将解码后的 host 或 device NV12 包装为 `VideoFrame`/`FrameSurface`。
- 保留原始 `AVFrame` 生命周期，直到所有消费者完成。
- 拒绝未知像素格式、无效 plane、pitch、size 或设备地址。

Ascend `image_to_tensor` 节点负责：

- 校验输入帧格式和 surface 边界。
- 等比例缩放、居中 padding，padding color 与现有模型配置一致。
- 使用测试机实际支持的 DVPP 输出格式。
- 将归一化和通道顺序固定在 DVPP/AIPP/模型转换契约中，避免运行时重复处理。
- 输出模型输入 buffer，并记录上传、DVPP 和同步耗时。

预览、抓图和 OSD 不属于推理零拷贝保证；需要时可以下载到 host 并复用通用实现。

## OM 模型契约

首期从同一 YOLO26 ONNX 基线生成 FP16 OM：

```text
ATC soc_version: Ascend310P3
input: batch=1, 640x640, fixed shape
outputs: reg0, cls0, reg1, cls1, reg2, cls2
output dtype: FP16
output layout: NCHW
```

模型转换记录必须包含：

- 原始 ONNX SHA256。
- ATC、CANN 版本。
- 完整 ATC 参数和 AIPP 配置。
- 生成 OM SHA256。
- 六路输入输出 metadata。

首期只记录一条可复现 ATC 命令，不建设模型转换框架。仅当多个模型需要稳定批量转换时，
再增加转换脚本。

模型包必须满足：

```json
{
  "chip_type": "ASCEND310P3",
  "model_type": "yolo26_det",
  "models": [
    {
      "file_name": "model.om",
      "max_batch": 1,
      "params": {
        "preprocess_mode": "image_to_tensor",
        "output_format": "yolo26_raw",
        "input_size": [640, 640],
        "padding_color": [114, 114, 114],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "top_k": 300,
        "reg_max": 1
      }
    }
  ]
}
```

包内只允许一个显式 `.om` 制品。导入时通过 AscendCL metadata 校验输入、六路输出、
固定 batch、shape 和 FP16 dtype。

## FP16 raw 后处理

现有 `yolo26_raw` 节点只接受 RK3588 INT8 affine 输出，需要扩展为：

- 保留 INT8 + scale/zero-point 路径，保证 RK3588 无回归。
- 新增 FP16 和 FP32读取路径。
- FP16/FP32 不要求量化参数。
- 两种路径共用候选框生成、sigmoid、class-aware NMS、top-k 和坐标恢复。
- 六路输出必须使用相同 dtype，shape 和类别通道必须一致。

首期不做 W8A8。只有阶段性能数据证明模型执行是主要瓶颈，并且存在代表性标定集时，
才单独评估量化。

## 故障策略

以下情况必须在初始化或运行阶段明确失败：

- CANN、AscendCL、DVPP 或定制 FFmpeg 依赖缺失或 ABI 不匹配。
- 310P3 不可见、context/stream 创建失败。
- 找不到指定硬解码器或得到未知硬件帧格式。
- FrameSurface 地址、plane、pitch、size 或生命周期无效。
- DVPP 不支持目标格式、resize、padding 或目标 buffer。
- OM 不是单输入、固定 batch=1、六路 FP16 raw-head 契约。
- ACL 模型加载、执行、同步或内存复制失败。

不得静默回退软件解码、CPU resize 或 ONNX Runtime。阶段一允许的 host/device 搬运必须
有日志和性能指标，便于确认阶段二是否真正消除。

## 测试与验收

### 本地门禁

- 平台 Profile 覆盖 `x86`、`sophon`、`rk3588` 和 `ascend310p3`。
- 使用假 SDK 验证缺头文件、缺库和非法平台组合会配置失败。
- 模型包测试覆盖合法 OM、错误扩展名、多个制品、错误 chip/model type。
- FP16 raw-head 黄金测试覆盖阈值、三尺度解码、NMS、top-k 和 letterbox 坐标恢复。
- Graph 测试覆盖 Ascend host/device 复制边界和 external-owned Blob 生命周期。
- 现有 x86、Sophon、RK3588 测试全部通过。

### 310P3 测试机

- H.264/H.265 本地文件 smoke test。
- RTSP H.264/H.265 断流、重连和持续运行测试。
- 日志确认硬解码、DVPP 和 AscendCL 均生效。
- 阶段二确认硬解码帧设备地址直接进入 DVPP，不生成中间 host 图像。
- 缺设备、错误 OM、硬解码器缺失和 DVPP 失败的故障测试。

#### Issue #27：FP16 raw-head 真机验证记录

执行时间与主机：2026-08-04，`ssh -p 1022 root@35623rcqc768.vicp.fun`（x86_64 / Ubuntu 22.04 / gcc 11.4，`npu-smi info` 设备 0 正常）。以下构建基线只在本机生效，未入库。

- 分支 worktree rsync 到 `/root/cosmo-edge-issue27`；`prebuild/ffmpeg/x86_64/{include,lib}` 符号链接到 `/opt/ffmpeg-4.4.1/ascend`（Ascend 编译的 FFmpeg 4.4.1），`ldd cosmo-tests` 确认 `libavcodec.so.58`、`libswscale.so.5` 等均来自 `/opt/ffmpeg-4.4.1/ascend/lib`。
- 需 `apt-get install -y automake autoconf libtool`（srs external 构建依赖，已获批准安装）。
- 三个源文件按 FFmpeg 4.4 API 打了临时补丁（`AVInputFormat*`、`FF_PROFILE_*` 宏、`AVCodec*`），仅用于本机构建、未提交；属预存兼容问题，与 Issue #27 无关。
- `cmake --build . --target cosmo-tests -j12` 构建成功。

测试命令与结果（`LD_LIBRARY_PATH` 同 CI：`prebuild/ffmpeg/x86_64/lib`、`3rd/onnxruntime-linux-x64-1.26.0/lib`、`build-cpu/thirdparty_install/{openssl,curl,event,glog,mp4v2,uuid}/lib`）：

- `./cosmo-tests "[yolo26]"` → `All tests passed (120 assertions in 9 test cases)`。覆盖 FP16 黄金（阈值、三尺度、NMS、top-k、letterbox 坐标恢复）、FP32、混合 dtype / 头数量 / shape / 类别通道错误、非 NCHW layout 拒绝、NaN/Inf 跳过，以及 INT8 affine 回归。
- `./cosmo-tests`（全量）→ 888 个用例中 887 个通过、1 个失败：`test_video_frame_proc_nv12.cc` 的 `NV12ToI420` 在 Ascend `libswscale.so.5` 的 `sws_scale` 内 SIGSEGV（该崩溃会让全量跑在随机顺序下提前终止，统计不稳定）。已用最小 standalone 程序在真机复现：NV12→YUV420P 三平面输出即崩；本地仓库自带 prebuild FFmpeg 下同一用例通过。属 Ascend FFmpeg swscale 的预存环境问题，与 Issue #27 改动（仅 `yolo26_raw_decode_node`、`detection_pipeline` 与其测试）无关，未在本 issue 修复。

RK3588 板回归：未执行。原 INT8 affine 路径由本地 `[yolo26]` INT8 用例覆盖且全绿；板端 SDK 缺少可用的 `librknnrt` sysroot，完整板端构建成本高，留给 RK3588 专项验证。

用固定视频比较前 100 个有效帧与 ONNX FP32 基线：

- 检测类别一致。
- 匹配框 IoU 不低于 `0.98`。
- 置信度绝对误差不超过 `0.01`。

分别运行 1 个和 3 个 detector 实例至少 30 分钟，确认无崩溃、无持续内存增长，任务释放
后设备内存回收。记录 decode、传输、DVPP、ACL、后处理、端到端时延/FPS，以及 CPU、
NPU 和内存峰值。首期输出性能报告，但不设置 FPS 阻塞门槛。

所有硬件测试使用 `AGENTS.md` 中的本地锁串行执行，并在 pull request 中记录准确命令和
相关结果。

## 首期不做

- 多卡枚举、任务绑卡和跨卡调度。
- 动态 batch、动态 shape 或多尺寸模型。
- YOLO26 之外的分类、OCR、DINO、SAM、VLM 和大模型。
- W8A8、ModelSlim、NPU NMS 或自定义算子。
- Docker 发布和多 CANN/FFmpeg 版本兼容层。
- 为一个测试环境设计运行时插件或通用厂商抽象。
