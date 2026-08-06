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

- 架构参数化：`COSMO_TARGET_ARCH` 合法集 `{x86_64, aarch64}`，默认 `x86_64`；
  aarch64 时 x86_64 主机走现有 aarch64 交叉 toolchain，aarch64 主机用宿主工具链。
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
2. 校验一个固定 batch 输入和一个 FP16 输出（见 OM 模型契约）。
3. 读取输入输出名称、shape、dtype 和 buffer size。
4. 一次性创建 dataset 和输入输出 buffer，后续推理复用。

推理时：

1. 将预处理结果绑定或复制到模型输入。
2. 在实例 stream 上执行模型。
3. 同步 stream。
4. 将单张输出复制到 host Blob。
5. 交给 `yolo_e2e` 后处理节点（端到端输出已含 NMS，host 只做坐标恢复）。

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

> 2026-08-05 决策(B 方案):310P3 算力强于 RK3588,无需 RK3588 那种
> "最大化 NPU、host 只做 raw 解码"的极端优化。不要求 OM 导出六路 raw heads;
> 直接消费 Ultralytics 导出(或 ATC 转换后)的单张后处理输出,host 只做
> NMS/top-k/letterbox 坐标恢复。六路 raw-head 契约保留给 RK3588 INT8 路径。

首期从 YOLO26 end2end ONNX 基线生成 FP16 OM（真实模型 Detect 头 `end2end: True`，
NMS 已固化，导出即单输出 `[1, max_det, 6]`）：

```text
ATC soc_version: Ascend310P3
input: images, batch=1, 960x960, fixed shape, NCHW FP16
output: output0, [1, 300, 6], ND FP16（end2end：x1,y1,x2,y2,score,class_id）
```

> 输入值域（2026-08-05 Issue #29 实测确认）：模型输入为 **RGB 归一化 0~1**
> 的 FP16 NCHW（与 Ultralytics 预处理张量一致，即 `image_to_tensor` 负责
> `/255` 归一化与 BGR→RGB）。同一张图喂 0~255 原始像素时检测置信度塌缩为 0，
> 喂 0~1 时与 Ultralytics 参考输出一致（见 Issue #29 真机记录）。首期 ATC 不插入
> static AIPP，归一化由 host 侧 `image_to_tensor` 完成。

> B 方案（2026-08-05 决策）：310P3 算力强于 RK3588，不要求 OM 导出六路 raw heads；
> 直接消费图内已完成的解码/NMS 输出，host 只做 top-k 与 letterbox 坐标恢复。
> 六路 raw-head 契约保留给 RK3588 INT8 路径。若后续接入非 end2end 的
> Ultralytics 导出（`[1, 4+nc, N]` channel-major），`output_format` 使用
> `yolo26_ultralytics`，由 `yolo26_ultralytics_postprocess` 解码（PR #36 已实现）。

模型转换记录必须包含：

- 原始 ONNX SHA256。
- ATC、CANN 版本。
- 完整 ATC 参数和 AIPP 配置（首期不插入 static AIPP，见 ATC 记录）。
- 生成 OM SHA256。
- 输入输出 metadata。

首期只记录一条可复现 ATC 命令，不建设模型转换框架。仅当多个模型需要稳定批量转换时，
再增加转换脚本。完整转换记录见 `docs/development/ascend310p3-yolo26-om-atc.md`。

模型包必须满足：

```json
{
  "chip_type": "ASCEND310P3",
  "model_type": "yolo26_det",
  "models": [
    {
      "file_name": "model.om",
      "max_batch": 1,
      "inputs": [
        { "name": "images", "shape": [1, 3, 960, 960], "data_type": 2 }
      ],
      "outputs": [
        { "name": "output0", "shape": [1, 300, 6], "data_type": 2 }
      ],
      "params": {
        "preprocess_mode": "image_to_tensor",
        "output_format": "yolo_e2e",
        "input_size": [960, 960],
        "padding_color": [114, 114, 114],
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "top_k": 300
      }
    }
  ]
}
```

包内只允许一个显式 `.om` 制品。导入时通过 AscendCL metadata 校验一个固定 batch-1
输入、一个输出、固定 shape `[1,300,6]`、ND format 与 FP16 dtype（`data_type` 2 = HALF）。

## FP16 raw 后处理

现有 `yolo26_raw` 节点只接受 RK3588 INT8 affine 输出，需要扩展为：

- 保留 INT8 + scale/zero-point 路径，保证 RK3588 无回归。
- 新增 FP16 和 FP32读取路径。
- FP16/FP32 不要求量化参数。
- 两种路径共用候选框生成、sigmoid、class-aware NMS、top-k 和坐标恢复。
- 六路输出必须使用相同 dtype，shape 和类别通道必须一致。

首期不做 W8A8。只有阶段性能数据证明模型执行是主要瓶颈，并且存在代表性标定集时，
才单独评估量化。

### B 方案:Ultralytics 单输出解码(`yolo26_ultralytics_postprocess`)

新增独立节点 `NODE_YOLO26_ULTRALYTICS_DECODE`,与六路 `yolo26_raw` 路径并存、互不影响:

- 输入:单张 FP32/FP16 `[1, 4+nc, N]`(channel-major),rows 0-3 = cx/cy/w/h(网络输入像素),
  rows 4+ = 各类别 sigmoid 概率。dist2bbox 与 class sigmoid 已固化在导出图内,
  host 不要求任何量化 scale/zero-point 元数据。
- 处理:置信度直接比较 sigmoid 分数(严格大于),class-aware NMS、top-k,
  输出 `[1, top_k, 6]`(cx, cy, w, h, score, class_id),letterbox 坐标恢复复用
  `PickDetectionObjects`。
- 失败契约:输出非 3 维、通道数 < 5(4 box + 1 class)、dtype 非 FP32/FP16、
  或 class score 超出 [0,1](说明 buffer 不是 channel-major,例如 cell-major 张量被误读)
  均返回可操作错误,不产出垃圾检测。

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

#### OM 契约探测与端到端推理冒烟（2026-08-05）

- 模型：`/opt/convert/cam_p2_distill_ascend_model/best_Ascend310P3.om`（md5 `76104ca44c95c916de6e841a56deb786`，与旧目录 `cki_drone_ascend_model` 为同一文件）。
- AscendCL 描述符探测（`aclmdlGetNumOutputs` / `aclmdlGetOutputFormat` / `aclmdlGetOutputDataType` / `aclmdlGetOutputDims`）：
  - input：`dtype=FP32 format=NCHW dims=[1,3,960,960] size=11059200`
  - output：`dtype=FP32 format=ND dims=[1,300,6] size=7200`（端到端输出，模型内已完成 NMS；`metadata.yaml` 的 `nms: false` 与实际制品不符，`/opt/convert/export_model.py` 硬编码 `nms=True`）。
- 真实推理冒烟（`aclmdlLoadFromFile` + `aclmdlExecute`）：相机帧 1280x720 letterbox 到 960x960（RGB、/255）→ 检出 2 个框：`cls=3 铲车 score=0.9517`、`cls=1 混凝土罐车 score=0.2510`；输出结构与仓库现有 `yolo_e2e_postprocess` 路径（消费 [1,300,6]）匹配。
- 结论：该 OM 证明 310P3 设备推理链路可用，但不能验证 Issue #27 的六路 FP16 raw-head host 解码——端到端输出绕开了本 issue 实现的解码。Issue #27 端到端验收需要 `nms=False` 且 reg/cls 分离的六路 raw-head OM 导出；当前目录无此制品，验证程序已留档在真机 `/tmp/om_probe.c`、`/tmp/om_run.c`，待 raw-head OM 就绪后执行。

#### raw-head 导出复测（2026-08-05，`end2end: false` + `nms: false`）

- 模型重新生成：`/opt/convert/cam_p2_distill_ascend_model/best_Ascend310P3.om`（7332985 B，metadata 为 `end2end: false, nms: false, quantize: 16`）。
- 探测契约：input `FP32 NCHW [1,3,960,960]`；output `FP32 ND [1,10,76500]`（4 尺度 240²/120²/60²/30² × 10 通道）。
- 值分析（同相机帧推理，dump 为 `/tmp/out_raw.raw`）：每 cell 内容 = `[cx, cy, w, h, 6 个 class score]`，class 通道为 sigmoid 激活后的分数 ∈ (0,1)；与端到端模型检出框一致（30² 尺度 grid (16,5) 的 `[162.1, 532.0, 176.8, 134.5]` + class3 0.9795 ≈ E2E 铲车 `[cx=161.5, cy=530.5, w=179.5, h=134]`）。
- 存储布局（numpy 复核，见下节 B 方案真机验证）：channel-major（rows 0-3 ∈ [1.7, 961]，rows 4-9 ∈ [0, 0.979]），即标准 Ultralytics 导出格式；早前把该 buffer 解读为 cell-major 是误读。
- 临时 harness（未入库）把 [1,10,76500] 拆成 6 个头喂 `yolo26_raw_postprocess`：产出 160 个假检测（框坐标超出画幅）——解码器按 raw logits/距离处理，而图内已完成 dist2bbox + class sigmoid。
- 结论：即使 `end2end: false` + `nms: false`，当前蒸馏/Detect 头导出仍把 bbox 解码与 class sigmoid 固化进图，输出不是 raw logits/distances，不符合六路 raw-head 契约。Issue #27 验收需要导出真正未解码的 raw head：每尺度分开的 reg（[1,4,H,W]，reg_max=1 距离）与 cls（[1,6,H,W]，logits）共 6 个 NCHW 张量；验证程序留档在真机 `/tmp/om_probe.c`、`/tmp/om_run.c`，raw-head OM 就绪后即可跑通。

#### B 方案真机验证:Ultralytics 单输出解码(2026-08-05)

- 模型:`/opt/convert/cam_p2_distill_ascend_model/best_Ascend310P3.om`(7332985 B,
  metadata `end2end: false, nms: false, quantize: 16`,Ultralytics YOLO26n-p2、6 类)。
- 探测(`/tmp/om_probe.c`,CANN `/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux`):
  input `FP32 NCHW [1,3,960,960]`;output `FP32 ND [1,10,76500]`(4 尺度 240²/120²/60²/30² × 10 通道)。
- 布局复核(numpy,同相机帧推理 dump `/tmp/out_raw2.raw`):channel-major,rows 0-3 = cx/cy/w/h
  ∈ [1.7, 961],rows 4-9 = 6 类 sigmoid 分数 ∈ [0, 0.979]。
- 候选统计:score ≥ 0.25 共 13 个(铲车 9、混凝土罐车 4)。
- 新节点验证:把真实 dump 构造成 `[1,10,76500]` blob 喂 `yolo26_ultralytics_postprocess`
  (阈值 0.25、NMS 0.45、top-k 300),输出 2 个检测,与 E2E 冒烟一致:
  - `cx=162.1 cy=532.0 w=176.8 h=134.5 score=0.9795 cls=3 铲车`(≈E2E `[161.5, 530.5, 179.5, 134]`)
  - `cx=785.0 cy=606.0 w=358.0 h=304.5 score=0.4746 cls=1 混凝土罐车`
- 测试命令与结果(构建基线 rsync 到 `/root/cosmo-edge-issue27`,`LD_LIBRARY_PATH` 同 CI):
  - `cmake --build build-cpu --target cosmo-tests -j8` → 构建成功。
  - `./build-cpu/cosmo-tests "[nn][yolo26]"` → `All tests passed (173 assertions in 14 test cases)`:
    新增 5 个 ultralytics 用例(黄金阈值/NMS/top-k、letterbox 恢复、FP16、错误契约),
    既有六路 raw/INT8/FP16 黄金用例全部保持绿色。
- 环境说明:本机全量 `cosmo-tests` 链接被预存的 FFmpeg API 不匹配阻塞
  (HEAD 的 `VideoDemuxerStream.cc` 等使用 FFmpeg 5.x `const AVCodec*`,本地 prebuild 与真机
  `/opt/ffmpeg-4.4.1`(Ascend 编译)均为 4.x);真机 checkout 早于该改动,可正常构建。
  与本 issue 改动无关,未在本 issue 修复。

#### Issue #29：YOLO26 AscendCL 推理冒烟（2026-08-05，已执行）

仓库新增 `AscendNetNode`（`src/nn/device/ascend/`），进程级单次 `aclInit`
（`ascend::EnsureAclInitialized()`，CANN 8.0 下第二次 `aclInit` 返回
`100002`/`ACL_ERROR_REPEAT_INITIALIZE`），每个 Graph 独立 context/stream/模型/
dataset/device buffer，推理复用预分配 buffer。输入输出都声明为 host 内存
（节点内部 H2D→`aclmdlExecuteAsync`→stream 同步→D2H 并 FP16→FP32 转换），
因此 Graph 不插入复制节点；不依赖 ONNX Runtime，无 CPU 回退。冒烟程序
`test/ascend310p3/ascend_acl_smoke.cc` 直接驱动真实引擎路径
（`Graph::Init/Forward/Output` + `yolo_e2e` 解码）。

构建与运行（310P3 测试机 `/root/cosmo-edge-issue29`，rsync 自本分支）：

```bash
ln -sfn /root/cosmo-edge-issue29/fmt-7.1.2 /root/cosmo-edge-issue29/3rd/fmt-7.1.2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND -I src -I 3rd/fmt-7.1.2/include \
    -I"${ASCEND_TOOLKIT_HOME}/include" test/ascend310p3/ascend_acl_smoke.cc \
    src/nn/device/ascend/ascend_net_node.cc src/nn/device/ascend/ascend_node_creator.cc \
    src/nn/core/status.cc src/nn/core/blob.cc src/nn/core/blob_impl.cc \
    src/nn/core/abstract_device.cc src/nn/core/abstract_context.cc \
    src/nn/core/shared_resource.cc src/nn/core/blob_store.cc src/nn/core/graph.cc \
    src/nn/node/node.cc src/nn/node/net_node.cc src/nn/node/node_type_utils.cc \
    src/nn/node/node_creator.cc src/nn/node/input_node.cc src/nn/node/identity_node.cc \
    src/nn/node/yolo_e2e_decode_node.cc src/nn/utils/op.cc src/nn/utils/string_format.cc \
    src/nn/utils/timer.cc src/nn/utils/dims_vector_utils.cc \
    src/nn/utils/blob_memory_size_info.cc src/nn/utils/blob_memory_size_utils.cc \
    src/nn/utils/data_type_utils.cc src/nn/device/naive/naive_device.cc \
    src/nn/device/naive/naive_context.cc 3rd/fmt-7.1.2/src/format.cc \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" \
    -lpthread -o ascend_acl_smoke
```

地面真值输入准备（Ultralytics 8.4.112 预处理张量，与 OM 输入逐字节对应）：

```bash
# 720p 相机流 sample_ascend.264 抽第 164 帧，letterbox 到 960x960（灰边 114）
/opt/ffmpeg-4.4.1/ascend/bin/ffmpeg -y -i /opt/ffmpeg-4.4.1/ascend/sample_ascend.264 \
  -vf "select=eq(n\,164)" -vframes 1 /tmp/sa_164.png
# cv2 等比例缩放+居中 padding 到 960x960，再跑 ultralytics 拿预处理张量
# （RGB NCHW FP32 0~1，/255），转 FP16 写 /tmp/input_sa164_0to1.f16
```

ultralytics 参考（同一张 960x960 输入，`imgsz=960`，conf=0.001）：3 个候选，
最高 `cls=3 conf=0.8946 xyxy=[339.6, 326.6, 960, 746.5]`。

运行结果（`uname -a`：`Linux tjx-Default-string 5.15.0-25-generic x86_64`；
`npu-smi info`：`310P3` Health OK，设备 0，`npu-smi 24.1.1.1`）：

```text
$ ./ascend_acl_smoke --om /opt/convert/bjsubway-yolo26/model.om \
    --input /tmp/input_sa164_0to1.f16 --conf 0.25 --iters 3 --expect-detections
om=/opt/convert/bjsubway-yolo26/model.om input_dims=[1,3,960,960] input_bytes=5529600
graph_init device=DEVICE_ASCEND...
iteration=0 output_dims=[1,300,6] dtype=FP32
det[0] cx=650.8 cy=536.5 w=622.5 h=420.0 score=0.8940 class=3
parsed_detections=1 (conf>=0.25)
iteration=1 output identical to iteration 0 (buffers reused)
iteration=2 output identical to iteration 0 (buffers reused)
smoke OK: 3 iteration(s), deterministic parsed detections on device 0
```

对照：OM 输出 `score=0.8940`、中心 `(650.8, 536.5)`、`w/h=622.5/420.0`
（≈xyxy `[339.5, 326.5, 962, 746.5]`）与 Ultralytics 参考
`conf=0.8946`、`xyxy=[339.6, 326.6, 960, 746.5]` 一致（差值来自 FP16 舍入）。
三个迭代输出逐字节一致，确认预分配 buffer 复用；同一输入喂 0~255 时 0 检测，
确认 OM 输入契约为 0~1 归一化。

> 说明：验收标准中 "returns six host FP16 blobs" 按本方案的 B 型契约解释为
> 单张 end2end 输出（每行六个值 `x1,y1,x2,y2,score,class_id`）；设备侧为 FP16，
> host 侧以 FP32 blob 提供（`yolo_e2e` 后处理节点按 FP32 读取）。

排查记录：320x240 基线帧（`/tmp/cosmo_baseline_h264.mp4`）的目标过小，在
960x960 letterbox 下 0 检测（Ultralytics 的 `auto=True` letterbox 实际送
736x960 才有检测，OM 固定 960x960 不接受），故改用 720p `sample_ascend.264`
帧作为地面真值；这是输入侧问题，与推理链路无关。

#### Issue #31：DVPP `image_to_tensor` 视频检测冒烟（2026-08-06，已执行）

仓库新增 `AscendImageToTensorNode`（`src/nn/device/ascend/ascend_image_to_tensor_node.*`）：
host NV12 `FrameSurface` → H2D 上传 → DVPP VPC 单任务完成「居中 letterbox + 等比缩放 +
NV12→RGB888」（`hi_mpi_vpc_crop_resize_make_border`，RGB 边 114 直接由
`scalar_value` 给定）→ D2H 下载 → host 侧 `/255` + FP16 NCHW `{1,3,960,960}`
（`FloatToFp16` 四舍五入到最近偶数，`AscendNodeCreator` 注册
`NODE_IMAGE_TO_TENSOR`）。上传 / DVPP / 下载 / 归一化分阶段计时并打日志，
与 `AscendNetNode` 的 H2D+执行+同步+D2H 计时一起由冒烟程序逐帧输出；
无软件解码、无 CPU resize 兜底。`AiComponment` 在 Ascend 后端下把 host NV12
surface 直接包装为输入 blob（`MakeAscendSurfaceBlob`），不走 CPU 拷贝。

真机排查得到两个 DVPP 关键修复（CANN 8.0 / `hi_dvpp` VPC）：

1. **D2H 拷贝与 VPC 写竞争**：`hi_mpi_vpc_crop_resize_make_border` 提交后不等待
   任务完成就 `aclrtMemcpy` D2H，读到的输出不完整（310P3 上典型表现为下半帧全 0，
   首帧偶发全 0）。修复：提交后 `hi_mpi_vpc_get_process_result(chn, task_id, -1)`
   阻塞等待任务完成再下载。
2. **复用通道 16 任务后死锁**：SDK 的发送环不回收已完成任务，复用同一通道
   连续提交会在第 16 次 `WaitForSend` 永久阻塞（`aclrtSynchronizeDevice` 无效，
   因为 VPC 任务不在 ACL stream 上）。`get_process_result` 会弹出已完成任务，
   环位随之释放；修复后单通道持久复用跑 165 帧无死锁，不再需要每帧
   destroy/create 通道。

冒烟程序 `test/ascend310p3/ascend_video_detect_smoke.cc` 直接驱动真实引擎路径：
demux（文件/RTSP，`mp4toannexb`）→ `h264_ascend`/`h265_ascend` 硬解码 →
Graph（`image_to_tensor_0`(DVPP) → `net_0`(AscendCL) → `yolo_e2e_decode_0`(host)），
逐帧打印 `wall_delta` 与四段耗时，支持 `--dump-frame/--dump-file` 导出 FP16 张量。

构建与运行（310P3 测试机 `/root/cosmo-edge-issue31`，rsync 自本分支；命令同时
写在冒烟文件头注释）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ln -sfn /root/cosmo-edge-issue31/fmt-7.1.2 /root/cosmo-edge-issue31/3rd/fmt-7.1.2
g++ -std=c++17 -O2 -DCOSMO_NN_USE_ASCEND_BACKEND -I src -I 3rd/fmt-7.1.2/include     -I"${ASCEND_TOOLKIT_HOME}/include" test/ascend310p3/ascend_video_detect_smoke.cc     src/nn/device/ascend/ascend_net_node.cc src/nn/device/ascend/ascend_node_creator.cc     src/nn/device/ascend/ascend_image_to_tensor_node.cc     src/nn/core/status.cc src/nn/core/blob.cc src/nn/core/blob_impl.cc     src/nn/core/abstract_device.cc src/nn/core/abstract_context.cc     src/nn/core/shared_resource.cc src/nn/core/blob_store.cc src/nn/core/graph.cc     src/nn/node/node.cc src/nn/node/net_node.cc src/nn/node/node_type_utils.cc     src/nn/node/node_creator.cc src/nn/node/input_node.cc src/nn/node/identity_node.cc     src/nn/node/yolo_e2e_decode_node.cc src/nn/utils/op.cc src/nn/utils/string_format.cc     src/nn/utils/timer.cc src/nn/utils/dims_vector_utils.cc     src/nn/utils/blob_memory_size_info.cc src/nn/utils/blob_memory_size_utils.cc     src/nn/utils/data_type_utils.cc src/nn/device/naive/naive_device.cc     src/nn/device/naive/naive_context.cc src/media/VideoDecoder.cc     src/media/VideoDecoderCreateAscend.cc src/media/VideoDecoderAscend.cc     src/media/VideoFrame.cc src/media/PixelFormatUtils.cc     src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc src/mem/AllocatorCpu.cc     src/mem/BlockFreqCalc.cc src/util/Thread.cc src/util/ThreadRegistry.cc     src/util/TimeUtil.cc 3rd/fmt-7.1.2/src/format.cc     -I/opt/ffmpeg-4.4.1/ascend/include -L/opt/ffmpeg-4.4.1/ascend/lib     -lavformat -lavcodec -lavutil -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib     -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl -lacl_dvpp -lacl_dvpp_mpi     -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread -ldl -lm     -o ascend_video_detect_smoke
```

地面真值（同一帧直接对照 Ultralytics，避免预处理来源不一致）：

```bash
# 取一张地铁场景测试图（训练集 eval 图片，模型 conf=0.966 的检测样本），
# 缩放编码成 1280x720 H.264 / H.265 各 75 帧
ffmpeg -y -loop 1 -i upload_2261802601891843740_9b34f57f-6e5f-4860-92e3-a1c7d8abe735.jpg   -t 3 -r 25 -vf scale=1280:720 -pix_fmt yuv420p -c:v libx264 subway_test_h264.mp4
ffmpeg -y -loop 1 -i upload_2261802601891843740_9b34f57f-6e5f-4860-92e3-a1c7d8abe735.jpg   -t 3 -r 25 -vf scale=1280:720 -pix_fmt yuv420p -c:v libx265 subway_test_h265.mp4
# Ultralytics 对同一 1280x720 帧 imgsz=960 的参考检测：
#   cls=2 conf=0.9363 xyxy=[304, 66, 1190.9, 473.7] → letterbox(0.75, pad_y=210)
#   后 cxcywh = (560.6, 412.4, 665.2, 305.8)
```

运行结果（`uname -a`：`Linux tjx-Default-string 5.15.0-25-generic x86_64`；
`npu-smi info`：310P3 Health OK，设备 0，`npu-smi 24.1.1.1`）：

```text
$ ./ascend_video_detect_smoke --om /opt/convert/bjsubway-yolo26/model.om     --video /tmp/subway_test_h264.mp4 --frames 75 --conf 0.25 --expect-detections
video=/tmp/subway_test_h264.mp4 1280x720 codec=h264_ascend
frame=74 wall_delta=36.50ms
frame=74 e2e_graph=36.22ms image_to_tensor=17.19ms acl=19.01ms postprocess=0.00ms
  det[0] cx=559.9 cy=417.1 w=666.2 h=299.8 score=0.9468 class=2
  parsed_detections=1 (conf>=0.25)
[h264_ascend] Decode hw send packet count is: 75.
[h264_ascend] Decode hw out frame count is: 75.
e2e=75 frames in 2.79s (26.9 fps incl. decode)
detected_frames=75/75 (conf>=0.25)
smoke OK

$ ./ascend_video_detect_smoke --om /opt/convert/bjsubway-yolo26/model.om     --video /tmp/subway_test_h265.mp4 --frames 75 --conf 0.25 --expect-detections
  det[0] cx=560.2 cy=416.9 w=667.5 h=300.2 score=0.9463 class=2
e2e=75 frames in 2.77s (27.1 fps incl. decode)
detected_frames=75/75 (conf>=0.25)
smoke OK

# 空场景样本（sample_ascend.264，全片无可检目标，与 Ultralytics 一致）
$ ./ascend_video_detect_smoke --om /opt/convert/bjsubway-yolo26/model.om     --video /opt/ffmpeg-4.4.1/ascend/sample_ascend.264 --frames 165
e2e=165 frames in 5.99s (27.6 fps incl. decode)
detected_frames=0/165 (conf>=0.25)
smoke OK
```

对照：DVPP 链路检测 `cxcywh=(559.9, 417.1, 666.2, 299.8) score=0.9468 class=2`
与 Ultralytics 同帧参考 `(560.6, 412.4, 665.2, 305.8) score=0.9363 class=2`
一致（FP16/DVPP 双线性与 Ultralytics INTER_AREA 的微小差异），证明
NV12→RGB888 颜色顺序、letterbox 几何、`/255` 归一化与 OM 契约匹配。

> 说明：Issue #29 记录的参考张量 `/tmp/input_sa164_0to1.f16`（Ultralytics
> 预处理输出）经核对并不对应 `sample_ascend.264` 的任何一帧（该视频全片为空场景，
> 系统 ffmpeg 与自定义 ascend ffmpeg 的 150~180 帧均与参考 PNG MAE≈37）；
> 因此本 issue 改用「同一解码帧直接对照 Ultralytics」的地面真值，不引用该张量。

RTSP 实时流检测与 `COSMO_TARGET_ARCH=aarch64`（经跳板机）冒烟：
未执行（无主机访问权限）。

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


#### Issue #33：本地文件 / RTSP YOLO26 检测任务（2026-08-06，已执行）

把已打通的全链路（demux → `h264_ascend`/`h265_ascend` 硬解码 → DVPP
`image_to_tensor` → AscendCL `net` → host `yolo_e2e_decode`）接入产品任务生命周期：
本地 H.264/H.265 文件、RTSP 源、RTSP 断线重连、重复启停、单卡 1/3 实例池策略、
错误上抛（无软件推理兜底）、preview 等任务侧消费方的 host-copy 路径。

冒烟程序 `test/ascend310p3/ascend_task_smoke.cc` 直接驱动真实引擎路径并复用
`AscendVideoDetectSmoke` 的构建方式（命令见文件头注释；构建脚本
`/root/cosmo-edge-issue33/build_task_smoke.sh`，rsync 自本分支，含
`src/media/VideoFrameProcCpu.cc`、`src/flow/channel/AlgChannelDecode.cc` 等）。

**真机发现的关键驱动问题（VDEC/VPC 楔死）**

1. **VDEC 通道销毁后无法重开**：一旦 VDEC 通道与 VPC 通道共存后销毁（RTSP
   重连 / 文件重启时 decoder Close→Open），之后每次 VDEC 重开会收包但回调线程
   永久循环在 `HI_ERR_VDEC_BUF_EMPTY`（`0xA005800E`），drain 0 帧；teardown
   时 `Decode sem_timewait=-1` 直接 abort（rc=134）。尝试过的
   `hi_mpi_sys_exit`、每轮换通道号、`hi_mpi_vdec_reset_chn`、VDEC/VPC 销毁顺序
   调换、整库 deinit 均无效（`hi_mpi_sys_init/exit` 是引用计数的）。
2. **可用模式**：一个持久 VDEC 通道跨 demux 会话复用（正是产品 RTSP 重连/文件
   重启的形态），VPC 通道存活时复测 r1/r2/r3 均 75/75。
3. **两个配套约束**：Ascend FFmpeg 解码器只在包到达时近似实时出帧（批量喂 ≈1/75，
   10~50ms 节奏喂 ≈70/75），冒烟按节奏喂包；EOS flush 会永久禁用该通道
   （`send err -541478725`），任务中途**绝不 flush**。

产品修复（`src/flow/channel/AlgChannelDecode.cc` + `src/media/VideoDecoder.h` /
`VideoDecoderAscend.h`）：`VideoDecoder` 新增 `ShouldReuseAcrossStreamChange()`，
Ascend 后端返回 true——流切换时只重置逐流簿记（`frame_info_`、`stream_index_`、
`decode_count_`、`frame_index_`），不再 Close/Open VDEC 通道；异常路径
（`codec_reset_sign_`）同样不再 Close/Open（避免触发楔死），只重置簿记并清除
复位标记；`decoder_->Open()` 失败会置 `DecoderFrameFailed` 任务状态（无软件
推理兜底）。其他后端行为不变。持久通道固定首个流的 codec/分辨率，中途换码流
需重启任务（Close/Open 在 VPC 存活时会楔死 310P3），换码流会以解码失败上抛。

预览 host-copy 修复（`src/media/VideoFrameProcCpu.cc`）：Ascend VDEC 的 NV12
`FrameSurface` 是 Y/UV 两块独立 host plane（`GetContiguousData()` 为空），
`EnsureHostData()` 改为同时接受「有有效 plane 的 host surface」；
`ConvertPixelFormat`/`CopyFrame` 在无连续基址时改用
`planes[i].virt_addr + offset` 与 `linesize(pitch)` 喂 sws / 逐行拷贝，覆盖
`StreamViewerEncoder` 预览、`TaskAlarmPicture` 抓拍、`EncodeJpeg` 等任务侧
host-copy 消费方。

实例池（`src/service/ai/impl/InferPoolServiceImpl.cc`）：Ascend 后端
`DetectorPool(alg_code, 1, 3)`——每个任务一个 ACL 实例，最多三个，不加第二
worker 池；`ascend_task_smoke` 的 `CheckDetectorPoolPolicy` 验证三任务各持一
实例、第四个任务拿不到实例。

构建与运行（310P3 测试机 `/root/cosmo-edge-issue33`，`uname -a`：
`Linux tjx-Default-string 5.15.0-25-generic x86_64`；`npu-smi`：310P3 Health OK，
设备 0）：

```bash
# 冒烟构建
bash /root/cosmo-edge-issue33/build_task_smoke.sh   # 末尾打印 SMOKE_BUILD_OK
# H.264 / H.265 本地文件，各 2 轮（重复启停）
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video /tmp/subway_test_h264.mp4 --frames 70 --rounds 2 --expect-detections
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video /tmp/subway_test_h265.mp4 --frames 70 --rounds 2 --expect-detections
# 单卡 3 实例
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video /tmp/subway_test_h264.mp4 --frames 70 --rounds 2 --instances 3 --expect-detections
# RTSP 断线重连（任务中途 kill 掉发布进程再重启）
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video rtsp://127.0.0.1:8554/stream --frames 120 --rtsp-reconnect \
  --rtsp-source /tmp/subway_test_h264.264 \
  --rtsp-python /root/cosmo-edge-issue33/test/ascend310p3/rtsp_test_source.py --expect-detections
# preview host-copy（NV12 → I420 + JPEG 抓拍）
./ascend_task_smoke --om /opt/convert/bjsubway-yolo26/model.om \
  --video /tmp/subway_test_h264.mp4 --frames 10 --preview-check
```

运行结果（`--expect-detections`，模型 `/opt/convert/bjsubway-yolo26/model.om`，
素材与 Issue #31 同源；地面真值 `cx≈560 cy≈417 w≈666 h≈300 score≈0.9468 class=2`）：

```text
H.264  --frames 70 --rounds 2   : decoded 70+70，detected_frames=140/140，task smoke OK
H.265  --frames 70 --rounds 2   : decoded 70+70，detected_frames=140/140，task smoke OK
3 实例 --frames 70 --rounds 2   : e2e=140 frames across 3 instances x 2 rounds，
                                  detected_frames=140/140，task smoke OK
RTSP 重连 --frames 120          : 发布进程被 kill 后重启，会话 75+120，
                                  detected_frames=195/195，task smoke OK
preview-check                   : preview sws NV12ToI420: ok；
                                  preview host-copy passed: NV12 -> I420 host,
                                  capture JPEG bytes=550291；task smoke OK
```

错误路径（每次运行先执行，硬件无关）：不支持的 codec（Mjpeg）拒绝 Open，坏 OM
在 `Graph::Init` 抛错，均无软件推理兜底；`CheckErrorPaths` 打印
`error paths passed: bad OM rejected / unsupported codec rejected with no fallback`。

RK3588 板回归：未执行。本 issue 的 `ShouldReuseAcrossStreamChange()` 改动位于公共
`AlgChannelDecode` 路径，默认返回 false 保持原有 Close/Open 行为；本地
`scripts/test_target_platform_profiles.sh` 全绿覆盖其余后端编译。

## 首期不做

- 多卡枚举、任务绑卡和跨卡调度。
- 动态 batch、动态 shape 或多尺寸模型。
- YOLO26 之外的分类、OCR、DINO、SAM、VLM 和大模型。
- W8A8、ModelSlim、NPU NMS 或自定义算子。
- Docker 发布和多 CANN/FFmpeg 版本兼容层。
- 为一个测试环境设计运行时插件或通用厂商抽象。
