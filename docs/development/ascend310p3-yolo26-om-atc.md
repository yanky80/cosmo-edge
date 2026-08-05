---
title: Ascend310P3 YOLO26 OM 转换记录（ATC/AIPP）
description: 与 OM 模型契约配套的可复现 ATC/AIPP 转换记录，包含 ONNX/OM 哈希、CANN 版本、完整参数与输入输出 metadata，以及真机复验结果。
---

# Ascend310P3 YOLO26 OM 转换记录

本记录对应 `docs/development/ascend310p3-adaptation-plan.md` 的
[OM 模型契约](#om-模型契约)。首期 310P3 契约为**单输出 end2end**（B 方案：
NMS 固化进图，host 只做 letterbox 坐标恢复），不再是六路 raw heads；
六路 raw-head 契约保留给 RK3588 INT8 路径。平台导入时用 AscendCL metadata
校验该契约（见 `src/service/model/impl/ModelImporter.cc`）。

## 转换记录（2026-08-05，已在 310P3 真机执行）

ONNX 基线由训练权重导出并完成 ATC 转换，全部命令在 310P3 测试机
（`root@35623rcqc768.vicp.fun:1022`）执行。源文件：

| 项目 | 路径 |
| --- | --- |
| 训练权重 | `/mnt/ml-storage/model/bjsubway-cam-260611/train/weights/best.pt`（yolo26m，960×960，6 类） |
| 导出工作目录 | `/opt/convert/bjsubway-yolo26/`（测试机） |

### ONNX 基线导出

模型 Detect 头为 `end2end: True`（NMS 固化，`max_det=300`），因此导出即单输出
`[1,300,6]`；CANN 8.0.0 的 ONNX parser 不支持 opset 19，使用 opset 12 导出：

```bash
flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun \
    "python3 -c \"from ultralytics import YOLO; YOLO(\\\"best.pt\\\").export(format=\\\"onnx\\\", imgsz=960, half=False, simplify=False, opset=12)\""'
```

| 项目 | 值（实测） |
| --- | --- |
| Ultralytics | 8.4.112（测试机 `python3`） |
| ONNX opset | 12 |
| ONNX SHA256 | `e5406bd45d70140a1d2931cee593db78949d3ee49c1369bf5e7b0f261148863a` |

### 转换命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=yolo26_det.onnx \
    --output=model \
    --soc_version=Ascend310P3 \
    --framework=5 \
    --input_format=NCHW \
    --input_shape="images:1,3,960,960" \
    --output_type=FP16 \
    --input_fp16_nodes=images \
    --log=error
```

| 项目 | 值（实测） |
| --- | --- |
| ATC 版本 | CANN 8.0.0（`version.cfg`：runtime/compiler/hccl/opp/toolkit/aoe/ncs `[7.6.0.1.220:8.0.0]`） |
| CANN 版本 | 8.0.0（`/usr/local/Ascend/ascend-toolkit/latest`） |
| 生成 OM | `model.om` |
| OM SHA256 | `e8a14da469c6fc4804be86f2687fc395d9d8fe44ff1aed0b57dc2bc6676d76f8` |

> **可复现性说明**：ATC 转换耗时约 29s，重复转换的 OM 字节不完全一致
> （实测两次 SHA256 分别为 `e8a14da4…` 与 `a73f3469…`，文件大小差 506B，
> 为 ATC 写入的时间戳/版本信息），功能等价。本记录以首次转换哈希为准。

### AIPP 配置（首期不使用）

首期契约为固定 shape FP16 流水线：OM 输入保持 `images: NCHW FP16 [1,3,960,960]`，
归一化与通道顺序由 `image_to_tensor`（host/DVPP 链路）负责，**不在 ATC 阶段插入
static AIPP**。若后续 DVPP 直通需要 AIPP，则必须同步调整导入校验契约：
插入 static AIPP 后 `aclmdlGetInputDims/Format/DataType` 报告的是 AIPP 输入格式
（如 `RGB888_U8`），与 `NCHW FP16 [1,3,960,960]` 契约不一致。参考配置如下，使用前
需按“转换记录 ↔ 导入校验”一致原则重新探测并回填：

```ini
aipp_op {
  aipp_mode: static
  input_format: RGB888_U8
  src_image_size_w: 960
  src_image_size_h: 960
  csc_switch: true
  rbuv_swap_switch: false
  mean_chn_0: 0
  mean_chn_1: 0
  mean_chn_2: 0
  min_chn_0: 0.003921569
  min_chn_1: 0.003921569
  min_chn_2: 0.003921569
}
```

### 输入输出 metadata（AscendCL 实测）

```bash
# /opt/convert/bjsubway-yolo26/om_probe2（aclmdlGetInputDims/OutputDims）
./om_probe2 model.om
```

| 张量 | 名称（runtime） | shape | dtype | 布局 | buffer size |
| --- | --- | --- | --- | --- | --- |
| 输入 | `images` | 1,3,960,960 | FP16 | NCHW | 5529600 B |
| 输出 | `/model.23/Concat_6:0:output0` | 1,300,6 | FP16 | ND | 3600 B |

> 输出 tensor 的 AscendCL runtime 名为 ATC 内部名 `/model.23/Concat_6:0:output0`，
> 对应 ONNX 输出 `output0`。导入校验按 shape/dtype/format/count 匹配契约，
> 不比对 runtime 名字；config.json 模板中的输出名沿用 ONNX 名 `output0`。
> 输出语义为 end2end `[1, max_det, 6]`（每行 `x1,y1,x2,y2,score,class_id`），
> 由 `yolo_e2e` 后处理节点消费（`yolo_e2e` 节点当前按 FP32 读取；
> FP16 OM 输出的 FP16 读取是推理链路后续问题，不在本 issue 导入范围内）。

## 真机复验（2026-08-05 已执行）

```bash
flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun \
    "uname -a && npu-smi info | head -8 && sha256sum /opt/convert/bjsubway-yolo26/yolo26_det.onnx /opt/convert/bjsubway-yolo26/model.om && cat \${ASCEND_TOOLKIT_HOME}/version.cfg | head -4"'
```

实测结果：

- 主机：`Linux tjx-Default-string 5.15.0-25-generic x86_64`；`npu-smi 24.1.1.1`，
  NPU 8（310P3，`0000:01:00.0`），Health OK。
- `yolo26_det.onnx` SHA256 `e5406bd4…`；`model.om` SHA256 `e8a14da4…`。
- CANN `version.cfg`：`runtime_running_version=[7.6.0.1.220:8.0.0]`。

### ascend310p3 profile 构建验证（已执行）

```bash
export PATH=/root/.cargo/bin:$PATH   # tokenizers_external 需要 Rust（测试机 /root/.cargo）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build_ascend -DCOSMO_TARGET_PLATFORM=ascend310p3 -DCOSMO_TARGET_ARCH=x86_64 -DCMAKE_BUILD_TYPE=Release
cmake --build build_ascend --target cosmo_service -j8
```

实测结果：配置阶段找到真实 CANN SDK（`/usr/local/Ascend/ascend-toolkit/latest`，
AscendCL/DVPP）与定制 FFmpeg（`/opt/ffmpeg-4.4.1/ascend/include`，
见 `docs/development/ascend310p3-test-host-baseline.md`）；`cosmo_service`
（含 `ModelImporter.cc` 的 AscendCL metadata loader）构建成功。

### 推理冒烟（已执行）

```bash
# /opt/convert/bjsubway-yolo26/om_run28：aclmdlLoadFromFile + aclmdlExecute，
# 全零 FP16 输入
./om_run28 model.om
```

实测结果：`aclmdlExecute ret=0`，输出 `[1,300,6]` FP16，全零输入下
score>0.5 命中 0（符合预期），OM 可在 310P3 真实加载执行。
