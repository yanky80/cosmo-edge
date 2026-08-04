---
title: Ascend310P3 YOLO26 OM 转换记录（ATC/AIPP）
description: 与 OM 模型契约配套的可复现 ATC/AIPP 转换记录，包含 ONNX/OM 哈希、CANN 版本、完整参数与输入输出 metadata。
---

# Ascend310P3 YOLO26 OM 转换记录

本记录对应 `docs/development/ascend310p3-adaptation-plan.md` 的
[OM 模型契约](#om-模型契约)：单一 `.om` 制品、固定 batch-1 NCHW FP16 输入、
六个按 `reg0, cls0, reg1, cls1, reg2, cls2` 排序的 FP16 NCHW 输出。
平台导入时用 AscendCL metadata 校验该契约。

## 转换记录（待真机执行）

> 状态：**未执行：无主机访问权限**。以下为可复现的转换与采集命令；取得 ONNX 基线后，
> 在 `AGENTS.md` 记录的 310P3 测试机（CANN 8.0.0，ATC 位于
> `${ASCEND_TOOLKIT_HOME}/atc/bin/atc`）上执行并回填哈希与版本。

### 基线输入

| 项目 | 值 |
| --- | --- |
| ONNX 基线 | `yolo26_det.onnx`（六路 raw-head，输出名 `reg0/cls0/reg1/cls1/reg2/cls2`） |
| ONNX SHA256 | `<待回填 sha256sum yolo26_det.onnx>` |
| 输入 | `[1,3,640,640]` FP16，固定 shape（ATC `--input-shape`） |
| 输出 | `[1,4,80,80]`、`[1,1,80,80]`、`[1,4,40,40]`、`[1,1,40,40]`、`[1,4,20,20]`、`[1,1,20,20]` FP16 NCHW |

### 转换命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=yolo26_det.onnx \
    --output=model \
    --soc_version=Ascend310P3 \
    --framework=5 \
    --input_format=NCHW \
    --input_shape="images:1,3,640,640" \
    --output_type=FP16 \
    --insert_op_conf=aipp_yolo26.cfg \
    --log=error
```

| 项目 | 值 |
| --- | --- |
| ATC 版本 | CANN 8.0.0（`version.cfg` 为准，ATC 不支持 `--version`） |
| CANN 版本 | 8.0.0（`${ASCEND_TOOLKIT_HOME}/version.cfg`） |
| 生成 OM | `model.om` |
| OM SHA256 | `<待回填 sha256sum model.om>` |

### AIPP 配置（aipp_yolo26.cfg）

```ini
aipp_op {
  aipp_mode: static
  input_format: RGB888_U8
  src_image_size_w: 640
  src_image_size_h: 640
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

> AIPP 将 RGB888 U8 图像归一化到 `[0,1]` 并完成通道顺序固定；模型输入为
> NCHW FP16。若真机 DVPP 直接产出 BGR，切换 `input_format` 与 `rbuv_swap_switch`
> 并在转换记录中注明，保持归一化/通道顺序固定在该契约内。

> **真机确认项**：插入 static AIPP 后，`aclmdlGetInputDims/Format/DataType` 报告的
> 输入张量可能跟随 AIPP 输入（RGB888 U8 而非 NCHW FP16），与导入校验的
> `NCHW FP16 [1,3,640,640]` 契约不一致。执行转换后必须在测试机读取 model.om 的
> AscendCL 输入 metadata 并回填下表；若输入实际为 AIPP 输入格式，则调整 ATC 参数
> （去掉 static AIPP、由 DVPP/宿主完成归一化与通道顺序，或按实际格式同步导入校验契约）
> 后重新转换，保证“转换记录 ↔ 导入校验”两者一致。未确认前不得视为验收完成。

### 输入输出 metadata 采集

转换后按 AscendCL metadata 回填（与导入校验同一来源）：

```bash
# 通过平台导入校验输出：stage=ascend/input/output 由 cosmo-engine 打印
# 或临时以 aclmdlGetInputDims/aclmdlGetOutputDims 读取 model.om
```

| 张量 | shape | dtype | 布局 |
| --- | --- | --- | --- |
| images | 1,3,640,640 | FP16 | NCHW |
| reg0 | 1,4,80,80 | FP16 | NCHW |
| cls0 | 1,1,80,80 | FP16 | NCHW |
| reg1 | 1,4,40,40 | FP16 | NCHW |
| cls1 | 1,1,40,40 | FP16 | NCHW |
| reg2 | 1,4,20,20 | FP16 | NCHW |
| cls2 | 1,1,20,20 | FP16 | NCHW |

## 复验命令（真机）

```bash
flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun "uname -a && npu-smi info"'
source /usr/local/Ascend/ascend-toolkit/set_env.sh
sha256sum yolo26_det.onnx model.om
cat "${ASCEND_TOOLKIT_HOME}/version.cfg"
```
