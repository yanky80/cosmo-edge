---
title: 310P3 测试主机基线
description: 华为昇腾 310P3 测试主机的软件/媒体基线记录，以及复验命令。
---

# 310P3 测试主机基线

采集日期：2026-08-04。主机为 `AGENTS.md` 记录的共享测试机
`root@35623rcqc768.vicp.fun:1022`。所有命令在本地锁下执行：

```bash
flock -w 1800 /tmp/cosmo-edge-ascend310p3-hw.lock -c \
  'ssh -o ConnectTimeout=10 -p 1022 root@35623rcqc768.vicp.fun "<command>"'
```

构建与发布基线以本记录为准，首期不维护多个 CANN/FFmpeg 兼容分支。

## CPU / OS

| 项目 | 值 |
| --- | --- |
| CPU | Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz，12 线程，x86_64 |
| 内存 | 31 GB |
| 内核 | Linux 5.15.0-25-generic (Ubuntu SMP) |
| 发行版 | Ubuntu 22.04 LTS (Jammy) |
| 主机名 | tjx-Default-string |

```bash
uname -a
grep -m1 "model name" /proc/cpuinfo
cat /etc/os-release
```

## 310P3 设备、驱动与固件

| 项目 | 值 |
| --- | --- |
| 芯片 | Ascend 310P3 (Chip Version V1) |
| NPU ID / Chip ID | 8 / 0 |
| PCIe | 0000:01:00.0，Board ID 0x6e |
| 显存 | 21527 MB（采集时已用 1836 MB） |
| 驱动 | 24.1.1.1（ascendhal 7.35.23，aicpu 1.0，dvppkernels 1.1） |
| 固件 | 7.5.0.5.220 |
| 驱动 Inner version | V100R001C19SPC007B220 |
| 兼容固件范围 | [6.4.0,6.4.99]、[7.0.0,7.6.99] |

```bash
npu-smi info
npu-smi info -t board -i 8 -c 0
cat /usr/local/Ascend/driver/version.info
```

## CANN / AscendCL / DVPP / ATC

| 项目 | 值 |
| --- | --- |
| Toolkit 根 | `/usr/local/Ascend/ascend-toolkit/latest`（软链到 `8.0.0`） |
| CANN 版本 | 8.0.0（`version.cfg`：runtime/compiler/hccl/opp/toolkit `[7.6.0.1.220:8.0.0]`） |
| 环境脚本 | `/usr/local/Ascend/ascend-toolkit/set_env.sh`（导出 `ASCEND_TOOLKIT_HOME` 等） |
| AscendCL 头文件 | `include/acl/acl.h`、`acl_base.h`、`acl_rt.h`、`acl_mdl.h` 等 |
| AscendCL 库 | `lib64/libascendcl.so` |
| DVPP 头文件 | `include/acl/dvpp/hi_dvpp*.h`（`hi_dvpp.h`、`hi_dvpp_vpc.h`、`hi_dvpp_vdec.h` 等）及 `include/acl/media/` |
| DVPP 库 | `lib64/libacl_dvpp.so`（另有 `libacl_dvpp_mpi.so`、`libacl_dvpp_op.so`、`libdvpp_cmdlist.so`） |
| ATC | `latest/atc/bin/atc`（`atc --version` 不支持，版本以 toolkit `version.cfg` 为准） |

CANN 不提供 `pkg-config` 文件，CMake 直接按上述头文件/库路径检查。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cat "${ASCEND_TOOLKIT_HOME}/version.cfg"
ls "${ASCEND_TOOLKIT_HOME}/include/acl/"
ls "${ASCEND_TOOLKIT_HOME}/include/acl/dvpp/"
ls "${ASCEND_TOOLKIT_HOME}/lib64/" | grep -E "ascendcl|dvpp"
```

## 编译器与 C 库

| 项目 | 值 |
| --- | --- |
| gcc / g++ | 11.4.0（Ubuntu 11.4.0-1ubuntu1~22.04.2） |
| glibc | 2.35（Ubuntu GLIBC 2.35-0ubuntu3.10） |
| C++ ABI | GCC 11 默认（Itanium C++ ABI，`_GLIBCXX_USE_CXX11_ABI=1`） |

## FFmpeg 基线（重要）

适配方案假设测试机已有“定制 FFmpeg 硬解码”，**实测不成立**：测试机安装的是
Ubuntu 自带 FFmpeg，**没有任何昇腾硬件解码器或 hwaccel**。构建 Profile 按计划
使用系统 FFmpeg；定制 Ascend FFmpeg 属于后续交付，其基线待其安装后复采。

| 项目 | 值 |
| --- | --- |
| 路径 | `/usr/bin/ffmpeg`（无 `/usr/local` 下的定制构建） |
| 版本 | 6.1-1build2~22.04（libavcodec 60.31.102，libavutil 58.29.100） |
| 来源 | Ubuntu 官方软件包 |
| 硬件解码器 | 无 Ascend；仅有 `h264_v4l2m2m`、`hevc_v4l2m2m`、`h264_qsv`、`hevc_qsv` |
| hwaccels | vdpau / vaapi / qsv / drm / opencl（无 Ascend） |
| 软件解码像素格式 | H.264 High → `yuv420p`；HEVC Main → `yuv420p`（实测 ffprobe） |

```bash
ffmpeg -version | head -3
ffmpeg -hide_banner -decoders | grep -E "h264|hevc|ascend"
ffmpeg -hide_banner -hwaccels
```

测试用样本（生成于采集时，`/tmp` 下）：`/tmp/cosmo_baseline_h264.mp4`（H.264
High，320x240，`yuv420p`）、`/tmp/cosmo_baseline_h265.mp4`（HEVC Main，
320x240，`yuv420p`）。

## AVFrame 所有权契约

未执行：当前主机**没有**可导出硬件帧的定制 FFmpeg，且未安装
`libavcodec-dev`（无法在不安装系统包的前提下编译探针；`AGENTS.md` 禁止未经
批准安装系统包）。主机已确认在线可达，其余基线均为实测值。

待定制 Ascend FFmpeg 就位后，用以下探针步骤复采并回填本表：

1. 安装/部署定制 FFmpeg 的 dev 头文件，编译一个最小解码探针：用
   `avcodec_find_decoder` 指定硬件解码器，逐帧打印
   `AVFrame::format`、`width/height`、`linesize[]`、`buf[]` 个数与
   `AVBufferRef` 类型、`AVFrame::data[]` 是否指向设备地址。
2. 记录硬件像素格式（预期如 `AV_PIX_FMT_NV12` 或昇腾私有格式）和
   `hw_frames_ctx` 的设备类型。
3. 验证所有权契约：解码器返回的帧由 libavcodec 引用计数管理
   （`av_frame_ref`/`av_frame_unref`），消费者（DVPP 输入、预览、抓图）必须
   在各自使用期间持有引用；设备帧地址仅在该引用生命周期内有效。

该契约是 `docs/development/ascend310p3-adaptation-plan.md` 阶段二
“设备帧直通”的前提，未确认前按阶段一“host NV12 + 一次 H2D”执行。

## 复验命令汇总

上述全部命令按小节顺序执行一遍即为完整复验；任何值与上表不一致时，以复验
结果更新本文件并同步调整 `cmake/ascend_sdk.cmake` 的检查项。
