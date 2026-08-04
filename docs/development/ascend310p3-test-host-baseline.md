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

测试机有两套 FFmpeg：Ubuntu 自带 FFmpeg 6.1（工具/预览用）和
`/opt/ffmpeg-4.4.1` 的**定制 Ascend FFmpeg 4.4.1**（昇腾媒体后端，媒体后端以
此为准）。构建 Profile 使用定制版本，不从仓库复制任何 FFmpeg 二进制。

### 定制 Ascend FFmpeg（媒体后端）

| 项目 | 值 |
| --- | --- |
| 源码树 | `/opt/ffmpeg-4.4.1`（就地构建，二进制 `/opt/ffmpeg-4.4.1/ffmpeg`） |
| 安装前缀 | `/opt/ffmpeg-4.4.1/ascend`（`bin/`、`include/`、`lib/`、`share/`） |
| 版本 | 4.4.1（libavcodec 58.134.100，libavutil 56.70.100） |
| 编译器 | clang 14.0.0-1ubuntu1.1 |
| 配置 | `--enable-cross-compile --enable-shared --enable-ascend --enable-sdl --enable-ffplay --prefix=./ascend`，`--extra-libs='-lacl_dvpp_mpi -lascendcl'`（CANN acllib） |
| 解码器 | `h264_ascend`、`h265_ascend`、`mjpeg_ascend`（Ascend HiMpi） |
| 编码器 | `h264_ascend`、`h265_ascend` |
| hwaccel | `ascend` |
| 解码像素格式 | `h264_ascend`/`h265_ascend` 支持 `ascend nv12`（`ffmpeg -h decoder=...`） |
| 编码像素格式 | `h264_ascend`/`h265_ascend` 支持 `nv12 ascend` |
| 解码器 AVOptions | `device_id`(0-8，默认 0)、`channel_id`(0-255)、`resize`(WxH) |
| 编码器 AVOptions | `device_id`、`channel_id`(0-127)、`profile`(0 baseline/1 main/2 high)、`rc_mode`(0 CBR/1 VBR)、`gop`、`frame_rate`、`max_bit_rate`、`movement_scene` |
| 运行时依赖 | `/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/lib64/libacl_dvpp_mpi.so`、`libascendcl.so`、`libdvpp_op_base.so`（`ldd` 实测） |
| 样本 | `/opt/ffmpeg-4.4.1/ascend/sample_ascend.264`（1280x720 H.264） |

```bash
/opt/ffmpeg-4.4.1/ffmpeg -version | head -3
/opt/ffmpeg-4.4.1/ffmpeg -hide_banner -decoders | grep -E "ascend|h264|hevc"
/opt/ffmpeg-4.4.1/ffmpeg -hide_banner -encoders | grep ascend
/opt/ffmpeg-4.4.1/ffmpeg -hide_banner -hwaccels
/opt/ffmpeg-4.4.1/ffmpeg -hide_banner -h decoder=h264_ascend
ldd /opt/ffmpeg-4.4.1/ffmpeg
```

### 系统 FFmpeg（工具/预览）

| 项目 | 值 |
| --- | --- |
| 路径 | `/usr/bin/ffmpeg` |
| 版本 | 6.1-1build2~22.04（libavcodec 60.31.102） |
| 硬件能力 | 无 Ascend 解码器/hwaccel；仅有 v4l2m2m、qsv 等 |
| 软件解码像素格式 | H.264 High → `yuv420p`；HEVC Main → `yuv420p`（实测 ffprobe） |

测试用样本（`/tmp` 下）：`/tmp/cosmo_baseline_h264.mp4`、`/tmp/cosmo_baseline_h265.mp4`。

## AVFrame 所有权契约（实测）

用最小探针（`avcodec_find_decoder_by_name("h264_ascend")`，链接
`/opt/ffmpeg-4.4.1/ascend` 的头文件和共享库）解码
`sample_ascend.264`，前 3 帧实测：

```text
decoder=h264_ascend hw_device_ctx=(nil)
frame=0 format=nv12(23) w=1280 h=720 linesize=[1280,1280,0]
       buf0=0x... buf1=0x... buf0_data=0x... hw_frames_ctx=(nil) data0=0x...
```

结论（默认路径，未设置 `-device_frame` 类选项）：

- 输出像素格式为 `AV_PIX_FMT_NV12`（host 内存），不是 `ascend` 设备格式。
- `buf[0]`/`buf[1]` 为有效 `AVBufferRef`，`data[0]` 指向 host 地址：
  帧由 libavcodec 引用计数托管，生命周期用 `av_frame_ref`/`av_frame_unref`
  管理；`hw_frames_ctx = NULL`，无设备帧上下文。
- `h264_ascend` 同时支持 `ascend` 设备像素格式（解码器列表
  `Supported pixel formats: ascend nv12`）；设备帧路径及其所有权归属
  （阶段二“设备帧直通”）留待后续在真实设备帧场景下复采。

探针编译命令：

```bash
cc -o /tmp/avprobe /tmp/avprobe.c \
  -I/opt/ffmpeg-4.4.1/ascend/include \
  -L/opt/ffmpeg-4.4.1/ascend/lib \
  -lavformat -lavcodec -lavutil -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib
/tmp/avprobe /opt/ffmpeg-4.4.1/ascend/sample_ascend.264
```

该契约满足 `docs/development/ascend310p3-adaptation-plan.md` 阶段一
“host NV12 + 一次 H2D”路径；阶段二设备帧直通需等设备帧探针确认。

## 复验命令汇总

上述全部命令按小节顺序执行一遍即为完整复验；任何值与上表不一致时，以复验
结果更新本文件并同步调整 `cmake/ascend_sdk.cmake` 的检查项。

## Issue #30 验收：Ascend 解码帧 → FrameSurface（2026-08-04）

`src/media/VideoDecoderAscend.{h,cc}`（`h264_ascend`/`h265_ascend` → host NV12
`FrameSurface`）+ 独立 smoke（`test/ascend310p3/ascend_decoder_smoke.cc`）。
全部验收在本机锁下执行，媒体后端统一使用 `/opt/ffmpeg-4.4.1`。

### 关键结论

- Gate 0 解码器按名称显式选择（`avcodec_find_decoder_by_name`），不支持的解码
  类型拒绝打开，无软件回退。
- 非 Gate 0 设备号在工厂处明确拒绝：`VideoDecoder::Create(device_id, ...)`
  在 `device_id != 0` 时返回 nullptr（invalid device address 拒绝路径）；打开/
  发送/接收失败统一附带 `av_strerror` 文本。
- 解码输出为 host NV12（`buf[0/1]` 有效 `AVBufferRef`，`hw_frames_ctx=NULL`），
  `FrameSurface` 的 `lifetime` 持有 `AVFrame`（`av_frame_move_ref`），平面/行距/
  尺寸保留在 `FrameSurface`；帧索引与时间戳（`SetFrameIndex`/`SetTimestamp`）
  同为 demux 包索引（`SendPacket` 把 pts/dts 设为 `frame_idx`）——引擎下游经
  `FrameInfoSave`/`FrameInfoGet` 用该索引还原流的真实时间戳与 stream identity，
  smoke 校验每个解码帧的索引/时间戳都能映射回已发送的视频流包。
- `h265_ascend` 的帧输出是异步的：文件流结束后若不做 EOS flush，硬件队列里
  的帧不会送达 `receive_frame`（只调 receive 会一直 EAGAIN）。`VideoDecoder`
  新增 `Flush()`（Ascend 实现为 `avcodec_send_packet(NULL)` 触发 HiMpi drain），
  smoke 在流结束后 flush + 轮询取帧。加 flush 后 h265 本地/RTSP 稳定 30/30。

> 注：本 issue 不改动既有 demux/reconnect 流程，验收以 smoke 直接驱动真实
> `VideoDecoderAscend` 类为准（310P3 主机上完整引擎构建尚未就绪）。引擎侧
> `AlgChannelDecode` 接入 h265 的每包 drain + EOS flush 属后续 issue。

### 样本与 RTSP 源

```bash
# 本地样本（已在 /tmp）
/opt/ffmpeg-4.4.1/ffmpeg -y -loglevel error -i /tmp/cosmo_baseline_h264.mp4 \
  -c copy -bsf:v h264_mp4toannexb -f h264 /tmp/cosmo_baseline_h264.264
/opt/ffmpeg-4.4.1/ffmpeg -y -loglevel error -i /tmp/cosmo_baseline_h265.mp4 \
  -c copy -bsf:v hevc_mp4toannexb -f hevc /tmp/cosmo_baseline_h265.h265

# 依赖零、stdlib-only 的 RTSP 测试源（TCP interleaved，与引擎 RtspDemuxStrategy 一致）
python3 test/ascend310p3/rtsp_test_source.py --file /tmp/cosmo_baseline_h265.h265 \
  --codec h265 --port 8554
```

主机上 mediamtx（docker 拉取被网络阻断）、VLC RTSP（仅 UDP 且 H.265 RTP 打
包/参数集不完整）均不可用，故用上述最小 RTSP 源验证 `--rtsp` 路径。

### smoke 构建（主机）

```bash
g++ -std=c++17 -O2 -I src -I 3rd/fmt-7.1.2/include \
  test/ascend310p3/ascend_decoder_smoke.cc \
  3rd/fmt-7.1.2/src/format.cc \
  src/media/VideoDecoder.cc src/media/VideoDecoderCreateAscend.cc \
  src/media/VideoDecoderAscend.cc src/media/VideoFrame.cc \
  src/media/PixelFormatUtils.cc \
  src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
  src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
  src/util/Thread.cc src/util/ThreadRegistry.cc src/util/TimeUtil.cc \
  -I/opt/ffmpeg-4.4.1/ascend/include \
  -L/opt/ffmpeg-4.4.1/ascend/lib -lavformat -lavcodec -lavutil \
  -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib -lpthread \
  -o /tmp/ascend_decoder_smoke
```

### 实测结果（310P3 主机，全部通过）

```text
$ /tmp/ascend_decoder_smoke --h264 /tmp/cosmo_baseline_h264.mp4 --h265 /tmp/cosmo_baseline_h265.mp4 --frames 30
gate0 decoders present: h264_ascend h265_ascend
error paths passed: unsupported codec and invalid device id rejected
source passed: /tmp/cosmo_baseline_h264.mp4 codec=h264_ascend frames=30 fmt=nv12 planes=2 pitch=320 rows=240 size=116318
source passed: /tmp/cosmo_baseline_h265.mp4 codec=h265_ascend frames=30 fmt=nv12 planes=2 pitch=320 rows=240 size=115358
ascend decoder smoke passed

$ /tmp/ascend_decoder_smoke --h265 /tmp/cosmo_baseline_h265.mp4 --frames 30   # 连跑 3 次均通过

$ /tmp/ascend_decoder_smoke --rtsp rtsp://127.0.0.1:8554/stream --frames 30   # h265 RTSP
source passed: rtsp://127.0.0.1:8554/stream codec=h265_ascend frames=30 fmt=nv12 planes=2 pitch=320 rows=240 size=115358

$ /tmp/ascend_decoder_smoke --rtsp rtsp://127.0.0.1:8554/stream --frames 30   # h264 RTSP（rtsp_test_source.py 切 h264）
source passed: rtsp://127.0.0.1:8554/stream codec=h264_ascend frames=30 fmt=nv12 planes=2 pitch=320 rows=240 size=116318
```

契约检查（`scripts/ascend_decoder_contract_check.sh`）在主机
`ASCEND_FFMPEG_PREFIX=/opt/ffmpeg-4.4.1/ascend` 下与本地（系统 FFmpeg 头）均通过：

```text
All tests passed (83 assertions in 5 test cases)
```
