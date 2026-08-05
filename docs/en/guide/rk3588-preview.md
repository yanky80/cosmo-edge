---
title: RK3588 Preview Operations
description: RK3588 packaging, deployment, zero-copy, diagnostics, and benchmark procedure.
prev:
  text: Build Guide
  link: /en/guide/build
next:
  text: Troubleshooting
  link: /en/guide/troubleshooting
---

# RK3588 Preview Operations

The RK3588 profile selects `data/resource/aiboxresource_rk3588` and installs
it as `resource/`. Configure it with external
`COSMO_RK3588_SDK_ROOT` and `COSMO_RK3588_SYSROOT`; CMake validates RKNN/RGA/
DRM headers, `librknnrt.so`, `librga.so`, `libdrm.so`, Rockchip FFmpeg shared
libraries, and `pkg-config` modules for DRM, MPP, and FFmpeg. SDK binaries are
never vendored. The package accepts `.rknn` models only.

The supported path is H.264/H.265 local file or RTSP → `h264_rkmpp`/
`hevc_rkmpp` → DRM PRIME NV12 DMA-BUF → RGA RGB letterbox into RKNN tensor
memory → RKNN inference → CPU YOLO26 raw-head postprocess. The zero-copy
boundary ends at the RKNN input tensor; outputs and postprocess are host-side.
Required device nodes are `/dev/mpp_service`, `/dev/rga`, and a usable DRM
card/render node under `/dev/dri`. The model contract is one NHWC RGB input and
six NCHW INT8 affine outputs.

Software decode/inference fallback, unsupported model families, MJPEG/USB in
this path, and RKNN output-tensor zero-copy are out of scope. Failures must be
explicit.

Troubleshoot with `ffmpeg -decoders`, `ls -l /dev/rga /dev/mpp_service /dev/dri`,
the DRM plane fd/offset/pitch/stride checks, and the RKNN one-input/six-output
and affine-quantization checks. The smoke proves no software fallback by
requiring `*_rkmpp`, `DRM_PRIME` DMA-BUF output, device-node access, and RGA
destination fd equal to the RKNN input fd.

Use [`docs/benchmarks/rk3588-preview/README.md`](../../benchmarks/rk3588-preview/README)
for the locked board procedure and reproducible 1/3-instance report. It
records decode, RGA, RKNN, postprocess, end-to-end throughput, and memory
fields without an FPS pass/fail gate. Release verification also runs unchanged
x86/Sophon builds/imports, Graph contracts, and existing tests.
