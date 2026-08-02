---
title: RK3588 Preview Operations
description: RK3588 package, deployment, zero-copy, diagnostics, and benchmark procedure.
prev:
  text: 构建指南
  link: /guide/build
next:
  text: 故障排查
  link: /guide/troubleshooting
---

# RK3588 Preview Operations

## Package and external dependencies

Configure with `-DCOSMO_TARGET_PLATFORM=rk3588` and an external
`COSMO_RK3588_SDK_ROOT` plus `COSMO_RK3588_SYSROOT`. The profile selects
`data/resource/aiboxresource_rk3588`, installs it as `resource/`, and accepts
`.rknn` models only. CMake checks `rknn_api.h`, `RgaApi.h`, `libdrm/drm.h`,
`librknnrt.so`, `librga.so`, `libdrm.so`, the Rockchip FFmpeg shared libraries,
and the `pkg-config` modules `libdrm`, `rockchip_mpp`, `libavcodec`,
`libavformat`, and `libavutil`.

SDK and sysroot binaries are external inputs; no vendor SDK binary belongs in
this repository or release source tree.

## Runtime contract

The supported path is H.264/H.265 local file or RTSP → `h264_rkmpp`/
`hevc_rkmpp` → DRM PRIME NV12 DMA-BUF → RGA RGB letterbox into RKNN tensor
memory → RKNN inference → CPU YOLO26 raw-head postprocess. The zero-copy
boundary ends at the RKNN input tensor: decoded DMA-BUF is not copied through
host RGB memory, while RKNN outputs and postprocess are host-visible.

Required device access is `/dev/mpp_service`, `/dev/rga`, and a usable DRM
card/render node under `/dev/dri`. The model package is a resource directory
with `model_template/yolo26_det.json` and a model entry whose file is `.rknn`;
the RKNN contract is one NHWC RGB input and six NCHW INT8 affine outputs.

Unsupported in this preview: software decode or inference fallback, MJPEG/USB
camera input in the RK zero-copy path, other model families, and RKNN output
tensor-memory zero-copy. A failure is preferable to silently changing paths.

## Diagnostics and troubleshooting

- Missing `h264_rkmpp` or `hevc_rkmpp`: install/use the board's external
  ffmpeg-rockchip runtime and verify `ffmpeg -decoders`. The engine must fail;
  it must not select a software decoder.
- RGA access: check `ls -l /dev/rga` and process permissions. RGA import or
  conversion errors are fatal.
- DRM plane validation: verify a DRM PRIME frame, NV12, two planes, valid fd,
  offset, pitch, and vertical stride. Invalid descriptors are rejected.
- RKNN tensor binding: verify one input, six outputs, affine scale/zero-point,
  and that RGA's destination fd equals the RKNN input fd. A model with one or
  nine outputs is not this preview model format.
- Prove no software fallback by checking the selected `*_rkmpp` decoder, the
  `DRM_PRIME`/DMA-BUF assertion, `/dev/mpp_service` and `/dev/rga` access, and
  the source/destination fd provenance in the smoke output.

## Board check and benchmark

Run the checks in [`docs/benchmarks/rk3588-preview/README.md`](../benchmarks/rk3588-preview/)
under the shared `flock` lock. Record board identity, fixed model/video,
decode/RGA/RKNN/postprocess status, end-to-end throughput, memory sampling,
and the 1/3-instance result. The report has no FPS gate.

Before release, also run the unchanged x86 and Sophon profile/configure paths,
model imports, Graph contract tests, and existing unit/test suites. See
[`docs/guide/build.md`](build) and [`docs/guide/test-cases.md`](test-cases).
