---
title: Ascend 310P3 Preview Operations
description: Ascend 310P3 package, deployment, environment, diagnostics, and benchmark procedure.
prev:
  text: Build Guide
  link: /en/guide/build
next:
  text: Troubleshooting
  link: /en/guide/troubleshooting
---

# Ascend 310P3 Preview Operations

## Package and external dependencies

Configure with `-DCOSMO_TARGET_PLATFORM=ascend310p3`. The profile selects
`data/resource/aiboxresource_ascend310p3`, installs it as `resource/`, and
accepts `.om` models only. CMake locates `acl/acl.h`, `acl/dvpp/hi_dvpp.h`,
`libascendcl.so`, and `libacl_dvpp*.so` from the external CANN toolkit, and
the custom Ascend FFmpeg under `/opt/ffmpeg-4.4.1/ascend`
(`h264_ascend`/`h265_ascend`); a library whose ELF architecture does not
match the target fails at configure time. CANN, driver, firmware, and FFmpeg
are external inputs; no vendor binary belongs in the repository or the
release package.

## Runtime contract

Supported path: H.264/H.265 local file or RTSP → `h264_ascend`/`h265_ascend`
hardware decode → DVPP `image_to_tensor` (centered letterbox to 960x960,
NV12→RGB, host /255 + FP16) → AscendCL inference (FP16 OM, input
`[1,3,960,960]`, output `[1,300,6]` end2end) → host `yolo_e2e` coordinate
restore. The decoder emits device frames (`AV_PIX_FMT_ASCEND`) by default and
inference consumes them directly with no H2D upload; preview/capture download
to host on demand. There is no software decode, CPU resize, or ONNX Runtime
fallback; failures are surfaced, never silently downgraded.

The model package is a resource directory plus an `.om` entry
(`model_template/yolo26_det.json` template, `chip_type: ASCEND310P3`); import
validates one batch-1 input and a `[1,300,6]` output via AscendCL metadata.

## Diagnostics and troubleshooting

- Missing `h264_ascend`/`h265_ascend`: check the custom FFmpeg at
  `/opt/ffmpeg-4.4.1/ascend` (`ffmpeg -decoders | grep ascend`). The engine
  must fail; it must not pick a software decoder.
- Device check: `npu-smi info` Health OK, driver/firmware versions,
  `/usr/local/Ascend/driver/version.info`.
- Library/ELF issues: configure-time errors are described in
  `docs/development/ascend310p3-native-package.md`.
- Model contract: a bad OM or wrong shape fails in `Graph::Init`; logs carry
  the failing stage, device id, model path, tensor description, and ACL error
  code.
- Stream switch/hang: the VDEC channel must not be closed while a VPC channel
  is alive; switching streams requires a task restart.

## Validation and benchmarks

Full validation, the 100-frame ONNX FP32 comparison, 1/3-instance 30-minute
soaks, and per-stage benchmarks live in
[`docs/benchmarks/ascend310p3-preview/README.md`](../../benchmarks/ascend310p3-preview/README).
All hardware commands run under the shared `flock` lock and are recorded with
their outputs in the pull request. The report has no FPS gate.

Before release, also run the unchanged x86/Sophon profile configure paths,
model imports, Graph contract tests, and existing unit suites. See
[`docs/guide/build.md`](build) and [`docs/guide/test-cases.md`](test-cases).
