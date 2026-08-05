---
title: Build Guide
description: Confirmed build paths for target-platform profiles, x86 Docker, Sophon release packages, RK3588 SDK validation, CPU test builds, and docs.
prev:
  text: Documentation Home
  link: /en/
next:
  text: Deployment Guide
  link: /en/guide/deployment
---

# Build Guide

This page documents build paths that are confirmed and available in the repository.

> **💡 Docker Compose Version Note**
> This documentation uses the latest Docker Compose V2 command format (`docker compose`). If you are using an older Docker environment, please replace `docker compose` with the hyphenated `docker-compose` in all commands.

## Build Path Overview

| Target | Entry Point | Notes |
| --- | --- | --- |
| Target-platform profile validation | `bash scripts/test_target_platform_profiles.sh` | Verifies `x86`, `sophon`, `rk3588`, `ascend310p3`, legacy compatibility, and invalid combinations at CMake configure time. |
| x86 Docker runtime | `docker-compose.x86.yml` / `docker-compose.x86.windows.yml` | Starts the containerized development/runtime environment. |
| Sophon release package | `docker compose -f docker-compose.sophon.yml run --rm cosmo-sophon-package` | Creates the target-device release package. |
| RK3588 profile configure | `cmake -S . -B build_rk3588 -DCOSMO_TARGET_PLATFORM=rk3588 ...` | Validates an external RK SDK/sysroot without committing vendor binaries. |
| Ascend 310P3 profile configure | `cmake -S . -B build_ascend310p3 -DCOSMO_TARGET_PLATFORM=ascend310p3 ...` | Validates the external CANN SDK and system FFmpeg at configure time. |
| CPU test build | `scripts/build_cpu_test.sh` | Builds `cosmo-tests` for x86 CPU validation. |
| Documentation site | `npm ci` and `npm run docs:build` | Builds this VitePress site. |

## Target-Platform Profile

CosmoEdge now selects one static build profile with:

```bash
-DCOSMO_TARGET_PLATFORM=x86|sophon|rk3588|ascend310p3
```

The profile derives:

- target architecture and toolchain
- inference and media backend selection
- compile definitions and linked vendor/runtime dependencies
- model artifact metadata (`.onnx`, `.nn` / `.bmodel`, `.rknn`)
- default `RESOURCE_DIR` and package contents

Legacy inputs such as `COSMO_TARGET_ARCH` and the old CPU/Sophon backend toggles are still accepted for compatibility, but CMake warns and rejects conflicts.

## x86 Docker Development Runtime

These entry points are from:

- `docker-compose.x86.yml` (Linux)
- `docker-compose.x86.windows.yml` (Windows)
- `Dockerfile.x86`
- `scripts/build_cpu.sh`

Confirmed CMake parameters:

| Parameter | Value |
| --- | --- |
| `COSMO_TARGET_PLATFORM` | `x86` |
| `COSMO_ENABLE_OPENH264` | `ON` |
| `COSMO_DEV_MODE` | `ON` |
| `RESOURCE_DIR` | `data/resource/aiboxresource_x86` |

Linux:

```bash
docker compose -f docker-compose.x86.yml up -d --build
docker compose -f docker-compose.x86.yml ps
```

Windows (PowerShell/CMD):

```powershell
docker compose -f docker-compose.x86.windows.yml up -d --build
docker compose -f docker-compose.x86.windows.yml ps
```

After build:

- Web console available at `http://127.0.0.1:8080`.
- Release packages and build artifacts exported to `build_output/`.
- Runtime data stored in Docker volume `cosmo-x86-data`.
- Resource directory mounted to Docker volume `cosmo-x86-app-resource`.

## Sophon Release Package

```bash
docker compose -f docker-compose.sophon.yml run --rm cosmo-sophon-package
```

Windows PowerShell:

```powershell
.\scripts\build_sophon_package.ps1
```

This path is from:

- `docker-compose.sophon.yml`
- `scripts/build_sophon_package.ps1` (Windows: restores `.so` symlinks before building)
- `scripts/build.sh`

Confirmed behavior:

- Base image uses the pre-built GHCR image: `ghcr.io/cosmo-wander-ai/cosmo_edge-build-env_sophon:v1` (unified build environment, speeding up local start time).
- Builds with `scripts/build.sh -m data/resource/aiboxresource`.
- Exports the release package only (does not start services).
- Package output under `build_output/`.

Confirmed profile:

| Parameter | Value |
| --- | --- |
| `COSMO_TARGET_PLATFORM` | `sophon` |

## RK3588 Profile Configure

The RK3588 profile validates build inputs from an external SDK/sysroot and does not use vendored RK binaries from this repository.

```bash
cmake -S . -B build_rk3588 \
  -DCOSMO_TARGET_PLATFORM=rk3588 \
  -DCOSMO_RK3588_SDK_ROOT=/path/to/rk-sdk \
  -DCOSMO_RK3588_SYSROOT=/path/to/rk-sysroot
```

Configure-time checks cover:

- `rknn_api.h`, `RgaApi.h`, and `libdrm/drm.h`
- aarch64 shared libraries such as `librknnrt.so`, `librga.so`, `libdrm.so`, and FFmpeg libs
- required `pkg-config` modules for `libdrm`, `rockchip_mpp`, and FFmpeg

See the [RK3588 Preview Operations](rk3588-preview) guide for the package, runtime/device requirements, zero-copy boundary, diagnostics, and board benchmark.

## Ascend 310P3 Profile Configure

The Ascend 310P3 profile builds for x86_64 by default (locked test-host
baseline); the target architecture is parameterized with `COSMO_TARGET_ARCH`
and can be `aarch64` (Kunpeng servers or on-board SoC deployments). The
profile validates the external CANN toolkit and the custom Ascend FFmpeg at
configure time. No CANN, driver, firmware, or custom FFmpeg binaries are
committed to the repository.

x86_64 default build (behavior identical to the locked baseline):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # exports ASCEND_TOOLKIT_HOME
cmake -S . -B build_ascend310p3 \
  -DCOSMO_TARGET_PLATFORM=ascend310p3
```

aarch64 cross build (x86_64 host + aarch64 cross toolchain, FFmpeg via
sysroot):

```bash
cmake -S . -B build_ascend310p3_aarch64 \
  -DCOSMO_TARGET_PLATFORM=ascend310p3 \
  -DCOSMO_TARGET_ARCH=aarch64 \
  -DCOSMO_ASCEND_SDK_ROOT=/path/to/aarch64-cann \
  -DCOSMO_ASCEND_SYSROOT=/path/to/aarch64-sysroot
```

Native aarch64 builds (aarch64 host) use the host toolchain: pass
`COSMO_TARGET_ARCH=aarch64` and `COSMO_ASCEND_SDK_ROOT` pointing at an aarch64
CANN install; no sysroot or toolchain arguments are needed.

Configure-time checks cover:

- host/target combinations: native builds require host == target; cross builds
  only allow an x86_64 host with an aarch64 target
- `libascendcl.so` and `libacl_dvpp.so` ELF architecture matches the target
  (`X86-64` / `AArch64`)
- `include/acl/acl.h` and `include/acl/dvpp/hi_dvpp.h`
- `lib64/libascendcl.so` and `lib64/libacl_dvpp.so`

FFmpeg lookup order:

1. `COSMO_ASCEND_SYSROOT` (sysroot for cross builds and hermetic tests)
2. `COSMO_ASCEND_FFMPEG_ROOT` custom FFmpeg root (defaults to
   `/opt/ffmpeg-4.4.1/ascend` on x86_64, the locked test-host baseline)
3. system FFmpeg dev packages (multiarch fallback via
   `CMAKE_LIBRARY_ARCHITECTURE`)

CANN environment initialization stays with the vendor-provided `set_env.sh`;
the locked test-host software and media baseline is recorded in
[Ascend 310P3 Test-Host Baseline](../development/ascend310p3-test-host-baseline).
The aarch64 deployment baseline (aarch64 CANN, custom FFmpeg aarch64 path,
driver/firmware versions) lives in that document's 「aarch64 部署基线（占位）」
section until real-device collection.

## CPU Test Build

```bash
bash scripts/build_cpu_test.sh
```

This script configures CMake with the CPU backend and `BUILD_TESTS=ON`, producing:

```sh
build_cpu/cosmo-tests
```

Useful for smoke testing C++ compilation and packaging logic without a target edge device.

## Documentation Build

```bash
npm ci
npm run docs:build
```

The build output is generated under `docs/.vitepress/dist` and should not be committed.
