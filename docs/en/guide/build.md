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
| Target-platform profile validation | `bash scripts/test_target_platform_profiles.sh` | Verifies `x86`, `sophon`, `rk3588`, legacy compatibility, and invalid combinations at CMake configure time. |
| x86 Docker runtime | `docker-compose.x86.yml` / `docker-compose.x86.windows.yml` | Starts the containerized development/runtime environment. |
| Sophon release package | `docker compose -f docker-compose.sophon.yml run --rm cosmo-sophon-package` | Creates the target-device release package. |
| RK3588 profile configure | `cmake -S . -B build_rk3588 -DCOSMO_TARGET_PLATFORM=rk3588 ...` | Validates an external RK SDK/sysroot without committing vendor binaries. |
| CPU test build | `scripts/build_cpu_test.sh` | Builds `cosmo-tests` for x86 CPU validation. |
| Documentation site | `npm ci` and `npm run docs:build` | Builds this VitePress site. |

## Target-Platform Profile

CosmoEdge now selects one static build profile with:

```bash
-DCOSMO_TARGET_PLATFORM=x86|sophon|rk3588
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

The zero-copy runtime path is still implemented in follow-up RK3588 issues; this issue only introduces the profile and SDK/sysroot validation seam.

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
