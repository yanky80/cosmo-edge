---
title: 构建指南
description: 目标平台 profile、x86 Docker、Sophon 发布包、RK3588 SDK 校验和 CPU 测试构建路径。
prev:
  text: 文档首页
  link: /
next:
  text: 部署指南
  link: /guide/deployment
---

# 构建指南

本文只记录当前仓库中已经确认的构建路径。历史文档或旧脚本中出现过、但当前仓库无法验证的路径，不作为公开支持路径。

> **💡 Docker Compose 版本提示**
> 本文档统一使用最新的 Docker Compose V2 命令格式 (`docker compose`)。如果你使用的是旧版 Docker 环境（如自带独立的 V1 插件），请将文中的 `docker compose` 替换为带横杠的 `docker-compose`。

## 构建路径总览

| 路径 | 用途 | 是否启动服务 | 输出 |
| --- | --- | --- | --- |
| `scripts/test_target_platform_profiles.sh` | 验证 `x86` / `sophon` / `rk3588` profile、旧参数兼容和非法组合 | 否 | 临时 CMake 配置目录 |
| x86 Docker 开发运行环境 | 首次体验、开发评估、生成 x86 发布包 | 是 | `build_output/` |
| Sophon 发布包构建 | 生成 aarch64/Sophon 部署包 | 否 | `build_output/` |
| RK3588 profile 配置 | 校验外部 RK SDK/sysroot | 否 | `build_rk3588/` |
| CPU 测试构建 | 构建 `cosmo-tests` | 否 | `build_cpu/cosmo-tests` |

## 目标平台 Profile

CosmoEdge 现在通过单一参数选择静态构建 profile：

```bash
-DCOSMO_TARGET_PLATFORM=x86|sophon|rk3588
```

该 profile 统一派生：

- 目标架构和 toolchain
- 推理/媒体后端
- 编译宏和链接依赖
- 模型制品元数据（`.onnx`、`.nn` / `.bmodel`、`.rknn`）
- 默认 `RESOURCE_DIR` 和打包内容

旧的 `COSMO_TARGET_ARCH` 和 CPU/Sophon backend 开关仍可兼容输入，但 CMake 会给出弃用警告，并在冲突时直接失败。

## x86 Docker 开发运行环境

Linux:

```bash
docker compose -f docker-compose.x86.yml up -d --build
```

Windows (PowerShell/CMD):

```powershell
docker compose -f docker-compose.x86.windows.yml up -d --build
```

该路径来自：

- `docker-compose.x86.yml` (Linux)
- `docker-compose.x86.windows.yml` (Windows)
- `Dockerfile.x86`
- `scripts/build_cpu.sh`

已确认构建参数：

| 参数 | 值 |
| --- | --- |
| `COSMO_TARGET_PLATFORM` | `x86` |
| `COSMO_ENABLE_OPENH264` | `ON` |
| `COSMO_DEV_MODE` | `ON` |
| `RESOURCE_DIR` | `data/resource/aiboxresource_x86` |

构建完成后：

- Web 控制台通过 `http://127.0.0.1:8080` 访问。
- 发布包和构建产物导出到 `build_output/`。
- 运行数据保存在 Docker volume `cosmo-x86-data`。
- 资源目录挂载到 Docker volume `cosmo-x86-app-resource`。

## Sophon 发布包构建

Linux / Bash：

```bash
docker compose -f docker-compose.sophon.yml run --rm cosmo-sophon-package
```

Windows PowerShell：

```powershell
.\scripts\build_sophon_package.ps1
```

该路径来自：

- `docker-compose.sophon.yml`
- `scripts/build_sophon_package.ps1`（Windows：构建前自动修复 `.so` 软链接）
- `scripts/build.sh`

已确认行为：

- 基础镜像使用预先构建的 GHCR 镜像：`ghcr.io/cosmo-wander-ai/cosmo_edge-build-env_sophon:v1`（统一的编译环境，加速了本地启动时间）。
- 使用 `scripts/build.sh -m data/resource/aiboxresource` 构建（生产包不启用 dev mode，故不传 `-t`）。
- 只导出发布包，不启动服务。
- 发布包导出到 `build_output/`。

已确认 profile：

| 参数 | 值 |
| --- | --- |
| `COSMO_TARGET_PLATFORM` | `sophon` |

## RK3588 Profile 配置

RK3588 profile 只校验外部 SDK/sysroot，不向仓库提交 RK 厂商二进制：

```bash
cmake -S . -B build_rk3588 \
  -DCOSMO_TARGET_PLATFORM=rk3588 \
  -DCOSMO_RK3588_SDK_ROOT=/path/to/rk-sdk \
  -DCOSMO_RK3588_SYSROOT=/path/to/rk-sysroot
```

配置阶段会检查：

- `rknn_api.h`、`RgaApi.h`、`libdrm/drm.h`
- `librknnrt.so`、`librga.so`、`libdrm.so` 和 FFmpeg 共享库
- `libdrm`、`rockchip_mpp`、FFmpeg 的 `pkg-config` 模块

MPP DRM PRIME 到 RKNN 输入 tensor 的零拷贝运行时链路仍在后续 RK3588 issue 中实现；本 issue 只建立 profile 和 SDK/sysroot 校验面。

## CPU 测试构建

```bash
bash scripts/build_cpu_test.sh
```

该脚本会使用 CPU 后端配置 CMake，并开启：

```text
BUILD_TESTS=ON
```

目标产物：

```text
build_cpu/cosmo-tests
```
