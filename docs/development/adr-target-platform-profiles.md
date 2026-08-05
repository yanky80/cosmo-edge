# ADR: Static Target-Platform Profiles

## Status

Accepted on July 31, 2026.

## Context

CosmoEdge now targets four build profiles: `x86`, `sophon`, `rk3588`, and `ascend310p3`. The old build inputs mixed architecture, inference backend, media backend, and resource selection across multiple CMake cache flags. That made it easy to create illegal combinations and leaked vendor choices into the build entrypoints.

## Decision

- CosmoEdge selects one static target platform with `COSMO_TARGET_PLATFORM=x86|sophon|rk3588|ascend310p3`.
- Each target platform derives one inference backend, one media backend, one default resource root, and one model-artifact profile. The architecture/toolchain is derived per platform: ascend310p3 parameterizes it with `COSMO_TARGET_ARCH` (`x86_64` default, `aarch64` via cross or native toolchain), while the other profiles pin one architecture.
- Media and inference backends remain independent concepts, but each static platform profile pins one supported pair.
- One model package carries artifacts for exactly one target platform.
- CosmoEdge does not add runtime plugin loading for vendor backends.

## Consequences

- Existing x86 and Sophon entrypoints become thinner because they only choose a target platform and optional resource overrides.
- Legacy backend toggles remain accepted for compatibility, but CMake warns and rejects conflicts.
- RK3588 configuration validates external SDK/sysroot inputs at configure time without committing vendor binaries into the repository.
- Ascend 310P3 configuration validates the external CANN toolkit (AscendCL/DVPP headers and libraries whose ELF architecture matches the target: `X86-64` or `AArch64`) and the custom Ascend FFmpeg (falling back to system FFmpeg) at configure time, without committing CANN, driver, firmware, or custom FFmpeg binaries. The target architecture is parameterized with `COSMO_TARGET_ARCH` (`x86_64` default, `aarch64` supported); native builds require host == target and cross builds allow an x86_64 host with an aarch64 target.
- Follow-up RK3588 runtime issues can add source implementations behind the profile without reopening the public build interface.
