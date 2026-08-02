# RK3588 Resource Root

This resource root is selected by the `COSMO_TARGET_PLATFORM=rk3588` package
profile and installed as the package `resource/` directory.

The preview package contains the YOLO26 detector template and platform-neutral
resource metadata. Runtime SDK/sysroot libraries remain external; model files
must be `.rknn` files matching the one-input/six-output RKNN contract described
in `docs/guide/rk3588-preview.md`.
