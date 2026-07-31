#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
TMP_DIR=$(mktemp -d)
HOST_CC=$(command -v cc)
HOST_CXX=$(command -v c++)
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
    local file="$1"
    local needle="$2"
    if ! grep -Fq "$needle" "$file"; then
        echo "Expected '$needle' in $file" >&2
        exit 1
    fi
}

run_configure() {
    local name="$1"
    shift
    cmake -S "$ROOT_DIR" -B "$TMP_DIR/$name" "$@" >"$TMP_DIR/$name.out" 2>"$TMP_DIR/$name.err"
}

run_configure_fail() {
    local name="$1"
    shift
    if cmake -S "$ROOT_DIR" -B "$TMP_DIR/$name" "$@" >"$TMP_DIR/$name.out" 2>"$TMP_DIR/$name.err"; then
        echo "Expected configure failure for $name" >&2
        exit 1
    fi
}

create_fake_rk_sdk() {
    local sdk_root="$TMP_DIR/rk-sdk"
    local sysroot="$TMP_DIR/rk-sysroot"
    mkdir -p \
        "$sdk_root/include/rga" \
        "$sdk_root/lib" \
        "$sysroot/usr/include/libdrm" \
        "$sysroot/usr/lib/pkgconfig"
    touch \
        "$sdk_root/include/rknn_api.h" \
        "$sdk_root/include/rga/RgaApi.h" \
        "$sysroot/usr/include/libdrm/drm.h" \
        "$sdk_root/lib/librknnrt.so" \
        "$sdk_root/lib/librga.so" \
        "$sysroot/usr/lib/libdrm.so" \
        "$sysroot/usr/lib/libavcodec.so" \
        "$sysroot/usr/lib/libavdevice.so" \
        "$sysroot/usr/lib/libavfilter.so" \
        "$sysroot/usr/lib/libavformat.so" \
        "$sysroot/usr/lib/libavutil.so" \
        "$sysroot/usr/lib/libswresample.so" \
        "$sysroot/usr/lib/libswscale.so"
    cat >"$TMP_DIR/pkg-config" <<'EOF'
#!/bin/sh
if [ "$1" = "--exists" ]; then
  exit 0
fi
exit 1
EOF
    chmod +x "$TMP_DIR/pkg-config"
    RK_SDK_ROOT="$sdk_root"
    RK_SYSROOT="$sysroot"
}

create_fake_rk_sdk

run_configure x86 -DCOSMO_TARGET_PLATFORM=x86
assert_contains "$TMP_DIR/x86/CMakeCache.txt" "COSMO_TARGET_PLATFORM:STRING=x86"
assert_contains "$TMP_DIR/x86/CMakeCache.txt" "COSMO_TARGET_ARCH:STRING=x86_64"
assert_contains "$TMP_DIR/x86/CMakeCache.txt" "COSMO_NN_USE_CPU_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/x86/CMakeCache.txt" "COSMO_MEDIA_USE_CPU_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/x86/CMakeCache.txt" "RESOURCE_DIR:PATH=${ROOT_DIR}/data/resource/aiboxresource_x86"

run_configure sophon \
    -DCOSMO_TARGET_PLATFORM=sophon \
    -DCMAKE_C_COMPILER="$HOST_CC" \
    -DCMAKE_CXX_COMPILER="$HOST_CXX"
assert_contains "$TMP_DIR/sophon/CMakeCache.txt" "COSMO_TARGET_PLATFORM:STRING=sophon"
assert_contains "$TMP_DIR/sophon/CMakeCache.txt" "COSMO_TARGET_ARCH:STRING=aarch64"
assert_contains "$TMP_DIR/sophon/CMakeCache.txt" "COSMO_NN_USE_SOPHON_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/sophon/CMakeCache.txt" "COSMO_MEDIA_USE_SOPHON_BACKEND:BOOL=ON"

run_configure legacy-x86 \
    -DCOSMO_TARGET_PLATFORM=x86 \
    -DCOSMO_TARGET_ARCH=x86_64 \
    -DCOSMO_NN_USE_CPU_BACKEND=ON \
    -DCOSMO_NN_USE_SOPHON_BACKEND=OFF \
    -DCOSMO_MEDIA_USE_CPU_BACKEND=ON \
    -DCOSMO_MEDIA_USE_SOPHON_BACKEND=OFF
assert_contains "$TMP_DIR/legacy-x86.err" "deprecated"

run_configure rk3588 \
    -DCOSMO_TARGET_PLATFORM=rk3588 \
    -DCMAKE_C_COMPILER="$HOST_CC" \
    -DCMAKE_CXX_COMPILER="$HOST_CXX" \
    -DCOSMO_RK3588_SDK_ROOT="$RK_SDK_ROOT" \
    -DCOSMO_RK3588_SYSROOT="$RK_SYSROOT" \
    -DCOSMO_PKG_CONFIG_EXECUTABLE="$TMP_DIR/pkg-config"
assert_contains "$TMP_DIR/rk3588/CMakeCache.txt" "COSMO_TARGET_PLATFORM:STRING=rk3588"
assert_contains "$TMP_DIR/rk3588/CMakeCache.txt" "COSMO_NN_USE_RKNN_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/rk3588/CMakeCache.txt" "COSMO_MEDIA_USE_RK3588_BACKEND:BOOL=ON"

run_configure_fail conflict \
    -DCOSMO_TARGET_PLATFORM=x86 \
    -DCOSMO_NN_USE_SOPHON_BACKEND=ON
assert_contains "$TMP_DIR/conflict.err" "conflicts"

run_configure_fail rk-missing-sdk \
    -DCOSMO_TARGET_PLATFORM=rk3588 \
    -DCMAKE_C_COMPILER="$HOST_CC" \
    -DCMAKE_CXX_COMPILER="$HOST_CXX"
assert_contains "$TMP_DIR/rk-missing-sdk.err" "COSMO_RK3588_SDK_ROOT"

echo "Target platform profile configure tests passed."
