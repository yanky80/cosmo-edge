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

create_fake_ascend_sdk() {
    local sdk_root="$TMP_DIR/ascend-sdk"
    local sysroot="$TMP_DIR/ascend-sysroot"
    local dummy="$TMP_DIR/ascend-dummy.c"
    mkdir -p \
        "$sdk_root/include/acl/dvpp" \
        "$sdk_root/lib64" \
        "$sysroot/usr/include/libavcodec" \
        "$sysroot/usr/lib/x86_64-linux-gnu"
    touch \
        "$sdk_root/include/acl/acl.h" \
        "$sdk_root/include/acl/dvpp/hi_dvpp.h"
    # ACL libs must be real x86_64 ELF files: the SDK check verifies the
    # library architecture with readelf at configure time.
    printf 'int cosmo_fake_ascendcl_dummy;\n' >"$dummy"
    "$HOST_CC" -shared -fPIC "$dummy" -o "$sdk_root/lib64/libascendcl.so"
    "$HOST_CC" -shared -fPIC "$dummy" -o "$sdk_root/lib64/libacl_dvpp.so"
    touch \
        "$sysroot/usr/include/libavcodec/avcodec.h" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libavcodec.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libavdevice.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libavfilter.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libavformat.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libavutil.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libswresample.so" \
        "$sysroot/usr/lib/x86_64-linux-gnu/libswscale.so"
    ASCEND_SDK_ROOT="$sdk_root"
    ASCEND_SYSROOT="$sysroot"
}

create_fake_ascend_sdk

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

run_configure ascend310p3 \
    -DCOSMO_TARGET_PLATFORM=ascend310p3 \
    -DCOSMO_ASCEND_SDK_ROOT="$ASCEND_SDK_ROOT" \
    -DCOSMO_ASCEND_SYSROOT="$ASCEND_SYSROOT"
assert_contains "$TMP_DIR/ascend310p3/CMakeCache.txt" "COSMO_TARGET_PLATFORM:STRING=ascend310p3"
assert_contains "$TMP_DIR/ascend310p3/CMakeCache.txt" "COSMO_TARGET_ARCH:STRING=x86_64"
assert_contains "$TMP_DIR/ascend310p3/CMakeCache.txt" "COSMO_NN_USE_ASCEND_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/ascend310p3/CMakeCache.txt" "COSMO_MEDIA_USE_ASCEND_BACKEND:BOOL=ON"
assert_contains "$TMP_DIR/ascend310p3/CMakeCache.txt" "RESOURCE_DIR:PATH=${ROOT_DIR}/data/resource/aiboxresource_ascend310p3"
assert_contains "$TMP_DIR/ascend310p3.out" "Model ext: .om"
assert_contains "$TMP_DIR/ascend310p3.out" "ASCEND310P3"

run_configure_fail ascend-missing-sdk \
    -DCOSMO_TARGET_PLATFORM=ascend310p3 \
    -DCOSMO_ASCEND_SYSROOT="$ASCEND_SYSROOT"
assert_contains "$TMP_DIR/ascend-missing-sdk.err" "COSMO_ASCEND_SDK_ROOT"

bad_arch_sdk="$TMP_DIR/ascend-badarch"
mkdir -p "$bad_arch_sdk/include/acl/dvpp" "$bad_arch_sdk/lib64"
touch \
    "$bad_arch_sdk/include/acl/acl.h" \
    "$bad_arch_sdk/include/acl/dvpp/hi_dvpp.h" \
    "$bad_arch_sdk/lib64/libascendcl.so" \
    "$bad_arch_sdk/lib64/libacl_dvpp.so"
run_configure_fail ascend-bad-arch \
    -DCOSMO_TARGET_PLATFORM=ascend310p3 \
    -DCOSMO_ASCEND_SDK_ROOT="$bad_arch_sdk" \
    -DCOSMO_ASCEND_SYSROOT="$ASCEND_SYSROOT"
assert_contains "$TMP_DIR/ascend-bad-arch.err" "x86_64 ELF"

echo "Target platform profile configure tests passed."
