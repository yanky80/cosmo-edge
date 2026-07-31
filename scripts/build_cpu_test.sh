#!/bin/bash
set -e
# CPU-backend test build script — native x86_64 compilation with ONNX Runtime.
# Uses a separate build directory (build_cpu/) to avoid conflicts with aarch64 builds.

if [ -z $PROJECT_ROOT_PATH ]
then
    PROJECT_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

BUILD_DIR=${PROJECT_ROOT_PATH}/build_cpu

clean_external_project() {
    local name="$1"
    rm -rf "${BUILD_DIR}/${name}-prefix"
    rm -rf "${BUILD_DIR}/CMakeFiles/${name}.dir"
}

clean_external_project openssl_external
clean_external_project srs_external
rm -rf "${BUILD_DIR}/srs_source"





mkdir -p ${BUILD_DIR}

OPENSSL_SSL_LIB="${BUILD_DIR}/thirdparty_install/openssl/lib/libssl.so"
if [ -f "${OPENSSL_SSL_LIB}" ] && ! file "${OPENSSL_SSL_LIB}" | grep -q "x86-64"; then
    echo "Removing non-x86_64 OpenSSL from CPU build cache: ${OPENSSL_SSL_LIB}"
    rm -rf "${BUILD_DIR}/thirdparty_install/openssl"
    clean_external_project openssl_external
    clean_external_project curl_external
    clean_external_project event_external
fi

if [ -d "${BUILD_DIR}/tokenizers_source" ] &&
   [ ! -f "${BUILD_DIR}/thirdparty_install/tokenizers/lib/libtokenizers_c.a" ]; then
    echo "Removing stale tokenizers build cache from CPU build directory"
    clean_external_project tokenizers_external
fi

cd ${BUILD_DIR}

echo "Configuring with tests enabled (CPU backend)..."
echo "Requires: pkg-config and openh264 development package (for x86 realtime OSD H264 encoding)"
cmake   -DCMAKE_BUILD_TYPE=Release \
        -U CMAKE_TOOLCHAIN_FILE \
        -U CMAKE_C_COMPILER \
        -U CMAKE_CXX_COMPILER \
        -U COSMO_TARGET_ARCH \
        -U COSMO_NN_USE_SOPHON_BACKEND \
        -U COSMO_NN_USE_CPU_BACKEND \
        -U COSMO_MEDIA_USE_SOPHON_BACKEND \
        -U COSMO_MEDIA_USE_CPU_BACKEND \
        -DCOSMO_TARGET_PLATFORM=x86 \
        -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DBUILD_TESTS=ON \
        ..



# Symlink compile_commands.json to project root for IDE and static analysis tools
ln -sf "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT_PATH}/compile_commands.json" 2>/dev/null || true

echo "Building cosmo-tests (CPU backend)..."
cmake --build . --target cosmo-tests -j$(nproc)

echo ""
echo "Build complete: ${BUILD_DIR}/cosmo-tests"
