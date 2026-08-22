#!/bin/bash
set -e
export LC_ALL=C.UTF-8

# Native x86_64 build for a Huawei Ascend 310P3 card.
RESOURCE_DIR=""
DEV_MODE=OFF
BUILD_TESTS_FLAG=OFF
while getopts "m:tTh" opt; do
    case $opt in
        m) RESOURCE_DIR="$OPTARG" ;;
        t) DEV_MODE=ON ;;
        T) BUILD_TESTS_FLAG=ON ;;
        h) echo "Usage: $0 [-m <resource_repo_path>] [-t (enable dev mode)] [-T (also build cosmo-tests)]"; exit 0 ;;
        *) echo "Usage: $0 [-m <resource_repo_path>] [-t (enable dev mode)] [-T (also build cosmo-tests)]"; exit 1 ;;
    esac
done

if [ -z "${PROJECT_ROOT_PATH:-}" ]; then
    PROJECT_ROOT_PATH=$(cd "$(dirname "$0")/.." && pwd)
fi

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    . /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if [ -z "${RESOURCE_DIR}" ]; then
    RESOURCE_DIR="${PROJECT_ROOT_PATH}/data/resource/aiboxresource_ascend310p3"
elif [ "${RESOURCE_DIR#/}" = "${RESOURCE_DIR}" ]; then
    RESOURCE_DIR="${PROJECT_ROOT_PATH}/${RESOURCE_DIR}"
fi

if [ ! -d "${RESOURCE_DIR}" ]; then
    echo "ERROR: Resource directory not found: ${RESOURCE_DIR}" >&2
    exit 1
fi

BUILD_DIR="${PROJECT_ROOT_PATH}/build_ascend310p3"
INSTALL_DIR="${BUILD_DIR}/install"
rm -rf "${INSTALL_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Dev mode: ${DEV_MODE}"
echo "Backend: Huawei Ascend 310P3 (CANN/DVPP)"
echo "Resource dir: ${RESOURCE_DIR}"
echo "Configuring..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -U CMAKE_TOOLCHAIN_FILE \
      -U CMAKE_C_COMPILER \
      -U CMAKE_CXX_COMPILER \
      -U COSMO_TARGET_ARCH \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DCOSMO_TARGET_PLATFORM=ascend310p3 \
      -DCOSMO_TARGET_ARCH=x86_64 \
      -DBUILD_TESTS="${BUILD_TESTS_FLAG}" \
      -DCMAKE_C_COMPILER=/usr/bin/cc \
      -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
      -DCOSMO_DEV_MODE="${DEV_MODE}" \
      -DRESOURCE_DIR="${RESOURCE_DIR}" \
      ..

ln -sf "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT_PATH}/compile_commands.json" 2>/dev/null || true

echo "Building Cosmo (Ascend 310P3 backend) ..."
build_targets=(--target install)
if [ "${BUILD_TESTS_FLAG}" = "ON" ]; then
    build_targets+=(--target cosmo-tests)
fi
cmake --build . "${build_targets[@]}" -j"$(nproc)"

echo "Packaging..."
cmake --build . --target package_all
echo "Build complete: ${INSTALL_DIR}/bin/cosmo-engine"
