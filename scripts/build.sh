#!/bin/bash
export LC_ALL=C.UTF-8

# ── Parse options ──
# -t = dev mode (disable watchdog); -T = also build cosmo-tests in this pass.
RESOURCE_DIR=""
DEV_MODE=OFF
BUILD_TESTS_FLAG=OFF
while getopts "m:tT" opt; do
    case $opt in
        m) RESOURCE_DIR="$OPTARG" ;;
        t) DEV_MODE=ON ;;
        T) BUILD_TESTS_FLAG=ON ;;
        *) echo "Usage: $0 [-m <resource_repo_path>] [-t (enable dev mode)] [-T (also build cosmo-tests)]"; exit 1 ;;
    esac
done

if [ -z "${PROJECT_ROOT_PATH:-}" ]
then
    PROJECT_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

if [ -z "${RESOURCE_DIR}" ]; then
    RESOURCE_DIR="${PROJECT_ROOT_PATH}/data/resource/aiboxresource"
elif [ "${RESOURCE_DIR#/}" = "${RESOURCE_DIR}" ]; then
    RESOURCE_DIR="${PROJECT_ROOT_PATH}/${RESOURCE_DIR}"
fi

if [ ! -d "${RESOURCE_DIR}" ]; then
    echo "ERROR: Resource directory not found: ${RESOURCE_DIR}" >&2
    exit 1
fi

BUILD_DIR=${PROJECT_ROOT_PATH}/build
INSTALL_DIR=${BUILD_DIR}/install

if [ -d ${INSTALL_DIR} ]
then 
    rm -rf ${INSTALL_DIR}
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "Dev mode: ${DEV_MODE}"
echo "Resource dir: ${RESOURCE_DIR}"
echo "Configuring..."
cmake   -DCMAKE_BUILD_TYPE=Release \
        -U CMAKE_C_COMPILER \
        -U CMAKE_CXX_COMPILER \
        -U COSMO_TARGET_ARCH \
        -U COSMO_NN_USE_SOPHON_BACKEND \
        -U COSMO_NN_USE_CPU_BACKEND \
        -U COSMO_MEDIA_USE_SOPHON_BACKEND \
        -U COSMO_MEDIA_USE_CPU_BACKEND \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
        -DCOSMO_TARGET_PLATFORM=sophon \
        -DBUILD_TESTS=${BUILD_TESTS_FLAG} \
        -DCOSMO_DEV_MODE=${DEV_MODE} \
        -DRESOURCE_DIR="${RESOURCE_DIR}" \
        ..

# Symlink compile_commands.json to project root for IDE and static analysis tools
ln -sf "${BUILD_DIR}/compile_commands.json" "${PROJECT_ROOT_PATH}/compile_commands.json" 2>/dev/null || true

echo "Building Cosmo ..."
build_targets=(--target install)
if [ "${BUILD_TESTS_FLAG}" = "ON" ]; then
    echo "Also building cosmo-tests in this pass..."
    build_targets+=(--target cosmo-tests)
fi
cmake --build . "${build_targets[@]}" -j$(nproc)

echo "Packaging..."
cmake --build . --target package_all
