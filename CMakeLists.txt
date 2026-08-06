cmake_minimum_required(VERSION 3.16)

if(POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "FATAL: In-source builds are not allowed. You should create a separate directory for build files.")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
endif()

include(cmake/target_platform.cmake)
set_property(CACHE COSMO_TARGET_ARCH PROPERTY STRINGS aarch64 x86_64)

project(Cosmo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#Generate compile_commands.json for static analysis tools(clang - tidy, IDE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Baseline : suppress noisy warnings from third - party headers(global scope)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wno-deprecated-declarations -Wno-ignored-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-ignored-attributes")

#Strict warnings applied only to project OBJECT libraries(not third - party)
set(COSMO_STRICT_WARNINGS
    -Wall
    -Wextra
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wpedantic
    # -Wconversion
    # -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wmisleading-indentation
    -Wduplicated-cond
    -Wduplicated-branches
    -Wlogical-op
)

option(BUILD_TESTS "Build unit tests" OFF)
option(COSMO_ENABLE_COVERAGE "Enable gcov code-coverage instrumentation for the test build (adds --coverage; defines the 'coverage' lcov target)" OFF)
if(BUILD_TESTS AND COSMO_ENABLE_COVERAGE)
    message(STATUS "Enabling code coverage for tests")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

option(COSMO_DEV_MODE "Development mode: disable watchdog and other production-only features" OFF)
if(COSMO_DEV_MODE)
    message(STATUS "Development mode enabled (watchdog disabled)")
    add_compile_definitions(COSMO_DEV_MODE)
endif()

if(COSMO_TARGET_PLATFORM STREQUAL "x86")
    add_compile_definitions(COSMO_TARGET_PLATFORM_X86)
elseif(COSMO_TARGET_PLATFORM STREQUAL "sophon")
    add_compile_definitions(COSMO_TARGET_PLATFORM_SOPHON)
elseif(COSMO_TARGET_PLATFORM STREQUAL "ascend310p3")
    add_compile_definitions(COSMO_TARGET_PLATFORM_ASCEND310P3)
else()
    add_compile_definitions(COSMO_TARGET_PLATFORM_RK3588)
endif()

#-- Backend selection(mutually exclusive) --
set(_cosmo_nn_backend_count 0)
foreach(_nn_backend_option
        COSMO_NN_USE_SOPHON_BACKEND
        COSMO_NN_USE_CPU_BACKEND
        COSMO_NN_USE_RKNN_BACKEND
        COSMO_NN_USE_ASCEND_BACKEND)
    if(${_nn_backend_option})
        math(EXPR _cosmo_nn_backend_count "${_cosmo_nn_backend_count} + 1")
    endif()
endforeach()
if(NOT _cosmo_nn_backend_count EQUAL 1)
    message(FATAL_ERROR "Exactly one NN backend must be enabled: Sophon, CPU, RKNN, or Ascend")
endif()

set(_cosmo_media_backend_count 0)
foreach(_media_backend_option
        COSMO_MEDIA_USE_SOPHON_BACKEND
        COSMO_MEDIA_USE_CPU_BACKEND
        COSMO_MEDIA_USE_RK3588_BACKEND
        COSMO_MEDIA_USE_ASCEND_BACKEND)
    if(${_media_backend_option})
        math(EXPR _cosmo_media_backend_count "${_cosmo_media_backend_count} + 1")
    endif()
endforeach()
if(NOT _cosmo_media_backend_count EQUAL 1)
    message(FATAL_ERROR "Exactly one media backend must be enabled: Sophon, CPU, RK3588, or Ascend")
endif()
if(COSMO_MEDIA_USE_SOPHON_BACKEND AND NOT COSMO_NN_USE_SOPHON_BACKEND)
    message(FATAL_ERROR "Sophon media requires the Sophon device runtime")
endif()
if(COSMO_MEDIA_USE_RK3588_BACKEND AND NOT COSMO_NN_USE_RKNN_BACKEND)
    message(FATAL_ERROR "RK3588 media requires the RKNN device runtime")
endif()
if(COSMO_MEDIA_USE_ASCEND_BACKEND AND NOT COSMO_NN_USE_ASCEND_BACKEND)
    message(FATAL_ERROR "Ascend media requires the Ascend device runtime")
endif()

if(COSMO_MEDIA_USE_SOPHON_BACKEND)
    add_compile_definitions(COSMO_MEDIA_USE_SOPHON_BACKEND)
elseif(COSMO_MEDIA_USE_RK3588_BACKEND)
    add_compile_definitions(COSMO_MEDIA_USE_RK3588_BACKEND)
elseif(COSMO_MEDIA_USE_ASCEND_BACKEND)
    add_compile_definitions(COSMO_MEDIA_USE_ASCEND_BACKEND)
else()
    add_compile_definitions(COSMO_MEDIA_USE_CPU_BACKEND)
endif()

if(COSMO_MEDIA_USE_CPU_BACKEND)
    set(COSMO_ENABLE_OPENH264 ON)
    set(COSMO_OPENH264_USE_ASM OFF)
else()
    set(COSMO_ENABLE_OPENH264 OFF)
    set(COSMO_OPENH264_USE_ASM OFF)
endif()

if(COSMO_NN_USE_SOPHON_BACKEND)
    set(COSMO_MODEL_GUARD ON)
else()
    set(COSMO_MODEL_GUARD OFF)
endif()

if(COSMO_NN_USE_RKNN_BACKEND)
    add_compile_definitions(COSMO_NN_USE_RKNN_BACKEND)
elseif(COSMO_NN_USE_SOPHON_BACKEND)
    add_compile_definitions(COSMO_NN_USE_SOPHON_BACKEND)
elseif(COSMO_NN_USE_ASCEND_BACKEND)
    add_compile_definitions(COSMO_NN_USE_ASCEND_BACKEND)
else()
    add_compile_definitions(COSMO_NN_USE_CPU_BACKEND)
endif()

if(COSMO_MODEL_GUARD)
    # .so is pre-installed in system image at /usr/lib/ or provided in prebuild/
    find_library(MODEL_GUARD_LIB cosmo_model_guard
                 PATHS /usr/lib ${CMAKE_SOURCE_DIR}/prebuild
                 NO_DEFAULT_PATH)
    if(MODEL_GUARD_LIB)
        message(STATUS "Model guard enabled: ${MODEL_GUARD_LIB}")
    else()
        message(WARNING "COSMO_MODEL_GUARD=ON but libcosmo_model_guard.so not found — "
                        "encrypted models will fail at runtime")
    endif()
endif()

message(STATUS "=======================")
message(STATUS "Cosmo Build Info:")
message(STATUS "\tSystem:    ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tCompiler:  ${CMAKE_CXX_COMPILER}, Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "\tToolchain: ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "\tPlatform:  ${COSMO_TARGET_PLATFORM}")
message(STATUS "\tTarget:    ${COSMO_TARGET_ARCH}")
message(STATUS "\tResource:  ${RESOURCE_DIR}")
message(STATUS "\tModel ext: ${COSMO_MODEL_FILE_EXTENSIONS}")
message(STATUS "\tEngine:    ${COSMO_ENGINE_TYPE}")

if(COSMO_TARGET_ARCH STREQUAL "aarch64")
    set(COSMO_EXTERNAL_PROJECT_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchains/aarch64-linux.toolchain.cmake")
    set(COSMO_EXTERNAL_PROJECT_CC /usr/bin/aarch64-linux-gnu-gcc)
    set(COSMO_EXTERNAL_PROJECT_CXX /usr/bin/aarch64-linux-gnu-g++)
    set(COSMO_EXTERNAL_PROJECT_AR /usr/bin/aarch64-linux-gnu-ar)
    set(COSMO_EXTERNAL_PROJECT_LD /usr/bin/aarch64-linux-gnu-ld)
    set(COSMO_EXTERNAL_PROJECT_RANLIB /usr/bin/aarch64-linux-gnu-ranlib)
    set(COSMO_EXTERNAL_PROJECT_STRIP /usr/bin/aarch64-linux-gnu-strip)
    set(COSMO_EXTERNAL_PROJECT_CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${COSMO_EXTERNAL_PROJECT_TOOLCHAIN_FILE}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_SYSTEM_NAME=Linux
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DCMAKE_C_COMPILER=${COSMO_EXTERNAL_PROJECT_CC}
        -DCMAKE_CXX_COMPILER=${COSMO_EXTERNAL_PROJECT_CXX}
        -DCMAKE_AR=${COSMO_EXTERNAL_PROJECT_AR}
        -DCMAKE_LINKER=${COSMO_EXTERNAL_PROJECT_LD}
        -DCMAKE_RANLIB=${COSMO_EXTERNAL_PROJECT_RANLIB}
        -DCMAKE_STRIP=${COSMO_EXTERNAL_PROJECT_STRIP}
        -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
        -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
    )
else()
    set(COSMO_EXTERNAL_PROJECT_CC ${CMAKE_C_COMPILER})
    set(COSMO_EXTERNAL_PROJECT_CXX ${CMAKE_CXX_COMPILER})
    set(COSMO_EXTERNAL_PROJECT_AR ${CMAKE_AR})
    set(COSMO_EXTERNAL_PROJECT_LD ${CMAKE_LINKER})
    set(COSMO_EXTERNAL_PROJECT_RANLIB ${CMAKE_RANLIB})
    set(COSMO_EXTERNAL_PROJECT_STRIP ${CMAKE_STRIP})
    set(COSMO_EXTERNAL_PROJECT_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )
    if(CMAKE_TOOLCHAIN_FILE)
        list(APPEND COSMO_EXTERNAL_PROJECT_CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        )
    endif()
endif()

##########################################################
#Third - party Dependencies
##########################################################
set(THIRDPARTY_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/thirdparty_install CACHE PATH "thirdparty install dir")
add_custom_target(third_build)

# Pre-create include directories for all ExternalProject-managed libraries.
# CMake validates INTERFACE_INCLUDE_DIRECTORIES paths at configure time, but
# ExternalProject only installs headers at build time.  Without this, a clean
# build (rm -rf build && ./build.sh) fails because the paths don't exist yet.
# Note: ffmpeg and openh264 are excluded — they are now provided as prebuilt
# binaries under prebuild/ffmpeg/<arch>/ and do not use ExternalProject.
foreach(_tp_subdir fmt glog zlib sqlitecpp uuid uSockets mp4v2 openssl curl mqtt event cryptopp tokenizers)
    file(MAKE_DIRECTORY "${THIRDPARTY_INSTALL_PREFIX}/${_tp_subdir}/include")
endforeach()

include(ExternalProject)

include(cmake/fmt.cmake)
include(cmake/glog.cmake)
include(cmake/zlib.cmake)
include(cmake/sqlitecpp.cmake)
include(cmake/uuid.cmake)
include(cmake/uSockets.cmake)
include(cmake/mp4v2.cmake)
include(cmake/openssl.cmake)
include(cmake/curl.cmake)
include(cmake/mqtt.cmake)
include(cmake/event.cmake)
if(COSMO_TARGET_PLATFORM STREQUAL "rk3588")
    include(cmake/rk3588_sdk.cmake)
endif()
if(COSMO_TARGET_PLATFORM STREQUAL "ascend310p3")
    include(cmake/ascend_sdk.cmake)
endif()
include(cmake/ffmpeg.cmake)
include(cmake/cryptopp.cmake)
include(cmake/uWebSockets.cmake)
include(cmake/tokenizers.cmake)
if(COSMO_NN_USE_SOPHON_BACKEND)
    include(cmake/device.cmake)
endif()
if(COSMO_NN_USE_CPU_BACKEND)
    include(cmake/onnxruntime.cmake)
endif()
include(cmake/pcap.cmake)
include(cmake/srs.cmake)

#Header - only third - party libraries
add_library(nlohmann INTERFACE)
target_include_directories(nlohmann SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/3rd/include)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/3rd/include)

add_library(eigen3 INTERFACE)
target_include_directories(eigen3 SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/3rd/eigen3-3.2.92)

# Minimal common dependencies shared by nearly all OBJECT libraries.
# Provides SYSTEM include directories for fmt, glog and nlohmann-json,
# which suppresses compiler warnings from these ubiquitous third-party headers.
# Each module declares its own additional third-party deps explicitly.
add_library(cosmo_common INTERFACE)
target_link_libraries(cosmo_common INTERFACE fmt glog nlohmann)

##########################################################
#OBJECT Libraries(per - module incremental compilation)
##########################################################

#-- - Layer 0 : Foundation -- -
add_subdirectory(src/util)

add_subdirectory(src/db)

add_subdirectory(src/platform)

add_subdirectory(src/mem)

add_subdirectory(src/nn)

#-- - Layer 1 : Infrastructure -- -
add_subdirectory(src/media)

add_subdirectory(src/infer)

add_subdirectory(src/network)

#-- - Layer 2 : Business Logic -- -
add_subdirectory(src/linkage)

add_subdirectory(src/service)

add_subdirectory(src/flow)

#-- - Layer 3 : Interface -- -
add_subdirectory(src/api)

#-- - App Entry -- -
file(GLOB_RECURSE APP_SRC CONFIGURE_DEPENDS "src/app/*.h" "src/app/*.cc")

#All OBJECT library targets(grouped by dependency layer)
#Note : cosmo_nn is excluded — it uses its own compile options(see src/nn/CMakeLists.txt)
set(ALL_OBJECT_LIBS
#Layer 0 — Foundation(no upward dependencies)
    cosmo_util cosmo_db cosmo_platform cosmo_mem
#Layer 1 — Infrastructure
    cosmo_media cosmo_infer cosmo_network
#Layer 2 — Business(linkage depends on service - layer DTO types)
    cosmo_linkage cosmo_service cosmo_flow
#Layer 3 — Interface
    cosmo_api
)

#Apply strict compiler warnings and third-party build ordering to all project OBJECT libraries.
#OBJECT libraries do NOT transitively inherit build-order from IMPORTED targets,
#so we must explicitly add_dependencies to guarantee ExternalProject headers are installed first.
foreach(lib ${ALL_OBJECT_LIBS})
    target_compile_options(${lib} PRIVATE ${COSMO_STRICT_WARNINGS})
    add_dependencies(${lib} third_build)
endforeach()

#Collect all OBJECT outputs for linking
set(ALL_OBJECTS
    $<TARGET_OBJECTS:cosmo_util>
    $<TARGET_OBJECTS:cosmo_db>
    $<TARGET_OBJECTS:cosmo_platform>
    $<TARGET_OBJECTS:cosmo_mem>
    $<TARGET_OBJECTS:cosmo_linkage>
    $<TARGET_OBJECTS:cosmo_media>
    $<TARGET_OBJECTS:cosmo_infer>
    $<TARGET_OBJECTS:cosmo_network>
    $<TARGET_OBJECTS:cosmo_service>
    $<TARGET_OBJECTS:cosmo_flow>
    $<TARGET_OBJECTS:cosmo_api>
    $<TARGET_OBJECTS:cosmo_nn>
)

#Common link libraries(shared by main executable and tests)
set(COMMON_LIBS
    stdc++fs fmt glog cryptopp SQLiteCpp sqlite3 uSockets uuid z pcap
    openssl_ssl openssl_crypto curl
    event_core event_extra event
    mqtt_c mqtt_cs
    ffmpeg_avcodec ffmpeg_avdevice ffmpeg_avfilter
    ffmpeg_avformat ffmpeg_avutil ffmpeg_swresample ffmpeg_swscale
    mp4v2
    tokenizers_cpp tokenizers_c
    pthread dl m
)

#Backend - specific link libraries
if(COSMO_NN_USE_SOPHON_BACKEND)
    list(APPEND COMMON_LIBS bmlib bmrt bmcv bmvd bmvenc)
endif()
if(COSMO_NN_USE_CPU_BACKEND)
    list(APPEND COMMON_LIBS onnxruntime)
endif()
if(COSMO_NN_USE_RKNN_BACKEND)
    list(APPEND COMMON_LIBS
        "${COSMO_RK3588_RKNNRT_LIB}"
        "${COSMO_RK3588_RGA_LIB}"
        "${COSMO_RK3588_DRM_LIB}")
endif()
if(COSMO_NN_USE_ASCEND_BACKEND)
    list(APPEND COMMON_LIBS
        "${COSMO_ASCENDCL_LIB}"
        "${COSMO_ASCEND_DVPP_LIB}"
        "${COSMO_ASCEND_DVPP_MPI_LIB}")
endif()

##########################################################
#Main Executable
##########################################################
set(EXECUTABLE_NAME "cosmo-engine")

add_executable(${EXECUTABLE_NAME} ${APP_SRC} ${ALL_OBJECTS})
target_include_directories(${EXECUTABLE_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src)
# Third-party headers come from IMPORTED targets via COMMON_LIBS + header-only libs
target_link_libraries(${EXECUTABLE_NAME} PRIVATE nlohmann stb uWebSockets)
if(COSMO_NN_USE_SOPHON_BACKEND)
    target_link_directories(${EXECUTABLE_NAME} PRIVATE ${DEVICE_LIB_DIR})
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE COSMO_NN_USE_SOPHON_BACKEND)
endif()
if(COSMO_MODEL_GUARD)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE COSMO_HAS_MODEL_GUARD=1)
    if(MODEL_GUARD_LIB)
        target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${MODEL_GUARD_LIB})
    endif()
endif()
if(COSMO_NN_USE_CPU_BACKEND)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE
        COSMO_NN_USE_CPU_BACKEND
        COSMO_NN_USE_HOST_BACKEND
        COSMO_NN_USE_ONNX_BACKEND)
endif()
if(COSMO_NN_USE_RKNN_BACKEND)
    target_compile_definitions(${EXECUTABLE_NAME} PRIVATE COSMO_NN_USE_RKNN_BACKEND)
endif()
#Build order: OBJECT libs → third_build (explicit, see foreach above)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${COMMON_LIBS})

##########################################################
#Unit Tests(optional, cmake - DBUILD_TESTS = ON)
##########################################################
if(BUILD_TESTS)
    file(GLOB TEST_SRC "test/*.cc" "test/*.cpp" "test/mock/*.cc")

    #App sources without main.cc
    set(APP_SRC_NO_MAIN ${APP_SRC})
    list(FILTER APP_SRC_NO_MAIN EXCLUDE REGEX ".*app/main\\.cc$")

    add_executable(cosmo-tests ${APP_SRC_NO_MAIN} ${ALL_OBJECTS} ${TEST_SRC})
    add_dependencies(cosmo-tests third_build)
    target_include_directories(cosmo-tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/test)
    target_precompile_headers(cosmo-tests PRIVATE
        ${CMAKE_SOURCE_DIR}/test/catch_amalgamated.hpp
        ${CMAKE_SOURCE_DIR}/test/catch2/trompeloeil.hpp
    )
    target_link_libraries(cosmo-tests PRIVATE nlohmann stb uWebSockets)
    if(COSMO_NN_USE_SOPHON_BACKEND)
        target_link_directories(cosmo-tests PRIVATE ${DEVICE_LIB_DIR})
        target_compile_definitions(cosmo-tests PRIVATE COSMO_NN_USE_SOPHON_BACKEND)
    endif()
    if(COSMO_MODEL_GUARD)
        target_compile_definitions(cosmo-tests PRIVATE COSMO_HAS_MODEL_GUARD=1)
        if(MODEL_GUARD_LIB)
            target_link_libraries(cosmo-tests PRIVATE ${MODEL_GUARD_LIB})
        endif()
    endif()
    if(COSMO_NN_USE_CPU_BACKEND)
        target_compile_definitions(cosmo-tests PRIVATE
            COSMO_NN_USE_CPU_BACKEND
            COSMO_NN_USE_HOST_BACKEND
            COSMO_NN_USE_ONNX_BACKEND)
    endif()
    if(COSMO_NN_USE_RKNN_BACKEND)
        target_compile_definitions(cosmo-tests PRIVATE COSMO_NN_USE_RKNN_BACKEND)
        target_include_directories(cosmo-tests PRIVATE
            "${COSMO_RK3588_SDK_ROOT}/include"
            "${COSMO_RK3588_SDK_ROOT}/include/rga"
            "${COSMO_RK3588_SYSROOT}/usr/include/libdrm")
        target_link_libraries(cosmo-tests PRIVATE
            "${COSMO_RK3588_RKNNRT_LIB}"
            "${COSMO_RK3588_RGA_LIB}"
            "${COSMO_RK3588_DRM_LIB}")
    endif()
    if(COSMO_NN_USE_ASCEND_BACKEND)
        # Ascend tests (e.g. test_ascend_image_to_tensor.cc) include the CANN
        # headers directly; the AscendCL/DVPP libraries propagate through
        # cosmo_nn and COMMON_LIBS.
        target_include_directories(cosmo-tests PRIVATE "${COSMO_ASCEND_SDK_ROOT}/include")
    endif()
    target_link_libraries(cosmo-tests PRIVATE ${COMMON_LIBS})
    if(COSMO_ENABLE_COVERAGE)
        # gcov runtime is only needed when the test binary was instrumented.
        target_link_libraries(cosmo-tests PRIVATE gcov)

        add_custom_target(coverage
            COMMAND lcov --capture --directory . --output-file coverage.info
            COMMAND lcov --remove coverage.info '/usr/*' '*/3rd/*' '*/test/*' '*/thirdparty_install/*' --output-file coverage.info
            COMMAND genhtml coverage.info --output-directory coverage_report
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report..."
        )
    endif()
endif()




##########################################################
# Web Frontend Build (Vue 3 + Vite)
##########################################################
include(cmake/web_frontend.cmake)

##########################################################
# Install
##########################################################

# Binary
install(TARGETS ${EXECUTABLE_NAME}
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ GROUP_EXECUTE
)

# Nginx configuration (uses system nginx, no custom binary)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nginx/conf
    DESTINATION bin/nginx_conf
    FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ
)

# Web Frontend
install(DIRECTORY ${WEB_BUILD_DIR}/web_unified/dist/
    DESTINATION web
    FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ
)

# Data
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
    DESTINATION files
    FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ
    PATTERN "*.json" EXCLUDE
    PATTERN "*.OTF"  EXCLUDE
    PATTERN "netplan" EXCLUDE
    PATTERN "resource" EXCLUDE
    PATTERN "test-video" EXCLUDE
)
# Font
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data/SOURCEHANSANSCN-REGULAR.OTF
    DESTINATION font
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ
)

# Resource (models, algorithm, layout, etc. from external repository)
if(DEFINED RESOURCE_DIR AND IS_DIRECTORY "${RESOURCE_DIR}")
    message(STATUS "Resource directory: ${RESOURCE_DIR}")
    install(DIRECTORY ${RESOURCE_DIR}/
        DESTINATION resource
        FILE_PERMISSIONS OWNER_WRITE OWNER_READ GROUP_WRITE GROUP_READ
        PATTERN ".git" EXCLUDE
    )
else()
    message(STATUS "No resource directory specified — package will not contain models")
endif()

# Scripts (exclude dev-only tools from device deployment)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
    DESTINATION scripts
    FILES_MATCHING
        PATTERN "build*" EXCLUDE
        PATTERN "format_check*" EXCLUDE
        PATTERN "static_analysis*" EXCLUDE
        PATTERN "install-hooks*" EXCLUDE
        PATTERN "pre-commit" EXCLUDE
        PATTERN "*.sh"
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ GROUP_EXECUTE
)

# Netplan failsafe config (deployed to scripts/ to keep device-side path unchanged)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/netplan/
    DESTINATION scripts
    FILES_MATCHING
        PATTERN "*.bak"
            PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ
)

##########################################################
# Packaging
##########################################################
file(STRINGS "${CMAKE_SOURCE_DIR}/src/util/Version.h" _ver_major REGEX "kVersionMajor")
file(STRINGS "${CMAKE_SOURCE_DIR}/src/util/Version.h" _ver_minor REGEX "kVersionMinor")
file(STRINGS "${CMAKE_SOURCE_DIR}/src/util/Version.h" _ver_patch REGEX "kVersionPatch")
string(REGEX REPLACE ".*kVersionMajor = ([0-9]+).*" "\\1" VER_MAJOR "${_ver_major}")
string(REGEX REPLACE ".*kVersionMinor = ([0-9]+).*" "\\1" VER_MINOR "${_ver_minor}")
string(REGEX REPLACE ".*kVersionPatch = ([0-9]+).*" "\\1" VER_PATCH "${_ver_patch}")

set(PACKAGE_PREFIX  "cosmo")
set(PACKAGE_VERSION "V${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")
set(PACKAGE_NAME    "${PACKAGE_PREFIX}-${PACKAGE_VERSION}")

# Generate and install version.txt
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version.txt" "${PACKAGE_VERSION}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.txt"
    DESTINATION bin
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# Install model guard shared library to lib/ (used at runtime via dlopen or direct link)
if(COSMO_MODEL_GUARD AND MODEL_GUARD_LIB)
    install(FILES ${MODEL_GUARD_LIB}
        DESTINATION lib
        PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    )
else()
    # Create empty lib directory to satisfy legacy upgrade checks on target device
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
        DESTINATION .
    )
endif()

set(CPACK_PACKAGE_NAME              "${PACKAGE_PREFIX}")
set(CPACK_PACKAGE_VENDOR            "Nanjing Cosmo Wander AI Technology Co., Ltd.")
set(CPACK_PACKAGE_VERSION           "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")
set(CPACK_PACKAGE_VERSION_MAJOR     "${VER_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR     "${VER_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH     "${VER_PATCH}")
set(CPACK_PACKAGE_DESCRIPTION       "Cosmo Package")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Cosmo - AI Inference Engine")
set(CPACK_PACKAGE_CONTACT           "hello@cosmowander.ai")
set(CPACK_GENERATOR                 "TGZ")
set(CPACK_PACKAGE_FILE_NAME         "${PACKAGE_NAME}")
set(CPACK_OUTPUT_FILE_PREFIX        "${CMAKE_BINARY_DIR}/packages")

include(CPack)

add_custom_target(package_all
    COMMAND cpack
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/package_md5_rename.sh
        ${CMAKE_BINARY_DIR}/packages ${PACKAGE_NAME}
    DEPENDS ${EXECUTABLE_NAME} web_frontend
    COMMENT "Creating package with MD5 in filename"
)
