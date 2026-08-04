# FFmpeg prebuilt binary integration
#
# Instead of compiling FFmpeg from source via ExternalProject_Add, we use
# prebuilt shared libraries placed under prebuild/ffmpeg/<arch>/.
#
# Prerequisites:
#   - prebuild/ffmpeg/aarch64/{include/,lib/} for the Sophon/aarch64 target
#   - prebuild/ffmpeg/x86_64/{include/,lib/}  for the CPU/x86_64 target
#   - All .so soname symlinks must be present (e.g. libavcodec.so -> libavcodec.so.58)
#   - If COSMO_ENABLE_OPENH264 is ON, libopenh264.so must also be present in the
#     same lib/ directory (FFmpeg must have been built with --enable-libopenh264).
#
# License rationale: using prebuilt binaries with known configure flags (no GPL
# components) eliminates the risk of accidentally enabling GPL-only encoders or
# decoders during a developer's local source build.

if(COSMO_TARGET_PLATFORM STREQUAL "rk3588")
    set(FFMPEG_PREBUILD_DIR "")
    set(FFMPEG_HEADERS "${COSMO_RK3588_FFMPEG_INCLUDE_DIR}")
    set(FFMPEG_AVCODEC_LIB "${COSMO_RK3588_AVCODEC_LIB}")
    set(FFMPEG_AVDEVICE_LIB "${COSMO_RK3588_AVDEVICE_LIB}")
    set(FFMPEG_AVFILTER_LIB "${COSMO_RK3588_AVFILTER_LIB}")
    set(FFMPEG_AVFORMAT_LIB "${COSMO_RK3588_AVFORMAT_LIB}")
    set(FFMPEG_AVUTIL_LIB "${COSMO_RK3588_AVUTIL_LIB}")
    set(FFMPEG_SWRESAMPLE_LIB "${COSMO_RK3588_SWRESAMPLE_LIB}")
    set(FFMPEG_SWSCALE_LIB "${COSMO_RK3588_SWSCALE_LIB}")
    message(STATUS "FFmpeg: using RK3588 sysroot libraries from ${COSMO_RK3588_SYSROOT}")
elseif(COSMO_TARGET_PLATFORM STREQUAL "ascend310p3")
    # Custom Ascend FFmpeg (h264_ascend/h265_ascend decoders and encoders);
    # never copied from prebuild/. Lookup order:
    #   1. COSMO_ASCEND_SYSROOT  (cross builds and hermetic tests)
    #   2. COSMO_ASCEND_FFMPEG_ROOT (defaults to /opt/ffmpeg-4.4.1/ascend on
    #      x86_64, the locked test-host baseline)
    #   3. system FFmpeg dev packages via CMAKE_LIBRARY_ARCHITECTURE
    set(FFMPEG_PREBUILD_DIR "")
    set(_cosmo_system_ffmpeg_include_dirs "")
    set(_cosmo_system_ffmpeg_lib_dirs "")

    # Debian-style multiarch triplet for the target (e.g. aarch64-linux-gnu).
    set(_cosmo_ffmpeg_lib_arch "${CMAKE_LIBRARY_ARCHITECTURE}")
    if(NOT _cosmo_ffmpeg_lib_arch)
        if(COSMO_TARGET_ARCH STREQUAL "aarch64")
            set(_cosmo_ffmpeg_lib_arch "aarch64-linux-gnu")
        else()
            set(_cosmo_ffmpeg_lib_arch "x86_64-linux-gnu")
        endif()
    endif()

    if(COSMO_ASCEND_SYSROOT)
        set(_cosmo_system_ffmpeg_no_default_path NO_DEFAULT_PATH)
        set(_cosmo_system_ffmpeg_include_dirs "${COSMO_ASCEND_SYSROOT}/usr/include")
        set(_cosmo_system_ffmpeg_lib_dirs
            "${COSMO_ASCEND_SYSROOT}/usr/lib/${_cosmo_ffmpeg_lib_arch}"
            "${COSMO_ASCEND_SYSROOT}/usr/lib64"
            "${COSMO_ASCEND_SYSROOT}/usr/lib")
    else()
        set(_cosmo_ascend_ffmpeg_root "${COSMO_ASCEND_FFMPEG_ROOT}")
        if(NOT _cosmo_ascend_ffmpeg_root AND COSMO_TARGET_ARCH STREQUAL "x86_64")
            # Locked test-host baseline (docs/development/ascend310p3-test-host-baseline.md).
            set(_cosmo_ascend_ffmpeg_root "/opt/ffmpeg-4.4.1/ascend")
        endif()
        if(EXISTS "${_cosmo_ascend_ffmpeg_root}/include/libavcodec/avcodec.h")
            set(_cosmo_system_ffmpeg_no_default_path NO_DEFAULT_PATH)
            set(_cosmo_system_ffmpeg_include_dirs "${_cosmo_ascend_ffmpeg_root}/include")
            set(_cosmo_system_ffmpeg_lib_dirs "${_cosmo_ascend_ffmpeg_root}/lib")
        else()
            # Fallback: system FFmpeg dev packages (multiarch-aware).
            set(_cosmo_system_ffmpeg_no_default_path "")
            set(_cosmo_system_ffmpeg_lib_dirs
                "/usr/lib/${_cosmo_ffmpeg_lib_arch}"
                "/lib/${_cosmo_ffmpeg_lib_arch}"
                "/usr/lib64")
        endif()
    endif()
    find_path(FFMPEG_HEADERS libavcodec/avcodec.h
        PATHS ${_cosmo_system_ffmpeg_include_dirs}
        ${_cosmo_system_ffmpeg_no_default_path})
    if(NOT FFMPEG_HEADERS)
        message(FATAL_ERROR
            "COSMO_TARGET_PLATFORM=ascend310p3 requires system FFmpeg headers "
            "(libavcodec/avcodec.h not found)")
    endif()
    foreach(_cosmo_system_ffmpeg_lib
            avcodec avdevice avfilter avformat avutil swresample swscale)
        string(TOUPPER "${_cosmo_system_ffmpeg_lib}" _cosmo_system_ffmpeg_lib_upper)
        find_library(FFMPEG_${_cosmo_system_ffmpeg_lib_upper}_LIB
            "${_cosmo_system_ffmpeg_lib}"
            PATHS ${_cosmo_system_ffmpeg_lib_dirs}
            ${_cosmo_system_ffmpeg_no_default_path})
        if(NOT FFMPEG_${_cosmo_system_ffmpeg_lib_upper}_LIB)
            message(FATAL_ERROR
                "COSMO_TARGET_PLATFORM=ascend310p3 requires system FFmpeg library "
                "lib${_cosmo_system_ffmpeg_lib}.so (not found)")
        endif()
    endforeach()
    message(STATUS "FFmpeg: using external Ascend/system libraries (headers: ${FFMPEG_HEADERS})")
elseif(COSMO_TARGET_ARCH STREQUAL "aarch64")
    set(FFMPEG_PREBUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/ffmpeg/aarch64)
elseif(COSMO_TARGET_ARCH STREQUAL "x86_64")
    set(FFMPEG_PREBUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/prebuild/ffmpeg/x86_64)
else()
    message(FATAL_ERROR "Unsupported architecture for FFmpeg prebuilt: ${COSMO_TARGET_ARCH}")
endif()

if(FFMPEG_PREBUILD_DIR)
    set(FFMPEG_HEADERS         ${FFMPEG_PREBUILD_DIR}/include)
    set(FFMPEG_AVCODEC_LIB     ${FFMPEG_PREBUILD_DIR}/lib/libavcodec.so)
    set(FFMPEG_AVDEVICE_LIB    ${FFMPEG_PREBUILD_DIR}/lib/libavdevice.so)
    set(FFMPEG_AVFILTER_LIB    ${FFMPEG_PREBUILD_DIR}/lib/libavfilter.so)
    set(FFMPEG_AVFORMAT_LIB    ${FFMPEG_PREBUILD_DIR}/lib/libavformat.so)
    set(FFMPEG_AVUTIL_LIB      ${FFMPEG_PREBUILD_DIR}/lib/libavutil.so)
    set(FFMPEG_SWRESAMPLE_LIB  ${FFMPEG_PREBUILD_DIR}/lib/libswresample.so)
    set(FFMPEG_SWSCALE_LIB     ${FFMPEG_PREBUILD_DIR}/lib/libswscale.so)
endif()

if(FFMPEG_PREBUILD_DIR)
    message(STATUS "FFmpeg: using prebuilt libraries from ${FFMPEG_PREBUILD_DIR}")
endif()

# ── OpenH264 (optional, required when COSMO_ENABLE_OPENH264=ON) ──────────────
if(COSMO_ENABLE_OPENH264)
    set(OPENH264_LIBRARY ${FFMPEG_PREBUILD_DIR}/lib/libopenh264.so)
    if(NOT EXISTS "${OPENH264_LIBRARY}")
        message(FATAL_ERROR
            "COSMO_ENABLE_OPENH264=ON but prebuilt libopenh264.so not found: "
            "${OPENH264_LIBRARY}\n"
            "Place a prebuilt libopenh264.so (built with matching FFmpeg) in "
            "${FFMPEG_PREBUILD_DIR}/lib/")
    endif()
    message(STATUS "FFmpeg OpenH264: using prebuilt ${OPENH264_LIBRARY}")

    add_library(openh264 SHARED IMPORTED)
    set_target_properties(openh264 PROPERTIES
        IMPORTED_LOCATION              ${OPENH264_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES  "${FFMPEG_HEADERS}"
    )
endif()

# ── FFmpeg IMPORTED targets ───────────────────────────────────────────────────
# Target names are identical to the previous ExternalProject-based setup so
# that all consumers (src/media/CMakeLists.txt, CMakeLists.txt COMMON_LIBS)
# require no changes.

add_library(ffmpeg_avcodec SHARED IMPORTED)
set_target_properties(ffmpeg_avcodec PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_AVCODEC_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)
if(COSMO_ENABLE_OPENH264)
    target_link_libraries(ffmpeg_avcodec INTERFACE openh264)
endif()

add_library(ffmpeg_avdevice SHARED IMPORTED)
set_target_properties(ffmpeg_avdevice PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_AVDEVICE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

add_library(ffmpeg_avfilter SHARED IMPORTED)
set_target_properties(ffmpeg_avfilter PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_AVFILTER_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

add_library(ffmpeg_avformat SHARED IMPORTED)
set_target_properties(ffmpeg_avformat PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_AVFORMAT_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

add_library(ffmpeg_avutil SHARED IMPORTED)
set_target_properties(ffmpeg_avutil PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_AVUTIL_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

add_library(ffmpeg_swresample SHARED IMPORTED)
set_target_properties(ffmpeg_swresample PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_SWRESAMPLE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

add_library(ffmpeg_swscale SHARED IMPORTED)
set_target_properties(ffmpeg_swscale PROPERTIES
    IMPORTED_LOCATION             ${FFMPEG_SWSCALE_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_HEADERS}"
)

# ── Install: copy prebuilt .so files (with soname symlinks) into package ──────
if(FFMPEG_PREBUILD_DIR)
    install(DIRECTORY ${FFMPEG_PREBUILD_DIR}/lib/
        DESTINATION lib
        FILES_MATCHING
            PATTERN "*.so*"
    )
endif()
