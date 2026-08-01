set(COSMO_RK3588_SDK_ROOT "" CACHE PATH "External RK3588 SDK root (headers and vendor runtime)")
set(COSMO_RK3588_SYSROOT "" CACHE PATH "External RK3588 sysroot root")
set(COSMO_PKG_CONFIG_EXECUTABLE "" CACHE FILEPATH "Optional pkg-config override")

if(NOT COSMO_RK3588_SDK_ROOT)
    message(FATAL_ERROR "COSMO_TARGET_PLATFORM=rk3588 requires COSMO_RK3588_SDK_ROOT")
endif()
if(NOT COSMO_RK3588_SYSROOT)
    message(FATAL_ERROR "COSMO_TARGET_PLATFORM=rk3588 requires COSMO_RK3588_SYSROOT")
endif()

set(_cosmo_rk_headers
    "${COSMO_RK3588_SDK_ROOT}/include/rknn_api.h"
    "${COSMO_RK3588_SDK_ROOT}/include/rga/RgaApi.h"
    "${COSMO_RK3588_SYSROOT}/usr/include/libdrm/drm.h"
)
foreach(_cosmo_header IN LISTS _cosmo_rk_headers)
    if(NOT EXISTS "${_cosmo_header}")
        message(FATAL_ERROR "RK3588 SDK header not found: ${_cosmo_header}")
    endif()
endforeach()

set(_cosmo_rk_lib_search_dirs
    "${COSMO_RK3588_SDK_ROOT}/lib"
    "${COSMO_RK3588_SDK_ROOT}/lib64"
    "${COSMO_RK3588_SYSROOT}/usr/local/lib"
    "${COSMO_RK3588_SYSROOT}/usr/local/lib64"
    "${COSMO_RK3588_SYSROOT}/usr/lib"
    "${COSMO_RK3588_SYSROOT}/usr/lib64"
    "${COSMO_RK3588_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    "${COSMO_RK3588_SYSROOT}/lib"
    "${COSMO_RK3588_SYSROOT}/lib64"
)

function(_cosmo_find_rk_lib out_var lib_name)
    foreach(_cosmo_dir IN LISTS _cosmo_rk_lib_search_dirs)
        if(EXISTS "${_cosmo_dir}/${lib_name}")
            set(${out_var} "${_cosmo_dir}/${lib_name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "RK3588 shared library not found: ${lib_name}")
endfunction()

_cosmo_find_rk_lib(COSMO_RK3588_RKNNRT_LIB librknnrt.so)
_cosmo_find_rk_lib(COSMO_RK3588_RGA_LIB librga.so)
_cosmo_find_rk_lib(COSMO_RK3588_DRM_LIB libdrm.so)
_cosmo_find_rk_lib(COSMO_RK3588_AVCODEC_LIB libavcodec.so)
_cosmo_find_rk_lib(COSMO_RK3588_AVDEVICE_LIB libavdevice.so)
_cosmo_find_rk_lib(COSMO_RK3588_AVFILTER_LIB libavfilter.so)
_cosmo_find_rk_lib(COSMO_RK3588_AVFORMAT_LIB libavformat.so)
_cosmo_find_rk_lib(COSMO_RK3588_AVUTIL_LIB libavutil.so)
_cosmo_find_rk_lib(COSMO_RK3588_SWRESAMPLE_LIB libswresample.so)
_cosmo_find_rk_lib(COSMO_RK3588_SWSCALE_LIB libswscale.so)

if(COSMO_PKG_CONFIG_EXECUTABLE)
    set(_cosmo_pkg_config "${COSMO_PKG_CONFIG_EXECUTABLE}")
else()
    find_program(_cosmo_pkg_config pkg-config)
endif()
if(NOT _cosmo_pkg_config)
    message(FATAL_ERROR "COSMO_TARGET_PLATFORM=rk3588 requires pkg-config")
endif()

set(_cosmo_pkgconfig_dirs "")
foreach(_cosmo_dir
        "${COSMO_RK3588_SYSROOT}/usr/lib/pkgconfig"
        "${COSMO_RK3588_SYSROOT}/usr/lib64/pkgconfig"
        "${COSMO_RK3588_SYSROOT}/usr/local/lib/pkgconfig"
        "${COSMO_RK3588_SYSROOT}/usr/local/lib64/pkgconfig"
        "${COSMO_RK3588_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/pkgconfig"
        "${COSMO_RK3588_SYSROOT}/usr/share/pkgconfig")
    if(EXISTS "${_cosmo_dir}")
        list(APPEND _cosmo_pkgconfig_dirs "${_cosmo_dir}")
    endif()
endforeach()
list(JOIN _cosmo_pkgconfig_dirs ":" _cosmo_pkgconfig_path)

foreach(_cosmo_pkg libdrm rockchip_mpp libavcodec libavformat libavutil)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "PKG_CONFIG_SYSROOT_DIR=${COSMO_RK3588_SYSROOT}"
            "PKG_CONFIG_LIBDIR=${_cosmo_pkgconfig_path}"
            "${_cosmo_pkg_config}" --exists "${_cosmo_pkg}"
        RESULT_VARIABLE _cosmo_pkg_result
    )
    if(NOT _cosmo_pkg_result EQUAL 0)
        message(FATAL_ERROR "RK3588 pkg-config module not found: ${_cosmo_pkg}")
    endif()
endforeach()

set(COSMO_RK3588_FFMPEG_INCLUDE_DIR "${COSMO_RK3588_SYSROOT}/usr/include")
set(COSMO_RK3588_FFMPEG_LIB_DIR "${COSMO_RK3588_SYSROOT}/usr/lib")
message(STATUS "RK3588 SDK root: ${COSMO_RK3588_SDK_ROOT}")
message(STATUS "RK3588 sysroot: ${COSMO_RK3588_SYSROOT}")
