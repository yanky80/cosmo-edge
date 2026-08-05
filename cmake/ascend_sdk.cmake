# External Huawei CANN SDK integration for the ascend310p3 target profile.
#
# The ascend310p3 profile is an x86_64 host build (Ascend 310P3 PCIe card on an
# x86_64 server). This file locates AscendCL (libascendcl) and DVPP
# (libacl_dvpp) headers and shared libraries from a CANN installation and
# fails at configure time when the SDK is missing, the library architecture
# does not match, or the profile is used on a non-x86_64 host. No CANN,
# driver, or firmware binaries are committed to the repository.

if(NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    message(FATAL_ERROR
        "COSMO_TARGET_PLATFORM=ascend310p3 requires an x86_64 host "
        "(got ${CMAKE_HOST_SYSTEM_PROCESSOR})")
endif()

set(COSMO_ASCEND_SDK_ROOT "" CACHE PATH "External CANN toolkit root (AscendCL/DVPP headers and runtime)")
set(COSMO_ASCEND_SYSROOT "" CACHE PATH "Optional system-root override for the ascend310p3 FFmpeg lookup")

if(NOT COSMO_ASCEND_SDK_ROOT)
    if(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND EXISTS "$ENV{ASCEND_TOOLKIT_HOME}")
        set(COSMO_ASCEND_SDK_ROOT "$ENV{ASCEND_TOOLKIT_HOME}"
            CACHE PATH "External CANN toolkit root (AscendCL/DVPP headers and runtime)" FORCE)
    elseif(EXISTS "/usr/local/Ascend/ascend-toolkit/latest")
        set(COSMO_ASCEND_SDK_ROOT "/usr/local/Ascend/ascend-toolkit/latest"
            CACHE PATH "External CANN toolkit root (AscendCL/DVPP headers and runtime)" FORCE)
    endif()
endif()
if(NOT COSMO_ASCEND_SDK_ROOT)
    message(FATAL_ERROR
        "COSMO_TARGET_PLATFORM=ascend310p3 requires a CANN toolkit. "
        "Set COSMO_ASCEND_SDK_ROOT or source the CANN set_env.sh "
        "(which exports ASCEND_TOOLKIT_HOME).")
endif()

set(_cosmo_ascend_headers
    "${COSMO_ASCEND_SDK_ROOT}/include/acl/acl.h"
    "${COSMO_ASCEND_SDK_ROOT}/include/acl/dvpp/hi_dvpp.h"
)
foreach(_cosmo_header IN LISTS _cosmo_ascend_headers)
    if(NOT EXISTS "${_cosmo_header}")
        message(FATAL_ERROR "Ascend SDK header not found: ${_cosmo_header}")
    endif()
endforeach()

set(_cosmo_ascend_lib_search_dirs
    "${COSMO_ASCEND_SDK_ROOT}/lib64"
)

function(_cosmo_find_ascend_lib out_var lib_name)
    foreach(_cosmo_dir IN LISTS _cosmo_ascend_lib_search_dirs)
        if(EXISTS "${_cosmo_dir}/${lib_name}")
            set(${out_var} "${_cosmo_dir}/${lib_name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    message(FATAL_ERROR "Ascend shared library not found: ${lib_name}")
endfunction()

_cosmo_find_ascend_lib(COSMO_ASCENDCL_LIB libascendcl.so)
_cosmo_find_ascend_lib(COSMO_ASCEND_DVPP_LIB libacl_dvpp.so)

# The locked baseline targets the x86_64 CANN build; reject anything else early.
find_program(_cosmo_ascend_readelf readelf)
if(_cosmo_ascend_readelf)
    foreach(_cosmo_ascend_lib IN ITEMS
            "${COSMO_ASCENDCL_LIB}"
            "${COSMO_ASCEND_DVPP_LIB}")
        execute_process(
            COMMAND "${_cosmo_ascend_readelf}" -h "${_cosmo_ascend_lib}"
            OUTPUT_VARIABLE _cosmo_ascend_elf_header
            RESULT_VARIABLE _cosmo_ascend_elf_result
        )
        if(NOT _cosmo_ascend_elf_result EQUAL 0 OR
           NOT _cosmo_ascend_elf_header MATCHES "X86-64")
            message(FATAL_ERROR
                "Ascend library is not an x86_64 ELF: ${_cosmo_ascend_lib}")
        endif()
    endforeach()
else()
    message(STATUS "readelf not found — skipping the Ascend ELF architecture check")
endif()

message(STATUS "Ascend CANN SDK root: ${COSMO_ASCEND_SDK_ROOT}")
message(STATUS "AscendCL library: ${COSMO_ASCENDCL_LIB}")
message(STATUS "Ascend DVPP library: ${COSMO_ASCEND_DVPP_LIB}")
