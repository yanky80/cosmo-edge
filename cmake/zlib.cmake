set(Z_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/zlib-1.3.1)
set(Z_INSTALL_DIR ${THIRDPARTY_INSTALL_PREFIX}/zlib)
set(Z_HEADERS ${Z_INSTALL_DIR}/include)
set(Z_LIB ${Z_INSTALL_DIR}/lib/libz.so)

ExternalProject_Add(
    z_external

    SOURCE_DIR ${Z_SOURCE_DIR}

    CMAKE_ARGS
        ${COSMO_EXTERNAL_PROJECT_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${Z_INSTALL_DIR}
        -DINSTALL_BIN_DIR=${Z_INSTALL_DIR}/bin
        -DINSTALL_LIB_DIR=${Z_INSTALL_DIR}/lib
        -DINSTALL_INC_DIR=${Z_INSTALL_DIR}/include
        -DINSTALL_MAN_DIR=${Z_INSTALL_DIR}/share/man
        -DINSTALL_PKGCONFIG_DIR=${Z_INSTALL_DIR}/share/pkgconfig
        -DZLIB_BUILD_EXAMPLES=OFF
    
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --prefix ${Z_INSTALL_DIR}

    UPDATE_COMMAND ""
    BUILD_ALWAYS OFF

    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
)
add_dependencies(third_build z_external)

add_library(z SHARED IMPORTED)
set_target_properties(z PROPERTIES
    IMPORTED_LOCATION ${Z_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${Z_HEADERS}"
)
add_dependencies(z z_external)

install(DIRECTORY ${Z_INSTALL_DIR}/lib/
    DESTINATION lib
    FILES_MATCHING
        PATTERN "*so*"
)
