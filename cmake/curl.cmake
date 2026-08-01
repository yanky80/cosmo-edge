set(CURL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd/curl-8.17.0)
set(CURL_INSTALL_DIR ${THIRDPARTY_INSTALL_PREFIX}/curl)
set(CURL_HEADERS ${CURL_INSTALL_DIR}/include)
set(CURL_LIB ${CURL_INSTALL_DIR}/lib/libcurl.so)

ExternalProject_Add(
    curl_external

    SOURCE_DIR ${CURL_SOURCE_DIR}

    CMAKE_ARGS
        ${COSMO_EXTERNAL_PROJECT_CMAKE_ARGS}
        -DCMAKE_INSTALL_PREFIX=${CURL_INSTALL_DIR}
        -DOPENSSL_ROOT_DIR=${THIRDPARTY_INSTALL_PREFIX}/openssl
        -DBUILD_SHARED_LIBS=ON
        # Cross-compilation skips curl's host CA auto-detection. This path is
        # resolved on the target at runtime and must be provided by the image.
        -DCURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
        -DCURL_CA_PATH=none
        -DBUILD_LIBCURL_DOCS=OFF
        -DBUILD_MISC_DOCS=OFF
        -DENABLE_CURL_MANUAL=OFF
        -DBUILD_TESTING=OFF
        -DBUILD_CURL_EXE=OFF
        -DBUILD_EXAMPLES=OFF

    CMAKE_CACHE_ARGS
        -DOPENSSL_ROOT_DIR:PATH=${THIRDPARTY_INSTALL_PREFIX}/openssl
        -DOPENSSL_INCLUDE_DIR:PATH=${THIRDPARTY_INSTALL_PREFIX}/openssl/include
        -DOPENSSL_SSL_LIBRARY:FILEPATH=${THIRDPARTY_INSTALL_PREFIX}/openssl/lib/libssl.so
        -DOPENSSL_CRYPTO_LIBRARY:FILEPATH=${THIRDPARTY_INSTALL_PREFIX}/openssl/lib/libcrypto.so
        -DCURL_USE_OPENSSL:BOOL=ON
        -DCURL_USE_PKGCONFIG:BOOL=OFF
        -DCURL_USE_LIBPSL:BOOL=OFF
        -DCURL_ZLIB:STRING=OFF
        -DCURL_BROTLI:STRING=OFF
        -DCURL_ZSTD:STRING=OFF
        -DCURL_DISABLE_LDAP:BOOL=ON
        -DCURL_DISABLE_LDAPS:BOOL=ON
        -DUSE_LIBIDN2:BOOL=OFF
        -DCURL_USE_LIBSSH2:BOOL=OFF
        -DENABLE_ARES:BOOL=OFF
    
    INSTALL_COMMAND ${CMAKE_COMMAND} --install . --prefix ${CURL_INSTALL_DIR}

    DEPENDS openssl_external

    UPDATE_COMMAND ""
    BUILD_ALWAYS OFF

    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
)

add_dependencies(third_build curl_external)

add_library(curl SHARED IMPORTED)
set_target_properties(curl PROPERTIES
    IMPORTED_LOCATION ${CURL_LIB}
    INTERFACE_INCLUDE_DIRECTORIES "${CURL_HEADERS}"
)
add_dependencies(curl curl_external)

install(DIRECTORY ${CURL_INSTALL_DIR}/lib/
    DESTINATION lib
    FILES_MATCHING
        PATTERN "*curl*"
        PATTERN "*so*"
)
