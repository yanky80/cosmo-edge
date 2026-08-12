#!/bin/bash
set -euo pipefail
# Hermetic unit contract check for the Ascend media decoder (issue #30).
#
# Compiles the committed Catch2 contract tests plus the real decoder sources
# with a small logger shim (the full engine links glog) and runs them against
# the FFmpeg available on the build machine:
#   - locally:  system FFmpeg dev headers/libs (pkg-config)
#   - 310P3:    ASCEND_FFMPEG_PREFIX=/opt/ffmpeg-4.4.1/ascend
#
# The decoder class methods are compiled but unreferenced by the contract
# tests, so --gc-sections drops them and their app-only dependencies
# (VideoFrame/memory pool) never need to link here.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${TMPDIR:-/tmp}/cosmo-ascend-contract-check"
CXX="${CXX:-g++}"
mkdir -p "$BUILD_DIR"

if [ -n "${ASCEND_FFMPEG_PREFIX:-}" ]; then
    FFMPEG_CFLAGS="-I${ASCEND_FFMPEG_PREFIX}/include"
    FFMPEG_LIBS="-L${ASCEND_FFMPEG_PREFIX}/lib -lavformat -lavcodec -lavutil -Wl,-rpath,${ASCEND_FFMPEG_PREFIX}/lib"
else
    FFMPEG_CFLAGS=$(pkg-config --cflags libavcodec libavutil)
    FFMPEG_LIBS=$(pkg-config --libs libavcodec libavutil)
fi

cat > "$BUILD_DIR/log_shim.cc" <<'SHIM'
// Standalone check: provide the logging entry point that util/Log.h declares
// (the full engine links glog; this check only needs the fmt-format callback).
#include "util/Log.h"

namespace cosmo::log {
size_t LogFormatArg(const char* file, const char* function, int line, int module, int severity,
                    fmt::string_view format, fmt::format_args args) {
    static_cast<void>(module);
    static_cast<void>(severity);
    const std::string text = fmt::vformat(format, args);
    std::fprintf(stderr, "%s:%d %s: %s\n", file, line, function, text.c_str());
    return text.size();
}
}  // namespace cosmo::log
SHIM

$CXX -std=c++17 -O2 -ffunction-sections -fdata-sections \
    -DCOSMO_MEDIA_USE_ASCEND_BACKEND \
    -I "$ROOT_DIR/src" -I "$ROOT_DIR/test" -I "$ROOT_DIR/3rd/fmt-7.1.2/include" \
    $FFMPEG_CFLAGS \
    "$ROOT_DIR/test/catch_amalgamated.cpp" \
    "$ROOT_DIR/test/test_video_decoder_ascend.cc" \
    "$ROOT_DIR/src/media/VideoDecoderAscend.cc" \
    "$BUILD_DIR/log_shim.cc" \
    "$ROOT_DIR/3rd/fmt-7.1.2/src/format.cc" \
    -Wl,--gc-sections \
    $FFMPEG_LIBS -lpthread \
    -o "$BUILD_DIR/ascend_decoder_contract_check"

"$BUILD_DIR/ascend_decoder_contract_check"
