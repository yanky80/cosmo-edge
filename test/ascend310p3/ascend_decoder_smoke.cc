// Host-level runnable check for issue #30/#32: decode existing local-file and
// RTSP packets with the test host's custom Ascend FFmpeg and expose device
// (AV_PIX_FMT_ASCEND) NV12 surfaces via VideoFrame/FrameSurface.
//
// The smoke drives the real engine decoder class where the full engine build
// is not available yet on the 310P3 host (the Ascend NN backend is a separate
// follow-up issue):
//   - media::VideoDecoderAscend (h264_ascend/h265_ascend, device NV12
//     surfaces; host NV12 fallback kept for sources without device export)
//   - media::DownloadAscendDeviceSurfaceToHost (on-demand D2H copy for
//     preview/snapshot/OSD consumers; never used by the inference path)
//
// Built standalone on the 310P3 host against /opt/ffmpeg-4.4.1/ascend:
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   g++ -std=c++17 -O2 -I src -I 3rd/fmt-7.1.2/include \
//     test/ascend310p3/ascend_decoder_smoke.cc \
//     3rd/fmt-7.1.2/src/format.cc \
//     src/media/VideoDecoder.cc \
//     src/media/VideoDecoderCreateAscend.cc \
//     src/media/VideoDecoderAscend.cc src/media/VideoDecoderAscendHostCopy.cc \
//     src/media/VideoFrame.cc src/media/PixelFormatUtils.cc \
//     src/mem/MemoryPoolMng.cc src/mem/FixedBlockPool.cc \
//     src/mem/AllocatorCpu.cc src/mem/BlockFreqCalc.cc \
//     src/util/Thread.cc src/util/ThreadRegistry.cc src/util/TimeUtil.cc \
//     -I/opt/ffmpeg-4.4.1/ascend/include \
//     -L/opt/ffmpeg-4.4.1/ascend/lib -lavformat -lavcodec -lavutil \
//     -Wl,-rpath,/opt/ffmpeg-4.4.1/ascend/lib \
//     -L"${ASCEND_TOOLKIT_HOME}/lib64" -lascendcl \
//     -Wl,-rpath,"${ASCEND_TOOLKIT_HOME}/lib64" -lpthread \
//     -o ascend_decoder_smoke
//
// Usage: ascend_decoder_smoke [--h264 file] [--h265 file] [--rtsp url]
//                             [--frames N]
//
// The --rtsp path needs an RTSP source; on hosts without a media server use
// the bundled minimal source: python3 test/ascend310p3/rtsp_test_source.py
//   --file <annexb.h265> --codec h265 --port 8554

#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
}

#include "media/FrameSurface.h"
#include "media/PixelFormat.h"
#include "media/VideoDecoder.h"
#include "media/VideoDecoderAscend.h"
#include "media/VideoFrame.h"
#include "mem/AllocatorCpu.h"
#include "mem/MemoryPoolMng.h"

// Standalone smoke: provide the logging entry point that util/Log.h declares
// (the full engine links glog; this check only needs the fmt-format callback).
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

// The full engine links libuuid; the standalone check does not, so provide
// the small entry point used by util/Thread.
namespace cosmo::util {
std::string GenerateUUID() {
    static std::atomic<uint64_t> counter{0};
    return "ascend-decoder-smoke-" + std::to_string(counter.fetch_add(1));
}
}  // namespace cosmo::util

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string AvError(int code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
    av_strerror(code, text.data(), text.size());
    return text.data();
}

bool IsRtspUrl(const std::string& url) {
    return url.rfind("rtsp://", 0) == 0;
}

// ── Demux: file or RTSP, mirroring the engine demux lifecycle ───────────
class Demux {
public:
    ~Demux() {
        Close();
    }

    bool Open(const std::string& url) {
        Close();
        AVDictionary* options = nullptr;
        if (IsRtspUrl(url)) {
            // Mirror the engine demux (RtspDemuxStrategy: tcp); VLC's RTSP
            // server only accepts UDP, so SMOKE_RTSP_TRANSPORT overrides it.
            // SMOKE_RTSP_NO_NOBUFFER drops the engine's "nobuffer" flag:
            // with UDP RTP the 4.4.1 demuxer then drops reordered packets
            // and never delivers frames (observed against VLC on the host).
            const char* transport = std::getenv("SMOKE_RTSP_TRANSPORT");
            av_dict_set(&options, "rtsp_transport", transport != nullptr ? transport : "tcp", 0);
            av_dict_set(&options, "stimeout", "5000000", 0);
            av_dict_set(&options, "max_delay", "200000", 0);
            if (std::getenv("SMOKE_RTSP_NO_NOBUFFER") == nullptr) {
                av_dict_set(&options, "fflags", "nobuffer", 0);
            }
        }
        const int ret = avformat_open_input(&format_, url.c_str(), nullptr, &options);
        av_dict_free(&options);
        if (ret < 0 || format_ == nullptr) {
            if (format_ != nullptr) {
                avformat_close_input(&format_);
                format_ = nullptr;
            }
            std::cerr << "demux open failed: " << AvError(ret) << "\n";
            return false;
        }
        if (avformat_find_stream_info(format_, nullptr) < 0) {
            Close();
            return false;
        }
        stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index_ < 0) {
            Close();
            return false;
        }
        // MP4 containers store H.264/H.265 in AVCC format, so the demuxer
        // applies the mp4toannexb bitstream filter before packets reach the
        // Ascend hardware decoder. RTSP already delivers Annex-B.
        if (!IsRtspUrl(url)) {
            const AVCodecParameters* params = format_->streams[stream_index_]->codecpar;
            const char* bsf_name            = nullptr;
            if (params->codec_id == AV_CODEC_ID_H264) {
                bsf_name = "h264_mp4toannexb";
            } else if (params->codec_id == AV_CODEC_ID_HEVC) {
                bsf_name = "hevc_mp4toannexb";
            }
            if (bsf_name != nullptr) {
                const AVBitStreamFilter* filter = av_bsf_get_by_name(bsf_name);
                if (filter == nullptr || av_bsf_alloc(filter, &bsf_ctx_) < 0 ||
                    avcodec_parameters_copy(bsf_ctx_->par_in, params) < 0 || av_bsf_init(bsf_ctx_) < 0) {
                    Close();
                    return false;
                }
            }
        }
        return true;
    }

    bool Next(AVPacket* packet) {
        if (format_ == nullptr) {
            return false;
        }
        while (true) {
            const int ret = av_read_frame(format_, packet);
            if (ret < 0) {
                std::cerr << "demux read failed: " << AvError(ret) << "\n";
                return false;
            }
            if (packet->stream_index != stream_index_) {
                av_packet_unref(packet);
                continue;
            }
            if (bsf_ctx_ != nullptr) {
                const int send_ret = av_bsf_send_packet(bsf_ctx_, packet);
                av_packet_unref(packet);
                if (send_ret < 0) {
                    std::cerr << "demux bsf send failed: " << AvError(send_ret) << "\n";
                    return false;
                }
                while (true) {
                    const int recv_ret = av_bsf_receive_packet(bsf_ctx_, packet);
                    if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                        break;
                    }
                    if (recv_ret < 0) {
                        std::cerr << "demux bsf receive failed: " << AvError(recv_ret) << "\n";
                        return false;
                    }
                    return true;
                }
                continue;
            }
            return true;
        }
    }

    AVCodecID CodecId() const {
        return format_->streams[stream_index_]->codecpar->codec_id;
    }

    int Width() const {
        return format_->streams[stream_index_]->codecpar->width;
    }

    int Height() const {
        return format_->streams[stream_index_]->codecpar->height;
    }

private:
    void Close() {
        if (bsf_ctx_ != nullptr) {
            av_bsf_free(&bsf_ctx_);
            bsf_ctx_ = nullptr;
        }
        if (format_ != nullptr) {
            avformat_close_input(&format_);
            format_ = nullptr;
        }
        stream_index_ = -1;
    }

    AVFormatContext* format_ = nullptr;
    AVBSFContext* bsf_ctx_   = nullptr;
    int stream_index_        = -1;
};

// ── Surface contract checks on real decoded frames ──────────────────────
void CheckDecodedFrame(const cosmo::media::VideoFramePtr& frame, int width, int height,
                       uint64_t packets_sent) {
    Require(frame != nullptr && frame->Active(), "decoder returned an inactive frame");
    Require(frame->GetPixelFormat() == cosmo::media::PixelFormat::PIXEL_NV12,
            "decoded frame format is not NV12");
    Require(
        frame->GetWidth() == static_cast<size_t>(width) && frame->GetHeight() == static_cast<size_t>(height),
        "decoded frame dimensions changed");
    // The frame index/timestamp is the demux packet index this decoder was
    // fed (SendPacket sets pts/dts from frame_idx); the engine recovers the
    // demuxed timestamp and stream identity with that index via
    // FrameInfoSave/FrameInfoGet, so it must point back at a sent packet.
    Require(frame->GetFrameIndex() < packets_sent,
            "decoded frame index does not map back to a sent video-stream packet");
    Require(frame->GetTimestamp() == static_cast<int64_t>(frame->GetFrameIndex()),
            "decoded frame timestamp does not match its packet index");

    const auto surface = frame->GetSurface();
    Require(surface != nullptr && surface->IsValid(), "decoded frame surface is invalid");
    Require(surface->memory_type == cosmo::media::FrameSurfaceMemoryType::Device,
            "decoded surface is not Ascend device memory (issue #32 device frames)");
    Require(surface->lifetime != nullptr, "decoded surface has no AVFrame lifetime holder");
    Require(surface->planes.size() == 2, "NV12 surface must expose two planes");
    for (int plane = 0; plane < 2; ++plane) {
        const auto& p = surface->planes[static_cast<size_t>(plane)];
        Require(p.virt_addr != nullptr, "NV12 plane has no device pointer");
        Require(p.pitch >= static_cast<size_t>(width), "NV12 plane pitch is shorter than the width");
        const size_t min_rows = plane == 0 ? static_cast<size_t>(height) : static_cast<size_t>(height) / 2;
        Require(p.vertical_stride >= min_rows, "NV12 plane vertical stride is shorter than the height");
        Require(p.size >= p.pitch * p.vertical_stride, "NV12 plane backing size is too small");
        Require(p.offset + p.pitch * min_rows <= p.size, "NV12 plane span exceeds its backing buffer");
    }
    // Preview/snapshot/OSD consumers can request a host copy on demand; the
    // inference path never calls this.
    std::vector<uint8_t> host_nv12;
    std::string copy_error;
    Require(cosmo::media::DownloadAscendDeviceSurfaceToHost(*surface, width, height, host_nv12, copy_error),
            "on-demand device->host copy failed: " + copy_error);
    const size_t expect_bytes =
        static_cast<size_t>(surface->planes[0].pitch) * static_cast<size_t>(height) * 3 / 2;
    Require(host_nv12.size() == expect_bytes, "host copy has the wrong byte size");
    bool any_nonzero = false;
    for (const uint8_t byte : host_nv12) {
        if (byte != 0) {
            any_nonzero = true;
            break;
        }
    }
    Require(any_nonzero, "host copy is all zeros");
    Require(frame->GetHostData() == nullptr,
            "inference frame must not construct host data (zero-copy device path)");
}

void CheckGate0Decoders() {
    Require(avcodec_find_decoder_by_name("h264_ascend") != nullptr,
            "h264_ascend decoder is unavailable on this FFmpeg");
    Require(avcodec_find_decoder_by_name("h265_ascend") != nullptr,
            "h265_ascend decoder is unavailable on this FFmpeg");
    std::cout << "gate0 decoders present: h264_ascend h265_ascend\n";
}

// The hardware path must never silently fall back to FFmpeg software
// decoding: an unsupported codec fails the open instead.
void CheckErrorPaths() {
    auto decoder = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(cosmo::media::VideoCodecType::kMjpeg, 1920, 1080);
    Require(!decoder->Open(), "Ascend decoder must refuse unsupported codecs (no software fallback)");
    Require(cosmo::media::VideoDecoder::Create(1, nullptr) == nullptr,
            "Ascend factory must reject non-Gate-0 device ids");
    std::cout << "error paths passed: unsupported codec and invalid device id rejected\n";
}

// Decodes a source (local file or RTSP) and validates the exposed surfaces.
// The first decoded frame is kept alive while more frames decode, proving the
// AVFrame-backed surface lifetime outlives the decoder's own frame slot.
void RunSource(const std::string& url, int frames_needed) {
    Demux demux;
    Require(demux.Open(url), "demux open failed: " + url);

    const cosmo::media::VideoCodecType codec_type = demux.CodecId() == AV_CODEC_ID_HEVC
                                                        ? cosmo::media::VideoCodecType::kH265
                                                        : cosmo::media::VideoCodecType::kH264;
    auto decoder                                  = cosmo::media::VideoDecoder::Create(0, nullptr);
    decoder->SetCodecType(codec_type, demux.Width(), demux.Height());
    Require(decoder->Open(), "Ascend decoder open failed (no software fallback): " + url);

    AVPacket* packet = av_packet_alloc();
    Require(packet != nullptr, "av_packet_alloc failed");

    std::vector<cosmo::media::VideoFramePtr> frames;
    cosmo::media::VideoFramePtr first_frame;
    const size_t held_at = static_cast<size_t>(frames_needed) / 2;
    int64_t packets_sent = 0;

    auto push = [&](cosmo::media::VideoFramePtr frame) {
        if (!frame) {
            return;
        }
        CheckDecodedFrame(frame, demux.Width(), demux.Height(), static_cast<uint64_t>(packets_sent));
        if (first_frame == nullptr) {
            first_frame = frame;
        }
        frames.push_back(frame);

        // Keep decoding while the first frame's AVFrame stays referenced.
        if (frames.size() == held_at) {
            Require(first_frame->GetSurface() != nullptr && first_frame->GetSurface()->IsValid(),
                    "held frame surface died while decoding continued");
            Require(first_frame->GetSurface()->planes[0].virt_addr != nullptr,
                    "held frame plane pointer was released while decoding continued");
        }
    };

    while (frames.size() < static_cast<size_t>(frames_needed) && demux.Next(packet)) {
        bool ok = false;
        const auto decoded =
            decoder->Decode(packet->data, static_cast<size_t>(packet->size), packets_sent, ok);
        ++packets_sent;
        push(decoded);
        av_packet_unref(packet);

        // The Ascend decoder delivers decoded frames asynchronously, so a
        // single receive per packet can miss H.265 frames; drain after every
        // packet and again once the demuxer reaches the end of the source.
        while (frames.size() < static_cast<size_t>(frames_needed)) {
            auto extra = decoder->GetFrame();
            if (!extra) {
                break;
            }
            push(extra);
        }
    }

    // End-of-stream: flush the Ascend HiMpi pipeline so frames still queued
    // in the hardware decoder are delivered, then drain with a short poll.
    Require(decoder->Flush(), "Ascend decoder flush failed: " + url);
    const auto flush_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    int empty_rounds          = 0;
    while (frames.size() < static_cast<size_t>(frames_needed) &&
           std::chrono::steady_clock::now() < flush_deadline && empty_rounds < 200) {
        auto extra = decoder->GetFrame();
        if (!extra) {
            ++empty_rounds;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        empty_rounds = 0;
        push(extra);
    }
    av_packet_free(&packet);

    Require(frames.size() == static_cast<size_t>(frames_needed),
            "source ended before " + std::to_string(frames_needed) + " valid frames: " + url + " (received " +
                std::to_string(frames.size()) + ")");

    const auto& first = frames.front()->GetSurface();
    std::cout << "source passed: " << url << " codec="
              << (codec_type == cosmo::media::VideoCodecType::kH265 ? "h265_ascend" : "h264_ascend")
              << " frames=" << frames.size() << " fmt=ascend-device planes=" << first->planes.size()
              << " pitch=" << first->planes[0].pitch << " rows=" << first->planes[0].vertical_stride
              << " size=" << first->TotalSize() << "\n";
}

}  // namespace

struct Options {
    std::string h264_file;
    std::string h265_file;
    std::string rtsp_url;
    int frames = 30;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    auto value = [&](int index) {
        Require(index + 1 < argc, std::string("missing value for ") + argv[index]);
        return std::string(argv[index + 1]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--h264")
            options.h264_file = value(i++);
        else if (argument == "--h265")
            options.h265_file = value(i++);
        else if (argument == "--rtsp")
            options.rtsp_url = value(i++);
        else if (argument == "--frames")
            options.frames = std::stoi(value(i++));
        else
            throw std::runtime_error("unknown option: " + argument);
    }
    Require(!options.h264_file.empty() || !options.h265_file.empty() || !options.rtsp_url.empty(),
            "one of --h264/--h265/--rtsp is required");
    Require(options.frames > 0, "--frames must be positive");
    return options;
}

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::cout << "ascend decoder smoke frames=" << options.frames << "\n";

        avformat_network_init();

        if (std::getenv("SMOKE_SKIP_POOL") == nullptr) {
            cosmo::mem::MemoryPoolMng memory_pool(
                std::make_unique<cosmo::mem::AllocatorCpu>(),
                {640 * 640 * 3 / 2, 1280 * 720 * 3 / 2, 1920 * 1088 * 3 / 2, 3840 * 2160 * 3 / 2});
            cosmo::mem::SetMemoryPoolContext(&memory_pool);
        }
        if (std::getenv("SMOKE_SKIP_GATE0") == nullptr) {
            CheckGate0Decoders();
        }
        if (std::getenv("SMOKE_SKIP_ERRORCHECK") == nullptr) {
            CheckErrorPaths();
        }

        for (const std::string& source : {options.h264_file, options.h265_file, options.rtsp_url}) {
            if (source.empty()) {
                continue;
            }
            RunSource(source, options.frames);
        }

        std::cout << "ascend decoder smoke passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ascend decoder smoke failed: " << e.what() << "\n";
        return 1;
    }
}
