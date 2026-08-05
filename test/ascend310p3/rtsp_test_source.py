#!/usr/bin/env python3
"""Minimal single-stream RTSP source for the decoder host smokes (issue #30).

Serves one raw Annex-B H.264/H.265 file over RTSP with interleaved
RTP/TCP (rtsp_transport=tcp), the transport the engine's RtspDemuxStrategy
uses. The 310P3/RK3588 test hosts have no media server installed, so this
dependency-free (stdlib-only) source makes the --rtsp path of
ascend_decoder_smoke verifiable on real hardware.

Usage:
  rtsp_test_source.py --file stream.264  --codec h264 --port 8554
  rtsp_test_source.py --file stream.h265 --codec h265 --port 8554
"""

import argparse
import base64
import socket
import struct
import threading
import time

RTP_PT = 96
CLOCK = 90000  # RTP clock rate for H.264/H.265


def split_annexb(data):
    """Split Annex-B bytes into NAL units (start codes stripped)."""
    nals = []
    i = 0
    n = len(data)
    while i < n:
        if data[i:i + 4] == b"\x00\x00\x00\x01":
            start = i + 4
        elif data[i:i + 3] == b"\x00\x00\x01":
            start = i + 3
        else:
            i += 1
            continue
        j = start
        while j < n and not (data[j:j + 4] == b"\x00\x00\x00\x01" or data[j:j + 3] == b"\x00\x00\x01"):
            j += 1
        if j > start:
            nals.append(data[start:j])
        i = j
    return nals


class Stream:
    def __init__(self, path, codec, fps):
        with open(path, "rb") as fh:
            self.nals = split_annexb(fh.read())
        if not self.nals:
            raise SystemExit(f"no NAL units found in {path}")
        self.codec = codec
        self.fps = fps
        self.sps = None
        self.pps = None
        self.vps = None
        for nal in self.nals:
            if codec == "h264":
                ntype = nal[0] & 0x1F
                if ntype == 7 and self.sps is None:
                    self.sps = nal
                elif ntype == 8 and self.pps is None:
                    self.pps = nal
            else:
                ntype = (nal[0] >> 1) & 0x3F
                if ntype == 32 and self.vps is None:
                    self.vps = nal
                elif ntype == 33 and self.sps is None:
                    self.sps = nal
                elif ntype == 34 and self.pps is None:
                    self.pps = nal
        if self.sps is None or self.pps is None:
            raise SystemExit(f"no SPS/PPS found in {path}")

    def sprop(self):
        """RFC 3984/7798 sprop-parameter-sets (base64, comma separated)."""
        units = []
        if self.codec == "h265" and self.vps is not None:
            units.append(base64.b64encode(self.vps).decode())
        units.append(base64.b64encode(self.sps).decode())
        units.append(base64.b64encode(self.pps).decode())
        return ",".join(units)

    def sdp(self):
        if self.codec == "h264":
            codec_name = "H264"
            # RFC 3984: comma-separated sprop-parameter-sets (SPS,PPS).
            fmtp = f"packetization-mode=1;sprop-parameter-sets={self.sprop()}"
        else:
            codec_name = "H265"
            # RFC 7798: separate sprop-vps/sps/pps attributes.
            vps = base64.b64encode(self.vps).decode() if self.vps is not None else ""
            fmtp = (f"sprop-vps={vps};sprop-sps="
                    f"{base64.b64encode(self.sps).decode()};sprop-pps="
                    f"{base64.b64encode(self.pps).decode()}")
        return (
            "v=0\r\n"
            f"o=- 0 0 IN IP4 127.0.0.1\r\n"
            "s=cosmo rtsp test source\r\n"
            "c=IN IP4 127.0.0.1\r\n"
            "t=0 0\r\n"
            f"m=video 0 RTP/AVP {RTP_PT}\r\n"
            f"a=rtpmap:{RTP_PT} {codec_name}/90000\r\n"
            f"a=fmtp:{RTP_PT} {fmtp}\r\n"
            "a=control:streamid=0\r\n"
        )


def rtp_header(seq, ts, ssrc, marker=False):
    second_byte = ((1 if marker else 0) << 7) | RTP_PT
    return struct.pack(">BBHII", 0x80, second_byte, seq & 0xFFFF, ts & 0xFFFFFFFF,
                       ssrc & 0xFFFFFFFF)


class RtspSource:
    def __init__(self, stream, port):
        self.stream = stream
        self.port = port
        self.ssrc = 0x0BADF00D
        self.seq = 0
        self.started = time.monotonic()

    def _interleave(self, channel, payload):
        return b"$" + bytes([channel]) + struct.pack(">H", len(payload)) + payload

    def _rtcp_sr(self):
        now = time.monotonic()
        ntp_sec = int(now)
        ntp_frac = int((now - ntp_sec) * 2**32)
        rtp_ts = int(now * CLOCK) & 0xFFFFFFFF
        return struct.pack(">BBHIIIIII", 0x80, 200, 6, self.ssrc & 0xFFFFFFFF,
                           ntp_sec & 0xFFFFFFFF, ntp_frac & 0xFFFFFFFF, rtp_ts, 0, 0)

    def _rtp_for(self, nal, ts, marker):
        if self.stream.codec == "h264":
            return self._rtp_h264(nal, ts, marker)
        return self._rtp_h265(nal, ts, marker)

    def _pack(self, ts, payload, marker):
        packet = self._interleave(0, rtp_header(self.seq, ts, self.ssrc, marker) + payload)
        self.seq = (self.seq + 1) & 0xFFFF
        return packet

    def _rtp_h264(self, nal, ts, marker):
        # RFC 3984: single NAL (1-byte header + payload) or FU-A fragments.
        ntype = nal[0] & 0x1F
        chunks = []
        if len(nal) <= 1400:
            chunks.append(nal)
        else:
            fu_ind = (nal[0] & 0xE0) | 28
            for i in range(1, len(nal), 1400):
                piece = nal[i:i + 1400]
                start = 1 if i == 1 else 0
                end = 1 if i + 1400 >= len(nal) else 0
                chunks.append(bytes([fu_ind, (start << 7) | (end << 6) | ntype]) + piece)
        return [self._pack(ts, chunk, marker and idx == len(chunks) - 1)
                for idx, chunk in enumerate(chunks)]

    def _rtp_h265(self, nal, ts, marker):
        # RFC 7798: every RTP packet carries a 2-byte payload header; large
        # NALs use the FU format (payload header Type=49 + FU header).
        ntype = (nal[0] >> 1) & 0x3F
        chunks = []
        if len(nal) <= 1400:
            payload_header = struct.pack(">H", (ntype << 9) | 1)
            chunks.append(payload_header + nal[2:])
        else:
            payload_header = struct.pack(">H", (49 << 9) | 1)
            for i in range(2, len(nal), 1400):
                piece = nal[i:i + 1400]
                start = 1 if i == 2 else 0
                end = 1 if i + 1400 >= len(nal) else 0
                chunks.append(payload_header + bytes([(start << 7) | (end << 6) | ntype]) + piece)
        return [self._pack(ts, chunk, marker and idx == len(chunks) - 1)
                for idx, chunk in enumerate(chunks)]

    def handle(self, conn):
        streaming = threading.Event()
        stream_thread = None
        reader = None
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\r\n\r\n" in buf:
                    head, buf = buf.split(b"\r\n\r\n", 1)
                    lines = head.decode("latin1").split("\r\n")
                    request = lines[0].split(" ")
                    method = request[0] if request else ""
                    cseq = ""
                    for line in lines[1:]:
                        if line.lower().startswith("cseq:"):
                            cseq = line.split(":", 1)[1].strip()
                    self._reply(conn, method, cseq)
                    if method == "PLAY":
                        streaming.set()
                        if stream_thread is None:
                            # After PLAY the client sends interleaved RTCP on
                            # the same connection; drain it so the socket never
                            # fills up and stalls the RTP stream.
                            reader = threading.Thread(target=self._reader, args=(conn,), daemon=True)
                            reader.start()
                            stream_thread = threading.Thread(
                                target=self._stream, args=(conn, streaming), daemon=True)
                            stream_thread.start()
                    elif method == "TEARDOWN":
                        streaming.clear()
                        return
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            streaming.clear()

    def _reader(self, conn):
        # Drain interleaved RTCP (and any other client bytes) so the socket
        # never fills up and stalls the RTP stream.
        try:
            while True:
                prefix = conn.recv(4)
                if len(prefix) < 4:
                    return
                if prefix[0:1] != b"$":
                    continue
                remaining = struct.unpack(">H", prefix[2:4])[0]
                while remaining > 0:
                    chunk = conn.recv(remaining)
                    if not chunk:
                        return
                    remaining -= len(chunk)
        except OSError:
            pass

    def _reply(self, conn, method, cseq):
        if method == "OPTIONS":
            body = ""
            status = "200 OK"
            extra = "Public: OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, GET_PARAMETER\r\n"
        elif method == "DESCRIBE":
            body = self.stream.sdp()
            status = "200 OK"
            extra = f"Content-Type: application/sdp\r\nContent-Length: {len(body)}\r\n"
        elif method == "SETUP":
            body = ""
            status = "200 OK"
            extra = "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n"
        elif method == "PLAY":
            body = ""
            status = "200 OK"
            extra = "Range: npt=0.000-\r\n"
        elif method in ("GET_PARAMETER",):
            body = ""
            status = "200 OK"
            extra = ""
        elif method == "TEARDOWN":
            body = ""
            status = "200 OK"
            extra = ""
        else:
            body = ""
            status = "501 Not Implemented"
            extra = ""
        reply = (f"RTSP/1.0 {status}\r\n"
                 f"CSeq: {cseq}\r\n"
                 f"{extra}"
                 f"Session: 1\r\n"
                 f"Content-Length: {len(body)}\r\n"
                 f"\r\n"
                 f"{body}")
        conn.sendall(reply.encode("latin1"))

    def _stream(self, conn, streaming):
        frame_interval = CLOCK / self.stream.fps
        nals = self.stream.nals
        i = 0
        next_sps = 0
        try:
            while streaming.is_set():
                nal = nals[i]
                ts = int(time.monotonic() * CLOCK) & 0xFFFFFFFF
                # Repeat parameter sets in-band so late joiners can parse.
                if i % 30 == 0 and self.stream.sps is not None:
                    params = ([self.stream.vps] if self.stream.vps else []) + \
                             [self.stream.sps, self.stream.pps]
                    for param in params:
                        if param:
                            conn.sendall(b"".join(self._rtp_for(param, ts, False)))
                marker = i == 0 or self._is_slice_start(nal)
                conn.sendall(b"".join(self._rtp_for(nal, ts, marker)))
                conn.sendall(self._interleave(1, self._rtcp_sr()))
                i = (i + 1) % len(nals)
                time.sleep(frame_interval / CLOCK)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            streaming.clear()

    def _is_slice_start(self, nal):
        if self.stream.codec == "h264":
            return (nal[0] & 0x1F) == 5  # IDR
        return ((nal[0] >> 1) & 0x3F) in (19, 20, 21)  # IDR_W_RADL etc.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--codec", required=True, choices=["h264", "h265"])
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    stream = Stream(args.file, args.codec, args.fps)
    source = RtspSource(stream, args.port)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"rtsp test source: rtsp://127.0.0.1:{args.port}/stream ({args.codec}, {len(stream.nals)} NALs)")
    while True:
        conn, _ = server.accept()
        threading.Thread(target=source.handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
