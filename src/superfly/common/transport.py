#!/usr/bin/env python
"""
Shared depth-frame transport between the Isaac Sim process (publisher) and the
offboard policy process (subscriber).

Design rationale:
  - The sim ships the RAW metric depth at each method's native training
    resolution and lets the policy process do its own normalization/pooling —
    this keeps the wire format identical to what a real depth-camera driver
    hands us on the Starling/VOXL2.
  - The depth value is planar/optical-axis Z-depth ("distance to image
    plane"), NOT Euclidean range; methods that train on range (diffaero)
    convert in their own perception builder.

Wire format (little-endian):
    uint32 seq | uint32 height | uint32 width | uint32 codec | body
Depth is in metres; no-return / non-finite pixels must be set to a large value
(>= 24.0) by the publisher. Three body encodings (`codec`):
    0 = raw float32[height*width] metres (the small-frame policies).
    1 = zlib(uint16[height*width] millimetres). A single UDP datagram caps at
        65507 bytes, so a 224x224 float32 frame (200 KB) will not fit; the agile
        path (which ships a 224x224 depth to match Loquercio's training input)
        uses this codec -- uint16 mm halves the raw size and zlib takes a
        typical depth frame to ~10-15 KB.
    2 = fragmented codec 1: same zlib(uint16 mm) stream, split across several
        datagrams when the compressed frame exceeds one datagram. High-entropy
        views (a km-scale USD scene full of gravel/vegetation at mm precision)
        compress to >65507 bytes, which made sendto() raise EMSGSIZE and killed
        the sim loop mid-flight (observed on ConstructionSite, 2026-07-09).
        Header gains `uint32 part | uint32 n_parts` before the chunk bytes; the
        subscriber reassembles by seq and discards incomplete frames.
The subscriber returns float32 metres for every codec, so callers are
codec-agnostic. send() never raises on transport errors -- losing one depth
frame must never kill the sim loop.
"""

import socket
import struct
import sys
import zlib
import numpy as np

DEPTH_PORT = 15001            # local UDP port for depth frames

CODEC_RAW_F32 = 0
CODEC_ZLIB_U16_MM = 1
CODEC_ZLIB_U16_MM_FRAG = 2

_HEADER = struct.Struct("<IIII")       # seq, height, width, codec
_FRAG_HEADER = struct.Struct("<IIIIII")  # seq, height, width, codec, part, n_parts

# Max body bytes per datagram: 65507 (UDP/IPv4 payload cap) minus headers,
# rounded down with margin.
_CHUNK = 60000


class DepthPublisher:
    """Sends metric depth frames over UDP from the sim process (metres)."""

    def __init__(self, host="127.0.0.1", port=DEPTH_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0
        self._send_errors = 0

    def send(self, depth_m: np.ndarray, compress: bool = False):
        """depth_m: (H, W) float32 array in metres, already oriented to match the
        native convention (row 0 = top/up, col 0 = left).

        compress=True encodes the frame as zlib(uint16 mm) so a 224x224 depth
        fits a single UDP datagram (raw float32 would be 200 KB > 65507); frames
        that compress to more than one datagram are fragmented (codec 2). The
        subscriber transparently decodes back to float32 metres. Never raises
        on transport errors."""
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth_m.shape
        if compress:
            mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
            body = zlib.compress(mm.tobytes(), 6)
            if len(body) <= _CHUNK:
                packets = [_HEADER.pack(self._seq, h, w, CODEC_ZLIB_U16_MM) + body]
            else:
                n_parts = -(-len(body) // _CHUNK)
                packets = [
                    _FRAG_HEADER.pack(self._seq, h, w, CODEC_ZLIB_U16_MM_FRAG,
                                      i, n_parts)
                    + body[i * _CHUNK:(i + 1) * _CHUNK]
                    for i in range(n_parts)
                ]
        else:
            packets = [_HEADER.pack(self._seq, h, w, CODEC_RAW_F32)
                       + depth_m.tobytes()]
        try:
            for pkt in packets:
                self._sock.sendto(pkt, self._addr)
        except OSError as e:
            # A dropped frame is recoverable (the subscriber keeps the last one);
            # an exception here killed the sim loop mid-flight once. Warn sparsely.
            self._send_errors += 1
            if self._send_errors < 3 or self._send_errors % 100 == 0:
                print(f"[depth_transport] send failed ({e}); "
                      f"{self._send_errors} frames dropped so far.",
                      file=sys.stderr, flush=True)
        self._seq += 1


class DepthSubscriber:
    """Receives the latest depth frame in the policy process (non-blocking)."""

    def __init__(self, host="127.0.0.1", port=DEPTH_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._last = None  # (H, W) float32 metres
        # codec-2 reassembly state: parts of the frame currently being received
        self._frag_seq = None
        self._frag_parts = {}   # part index -> chunk bytes
        self._frag_n = 0

    def _decode_zlib_u16_mm(self, body, h, w):
        raw = zlib.decompress(body)
        if len(raw) != h * w * 2:
            return None
        return (np.frombuffer(raw, dtype=np.uint16, count=h * w)
                .astype(np.float32) / 1000.0)

    def latest(self):
        """Drain the socket and return the most recent depth frame (metres), or
        the last-seen frame, or None if nothing has arrived yet."""
        while True:
            try:
                data = self._sock.recv(1 << 20)
            except BlockingIOError:
                break
            if len(data) < _HEADER.size:
                continue
            seq, h, w, codec = _HEADER.unpack_from(data, 0)
            body = data[_HEADER.size:]
            try:
                if codec == CODEC_RAW_F32:
                    if len(body) != h * w * 4:
                        continue
                    frame = np.frombuffer(body, dtype=np.float32, count=h * w)
                elif codec == CODEC_ZLIB_U16_MM:
                    frame = self._decode_zlib_u16_mm(body, h, w)
                    if frame is None:
                        continue
                elif codec == CODEC_ZLIB_U16_MM_FRAG:
                    if len(data) < _FRAG_HEADER.size:
                        continue
                    seq, h, w, codec, part, n_parts = _FRAG_HEADER.unpack_from(data, 0)
                    if n_parts == 0 or part >= n_parts:
                        continue
                    if seq != self._frag_seq:      # new frame: drop stale parts
                        self._frag_seq = seq
                        self._frag_parts = {}
                        self._frag_n = n_parts
                    self._frag_parts[part] = data[_FRAG_HEADER.size:]
                    if len(self._frag_parts) < self._frag_n:
                        continue
                    frame = self._decode_zlib_u16_mm(
                        b"".join(self._frag_parts[i] for i in range(self._frag_n)),
                        h, w)
                    self._frag_seq, self._frag_parts = None, {}
                    if frame is None:
                        continue
                else:
                    continue
            except (zlib.error, ValueError):
                continue
            self._last = frame.reshape(h, w)
        return self._last
