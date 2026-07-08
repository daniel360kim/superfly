#!/usr/bin/env python
"""
Shared depth-frame transport between the Isaac Sim process (publisher) and the
offboard policy process (subscriber).

Design rationale (verified against DiffPhysDrone source):
  - The native sim renders depth at H=48, W=64 (env_cuda.py: Env(B, 64, 48)),
    then the POLICY applies `3/d.clamp(0.3,24) - 0.6` and `F.max_pool2d(.,4,4)`
    to get 12x16 (main_cuda.py:156-157). We therefore ship the RAW 48x64 metric
    depth and let the policy process do the normalization+pooling itself — this
    keeps the wire format identical to what a real depth-camera driver hands us
    on the Starling/VOXL2.
  - The native depth value is planar/optical-axis Z-depth (the render ray has a
    unit forward component; quadsim_kernel.cu:34-39,157), so the Isaac Sim camera
    must publish "distance to image plane" (Z-depth), NOT Euclidean range.

Wire format (little-endian):
    uint32 seq | uint32 height | uint32 width | uint32 codec | body
Depth is in metres; no-return / non-finite pixels must be set to a large value
(>= 24.0) by the publisher. Two body encodings (`codec`):
    0 = raw float32[height*width] metres (the small-frame policies).
    1 = zlib(uint16[height*width] millimetres). A single UDP datagram caps at
        65507 bytes, so a 224x224 float32 frame (200 KB) will not fit; the agile
        path (which ships a 224x224 depth to match Loquercio's training input)
        uses this codec -- uint16 mm halves the raw size and zlib takes a real
        depth frame to ~10-15 KB. The subscriber returns float32 metres either
        way, so callers are codec-agnostic.
"""

import socket
import struct
import zlib
import numpy as np

DEPTH_PORT = 15001            # local UDP port for depth frames
RENDER_H, RENDER_W = 48, 64   # resolution the policy expects BEFORE 4x4 pooling

CODEC_RAW_F32 = 0
CODEC_ZLIB_U16_MM = 1

_HEADER = struct.Struct("<IIII")  # seq, height, width, codec


class DepthPublisher:
    """Sends metric depth frames over UDP from the sim process (metres)."""

    def __init__(self, host="127.0.0.1", port=DEPTH_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, depth_m: np.ndarray, compress: bool = False):
        """depth_m: (H, W) float32 array in metres, already oriented to match the
        native convention (row 0 = top/up, col 0 = left).

        compress=True encodes the frame as zlib(uint16 mm) so a 224x224 depth
        fits a single UDP datagram (raw float32 would be 200 KB > 65507). The
        subscriber transparently decodes back to float32 metres."""
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth_m.shape
        if compress:
            mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
            body = zlib.compress(mm.tobytes(), 6)
            codec = CODEC_ZLIB_U16_MM
        else:
            body = depth_m.tobytes()
            codec = CODEC_RAW_F32
        payload = _HEADER.pack(self._seq, h, w, codec) + body
        self._sock.sendto(payload, self._addr)
        self._seq += 1


class DepthSubscriber:
    """Receives the latest depth frame in the policy process (non-blocking)."""

    def __init__(self, host="127.0.0.1", port=DEPTH_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._last = None  # (H, W) float32 metres

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
                    raw = zlib.decompress(body)
                    if len(raw) != h * w * 2:
                        continue
                    frame = (np.frombuffer(raw, dtype=np.uint16, count=h * w)
                             .astype(np.float32) / 1000.0)
                else:
                    continue
            except (zlib.error, ValueError):
                continue
            self._last = frame.reshape(h, w)
        return self._last
