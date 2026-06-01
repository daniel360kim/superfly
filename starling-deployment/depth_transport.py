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
    uint32 seq | uint32 height | uint32 width | float32[height*width] depth_m
Depth is in metres; no-return / non-finite pixels must be set to a large value
(>= 24.0) by the publisher.
"""

import socket
import struct
import numpy as np

DEPTH_PORT = 15001            # local UDP port for depth frames
RENDER_H, RENDER_W = 48, 64   # resolution the policy expects BEFORE 4x4 pooling

_HEADER = struct.Struct("<III")  # seq, height, width


class DepthPublisher:
    """Sends 48x64 float32 metric depth frames over UDP from the sim process."""

    def __init__(self, host="127.0.0.1", port=DEPTH_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, depth_m: np.ndarray):
        """depth_m: (H, W) float32 array in metres, already oriented to match the
        native convention (row 0 = top/up, col 0 = left)."""
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth_m.shape
        payload = _HEADER.pack(self._seq, h, w) + depth_m.tobytes()
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
            seq, h, w = _HEADER.unpack_from(data, 0)
            if len(data) != _HEADER.size + h * w * 4:
                continue
            self._last = np.frombuffer(
                data, dtype=np.float32, count=h * w, offset=_HEADER.size
            ).reshape(h, w)
        return self._last
