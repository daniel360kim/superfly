"""Lossless fragmented UDP transport for 224x224 RGB policy frames."""

from __future__ import annotations

import socket
import struct
import sys
import zlib

import numpy as np


RGB_PORT = 15003
MAGIC = b"SFRG"
VERSION = 1
_HEADER = struct.Struct("<4sB3xIIIII")
_CHUNK = 60000


class RGBPublisher:
    """Send uint8 RGB frames losslessly, fragmenting above the UDP limit."""

    def __init__(self, host: str = "127.0.0.1", port: int = RGB_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0
        self._send_errors = 0

    def send(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"RGB frame must be HxWx3, got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        h, w, _ = frame.shape
        body = zlib.compress(frame.tobytes(), 3)
        n_parts = max(1, -(-len(body) // _CHUNK))
        try:
            for part in range(n_parts):
                chunk = body[part * _CHUNK:(part + 1) * _CHUNK]
                packet = _HEADER.pack(
                    MAGIC, VERSION, self._seq, h, w, part, n_parts
                ) + chunk
                self._sock.sendto(packet, self._addr)
        except OSError as exc:
            self._send_errors += 1
            if self._send_errors < 3 or self._send_errors % 100 == 0:
                print(
                    f"[rgb_transport] send failed ({exc}); "
                    f"{self._send_errors} frames dropped so far.",
                    file=sys.stderr,
                    flush=True,
                )
        self._seq += 1


class RGBSubscriber:
    """Drain UDP and return the newest completely reassembled RGB frame."""

    def __init__(self, host: str = "127.0.0.1", port: int = RGB_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._last = None
        self._seq = None
        self._shape = None
        self._n_parts = 0
        self._parts = {}

    def latest(self):
        while True:
            try:
                data = self._sock.recv(65535)
            except BlockingIOError:
                break
            if len(data) < _HEADER.size:
                continue
            magic, version, seq, h, w, part, n_parts = _HEADER.unpack_from(data)
            if (magic != MAGIC or version != VERSION or h == 0 or w == 0
                    or n_parts == 0 or part >= n_parts):
                continue
            if seq != self._seq:
                self._seq = seq
                self._shape = (h, w)
                self._n_parts = n_parts
                self._parts = {}
            if self._shape != (h, w) or self._n_parts != n_parts:
                continue
            self._parts[part] = data[_HEADER.size:]
            if len(self._parts) != self._n_parts:
                continue
            try:
                packed = b"".join(self._parts[i] for i in range(self._n_parts))
                raw = zlib.decompress(packed)
            except (KeyError, zlib.error):
                self._parts = {}
                continue
            if len(raw) != h * w * 3:
                self._parts = {}
                continue
            self._last = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            self._parts = {}
        return self._last
