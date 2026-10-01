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

Policy RGB (RGB students, 2026-10-01; RgbPublisher / RgbSubscriber). The
frame is the camera's own 640x480 uint8 RGB -- what a real nose-camera driver
hands the policy; the policy process does the resize
(superfly.policies.rgb_preproc), exactly as onboard. It does NOT go over UDP:
a 0.9 MB frame is 16 datagrams, the kernel caps the receive buffer at
net.core.rmem_max = 212992 B (gs2 and airstation03), and with the offboard's
control loop holding the GIL a receive thread completed 10 of 150 frames at
30 Hz in a loopback stress test (W5, 2026-10-01). Instead the sim writes the
newest frame into a shared-memory file, /dev/shm/superfly_rgb_<uid>_<i>
(superfly.common.instance.rgb_shm_path; i = SUPERFLY_INSTANCE or "x"), under
a seqlock; the subscriber copies the newest complete frame on demand. Never
blocks, never tears, never drops the newest frame. Layout (little-endian):
    0  u32 magic 0x52474231 ("RGB1") | 4 u32 version 1 | 8 u64 seq (odd while writing)
   16  u32 height | 20 u32 width | 24 f64 stamp (sim time, s) | 32 u64 frames written
   40  f64 wall time of the write (time.time(), s)
   64  uint8[height*width*3] RGB, row 0 = top, col 0 = left
"""

import socket
import struct
import sys
import zlib
import numpy as np

import mmap
import os
import time

from superfly.common.instance import depth_port, rgb_shm_path

# local UDP port for depth frames: 15001, +10 per SUPERFLY_INSTANCE (parallel
# campaigns; unset = 15001)
DEPTH_PORT = depth_port()

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



_SHM_MAGIC = 0x52474231
_SHM_HDR = 64
_SHM_MAX = (640 * 2) * (480 * 2) * 3          # room for up to 1280x960 RGB


class RgbPublisher:
    """Writes the newest uint8 RGB policy frame into the shared-memory seqlock
    (module docstring). Never raises: a failed write loses one frame."""

    def __init__(self, path=None, max_bytes=_SHM_MAX):
        self.path = rgb_shm_path() if path is None else str(path)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, _SHM_HDR + int(max_bytes))
            self._mm = mmap.mmap(fd, _SHM_HDR + int(max_bytes))
        finally:
            os.close(fd)
        self._cap = int(max_bytes)
        self._mm[0:_SHM_HDR] = bytes(_SHM_HDR)                 # fresh session: seq 0, no frame
        struct.pack_into("<II", self._mm, 0, _SHM_MAGIC, 1)
        self._seq = 0
        self._n = 0
        self._errors = 0

    def send(self, rgb: np.ndarray, stamp: float = 0.0):
        """rgb: (H, W, 3|4) uint8, row 0 = top, col 0 = left (alpha dropped).
        stamp: the frame's sim time [s]."""
        try:
            a = np.asarray(rgb)[..., :3]
            h, w = a.shape[:2]
            if h * w * 3 > self._cap:
                raise ValueError(f"frame {h}x{w} exceeds the {self._cap} B buffer")
            self._seq += 1                                      # odd: writing
            struct.pack_into("<Q", self._mm, 8, self._seq)
            struct.pack_into("<IId", self._mm, 16, h, w, float(stamp))
            dst = np.frombuffer(self._mm, dtype=np.uint8, count=h * w * 3, offset=_SHM_HDR)
            dst[:] = np.ascontiguousarray(a, dtype=np.uint8).reshape(-1)
            del dst
            self._n += 1
            struct.pack_into("<Qd", self._mm, 32, self._n, time.time())
            self._seq += 1                                      # even: complete
            struct.pack_into("<Q", self._mm, 8, self._seq)
        except Exception as e:
            self._errors += 1
            if self._errors < 3 or self._errors % 100 == 0:
                print(f"[rgb_transport] write failed ({e}); {self._errors} frames lost so far.",
                      file=sys.stderr, flush=True)


class RgbSubscriber:
    """Reads the newest complete RGB frame (H, W, 3) uint8 from the shared-memory
    seqlock (module docstring). Opens the file lazily (the sim may start
    later); latest() never blocks and returns None until a frame exists."""

    def __init__(self, path=None):
        self.path = rgb_shm_path() if path is None else str(path)
        self._mm = None
        self._last = None
        self._last_seq = -1
        self._last_stamp = None
        self._last_wall = None
        self.frames = 0          # distinct complete frames read
        self.incomplete = 0      # reads that met a frame being written (retried)
        self.written = 0         # frames the publisher has written (its counter)

    def _open(self):
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError:
            return False
        try:
            size = os.fstat(fd).st_size
            if size < _SHM_HDR:
                return False
            self._mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        return True

    def _read(self):
        mm = self._mm
        for _ in range(4):
            magic, = struct.unpack_from("<I", mm, 0)
            s1, = struct.unpack_from("<Q", mm, 8)
            if magic != _SHM_MAGIC or s1 == 0:
                return None
            if s1 & 1:
                self.incomplete += 1
                time.sleep(0.0005)
                continue
            if s1 == self._last_seq:
                return "same"
            h, w, st = struct.unpack_from("<IId", mm, 16)
            n, wall = struct.unpack_from("<Qd", mm, 32)
            if h * w * 3 > len(mm) - _SHM_HDR:
                return None
            frame = np.frombuffer(mm, dtype=np.uint8, count=h * w * 3, offset=_SHM_HDR).copy()
            s2, = struct.unpack_from("<Q", mm, 8)
            if s2 != s1:
                self.incomplete += 1
                continue
            self._last, self._last_seq = frame.reshape(h, w, 3), s1
            self._last_stamp, self._last_wall, self.written = st, wall, n
            self.frames += 1
            return "new"
        return None

    def latest(self):
        if self._mm is None and not self._open():
            return None
        self._read()
        return self._last

    def latest_stamped(self):
        """(frame, publisher sim time [s], publisher wall time of the write [s]) or Nones."""
        f = self.latest()
        return f, (self._last_stamp if f is not None else None), (self._last_wall if f is not None else None)

    def close(self):
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
