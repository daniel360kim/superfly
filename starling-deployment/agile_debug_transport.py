"""UDP transport for Agile Autonomy debug viz (offboard -> sim).

The sim renders an overhead map (obstacles, goal, drone, predicted trajectories).
Trajectories are in PX4-local ENU (same frame as LOCAL_POSITION_NED -> ENU in the
offboard). The sim converts to world ENU via world = spawn_pos + local.

Wire format (little-endian, UDP port AGILE_DEBUG_PORT):
    magic b'AGDB' | uint8 version=1 | uint32 seq
    uint8 n_modes | uint8 n_wp | uint8 mode_idx | uint8 tracker
    float32 pos[3]   local ENU drone position
    float32 yaw      ENU yaw [rad] (body +x heading)
    float32 alphas[3]
    float32 trajectories[n_modes * n_wp * 3]  row-major, local ENU waypoints
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np

AGILE_DEBUG_PORT = 15002
_MAGIC = b"AGDB"
_VERSION = 1
_HEADER = struct.Struct("<4sBIBBBBB")  # magic, ver, seq, n_modes, n_wp, mode_idx, tracker, pad
_META = struct.Struct("<3f f 3f")    # pos[3], yaw, alphas[3]


@dataclass
class AgileDebugFrame:
    seq: int
    pos_local: np.ndarray          # (3,) float64
    yaw: float
    alphas: np.ndarray             # (3,) float64
    trajectories_local: np.ndarray # (n_modes, n_wp, 3) float64
    mode_idx: int
    tracker: str                   # "mpc" | "pd"


def pack_debug_frame(
    seq: int,
    pos_local: np.ndarray,
    yaw: float,
    alphas: np.ndarray,
    trajectories_local: np.ndarray,
    mode_idx: int,
    tracker: str,
) -> bytes:
    traj = np.asarray(trajectories_local, dtype=np.float32)
    n_modes, n_wp, _ = traj.shape
    if n_modes > 3 or n_wp > 16:
        raise ValueError(f"trajectory shape {traj.shape} exceeds wire limits")
    pos = np.asarray(pos_local, dtype=np.float32).reshape(3)
    alph = np.asarray(alphas, dtype=np.float32).reshape(3)
    tracker_id = 0 if tracker == "mpc" else 1
    header = _HEADER.pack(_MAGIC, _VERSION, int(seq), n_modes, n_wp, mode_idx, tracker_id, 0)
    meta = _META.pack(float(pos[0]), float(pos[1]), float(pos[2]), float(yaw),
                      float(alph[0]), float(alph[1]), float(alph[2]))
    return header + meta + traj.reshape(-1).tobytes()


def unpack_debug_frame(data: bytes) -> AgileDebugFrame | None:
    if len(data) < _HEADER.size + _META.size:
        return None
    magic, ver, seq, n_modes, n_wp, mode_idx, tracker_id, _pad = _HEADER.unpack_from(data, 0)
    if magic != _MAGIC or ver != _VERSION or n_modes == 0 or n_wp == 0:
        return None
    need = _HEADER.size + _META.size + n_modes * n_wp * 3 * 4
    if len(data) != need:
        return None
    pos_x, pos_y, pos_z, yaw, a0, a1, a2 = _META.unpack_from(data, _HEADER.size)
    off = _HEADER.size + _META.size
    traj = np.frombuffer(data, dtype=np.float32, count=n_modes * n_wp * 3, offset=off)
    traj = traj.reshape(n_modes, n_wp, 3).astype(np.float64)
    return AgileDebugFrame(
        seq=int(seq),
        pos_local=np.array([pos_x, pos_y, pos_z], dtype=np.float64),
        yaw=float(yaw),
        alphas=np.array([a0, a1, a2], dtype=np.float64),
        trajectories_local=traj,
        mode_idx=int(mode_idx),
        tracker="mpc" if tracker_id == 0 else "pd",
    )


class AgileDebugPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = AGILE_DEBUG_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send(self, frame: AgileDebugFrame) -> None:
        payload = pack_debug_frame(
            self._seq,
            frame.pos_local,
            frame.yaw,
            frame.alphas,
            frame.trajectories_local,
            frame.mode_idx,
            frame.tracker,
        )
        self._sock.sendto(payload, self._addr)
        self._seq += 1


class AgileDebugSubscriber:
    def __init__(self, host: str = "127.0.0.1", port: int = AGILE_DEBUG_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.setblocking(False)
        self._last: AgileDebugFrame | None = None

    def latest(self) -> AgileDebugFrame | None:
        while True:
            try:
                data = self._sock.recv(65535)
            except BlockingIOError:
                break
            frame = unpack_debug_frame(data)
            if frame is not None:
                self._last = frame
        return self._last
