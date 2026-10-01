"""RGB student input preprocessing -- ONE function shared by the Isaac harness,
the training loader and onboard (numpy only, no cv2 / torch / PIL needed).

Contract (RGB_START_PROPOSAL 7, Starling nose camera, 2026-10-01):
  camera   640x480 (4:3), 87 deg HFOV rectilinear, square pixels, uint8 RGB
           (channel order R, G, B; row 0 = top, col 0 = left)
  resize   ONE antialiased AREA resize 640x480 -> 256x192 (factor 2.5): every
           output pixel is the exact area-weighted mean of the source pixels its
           footprint covers (= cv2.INTER_AREA = PIL Image.BOX, to float rounding).
           NOT torch F.interpolate(mode='area'), which is adaptive_avg_pool and
           takes floor/ceil windows at a non-integer factor (up to 1/2 px shift
           and a different kernel), and NOT plain bilinear (point-samples thin
           obstacles; PRE_RGB_FINDINGS 2.3).
  scale    /255 -> [0, 1] float32
  norm     ImageNet mean/std, either here (norm="imagenet") or inside the ONNX
           graph (norm="in_graph": this function stops at [0, 1])
  layout   NCHW (1, 3, 192, 256) unless the graph asks for NHWC

The resize is separable: out = Wy @ img @ Wx^T per channel, with Wy (192x480)
and Wx (256x640) the exact 1-D area-overlap matrices (rows sum to 1). Computed
in float64 then cast, so the result does not depend on the caller's BLAS.
"""

from functools import lru_cache

import numpy as np

SRC_W, SRC_H = 640, 480
NET_W, NET_H = 256, 192
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@lru_cache(maxsize=16)
def area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) exact area-overlap weights: output cell j covers source
    interval [j*s, (j+1)*s), s = n_in / n_out; weight = overlap length / s."""
    s = n_in / n_out
    W = np.zeros((n_out, n_in), np.float64)
    for j in range(n_out):
        a, b = j * s, (j + 1) * s
        i0, i1 = int(np.floor(a)), int(np.ceil(b))
        for i in range(i0, min(i1, n_in)):
            ov = min(b, i + 1) - max(a, i)
            if ov > 0:
                W[j, i] = ov / s
    W.setflags(write=False)
    return W


@lru_cache(maxsize=16)
def area_taps(n_in: int, n_out: int):
    """area_matrix as K taps per output: (idx (K, n_out) int, w (K, n_out))."""
    W = area_matrix(n_in, n_out)
    K = int((W > 0).sum(1).max())
    idx = np.zeros((K, n_out), np.int64)
    wt = np.zeros((K, n_out), np.float64)
    for j in range(n_out):
        nz = np.flatnonzero(W[j])
        idx[:len(nz), j], wt[:len(nz), j] = nz, W[j, nz]
        idx[len(nz):, j] = nz[-1]                       # zero-weight padding
    return idx, wt


def area_resize(img, out_wh=(NET_W, NET_H)) -> np.ndarray:
    """Exact area (box) resize of an (H, W[, C]) image to out_wh = (W, H).
    Returns float64 in the input's value range (no rounding). Separable, as
    K-tap gathers (K = 4 at 2.5x): ~ms, no BLAS. Equals area_matrix products."""
    x = np.asarray(img)
    h, w = x.shape[:2]
    iy, wy = area_taps(h, out_wh[1])
    ix, wx = area_taps(w, out_wh[0])
    ex = (slice(None),) + (None,) * (x.ndim - 1)
    y = sum(wy[k][ex] * x[iy[k]] for k in range(len(iy)))       # (H', W[, C]) float64
    ex = (None, slice(None)) + (None,) * (x.ndim - 2)
    return sum(wx[k][ex] * y[:, ix[k]] for k in range(len(ix)))  # (H', W'[, C])


def rgb_to_net(img_u8, norm: str = "imagenet", layout: str = "nchw",
               out_wh=(NET_W, NET_H), dtype=np.float32) -> np.ndarray:
    """Camera frame (H, W, 3|4) uint8 RGB(A) -> network input with a batch dim.
    norm: "imagenet" (mean/std here) | "in_graph" or "none" ([0, 1] only) |
    "uint8" (area-resized, rounded, uint8 -- for a graph that takes bytes)."""
    x = np.asarray(img_u8)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {x.shape}")
    x = x[..., :3]
    if x.shape[:2] != (out_wh[1], out_wh[0]):
        y = area_resize(x, out_wh)
    else:
        y = x.astype(np.float64)
    if norm == "uint8":
        y = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    else:
        y = (y / 255.0).astype(np.float32)
        if norm == "imagenet":
            y = (y - IMAGENET_MEAN) / IMAGENET_STD
        elif norm not in ("in_graph", "none"):
            raise ValueError(f"unknown norm {norm!r}")
        y = y.astype(dtype)
    if layout == "nchw":
        y = np.transpose(y, (2, 0, 1))
    elif layout != "nhwc":
        raise ValueError(f"unknown layout {layout!r}")
    return np.ascontiguousarray(y[None])


def blank_frame(w: int = SRC_W, h: int = SRC_H) -> np.ndarray:
    """Mid-grey camera frame (before the first real one / forward timing)."""
    return np.full((h, w, 3), 128, np.uint8)
