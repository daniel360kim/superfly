"""RGB student path (2026-10-01): area resize, RGB transport, ChunkPolicy rgb routing.

Run: python -m pytest tests/test_chunk_rgb.py  (needs numpy; onnx + onnxruntime for the
routing tests, cv2 for the INTER_AREA cross-check -- each skipped when missing)."""
import json
import time

import numpy as np
import pytest

from superfly.policies.rgb_preproc import (area_matrix, area_resize, rgb_to_net,
                                           IMAGENET_MEAN, IMAGENET_STD)


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)


def test_area_matrix_rows_sum_to_one():
    for n_in, n_out in ((480, 192), (640, 256), (10, 4)):
        W = area_matrix(n_in, n_out)
        assert np.allclose(W.sum(1), 1.0)
        assert np.allclose(W.sum(0), n_out / n_in)        # every source pixel counted exactly once


def test_area_resize_equals_matrix_form():
    x = _img()
    ref = np.stack([area_matrix(480, 192) @ x[..., c].astype(float) @ area_matrix(640, 256).T
                    for c in range(3)], -1)
    assert np.abs(area_resize(x) - ref).max() < 1e-9


def test_area_resize_matches_cv2_inter_area():
    cv2 = pytest.importorskip("cv2")
    x = _img(1)
    assert np.abs(area_resize(x) - cv2.resize(x.astype(np.float32), (256, 192),
                                              interpolation=cv2.INTER_AREA)).max() < 1e-3
    # uint8 INTER_AREA == round(exact)
    assert np.array_equal(np.rint(area_resize(x)).astype(np.uint8),
                          cv2.resize(x, (256, 192), interpolation=cv2.INTER_AREA))


def test_rgb_to_net_norms_and_layouts():
    x = _img(2)
    a = rgb_to_net(x)                                       # nchw imagenet
    assert a.shape == (1, 3, 192, 256) and a.dtype == np.float32
    b = rgb_to_net(x, norm="in_graph", layout="nhwc")
    assert b.shape == (1, 192, 256, 3) and 0.0 <= b.min() and b.max() <= 1.0
    assert np.allclose(np.transpose(a[0], (1, 2, 0)), (b[0] - IMAGENET_MEAN) / IMAGENET_STD, atol=1e-5)
    u = rgb_to_net(x, norm="uint8")
    assert u.dtype == np.uint8
    rgba = np.concatenate([x, np.full((480, 640, 1), 255, np.uint8)], -1)
    assert np.array_equal(rgb_to_net(rgba), a)              # alpha dropped


def test_rgb_transport_roundtrip(tmp_path):
    from superfly.common.transport import RgbPublisher, RgbSubscriber
    path = tmp_path / "rgb_shm"
    sub = RgbSubscriber(path=path)
    assert sub.latest() is None                             # no publisher yet: lazy open, no frame
    pub = RgbPublisher(path=path)
    assert sub.latest() is None                             # publisher up, nothing written
    frames = [_img(k) for k in range(3)]
    for k in range(9):
        pub.send(frames[k % 3], stamp=k / 30.0)
        if k % 2 == 0:
            f, st, wall = sub.latest_stamped()
            assert np.array_equal(f, frames[k % 3]) and abs(st - k / 30.0) < 1e-9
    f, st, _ = sub.latest_stamped()
    assert np.array_equal(f, frames[8 % 3]) and abs(st - 8 / 30.0) < 1e-9
    assert sub.written == 9 and sub.frames == 5
    rgba = np.concatenate([frames[0], np.zeros((480, 640, 1), np.uint8)], -1)
    pub.send(rgba, stamp=1.0)
    assert np.array_equal(sub.latest(), frames[0])          # alpha dropped
    sub.close()


def test_rgb_transport_concurrent_no_tearing(tmp_path):
    """A writer process at full speed vs a reader: every frame read is one whole frame."""
    import multiprocessing as mp
    from superfly.common.transport import RgbSubscriber
    path = tmp_path / "rgb_shm"
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_writer, args=(str(path), 300))
    p.start()
    sub = RgbSubscriber(path=path)
    seen = 0
    t_end = time.time() + 20
    while p.is_alive() and time.time() < t_end:
        f = sub.latest()
        if f is not None:
            v = int(f[0, 0, 0])
            assert (f == v).all()                           # a torn frame mixes two values
            seen += 1
    p.join()
    assert seen > 0


def _writer(path, n):
    from superfly.common.transport import RgbPublisher
    pub = RgbPublisher(path=path)
    for k in range(n):
        pub.send(np.full((480, 640, 3), k % 251, np.uint8), stamp=k)


def _tiny_rgb_graph(path, layout="nchw", with_depth=True):
    """chunk = 0, gate[h] = mean(rgb input) (+ mean(depth) on head 1) -- exposes what was fed."""
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    shp = [1, 3, 192, 256] if layout == "nchw" else [1, 192, 256, 3]
    ins = [helper.make_tensor_value_info("imu", TensorProto.FLOAT, [1, 1, 22]),
           helper.make_tensor_value_info("prev_chunk", TensorProto.FLOAT, [1, 60]),
           helper.make_tensor_value_info("rgb", TensorProto.FLOAT, shp)]
    nodes = [helper.make_node("ReduceMean", ["rgb"], ["m"], keepdims=0),
             helper.make_node("ReduceMean", ["imu"], ["mi"], keepdims=0),
             helper.make_node("ReduceMean", ["prev_chunk"], ["mp"], keepdims=0)]
    if with_depth:
        ins.append(helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 1, 224, 224, 3]))
        nodes.append(helper.make_node("ReduceMean", ["depth"], ["md"], keepdims=0))
    else:
        nodes.append(helper.make_node("Constant", [], ["md"],
                                      value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0])))
    zero = helper.make_tensor("zc", TensorProto.FLOAT, [1, 5, 60], [0.0] * 300)
    nodes += [helper.make_node("Constant", [], ["chunk"], value=zero),
              helper.make_node("Mul", ["mi", "mp"], ["z0"]),
              helper.make_node("Add", ["m", "z0"], ["m2"]),
              helper.make_node("Constant", [], ["sh"], value=helper.make_tensor(
                  "s", TensorProto.INT64, [2], [1, 1])),
              helper.make_node("Reshape", ["m2", "sh"], ["g0"]),
              helper.make_node("Reshape", ["md", "sh"], ["g1"]),
              helper.make_node("Concat", ["g0", "g1", "g0", "g0", "g0"], ["gate"], axis=1)]
    outs = [helper.make_tensor_value_info("chunk", TensorProto.FLOAT, [1, 5, 60]),
            helper.make_tensor_value_info("gate", TensorProto.FLOAT, [1, 5])]
    m = helper.make_model(helper.make_graph(nodes, "tiny_rgb", ins, outs),
                          opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.save(m, str(path))
    json.dump({"arch": "chunk_v1", "chunk_steps": 15, "chunk_dt": 0.1,
               "heads": ["straight", "left", "right", "over", "under"]},
              open(str(path) + ".json", "w"))


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_chunk_policy_routes_rgb_by_name(tmp_path, layout):
    pytest.importorskip("onnxruntime")
    from superfly.policies.chunk import ChunkPolicy
    p = tmp_path / "s.onnx"
    _tiny_rgb_graph(p, layout)
    pol = ChunkPolicy(p, threads=1)
    assert pol.modality == "rgb" and pol.rgb["layout"] == layout and pol.rgb["size"] == [256, 192]
    assert pol.rgb["norm"] == "imagenet" and not pol.rgb["norm_from_sidecar"]
    x = _img(3)
    pol.decide(0.0, [0, 0, 2], np.eye(3), [0, 0, 0], [0, 0, 0], [5, 0, 2],
               np.ones((224, 224), np.float32), rgb=x)
    g = pol.last["gate"]
    assert abs(g[0] - rgb_to_net(x).mean()) < 1e-5            # the rgb input got the normalised frame
    assert abs(g[1] - 250.0) < 1e-3                            # depth input: blank (20 m -> 250), not the 1 m frame


def test_depth_student_unchanged(tmp_path):
    """A graph without an rgb input keeps the depth routing (rank-5 -> depth)."""
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto
    from superfly.policies.chunk import ChunkPolicy
    ins = [helper.make_tensor_value_info("imu", TensorProto.FLOAT, [1, 1, 22]),
           helper.make_tensor_value_info("x5", TensorProto.FLOAT, [1, 1, 224, 224, 3]),
           helper.make_tensor_value_info("prev_chunk", TensorProto.FLOAT, [1, 60])]
    nodes = [helper.make_node("ReduceMean", ["x5"], ["md"], keepdims=0),
             helper.make_node("ReduceMean", ["imu"], ["mi"], keepdims=0),
             helper.make_node("ReduceMean", ["prev_chunk"], ["mp"], keepdims=0),
             helper.make_node("Mul", ["mi", "mp"], ["z"]),
             helper.make_node("Add", ["md", "z"], ["m"]),
             helper.make_node("Constant", [], ["sh"], value=helper.make_tensor("s", TensorProto.INT64, [2], [1, 1])),
             helper.make_node("Reshape", ["m", "sh"], ["g0"]),
             helper.make_node("Concat", ["g0", "g0", "g0", "g0", "g0"], ["gate"], axis=1),
             helper.make_node("Constant", [], ["chunk"], value=helper.make_tensor(
                 "zc", TensorProto.FLOAT, [1, 5, 60], [0.0] * 300))]
    outs = [helper.make_tensor_value_info("chunk", TensorProto.FLOAT, [1, 5, 60]),
            helper.make_tensor_value_info("gate", TensorProto.FLOAT, [1, 5])]
    m = helper.make_model(helper.make_graph(nodes, "tiny_depth", ins, outs),
                          opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    p = tmp_path / "d.onnx"
    onnx.save(m, str(p))
    json.dump({"arch": "chunk_v1"}, open(str(p) + ".json", "w"))
    pol = ChunkPolicy(p, threads=1)
    assert pol.modality == "depth" and pol.rgb is None
    pol.decide(0.0, [0, 0, 2], np.eye(3), [0, 0, 0], [0, 0, 0], [5, 0, 2], np.ones((224, 224), np.float32))
    assert abs(pol.last["gate"][0] - 1000.0 / 80.0) < 1e-3      # 1 m -> mm/80 = 12.5


def test_quantized_input_equals_training_shards():
    """rgb_to_net (default quantize) == cv2.INTER_AREA uint8 / 255 (the W3 shard path) exactly."""
    cv2 = pytest.importorskip("cv2")
    x = _img(5)
    q = rgb_to_net(x, norm="in_graph")[0].transpose(1, 2, 0)
    ref = cv2.resize(x, (256, 192), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    assert np.array_equal(q, ref)
    f = rgb_to_net(x, norm="in_graph", quantize=False)[0].transpose(1, 2, 0)
    assert 0 < np.abs(f - ref).max() <= 0.5 / 255 + 1e-6
