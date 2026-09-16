"""Loquercio et al. agile_autonomy (uzh-rpg/agile_autonomy) PlaNet trajectory
network: Keras model definition + TF2 checkpoint loading + inference.

Ported from SAFE_Benchmark's integrations/aerial_nav/agents/model_agile_autonomy.py
(itself a Keras re-implementation of agile_autonomy's planner_learning network),
so this deployment never imports from SAFE_Benchmark at runtime. The model code
is a faithful copy -- any behavioural deviation from SAFE is called out with a
"DEVIATION:" comment (there are none in this file; the known SAFE bugs are in
its reference/tracking layer, fixed in agile_mpc.py / agile_core.py).

The network maps (depth image sequence, 21-dim IMU-style state) -> `modes`
candidate local trajectories, each `1 (alpha cost) + 3*out_seq_len` numbers:
out_seq_len waypoints at a fixed 0.1 s spacing in the CAMERA/BODY frame.
Checkpoints are TF2 object checkpoints saved as a PREFIX (ckpt-50.index +
ckpt-50.data-*); pass the prefix or a directory containing one.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class LoquercioModelConfig:
    img_width: int = 224
    img_height: int = 224
    seq_len: int = 1
    modes: int = 3
    state_dim: int = 3
    out_seq_len: int = 10
    use_rgb: bool = False
    use_depth: bool = True
    use_position: bool = False
    use_attitude: bool = True
    use_bodyrates: bool = True
    freeze_backbone: bool = False

    @property
    def raw_state_dim(self) -> int:
        return 21 if self.use_bodyrates else 18

    @property
    def output_dim_per_mode(self) -> int:
        return 1 + self.state_dim * self.out_seq_len


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the agile method. Run this under "
            "starling-deployment/agile_python.sh (the agile venv at "
            "starling-deployment/.venv)."
        ) from exc
    return tf


def create_network(config: LoquercioModelConfig):
    tf = _tf()
    return PlaNet(tf, config)


class PlaNet:
    """Keras model structure used by uzh-rpg/agile_autonomy planner_learning."""

    def __init__(self, tf, config: LoquercioModelConfig):
        self.tf = tf
        self.config = config
        self.model = self._create_model()

    def __call__(self, inputs):
        return self.model(inputs)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def variables(self):
        return self.model.variables

    def _create_model(self):
        tf = self.tf
        config = self.config

        class _PlaNetModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                channels = 3 * int(config.use_rgb) + 3 * int(config.use_depth)
                input_size = (config.img_height, config.img_width, channels)

                if config.use_rgb or config.use_depth:
                    self.backbone = [
                        tf.keras.applications.MobileNet(
                            include_top=False,
                            weights=None,
                            input_shape=input_size,
                            pooling=None,
                        )
                    ]
                    self.backbone[0].trainable = not config.freeze_backbone
                    self.resize_op = [tf.keras.layers.Conv1D(128, 1, padding='valid')]
                    self.img_mergenet = [
                        tf.keras.layers.Conv1D(128, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(64, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(64, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(32, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                    ]
                    self.resize_op_2 = [
                        tf.keras.layers.Conv1D(config.modes, 3, padding='valid')
                    ]

                self.states_conv = [
                    tf.keras.layers.Conv1D(64, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                ]
                self.resize_op_3 = [
                    tf.keras.layers.Conv1D(config.modes, 3, padding='valid')
                ]
                output_dim = config.output_dim_per_mode
                self.plan_module = [
                    tf.keras.layers.Conv1D(64, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(128, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(128, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(output_dim, 1, padding='same'),
                ]

            def call(self, inputs):
                if config.use_position:
                    imu_obs = inputs['imu']
                else:
                    imu_obs = inputs['imu'][:, :, 3:]
                if not config.use_attitude:
                    if config.use_position:
                        raise ValueError('Loquercio config cannot use position without attitude.')
                    imu_obs = inputs['imu'][:, :, 12:]

                imu_embeddings = self._imu_branch(imu_obs)
                img_embeddings = self._preprocess_frames(inputs)
                if img_embeddings is not None:
                    total_embeddings = tf.concat((img_embeddings, imu_embeddings), axis=-1)
                else:
                    total_embeddings = imu_embeddings
                return self._plan_branch(total_embeddings)

            def _conv_branch(self, image):
                x = tf.keras.applications.mobilenet.preprocess_input(image)
                for layer in self.backbone:
                    x = layer(x)
                x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
                for layer in self.resize_op:
                    x = layer(x)
                return tf.reshape(x, (tf.shape(x)[0], -1))

            def _image_branch(self, img_seq):
                img_fts = tf.map_fn(
                    self._conv_branch,
                    elems=img_seq,
                    parallel_iterations=config.seq_len,
                    fn_output_signature=tf.float32,
                )
                img_fts = tf.transpose(img_fts, (1, 0, 2))
                x = img_fts
                for layer in self.img_mergenet:
                    x = layer(x)
                x = tf.transpose(x, (0, 2, 1))
                for layer in self.resize_op_2:
                    x = layer(x)
                return tf.transpose(x, (0, 2, 1))

            def _imu_branch(self, embeddings):
                x = embeddings
                for layer in self.states_conv:
                    x = layer(x)
                x = tf.transpose(x, (0, 2, 1))
                for layer in self.resize_op_3:
                    x = layer(x)
                return tf.transpose(x, (0, 2, 1))

            def _plan_branch(self, embeddings):
                x = embeddings
                for layer in self.plan_module:
                    x = layer(x)
                return x

            def _preprocess_frames(self, inputs):
                if config.use_rgb and config.use_depth:
                    img_seq = tf.concat((inputs['rgb'], inputs['depth']), axis=-1)
                elif config.use_rgb:
                    img_seq = inputs['rgb']
                elif config.use_depth:
                    img_seq = inputs['depth']
                else:
                    return None
                img_seq = tf.transpose(img_seq, (1, 0, 2, 3, 4))
                return self._image_branch(img_seq)

        return _PlaNetModel()


# ---------------------------------------------------------------------------
# TensorFlow checkpoint loading + inference
# ---------------------------------------------------------------------------

def resolve_checkpoint_prefix(path: str) -> str:
    raw = Path(path).expanduser()
    if raw.is_dir():
        tf = _tf()
        latest = tf.train.latest_checkpoint(str(raw))
        if latest:
            return latest
        # No `checkpoint` pointer file (our ckpt-50 copy has none): fall back to
        # the newest *.index in the directory.
        indexes = sorted(raw.glob("*.index"))
        if indexes:
            return str(indexes[-1])[: -len(".index")]
        raise FileNotFoundError(f"No TensorFlow checkpoint found in {raw}")
    if raw.suffix in ('.index', '.data'):
        return str(raw).split('.index')[0].split('.data')[0]
    if Path(str(raw) + '.index').exists():
        return str(raw)
    raise FileNotFoundError(
        f"Loquercio checkpoint prefix not found: {raw}. Expected {raw}.index or a checkpoint directory."
    )


def _checkpoint_value_key(prefix: str, weight) -> str:
    path = str(getattr(weight, "path", weight.name))
    leaf = path.rsplit("/", 1)[-1].split(":", 1)[0]
    return f"{prefix}/{leaf}/.ATTRIBUTES/VARIABLE_VALUE"


def _assign_checkpoint_layer(tf, checkpoint_prefix: str, layer, key_prefix: str) -> int:
    available = {
        name: tuple(shape)
        for name, shape in tf.train.list_variables(checkpoint_prefix)
        if name.startswith(key_prefix + "/") and ".OPTIMIZER_SLOT/" not in name
    }
    used = set()
    assigned = 0
    for weight in layer.weights:
        key = _checkpoint_value_key(key_prefix, weight)
        if key not in available:
            shape_matches = [
                name for name, shape in available.items()
                if name not in used and shape == tuple(weight.shape)
            ]
            if len(shape_matches) != 1:
                raise ValueError(
                    f"Missing unambiguous Loquercio tensor for {weight.path} "
                    f"under {key_prefix}; candidates={shape_matches}"
                )
            key = shape_matches[0]
        value = tf.train.load_variable(checkpoint_prefix, key)
        if tuple(value.shape) != tuple(weight.shape):
            raise ValueError(
                f"Loquercio checkpoint shape mismatch for {key}: "
                f"{tuple(value.shape)} != {tuple(weight.shape)}"
            )
        weight.assign(value)
        used.add(key)
        assigned += 1
    return assigned


def _restore_keras3_checkpoint(tf, model, checkpoint_prefix: str) -> int:
    """Load a TF/Keras 2 object checkpoint into the Keras 3 model."""
    variable_names = [
        name for name, _ in tf.train.list_variables(checkpoint_prefix)
        if ".OPTIMIZER_SLOT/" not in name
    ]
    backbone_indices = sorted({
        int(match.group(1))
        for name in variable_names
        if (match := re.match(r"net/backbone/0/layer_with_weights-(\d+)/", name))
    })
    backbone_layers = [layer for layer in model.backbone[0].layers if layer.weights]
    if len(backbone_indices) != len(backbone_layers):
        raise ValueError(
            "Loquercio backbone layer mismatch: checkpoint has "
            f"{len(backbone_indices)}, model has {len(backbone_layers)}"
        )

    assigned = 0
    for index, layer in zip(backbone_indices, backbone_layers):
        assigned += _assign_checkpoint_layer(
            tf, checkpoint_prefix, layer,
            f"net/backbone/0/layer_with_weights-{index}",
        )

    for group_name in (
        "resize_op", "img_mergenet", "resize_op_2",
        "states_conv", "resize_op_3", "plan_module",
    ):
        for index, layer in enumerate(getattr(model, group_name)):
            if layer.weights:
                assigned += _assign_checkpoint_layer(
                    tf, checkpoint_prefix, layer, f"net/{group_name}/{index}"
                )

    if assigned != len(model.weights):
        raise ValueError(
            f"Loaded {assigned} Loquercio weights, but model has {len(model.weights)}"
        )
    return assigned


class TensorFlowLoquercioBackend:
    """Loads the agile_autonomy PlaNet checkpoint and runs depth+IMU inference."""

    def __init__(self, checkpoint_path: str, config: LoquercioModelConfig):
        self.tf = _tf()
        self.config = config
        self.network = create_network(config)
        self._warm_start_network()
        checkpoint_prefix = resolve_checkpoint_prefix(checkpoint_path)
        keras_major = int(str(self.tf.keras.__version__).split(".", 1)[0])
        if keras_major >= 3:
            self.loaded_weight_count = _restore_keras3_checkpoint(
                self.tf, self.network.model, checkpoint_prefix
            )
        else:
            checkpoint = self.tf.train.Checkpoint(net=self.network.model)
            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched()
            status.expect_partial()
            self.loaded_weight_count = len(self.network.model.weights)
        self.checkpoint_prefix = checkpoint_prefix

    def _warm_start_network(self) -> None:
        inputs = {
            'depth': np.zeros(
                (1, self.config.seq_len, self.config.img_height, self.config.img_width, 3),
                dtype=np.float32,
            ),
            'imu': np.zeros(
                (1, self.config.seq_len, self.config.raw_state_dim),
                dtype=np.float32,
            ),
        }
        _ = self.network(inputs)

    def infer(self, depth: np.ndarray, imu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (alphas, trajectories) with modes sorted ascending by |alpha|
        (index 0 = the network's lowest-cost prediction); trajectories is
        (modes, 3*out_seq_len) in the camera/body frame."""
        inputs = {
            'depth': depth.astype(np.float32, copy=False),
            'imu': imu.astype(np.float32, copy=False),
        }
        pred = self.network(inputs).numpy()
        pred = pred[:, np.abs(pred[0, :, 0]).argsort(), :]
        alphas = np.abs(pred[0, :, 0])
        trajectories = pred[0, :, 1:]
        return alphas.astype(np.float32), trajectories.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX student backend (anyanything TRAINING, test-5 students)
# ---------------------------------------------------------------------------
# The students trained in `anyanything` are exported to ONNX with the contract
# in ~/anyanything/agile_student/INPUTS.md:
#
#   inputs   imu   (1, 1, 22) = [pos(3), R(9), v_body(3), omega_body(3),
#                                goal_body(3), v_goal(1)]
#            depth (1, 1, 224, 224, 3) = mm/80, clipped at 20 m, tiled x3
#   output   (1, M, 1 + 3N) per mode [alpha, x_1..N, y_1..N, z_1..N] in the
#            body frame at t = 0.5 j s (ABSOLUTE metres, not speed-normalised)
#
# M (modes) and N (waypoints) vary by run -- the shipped students are
# (M=2, N=10), (M=3, N=5) and (M=3, N=10) -- so both come from the graph, never
# from a hard-coded config. The reference decoder this mirrors byte-for-byte is
# superfly_expert_sampler.sim_episode.OnnxPolicy.decode_output (the test-5
# evaluation harness); tests/test_agile_student.py asserts they agree.


@dataclass(frozen=True)
class StudentModelConfig:
    """LoquercioModelConfig's shape for the student: 22-dim state, and mode /
    waypoint counts read off the ONNX graph rather than fixed at 3 / 10."""
    modes: int = 3
    out_seq_len: int = 10
    img_width: int = 224
    img_height: int = 224
    seq_len: int = 1
    state_dim: int = 3
    use_rgb: bool = False
    use_depth: bool = True
    use_position: bool = False
    use_attitude: bool = True
    use_bodyrates: bool = True
    freeze_backbone: bool = False

    @property
    def raw_state_dim(self) -> int:
        return 22

    @property
    def output_dim_per_mode(self) -> int:
        return 1 + self.state_dim * self.out_seq_len


def is_onnx_checkpoint(path) -> bool:
    """The student is selected purely by artifact extension, so nothing in the
    harness has to special-case a method name."""
    return str(path).lower().endswith(".onnx")


def decode_student_output(out) -> Tuple[np.ndarray, np.ndarray]:
    """(1, M, 1+3N) -> (alphas (M,), trajectories (M, 3N)), modes sorted
    ascending by |alpha| so index 0 is the net's lowest-cost prediction --
    the same ordering TensorFlowLoquercioBackend.infer returns, and the same
    ordering sim_episode.select_mode's argsort produces.

    `trajectories` keeps the flat [x_1..N | y_1..N | z_1..N] layout, which is
    what agile_core._adopt_plan reshapes as (state_dim=3, out_seq_len).
    A two-tensor graph (modes (1,M,N,3), costs (1,M)) is accepted too, matching
    sim_episode.OnnxPolicy.decode_output."""
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        m = np.asarray(out[0], dtype=np.float64)[0]
        modes = m.shape[0]
        wps = m.reshape(modes, m.size // (3 * modes), 3)          # (M, N, 3)
        alpha = np.abs(np.asarray(out[1], dtype=np.float64)[0].reshape(modes))
        flat = np.transpose(wps, (0, 2, 1)).reshape(modes, -1)    # -> [x..|y..|z..]
    else:
        o = np.asarray(out[0] if isinstance(out, (list, tuple)) else out, dtype=np.float64)
        o = o.reshape(-1, o.shape[-1])
        alpha = np.abs(o[:, 0])
        flat = o[:, 1:]
    order = np.argsort(alpha, kind="stable")
    return (alpha[order].astype(np.float32), flat[order].astype(np.float32))


class OnnxStudentBackend:
    """onnxruntime (CPU) drop-in for TensorFlowLoquercioBackend.

    Exposes the same `infer(depth, imu) -> (alphas, trajectories)` contract, so
    agile_core's plan pipeline is unchanged apart from the encoders. `modes` and
    `out_seq_len` are discovered from the graph (static output shape when the
    exporter wrote one, otherwise a single zero-input probe run)."""

    #: Intra-op threads for the CPU provider. Measured on gs2 (32 cores, load
    #: average 39 -- i.e. pessimistic) on t5fix_s_r1, median of 15-20 runs:
    #:   default 174 ms | 1 thread 214 | 2: 162 | 4: 149 | 6: 136 | 8: 124
    #:   12: 119 | 16: 118  (ORT_ENABLE_ALL: 8 -> 136, 12 -> 132)
    #: Returns flatten past 8, and airstation03 is a shared 24-core box running
    #: Isaac + PX4 SITL + an acados MPC in the same breath, so 8 buys nearly all
    #: of the speed-up while leaving the rest of the machine alone. Override
    #: with AGILE_ONNX_THREADS. Even at 8 threads one pass costs more than the
    #: 1/15 s decision period, which is why the student runs --net-thread: the
    #: forward pass must not block the 100 Hz attitude stream to PX4.
    DEFAULT_INTRA_OP_THREADS = 8

    def __init__(self, checkpoint_path: str, providers=("CPUExecutionProvider",),
                 intra_op_threads: int | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required for the agile_student method. Install it "
                "into the agile venv: <repo>/.venv/bin/python -m pip install onnxruntime"
            ) from exc
        path = Path(checkpoint_path)
        if path.is_dir():                               # accept the run folder too
            path = path / "student.onnx"
        if not path.exists():
            raise FileNotFoundError(f"ONNX student not found: {path}")
        self.checkpoint_prefix = str(path)
        # CPU on purpose: the offboard shares the box with Isaac + PX4 SITL.
        if intra_op_threads is None:
            intra_op_threads = int(os.environ.get("AGILE_ONNX_THREADS",
                                                  self.DEFAULT_INTRA_OP_THREADS))
        self.intra_op_threads = int(intra_op_threads)
        opts = ort.SessionOptions()
        if self.intra_op_threads > 0:
            opts.intra_op_num_threads = self.intra_op_threads
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), opts, providers=list(providers))
        self._input_shapes = {i.name: list(i.shape) for i in self.session.get_inputs()}
        self._depth_input, self._state_input = self._classify_inputs()
        self.modes, self.out_seq_len = self._probe_output_shape()
        self.config = StudentModelConfig(modes=self.modes, out_seq_len=self.out_seq_len)
        self.loaded_weight_count = len(self.session.get_inputs())   # diagnostic only
        self.forward_ms = self._time_one_forward()
        print(f"[agile] onnx forward {self.forward_ms:.0f} ms "
              f"({self.intra_op_threads} intra-op threads) -> at most "
              f"{1000.0 / max(self.forward_ms, 1e-6):.1f} Hz of decisions on this "
              f"box; the evaluation harness decides at 15 Hz.", flush=True)

    def _time_one_forward(self) -> float:
        """Median of a few zero-input passes, logged at startup: the decision
        rate is the number that decides whether a flight is comparable to the
        test-5 evaluation, and it is a property of the box, not of the code."""
        import time as _time
        depth = np.zeros((1, 1, 224, 224, 3), dtype=np.float32)
        imu = np.zeros((1, 1, self.config.raw_state_dim), dtype=np.float32)
        feed = {self._depth_input: depth, self._state_input: imu}
        for _ in range(2):
            self.session.run(None, feed)
        ts = []
        for _ in range(5):
            t0 = _time.perf_counter()
            self.session.run(None, feed)
            ts.append((_time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))

    def _classify_inputs(self):
        """Same rule as sim_episode.OnnxPolicy.__call__: the 5-D (or
        depth/img-named) tensor is the image, the other one is the state."""
        depth_name = state_name = None
        for name, shape in self._input_shapes.items():
            if "depth" in name.lower() or "img" in name.lower() or len(shape) == 5:
                depth_name = name
            else:
                state_name = name
        if depth_name is None or state_name is None:
            raise ValueError(
                f"cannot classify ONNX inputs {list(self._input_shapes)} into "
                "(depth, state); expected one 5-D image tensor and one state vector")
        return depth_name, state_name

    def _probe_output_shape(self):
        shape = list(self.session.get_outputs()[0].shape)
        if len(shape) == 3 and all(isinstance(d, int) and d > 0 for d in shape[1:]):
            modes, width = int(shape[1]), int(shape[2])
            return modes, (width - 1) // 3
        # dynamic axes: one zero-input forward pass settles it
        depth = np.zeros((1, 1, 224, 224, 3), dtype=np.float32)
        imu = np.zeros((1, 1, 22), dtype=np.float32)
        alphas, trajectories = self.infer(depth, imu, _bootstrap=True)
        return int(alphas.shape[0]), int(trajectories.shape[1] // 3)

    def infer(self, depth: np.ndarray, imu: np.ndarray, _bootstrap: bool = False):
        """(alphas (M,), trajectories (M, 3*out_seq_len)) sorted by |alpha|."""
        feed = {self._depth_input: np.ascontiguousarray(depth, dtype=np.float32),
                self._state_input: np.ascontiguousarray(imu, dtype=np.float32)}
        return decode_student_output(self.session.run(None, feed))
