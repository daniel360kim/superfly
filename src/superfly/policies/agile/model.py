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
