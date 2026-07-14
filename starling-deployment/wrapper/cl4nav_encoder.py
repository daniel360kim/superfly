"""Frozen CL4Nav RGB encoder used by the Agile RGB deployment path."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class FrozenCL4NavEncoder:
    """Run CL4Nav ONNX outside TensorFlow, matching the training input path."""

    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224

    def __init__(self, model_path: str, tf, feature_dim: int = 128,
                 execution_provider: str = "CUDAExecutionProvider"):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for agile_rgb. Install onnxruntime-gpu "
                "in the tf_gpu runtime (or onnxruntime for CPU inference)."
            ) from exc

        self.tf = tf
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.feature_dim = int(feature_dim)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"CL4Nav ONNX model not found: {self.model_path}")

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = ort.get_available_providers()
        if execution_provider not in available:
            raise RuntimeError(
                f"CL4Nav ONNX provider {execution_provider!r} is unavailable; "
                f"available providers: {available}. Use --cl4nav-provider "
                "CPUExecutionProvider to run on CPU."
            )
        providers = [execution_provider]
        if (execution_provider != "CPUExecutionProvider"
                and "CPUExecutionProvider" in available):
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        if execution_provider not in self.session.get_providers():
            raise RuntimeError(f"Failed to activate ONNX provider {execution_provider!r}")

        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()
        if len(model_inputs) != 1 or not model_outputs:
            raise ValueError("CL4Nav ONNX model must have one input and at least one output")
        self.input_name = model_inputs[0].name
        self.output_name = model_outputs[0].name
        self._validate_contract(model_inputs[0].shape, model_outputs[0].shape)

    def _validate_contract(self, input_shape, output_shape):
        if len(input_shape) != 4:
            raise ValueError(f"CL4Nav ONNX input must have rank 4, got {input_shape}")
        for actual, expected in zip(input_shape[1:], (3, 224, 224)):
            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    f"CL4Nav ONNX input must be [N,3,224,224], got {input_shape}"
                )
        if len(output_shape) != 2:
            raise ValueError(f"CL4Nav ONNX output must have rank 2, got {output_shape}")
        if isinstance(output_shape[-1], int) and output_shape[-1] != self.feature_dim:
            raise ValueError(
                f"Configured feature dimension {self.feature_dim} does not match "
                f"ONNX output {output_shape[-1]}"
            )

    def _preprocess(self, rgb):
        if self.tf.is_tensor(rgb):
            rgb = rgb.numpy()
        images = np.asarray(rgb)
        if images.ndim != 5 or images.shape[-1] != 3:
            raise ValueError(f"RGB input must have shape [B,T,H,W,3], got {images.shape}")
        batch_size, seq_len, height, width, _ = images.shape
        if batch_size < 1 or seq_len < 1:
            raise ValueError("RGB batch and sequence dimensions must be non-empty")

        images = images.reshape(batch_size * seq_len, height, width, 3).astype(
            np.float32, copy=False)
        if not np.all(np.isfinite(images)):
            raise ValueError("RGB input contains NaN or Inf")
        image_min = float(images.min())
        image_max = float(images.max())
        if image_min < 0.0 or image_max > 255.0:
            raise ValueError("RGB values must be in [0,1] or [0,255]")
        if image_max > 1.0:
            images = images / 255.0
        if (height, width) != (self.INPUT_HEIGHT, self.INPUT_WIDTH):
            images = self.tf.image.resize(
                images, [self.INPUT_HEIGHT, self.INPUT_WIDTH], method="bilinear"
            ).numpy()
        images = np.clip(images, 0.0, 1.0)
        images = np.transpose(images, (0, 3, 1, 2))
        return np.ascontiguousarray(images, dtype=np.float32), batch_size, seq_len

    def __call__(self, rgb) -> np.ndarray:
        encoder_input, batch_size, seq_len = self._preprocess(rgb)
        features = self.session.run(
            [self.output_name], {self.input_name: encoder_input}
        )[0]
        features = np.asarray(features, dtype=np.float32)
        expected = (batch_size * seq_len, self.feature_dim)
        if features.shape != expected:
            raise ValueError(f"CL4Nav ONNX output must have shape {expected}, got {features.shape}")
        if not np.all(np.isfinite(features)):
            raise ValueError("CL4Nav ONNX output contains NaN or Inf")
        return features.reshape(batch_size, seq_len, self.feature_dim)
