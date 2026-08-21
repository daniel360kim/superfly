#!/usr/bin/env python3
"""Convert a DiffAero exported ONNX actor to TFLite, with built-in
numerical verification against onnxruntime.

Pipeline: exported_actor.onnx -> onnxsim (shape-inference + folding, on a
temp copy -- required, see convert_one) -> TF SavedModel (onnx2tf) ->
float32 .tflite (tf.lite.TFLiteConverter). Verification is not optional:
64 random inputs go through BOTH onnxruntime and the TFLite interpreter,
and the script exits nonzero if the fp32 max abs error exceeds 1e-4. A
sidecar `<out>.json` records the source-onnx sha256, tool versions,
verification stats, date and command line, so every .tflite in the repo
says exactly where it came from.

Given a run dir it also attempts `obs_fn.onnx` -> `obs_fn.tflite`
non-fatally: if that conversion fails it warns and continues -- the actor
is the deliverable.

This is CPU work; it runs fine on a dispatch box. It does NOT run in the
repo venvs -- build a fresh one. Verified working combination
(python 3.12.3, 2026-08-19):

    python3 -m venv ~/tflite-venv
    ~/tflite-venv/bin/pip install \
        tensorflow-cpu==2.21.0 onnx==1.20.1 onnxruntime==1.26.0 \
        onnx_graphsurgeon==0.6.1 sng4onnx==2.0.1 onnxsim==0.6.5 \
        ai-edge-litert==2.1.2 psutil==5.9.5 ml_dtypes flatbuffers
    ~/tflite-venv/bin/pip install --no-deps onnx2tf==1.29.24 tf_keras==2.21.0

onnx2tf MUST be 1.x and MUST go in with --no-deps: 2.x drops the
SavedModel stage entirely (different pipeline, different Python API), and
1.x's own dep pins drag in an onnxsim source build that fails on py3.12
without cmake -- the wheel pinned above works.

Usage:
    python scripts/export_tflite.py checkpoints/DiffAero/<run>        # run dir
    python scripts/export_tflite.py path/to/exported_actor.onnx       # one file
    python scripts/export_tflite.py <run> --fp16   # also write *_fp16.tflite
"""

import argparse
import hashlib
import json
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

N_VERIFY = 64
FP32_ATOL = 1e-4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_versions() -> dict:
    import onnx
    import onnx2tf
    import onnxruntime
    import onnxsim
    import tensorflow as tf
    return {
        "onnx2tf": onnx2tf.__version__,
        "onnxsim": onnxsim.__version__,
        "tensorflow": tf.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
    }


def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path,
                        input_names: list[str],
                        fixed_batch: int | None) -> None:
    """ONNX -> TF SavedModel via onnx2tf. Every input is pinned with
    keep_shape_absolutely_input_names: without it onnx2tf channel-last-ifies
    any rank>=3 input (the actor's [1,9,16] 'orientation' came out
    [1,16,9]), silently changing the I/O contract the offboard code feeds."""
    import onnx2tf
    kwargs = dict(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        keep_shape_absolutely_input_names=input_names,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )
    if fixed_batch is not None:
        kwargs["batch_size"] = fixed_batch
    onnx2tf.convert(**kwargs)


def saved_model_to_tflite(saved_model_dir: Path, out: Path,
                          fp16: bool = False) -> None:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    out.write_bytes(converter.convert())


def onnx_input_specs(onnx_path: Path):
    """[(name, concrete_shape, np_dtype)] for every model input; dynamic
    dims become 1 (batch 1)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    specs = []
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        dtype = {"tensor(float)": np.float32,
                 "tensor(double)": np.float64,
                 "tensor(int64)": np.int64,
                 "tensor(int32)": np.int32}[inp.type]
        specs.append((inp.name, shape, dtype))
    return sess, specs


def verify(onnx_path: Path, tflite_path: Path, n: int = N_VERIFY):
    """Run n random inputs through onnxruntime and the TFLite interpreter;
    return (max_abs_err, max_rel_err). TFLite tensors are matched to ONNX
    tensors by their copied names when present, positionally otherwise."""
    import tensorflow as tf
    sess, specs = onnx_input_specs(onnx_path)
    onnx_out_names = [o.name for o in sess.get_outputs()]

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    tfl_in = interp.get_input_details()
    tfl_out = interp.get_output_details()
    if len(tfl_in) != len(specs):
        raise RuntimeError(
            f"input count mismatch: onnx={len(specs)} tflite={len(tfl_in)}")

    def match(names, details):
        """Map each ONNX tensor name to a TFLite detail. onnx2tf copies the
        ONNX names into the tflite tensors when asked, wrapped as
        'serving_default_<name>:0', so match on the stripped base name and
        fall back to positional order (e.g. for 'StatefulPartitionedCall:0'
        outputs)."""
        def base(d):
            b = d["name"].split(":")[0]
            return b.removeprefix("serving_default_")
        out = []
        for i, name in enumerate(names):
            hit = [d for d in details if base(d) == name]
            out.append(hit[0] if len(hit) == 1 else details[i])
        return out

    tfl_in = match([s[0] for s in specs], tfl_in)
    tfl_out = match(onnx_out_names, tfl_out)

    rng = np.random.default_rng(0)
    max_abs = 0.0
    max_rel = 0.0
    for _ in range(n):
        feeds = {}
        for name, shape, dtype in specs:
            if np.issubdtype(dtype, np.floating):
                x = rng.standard_normal(shape).astype(dtype)
            else:
                x = rng.integers(0, 2, size=shape).astype(dtype)
            feeds[name] = x
        ref = sess.run(onnx_out_names, feeds)
        for (name, _, _), detail in zip(specs, tfl_in):
            interp.set_tensor(detail["index"],
                              feeds[name].reshape(detail["shape"]))
        interp.invoke()
        for r, detail in zip(ref, tfl_out):
            t = interp.get_tensor(detail["index"]).reshape(r.shape)
            abs_err = np.abs(r.astype(np.float64) - t.astype(np.float64))
            max_abs = max(max_abs, float(abs_err.max()))
            denom = np.maximum(np.abs(r.astype(np.float64)), 1e-8)
            max_rel = max(max_rel, float((abs_err / denom).max()))
    return max_abs, max_rel


def convert_one(onnx_path: Path, out: Path, fp16: bool) -> dict:
    """Full pipeline for one .onnx; returns the sidecar dict (also written
    to <out>.json). Raises on conversion or verification failure.

    onnx2tf 1.x runs onnxsim and WRITES THE SIMPLIFIED MODEL BACK OVER ITS
    INPUT FILE (weights embedded, graph folded), which would silently dirty
    the committed checkpoint -- so conversion always runs on a temp copy
    and the source file (the verification reference) is never touched."""
    import onnx
    src_sha = sha256(onnx_path)
    model = onnx.load(str(onnx_path), load_external_data=False)
    opset = max((imp.version for imp in model.opset_import
                 if imp.domain in ("", "ai.onnx")), default=None)
    dynamic = any(
        not (d.HasField("dim_value") and d.dim_value > 0)
        for inp in model.graph.input
        for d in inp.type.tensor_type.shape.dim)

    input_names = [inp.name for inp in model.graph.input]
    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        work = Path(tmp) / onnx_path.name
        # onnxsim first, and not as an optimization: on the raw torch
        # export onnx2tf mis-tracks the operand layout of the actor's
        # vector@matrix MatMul and dies ("Dimensions must be equal, 3 and
        # 1"); after shape-inference + constant-folding it converts
        # cleanly. onnx.load also inlines the .onnx.data external weights
        # here, so the temp copy is self-contained.
        import onnxsim
        simplified, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
        if not ok:
            raise RuntimeError("onnxsim.simplify() self-check failed")
        onnx.save(simplified, str(work))
        saved_model_dir = Path(tmp) / "saved_model"
        onnx_to_saved_model(work, saved_model_dir, input_names,
                            fixed_batch=1 if dynamic else None)
        saved_model_to_tflite(saved_model_dir, out, fp16=False)
        fp16_out = None
        if fp16:
            fp16_out = out.with_name(out.stem + "_fp16.tflite")
            saved_model_to_tflite(saved_model_dir, fp16_out, fp16=True)

    max_abs, max_rel = verify(onnx_path, out)
    print(f"  {out.name}: max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e} "
          f"({N_VERIFY} random inputs vs onnxruntime)")
    if max_abs > FP32_ATOL:
        raise RuntimeError(
            f"fp32 verification FAILED: max_abs_err {max_abs:.3e} > {FP32_ATOL}")

    sidecar = {
        "schema": "superfly-tflite-export-v1",
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": src_sha,
        "opset": opset,
        "dynamic_batch_fixed_to_1": dynamic,
        "tool_versions": tool_versions(),
        "verification": {
            "n_random_inputs": N_VERIFY,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "fp32_atol": FP32_ATOL,
        },
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
    }
    if fp16_out is not None:
        f16_abs, f16_rel = verify(onnx_path, fp16_out)
        print(f"  {fp16_out.name}: max_abs_err={f16_abs:.3e} "
              f"max_rel_err={f16_rel:.3e} (informational, no gate)")
        sidecar["fp16"] = {
            "file": fp16_out.name,
            "max_abs_err": f16_abs,
            "max_rel_err": f16_rel,
        }
    Path(str(out) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source",
                    help="DiffAero run dir (checkpoints/DiffAero/<run>) or a "
                         "direct .onnx path")
    ap.add_argument("--out", default=None,
                    help="Output .tflite path (default: alongside the onnx, "
                         "exported_actor.tflite)")
    ap.add_argument("--fp16", action="store_true",
                    help="Also write an fp16-quantized *_fp16.tflite "
                         "(verified informationally, not gated)")
    args = ap.parse_args()

    src = Path(args.source)
    if src.is_dir():
        ckpt_dir = src / "checkpoints" if (src / "checkpoints").is_dir() else src
        actor = ckpt_dir / "exported_actor.onnx"
        obs_fn = ckpt_dir / "obs_fn.onnx"
        if not actor.is_file():
            ap.error(f"no exported_actor.onnx under {ckpt_dir}")
    elif src.suffix == ".onnx":
        actor, obs_fn = src, None
    else:
        ap.error(f"{src}: not a directory or .onnx file")

    out = Path(args.out) if args.out else actor.with_suffix(".tflite")
    print(f"converting {actor} -> {out}")
    convert_one(actor, out, fp16=args.fp16)
    print(f"wrote {out} ({out.stat().st_size} bytes) + {out.name}.json")

    if obs_fn is not None and obs_fn.is_file():
        obs_out = obs_fn.with_suffix(".tflite")
        print(f"converting {obs_fn} -> {obs_out} (non-fatal)")
        try:
            convert_one(obs_fn, obs_out, fp16=args.fp16)
            print(f"wrote {obs_out} ({obs_out.stat().st_size} bytes)")
        except Exception as e:  # noqa: BLE001 -- best-effort side artifact
            print(f"WARNING: obs_fn.onnx conversion failed ({e}); "
                  f"continuing -- the actor is the deliverable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
