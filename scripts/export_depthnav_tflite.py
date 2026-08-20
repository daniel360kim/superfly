#!/usr/bin/env python3
"""Export a DepthNav MultiInputPolicy checkpoint to ONNX and TFLite, with
numerical verification at every hop (torch -> onnxruntime -> TFLite),
including a recurrent-rollout check.

The policy is recurrent (GRUCell, hidden 192) over a dict obs
{state:(7,), target:(4,), depth:(1,72,128)}. The export wraps it in a
flat-tensor module with static shapes:

    inputs : state[1,7], target[1,4], depth[1,1,72,128], latent[1,192]
    outputs: action[1,4], new_latent[1,192]

The caller owns the hidden state: init latent to zeros at handoff, feed
new_latent back every step (exactly what DepthNavPolicy.step does). Depth
input is metric metres already clamped to [0.25, 20] (base_env semantics);
the 1/(d+1e-6) inversion and the 72x128 -> 12x16 max-pool are INSIDE the
graph (ImageExtractor.preprocess_depth), so the deploy side feeds raw
clamped metres, same as the torch wrapper.

Two ops need special handling:
  * nn.GRUCell lowers to aten::gru_cell, which has no ONNX symbolic -- the
    wrapper re-implements the cell from its own weights (PyTorch gate order
    r,z,n) and the decomposition is verified against the real module in
    torch before anything is exported. LayerNormGRUCell (the other alias)
    is already plain ops and is called as-is.
  * F.adaptive_max_pool2d(72x128 -> 12x16) divides evenly, so the
    TorchScript exporter lowers it to a plain MaxPool -- fine as long as
    input_max_pool_H_W divides the depth resolution (asserted).

Two venvs, two stages (this box keeps torch and the TF stack in separate
envs -- see scripts/export_tflite.py on branch `triage` for the pinned TF
stack: onnx2tf 1.29.24 installed --no-deps, onnxsim 0.6.5, tensorflow-cpu
2.21.0, tf_keras, onnx 1.20.1, onnxruntime 1.26.0, ai-edge-litert 2.1.2):

  --stage torch2onnx    torch venv (methods/depthnav/.venv; needs
                        `pip install onnx==1.20.1` on top -- the TorchScript
                        exporter serializes through the onnx package since
                        torch 2.9): load ckpt,
                        wrap, torch.onnx.export (opset 17, dynamo=False --
                        the venv has no onnxscript), dump a reference .npz
                        of random inputs + torch outputs (64 single-step
                        samples and a 10-step recurrent rollout).
  --stage onnx2tflite   tflite venv: onnxsim (load-bearing, see triage
                        notes) -> onnx2tf (keep_shape_absolutely_input_names
                        on all four inputs so nothing gets channel-last-
                        ified) -> fp32 .tflite, then verify: ONNX vs torch
                        refs, TFLite vs onnxruntime, and the 10-step
                        recurrent rollout TFLite vs torch refs. Writes
                        <out>.tflite.json sidecar (sha256s, versions, stats).
  --stage all (default) run both by shelling out to the right interpreters
                        (--depthnav-python / --tflite-python, env
                        DEPTHNAV_PYTHON / TFLITE_PYTHON).

Everything is CPU work. onnx2tf mutates its input file (onnxsim writeback),
so conversion always runs on a temp copy -- the exported .onnx stays the
verification reference.

Usage (prototype):
    python scripts/export_depthnav_tflite.py \
        checkpoints/DepthNav/level1_4/level1_4_iteration_13500.pth \
        --out-dir /path/to/scratch \
        --tflite-python ~/tflite-venv/bin/python

    # velocity variant, architecture smoke test with random weights:
    python scripts/export_depthnav_tflite.py --random-weights \
        --cfg methods/depthnav/examples/navigation/policy_cfg/small_yaw_vel.yaml \
        --out-dir /path/to/scratch ...
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = (REPO_ROOT / "methods/depthnav/examples/navigation/"
               "policy_cfg/small_yaw.yaml")

STATE_DIM, TARGET_DIM = 7, 4
DEPTH_H, DEPTH_W = 72, 128
DEPTH_NEAR, DEPTH_FAR = 0.25, 20.0
LATENT_DIM = 192
OPSET = 17

N_VERIFY = 64          # single-step random samples
N_ROLLOUT = 10         # recurrent rollout length
FP32_ATOL = 1e-4       # gate: onnx-vs-torch and tflite-vs-onnx, single step
WRAPPER_ATOL = 1e-5    # gate: GRU decomposition vs real module, torch-vs-torch
ROLLOUT_ATOL = 1e-3    # gate: tflite-vs-torch after 10 recurrent steps


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_inputs(rng, n):
    """Plausible-range random inputs. Depth MUST be sampled in metric range
    [near, far]: the graph computes 1/(d+1e-6), so standard-normal depth
    would produce ~1e6 activations and garbage error measurements."""
    import numpy as np
    return {
        "state": rng.standard_normal((n, 1, STATE_DIM)).astype(np.float32),
        "target": rng.standard_normal((n, 1, TARGET_DIM)).astype(np.float32),
        "depth": rng.uniform(DEPTH_NEAR, DEPTH_FAR,
                             (n, 1, 1, DEPTH_H, DEPTH_W)).astype(np.float32),
        "latent": rng.uniform(-1.0, 1.0,
                              (n, 1, LATENT_DIM)).astype(np.float32),
    }


# --------------------------------------------------------------------------
# stage 1: torch -> onnx (+ reference outputs)  [runs in the depthnav venv]
# --------------------------------------------------------------------------

def stage_torch2onnx(args) -> int:
    depthnav_root = Path(os.environ.get("DEPTHNAV_ROOT",
                                        REPO_ROOT / "methods/depthnav"))
    sys.path.insert(0, str(depthnav_root))
    import numpy as np
    import torch as th
    import torch.nn.functional as F
    import yaml
    from gymnasium import spaces
    from depthnav.policies.multi_input_policy import (MultiInputPolicy,
                                                      LayerNormGRUCell)

    ckpt = resolve_ckpt(args.checkpoint)
    cfg_path = Path(args.cfg) if args.cfg else DEFAULT_CFG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    policy_kwargs = dict(cfg["policy"])
    policy_kwargs["device"] = "cpu"
    activation = str(policy_kwargs.get("output_activation_fn", ""))

    # sanity: max-pool target must divide the depth resolution, or
    # adaptive_max_pool2d does not lower to a static MaxPool in ONNX
    pool = (policy_kwargs.get("feature_extractor_kwargs", {})
            .get("net_arch", {}).get("depth", {}).get("input_max_pool_H_W"))
    if pool:
        assert DEPTH_H % pool[0] == 0 and DEPTH_W % pool[1] == 0, \
            f"input_max_pool_H_W {pool} does not divide {DEPTH_H}x{DEPTH_W}"

    obs_space = spaces.Dict({
        "state": spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
        "target": spaces.Box(-np.inf, np.inf, (TARGET_DIM,), dtype=np.float32),
        "depth": spaces.Box(0.0, np.inf, (1, DEPTH_H, DEPTH_W),
                            dtype=np.float32),
    })
    model = MultiInputPolicy(obs_space, **policy_kwargs)
    if args.random_weights:
        print(f"[torch2onnx] RANDOM WEIGHTS (smoke test), cfg={cfg_path.name}")
    else:
        model.load(str(ckpt))
        print(f"[torch2onnx] loaded {ckpt.name}, cfg={cfg_path.name}, "
              f"output_activation={activation}")
    model.eval()
    assert model.is_recurrent and model.latent_dim == LATENT_DIM, \
        f"expected recurrent policy with latent {LATENT_DIM}"

    class FlatPolicy(th.nn.Module):
        """Flat-tensor wrapper. Re-implements nn.GRUCell from its weights
        because aten::gru_cell has no ONNX symbolic."""

        def __init__(self, m):
            super().__init__()
            self.m = m
            self.cell = m.recurrent_extractor
            self.cell_is_module_ops = isinstance(self.cell, LayerNormGRUCell)

        def gru_cell(self, x, h):
            c = self.cell
            gi = F.linear(x, c.weight_ih, c.bias_ih)
            gh = F.linear(h, c.weight_hh, c.bias_hh)
            i_r, i_z, i_n = gi.chunk(3, 1)
            h_r, h_z, h_n = gh.chunk(3, 1)
            r = th.sigmoid(i_r + h_r)
            z = th.sigmoid(i_z + h_z)
            n = th.tanh(i_n + r * h_n)
            return (1 - z) * n + z * h

        def forward(self, state, target, depth, latent):
            obs = {"state": state, "target": target, "depth": depth}
            feats = self.m.feature_norm(self.m.feature_extractor(obs))
            if self.cell_is_module_ops:
                new_latent = self.cell(feats, latent)   # plain ops, exports
            else:
                new_latent = self.gru_cell(feats, latent)
            action = self.m.policy_net(new_latent)
            return action, new_latent

    wrapper = FlatPolicy(model).eval()
    rng = np.random.default_rng(0)
    single = make_inputs(rng, N_VERIFY)

    # --- verify the wrapper (esp. the GRU decomposition) against the real
    # model, torch vs torch, before trusting any exported graph ---
    wrap_err = 0.0
    ref_action = np.empty((N_VERIFY, 1, 4), np.float32)
    ref_latent = np.empty((N_VERIFY, 1, LATENT_DIM), np.float32)
    with th.no_grad():
        for i in range(N_VERIFY):
            t = {k: th.from_numpy(single[k][i]) for k in single}
            a_ref, l_ref = model(
                {"state": t["state"], "target": t["target"],
                 "depth": t["depth"]}, t["latent"])
            a, l = wrapper(t["state"], t["target"], t["depth"], t["latent"])
            wrap_err = max(wrap_err,
                           float((a - a_ref).abs().max()),
                           float((l - l_ref).abs().max()))
            ref_action[i], ref_latent[i] = a.numpy(), l.numpy()
    print(f"[torch2onnx] wrapper-vs-model max_abs_err={wrap_err:.3e} "
          f"({N_VERIFY} samples; GRU decomposition check)")
    if wrap_err > WRAPPER_ATOL:
        print(f"[torch2onnx] FAIL: wrapper mismatch > {WRAPPER_ATOL}")
        return 1

    # --- recurrent rollout reference: latent starts at zeros, feeds back ---
    roll = make_inputs(rng, N_ROLLOUT)
    roll_action = np.empty((N_ROLLOUT, 1, 4), np.float32)
    roll_latent = np.empty((N_ROLLOUT, 1, LATENT_DIM), np.float32)
    with th.no_grad():
        h = th.zeros(1, LATENT_DIM)
        for i in range(N_ROLLOUT):
            a, h = wrapper(th.from_numpy(roll["state"][i]),
                           th.from_numpy(roll["target"][i]),
                           th.from_numpy(roll["depth"][i]), h)
            roll_action[i], roll_latent[i] = a.numpy(), h.numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or ckpt.stem
    onnx_path = out_dir / f"{stem}.onnx"
    example = tuple(th.from_numpy(single[k][0])
                    for k in ("state", "target", "depth", "latent"))
    th.onnx.export(
        wrapper, example, str(onnx_path),
        input_names=["state", "target", "depth", "latent"],
        output_names=["action", "new_latent"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,   # TorchScript exporter: static shapes, no onnxscript dep
    )
    print(f"[torch2onnx] wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")

    np.savez_compressed(
        out_dir / f"{stem}_ref.npz",
        **{f"single_{k}": v for k, v in single.items()},
        single_action=ref_action, single_new_latent=ref_latent,
        **{f"roll_{k}": v for k, v in roll.items()},
        roll_action=roll_action, roll_new_latent=roll_latent,
    )
    meta = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": None if args.random_weights else sha256(ckpt),
        "random_weights": bool(args.random_weights),
        "cfg": str(cfg_path),
        "output_activation_fn": activation,
        "opset": OPSET,
        "torch": th.__version__,
        "wrapper_vs_model_max_abs_err": wrap_err,
        "io": {"inputs": {"state": [1, STATE_DIM], "target": [1, TARGET_DIM],
                          "depth": [1, 1, DEPTH_H, DEPTH_W],
                          "latent": [1, LATENT_DIM]},
               "outputs": {"action": [1, 4],
                           "new_latent": [1, LATENT_DIM]}},
    }
    (out_dir / f"{stem}.onnx.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    return 0


def resolve_ckpt(source: str) -> Path:
    src = Path(source)
    if src.is_dir():
        pths = sorted(src.glob("*.pth"))
        if len(pths) != 1:
            sys.exit(f"{src}: expected exactly one .pth, found "
                     f"{[p.name for p in pths]} -- pass the file directly")
        return pths[0]
    if src.suffix != ".pth":
        sys.exit(f"{src}: not a run dir or .pth file")
    return src


# --------------------------------------------------------------------------
# stage 2: onnx -> tflite + verification  [runs in the tflite venv]
# --------------------------------------------------------------------------

def stage_onnx2tflite(args) -> int:
    import numpy as np
    import onnx
    import onnxsim
    import onnx2tf
    import onnxruntime as ort
    import tensorflow as tf

    out_dir = Path(args.out_dir)
    stem = args.name or resolve_ckpt(args.checkpoint).stem
    onnx_path = out_dir / f"{stem}.onnx"
    ref = np.load(out_dir / f"{stem}_ref.npz")
    meta = json.loads((out_dir / f"{stem}.onnx.meta.json").read_text())
    tflite_path = out_dir / f"{stem}.tflite"
    input_names = ["state", "target", "depth", "latent"]
    output_names = ["action", "new_latent"]

    # conversion on a temp copy: onnx2tf's onnxsim pass writes the
    # simplified model back over its input file (triage lesson -- never
    # let it touch the reference .onnx)
    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmp:
        work = Path(tmp) / onnx_path.name
        # onnxsim first, load-bearing: onnx2tf mis-tracks operand layouts on
        # raw torch exports without shape-inference + folding
        simplified, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
        if not ok:
            print("[onnx2tflite] FAIL: onnxsim self-check failed")
            return 1
        onnx.save(simplified, str(work))
        saved_model_dir = Path(tmp) / "saved_model"
        # pin every input's shape or onnx2tf channel-last-ifies rank>=3
        # inputs (depth [1,1,72,128] would come out [1,72,128,1] and the
        # latent handling could silently change the I/O contract)
        onnx2tf.convert(
            input_onnx_file_path=str(work),
            output_folder_path=str(saved_model_dir),
            keep_shape_absolutely_input_names=input_names,
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
        )
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_dir))
        tflite_path.write_bytes(converter.convert())
    print(f"[onnx2tflite] wrote {tflite_path} "
          f"({tflite_path.stat().st_size} bytes)")

    # --- set up both runtimes ---
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()

    def match(names, details):
        """Match tflite tensors to onnx names ('serving_default_<name>:0'
        wrapping), positional fallback (e.g. 'StatefulPartitionedCall:0')."""
        def base(d):
            return d["name"].split(":")[0].removeprefix("serving_default_")
        return [next((d for d in details if base(d) == n), details[i])
                for i, n in enumerate(names)]

    tfl_in = match(input_names, interp.get_input_details())
    tfl_out = match(output_names, interp.get_output_details())

    def run_tflite(feeds):
        for n, d in zip(input_names, tfl_in):
            interp.set_tensor(d["index"], feeds[n].reshape(d["shape"]))
        interp.invoke()
        return [interp.get_tensor(d["index"]) for d in tfl_out]

    def maxabs(a, b):
        return float(np.abs(np.asarray(a, np.float64)
                            - np.asarray(b, np.float64)).max())

    # --- single-step: onnx vs torch refs, tflite vs onnx ---
    n = ref["single_state"].shape[0]
    onnx_vs_torch = tfl_vs_onnx = 0.0
    for i in range(n):
        feeds = {k: ref[f"single_{k}"][i] for k in input_names}
        o_act, o_lat = sess.run(output_names, feeds)
        onnx_vs_torch = max(onnx_vs_torch,
                            maxabs(o_act, ref["single_action"][i]),
                            maxabs(o_lat, ref["single_new_latent"][i]))
        t_act, t_lat = run_tflite(feeds)
        tfl_vs_onnx = max(tfl_vs_onnx,
                          maxabs(t_act.reshape(o_act.shape), o_act),
                          maxabs(t_lat.reshape(o_lat.shape), o_lat))
    print(f"[verify] single-step ({n} samples): "
          f"onnx-vs-torch max_abs_err={onnx_vs_torch:.3e}, "
          f"tflite-vs-onnxruntime max_abs_err={tfl_vs_onnx:.3e}")

    # --- recurrent rollout: tflite feeds its OWN new_latent back; compare
    # against the torch rollout at every step. This is the real risk:
    # per-step error compounding through the hidden state. ---
    steps = ref["roll_state"].shape[0]
    h = np.zeros((1, LATENT_DIM), np.float32)
    per_step = []
    for i in range(steps):
        feeds = {k: ref[f"roll_{k}"][i] for k in ("state", "target", "depth")}
        feeds["latent"] = h
        t_act, t_lat = run_tflite(feeds)
        h = np.asarray(t_lat, np.float32).reshape(1, LATENT_DIM)
        per_step.append(max(
            maxabs(t_act.reshape(1, 4), ref["roll_action"][i]),
            maxabs(h, ref["roll_new_latent"][i])))
    rollout_final = per_step[-1]
    print(f"[verify] {steps}-step recurrent rollout tflite-vs-torch: "
          f"step1={per_step[0]:.3e} ... step{steps}={rollout_final:.3e}")

    sidecar = {
        "schema": "superfly-depthnav-tflite-export-v1",
        "source_checkpoint": meta["checkpoint"],
        "source_checkpoint_sha256": meta["checkpoint_sha256"],
        "random_weights": meta["random_weights"],
        "cfg": meta["cfg"],
        "output_activation_fn": meta["output_activation_fn"],
        "source_onnx": str(onnx_path),
        "source_onnx_sha256": sha256(onnx_path),
        "opset": meta["opset"],
        "io": meta["io"],
        "tool_versions": {
            "torch": meta["torch"], "onnx2tf": onnx2tf.__version__,
            "onnxsim": onnxsim.__version__, "tensorflow": tf.__version__,
            "onnx": onnx.__version__, "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "verification": {
            "n_single_step": n,
            "wrapper_vs_model_max_abs_err":
                meta["wrapper_vs_model_max_abs_err"],
            "onnx_vs_torch_max_abs_err": onnx_vs_torch,
            "tflite_vs_onnxruntime_max_abs_err": tfl_vs_onnx,
            "rollout_steps": steps,
            "rollout_tflite_vs_torch_per_step_max_abs_err": per_step,
            "rollout_final_step_max_abs_err": rollout_final,
            "fp32_atol": FP32_ATOL, "rollout_atol": ROLLOUT_ATOL,
        },
        "tflite_bytes": tflite_path.stat().st_size,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
    }
    Path(str(tflite_path) + ".json").write_text(
        json.dumps(sidecar, indent=2) + "\n")

    fail = []
    if onnx_vs_torch > FP32_ATOL:
        fail.append(f"onnx-vs-torch {onnx_vs_torch:.3e} > {FP32_ATOL}")
    if tfl_vs_onnx > FP32_ATOL:
        fail.append(f"tflite-vs-onnxruntime {tfl_vs_onnx:.3e} > {FP32_ATOL}")
    if rollout_final > ROLLOUT_ATOL:
        fail.append(f"rollout step{steps} {rollout_final:.3e} > {ROLLOUT_ATOL}")
    if fail:
        print("[verify] FAILED: " + "; ".join(fail))
        return 1
    print(f"[verify] PASS -- sidecar {tflite_path.name}.json")
    return 0


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint",
                    help=".pth file or a run dir containing exactly one .pth "
                         "(e.g. checkpoints/DepthNav/level1_4)")
    ap.add_argument("--cfg", default=None,
                    help="policy yaml (default: policy_cfg/small_yaw.yaml; "
                         "pass small_yaw_vel.yaml for velocity checkpoints -- "
                         "the yaml's output_activation_fn is what selects the "
                         "action head, and it is recorded in the sidecar)")
    ap.add_argument("--out-dir", required=True,
                    help="where the .onnx/.tflite/refs land (keep it OUT of "
                         "the repo; checkpoints/ is committed)")
    ap.add_argument("--name", default=None,
                    help="artifact stem (default: checkpoint stem)")
    ap.add_argument("--stage", choices=["all", "torch2onnx", "onnx2tflite"],
                    default="all")
    ap.add_argument("--random-weights", action="store_true",
                    help="skip the checkpoint load; architecture/conversion "
                         "smoke test (sidecar marks random_weights=true)")
    ap.add_argument("--depthnav-python",
                    default=os.environ.get(
                        "DEPTHNAV_PYTHON",
                        str(REPO_ROOT / "methods/depthnav/.venv/bin/python")),
                    help="interpreter with torch + depthnav deps "
                         "(env DEPTHNAV_PYTHON)")
    ap.add_argument("--tflite-python",
                    default=os.environ.get(
                        "TFLITE_PYTHON",
                        str(Path.home() / "tflite-venv/bin/python")),
                    help="interpreter with the pinned onnx2tf/TF stack "
                         "(env TFLITE_PYTHON; see module docstring)")
    args = ap.parse_args()

    if args.stage == "torch2onnx":
        return stage_torch2onnx(args)
    if args.stage == "onnx2tflite":
        return stage_onnx2tflite(args)

    passthrough = [args.checkpoint, "--out-dir", args.out_dir]
    if args.cfg:
        passthrough += ["--cfg", args.cfg]
    if args.name:
        passthrough += ["--name", args.name]
    if args.random_weights:
        passthrough += ["--random-weights"]
    for py, stage in ((args.depthnav_python, "torch2onnx"),
                      (args.tflite_python, "onnx2tflite")):
        if not Path(py).is_file():
            sys.exit(f"interpreter for stage {stage} not found: {py} "
                     "(set --depthnav-python/--tflite-python or the env vars)")
        rc = subprocess.call([py, str(Path(__file__).resolve()),
                              "--stage", stage] + passthrough)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
