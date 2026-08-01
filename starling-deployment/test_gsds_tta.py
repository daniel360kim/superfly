"""CPU-only validation of the X15 TTA aggregation math in wrapper/gsds_core.

No GPU, no checkpoint, no MAVLink: exercises the pure aggregation function
(tta_select_average) and the env gate (_tta_n_from_env) on synthetic model
outputs (M=3 modes, known costs). Run with the gsds deploy interpreter:

    /home/danielkim/gs_drone_sim/.venv/bin/python test_gsds_tta.py

Covers:
  (a) N=1/unset gate off + S=1 aggregation bit-identical to plain argmin
  (b) select-then-average invariant to per-sample mode permutation
  (c) disagreement counting correctness
  (d) explicit (depth-guarded) mode override respected
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrapper.gsds_core import _tta_n_from_env, tta_select_average  # noqa: E402

M, T = 3, 10
rng = np.random.default_rng(0)
FAILURES = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILURES.append(name)


# ---- (a) env gate + S=1 bit-identity ---------------------------------------
print("(a) env gate and N=1 path")
for val, expect in [(None, 1), ("", 1), ("1", 1), ("0", 1), ("-3", 1),
                    ("2", 2), ("4", 4)]:
    if val is None:
        os.environ.pop("GSDS_TTA_N", None)
    else:
        os.environ["GSDS_TTA_N"] = val
    check(f"GSDS_TTA_N={val!r} -> {expect}", _tta_n_from_env() == expect)
os.environ["GSDS_TTA_N"] = "four"
try:
    _tta_n_from_env()
    check("garbage GSDS_TTA_N raises", False)
except ValueError:
    check("garbage GSDS_TTA_N raises", True)
os.environ.pop("GSDS_TTA_N", None)

# S=1 aggregation must equal the plain argmin-cost selection BITWISE
# (deploy's N<=1 branch never calls tta_select_average at all — this pins the
# math down anyway, so S=1 could be substituted with zero drift).
traj1 = rng.standard_normal((1, M, T, 3)).astype(np.float32)
cost1 = np.array([[0.7, 0.2, 0.9]], dtype=np.float32)
avg1, modes1, dis1 = tta_select_average(traj1, cost1)
check("S=1 avg bit-identical to argmin trajectory",
      np.array_equal(avg1, traj1[0, 1]) and avg1.dtype == traj1.dtype)
check("S=1 mode = argmin cost", modes1.tolist() == [1])
check("S=1 disagreement = 0.0", dis1 == 0.0)

# ---- (b) invariance to per-sample mode permutation -------------------------
print("(b) mode-permutation invariance")
S = 4
traj = rng.standard_normal((S, M, T, 3)).astype(np.float32)
cost = rng.standard_normal((S, M)).astype(np.float32)
avg, modes, dis = tta_select_average(traj, cost)

traj_p = np.empty_like(traj)
cost_p = np.empty_like(cost)
for s in range(S):
    perm = rng.permutation(M)          # independent permutation per sample
    traj_p[s] = traj[s, perm]
    cost_p[s] = cost[s, perm]
avg_p, modes_p, dis_p = tta_select_average(traj_p, cost_p)
check("averaged trajectory invariant under per-sample mode permutation",
      np.array_equal(avg, avg_p))
# and each sample still committed to the same PHYSICAL trajectory
same_pick = all(np.array_equal(traj[s, modes[s]], traj_p[s, modes_p[s]])
                for s in range(S))
check("per-sample committed trajectory unchanged by permutation", same_pick)

# counter-example documenting WHY: naive per-mode-index averaging is NOT
# permutation invariant (this failing to match is the expected behavior)
naive = traj.mean(axis=0)[cost.mean(axis=0).argmin()]
naive_p = traj_p.mean(axis=0)[cost_p.mean(axis=0).argmin()]
check("naive per-index averaging is permutation-SENSITIVE (sanity)",
      not np.allclose(naive, naive_p))

# ---- (c) disagreement counting ---------------------------------------------
print("(c) disagreement counting")
# known costs -> samples select modes [0, 0, 1, 2]: 2 of 3 jittered samples
# disagree with the clean sample -> rate 2/3
cost_k = np.array([[0.1, 0.5, 0.5],
                   [0.2, 0.9, 0.9],
                   [0.9, 0.1, 0.9],
                   [0.9, 0.9, 0.1]], dtype=np.float32)
traj_k = rng.standard_normal((4, M, T, 3)).astype(np.float32)
avg_k, modes_k, dis_k = tta_select_average(traj_k, cost_k)
check("modes = [0,0,1,2]", modes_k.tolist() == [0, 0, 1, 2])
check("disagreement = 2/3", np.isclose(dis_k, 2.0 / 3.0))
check("avg = mean of the three selected trajectories",
      np.allclose(avg_k, np.stack([traj_k[0, 0], traj_k[1, 0],
                                   traj_k[2, 1], traj_k[3, 2]]).mean(0)))
# unanimity -> 0.0
cost_u = np.tile(np.array([[0.9, 0.1, 0.9]], np.float32), (4, 1))
_, modes_u, dis_u = tta_select_average(traj_k, cost_u)
check("unanimous selection -> disagreement 0.0",
      dis_u == 0.0 and modes_u.tolist() == [1, 1, 1, 1])

# ---- (d) explicit mode override (depth-guard path) -------------------------
print("(d) guarded-mode override")
forced = np.array([2, 2, 2, 2])
avg_f, modes_f, dis_f = tta_select_average(traj_k, cost_k, modes=forced)
check("override modes respected + avg over forced picks",
      modes_f.tolist() == [2, 2, 2, 2] and dis_f == 0.0
      and np.allclose(avg_f, traj_k[:, 2].mean(axis=0)))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
