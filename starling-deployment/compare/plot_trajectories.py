#!/usr/bin/env python
"""Generate clean trajectory plots for a comparison results directory.

For each environment (subfolder) it writes, into plots/<env>/:
  topdown.png   -- XY overlay of all policies + analytic obstacles + start/goal
  altitude.png  -- altitude vs along-track progress for all policies
  speed.png     -- speed vs post-takeoff time for all policies
  dashboard.png -- the three panels + a per-policy metrics table

Trajectory / obstacle conventions match metrics.py:
  traj = [t, x, y, z, vx, vy, vz]
  spheres (cx,cy,cz,r) | boxes (cx,cy,cz,hx,hy,hz[,roll,pitch,yaw])
  cyl_v (cx,cy,r) axis +Z | cyl_h (cx,cy,cz,r) axis +X, length CYLH_LEN
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import takeoff_index, CYLH_LEN  # noqa: E402

# ---- palette (dataviz categorical slots 1-4, fixed pre-validated order) ------
POLICY_ORDER = ["diffaero", "diffphys", "depthnav", "agile"]
POLICY_COLOR = {
    "diffaero": "#2a78d6",  # blue
    "diffphys": "#1baf7a",  # aqua
    "depthnav": "#eda100",  # yellow
    "agile":    "#8a3ffc",  # purple
}
POLICY_LABEL = {
    "diffaero": "DiffAero",
    "diffphys": "DiffPhysics",
    "depthnav": "DepthNav",
    "agile":    "Agile",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e3df"
OBST_FILL = "#d6d6d1"
OBST_EDGE = "#b5b5ad"
START_C = "#199e70"
GOAL_C = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 22,
    "font.family": "sans-serif",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.8,
})


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def load_env(env_dir):
    """Return (fields dict, {policy: {t,P,V,speed,meta}}) for one environment."""
    runs = {}
    fields = None
    for policy in POLICY_ORDER:
        npz = os.path.join(env_dir, policy, "traj.npz")
        if not os.path.isfile(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        traj = np.asarray(z["traj"], float)
        if traj.shape[0] == 0:
            continue
        t, P, V = traj[:, 0], traj[:, 1:4], traj[:, 4:7]
        speed = np.linalg.norm(V, axis=1)
        k = takeoff_index(speed)
        k = 0 if k is None else k
        meta = {}
        mpath = os.path.join(env_dir, policy, "metrics.json")
        if os.path.isfile(mpath):
            meta = json.load(open(mpath))
        runs[policy] = dict(t=t, P=P, V=V, speed=speed, k=k, meta=meta,
                            start=np.asarray(z["start"], float),
                            goal=np.asarray(z["goal"], float))
        if fields is None and all(key in z for key in ("spheres", "boxes", "cyl_v", "cyl_h")):
            fields = {key: np.asarray(z[key], float) for key in ("spheres", "boxes", "cyl_v", "cyl_h")}
    return fields, runs


def draw_obstacles(ax, fields):
    if not fields:
        return
    for cx, cy, cz, r in fields["spheres"]:
        ax.add_patch(Circle((cx, cy), r, facecolor=OBST_FILL, edgecolor=OBST_EDGE,
                            linewidth=0.6, zorder=1))
    for b in fields["boxes"]:
        cx, cy = b[0], b[1]
        hx, hy = b[3], b[4]
        yaw = b[8] if len(b) >= 9 else 0.0
        rect = Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, facecolor=OBST_FILL,
                         edgecolor=OBST_EDGE, linewidth=0.6, zorder=1)
        rect.set_transform(Affine2D().rotate_around(cx, cy, yaw) + ax.transData)
        ax.add_patch(rect)
    for cx, cy, r in fields["cyl_v"]:
        ax.add_patch(Circle((cx, cy), r, facecolor=OBST_FILL, edgecolor=OBST_EDGE,
                            linewidth=0.6, zorder=1))
    for cx, cy, cz, r in fields["cyl_h"]:
        ax.add_patch(Rectangle((cx - CYLH_LEN / 2, cy - r), CYLH_LEN, 2 * r,
                              facecolor=OBST_FILL, edgecolor=OBST_EDGE,
                              linewidth=0.6, zorder=1))


def draw_start_goal_xy(ax, start, goal, goal_radius=1.0):
    ax.add_patch(Circle((goal[0], goal[1]), goal_radius, facecolor="none",
                        edgecolor=GOAL_C, linewidth=1.2, linestyle="--", zorder=3))
    ax.scatter([start[0]], [start[1]], s=90, marker="o", facecolor="white",
               edgecolor=START_C, linewidth=2.0, zorder=6)
    ax.scatter([goal[0]], [goal[1]], s=180, marker="*", facecolor=GOAL_C,
               edgecolor="white", linewidth=0.8, zorder=6)
    ax.annotate("start", (start[0], start[1]), textcoords="offset points",
                xytext=(8, 8), fontsize=18, color=INK2, weight="bold")
    ax.annotate("goal", (goal[0], goal[1]), textcoords="offset points",
                xytext=(8, 8), fontsize=18, color=INK2, weight="bold")


def policy_present(runs):
    return [p for p in POLICY_ORDER if p in runs]


def obstacle_bbox(fields):
    """(xmin,xmax,ymin,ymax) covering all obstacle footprints, or None."""
    xs, ys = [], []
    for cx, cy, cz, r in fields.get("spheres", []):
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    for b in fields.get("boxes", []):
        rad = math.hypot(b[3], b[4])  # rotation-safe conservative radius
        xs += [b[0] - rad, b[0] + rad]; ys += [b[1] - rad, b[1] + rad]
    for cx, cy, r in fields.get("cyl_v", []):
        xs += [cx - r, cx + r]; ys += [cy - r, cy + r]
    for cx, cy, cz, r in fields.get("cyl_h", []):
        xs += [cx - CYLH_LEN / 2, cx + CYLH_LEN / 2]; ys += [cy - r, cy + r]
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def square_limits(xmin, xmax, ymin, ymax, pad=0.10):
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(xmax - xmin, ymax - ymin) / 2 * (1 + pad)
    half = max(half, 1.0)
    return (cx - half, cx + half), (cy - half, cy + half)


def view_limits(fields, runs):
    """Square view: frame the mission (start/goal/obstacles) when an analytic
    field exists so a runaway can't dominate; else frame the full paths."""
    ref = next(iter(runs.values()))
    bb = obstacle_bbox(fields)
    xs = [ref["start"][0], ref["goal"][0]]
    ys = [ref["start"][1], ref["goal"][1]]
    if bb is not None:
        xs += [bb[0], bb[1]]; ys += [bb[2], bb[3]]
    else:  # USD scene: no obstacles -> include the flown paths
        for p in policy_present(runs):
            P = runs[p]["P"][runs[p]["k"]:]
            xs += [P[:, 0].min(), P[:, 0].max()]
            ys += [P[:, 1].min(), P[:, 1].max()]
    return square_limits(min(xs), max(xs), min(ys), max(ys))


def offframe_policies(runs, xlim, ylim):
    off = []
    for p in policy_present(runs):
        P = runs[p]["P"][runs[p]["k"]:]
        if (P[:, 0].max() > xlim[1] or P[:, 0].min() < xlim[0]
                or P[:, 1].max() > ylim[1] or P[:, 1].min() < ylim[0]):
            off.append(p)
    return off


def plot_topdown(ax, fields, runs):
    draw_obstacles(ax, fields)
    ref = next(iter(runs.values()))
    draw_start_goal_xy(ax, ref["start"], ref["goal"],
                       ref["meta"].get("hyperparams", {}).get("goal_radius", 1.0))
    for p in policy_present(runs):
        r = runs[p]
        P = r["P"][r["k"]:]
        ax.plot(P[:, 0], P[:, 1], color=POLICY_COLOR[p], linewidth=2.0,
                solid_capstyle="round", alpha=0.95, zorder=5, label=POLICY_LABEL[p])
        ax.scatter([P[-1, 0]], [P[-1, 1]], s=26, color=POLICY_COLOR[p],
                   edgecolor="white", linewidth=0.8, zorder=6)
    xlim, ylim = view_limits(fields, runs)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    off = offframe_policies(runs, xlim, ylim)
    if off:
        note = "exits frame: " + ", ".join(POLICY_LABEL[p] for p in off)
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=17,
                color=INK2, style="italic", va="bottom", ha="left")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    style_axes(ax)


def plot_altitude(ax, runs):
    ref = next(iter(runs.values()))
    start, goal = ref["start"], ref["goal"]
    axis = goal - start
    L = np.linalg.norm(axis[:2])
    u = axis[:2] / (L if L else 1.0)
    for p in policy_present(runs):
        r = runs[p]
        P = r["P"][r["k"]:]
        prog = (P[:, :2] - start[:2]) @ u
        ax.plot(prog, P[:, 2], color=POLICY_COLOR[p], linewidth=2.0,
                solid_capstyle="round", alpha=0.95, label=POLICY_LABEL[p])
    ax.axvline(L, color=GOAL_C, linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlim(-3, L + 4)  # focus on the start->goal corridor
    ax.annotate("goal", (L, ax.get_ylim()[1]), textcoords="offset points",
                xytext=(-46, -26), fontsize=18, color=INK2)
    ax.set_xlabel("progress toward goal  [m]")
    ax.set_ylabel("altitude  z  [m]")
    style_axes(ax)


def plot_speed(ax, runs):
    for p in policy_present(runs):
        r = runs[p]
        t = r["t"][r["k"]:] - r["t"][r["k"]]
        ax.plot(t, r["speed"][r["k"]:], color=POLICY_COLOR[p], linewidth=1.8,
                solid_capstyle="round", alpha=0.9, label=POLICY_LABEL[p])
    ax.set_xlabel("flight time  [s]")
    ax.set_ylabel("speed  [m/s]")
    style_axes(ax)


def legend_handles(runs):
    return [Line2D([0], [0], color=POLICY_COLOR[p], linewidth=2.5,
                   label=POLICY_LABEL[p]) for p in policy_present(runs)]


def save_single(path, plotfn, runs, fields=None, title=""):
    fig, ax = plt.subplots(figsize=(11, 11) if plotfn is plot_topdown else (11, 6.2))
    if fields is not None:
        plotfn(ax, fields, runs)
    else:
        plotfn(ax, runs)
    ax.legend(handles=legend_handles(runs), frameon=False, loc="best", fontsize=20)
    if title:
        ax.set_title(title, fontsize=26, weight="bold", color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(x, s="{:.1f}"):
    return "-" if x is None else s.format(x)


def save_dashboard(path, env, fields, runs):
    # top row: top-down + altitude; bottom: full-width metrics table (room for 2x font)
    fig = plt.figure(figsize=(21, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1.35, 0.5],
                          hspace=0.22, wspace=0.2)
    ax_td = fig.add_subplot(gs[0, 0])
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, :])

    plot_topdown(ax_td, fields, runs)
    plot_altitude(ax_alt, runs)

    # metrics table (spans full width so 2x-size headers never collide)
    ax_tbl.axis("off")
    cols = ["policy", "success", "collided", "time-to-goal", "mean speed", "peak speed", "min clearance"]
    rows = []
    colors = []
    for p in policy_present(runs):
        m = runs[p]["meta"]
        rows.append([
            POLICY_LABEL[p],
            "yes" if m.get("success") else "no",
            "yes" if m.get("collided") else "no",
            fmt(m.get("time_to_goal_s"), "{:.1f} s"),
            fmt(m.get("mean_speed_mps"), "{:.2f} m/s"),
            fmt(m.get("peak_speed_mps"), "{:.2f} m/s"),
            fmt(m.get("min_clearance_m"), "{:.2f} m"),
        ])
        colors.append(POLICY_COLOR[p])
    tbl = ax_tbl.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                       colWidths=[0.16, 0.12, 0.12, 0.16, 0.16, 0.14, 0.16])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(21)
    tbl.scale(1, 3.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#f0f0ec")
        elif col == 0:
            cell.set_text_props(weight="bold", color=colors[row - 1])

    fig.legend(handles=legend_handles(runs), frameon=False, ncol=len(runs),
               loc="lower center", fontsize=22, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(env, fontsize=30, weight="bold", color=INK, x=0.5, y=0.985)
    fig.text(0.06, 0.955, "trajectory comparison  ·  4 policies", fontsize=20, color=INK2)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.97)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else "results/combined_jul08"
    results = os.path.abspath(results)
    outroot = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results, "plots")
    outroot = os.path.abspath(outroot)

    envs = sorted(d for d in os.listdir(results)
                  if os.path.isdir(os.path.join(results, d)) and d != "plots")
    for env in envs:
        env_dir = os.path.join(results, env)
        fields, runs = load_env(env_dir)
        if not runs:
            print(f"[skip] {env}: no trajectories")
            continue
        outdir = os.path.join(outroot, env)
        os.makedirs(outdir, exist_ok=True)
        save_single(os.path.join(outdir, "topdown.png"), plot_topdown, runs,
                    fields=fields or {}, title=f"{env}  ·  top-down trajectories")
        save_single(os.path.join(outdir, "altitude.png"), plot_altitude, runs,
                    title=f"{env}  ·  altitude profile")
        save_single(os.path.join(outdir, "speed.png"), plot_speed, runs,
                    title=f"{env}  ·  speed profile")
        save_dashboard(os.path.join(outdir, "dashboard.png"), env, fields or {}, runs)
        tag = "analytic obstacles" if fields else "no obstacle field (USD scene)"
        print(f"[ok]   {env}: {len(runs)} policies, {tag} -> {outdir}")


if __name__ == "__main__":
    main()
