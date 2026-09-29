#!/usr/bin/env python
"""
Hover / step probe for an airframe (method `hover_probe` in the registry).

Measures what the offboards otherwise ASSUME about the vehicle PX4 is flying:
the hover throttle (PX4's own collective thrust setpoint, read back from the
ATTITUDE_TARGET stream while PX4's position controller holds a hover), and
whether PX4's rate/attitude loops -- tuned for the Iris airframe (none_iris,
SYS_AUTOSTART 10016) -- are stable on it (roll/pitch rate RMS in hover, and
the tilt/overshoot of a 2 m/s velocity step).

It flies no policy. Phases:
  CLIMB   position setpoint at --climb-alt, until settled
  HOVER   --hover-s seconds of position hold; thrust sampled when |vz| < 0.05
          and |alt err| < 0.10 (PX4 MPC_THR_HOVER is pushed to --hover-guess
          first; the probe reads what PX4 ACTUALLY commands, which is the
          hover throttle whatever the guess -- PX4's integrator and hover
          estimator absorb the difference)
  STEP    velocity setpoint --step-v m/s along -y ENU for --step-s, then 0
  LAND

Writes hover_probe.json next to $SUPERFLY_STATE_LOG (the runner sets it to the
trial dir) and prints it. Accepts and ignores the harness's usual offboard
arguments (--checkpoint, --depth, --goal).
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pymavlink import mavutil

from superfly.common.px4_offboard import (
    HEARTBEAT_HZ, DroneState, wait_for_heartbeat, request_stream_rates,
    set_offboard_mode, arm, retry_offboard_arm, send_position_target_ned,
    send_velocity_target_ned, send_land_command, send_heartbeat, set_param_float,
)
from superfly.common.sentinels import mark_policy_phase, mark_offboard_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)       # ignored (harness contract)
    ap.add_argument("--connect", default="udp:localhost:14550")
    ap.add_argument("--depth", action="store_true")     # ignored
    ap.add_argument("--goal", type=float, nargs="+", default=None)   # ignored
    ap.add_argument("--climb-alt", type=float, default=2.0)
    ap.add_argument("--hover-s", type=float, default=15.0)
    ap.add_argument("--step-v", type=float, default=2.0)
    ap.add_argument("--step-s", type=float, default=4.0)
    ap.add_argument("--hover-guess", type=float, default=0.5,
                    help="MPC_THR_HOVER pushed before arming (a starting point only).")
    args = ap.parse_args()

    mav = mavutil.mavlink_connection(args.connect)
    wait_for_heartbeat(mav)
    request_stream_rates(mav, 50.0)
    # ATTITUDE_TARGET (83): PX4's own attitude + collective thrust setpoint.
    mav.mav.command_long_send(mav.target_system, mav.target_component,
                              mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                              83.0, 20000.0, 0, 0, 0, 0, 0)

    state = DroneState()
    lock = threading.Lock()
    thr = {"t": 0.0, "thrust": None}
    stop = threading.Event()

    def rx():
        while not stop.is_set():
            msg = mav.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue
            ty = msg.get_type()
            if ty == "ATTITUDE_QUATERNION":
                state.update_from_attitude(msg)
            elif ty == "LOCAL_POSITION_NED":
                state.update_from_local_position(msg)
            elif ty == "HEARTBEAT" and msg.get_srcSystem() != 255:
                state.update_from_heartbeat(msg)
            elif ty == "STATUSTEXT" and msg.severity <= 4:
                print(f"[px4 statustext sev={msg.severity}] {msg.text}", flush=True)
            elif ty == "ATTITUDE_TARGET":
                with lock:
                    thr["t"], thr["thrust"] = time.time(), float(msg.thrust)
    threading.Thread(target=rx, daemon=True).start()

    set_param_float(mav, "MPC_THR_HOVER", float(args.hover_guess))
    time.sleep(0.5)
    pos0, _, _, yaw0 = state.get()
    xn, ye, zd = pos0[1], pos0[0], -args.climb_alt
    yaw_ned = math.atan2(math.sin(math.pi / 2 - yaw0), math.cos(math.pi / 2 - yaw0))
    for _ in range(30):
        send_position_target_ned(mav, xn, ye, zd, yaw_ned)
        send_heartbeat(mav)
        time.sleep(0.05)
    set_offboard_mode(mav)
    time.sleep(0.5)
    arm(mav)

    rows = []            # t, phase, z, vz, v_step (-vy ENU), thrust, tilt_deg, |omega_xy|
    phase, t_phase = "CLIMB", time.time()
    last_arm, last_hb = time.time(), 0.0
    t0 = time.time()
    dt = 0.02
    try:
        while True:
            now = time.time()
            if now - last_hb > 1.0 / HEARTBEAT_HZ:
                send_heartbeat(mav)
                last_hb = now
            pos, vel, R, om, yaw = state.get_full()
            with lock:
                th = thr["thrust"] if now - thr["t"] < 0.2 else None
            tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
            rows.append((now - t0, phase, float(pos[2]), float(vel[2]), -float(vel[1]),
                         th, tilt, float(np.hypot(om[0], om[1]))))
            if phase == "CLIMB":
                send_position_target_ned(mav, xn, ye, zd, yaw_ned)
                last_arm = retry_offboard_arm(mav, state, last_arm)
                if (abs(pos[2] - args.climb_alt) < 0.15 and np.linalg.norm(vel) < 0.15
                        and state.offboard and state.armed):
                    phase, t_phase = "HOVER", now
                    mark_policy_phase("start")
                    print(f"[probe] HOVER at z={pos[2]:.2f} (t={now - t0:.1f}s)", flush=True)
                elif now - t0 > 60:
                    print("[probe] never settled in CLIMB", flush=True)
                    break
            elif phase == "HOVER":
                send_position_target_ned(mav, xn, ye, zd, yaw_ned)
                if now - t_phase > args.hover_s:
                    phase, t_phase = "STEP", now
            elif phase == "STEP":
                v = args.step_v if now - t_phase < args.step_s else 0.0
                # -y ENU (south): away from the procedural fields, which run +y
                send_velocity_target_ned(mav, -v, 0.0, 0.0, yaw_ned)
                if now - t_phase > args.step_s + 4.0:
                    phase, t_phase = "LAND", now
                    mark_policy_phase("end")
                    send_land_command(mav)
            elif phase == "LAND":
                if not state.armed or now - t_phase > 30:
                    break
            time.sleep(dt)
    finally:
        stop.set()
        res = summarize(rows, args)
        out = os.environ.get("SUPERFLY_STATE_LOG")
        if out:
            Path(out).with_name("hover_probe.json").write_text(json.dumps(res, indent=2))
        print("[probe] RESULT " + json.dumps(res), flush=True)
        mav.mav.command_long_send(mav.target_system, mav.target_component,
                                  mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                  0, 0, 0, 0, 0, 0, 0, 0)
        mark_offboard_done()


def summarize(rows, args):
    res = {"climb_alt": args.climb_alt, "hover_guess": args.hover_guess}
    hov = [r for r in rows if r[1] == "HOVER"]
    if hov:
        # skip the first 3 s of the hover (integrator settling)
        t_h0 = hov[0][0]
        settled = [r for r in hov if r[0] - t_h0 > 3.0 and abs(r[3]) < 0.05
                   and abs(r[2] - args.climb_alt) < 0.10 and r[5] is not None]
        th = np.array([r[5] for r in settled])
        cos_t = np.cos(np.radians([r[6] for r in settled]))
        if th.size:
            res.update(hover_thrust_median=round(float(np.median(th * cos_t)), 4),
                       hover_thrust_p10=round(float(np.percentile(th, 10)), 4),
                       hover_thrust_p90=round(float(np.percentile(th, 90)), 4),
                       hover_samples=int(th.size))
        z = np.array([r[2] for r in hov])
        res.update(hover_z_std=round(float(np.std(z)), 4),
                   hover_omega_xy_rms=round(float(np.sqrt(np.mean(np.square([r[7] for r in hov])))), 4),
                   hover_tilt_max_deg=round(float(max(r[6] for r in hov)), 2))
    st = [r for r in rows if r[1] == "STEP"]
    if st:
        vx = np.array([r[4] for r in st])
        res.update(step_vx_max=round(float(vx.max()), 3),
                   step_overshoot=round(float(vx.max() - args.step_v), 3),
                   step_tilt_max_deg=round(float(max(r[6] for r in st)), 2),
                   step_omega_xy_rms=round(float(np.sqrt(np.mean(np.square([r[7] for r in st])))), 4),
                   step_z_min=round(float(min(r[2] for r in st)), 3),
                   step_z_max=round(float(max(r[2] for r in st)), 3))
    return res


if __name__ == "__main__":
    main()
