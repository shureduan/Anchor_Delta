"""Scripted expert for bottle -> box in sim: the same 12-step plan as the real script (real/scripts/11_pick_place.py),
executed as a stream of 10 Hz tool targets at 5 cm/s, using privileged object poses (the student only gets images).

Episode record (per 10 Hz step): wrist RGB, tool xyz, gripper opening, action = next tool target xyz + gripper cmd,
phase id. Success = after release and 1 s settling, bottle base inside the box's inner square, bottle upright.
"""
from __future__ import annotations

import numpy as np

from scene import BOTTLE_H, BOX_H, BOX_T, BOX_W, TABLE_Z, Sim

V = 0.05                     # m/s, as on the real robot
DT = 1.0 / Sim.CTRL_HZ
GRIP_BELOW_TOP = 0.012       # tool z below the cap top at the grasp (real script default)
PHASES = ['approach', 'descend', 'close', 'lift', 'transfer', 'lower', 'release', 'retreat']


def sample_scene(rng):
    """Bottle and box placements like the real runs (bottle-box 10-20 cm, both on the table, reachable)."""
    for _ in range(1000):
        b = np.array([rng.uniform(0.22, 0.48), rng.uniform(-0.05, 0.30)])
        ang = rng.uniform(0, 2 * np.pi); dist = rng.uniform(0.10, 0.20)
        x = b + dist * np.array([np.cos(ang), np.sin(ang)])
        if 0.20 <= x[0] <= 0.52 and -0.08 <= x[1] <= 0.32:
            return b, x, rng.uniform(0, np.pi / 2)
    raise RuntimeError('no placement')


def centre_offset(height_above_table, aim_above_table=0.08):
    """Table-frame xy offset from the tool to where the colour image centre looks (tool vertical), at a given height."""
    from scene import CAM_POS_TOOL, COLOR_K, R_TOOL, R_TOOL_CVCAM
    d = np.array([(COLOR_K['w'] / 2 - COLOR_K['cx']) / COLOR_K['fx'], (COLOR_K['h'] / 2 - COLOR_K['cy']) / COLOR_K['fy'], 1.0])
    d_w = R_TOOL @ R_TOOL_CVCAM @ d; c_w = R_TOOL @ CAM_POS_TOOL
    s = (aim_above_table - height_above_table - c_w[2]) / d_w[2]
    return (c_w + s * d_w)[:2]


def start_pose(rng, b, x):
    """Observation pose like the real --auto re-centring: the bottle-box midpoint near the image centre (+ noise)."""
    mid = (b + x) / 2
    z = rng.uniform(0.40, 0.46)
    xy = mid - centre_offset(z - TABLE_Z) + rng.normal(0, 0.025, 2)
    tilt = (180.0 + rng.normal(0, 1.0), rng.normal(0, 1.0), 97.5)
    return np.array([xy[0], xy[1], z]), tilt


def waypoints(cap_xy, cap_top, box_xy, observable=False):
    """observable=True (v5): no hidden waits before a gripper change -- close as soon as the grasp pose is reached and
    release as soon as the place height is reached, so the gripper command is a function of what the student can observe
    (v4 waited a fixed 0.3 s at a stationary pose, which no observation can time; real L3 deadlock 2026-11-22)."""
    w_pre = 0.0 if observable else 0.3
    tz = TABLE_Z
    g = cap_top - GRIP_BELOW_TOP
    hang = g - tz
    z_carry = tz + BOX_H + 0.05 + hang
    z_rim = tz + BOX_H + 0.02 + hang
    z_place = tz + 0.018 + hang
    c = np.array([cap_xy[0], cap_xy[1]]); bx = np.array(box_xy)
    return [  # (phase, target xyz, gripper cmd, hold seconds after reaching)
        ('approach', [*c, cap_top + 0.10], 0.0, 0.3),
        ('descend', [*c, g], 0.0, w_pre),
        ('close', [*c, g], 1.0, 1.5),
        ('lift', [*c, z_carry], 1.0, 0.2),
        ('transfer', [*bx, z_carry], 1.0, 0.3),
        ('lower', [*bx, z_rim], 1.0, 0.2),
        ('lower', [*bx, z_place], 1.0, w_pre),
        ('release', [*bx, z_place], 0.0, 1.0),
        ('retreat', [*bx, z_carry], 0.0, 0.0),
    ]


def run_episode(sim: Sim, rng, render=True, noise=0.0, randomize=True, observable=False):
    """Closed-loop expert: every 0.1 s the label is a step of <= V*DT from the CURRENT tool position towards the
    active waypoint; the executed action adds Gaussian noise (std `noise`, metres) in free-space motion only (approach,
    lift, transfer, retreat; not within 3 cm of a waypoint) so the data contain recoveries without breaking contacts."""
    b, x, yaw = sample_scene(rng)
    xyz0, tilt = start_pose(rng, b, x)
    sim.reset(xyz0, b, x, yaw, tool_euler_deg=tilt)
    if randomize:
        sim.randomize(rng)
    from scene import R_TOOL, mat2quat
    sim.d.mocap_quat[sim.mocap] = mat2quat(R_TOOL)          # commanded vertical from now on, as on the real robot
    bp, _ = sim.body_pose('bottle'); xp, _ = sim.body_pose('box')
    plan = waypoints(bp[:2], bp[2] + BOTTLE_H, xp[:2], observable=observable)
    rec = dict(rgb=[], seg=[], tool=[], grip=[], action=[], phase=[], scene=dict(bottle=b.tolist(), box=x.tolist(), box_yaw=float(yaw),
                                                                       start=xyz0.tolist(), tilt=list(tilt), noise=noise))
    cmd = sim.tool()                                       # last commanded (clean) target
    for phase, tgt, gc, hold in plan:
        tgt = np.array(tgt, float)
        n_hold = int(round(hold / DT)); held = 0; guard = 0
        while guard < 400:
            guard += 1
            cur = sim.tool()
            delta = tgt - cmd
            dist = np.linalg.norm(delta)
            step = delta if dist <= V * DT else delta / dist * V * DT
            label = cmd + step
            free = phase in ('approach', 'lift', 'transfer', 'retreat') and np.linalg.norm(tgt - cmd) > 0.03
            a_exec = label + (rng.normal(0, noise, 3) if (noise and free) else 0.0)   # no noise near contacts
            if render:
                rec['rgb'].append(sim.render())
                if getattr(sim, 'record_seg', False):
                    rec['seg'].append(sim.render_seg())
            rec['tool'].append(cur); rec['grip'].append(sim.gripper_open_frac())
            rec['action'].append(np.r_[label, gc]); rec['phase'].append(PHASES.index(phase))
            sim.step(a_exec, gc)
            cmd = label
            if np.linalg.norm(tgt - cmd) < 1e-6:
                held += 1
                if held > n_hold:
                    break
    for _ in range(10):                 # 1 s settling
        sim.step(cmd, 0.0)
    rec['success'], rec['info'] = success(sim)
    return rec


def success(sim):
    bp, bR = sim.body_pose('bottle'); xp, xR = sim.body_pose('box')
    rel = xR.T @ (bp - xp)              # bottle base in the box frame
    inner = BOX_W / 2 - BOX_T
    upright = bR[2, 2] > np.cos(np.radians(20))
    inside = abs(rel[0]) < inner and abs(rel[1]) < inner and rel[2] < 0.03
    return bool(inside and upright), dict(rel=rel.round(4).tolist(), tilt_deg=float(np.degrees(np.arccos(np.clip(bR[2, 2], -1, 1)))),
                                         bottle_z=float(bp[2]))
