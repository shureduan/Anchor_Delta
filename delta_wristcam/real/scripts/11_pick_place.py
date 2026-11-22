"""Pick the bottle by its cap (top-down) and put it into the green box. Every motion is printed and waits for the
operator. Keep a hand on the e-stop.

    conda activate delta-kinova
    python real/scripts/11_pick_place.py --auto          # detect bottle + box now (re-centres the camera if needed)
    python real/scripts/11_pick_place.py --auto --record # same, and log the episode (wrist RGB-D + robot state, 10 Hz)
                                                         #   -> real/out/demos/<stamp>/ ; asks for success at the end
    python real/scripts/11_pick_place.py --cap X Y TOP --box X Y --table Z

    X, Y, TOP, Z in metres, robot base frame (from 10_scene.py / box_fit.py). TOP = cap top height (base z).
    --auto: start with the gripper pointing down ~40 cm above the table, bottle and box roughly in view. If the box
    fit is not accepted (< 97 % of the green pixels inside the fitted box, < 75 % of the fitted box green, or > 5 % of it
    outside the image) it proposes a camera
    re-centring move (same height, <= 15 cm, waits for Enter) so that the bottle-box midpoint is in the image
    centre, and detects again (at most 3 times).

At every prompt:
    Enter (empty)  execute the shown motion           (anything not understood does nothing and asks again)
    a dx dy dz     shift this target by dx, dy, dz centimetres (base frame, each <= 5) and show it again
    s              skip this step
    q / quit       stop (the arm stays where it is)
Ctrl-C during a motion sends Stop.

Safety: Cartesian moves only, gripper kept vertical, translation speed <= 5 cm/s, rotation <= 20 deg/s, targets
outside X [0.10, 0.55], Y [-0.25, 0.30], Z [TOP - 0.03, 0.55] are refused (observation moves: Z [0.25, 0.55]).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import CALIB, Connection, pose_matrix, T_TOOL_COLOR, tool_pose  # noqa: E402

V_LIN, V_ROT = 0.05, 20.0
THETA = (180.0, 0.0, 97.5)      # gripper vertical, same yaw as the observation poses
XY_LO, XY_HI = np.array([0.10, -0.25]), np.array([0.55, 0.30])


def make_move(base, cyc, z_lo, z_hi):
    from kortex_api.autogen.messages import Base_pb2
    lo, hi = np.r_[XY_LO, z_lo], np.r_[XY_HI, z_hi]

    def move(xyz):
        xyz = np.asarray(xyz, float)
        if np.any(xyz < lo) or np.any(xyz > hi):
            print(f'REFUSED: target {xyz.round(3)} outside workspace {lo.round(3)} .. {hi.round(3)}'); return False
        act = Base_pb2.Action(); act.name = 'pp'
        p = act.reach_pose.target_pose
        p.x, p.y, p.z = map(float, xyz); p.theta_x, p.theta_y, p.theta_z = THETA
        act.reach_pose.constraint.speed.translation = V_LIN
        act.reach_pose.constraint.speed.orientation = V_ROT
        done = threading.Event(); result = {}

        def cb(n):
            if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                result['ev'] = Base_pb2.ActionEvent.Name(n.action_event); done.set()
        h = base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
        try:
            base.ExecuteAction(act)
            if not done.wait(60):
                base.Stop(); print('timeout -> Stop'); return False
        except KeyboardInterrupt:
            base.Stop(); print('Ctrl-C -> Stop'); raise
        finally:
            base.Unsubscribe(h)
        now = tool_pose(cyc)
        print(f'   {result.get("ev")}  tool now ({now[0]:.3f}, {now[1]:.3f}, {now[2]:.3f})  err {np.linalg.norm(now[:3] - xyz) * 1000:.1f} mm')
        return result.get('ev') == 'ACTION_END'
    return move


def ask(prompt):
    """Return 'go', 'skip', 'quit' or ('adj', d_metres). Only an empty Enter means go."""
    while True:
        ans = input(prompt).strip().lower()
        if ans == '':
            return 'go'
        if ans in ('q', 'quit', 'exit', 'stop'):
            return 'quit'
        if ans in ('s', 'skip'):
            return 'skip'
        if ans.startswith('a '):
            try:
                d = np.array([float(v) for v in ans.split()[1:4]])
                assert d.shape == (3,) and np.all(np.abs(d) <= 5)
                return ('adj', d / 100.0)
            except (ValueError, AssertionError):
                print('usage: a dx dy dz   (centimetres, each |d| <= 5)'); continue
        print(f'not understood: "{ans}" -- nothing done. Press Enter alone to execute.')


def recentre_target(pose, point_xy, table_z):
    """Tool xy (same height) that puts the table point point_xy at the colour image centre."""
    C = CALIB['color_intrinsics']
    Tc = pose_matrix(pose) @ T_TOOL_COLOR
    d = np.array([(C['width'] / 2 - C['cx']) / C['fx'], (C['height'] / 2 - C['cy']) / C['fy'], 1.0])
    d_base = Tc[:3, :3] @ d
    s = (table_z + 0.08 - Tc[2, 3]) / d_base[2]           # aim at mid-object height (~8 cm above the table)
    hit = Tc[:3, 3] + s * d_base
    shift = np.asarray(point_xy) - hit[:2]
    n = np.linalg.norm(shift)
    if n > 0.15:
        shift *= 0.15 / n
    return pose[:3] + np.r_[shift, 0.0]


def auto_detect(base, cyc, rec):
    import importlib.util
    import cv2
    from box_fit import corners, fit, green_mask, project
    from kinova_common import REAL
    spec = importlib.util.spec_from_file_location('scene', str(Path(__file__).parent / '10_scene.py'))
    sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    obs_move = make_move(base, cyc, 0.25, 0.55)
    out = REAL / 'out' / 'scene'; out.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        t_req = time.time()
        p_before = tool_pose(cyc)
        color, depth = rec.latest(fresh_after=t_req + 0.15)
        pose = tool_pose(cyc)
        if np.abs(pose[:3] - p_before[:3]).max() > 0.002:
            sys.exit('arm moved during detection')
        rep, ov = sc.detect(color, depth, pose)
        tz = rep['table_z']
        stamp = time.strftime('%H%M%S')
        np.savez(out / f'auto_{stamp}.npz', color=color, depth=depth, pose=pose)
        if 'bottle' not in rep:
            cv2.imwrite(str(out / f'auto_{stamp}_overlay.jpg'), ov)
            sys.exit(f'no bottle cap (yellow) found; image {out / f"auto_{stamp}_overlay.jpg"}')
        b = rep['bottle']; cap = [b['cap_xy'][0], b['cap_xy'][1], tz + b['cap_top_above_table']]
        print(f'\n[detect {attempt + 1}] table z {tz:.4f}; cap ({cap[0]:.4f}, {cap[1]:.4f}) top z {cap[2]:.4f} ({b["cap_top_above_table"] * 100:.1f} cm above table)')
        ok, box_xy = False, None
        if 'box' in rep:
            q, iou = fit([(color, pose)], tz, rep['box']['center_xy'])
            uv = project(corners(q, tz), pose)
            hull = cv2.convexHull(uv.astype(np.float32)); full = cv2.contourArea(hull)
            m = np.zeros(color.shape[:2], np.uint8); cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
            outside = 1.0 - m.sum() / max(full, 1.0)
            gmask = green_mask(color).astype(bool); mb = m.astype(bool)
            explained = (gmask & mb).sum() / max(gmask.sum(), 1)    # green pixels inside the fitted box (wrong pose -> low)
            filled = (gmask & mb).sum() / max(mb.sum(), 1)          # box area that is green (occlusion -> low; highlights lower it a bit)
            cv2.polylines(ov, [uv[:4].astype(np.int32)], True, (0, 0, 255), 2); cv2.polylines(ov, [uv[4:].astype(np.int32)], True, (0, 128, 255), 2)
            print(f'box centre ({q[0]:.4f}, {q[1]:.4f}) yaw {np.degrees(q[2]):.1f} deg | green inside fit {explained:.3f} (need >= 0.97), '
                  f'fit area green {filled:.3f} (need >= 0.75), outside the image {max(outside, 0):.1%} (need <= 5 %)')
            box_xy = q[:2]
            ok = explained >= 0.97 and filled >= 0.75 and outside <= 0.05
            reason = ', '.join(r for r, bad in (('green outside the fitted box', explained < 0.97), ('fitted box not filled by green (occluded?)', filled < 0.75),
                                                ('box partly outside the image', outside > 0.05)) if bad)
        else:
            print('box: not found in this view'); reason = 'no box'
        cv2.imwrite(str(out / f'auto_{stamp}_overlay.jpg'), ov)
        print(f'overlay {out / f"auto_{stamp}_overlay.jpg"}')
        if ok:
            if ask('use these detections? Enter (empty) = yes, q = abort > ') != 'go':
                sys.exit('aborted')
            return cap, [box_xy[0], box_xy[1]], tz, dict(cap=cap, box_xy=[float(box_xy[0]), float(box_xy[1])], box_yaw_deg=float(np.degrees(q[2])),
                                                     box_iou=float(iou), table_z=float(tz), detect_frame=f'auto_{stamp}.npz')
        if attempt == 3:
            sys.exit(f'box fit still not accepted after 3 re-centring moves ({reason}); move the arm or the box by hand')
        mid = np.asarray(cap[:2]) if box_xy is None else (np.asarray(cap[:2]) + box_xy) / 2
        tgt = recentre_target(pose, mid, tz)
        print(f'box not accepted ({reason}) -> re-centre the camera on the bottle-box midpoint ({mid[0]:.3f}, {mid[1]:.3f})')
        r = ask(f'[re-centre] move tool to ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}) [from ({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})]\n'
                '  Enter (empty) = go | q = quit > ')
        if r != 'go':
            sys.exit('stopped by operator')
        if not obs_move(tgt):
            sys.exit('re-centring move failed')
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto', action='store_true')
    ap.add_argument('--cap', nargs=3, type=float)
    ap.add_argument('--box', nargs=2, type=float)
    ap.add_argument('--table', type=float)
    ap.add_argument('--grip-below-top', type=float, default=0.012, help='finger-pad centre below the cap top (m)')
    ap.add_argument('--box-h', type=float, default=0.075)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    if not a.auto and (a.cap is None or a.box is None or a.table is None):
        sys.exit('give --auto or all of --cap --box --table')

    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2

    with Connection() as router:
        base, cyc = BaseClient(router), BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation(); mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)
        state = Base_pb2.ArmState.Name(base.GetArmState().active_state)
        print('arm state:', state)
        if state != 'ARMSTATE_SERVOING_READY':
            sys.exit('arm not ready (clear faults / release the e-stop first)')
        rec = None
        if a.auto or a.record:
            from recorder import Recorder
            rec = Recorder(); rec.start()
        meta = dict(note=a.note, started=time.strftime('%Y-%m-%d %H:%M:%S'), result='aborted', speed_m_s=V_LIN)
        try:
            run(a, base, cyc, rec, meta, Base_pb2)
        finally:
            if rec is not None:
                if a.record:
                    if meta['result'] != 'aborted':
                        r = input('\nWas the bottle placed INSIDE the box? y / n > ').strip().lower()
                        meta['result'] = 'success' if r.startswith('y') else 'failure'
                        if meta['result'] == 'failure':
                            meta['failure_note'] = input('what went wrong (free text) > ').strip()
                    out = rec.save(meta)
                    print('episode saved to', out, '| recorder error:', rec.err)
                rec.close()


def run(a, base, cyc, rec, meta, Base_pb2):
        if rec is not None and a.record:
            rec.start_logging()
        if a.auto:
            a.cap, a.box, a.table, det = auto_detect(base, cyc, rec)
            meta['detection'] = det
        meta.update(cap=list(map(float, a.cap)), box=list(map(float, a.box)), table=float(a.table))

        cx, cy, top = a.cap; bx, by = a.box; tz = a.table
        g = top - a.grip_below_top                  # tool z at the grasp
        hang = g - tz                               # tool z minus bottle bottom while held
        z_carry = tz + a.box_h + 0.05 + hang        # bottle bottom 5 cm above the rim
        z_rim = tz + a.box_h + 0.02 + hang          # bottle bottom 2 cm above the rim (check alignment here)
        z_place = tz + 0.018 + hang                 # bottle bottom ~1.5 cm above the box floor
        print(f'grasp tool z {g:.3f}, bottle hangs {hang:.3f} below the tool; carry z {z_carry:.3f}, rim check z {z_rim:.3f}, place z {z_place:.3f}')
        move = make_move(base, cyc, top - 0.03, 0.55)

        def gripper(v):
            cmd = Base_pb2.GripperCommand(); cmd.mode = Base_pb2.GRIPPER_POSITION
            f = cmd.gripper.finger.add(); f.finger_identifier = 1; f.value = float(v)
            base.SendGripperCommand(cmd)
            time.sleep(1.5)
            return cyc.RefreshFeedback().interconnect.gripper_feedback.motor[0].position

        def step(name, target=None, grip=None):
            t = None if target is None else np.array(target, float)
            if rec is not None:
                rec.set_step(name, t)
            while True:
                now = tool_pose(cyc)
                desc = f'gripper -> {grip}' if grip is not None else f'move to ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})  [from ({now[0]:.3f}, {now[1]:.3f}, {now[2]:.3f})]'
                r = ask(f'\n[{name}] {desc}\n  Enter (empty) = go | a dx dy dz (cm) | s = skip | q = quit > ')
                if r == 'quit':
                    sys.exit('stopped by operator')
                if r == 'skip':
                    return t
                if isinstance(r, tuple):
                    if t is not None:
                        t = t + r[1]
                        meta.setdefault('adjustments', []).append([name, (r[1] * 100).round(2).tolist()])
                        if rec is not None:
                            rec.set_step(name, t)
                    continue
                if grip is not None:
                    pos = gripper(grip); print(f'   gripper position now {pos:.1f} %')
                    return t
                if not move(t):
                    if input('   motion did not finish; Enter (empty) to continue, anything else stops > ').strip() != '':
                        sys.exit('stopped')
                return t

        step('1 above the cap', [cx, cy, top + 0.10])
        step('2 open gripper', grip=0.0)
        t = step('3 down to 4 cm above the grasp height', [cx, cy, g + 0.04])
        t = step('4 down to 2 cm above the grasp height (look: are the fingers around the cap?)', t - [0, 0, 0.02])
        t = step('5 down to the grasp height (adjust with  a 0 0 -1  if the pads are still above the cap)', t - [0, 0, 0.02])
        step('6 close gripper', grip=1.0)
        pos = cyc.RefreshFeedback().interconnect.gripper_feedback.motor[0].position
        if pos > 95:
            print(f'WARNING: gripper closed to {pos:.0f} % -> probably missed the cap')
        here = tool_pose(cyc)[:3]
        step('7 lift (bottle bottom 5 cm above the box rim)', [here[0], here[1], z_carry])
        step('8 over the box', [bx, by, z_carry])
        t = step('9 lower to 2 cm above the rim -- CHECK ALIGNMENT, adjust with  a dx dy 0  (0.5 cm steps)', [bx, by, z_rim])
        step('10 lower into the box', [t[0], t[1], z_place])
        step('11 open gripper', grip=0.0)
        step('12 lift away', [t[0], t[1], z_carry])
        meta['result'] = 'completed'
        meta['gripper_closed_pct'] = float(pos)
        print('\ndone')


if __name__ == '__main__':
    main()
