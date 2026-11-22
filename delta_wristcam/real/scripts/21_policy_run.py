"""Run a sim-trained ResNet policy (A dense or B anchors) on the real Kinova, zero-shot. Keep a hand on the e-stop.

Terminal 1 (policy, PyTorch):
    cd kinova_bottle_sim && .venv/bin/python scripts/policy_server.py
Terminal 2 (robot, Kortex):
    conda activate delta-kinova
    python real/scripts/21_policy_run.py --policy A
    python real/scripts/21_policy_run.py --policy B --rule fixed --K 20 --record --note "B K20 try 1"

Start pose: gripper pointing down, about 40-46 cm above the table, bottle and box both in view and their midpoint
near the image centre (as in the sim training starts). --start X Y Z moves there first (asks for Enter).

Every control step: fresh wrist colour frame + tool pose + gripper % -> policy server -> next tool target.
Safety:
  - the target is clipped to <= --max-step (5 mm) from the current tool position per step and moved with ReachPose at
    <= 5 cm/s, gripper kept vertical (theta = 180, 0, 97.5); targets are clamped into X [0.10, 0.55], Y [-0.25, 0.30],
    Z [--z-min, 0.55] (a warning is printed whenever clamping happens)
  - it asks for an empty Enter before every --confirm-every steps (default 10 = at most 5 cm of motion per Enter,
    approved by the operator on 2026-11-17; allowed 1..10) and ALWAYS before the gripper opens or closes; q / quit stops (the arm stays where it is); anything else does nothing and asks again
  - Ctrl-C during a motion sends Stop
The episode ends 20 steps after the first release that follows >= 15 closed steps (as in sim), or at --max-steps.
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import Connection, tool_pose  # noqa: E402

THETA = (180.0, 0.0, 97.5)
XY_LO, XY_HI = np.array([0.10, -0.25]), np.array([0.55, 0.30])
V_LIN, V_ROT = 0.05, 20.0


def ask(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans == '':
            return 'go'
        if ans in ('q', 'quit', 'exit', 'stop'):
            return 'quit'
        if ans.startswith('c '):
            try:
                n = int(ans.split()[1]); assert 1 <= n <= 10
                return ('every', n)
            except (ValueError, AssertionError, IndexError):
                print('usage: c N   (confirm every N steps, 1..10)'); continue
        print(f'not understood: "{ans}" -- nothing done. Press Enter alone to execute, q to stop.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['A', 'B'], required=True)
    ap.add_argument('--rule', default='dense', choices=['dense', 'fixed', 'adaptive'])
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--move', type=float, default=0.08)
    ap.add_argument('--start', nargs=3, type=float, default=None)
    ap.add_argument('--confirm-every', type=int, default=10, choices=range(1, 11), metavar='1..10')
    ap.add_argument('--max-step', type=float, default=0.005)
    ap.add_argument('--z-min', type=float, default=0.14, help='lowest tool z (m); the grasp is ~1.2 cm below the cap top')
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--port', type=int, default=6001)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--note', default='')
    ap.add_argument('--auto', action='store_true',
                    help='no Enter during the episode (incl. gripper changes; operator request 2026-11-22, for full videos): '
                         'type q + Enter at any time to stop at the next step; all motion limits stay')
    a = ap.parse_args()
    if a.policy == 'A' and a.rule != 'dense':
        sys.exit('policy A is dense by definition')
    lo, hi = np.r_[XY_LO, a.z_min], np.r_[XY_HI, 0.55]

    from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    from kortex_api.autogen.messages import Base_pb2
    from recorder import Recorder

    srv = Client(('127.0.0.1', a.port), authkey=b'delta-kinova')
    with Connection() as router:
        base, cyc = BaseClient(router), BaseCyclicClient(router)
        mode = Base_pb2.ServoingModeInformation(); mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)
        state = Base_pb2.ArmState.Name(base.GetArmState().active_state)
        print('arm state:', state)
        if state != 'ARMSTATE_SERVOING_READY':
            sys.exit('arm not ready (clear faults / release the e-stop first)')

        def move(xyz):
            act = Base_pb2.Action(); act.name = 'policy'
            p = act.reach_pose.target_pose
            p.x, p.y, p.z = map(float, xyz); p.theta_x, p.theta_y, p.theta_z = THETA
            act.reach_pose.constraint.speed.translation = V_LIN
            act.reach_pose.constraint.speed.orientation = V_ROT
            import threading
            done = threading.Event(); res = {}

            def cb(n):
                if n.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                    res['ev'] = Base_pb2.ActionEvent.Name(n.action_event); done.set()
            h = base.OnNotificationActionTopic(cb, Base_pb2.NotificationOptions())
            try:
                base.ExecuteAction(act)
                if not done.wait(30):
                    base.Stop(); print('timeout -> Stop'); return False
            except KeyboardInterrupt:
                base.Stop(); print('Ctrl-C -> Stop'); raise
            finally:
                base.Unsubscribe(h)
            return res.get('ev') == 'ACTION_END'

        def gripper(v):
            """Send the gripper command and WAIT until the fingers stop (arm held still, as the sim expert does while
            closing). Sending the next ReachPose right away aborts the gripper motion (2026-11-22 run 1: stalled at 5 %)."""
            cmd = Base_pb2.GripperCommand(); cmd.mode = Base_pb2.GRIPPER_POSITION
            f = cmd.gripper.finger.add(); f.finger_identifier = 1; f.value = float(v)
            base.SendGripperCommand(cmd)
            t0 = time.time(); last = grip_pct(); still = 0
            while time.time() - t0 < 2.5:
                time.sleep(0.1)
                now = grip_pct(); still = still + 1 if abs(now - last) < 0.3 else 0; last = now
                if still >= 3 and time.time() - t0 > 0.5:
                    break
            print(f'  gripper settled at {last:.1f} % after {time.time() - t0:.1f} s')

        def grip_pct():
            return float(cyc.RefreshFeedback().interconnect.gripper_feedback.motor[0].position)

        rec = Recorder(); rec.start()
        meta = dict(note=a.note, policy=a.policy, rule=a.rule, K=a.K, move=a.move, started=time.strftime('%Y-%m-%d %H:%M:%S'),
                    result='aborted', max_step=a.max_step, confirm_every=a.confirm_every)
        log = []; policy_frames = []          # the exact 256x144 frames the policy received (one per step)
        try:
            if a.start is not None:
                s = np.array(a.start)
                if np.any(s < np.r_[XY_LO, 0.25]) or np.any(s > hi):
                    sys.exit(f'start {s} outside the workspace')
                cur = tool_pose(cyc)[:3]
                if cur[2] < s[2] - 0.01:            # coming from low (e.g. after a failed run): straight up first, then across
                    up = np.array([cur[0], cur[1], s[2]])
                    if ask(f'\n[start 1/2] lift straight up to z {s[2]:.3f} (now {cur[2]:.3f})  Enter = go | q = quit > ') == 'quit':
                        sys.exit('stopped by operator')
                    move(up)
                if ask(f'\n[start] move to ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})  Enter = go | q = quit > ') == 'quit':
                    sys.exit('stopped by operator')
                move(s)
            if grip_pct() > 5:
                if ask(f'\n[start] open the gripper (now {grip_pct():.0f} %)  Enter = go | q = quit > ') == 'quit':
                    sys.exit('stopped by operator')
                gripper(0.0); time.sleep(1.5)
            srv.send({'cmd': 'reset', 'policy': a.policy, 'rule': a.rule, 'K': a.K, 'move': a.move})
            pinfo = srv.recv(); print('policy:', pinfo); meta['policy_server'] = pinfo
            if a.record:
                rec.start_logging()
            every = a.confirm_every; g_now = 0; closed_steps = 0; release_t = None; meta['result'] = 'running'
            lines = None
            if a.auto:
                if ask('\n[AUTO] run the whole episode without confirmations (q + Enter stops)  Enter = start | q = quit > ') == 'quit':
                    sys.exit('stopped by operator')
                import queue
                import threading
                lines = queue.Queue()

                def reader():
                    while True:
                        try:
                            lines.put(input())
                        except EOFError:
                            return
                threading.Thread(target=reader, daemon=True).start()
                meta['auto'] = True
            print('\a', end='', flush=True); meta['t_start'] = time.time()   # beep: sync mark for an external (phone) video
            for t in range(a.max_steps):
                if lines is not None:
                    stop = False
                    while not lines.empty():
                        ln = lines.get().strip().lower()
                        if ln in ('q', 'quit', 'stop', 'exit'):
                            stop = True
                        elif ln:
                            print(f'  (type q + Enter to stop; "{ln}" ignored)')
                    if stop:
                        print('  -> STOP by operator (arm stays here)'); meta['result'] = 'stopped'; break
                t_req = time.time()
                color, _ = rec.latest(fresh_after=t_req); t_frame = rec.last_t
                tool = tool_pose(cyc)[:3]; gp = grip_pct()
                small = cv2.resize(color, (256, 144), interpolation=cv2.INTER_AREA)
                jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(); policy_frames.append(jpg)
                srv.send({'cmd': 'step', 'jpg': jpg,
                          'tool': [float(v) for v in tool], 'grip_pct': gp})
                o = srv.recv(); t_policy = time.time()
                tgt_raw = np.array(o['target']); tgt = tgt_raw.copy(); d = tgt - tool; n = np.linalg.norm(d)
                if n > a.max_step:
                    tgt = tool + d / n * a.max_step
                clamped = np.clip(tgt, lo, hi)
                warn = '  [CLAMPED to workspace]' if np.any(np.abs(clamped - tgt) > 1e-6) else ''
                tgt = clamped
                g_change = o['grip_cmd'] != g_now
                desc = (f'step {t:3d}  tool ({tool[0]:.3f}, {tool[1]:.3f}, {tool[2]:.3f}) -> ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f})'
                        f'  move {np.linalg.norm(tgt - tool) * 1000:.1f} mm  gripper {gp:.0f} % (sim {o["grip_sim"]:.2f})'
                        f'  {"ENCODED" if o["encoded"] else "anchor age " + str(o["anchor_age"])}  {o["policy_ms"]:.1f} ms{warn}')
                print(desc)
                if not a.auto and (g_change or t % every == 0):
                    what = f'GRIPPER -> {"CLOSE" if o["grip_cmd"] else "OPEN"} and move' if g_change else f'next {every} step(s)'
                    r = ask(f'  [{what}]  Enter = go | c N = confirm every N steps | q = quit > ')
                    if r == 'quit':
                        meta['result'] = 'stopped'; break
                    if isinstance(r, tuple):
                        every = r[1]; print(f'  confirming every {every} steps (gripper changes always)')
                rec.set_step(f'policy step {t}', tgt)
                if g_change:
                    gripper(float(o['grip_cmd'])); g_now = o['grip_cmd']
                t_cmd = time.time()
                move(tgt)
                log.append(dict(t=t, tool=tool.tolist(), target=tgt.tolist(), grip_pct=gp, grip_cmd=o['grip_cmd'], encoded=o['encoded'], ms=o['policy_ms'],
                                # wall-clock sync (time.time(), same clock as the recorder's frame times 't')
                                t_req=t_req, t_frame=t_frame, t_policy=t_policy, t_cmd=t_cmd, t_done=time.time(),
                                # model outputs / algorithm state, for visualisation
                                target_raw=tgt_raw.tolist(), clamped=bool(warn), grip_change=bool(g_change), grip_sim=o.get('grip_sim'),
                                anchor_age=o.get('anchor_age'), anchor_step=o.get('anchor_step'), anchor_tool=o.get('anchor_tool'),
                                delta=o.get('delta'), flip_since=o.get('flip_since'), chunk=o.get('chunk'), chunk_grip_logit=o.get('chunk_grip_logit'),
                                n_enc=o.get('n_enc'), commit_fired=o.get('commit_fired')))
                closed_steps += g_now > 0
                if release_t is None and closed_steps > 15 and g_now == 0:
                    release_t = t; print('  -> released after holding: placing attempt; 20 more steps')
                if release_t is not None and t - release_t >= 20:
                    meta['result'] = 'done'; break
            else:
                meta['result'] = 'max_steps'
        finally:
            meta['t_end'] = time.time(); print('\a', end='', flush=True)
            srv.send({'cmd': 'bye'})
            meta.update(steps=len(log), encodes=int(sum(r['encoded'] for r in log)), policy_log=log)
            def read_line(prompt):
                if lines is None:
                    return input(prompt)
                print(prompt, end='', flush=True)
                while not lines.empty():          # drop anything typed during the run
                    lines.get()
                return lines.get()
            if a.record:
                if meta['result'] in ('done', 'max_steps', 'stopped'):
                    r = ''
                    while r not in ('y', 'n'):
                        r = read_line('\nWas the bottle placed INSIDE the box? type y or n > ').strip().lower()[:1]
                    meta['success'] = r == 'y'
                    meta['failure_note'] = '' if meta['success'] else read_line('what went wrong (free text) > ').strip()
                    if meta['success']:
                        meta['observation'] = read_line('anything notable, e.g. touched the rim (free text, Enter = nothing) > ').strip()
                out = rec.save(meta)
                if out is not None and policy_frames:
                    np.savez(out / 'policy_frames.npz', jpg=np.array(policy_frames[:len(log)] + policy_frames[len(log):], dtype=object),
                             step=np.arange(len(policy_frames)), t_frame=np.array([r['t_frame'] for r in log] + [np.nan] * (len(policy_frames) - len(log))))
                print('episode saved to', out, '| recorder error:', rec.err)
            rec.close()
            print(f'{meta["result"]}: {len(log)} steps, {meta["encodes"]} image encodings')


if __name__ == '__main__':
    main()
