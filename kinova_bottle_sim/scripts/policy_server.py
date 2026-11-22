"""Policy server for the real robot: runs the Controller (PyTorch) in this venv; the robot client
(delta_wristcam/real/scripts/21_policy_run.py, delta-kinova env, Kortex) talks to it over a local socket, so the
delta-kinova environment needs no PyTorch.

    .venv/bin/python scripts/policy_server.py                     # listens on 127.0.0.1:6001
    .venv/bin/python scripts/policy_server.py --A runs/A3/last.pt --B runs/B3/last.pt   # real-matched look models

Messages (plain Python types only, so numpy versions do not matter):
    {'cmd': 'reset', 'policy': 'A'|'B', 'rule': 'dense'|'fixed'|'adaptive', 'K': 20, 'move': 0.08}
        -> {'ok': True, 'mode': ...}
    {'cmd': 'step', 'jpg': <BGR JPEG bytes, any size, 16:9>, 'tool': [x, y, z], 'grip_pct': <Kortex gripper %>}
        -> {'target': [x, y, z], 'grip_cmd': 0|1, 'encoded': bool, 'policy_ms': float, 'step': int, 'grip_sim': float}
    {'cmd': 'bye'}
The image is resized to 128x72 (INTER_AREA) and the gripper % is mapped to the sim reading (real_offline.py).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from policy_runtime import Controller  # noqa: E402
from real_offline import real_to_sim_grip  # noqa: E402

CKPT = {'A': 'runs/A2/last.pt', 'B': 'runs/B2/last.pt'}
AUTHKEY = b'delta-kinova'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=6001)
    ap.add_argument('--A', default=CKPT['A']); ap.add_argument('--B', default=CKPT['B'])
    ap.add_argument('--grip-commit', action='store_true', help='gripper commit rule (policy_runtime.Controller), same for A and B')
    a = ap.parse_args()
    CKPT.update(A=a.A, B=a.B); print('checkpoints', CKPT, flush=True)
    ctls = {}
    with Listener(('127.0.0.1', a.port), authkey=AUTHKEY) as lis:
        print(f'policy server on 127.0.0.1:{a.port}', flush=True)
        while True:
            with lis.accept() as conn:
                print('client connected', flush=True); ctl = None
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg['cmd'] == 'reset':
                        key = (msg['policy'], msg.get('rule', 'dense'), msg.get('K', 20), msg.get('move', 0.08))
                        if key not in ctls:
                            ctls[key] = Controller(CKPT[msg['policy']], key[1], key[2], key[3], grip_commit=a.grip_commit)
                        ctl = ctls[key]; ctl.reset()
                        print('reset', key, flush=True)
                        conn.send({'ok': True, 'mode': ctl.mode, 'ckpt': CKPT[msg['policy']], 'grip_commit': bool(a.grip_commit)})
                    elif msg['cmd'] == 'step':
                        bgr = cv2.imdecode(np.frombuffer(msg['jpg'], np.uint8), cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(cv2.resize(bgr, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                        g = float(real_to_sim_grip(msg['grip_pct']))
                        o = ctl.step(img, np.asarray(msg['tool'][:3], np.float32), g)
                        conn.send({'target': [float(v) for v in o['target']], 'grip_cmd': int(o['grip_cmd']), 'encoded': bool(o['encoded']),
                                   'policy_ms': float(o['policy_ms']), 'step': int(o['step']), 'grip_sim': g, 'anchor_age': int(o['anchor_age']),
                                   'chunk': np.asarray(o['chunk'], float).round(5).tolist(), 'chunk_grip_logit': np.asarray(o['chunk_grip_logit'], float).round(3).tolist(),
                                   'anchor_step': int(o['anchor_step']), 'anchor_tool': [float(v) for v in o['anchor_tool']],
                                   'delta': [float(v) for v in o['delta']], 'flip_since': None if o['flip_since'] is None else int(o['flip_since']),
                                   'n_enc': int(o['n_enc']), 'commit_fired': bool(o['commit_fired'])})
                    elif msg['cmd'] == 'bye':
                        break
                print('client gone', flush=True)


if __name__ == '__main__':
    main()
