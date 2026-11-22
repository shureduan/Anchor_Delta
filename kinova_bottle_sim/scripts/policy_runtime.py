"""Step-by-step controller for the ResNet policies (A dense / B anchors), shared by the sim check and the real robot.

    ctl = Controller('runs/B2/last.pt', rule='fixed', K=20)
    ctl.reset()
    out = ctl.step(img_rgb_128x72, tool_xyz, grip_sim)   # -> dict(target=xyz, grip_cmd=0/1, encoded=bool, ...)

Same logic as eval.py: closed = gripper reading > 0.2; steps since the last gripper change; anchors from proprio only
(B); temporal ensembling over overlapping chunks. Inputs must be in sim conventions: tool position in the robot base
frame (m), gripper reading 0 (open) .. 1 (closed) with 0.411 = holding the 36 mm cap.

    python scripts/policy_runtime.py --ckpt runs/B2/last.pt --rule fixed --K 20 --n 10   # sim closed-loop check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from policy import POS_SCALE, Policy, proprio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Controller:
    REFRACTORY = 15            # grip_commit: no gripper change within 15 steps (1.5 s) of the previous one
    def __init__(self, ckpt, rule='dense', K=20, move=0.08, device=None, grip_commit=False):
        self.dev = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        ck = torch.load(ROOT / ckpt, map_location='cpu')
        self.mode, self.flip = ck['mode'], ck.get('flip', False)
        self.pol = Policy(self.mode, pretrained=False, flip=self.flip); self.pol.load_state_dict(ck['state']); self.pol.to(self.dev).eval()
        if self.mode == 'A' and rule != 'dense':
            raise ValueError('policy A is dense by definition')
        self.rule, self.K, self.move = rule, K, move
        # grip_commit (2026-11-22, real L3 deadlock): a stationary policy keeps predicting "change the gripper in ~6 steps"
        # forever (no elapsed-time signal once flip_since saturates), and the uniform ensemble can settle just below 0.
        # Commit rule: remember the earliest step at which a chunk planned the change; execute it when that step comes
        # if the current chunk still plans the change and the arm is (nearly) stationary. OR-ed with the ensemble rule.
        self.grip_commit = grip_commit
        self.reset()

    def reset(self):
        self.t = 0; self.feat = None; self.anc_t = 0; self.anc_tool = None; self.anc_grip = np.float32(0)
        self.prev_closed = False; self.last_flip = 0; self.hist = []; self.n_enc = 0
        self.g_state = 0.0; self.sched = None; self.commit_fired = 0; self.tools = []; self.last_change = -10 ** 6

    def step(self, img, tool, grip):
        t = self.t; tool = np.asarray(tool, np.float32); grip = np.float32(grip)
        closed = bool(grip > 0.2)
        if t > 0 and closed != self.prev_closed:
            self.last_flip = t
        if self.mode == 'A' or self.rule == 'dense':
            need = True
        elif self.rule == 'fixed':
            need = t % self.K == 0
        else:
            need = t == 0 or closed != self.prev_closed or np.linalg.norm(tool - self.anc_tool) > self.move
        self.prev_closed = closed
        fl = t - self.last_flip if self.flip else None
        t0 = time.perf_counter()
        with torch.no_grad():
            if need:
                self.feat = self.pol.encode(torch.from_numpy(np.ascontiguousarray(img)[None]).to(self.dev))
                self.n_enc += 1; self.anc_t, self.anc_tool, self.anc_grip = t, tool.copy(), np.float32(closed)
            pr = proprio(tool, grip, self.anc_tool, self.anc_grip, t - self.anc_t, flip=fl) if self.mode == 'B' else proprio(tool, grip, flip=fl)
            po, gl = self.pol(self.feat, torch.from_numpy(pr[None]).to(self.dev))
        po = po[0].cpu().numpy(); gl = gl[0].cpu().numpy()
        ms = (time.perf_counter() - t0) * 1e3
        self.hist.append((t, tool + po * POS_SCALE, gl)); self.hist = [h for h in self.hist if t - h[0] < len(po)]
        target = np.mean([h[1][t - h[0]] for h in self.hist], 0)
        g = float(np.mean([h[2][t - h[0]] for h in self.hist]) > 0)
        fired = False
        self.tools = (self.tools + [tool.copy()])[-4:]
        # stationary = the MEASURED tool barely moved over the last 3 steps (the commanded target may sit below a
        # workspace clamp, e.g. z_min on the real robot, so target - tool is not a reliable test)
        still = len(self.tools) == 4 and max(np.linalg.norm(self.tools[-1] - p) for p in self.tools[:-1]) < 0.002
        if self.grip_commit:
            want = 1.0 - self.g_state                                   # the other gripper state
            side = (gl > 0) == bool(want)
            # planned change = from some index k to the END of the chunk the logits stay on the new side
            ks = [k for k in range(len(side)) if side[k:].all()]
            if ks:
                cand = t + int(ks[0]); self.sched = cand if self.sched is None else min(self.sched, cand)
            else:
                self.sched = None                                       # current chunk no longer plans the change
            if g == self.g_state and self.sched is not None and t >= self.sched and still:
                g = want; fired = True; self.commit_fired += 1
        if self.grip_commit and g != self.g_state and t - self.last_change < self.REFRACTORY:
            g = self.g_state; fired = False                             # hold the new gripper state (expert holds >= 1 s)
        if g != self.g_state:
            self.g_state = g; self.sched = None; self.last_change = t
        self.t += 1
        return dict(target=target, grip_cmd=g, encoded=need, policy_ms=ms, step=t, anchor_age=t - self.anc_t,
                    # for visualisation / analysis only
                    chunk=tool + po * POS_SCALE, chunk_grip_logit=gl, anchor_step=self.anc_t,
                    anchor_tool=self.anc_tool.copy(), delta=tool - self.anc_tool, flip_since=fl, n_enc=self.n_enc,
                    commit_fired=fired, grip_sched=self.sched)


def sim_check():
    """Closed loop in sim through the Controller (must match eval.py)."""
    import cv2
    from expert import sample_scene, start_pose, success
    from scene import R_TOOL, Sim, mat2quat
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--rule', default='dense'); ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--grip-commit', action='store_true')
    ap.add_argument('--move', type=float, default=0.08); ap.add_argument('--n', type=int, default=10); ap.add_argument('--seed0', type=int, default=100000); ap.add_argument('--look', default='v2')
    a = ap.parse_args()
    ctl = Controller(a.ckpt, a.rule, a.K, a.move, grip_commit=a.grip_commit); sim = Sim(256, 144, look=a.look); ok_n = 0
    for ep in range(a.n):
        rng = np.random.default_rng(a.seed0 + ep)
        b, x, yaw = sample_scene(rng); xyz0, tilt = start_pose(rng, b, x)
        sim.reset(xyz0, b, x, yaw, tool_euler_deg=tilt); sim.randomize(rng); sim.d.mocap_quat[sim.mocap] = mat2quat(R_TOOL)
        ctl.reset(); closed_steps = 0; release_t = None; cmd = sim.tool()
        for t in range(400):
            img = cv2.resize(sim.render(), (128, 72), interpolation=cv2.INTER_AREA)
            o = ctl.step(img, sim.tool(), sim.gripper_open_frac())
            cmd = o['target'].copy(); cmd[2] = max(cmd[2], -0.02); sim.step(cmd, o['grip_cmd'])
            closed_steps += o['grip_cmd'] > 0
            if release_t is None and closed_steps > 15 and o['grip_cmd'] == 0:
                release_t = t
            if release_t is not None and t - release_t >= 20:
                break
        for _ in range(10):
            sim.step(cmd, 0.0)
        ok, info = success(sim); ok_n += ok
        print(f'ep {ep}: {"success" if ok else "FAIL"}  encodes {ctl.n_enc}/{t + 1}  commit_fired {ctl.commit_fired}', flush=True)
    print(f'{ok_n}/{a.n} success')


if __name__ == '__main__':
    sim_check()
