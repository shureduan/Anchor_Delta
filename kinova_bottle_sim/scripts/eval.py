"""Closed-loop evaluation in sim on unseen randomised scenes.

    python scripts/eval.py --ckpt runs/A/last.pt --n 50
    python scripts/eval.py --ckpt runs/B/last.pt --rule adaptive --n 50
    python scripts/eval.py --ckpt runs/B/last.pt --rule fixed --K 20 --n 50

Every 0.1 s the policy outputs a chunk; the first target (tool + offset) and gripper command are executed.
The episode ends 2 s after the first release that follows >= 1.5 s of holding (the placement attempt), or at max-steps.
A encodes the current frame every step. B encodes a frame only at anchors (rule from proprio only) and reuses the
cached feature otherwise. Reports success, image encodings per step, and measured policy time per step (MPS).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from expert import sample_scene, start_pose, success  # noqa: E402
from policy import POS_SCALE, Policy, proprio  # noqa: E402
from scene import R_TOOL, Sim, mat2quat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rule', default='dense', choices=['dense', 'fixed', 'adaptive'])
    ap.add_argument('--K', type=int, default=20)
    ap.add_argument('--move', type=float, default=0.08)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed0', type=int, default=100000)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--video', type=int, default=0, help='save a montage of the first N episodes')
    ap.add_argument('--out', default=None)
    ap.add_argument('--look', default='v2')
    ap.add_argument('--trace', type=int, default=0, help='print state every N steps')
    ap.add_argument('--ensemble', type=int, default=1, help='temporal ensembling over overlapping chunks (ACT-style, uniform)')
    a = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ck = torch.load(ROOT / a.ckpt, map_location='cpu')
    flip = ck.get('flip', False)
    pol = Policy(ck['mode'], pretrained=False, flip=flip); pol.load_state_dict(ck['state']); pol.to(dev).eval()
    mode = ck['mode']
    if mode == 'A' and a.rule != 'dense':
        sys.exit('policy A is dense by definition')
    sim = Sim(256, 144, look=a.look)
    res = []; t_enc, t_head = [], []
    for ep in range(a.n):
        rng = np.random.default_rng(a.seed0 + ep)
        b, x, yaw = sample_scene(rng)
        xyz0, tilt = start_pose(rng, b, x)
        sim.reset(xyz0, b, x, yaw, tool_euler_deg=tilt); sim.randomize(rng)
        sim.d.mocap_quat[sim.mocap] = mat2quat(R_TOOL)
        feat = None; anc_t = 0; anc_tool = None; anc_grip = 0.0; n_enc = 0; prev_closed = False
        frames = []; hist = []; last_flip = 0                  # (t_pred, absolute targets (H,3), grip logits (H,))
        cmd = sim.tool(); closed_steps = 0; release_t = None; steps_run = a.max_steps
        for t in range(a.max_steps):
            tool = sim.tool().astype(np.float32); grip = np.float32(sim.gripper_open_frac())
            closed = grip > 0.2
            if t > 0 and closed != prev_closed:
                last_flip = t
            fl = t - last_flip if flip else None
            if mode == 'A' or a.rule == 'dense':
                need = True
            elif a.rule == 'fixed':
                need = t % a.K == 0
            else:
                need = t == 0 or closed != prev_closed or np.linalg.norm(tool - anc_tool) > a.move
            prev_closed = closed
            if need:
                img = cv2.resize(sim.render(), (128, 72), interpolation=cv2.INTER_AREA)
                if ep < a.video:
                    frames.append(img)
                t0 = time.perf_counter()
                with torch.no_grad():
                    feat = pol.encode(torch.from_numpy(img[None]).to(dev))
                if dev == 'mps':
                    torch.mps.synchronize()
                t_enc.append(time.perf_counter() - t0)
                n_enc += 1; anc_t = t; anc_tool = tool.copy(); anc_grip = np.float32(closed)
            if mode == 'B':
                pr = proprio(tool, grip, anc_tool, anc_grip, t - anc_t, flip=fl)
            else:
                pr = proprio(tool, grip, flip=fl)
            t0 = time.perf_counter()
            with torch.no_grad():
                po, gl = pol(feat, torch.from_numpy(pr[None]).to(dev))
            po = po[0].cpu().numpy(); gl = gl[0].cpu().numpy()
            t_head.append(time.perf_counter() - t0)
            if a.ensemble:
                # a single frame cannot tell how long the robot has been waiting at the grasp pose, so the
                # first-step gripper logit stays just below 0 forever; averaging what earlier chunks predicted for
                # this step gives the policy a notion of elapsed time (ACT temporal ensembling)
                hist.append((t, tool + po * POS_SCALE, gl)); hist = [h for h in hist if t - h[0] < len(po)]
                cmd = np.mean([h[1][t - h[0]] for h in hist], 0)
                g_cmd = float(np.mean([h[2][t - h[0]] for h in hist]) > 0)
            else:
                cmd = tool + po[0] * POS_SCALE
                g_cmd = float(gl[0] > 0)
            cmd[2] = max(cmd[2], -0.02)
            sim.step(cmd, g_cmd)
            if a.trace and t % a.trace == 0:
                print(t, 'tool', tool.round(3), 'grip %.2f' % grip, 'g_cmd', g_cmd, 'since', t - anc_t, 'bottle', sim.body_pose('bottle')[0].round(3), flush=True)
            closed_steps = closed_steps + 1 if g_cmd > 0 else closed_steps
            if release_t is None and closed_steps > 15 and g_cmd == 0:
                release_t = t                      # first release after holding: the placement attempt
            if release_t is not None and t - release_t >= 20:
                steps_run = t + 1
                break
        for _ in range(10):
            sim.step(cmd, 0.0)
        ok, info = success(sim)
        res.append(dict(seed=a.seed0 + ep, success=ok, encodes=n_enc, steps=steps_run, released=release_t is not None, info=info))
        print(f'ep {ep}: {"success" if ok else "FAIL"}  encodes {n_enc}/{steps_run}  released {release_t}  {info}', flush=True)
        if ep < a.video and frames:
            idx = np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)
            tiles = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR) for i in idx] + [np.zeros_like(frames[0])] * (12 - len(idx))
            cv2.imwrite(str(ROOT / 'out' / f'eval_{mode}_{a.rule}_{ep}.jpg'), np.concatenate([np.concatenate(tiles[k:k + 6], 1) for k in (0, 6)], 0))
    k = sum(r['success'] for r in res)
    rep = dict(ckpt=a.ckpt, mode=mode, ensemble=a.ensemble, rule=a.rule, K=a.K if a.rule == 'fixed' else None, move=a.move if a.rule == 'adaptive' else None,
               n=a.n, success=k / a.n, wilson95=wilson(k, a.n), encodes_per_step=float(np.mean([r['encodes'] / r['steps'] for r in res])),
               encode_ms=float(np.median(t_enc) * 1e3), head_ms=float(np.median(t_head) * 1e3),
               policy_ms_per_step=float((np.sum(t_enc) + np.sum(t_head)) / sum(r['steps'] for r in res) * 1e3),
               mean_steps=float(np.mean([r['steps'] for r in res])), episodes=res)
    out = ROOT / (a.out or f'out/eval_{mode}_{a.rule}{a.K if a.rule == "fixed" else ""}.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'episodes'}, indent=1))


if __name__ == '__main__':
    main()
