"""Closed-loop sim evaluation of SmolVLA A / B with Real-Time Chunking and emulated inference latency.

    .venv-vla/bin/python scripts/eval_vla.py --ckpt runs/vla_A/last.pt --n 30
    .venv-vla/bin/python scripts/eval_vla.py --ckpt runs/vla_B/last.pt --rule adaptive --move 0.08 --n 30

Every --s control steps (0.1 s each) the policy plans a chunk of H_VLA actions. Real time is emulated: the simulator does
not wait for the network, but the new chunk only takes over after d = ceil(measured inference time / 0.1 s) steps; in
the meantime the robot keeps executing the previous chunk, and RTC (LeRobot RTCProcessor) inpaints the new chunk so its
first d actions agree with what is being executed. Both run [image tokens, task, state] through the VLM layers and then 10 denoising steps at every plan. A encodes
the current image (SigLIP + connector) at every plan; B only at anchors (rule from proprio only, checked at plan ticks)
and otherwise reuses the anchor's image tokens with the current state + anchor delta in the state token (vla.py).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval import wilson  # noqa: E402
from expert import sample_scene, start_pose, success  # noqa: E402
from policy import POS_SCALE  # noqa: E402
from scene import R_TOOL, Sim, mat2quat  # noqa: E402
from vla import VLA, anchor_delta, state_vec  # noqa: E402

from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rule', default='dense', choices=['dense', 'fixed', 'adaptive'])
    ap.add_argument('--K', type=int, default=20, help='fixed: re-anchor when the anchor is >= K steps old')
    ap.add_argument('--move', type=float, default=0.08)
    ap.add_argument('--s', type=int, default=10, help='plan every s steps (RTC execution horizon)')
    ap.add_argument('--rtc', type=int, default=1)
    ap.add_argument('--delay', default='measured', help="'measured' or a fixed number of steps")
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed0', type=int, default=100000)
    ap.add_argument('--max-steps', type=int, default=400)
    ap.add_argument('--trace', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    dev = 'mps'
    ck = torch.load(ROOT / a.ckpt, map_location='cpu'); mode = ck['mode']
    v = VLA(mode, dev, ckpt=ROOT / a.ckpt).eval()
    if a.rtc:
        v.pol.config.rtc_config = RTCConfig(enabled=True, execution_horizon=a.s); v.pol.init_rtc_processor()
    if mode == 'A' and a.rule != 'dense':
        sys.exit('policy A is dense by definition')
    sim = Sim(256, 144)
    res, t_img, t_vlm, t_den, delays = [], [], [], [], []
    dummy = torch.zeros(1, 8, 8, 3, dtype=torch.uint8, device=dev)
    for ep in range(a.n):
        rng = np.random.default_rng(a.seed0 + ep)
        b, x, yaw = sample_scene(rng)
        xyz0, tilt = start_pose(rng, b, x)
        sim.reset(xyz0, b, x, yaw, tool_euler_deg=tilt); sim.randomize(rng)
        sim.d.mocap_quat[sim.mocap] = mat2quat(R_TOOL)
        torch.manual_seed(a.seed0 + ep)
        queue = []                       # absolute [x, y, z, grip] for the coming steps
        feat = None; anc_t = 0; anc_tool = None; anc_grip = 0.0; n_enc = 0; n_plan = 0
        prev_closed = False; last_flip = 0; flip_since_anchor = False
        closed_steps = 0; release_t = None; steps_run = a.max_steps; cmd = sim.tool(); g_cmd = 0.0
        for t in range(a.max_steps):
            tool = sim.tool().astype(np.float32); grip = float(sim.gripper_open_frac())
            closed = grip > 0.2
            if t > 0 and closed != prev_closed:
                last_flip = t; flip_since_anchor = True
            prev_closed = closed
            if t % a.s == 0 or not queue:
                n_plan += 1
                if mode == 'A' or a.rule == 'dense':
                    need = True
                elif a.rule == 'fixed':
                    need = feat is None or t - anc_t >= a.K
                else:
                    need = feat is None or flip_since_anchor or np.linalg.norm(tool - anc_tool) > a.move
                t0 = time.perf_counter()
                if need:                                   # encode the image (SigLIP + connector): the cost B saves
                    img = torch.from_numpy(np.ascontiguousarray(sim.render())[None]).to(dev)
                    feat = v.image_features(img); torch.mps.synchronize(); n_enc += 1
                    anc_t, anc_tool, anc_grip, flip_since_anchor = t, tool.copy(), grip, False
                t1 = time.perf_counter()
                st = state_vec(tool, grip, t - last_flip)
                if mode == 'B':
                    st = np.r_[st, anchor_delta(tool, anc_tool, anc_grip, t - anc_t)]
                cache = v.prefix(dummy, torch.from_numpy(st[None]).to(dev), feat=feat); torch.mps.synchronize()
                t2 = time.perf_counter()
                left = None
                if a.rtc and queue:
                    q = np.array(queue[:20], np.float32)
                    left = torch.from_numpy(np.c_[(q[:, :3] - tool) / POS_SCALE, q[:, 3:4] * 2 - 1][None]).to(dev)
                # delay used for RTC guidance: predicted from the latency of earlier plans of the same kind
                med = lambda x, d0: float(np.median(x[-20:])) if x else d0
                est = (med(t_img, 0.2) if need else 0) + med(t_vlm, 0.06) + med(t_den, 0.25)
                d = int(a.delay) if a.delay != 'measured' else min(a.s, math.ceil(est / 0.1))
                rtc = dict(prev_chunk_left_over=left, inference_delay=d, execution_horizon=a.s) if left is not None else None
                ch = v.denoise(cache, None, rtc=rtc); torch.mps.synchronize()
                t3 = time.perf_counter()
                if need:
                    t_img.append(t1 - t0)
                t_vlm.append(t2 - t1); t_den.append(t3 - t2)
                d_real = int(a.delay) if a.delay != 'measured' else min(a.s, math.ceil((t3 - t0) / 0.1))
                delays.append(d_real)
                c = ch[0, :, :4].float().cpu().numpy()
                new = [np.r_[tool + c[k, :3] * POS_SCALE, float(c[k, 3] > 0)] for k in range(len(c))]
                keep = queue[:d_real] if queue else []
                queue = keep + new[len(keep):]
            act = queue.pop(0) if queue else np.r_[cmd, g_cmd]
            cmd = act[:3].copy(); cmd[2] = max(cmd[2], -0.02); g_cmd = float(act[3])
            sim.step(cmd, g_cmd)
            if a.trace and t % a.trace == 0:
                print(t, 'tool', tool.round(3), 'grip %.2f' % grip, 'g', g_cmd, 'bottle', sim.body_pose('bottle')[0].round(3), 'enc', n_enc, flush=True)
            closed_steps = closed_steps + 1 if g_cmd > 0 else closed_steps
            if release_t is None and closed_steps > 15 and g_cmd == 0:
                release_t = t
            if release_t is not None and t - release_t >= 20:
                steps_run = t + 1
                break
        for _ in range(10):
            sim.step(cmd, 0.0)
        ok, info = success(sim)
        res.append(dict(seed=a.seed0 + ep, success=ok, encodes=n_enc, plans=n_plan, steps=steps_run, released=release_t is not None, info=info))
        print(f'ep {ep}: {"success" if ok else "FAIL"}  encodes {n_enc}/{n_plan} plans/{steps_run} steps  released {release_t}  {info}', flush=True)
    k = sum(r['success'] for r in res)
    n_steps = sum(r['steps'] for r in res); n_plans = sum(r['plans'] for r in res)
    rep = dict(ckpt=a.ckpt, mode=mode, rule=a.rule, K=a.K if a.rule == 'fixed' else None, move=a.move if a.rule == 'adaptive' else None,
               s=a.s, rtc=a.rtc, n=a.n, success=k / a.n, wilson95=wilson(k, a.n),
               encodes_per_step=sum(r['encodes'] for r in res) / n_steps, encodes_per_plan=sum(r['encodes'] for r in res) / n_plans,
               image_enc_ms=float(np.median(t_img) * 1e3), vlm_ms=float(np.median(t_vlm) * 1e3), denoise_ms=float(np.median(t_den) * 1e3),
               plan_ms_mean=float((np.sum(t_img) + np.sum(t_vlm) + np.sum(t_den)) / n_plans * 1e3),
               compute_ms_per_step=float((np.sum(t_img) + np.sum(t_vlm) + np.sum(t_den)) / n_steps * 1e3),
               delay_steps_mean=float(np.mean(delays)), mean_steps=n_steps / a.n, episodes=res)
    out = ROOT / (a.out or f'out/vla_{mode}_{a.rule}{a.K if a.rule == "fixed" else ""}.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'episodes'}, indent=1))


if __name__ == '__main__':
    main()
