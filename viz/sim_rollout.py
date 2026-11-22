"""Simulation rollouts of A / B (same scene per seed) rendered for the project page.

    ../kinova_bottle_sim/.venv/bin/python sim_rollout.py --seeds 100003 100011        # -> out/sim_<seed>.mp4 + gif
Each frame: third-person view (floating gripper, no arm model) + wrist camera with the policy's image (B: anchor) and
HUD, for A dense | B fixed K=20 | B adaptive 8 cm side by side.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / 'kinova_bottle_sim'
sys.path.insert(0, str(SIM / 'scripts'))
from expert import sample_scene, start_pose, success  # noqa: E402
from policy_runtime import Controller  # noqa: E402
from scene import R_TOOL, Sim, mat2quat  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
CFGS = [('A  dense', 'runs/A5/last.pt', 'dense'), ('B  fixed K=20', 'runs/B5/last.pt', 'fixed'), ('B  adaptive 8 cm', 'runs/B5/last.pt', 'adaptive')]


def t(img, s, xy, sc=0.55, col=(255, 255, 255), th=1, bg=(0, 0, 0)):
    (w, h), b = cv2.getTextSize(s, FONT, sc, th)
    cv2.rectangle(img, (xy[0] - 3, xy[1] - h - 4), (xy[0] + w + 3, xy[1] + b + 1), bg, -1)
    cv2.putText(img, s, xy, FONT, sc, col, th, cv2.LINE_AA)


def rollout(seed, name, ck, rule):
    sim = Sim(256, 144, look='real'); ctl = Controller(str(SIM / ck), rule, 20, 0.08)
    third = mujoco.Renderer(sim.m, 360, 640); cam = mujoco.MjvCamera()
    rng = np.random.default_rng(seed)
    b, x, yaw = sample_scene(rng); xyz0, tilt = start_pose(rng, b, x)
    sim.reset(xyz0, b, x, yaw, tool_euler_deg=tilt); sim.randomize(rng); sim.d.mocap_quat[sim.mocap] = mat2quat(R_TOOL)
    mid = (b + x) / 2; cam.lookat[:] = [mid[0], mid[1], 0.08]; cam.distance = 0.75; cam.azimuth = 160; cam.elevation = -35
    frames = []; closed_steps = 0; release_t = None; cmd = sim.tool(); last_enc = -99
    for k in range(400):
        wrist = sim.render(); img = cv2.resize(wrist, (128, 72), interpolation=cv2.INTER_AREA)
        o = ctl.step(img, sim.tool(), sim.gripper_open_frac())
        if o['encoded']:
            last_enc = k
        hl = (sim.m.vis.headlight.ambient.copy(), sim.m.vis.headlight.diffuse.copy())     # brighter light for the
        sim.m.vis.headlight.ambient[:] = 0.45; sim.m.vis.headlight.diffuse[:] = 0.6               # third-person view only
        third.update_scene(sim.d, camera=cam); tp = cv2.cvtColor(third.render(), cv2.COLOR_RGB2BGR)
        sim.m.vis.headlight.ambient[:], sim.m.vis.headlight.diffuse[:] = hl
        wv = cv2.resize(cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR), (640, 360))
        if ctl.feat is not None:
            pass
        frames.append((tp, wv, k, o['anchor_age'], k - last_enc < 3, ctl.n_enc, o['grip_cmd'], o['delta'] * 100))
        cmd = o['target'].copy(); cmd[2] = max(cmd[2], -0.02); sim.step(cmd, o['grip_cmd'])
        closed_steps += o['grip_cmd'] > 0
        if release_t is None and closed_steps > 15 and o['grip_cmd'] == 0:
            release_t = k
        if release_t is not None and k - release_t >= 20:
            break
    for _ in range(10):
        sim.step(cmd, 0.0)
    ok, _ = success(sim)
    return frames, ok


def compose(cols, names, oks, seed):
    n = max(len(c) for c in cols); out = []
    for i in range(n):
        tiles = []
        for c, nm, ok in zip(cols, names, oks):
            tp, wv, k, age, flash, nenc, g, dl = c[min(i, len(c) - 1)]
            tp = tp.copy(); wv = wv.copy()
            t(tp, f'{nm}   step {k}', (10, 26), 0.6)
            if i >= len(c) - 1:
                t(tp, 'SUCCESS' if ok else 'FAIL', (10, 340), 0.8, (255, 255, 255), 2, (40, 150, 40) if ok else (40, 40, 200))
            t(wv, 'wrist camera (policy input)', (10, 24), 0.5)
            t(wv, f'encodes {nenc}/{k + 1}   anchor age {age}   gripper {"closed" if g else "open"}', (10, 344), 0.5)
            if 'B' in nm:
                t(wv, f'delta ({dl[0]:+.1f},{dl[1]:+.1f},{dl[2]:+.1f}) cm', (10, 316), 0.5, (0, 165, 255))
            if flash:
                cv2.rectangle(wv, (0, 0), (639, 359), (60, 60, 255), 6); t(wv, 'ENCODE', (530, 26), 0.6, (255, 255, 255), 2, (60, 60, 255))
            tiles.append(np.vstack([tp, wv]))
        out.append(np.hstack(tiles))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, nargs='+', default=[100003]); a = ap.parse_args()
    (HERE / 'out').mkdir(exist_ok=True)
    for seed in a.seeds:
        cols, oks = [], []
        for nm, ck, rule in CFGS:
            fr, ok = rollout(seed, nm, ck, rule); cols.append(fr); oks.append(ok); print(seed, nm, 'success' if ok else 'FAIL', len(fr), 'steps', flush=True)
        frames = compose(cols, [c[0] for c in CFGS], oks, seed)
        mp4 = HERE / 'out' / f'sim_{seed}.mp4'; H, W = frames[0].shape[:2]
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '10', '-i', '-',
                              '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(mp4)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close(); p.wait()
        gif = HERE.parent / 'kinova_report/media/gif' / f'sim_{seed}_3x.gif'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-vf',
                        'setpts=PTS/3,fps=10,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
                        str(gif)])
        print('->', mp4, gif)


if __name__ == '__main__':
    main()
