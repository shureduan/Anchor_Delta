"""Two focused animations from the real L3 v5 runs (one frame per 10 Hz policy step):

  delta_rgbd_<video>.gif   wrist RGB + depth now | the cached anchor image | delta = tool_now - tool_anchor from FK
  anchor_rules.gif         fixed K=20 (b_l3) vs adaptive (c_l3): why and when a new anchor is encoded

    ../kinova_bottle_sim/.venv/bin/python focus_gifs.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from render_run import depth_vis, load_run

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'kinova_report/media/gif'
FONT = cv2.FONT_HERSHEY_SIMPLEX
ORA, RED, WHITE, GREY = (0, 165, 255), (60, 60, 255), (255, 255, 255), (150, 150, 150)


def t(img, s, xy, sc=0.55, col=WHITE, th=1, bg=None):
    if bg is not None:
        (w, h), b = cv2.getTextSize(s, FONT, sc, th); cv2.rectangle(img, (xy[0] - 3, xy[1] - h - 4), (xy[0] + w + 3, xy[1] + b + 1), bg, -1)
    cv2.putText(img, s, xy, FONT, sc, col, th, cv2.LINE_AA)


def rec_i(S, k):
    return int(np.clip(np.searchsorted(S['t_rec'], S['t_pol'][k], 'right') - 1, 0, len(S['t_rec']) - 1))


def write(frames, name, fps=10, speed=2, width=960):
    H, W = frames[0].shape[:2]; H2, W2 = H + H % 2, W + W % 2      # libx264 needs even sizes
    frames = [cv2.copyMakeBorder(f, 0, H2 - H, 0, W2 - W, cv2.BORDER_CONSTANT, value=(22, 22, 22)) for f in frames]
    mp4 = HERE / 'out' / f'{name}.mp4'; H, W = H2, W2
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', str(mp4)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp4), '-vf',
                    f'setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=160:stats_mode=diff[p];'
                    '[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle', str(OUT / f'{name}.gif')])
    print('->', OUT / f'{name}.gif')


def delta_panel(S, k, w=480, h=540):
    img = np.full((h, w, 3), 22, np.uint8)
    a = S['anc'][k]; d = S['delta'][k] * 100; pa, pn = S['tool'][a], S['tool'][k]
    t(img, 'delta from robot kinematics (no image)', (14, 30), 0.6, ORA, 1)
    t(img, f'tool now    (FK) = ({pn[0]:.3f}, {pn[1]:.3f}, {pn[2]:.3f}) m', (14, 70), 0.5)
    t(img, f'tool anchor (FK) = ({pa[0]:.3f}, {pa[1]:.3f}, {pa[2]:.3f}) m', (14, 96), 0.5, ORA)
    t(img, f'delta = now - anchor = ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) cm', (14, 128), 0.55, WHITE, 1)
    # bars
    for j, (lab, col) in enumerate([('dx', (90, 90, 255)), ('dy', (90, 220, 90)), ('dz', (255, 160, 90))]):
        y = 180 + j * 52; x0 = 240; sc = 14.0     # px per cm
        cv2.line(img, (x0, y - 18), (x0, y + 18), GREY, 1)
        cv2.rectangle(img, (x0, y - 12), (int(x0 + np.clip(d[j], -16, 16) * sc), y + 12), col, -1)
        t(img, f'{lab} {d[j]:+5.1f} cm', (24, y + 6), 0.55, col)
    t(img, '-8 cm', (240 - 8 * 14 - 18, 350), 0.4, GREY); t(img, '+8 cm', (240 + 8 * 14 - 18, 350), 0.4, GREY)
    n = np.linalg.norm(d)
    t(img, f'|delta| = {n:4.1f} cm', (24, 395), 0.6, WHITE, 1)
    t(img, f'anchor age = {S["age"][k]} steps ({S["age"][k] / 10:.1f} s)', (24, 425), 0.55)
    t(img, f'gripper at anchor: {"closed" if S["L"][a]["grip_sim"] > 0.2 else "open"}   now: {"closed" if S["L"][k]["grip_sim"] > 0.2 else "open"}',
      (24, 455), 0.5)
    t(img, 'policy input = [cached anchor feature (512),', (24, 495), 0.45, GREY)
    t(img, '  proprio now (5), delta (5)]  -> 10-step chunk', (24, 518), 0.45, GREY)
    return img


def delta_rgbd(video_stem):
    man = json.load(open(HERE / 'sync_manifest.json')); S = load_run(man[video_stem + '.MOV']['run'])
    frames = []; last_enc = -9
    for k in range(len(S['L'])):
        i = rec_i(S, k)
        rgb = cv2.resize(cv2.imdecode(np.frombuffer(S['jpg'][i], np.uint8), 1), (480, 270))
        dep = cv2.resize(depth_vis(S['depth'][i]), (480, 270), interpolation=cv2.INTER_NEAREST)
        if S['enc'][k]:
            last_enc = k
        anc = cv2.resize(cv2.imdecode(np.frombuffer(S['pjpg'][S['anc'][k]], np.uint8), 1), (480, 270))
        flash = k - last_enc < 3
        cv2.rectangle(anc, (0, 0), (479, 269), RED if flash else ORA, 5)
        t(rgb, f'wrist RGB now  (step {k})', (8, 24), 0.55, bg=(0, 0, 0)); t(dep, 'wrist depth now', (8, 24), 0.55, bg=(0, 0, 0))
        t(anc, f'cached anchor image (step {S["anc"][k]})', (8, 24), 0.55, ORA, bg=(0, 0, 0))
        if flash:
            t(anc, 'ENCODE  (new anchor)', (8, 258), 0.6, WHITE, 2, RED)
        else:
            t(anc, 'not re-encoded: reused', (8, 258), 0.55, WHITE, 1, (0, 0, 0))
        left = np.vstack([np.hstack([rgb, dep]), np.hstack([anc, np.full((270, 480, 3), 22, np.uint8)])])
        right_info = delta_panel(S, k)
        left[270:540, 480:960] = right_info[0:270]
        frame = np.hstack([left, delta_panel(S, k)])
        frame[270:540, 480:960] = np.full((270, 480, 3), 22, np.uint8)
        # middle-bottom: encode counter
        mb = frame[270:540, 480:960]
        enc = int(S['enc'][:k + 1].sum())
        t(mb, 'image encoder calls', (16, 34), 0.6)
        cv2.rectangle(mb, (16, 60), (464, 90), (60, 60, 60), 1); cv2.rectangle(mb, (16, 60), (16 + int(448 * enc / (k + 1)), 90), RED, -1)
        t(mb, f'B: {enc} / {k + 1} steps ({100 * enc / (k + 1):.1f} %)', (16, 120), 0.55, RED)
        cv2.rectangle(mb, (16, 140), (464, 170), (110, 110, 110), -1)
        t(mb, f'A (dense): {k + 1} / {k + 1} steps (100 %)', (16, 200), 0.55, GREY)
        t(mb, 'between anchors the camera is not encoded;', (16, 236), 0.45, GREY)
        t(mb, 'the delta tells the policy how the hand moved', (16, 258), 0.45, GREY)
        frames.append(frame)
    write(frames, f'delta_rgbd_{video_stem}', speed=2, width=1100)


def rule_panel(S, k, rule, w=720, h=300):
    img = np.full((h, w, 3), 22, np.uint8)
    L = S['L']; a = S['anc'][k]; n = np.linalg.norm(S['delta'][k]) * 100
    closed_now = L[k]['grip_sim'] > 0.2; closed_anc = L[a]['grip_sim'] > 0.2
    if rule == 'fixed':
        t(img, 'rule: fixed K = 20   (re-encode when age reaches 20)', (12, 28), 0.55, ORA)
        age = S['age'][k]
        cv2.rectangle(img, (20, 60), (680, 100), (60, 60, 60), 1); cv2.rectangle(img, (20, 60), (20 + int(660 * age / 19), 100), ORA, -1)
        t(img, f'anchor age {age:2d} / 20 steps', (24, 130), 0.6)
    else:
        t(img, 'rule: adaptive  (re-encode if ANY trigger fires)', (12, 28), 0.55, ORA)
        cv2.rectangle(img, (20, 60), (680, 100), (60, 60, 60), 1)
        cv2.rectangle(img, (20, 60), (20 + int(660 * min(n, 10) / 10), 100), ORA if n < 8 else RED, -1)
        x8 = 20 + int(660 * 8 / 10); cv2.line(img, (x8, 52), (x8, 108), WHITE, 2)
        t(img, f'|delta| = {n:4.1f} cm   (threshold 8 cm)', (24, 130), 0.6)
        t(img, f'gripper (sensor): at anchor {"closed" if closed_anc else "open"} | now {"closed" if closed_now else "open"}',
          (24, 165), 0.55, RED if closed_now != closed_anc else WHITE)
    # encode log with reasons
    t(img, 'encodes so far:', (24, 205), 0.5, GREY)
    xs = [j for j in range(k + 1) if S['enc'][j]]
    for m, j in enumerate(xs[-6:]):
        t(img, f'{j}:{S["reason"][j]}', (24 + m * 115, 235), 0.42, RED)
    t(img, f'total {len(xs)} / {k + 1} steps', (24, 275), 0.55, RED)
    return img


def reasons(S, rule):
    r = {}
    for j in range(len(S['L'])):
        if not S['enc'][j]:
            continue
        if j == 0:
            r[j] = 'start'
        elif rule == 'fixed':
            r[j] = 'age=20'
        else:
            p = j - 1; a = S['anc'][p]
            if (S['L'][j]['grip_sim'] > 0.2) != (S['L'][p]['grip_sim'] > 0.2):
                r[j] = 'gripper'
            else:
                r[j] = '>8cm'
    return r


def anchor_rules():
    man = json.load(open(HERE / 'sync_manifest.json'))
    SB = load_run(man['b_l3_r1.MOV']['run']); SC = load_run(man['c_l3_r1.MOV']['run'])
    SB['reason'] = reasons(SB, 'fixed'); SC['reason'] = reasons(SC, 'adaptive')
    n = min(len(SB['L']), len(SC['L'])); frames = []; le = {'b': -9, 'c': -9}
    for k in range(n):
        cols = []
        for key, S, rule, title in [('b', SB, 'fixed', 'B  fixed K=20'), ('c', SC, 'adaptive', 'B  adaptive (gripper flip / moved > 8 cm)')]:
            if S['enc'][k]:
                le[key] = k
            i = rec_i(S, k)
            rgb = cv2.resize(cv2.imdecode(np.frombuffer(S['jpg'][i], np.uint8), 1), (360, 203))
            anc = cv2.resize(cv2.imdecode(np.frombuffer(S['pjpg'][S['anc'][k]], np.uint8), 1), (360, 203))
            flash = k - le[key] < 3
            cv2.rectangle(anc, (0, 0), (359, 202), RED if flash else ORA, 5)
            t(rgb, f'{title}', (6, 20), 0.45, bg=(0, 0, 0)); t(rgb, f'now step {k}', (6, 195), 0.45, bg=(0, 0, 0))
            t(anc, f'anchor @ {S["anc"][k]}', (6, 20), 0.45, ORA, bg=(0, 0, 0))
            if flash:
                t(anc, f'ENCODE: {S["reason"][le[key]]}', (6, 195), 0.55, WHITE, 2, RED)
            cols.append(np.vstack([np.hstack([rgb, anc]), rule_panel(S, k, rule)]))
        frames.append(np.hstack([cols[0], np.full((cols[0].shape[0], 8, 3), 60, np.uint8), cols[1]]))
    write(frames, 'anchor_rules_l3', speed=2, width=1200)


def main():
    (HERE / 'out').mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    import sys
    if 'rules' not in sys.argv:
        delta_rgbd('b_l3_r1')
    anchor_rules()


if __name__ == '__main__':
    main()
