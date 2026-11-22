"""Still figures for the report from the 9 grid runs (phone + wrist RGB-D + algorithm state).

    ../kinova_bottle_sim/.venv/bin/python make_figures.py      # -> ../kinova_report/media/figures/*.jpg
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from render_run import DEMOS, PHONE, depth_vis, load_run, phone_offset, txt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'kinova_report/media/figures'
GRID = [('L1', 'a_l1_r1', 'b_l1_r1', 'c_l1_r1'), ('L2', 'a_l2_r1', 'b_l2_r1', 'c_l2_r1'), ('L3', 'a_l3_r1', 'b_l3_r1', 'c_l3_r1')]
COLS = ['A  (encode every frame)', 'B  (anchor every 20 steps + delta)', 'B  (adaptive anchors + delta)']


def phases(S):
    """Policy-step indices of: start, above the cap, grasp, transfer, release, end."""
    ch = [i for i, r in enumerate(S['L']) if r.get('grip_change')]
    g, rel = ch[0], ch[-1]
    return dict(start=0, approach=max(g - 25, 0), grasp=g + 2, transfer=(g + rel) // 2, release=rel + 2, end=len(S['L']) - 1)


def phone_frame(video, t_video, w=640):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t_video:.3f}', '-i', str(video), '-frames:v', '1', '-vf', f'scale={w}:-1',
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, w, 3)


def rec_index(S, k):
    return int(np.clip(np.searchsorted(S['t_rec'], S['t_pol'][k], 'right') - 1, 0, len(S['t_rec']) - 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.load(open(HERE / 'sync_manifest.json'))
    import datetime
    cache = {}
    for _, *vids in GRID:
        for v in vids:
            S = load_run(man[v + '.MOV']['run']); video = PHONE / (v + '.MOV')
            c = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format_tags=com.apple.quicktime.creationdate', '-of', 'csv=p=0', str(video)],
                               capture_output=True, text=True).stdout.strip()
            v0, _ = phone_offset(video, S, datetime.datetime.strptime(c, '%Y-%m-%dT%H:%M:%S%z').timestamp())
            cache[v] = (S, video, v0, phases(S))
    # 1) phone 3x3 grid at the release moment + at the grasp
    for ph in ('grasp', 'transfer', 'release'):
        rows = []
        for lay, *vids in GRID:
            tiles = []
            for j, v in enumerate(vids):
                S, video, v0, P = cache[v]
                f = phone_frame(video, S['t_pol'][P[ph]] - v0, w=360).copy()     # native portrait aspect (360 x 640)
                txt(f, f'{lay} [{"v5" if lay == "L3" else "v4"}]', (8, 24), 0.55, (255, 255, 255), 1, bg=(0, 0, 0))
                txt(f, COLS[j].split('(')[0].strip() + ('' if j == 0 else (' K=20' if j == 1 else ' adaptive')), (8, 52), 0.5, (255, 255, 255), 1, bg=(0, 0, 0))
                tiles.append(f)
            rows.append(np.hstack(tiles))
        cv2.imwrite(str(OUT / f'phone_grid_{ph}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # 2) wrist RGB-D strips per run
    for lay, *vids in GRID:
        for j, v in enumerate(vids):
            S, video, v0, P = cache[v]
            top, bot = [], []
            for name in ('start', 'approach', 'grasp', 'transfer', 'release'):
                i = rec_index(S, P[name])
                rgb = cv2.resize(cv2.imdecode(np.frombuffer(S['jpg'][i], np.uint8), 1), (384, 216))
                txt(rgb, f'{name}  (step {P[name]})', (6, 20), 0.5, (255, 255, 255), 1, bg=(0, 0, 0))
                top.append(rgb); bot.append(cv2.resize(depth_vis(S['depth'][i]), (384, 216), interpolation=cv2.INTER_NEAREST))
            strip = np.vstack([np.hstack(top), np.hstack(bot)])
            cv2.imwrite(str(OUT / f'wrist_rgbd_{v}.jpg'), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    # 3) the 'what B sees' figure: current wrist view vs cached anchor frame + delta, at 4 moments of b_l3_r1
    S, video, v0, P = cache['b_l3_r1']
    tiles = []
    for k in (55, 99, 139, 166):                     # moments with an old anchor (K=20 run)
        i = rec_index(S, k); a = S['anc'][k]
        cur = cv2.resize(cv2.imdecode(np.frombuffer(S['jpg'][i], np.uint8), 1), (448, 252))
        anc = cv2.resize(cv2.imdecode(np.frombuffer(S['pjpg'][a], np.uint8), 1), (448, 252))
        txt(cur, f'now: step {k}', (6, 22), 0.6, (255, 255, 255), 1, bg=(0, 0, 0))
        txt(anc, f'policy image: anchor @ step {a} (age {k - a})', (6, 22), 0.6, (0, 165, 255), 1, bg=(0, 0, 0))
        d = S['delta'][k] * 100
        lab = np.full((40, 448, 3), 18, np.uint8); txt(lab, f'delta = ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) cm  from FK', (8, 27), 0.6, (0, 165, 255))
        tiles.append(np.vstack([cur, anc, lab]))
    cv2.imwrite(str(OUT / 'what_B_sees_l3.jpg'), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # 4) composite viz stills (encode moment / mid-anchor transfer)
    print('figures ->', OUT)


if __name__ == '__main__':
    main()
