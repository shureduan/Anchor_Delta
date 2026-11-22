"""Read-only: grab one wrist RGB-D frame + tool pose, find the table, the bottle (yellow cap) and the green box in the
robot base frame. Sends no command.

    python real/scripts/10_scene.py [--tag name]

Writes real/out/scene/<tag>.npz (raw frames + pose) and <tag>_overlay.jpg, prints the detections (metres, base frame).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import REAL, RGBD, Connection, cloud, pose_matrix, tool_pose  # noqa: E402

OUT = REAL / 'out' / 'scene'


def detect(color, depth, pose):
    Pb, uvc, uvd = cloud(depth, pose_matrix(pose))
    H, W = color.shape[:2]
    inside = (uvc[:, 0] >= 0) & (uvc[:, 0] < W - 1) & (uvc[:, 1] >= 0) & (uvc[:, 1] < H - 1)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    col = np.zeros((len(Pb), 3), np.uint8)
    col[inside] = hsv[uvc[inside, 1].astype(int), uvc[inside, 0].astype(int)]
    rep = {'tool_pose': pose.tolist(), 'points': int(len(Pb))}
    # table = most populated 5 mm height bin among points below the camera
    z = Pb[:, 2]
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.005, 0.005))
    zt0 = edges[np.argmax(hist)]
    near = np.abs(z - zt0) < 0.01
    table_z = float(np.median(z[near]))
    rep['table_z'] = table_z; rep['table_frac'] = float(near.mean())
    obj = (z > table_z + 0.015) & (z < table_z + 0.40)
    yellow = inside & (col[:, 0] >= 18) & (col[:, 0] <= 34) & (col[:, 1] > 130) & (col[:, 2] > 100)
    green = inside & (col[:, 0] >= 35) & (col[:, 0] <= 85) & (col[:, 1] > 50) & (col[:, 2] > 60)
    # clusters on a 5 mm grid of object points
    res = 0.005
    P = Pb[obj]
    clusters = []
    if len(P):
        lo = P[:, :2].min(0)
        ij = np.floor((P[:, :2] - lo) / res).astype(int)
        grid = np.zeros(ij.max(0) + 1, np.uint8); grid[ij[:, 0], ij[:, 1]] = 1
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(grid)
        L = lab[ij[:, 0], ij[:, 1]]
        idx = np.where(obj)[0]
        for k in range(1, n):
            m = L == k
            if m.sum() < 80:
                continue
            Q = P[m]; ii = idx[m]
            clusters.append(dict(n=int(m.sum()), z_top=float(np.percentile(Q[:, 2], 98)) - table_z,
                                 xy_mean=Q[:, :2].mean(0).tolist(), xy_min=Q[:, :2].min(0).tolist(), xy_max=Q[:, :2].max(0).tolist(),
                                 yellow=int(yellow[ii].sum()), green=int(green[ii].sum()), _pts=Q, _idx=ii))
    rep['clusters'] = [{k: v for k, v in c.items() if not k.startswith('_')} for c in clusters]
    # bottle: cap = yellow object points near the top of their cluster
    capm = obj & yellow
    if capm.sum() > 20:
        C = Pb[capm]
        top = C[:, 2] > np.percentile(C[:, 2], 30)
        rep['bottle'] = dict(cap_xy=C[top, :2].mean(0).tolist(), cap_top_above_table=float(np.percentile(C[:, 2], 95) - table_z),
                             n_cap_points=int(capm.sum()))
    # box: thin pale-green walls are nearly invisible to depth from above, so segment it in colour (H 27-50,
    # S 30-110: not the saturated yellow cap, not the grey table) and take every depth point (floor + walls) that
    # projects into the largest green region
    gm = ((hsv[..., 0] >= 27) & (hsv[..., 0] <= 50) & (hsv[..., 1] >= 30) & (hsv[..., 1] <= 110) & (hsv[..., 2] >= 60)).astype(np.uint8)
    gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(gm)
    if n > 1 and st[1:, 4].max() > 3000:
        k = 1 + int(np.argmax(st[1:, 4]))
        region = cv2.dilate((lab == k).astype(np.uint8), np.ones((7, 7), np.uint8))
        x, y, w_, h_ = st[k, :4]
        touches = bool(x <= 2 or y <= 2 or x + w_ >= W - 3 or y + h_ >= H - 3)
        sel = inside.copy()
        sel[inside] = region[uvc[inside, 1].astype(int), uvc[inside, 0].astype(int)] > 0
        sel &= z > table_z - 0.01
        Q = Pb[sel]
        if len(Q) > 100:
            (cx, cy), (w, h), ang = cv2.minAreaRect(Q[:, :2].astype(np.float32))
            rep['box'] = dict(center_xy=[float(cx), float(cy)], size_xy=[float(w), float(h)], yaw_deg=float(ang),
                              rim_above_table=float(np.percentile(Q[:, 2], 99) - table_z), n_points=int(len(Q)),
                              image_area_px=int(st[k, 4]), cut_by_image_border=touches)
        green = sel
    # overlay
    ov = color.copy()
    for m, bgr in ((obj, (255, 0, 255)), (capm, (0, 255, 255)), (obj & green, (0, 255, 0))):
        q = uvc[m & inside].astype(int)
        ov[q[:, 1], q[:, 0]] = bgr
    return rep, ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='scene')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
    cam = RGBD()
    try:
        with Connection() as router:
            cyc = BaseCyclicClient(router)
            p0 = tool_pose(cyc)
            color, depth = cam.latest()
            p1 = tool_pose(cyc)
    finally:
        cam.close()
    if np.abs(p1[:3] - p0[:3]).max() > 0.002:
        print('WARNING: the arm moved while grabbing (keep it still)')
    np.savez(OUT / f'{a.tag}.npz', color=color, depth=depth, pose=p1)
    rep, ov = detect(color, depth, p1)
    cv2.imwrite(str(OUT / f'{a.tag}_overlay.jpg'), ov)
    cv2.imwrite(str(OUT / f'{a.tag}_color.jpg'), color)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
