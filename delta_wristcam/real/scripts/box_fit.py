"""Fit the open green box as a cuboid (centre x, y, yaw; side s = 8.7 cm and height h = 7.5 cm measured) to the green silhouettes of one or more
wrist frames: the projected convex hull of the 8 cuboid corners should cover the green region (IoU).
Thin walls make depth useless here; the silhouette uses only colour plus the known camera pose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kinova_common import CALIB, T_TOOL_COLOR, pose_matrix  # noqa: E402


def green_mask(color):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 27) & (hsv[..., 0] <= 60) & (hsv[..., 1] >= 15) & (hsv[..., 1] <= 110) & (hsv[..., 2] >= 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    return (lab == k).astype(np.uint8)


def project(Pb, pose):
    C = CALIB['color_intrinsics']
    T = np.linalg.inv(pose_matrix(pose) @ T_TOOL_COLOR)
    P = Pb @ T[:3, :3].T + T[:3, 3]
    return np.stack([P[:, 0] / P[:, 2] * C['fx'] + C['cx'], P[:, 1] / P[:, 2] * C['fy'] + C['cy']], 1)


def corners(q, table_z):
    cx, cy, yaw, s, h = q
    c, sn = np.cos(yaw), np.sin(yaw)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * s / 2 @ np.array([[c, sn], [-sn, c]]) + [cx, cy]
    return np.concatenate([np.c_[sq, np.full(4, table_z)], np.c_[sq, np.full(4, table_z + h)]])


def iou(q, views, table_z, scale=0.25):
    tot = 0.0
    for color_mask, pose in views:
        uv = project(corners(q, table_z), pose) * scale
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        m = np.zeros_like(color_mask)
        cv2.fillConvexPoly(m, hull, 1)
        inter = (m & color_mask).sum(); uni = (m | color_mask).sum()
        tot += inter / max(uni, 1)
    return tot / len(views)


def fit(frames, table_z, init_xy, s0=0.087, h0=0.075, fixed_s=0.087, fixed_h=0.075):
    """frames: list of (color, pose). Side s and height h fixed (measured). Pattern search over (x, y, yaw), numpy only.
    Returns (cx, cy, yaw_rad, s, h), mean IoU."""
    views = []
    for color, pose in frames:
        g = green_mask(color)
        if g is not None:
            views.append((cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST), pose))
    if not views:
        return None, 0.0
    s, h = fixed_s or s0, fixed_h or h0
    f = lambda q: iou(np.array([q[0], q[1], q[2], s, h]), views, table_z)
    best_q, best_v = None, -1.0
    for yaw0 in np.radians(np.arange(0, 90, 7.5)):
        q = np.array([init_xy[0], init_xy[1], yaw0]); v = f(q)
        steps = np.array([0.02, 0.02, np.radians(8)])
        while steps[0] > 2e-4:
            improved = False
            for i in range(3):
                for sg in (1, -1):
                    q2 = q.copy(); q2[i] += sg * steps[i]; v2 = f(q2)
                    if v2 > v:
                        q, v, improved = q2, v2, True
            if not improved:
                steps /= 2
        if v > best_v:
            best_q, best_v = q, v
    return np.array([best_q[0], best_q[1], best_q[2] % (np.pi / 2), s, h]), best_v

if __name__ == '__main__':
    REAL = Path(__file__).resolve().parents[1]
    tags = sys.argv[1:] or ['obs1', 'obs2']
    fr = [(np.load(REAL / 'out' / 'scene' / f'{t}.npz')['color'], np.load(REAL / 'out' / 'scene' / f'{t}.npz')['pose']) for t in tags]
    for t, (c, p) in zip(tags, fr):
        q1, s1 = fit([(c, p)], -0.0249, (0.25, -0.025))
        print(f'{t}: centre ({q1[0]:.4f}, {q1[1]:.4f})  yaw {np.degrees(q1[2]):.1f}  IoU {s1:.3f}')
