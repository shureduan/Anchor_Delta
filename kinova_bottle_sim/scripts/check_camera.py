"""Render the sim wrist camera at the real obs2 pose and compare with the real frame; check the projection."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from scene import Sim, R_TOOL, R_TOOL_CVCAM, CAM_POS_TOOL, COLOR_K, TABLE_Z
REAL = Path(__file__).resolve().parents[2] / 'delta_wristcam/real/out/scene/obs2.npz'   # a recorded real observation
real = np.load(REAL); pose = real['pose']
W, H = 640, 360
sim = Sim(W, H)
sim.reset(pose[:3], (0.2774, 0.1299), (0.2575, -0.0163), np.radians(17.0), tool_euler_deg=pose[3:])
for _ in range(5): sim.step(pose[:3], 0.0)
img = sim.render()
print('tool target', pose[:3].round(3), 'sim tool', sim.tool().round(4), 'gripper', round(sim.gripper_open_frac(), 3))
# expected pixel of the cap centre (top) with the OpenCV model
cap = np.array([0.2774, 0.1299, TABLE_Z + 0.184])
from scene import rot_xyz_deg
RT = rot_xyz_deg(*pose[3:])
Pc = R_TOOL_CVCAM.T @ (RT.T @ (cap - sim.tool()) - CAM_POS_TOOL)
s = W / COLOR_K['w']
u = (Pc[0] / Pc[2] * COLOR_K['fx'] + COLOR_K['cx']) * s; v = (Pc[1] / Pc[2] * COLOR_K['fy'] + COLOR_K['cy']) * s
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); ys, xs = np.nonzero((hsv[..., 0] > 20) & (hsv[..., 0] < 35) & (hsv[..., 1] > 150))
print(f'cap centre expected ({u:.1f}, {v:.1f}) px; yellow blob centroid in render ({xs.mean():.1f}, {ys.mean():.1f}) n={len(xs)}')
rc = cv2.resize(real['color'], (W, H))
side = np.concatenate([rc, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)], 1)
cv2.circle(side, (int(W + u), int(v)), 6, (0, 0, 255), 2)
cv2.imwrite('out/check_camera.jpg', side)
# marker check: a tiny red sphere at a known point -> compare with the OpenCV projection
import mujoco
P = np.array([0.30, 0.02, TABLE_Z + 0.001])
sim.r.update_scene(sim.d, camera=sim.cam)
g = sim.r.scene.geoms[sim.r.scene.ngeom]; sim.r.scene.ngeom += 1
mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]), P, np.eye(3).ravel(), np.array([1, 0, 0, 1], np.float32))
im2 = sim.r.render()
red = (im2[..., 0] > 200) & (im2[..., 1] < 60) & (im2[..., 2] < 60); ys, xs = np.nonzero(red)
Pc = R_TOOL_CVCAM.T @ (RT.T @ (P - sim.tool()) - CAM_POS_TOOL)
u = (Pc[0] / Pc[2] * COLOR_K['fx'] + COLOR_K['cx']) * s; v = (Pc[1] / Pc[2] * COLOR_K['fy'] + COLOR_K['cy']) * s
print(f'marker: OpenCV projection ({u:.1f}, {v:.1f}) vs rendered ({xs.mean():.1f}, {ys.mean():.1f})')
# real vs sim: yellow cap blob centroid
def ycen(bgr):
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); m = (h[..., 0] >= 18) & (h[..., 0] <= 34) & (h[..., 1] > 130) & (h[..., 2] > 100)
    ys, xs = np.nonzero(m); return xs.mean(), ys.mean()
print('cap blob centroid: real (%.1f, %.1f)' % ycen(rc), ' sim (%.1f, %.1f)' % ycen(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
