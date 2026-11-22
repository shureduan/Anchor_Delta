"""MuJoCo scene that mirrors the real Kinova Gen3 bottle -> box setup (2026-11-15 lab session).

Frames and numbers (all from the real robot):
  world      = Kinova base frame (z up); table top at z = -0.025 m, table edge at y = -0.15 m (both measured
               with the wrist depth camera on 18 real frames)
  hand body  = Kortex tool frame (tool_frame of the Robotiq 2F-85 = end_effector_link + 0.12 m along z); the hand
               is driven by a mocap target through a weld (actions = tool target xyz + gripper), orientation fixed
               to the real observation/pick orientation theta = (180, 0, 97.5) deg (gripper vertical)
  gripper    = mujoco_menagerie robotiq_2f85, mounted at the flange (tool - 0.12 m), no extra rotation: its fingers
               close along the mount x axis = tool x
               (in the real wrist image the fingers appear at the left/right bottom corners = camera x = -tool x)
  colour cam = Kinova ros_kortex URDF color_module: tool-frame position (0, 0.05639, -0.12305), optical axes
               x_cam = -x_tool, y_cam = -y_tool, z_cam = z_tool; intrinsics read from the arm
               (1280x720, fx 1297.67, fy 1298.63, cx 620.91, cy 238.28), rendered at a lower resolution
  bottle     = 18.4 cm tall, body diameter 7 cm, cap diameter 3.6 cm (yellow), tea-brown body, orange label
  box        = open pale-green box, 8.7 x 8.7 cm outer, 7.5 cm tall, 3 mm walls
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRIPPER_XML = ROOT / 'assets' / 'menagerie' / 'robotiq_2f85' / '2f85.xml'
TABLE_Z = -0.025
TOOL_EULER_DEG = (180.0, 0.0, 97.5)          # Kortex theta_x, theta_y, theta_z (R = Rz Ry Rx)
COLOR_K = dict(w=1280, h=720, fx=1297.6728515625, fy=1298.63134765625, cx=620.9140014648438, cy=238.2803192138672)
CAM_POS_TOOL = np.array([0.0, 0.05639, -0.00305 - 0.12])
BOTTLE_H, BOTTLE_R, CAP_R = 0.184, 0.035, 0.018
BOX_W, BOX_H, BOX_T = 0.087, 0.075, 0.003


def rot_xyz_deg(tx, ty, tz):
    a, b, c = np.radians([tx, ty, tz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q


R_TOOL = rot_xyz_deg(*TOOL_EULER_DEG)
# MuJoCo camera frame (x right, y up, looks along -z) from the OpenCV optical frame (x right, y down, z forward)
R_TOOL_CVCAM = np.diag([-1.0, -1.0, 1.0])
R_TOOL_MJCAM = R_TOOL_CVCAM @ np.diag([1.0, -1.0, -1.0])


def bottle_xml():
    """Bottle along +z from its base: body cylinder, shoulder (stacked cylinders), neck, yellow cap."""
    g = []
    g.append(f'<geom name="b_body" type="cylinder" size="{BOTTLE_R} 0.0575" pos="0 0 0.0575" material="tea" mass="0.42"/>')
    g.append(f'<geom name="b_label" type="cylinder" size="{BOTTLE_R + 0.0005} 0.035" pos="0 0 0.055" material="label" contype="0" conaffinity="0" mass="0"/>')
    for i, (r, z) in enumerate([(0.032, 0.122), (0.027, 0.131), (0.022, 0.140), (0.0175, 0.149)]):
        g.append(f'<geom name="b_sh{i}" type="cylinder" size="{r} 0.0045" pos="0 0 {z}" material="tea" mass="0.01"/>')
    g.append(f'<geom name="b_neck" type="cylinder" size="0.0155 0.0075" pos="0 0 0.161" material="tea" mass="0.01"/>')
    g.append(f'<geom name="b_cap" type="cylinder" size="{CAP_R} 0.0080" pos="0 0 {BOTTLE_H - 0.008}" material="cap" mass="0.005" friction="1.2 0.01 0.001"/>')
    return '\n'.join(g)


def box_xml():
    w, h, t = BOX_W / 2, BOX_H / 2, BOX_T / 2
    g = [f'<geom name="x_floor" type="box" size="{w} {w} {t}" pos="0 0 {t}" material="boxg" mass="0.03"/>']
    for i, (px, py, sx, sy) in enumerate([(w - t, 0, t, w), (-(w - t), 0, t, w), (0, w - t, w, t), (0, -(w - t), w, t)]):
        g.append(f'<geom name="x_wall{i}" type="box" size="{sx} {sy} {h}" pos="{px} {py} {h}" material="boxg" mass="0.015"/>')
    return '\n'.join(g)


# Appearance sets. 'v2' = the original look (data v1/v2, ResNet A/B/A2/B2, SmolVLA). 'real' = colours matched to the
# real wrist frames (out/real_vs_sim.jpg): the real box is a dull sage green (rendered ~(112, 119, 101), darker than the
# table), the table a light grey-lilac (~(162, 153, 164)); the v2 box rendered ~(151, 173, 120), bright and saturated.
LOOKS = {
    'v2': dict(table1='0.80 0.78 0.84', table2='0.76 0.74 0.80', boxg='0.66 0.75 0.50'),
    'real': dict(table1='0.76 0.72 0.77', table2='0.72 0.68 0.73', boxg='0.49 0.52 0.44'),
}


def scene_xml(render_w=320, render_h=180, look='v2'):
    L = LOOKS[look]
    s = render_w / COLOR_K['w']
    # MuJoCo pinhole camera with explicit intrinsics: sensorsize (any length units), focal and principal in the same
    # units; principal = offset of the principal point from the sensor centre (both signs negated w.r.t. OpenCV)
    W, H = COLOR_K['w'], COLOR_K['h']
    focal = f"{COLOR_K['fx']} {COLOR_K['fy']}"
    # sign convention verified with a projected marker (scripts/check_camera.py): x offset enters negated
    principal = f"{-(COLOR_K['cx'] - W / 2)} {-(COLOR_K['cy'] - H / 2)}"
    cam_q = mat2quat(R_TOOL_MJCAM)
    return f"""
<mujoco model="kinova_bottle_box">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" cone="elliptic" impratio="10" integrator="implicitfast"/>
  <visual>
    <global offwidth="{max(render_w, 640)}" offheight="{max(render_h, 480)}"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.25 0.25 0.25" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="tabletex" type="2d" builtin="flat" rgb1="{L['table1']}" rgb2="{L['table2']}" mark="random" random="0.04" width="512" height="512"/>
    <material name="table" texture="tabletex" texrepeat="4 4" rgba="1 1 1 1"/>
    <material name="floor" rgba="0.35 0.33 0.32 1"/>
    <material name="tea" rgba="0.33 0.17 0.07 0.95" specular="0.6" shininess="0.8"/>
    <material name="label" rgba="0.90 0.42 0.12 1"/>
    <material name="cap" rgba="0.95 0.85 0.12 1"/>
    <material name="boxg" rgba="{L['boxg']} 1"/>
  </asset>
  <worldbody>
    <light name="top" pos="0.3 0.0 1.5" dir="0 0 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" material="floor"/>
    <geom name="table" type="box" size="0.50 0.45 0.02" pos="0.35 0.30 {TABLE_Z - 0.02}" material="table" friction="0.8 0.005 0.0001"/>
    <body name="target" mocap="true" pos="0.30 0.10 0.40" quat="{' '.join(map(str, mat2quat(R_TOOL)))}"/>
    <body name="hand" pos="0.30 0.10 0.40" quat="{' '.join(map(str, mat2quat(R_TOOL)))}">
      <freejoint name="hand_free"/>
      <inertial pos="0 0 -0.06" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <camera name="wrist" pos="{' '.join(map(str, CAM_POS_TOOL))}" quat="{' '.join(map(str, cam_q))}"
              sensorsize="{W} {H}" focal="{focal}" principal="{principal}" resolution="{render_w} {render_h}"/>
      <site name="tool" size="0.004" rgba="1 0 0 1" group="5"/>
      <frame name="flange" pos="0 0 -0.12"/>
    </body>
    <body name="bottle" pos="0.30 0.15 {TABLE_Z}">
      <freejoint name="bottle_free"/>
      {bottle_xml()}
    </body>
    <body name="box" pos="0.30 0.00 {TABLE_Z}">
      <freejoint name="box_free"/>
      {box_xml()}
    </body>
  </worldbody>
  <equality>
    <weld name="hand_weld" body1="target" body2="hand" solref="0.02 1" solimp="0.95 0.99 0.001"/>
  </equality>
</mujoco>
"""


def build(render_w=320, render_h=180, look='v2'):
    spec = mujoco.MjSpec.from_string(scene_xml(render_w, render_h, look))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    frame = spec.frame('flange')
    frame.attach_body(grip.worldbody.first_body(), 'g_', '')
    model = spec.compile()
    return model


class Sim:
    """Thin wrapper: set the tool target and the gripper command, step at 10 Hz, render the wrist camera."""
    CTRL_HZ = 10

    def __init__(self, render_w=320, render_h=180, look='v2'):
        self.look = look
        self.m = build(render_w, render_h, look)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, render_h, render_w)
        self.n_sub = int(round(1.0 / self.CTRL_HZ / self.m.opt.timestep))
        self.mocap = self.m.body('target').mocapid[0]
        self.act = self.m.actuator('g_fingers_actuator').id
        self.cam = self.m.camera('wrist').id
        self.hand = self.m.body('hand').id

    def reset(self, tool_xyz, bottle_xy, box_xy, box_yaw=0.0, tool_euler_deg=None):
        mujoco.mj_resetData(self.m, self.d)
        q = mat2quat(R_TOOL if tool_euler_deg is None else rot_xyz_deg(*tool_euler_deg))
        self.d.mocap_pos[self.mocap] = tool_xyz; self.d.mocap_quat[self.mocap] = q
        j = self.m.joint('hand_free').qposadr[0]; self.d.qpos[j:j + 3] = tool_xyz; self.d.qpos[j + 3:j + 7] = q
        j = self.m.joint('bottle_free').qposadr[0]; self.d.qpos[j:j + 3] = [*bottle_xy, TABLE_Z]; self.d.qpos[j + 3:j + 7] = [1, 0, 0, 0]
        j = self.m.joint('box_free').qposadr[0]; self.d.qpos[j:j + 3] = [*box_xy, TABLE_Z]
        self.d.qpos[j + 3:j + 7] = [np.cos(box_yaw / 2), 0, 0, np.sin(box_yaw / 2)]
        self.d.ctrl[self.act] = 0
        mujoco.mj_forward(self.m, self.d)
        for _ in range(100):
            mujoco.mj_step(self.m, self.d)

    def randomize(self, rng):
        """Domain randomisation of appearance and camera mounting (call after reset)."""
        m = self.m
        if not hasattr(self, '_base'):
            self._base = dict(mat=m.mat_rgba.copy(), cam_pos=m.cam_pos[self.cam].copy(), cam_quat=m.cam_quat[self.cam].copy(),
                              amb=m.vis.headlight.ambient.copy(), dif=m.vis.headlight.diffuse.copy(), ldif=m.light_diffuse.copy(),
                              ldir=m.light_dir.copy())
        B = self._base
        m.mat_rgba[:] = B['mat']
        for name, jit in (('table', 0.08), ('tea', 0.10), ('label', 0.12), ('cap', 0.08), ('boxg', 0.10), ('floor', 0.15)):
            if name == 'boxg' and self.look != 'v2':
                jit = 0.04                          # keep the hue green; saturation is varied separately below
            i = m.material(name).id
            m.mat_rgba[i, :3] = np.clip(B['mat'][i, :3] * rng.uniform(0.8, 1.15) + rng.normal(0, jit, 3), 0, 1)
        if self.look != 'v2':                       # also vary the box saturation (dull sage .. clearly green)
            i = m.material('boxg').id; c = m.mat_rgba[i, :3]; g = c.mean()
            m.mat_rgba[i, :3] = np.clip(g + (c - g) * rng.uniform(0.5, 2.0), 0, 1)
        k = rng.uniform(0.6, 1.4)
        m.vis.headlight.ambient[:] = np.clip(B['amb'] * k, 0, 1); m.vis.headlight.diffuse[:] = np.clip(B['dif'] * rng.uniform(0.5, 1.6), 0, 1)
        m.light_diffuse[:] = np.clip(B['ldif'] * rng.uniform(0.5, 1.6), 0, 1)
        d = B['ldir'][0] + rng.normal(0, 0.3, 3); d[2] = -abs(d[2]) - 0.5; m.light_dir[0] = d / np.linalg.norm(d)
        m.cam_pos[self.cam] = B['cam_pos'] + rng.normal(0, 0.003, 3)
        dq = np.r_[1.0, rng.normal(0, np.radians(1.0) / 2, 3)]; dq /= np.linalg.norm(dq)
        q = np.zeros(4); mujoco.mju_mulQuat(q, B['cam_quat'], dq); m.cam_quat[self.cam] = q

    def step(self, tool_xyz, grip01):
        self.d.mocap_pos[self.mocap] = tool_xyz
        self.d.ctrl[self.act] = 255.0 * float(np.clip(grip01, 0, 1))
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)

    def render(self):
        self.r.update_scene(self.d, camera=self.cam)
        return self.r.render()                       # RGB uint8 (H, W, 3)

    SEG_BG, SEG_BOTTLE, SEG_BOX, SEG_GRIPPER = 0, 1, 2, 3

    def render_seg(self):
        """Class map of the wrist view (H, W) uint8: 0 table / floor / nothing, 1 bottle, 2 box, 3 gripper.
        Training-time augmentation only (v4 background / object appearance randomisation), never a policy input."""
        if not hasattr(self, '_geom_cls'):
            cls = np.zeros(self.m.ngeom + 1, np.uint8)           # last entry: no geom (id -1)
            for i in range(self.m.ngeom):
                body = self.m.body(self.m.geom_bodyid[i]).name
                cls[i] = self.SEG_BOTTLE if body == 'bottle' else self.SEG_BOX if body == 'box' else \
                    self.SEG_BG if body == 'world' else self.SEG_GRIPPER
            self._geom_cls = cls
        self.r.enable_segmentation_rendering()
        try:
            self.r.update_scene(self.d, camera=self.cam)
            o = self.r.render()
        finally:
            self.r.disable_segmentation_rendering()
        ids = np.where((o[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (o[..., 0] >= 0), o[..., 0], -1)
        return self._geom_cls[ids]

    def tool(self):
        return self.d.xpos[self.hand].copy()

    def body_pose(self, name):
        b = self.m.body(name).id
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def gripper_open_frac(self):
        """0 = fully open, 1 = fully closed (driver joint 0 .. 0.8 rad)."""
        return float(self.d.qpos[self.m.joint('g_right_driver_joint').qposadr[0]] / 0.8)
