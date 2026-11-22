"""Shared helpers for the Kinova Gen3 (6-DoF, vision module, Robotiq 2F-85): connection, poses, wrist RGB-D,
point cloud in the robot base frame.

Camera mounting (Kinova ros_kortex gen3_macro.xacro, noetic-devel; parent = end_effector_link):
    depth_module  xyz (0.0275, 0.066, -0.00305)   rpy (pi, pi, 0)
    color_module  xyz (0, 0.05639, -0.00305)      rpy (pi, pi, 0)
Tool frame (Robotiq 2F-85 tool_frame, and the arm's own tool configuration): end_effector_link + (0, 0, 0.12).
Check: the colour - depth offset implied by these, expressed in the camera axes, equals the depth->colour
extrinsics the arm reports (-0.0271, -0.0100, -0.0047) m.
Kortex Cartesian angles: degrees, R = Rz(theta_z) Ry(theta_y) Rx(theta_x).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REAL = Path(__file__).resolve().parents[1]
CALIB = json.loads((REAL / 'calib' / 'vision_module.json').read_text())
IP = '192.168.2.10'

R_EE_CAM = np.diag([-1.0, -1.0, 1.0])                   # rpy (pi, pi, 0)
T_TOOL_EE = np.eye(4); T_TOOL_EE[2, 3] = -0.12          # inverse of ee -> tool (z + 0.12)


def homog(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


T_TOOL_DEPTH = T_TOOL_EE @ homog(R_EE_CAM, [0.0275, 0.066, -0.00305])
T_TOOL_COLOR = T_TOOL_EE @ homog(R_EE_CAM, [0.0, 0.05639, -0.00305])


def rot_xyz_deg(tx, ty, tz):
    a, b, c = np.radians([tx, ty, tz])
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose_matrix(p):
    """p = (x, y, z [m], theta_x, theta_y, theta_z [deg]) -> 4x4 base <- tool."""
    return homog(rot_xyz_deg(*p[3:]), p[:3])


class Connection:
    def __init__(self, ip=IP, user='admin', password='admin', port=10000):
        from kortex_api.RouterClient import RouterClient
        from kortex_api.TCPTransport import TCPTransport
        self.ip, self.port, self.cred = ip, port, (user, password)
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.session = None

    def __enter__(self):
        from kortex_api.autogen.messages import Session_pb2
        from kortex_api.SessionManager import SessionManager
        self.transport.connect(self.ip, self.port)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = self.cred
        info.session_inactivity_timeout = 60000
        info.connection_inactivity_timeout = 2000
        self.session = SessionManager(self.router)
        self.session.CreateSession(info)
        return self.router

    def __exit__(self, *exc):
        from kortex_api.RouterClient import RouterClientSendOptions
        if self.session is not None:
            opts = RouterClientSendOptions(); opts.timeout_ms = 1000
            self.session.CloseSession(opts)
        self.transport.disconnect()


def tool_pose(cyclic):
    b = cyclic.RefreshFeedback().base
    return np.array([b.tool_pose_x, b.tool_pose_y, b.tool_pose_z, b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z])


class RGBD:
    """Keeps both RTSP streams open; latest() returns the newest colour (BGR) and depth (mm) frames."""
    COLOR = ('rtspsrc location=rtsp://{ip}/color latency=30 ! rtph264depay ! avdec_h264 ! videoconvert ! '
             'video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true sync=false')
    DEPTH = ('rtspsrc location=rtsp://{ip}/depth latency=30 ! rtpgstdepay ! videoconvert ! '
             'video/x-raw,format=GRAY16_LE ! appsink name=sink max-buffers=1 drop=true sync=false')

    def __init__(self, ip=IP):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init([])
        self.Gst = Gst
        self.pipes = []
        for s in (self.COLOR, self.DEPTH):
            p = Gst.parse_launch(s.format(ip=ip)); p.set_state(Gst.State.PLAYING); self.pipes.append(p)
        self.sinks = [p.get_by_name('sink') for p in self.pipes]

    def _pull(self, sink, ch, dtype, timeout=2.0):
        s = sink.emit('try-pull-sample', int(timeout * self.Gst.SECOND))
        if s is None:
            return None
        buf, caps = s.get_buffer(), s.get_caps().get_structure(0)
        w, h = caps.get_value('width'), caps.get_value('height')
        ok, info = buf.map(self.Gst.MapFlags.READ)
        arr = np.frombuffer(info.data, dtype=dtype)[: w * h * ch].reshape(h, w, ch).copy()
        buf.unmap(info)
        return arr

    def latest(self, n_flush=3):
        """Drop a few frames so the result is fresh (streams lag ~1-2 frames)."""
        c = d = None
        for _ in range(n_flush):
            c2 = self._pull(self.sinks[0], 3, np.uint8)
            d2 = self._pull(self.sinks[1], 1, np.uint16)
            c = c2 if c2 is not None else c
            d = d2 if d2 is not None else d
        return c, None if d is None else d[..., 0]

    def close(self):
        for p in self.pipes:
            p.set_state(self.Gst.State.NULL)


def cloud(depth_mm, T_base_tool, stride=1):
    """Depth image -> (N, 3) base-frame points, (N, 2) colour pixel coords, (N, 2) depth pixel coords."""
    K = CALIB['depth_intrinsics']; C = CALIB['color_intrinsics']; t_dc = np.array(CALIB['depth_to_color_extrinsics']['t_m'])
    v, u = np.mgrid[0:depth_mm.shape[0]:stride, 0:depth_mm.shape[1]:stride]
    z = depth_mm[::stride, ::stride].astype(float) / 1000.0
    ok = z > 0
    u, v, z = u[ok], v[ok], z[ok]
    P = np.stack([(u - K['cx']) / K['fx'] * z, (v - K['cy']) / K['fy'] * z, z], 1)
    Pc = P + t_dc
    uc = Pc[:, 0] / Pc[:, 2] * C['fx'] + C['cx']; vc = Pc[:, 1] / Pc[:, 2] * C['fy'] + C['cy']
    T = T_base_tool @ T_TOOL_DEPTH
    Pb = P @ T[:3, :3].T + T[:3, 3]
    return Pb, np.stack([uc, vc], 1), np.stack([u, v], 1)
