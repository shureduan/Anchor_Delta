"""Read-only connectivity check for the Kinova Gen3. Sends no motion command.

Run in the ``delta-kinova`` env (Python 3.9, kortex_api 2.6, protobuf 3.5.1):

    conda activate delta-kinova
    python real/scripts/00_check_robot.py --ip 192.168.1.10

Writes ``real/calib/vision_module.json`` with what the arm reports about itself:

* joint angles, measured tool pose, gripper position (sanity check of feedback)
* colour / depth intrinsics of the wrist vision module
* the module's *depth -> colour* extrinsics

Note: Kortex's extrinsics relate the two sensors to each other. They are **not**
camera -> tool. The camera -> tool transform still comes from our own hand-eye
calibration (``real/calib/``), which is what FK-based ego-motion depends on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, VisionConfig_pb2

OUT = Path(__file__).resolve().parents[1] / "calib" / "vision_module.json"


class Connection:
    def __init__(self, ip: str, user: str, password: str, port: int = 10000):
        self.ip, self.port, self.cred = ip, port, (user, password)
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)
        self.session = None

    def __enter__(self):
        self.transport.connect(self.ip, self.port)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = self.cred
        info.session_inactivity_timeout = 10000
        info.connection_inactivity_timeout = 2000
        self.session = SessionManager(self.router)
        self.session.CreateSession(info)
        return self.router

    def __exit__(self, *exc):
        if self.session is not None:
            opts = RouterClientSendOptions()
            opts.timeout_ms = 1000
            self.session.CloseSession(opts)
        self.transport.disconnect()


def intrinsics_dict(x) -> dict:
    d = x.distortion_coeffs
    res = VisionConfig_pb2.Resolution.Name(x.resolution)  # e.g. RESOLUTION_480x270
    w, h = (int(v) for v in res.split("_")[1].split("x")) if "x" in res else (0, 0)
    return {
        "resolution": res,
        "width": w,
        "height": h,
        "fx": x.focal_length_x,
        "fy": x.focal_length_y,
        "cx": x.principal_point_x,
        "cy": x.principal_point_y,
        "distortion_k1_k2_p1_p2_k3": [d.k1, d.k2, d.p1, d.p2, d.k3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.10")
    ap.add_argument("-u", "--username", default="admin")
    ap.add_argument("-p", "--password", default="admin")
    args = ap.parse_args()

    with Connection(args.ip, args.username, args.password) as router:
        base, cyclic = BaseClient(router), BaseCyclicClient(router)
        devices, vision = DeviceManagerClient(router), VisionConfigClient(router)

        t0 = time.perf_counter()
        fb = cyclic.RefreshFeedback()
        fb_ms = (time.perf_counter() - t0) * 1e3
        joints = [a.position for a in fb.actuators]
        tool = fb.base
        pose = base.GetMeasuredCartesianPose()
        grip = [m.position for m in fb.interconnect.gripper_feedback.motor]

        handles = devices.ReadAllDevices().device_handle
        vis_ids = [h.device_identifier for h in handles if h.device_type == DeviceConfig_pb2.VISION]
        if not vis_ids:
            raise SystemExit("no vision module found on this arm")
        vid = vis_ids[0]

        sid = VisionConfig_pb2.SensorIdentifier()
        sid.sensor = VisionConfig_pb2.SENSOR_COLOR
        color = intrinsics_dict(vision.GetIntrinsicParameters(sid, vid))
        sid.sensor = VisionConfig_pb2.SENSOR_DEPTH
        depth = intrinsics_dict(vision.GetIntrinsicParameters(sid, vid))
        ext = vision.GetExtrinsicParameters(vid)
        r = ext.rotation
        depth_to_color = {
            "R": [
                [r.row1.column1, r.row1.column2, r.row1.column3],
                [r.row2.column1, r.row2.column2, r.row2.column3],
                [r.row3.column1, r.row3.column2, r.row3.column3],
            ],
            "t_m": [ext.translation.t_x, ext.translation.t_y, ext.translation.t_z],
        }

    report = {
        "ip": args.ip,
        "joint_deg": joints,
        "tool_pose_feedback": {
            "xyz_m": [tool.tool_pose_x, tool.tool_pose_y, tool.tool_pose_z],
            "theta_xyz_deg": [tool.tool_pose_theta_x, tool.tool_pose_theta_y, tool.tool_pose_theta_z],
        },
        "tool_pose_measured": {
            "xyz_m": [pose.x, pose.y, pose.z],
            "theta_xyz_deg": [pose.theta_x, pose.theta_y, pose.theta_z],
        },
        "gripper_position_pct": grip,
        "feedback_roundtrip_ms": fb_ms,
        "vision_device_id": vid,
        "color_intrinsics": color,
        "depth_intrinsics": depth,
        "depth_to_color_extrinsics": depth_to_color,
        "note": "depth_to_color is sensor-to-sensor, NOT camera-to-tool; hand-eye calibration is separate.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
