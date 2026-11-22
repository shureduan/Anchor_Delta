"""Background recorder: owns the wrist RGB-D streams and (optionally) logs an episode at ~10 Hz.

Per logged frame: colour (BGR, 640x360 JPEG), depth (480x270 uint16 mm), tool pose, joint angles, gripper position,
the current step label and target set by the controller, wall-clock time. The robot state is read through its own
Kortex session (read-only), so the controller's session is not shared across threads.

Saved to real/out/demos/<stamp>/episode.npz (+ colour JPEGs inside the npz as encoded bytes) and meta.json.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from kinova_common import REAL, RGBD, Connection


class Recorder(threading.Thread):
    def __init__(self, hz=10.0):
        super().__init__(daemon=True)
        self.hz = hz
        self.cam = RGBD()
        self.lock = threading.Lock()
        self.color = self.depth = None
        self.t_frame = 0.0
        self.logging = False
        self.label, self.target = 'idle', None
        self.rows = []
        self.stop_ev = threading.Event()
        self.err = None

    # ---- controller side
    def latest(self, fresh_after=None, timeout=3.0):
        """Newest colour/depth pair, captured after time `fresh_after` (default: now)."""
        t0 = time.time() if fresh_after is None else fresh_after
        while time.time() - t0 < timeout:
            with self.lock:
                if self.color is not None and self.depth is not None and self.t_frame > t0:
                    self.last_t = self.t_frame          # capture (arrival) time of the returned frame, for log sync
                    return self.color.copy(), self.depth.copy()
            time.sleep(0.02)
        raise RuntimeError('no fresh camera frame')

    def set_step(self, label, target=None):
        with self.lock:
            self.label = label
            self.target = None if target is None else [float(v) for v in target]

    def start_logging(self):
        with self.lock:
            self.rows = []; self.logging = True

    def save(self, meta):
        with self.lock:
            self.logging = False
            rows = self.rows; self.rows = []
        if not rows:
            return None
        out = REAL / 'out' / 'demos' / time.strftime('%Y%m%d_%H%M%S')
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'episode.npz',
                            t=np.array([r['t'] for r in rows]),
                            depth=np.stack([r['depth'] for r in rows]),
                            color_jpg=np.array([r['jpg'] for r in rows], dtype=object),
                            tool_pose=np.array([r['pose'] for r in rows]),
                            joints=np.array([r['joints'] for r in rows]),
                            gripper=np.array([r['grip'] for r in rows]),
                            label=np.array([r['label'] for r in rows]),
                            target=np.array([r['target'] if r['target'] is not None else [np.nan] * 3 for r in rows]))
        meta = dict(meta, frames=len(rows), hz=self.hz, duration_s=rows[-1]['t'] - rows[0]['t'],
                    color='640x360 JPEG q90 (BGR)', depth='480x270 uint16 mm', calib='real/calib/vision_module.json')
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        return out

    def close(self):
        self.stop_ev.set(); self.join(timeout=3); self.cam.close()

    # ---- thread
    def run(self):
        from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
        try:
            with Connection() as router:
                cyc = BaseCyclicClient(router)
                next_log = 0.0
                while not self.stop_ev.is_set():
                    c = self.cam._pull(self.cam.sinks[0], 3, np.uint8, timeout=0.5)
                    d = self.cam._pull(self.cam.sinks[1], 1, np.uint16, timeout=0.5)
                    now = time.time()
                    with self.lock:
                        if c is not None:
                            self.color = c
                        if d is not None:
                            self.depth = d[..., 0]
                        if c is not None and d is not None:
                            self.t_frame = now
                        logging = self.logging and now >= next_log and self.color is not None and self.depth is not None
                        if logging:
                            color, depth, label, target = self.color, self.depth, self.label, self.target
                    if logging:
                        next_log = now + 1.0 / self.hz
                        fb = cyc.RefreshFeedback(); b = fb.base
                        ok, jpg = cv2.imencode('.jpg', cv2.resize(color, (640, 360), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
                        row = dict(t=now, depth=depth.copy(), jpg=jpg.tobytes(),
                                   pose=[b.tool_pose_x, b.tool_pose_y, b.tool_pose_z, b.tool_pose_theta_x, b.tool_pose_theta_y, b.tool_pose_theta_z],
                                   joints=[a.position for a in fb.actuators],
                                   grip=fb.interconnect.gripper_feedback.motor[0].position, label=label, target=target)
                        with self.lock:
                            if self.logging:
                                self.rows.append(row)
        except Exception as e:  # keep the controller alive; it checks rec.err
            self.err = repr(e)
