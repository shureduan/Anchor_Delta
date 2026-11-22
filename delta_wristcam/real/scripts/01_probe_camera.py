"""Probe the Kinova Gen3 wrist RGB-D streams over RTSP (GStreamer). Read-only.

Run in the ``delta-kinova`` env:

    conda activate delta-kinova
    python real/scripts/01_probe_camera.py --ip 192.168.1.10 --seconds 5

Grabs frames from both streams, reports the achieved rate and inter-frame jitter,
and saves one colour PNG, one depth .npy (uint16, millimetres) and a depth
preview into ``real/out/camera_probe/``.

The pipelines follow Kinova's ROS vision driver: colour is H.264, depth is raw
16-bit carried with GStreamer's own RTP payloader (``rtpgstdepay``), which is why
OpenCV/FFmpeg alone cannot read it. They have NOT yet been run against the arm
from this machine; if a pipeline fails, run it by hand with ``gst-launch-1.0``
and ``GST_DEBUG=3`` to see which caps are negotiated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "camera_probe"

COLOR = (
    "rtspsrc location=rtsp://{ip}/color latency=30 ! rtph264depay ! avdec_h264 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink name=sink max-buffers=1 drop=true sync=false"
)
DEPTH = (
    "rtspsrc location=rtsp://{ip}/depth latency=30 ! rtpgstdepay ! "
    "videoconvert ! video/x-raw,format=GRAY16_LE ! appsink name=sink max-buffers=1 drop=true sync=false"
)


def grab(pipeline_str: str, seconds: float, channels: int, dtype):
    pipe = Gst.parse_launch(pipeline_str)
    sink = pipe.get_by_name("sink")
    pipe.set_state(Gst.State.PLAYING)
    stamps, last = [], None
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            sample = sink.emit("try-pull-sample", int(0.5 * Gst.SECOND))
            if sample is None:
                continue
            stamps.append(time.perf_counter())
            buf, caps = sample.get_buffer(), sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")
            ok, info = buf.map(Gst.MapFlags.READ)
            if ok:
                arr = np.frombuffer(info.data, dtype=dtype)
                last = arr[: w * h * channels].reshape(h, w, channels).copy()
                buf.unmap(info)
    finally:
        pipe.set_state(Gst.State.NULL)
    dt = np.diff(stamps) * 1e3 if len(stamps) > 1 else np.array([])
    stats = {
        "frames": len(stamps),
        "fps": float(len(stamps) / seconds),
        "dt_ms_median": float(np.median(dt)) if dt.size else None,
        "dt_ms_p95": float(np.percentile(dt, 95)) if dt.size else None,
        "shape": None if last is None else list(last.shape),
    }
    return last, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.1.10")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    Gst.init([])
    OUT.mkdir(parents=True, exist_ok=True)

    color, cs = grab(COLOR.format(ip=args.ip), args.seconds, 3, np.uint8)
    depth, ds = grab(DEPTH.format(ip=args.ip), args.seconds, 1, np.uint16)

    if color is not None:
        cv2.imwrite(str(OUT / "color.png"), color)
    if depth is not None:
        d = depth[..., 0]
        np.save(OUT / "depth_mm.npy", d)
        valid = d[d > 0]
        ds["depth_mm_valid_frac"] = float((d > 0).mean())
        ds["depth_mm_p5_p50_p95"] = (
            [float(np.percentile(valid, q)) for q in (5, 50, 95)] if valid.size else None
        )
        vis = cv2.applyColorMap(
            cv2.convertScaleAbs(d, alpha=255.0 / max(1, np.percentile(valid, 99) if valid.size else 1)),
            cv2.COLORMAP_TURBO,
        )
        cv2.imwrite(str(OUT / "depth_preview.png"), vis)

    report = {"color": cs, "depth": ds}
    (OUT / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
