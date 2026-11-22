# Real-robot scripts (Kinova Gen3 + Robotiq 2F-85 + wrist vision module)

Runs in a Python 3.9 environment with the Kortex API (`environment.yml`). The learned policy runs in a separate PyTorch process (`../../kinova_bottle_sim/scripts/policy_server.py`); the two communicate over a local socket.

| script | role |
|---|---|
| `00_check_robot.py`, `01_probe_camera.py` | read-only connectivity check, camera intrinsics, colour / depth stream probing |
| `10_scene.py`, `box_fit.py` | read-only detection of the bottle cap (colour + depth) and the box (silhouette cuboid fit) |
| `11_pick_place.py` | geometric scripted pick-and-place, used to record real demonstrations |
| `13_goto.py`, `12_retract.py` | start pose and retract; each motion is confirmed by the operator |
| `21_policy_run.py` | learned-policy client: per-step target clipping (≤ 5 mm), 5 cm/s motion, workspace clamp, operator stop, gripper waits until the fingers settle, full per-step logging (timestamps, plan, anchor, delta, exact policy input frames) |
| `recorder.py`, `kinova_common.py` | wrist RGB-D + robot-state recorder; Kortex connection helpers |

`calib/vision_module.json` holds the colour / depth intrinsics read from the arm. The camera mount comes from the Kinova URDF. The table top is at z = -0.025 m in the base frame.
