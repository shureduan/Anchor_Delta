# kinova_bottle_sim — simulation, policies and real-robot runtime for the Kinova bottle-into-box study

This folder contains the full technical pipeline behind the Kinova real-robot results of DELTA. The task is to pick a bottle by its cap and put it into an open green box, using **only a wrist-mounted camera**. Two policies are compared:

- **A — dense encoding.** The wrist image is encoded at every control step.
- **B — anchor + delta.** The wrist image is encoded only at sparse *anchors*. Between anchors, the policy receives the hand's motion since the anchor (the *delta*), which comes from forward kinematics and the gripper sensor.

Both are trained only in simulation and deployed **zero-shot** on a Kinova Gen3 with a Robotiq 2F-85 gripper.

The complete write-up (method, setups, results, failure analysis, media) is in the [repository overview](../README.md) and [`../report/`](../report/).

## Summary of results

| | A dense | B fixed K=20 | B adaptive (8 cm) |
|---|---|---|---|
| Simulation success, v5 models (n = 30, unseen scenes) | 29/30 | 30/30 | 30/30 |
| Real robot, 3 layouts (one successful run each, shown as videos) | 3/3 | 3/3 | 3/3 |
| Image encodes per real episode | 100 % | 5.1–5.4 % | 5.3–5.5 % |
| Policy compute per step (idle Apple M4 GPU) | 8.95–8.98 ms | 1.56–1.78 ms | 1.56–1.68 ms |
| Stored visual data per real episode | 2.1–2.4 MB | 117–138 kB | 110–147 kB |

## Design

| component | file | summary |
|---|---|---|
| Simulated cell | `scripts/scene.py` | MuJoCo replica of the real setup: Kinova tool frame, wrist camera with the real intrinsics and URDF mount, measured bottle and box, table height from depth. It also renders per-pixel class maps (background / bottle / box / gripper), which are used for training-time augmentation only |
| Scripted expert | `scripts/expert.py` | Privileged-state pick-and-place at 5 cm/s and 10 Hz, with free-space noise for recovery data. The v5 expert closes and releases the gripper **on arrival**, so every gripper decision is observable |
| Policy | `scripts/policy.py` | ResNet-18 encoder (ImageNet init, fine-tuned) + MLP head; outputs a 10-step chunk of tool-position offsets and gripper logits. B adds a 5-d delta (FK displacement since the anchor, gripper state at the anchor, anchor age) |
| Anchor rules | `scripts/policy.py` | Fixed every K steps, or adaptive (episode start, gripper flip, > 8 cm of motion), decided from proprioception only |
| Training | `scripts/train.py`, `scripts/dr_aug.py` | Behaviour cloning (L1 + 0.1 BCE) with class-map-driven appearance randomisation. B is trained on mixed anchor ages, so the anchor age cannot serve as a clock |
| Evaluation | `scripts/eval.py` | Closed loop on unseen scenes with ACT-style temporal ensembling; reports success, encodes per step and measured compute |
| Real-robot runtime | `scripts/policy_runtime.py`, `scripts/policy_server.py` | The same step controller in simulation and on the robot. The server talks to the Kortex client in `../delta_wristcam/real/scripts/21_policy_run.py` |
| Real-data analysis | `scripts/real_offline.py`, `real_phase.py`, `real_grip_check.py`, `real_retime.py` | Offline agreement with recorded real demonstrations, gripper checks on real frames, and exact replay for clean compute timing |
| Supplementary VLA line | `scripts/vla.py`, `train_vla.py`, `eval_vla.py` | The same A / B comparison on SmolVLA with real-time chunking |

## Model versions

| version | change | effect |
|---|---|---|
| v2 | both policies receive "steps since the gripper last changed"; B trained on mixed anchor ages | removes B's anchor-age shortcut (fixed-60 rises from 2/50 to 47/50 in simulation) |
| v3 | simulation colours matched to the real scene | lower agreement on real frames (69 %), showing the gap was not colour |
| v4 | class-map-driven appearance randomisation | real-frame agreement 97–98 %; first real successes |
| v5 | observable gripper timing; gripper fingers erased and bottle textures randomised in training | removes the two real gripper failure modes of v4; 3/3 on the layout where v4 failed |

Datasets (`data/`), checkpoints (`runs/`) and the MuJoCo Menagerie gripper model (`assets/menagerie/`, from google-deepmind/mujoco_menagerie) are kept outside the repository because of their size.

## Status

The v5 models are the current models. They have been validated in simulation and on one real layout with all three configurations. The next step is to evaluate them across all real layouts with more trials per configuration.
