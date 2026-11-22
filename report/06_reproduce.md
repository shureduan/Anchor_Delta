# 6. Reproduce and file map

## 6.1 Code

| path | role |
|---|---|
| `kinova_bottle_sim/scripts/scene.py` | MuJoCo scene, camera, looks, randomisation, class maps (`render_seg`) |
| `kinova_bottle_sim/scripts/expert.py` | scene sampling, start pose, scripted expert (`observable=True` for v5), success test |
| `kinova_bottle_sim/scripts/gen_data.py` | parallel demonstration generation (`--look real --seg --observable`) |
| `kinova_bottle_sim/scripts/policy.py` | network, proprio / delta assembly, anchor rules, `flip_since` |
| `kinova_bottle_sim/scripts/dr_aug.py` | class-map-driven appearance randomisation (training only) |
| `kinova_bottle_sim/scripts/train.py` | training of A / B (`--dr`, `--dr5`, `--freeze-enc`) |
| `kinova_bottle_sim/scripts/eval.py` | sim closed-loop evaluation |
| `kinova_bottle_sim/scripts/policy_runtime.py` | step controller shared by sim and robot |
| `kinova_bottle_sim/scripts/policy_server.py` | robot-side policy server |
| `kinova_bottle_sim/scripts/real_offline.py`, `real_phase.py`, `real_grip_check.py`, `real_retime.py` | offline checks on real data |
| `delta_wristcam/real/scripts/21_policy_run.py` | robot client: safety limits, `--auto`, logging |
| `delta_wristcam/real/scripts/13_goto.py`, `12_retract.py`, `11_pick_place.py` | start pose, retract, scripted demonstration |
| `viz/render_run.py`, `make_figures.py`, `analysis_figures.py` | per-frame visualisation videos and figures |

## 6.2 Pipeline

```bash
cd kinova_bottle_sim
scripts/run_v5.sh                 # data/v5 -> runs/A5, runs/B5 -> real-frame checks -> sim evals (~2 h on an M4)
.venv/bin/python scripts/real_grip_check.py runs/A5/last.pt runs/B5/last.pt
.venv/bin/python scripts/real_retime.py <run stamps...>          # clean compute timing from the recorded inputs
cd ../viz
../kinova_bottle_sim/.venv/bin/python render_run.py b_l3_r1.MOV  # per-frame algorithm visualisation
```

## 6.3 Data in this report (`data/`)

| file | content |
|---|---|
| `real_runs.csv` | all 26 real runs: configuration, model, result, steps, encodes, failure category, whether the run counts in the main comparison |
| `real_grid_summary.json` | the 3 x 3 grid: success, encodes, clean ms/step, stored visual bytes |
| `real_timing_clean.json` | clean timing per run and a flag that the anchors were reproduced |
| `real_phase_all_versions.json` | offline per-phase agreement, v2–v5 |

Raw robot recordings (~0.8 GB), phone videos (~1 GB), datasets (~11 GB) and checkpoints (~1.5 GB) are **not** in the repository.
