"""Clean compute timing of real runs: replay each run's exact policy inputs (policy_frames.npz + logged tool / gripper)
through the same Controller on an otherwise idle GPU (MPS), so the A/B ms/step is not polluted by other jobs that ran
during the lab session. Prints ms/step (mean), image-encoder calls, and checks that the replay reproduces the logged
anchor schedule.
    .venv/bin/python scripts/real_retime.py 20261122_123258 20261122_115909 ...
"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from policy_runtime import Controller   # noqa: E402
from real_offline import real_to_sim_grip   # noqa: E402
R = Path(__file__).resolve().parents[2] / 'delta_wristcam/real/out/demos'
out = {}
for run in sys.argv[1:]:
    m = json.load(open(R / run / 'meta.json')); L = m['policy_log']; J = np.load(R / run / 'policy_frames.npz', allow_pickle=True)['jpg']
    ck = (m.get('policy_server') or {}).get('ckpt') or {'A': 'runs/A4/last.pt', 'B': 'runs/B4/last.pt'}[m['policy']]
    ctl = Controller(ck, m['rule'], m['K'], m['move'])
    imgs = [cv2.cvtColor(cv2.resize(cv2.imdecode(np.frombuffer(J[t], np.uint8), 1), (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for t in range(len(L))]
    for rep in range(2):                     # rep 0 = warm-up
        ctl.reset(); ms = []; enc = []
        for t, r in enumerate(L):
            o = ctl.step(imgs[t], np.array(r['tool'], np.float32), np.float32(real_to_sim_grip(r['grip_pct'])))
            ms.append(o['policy_ms']); enc.append(o['encoded'])
    same = bool(np.all(np.array(enc) == np.array([r['encoded'] for r in L])))
    out[run] = dict(note=m['note'], ckpt=ck, steps=len(L), encodes=int(sum(enc)), ms_per_step=float(np.mean(ms)),
                    ms_encode_step=float(np.median([x for x, e in zip(ms, enc) if e])), ms_head_step=float(np.median([x for x, e in zip(ms, enc) if not e])) if not all(enc) else None,
                    anchors_reproduced=same, ms_logged_in_lab=float(np.mean([r['ms'] for r in L])))
    o = out[run]; print(f"{run} {o['note']:14s} {o['ms_per_step']:5.2f} ms/step (lab {o['ms_logged_in_lab']:5.2f})  enc {o['encodes']}/{o['steps']}  anchors reproduced {same}", flush=True)
json.dump(out, open(Path(__file__).resolve().parents[2] / 'kinova_report/data/real_timing_clean.json', 'w'), indent=1)
