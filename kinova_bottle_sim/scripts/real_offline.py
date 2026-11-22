"""Sim-to-real check WITHOUT moving the robot: run the sim-trained ResNet policies (A2 dense, B2 with anchors) on the
recorded real demos (delta_wristcam/real/out/demos) and compare their predictions with what the scripted controller did.

    .venv/bin/python scripts/real_offline.py
    .venv/bin/python scripts/real_offline.py --sim-ref 20      # also the same metrics on 20 sim validation episodes

Per frame where the scripted controller was moving towards a target (>= 1 cm away):
  direction agreement = cos(predicted 1 s displacement, target - tool)
Gripper agreement: predicted close command vs. the script's gripper intent (closed from 'close gripper' to 'open gripper').
Real gripper position (Kortex %, 0.44 open .. 55.3 holding the 36 mm cap) is mapped to the sim reading (0 .. 0.411 for
the same grasp). Images: real 640x360 JPEG -> 128x72 INTER_AREA, as the sim frames.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from policy import POS_SCALE, Policy, anchor_indices, flip_since, proprio  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REAL_DEMOS = ROOT.parent / 'delta_wristcam' / 'real' / 'out' / 'demos'
REAL_OPEN, REAL_HOLD, SIM_HOLD = 0.44, 55.26, 0.411


def real_to_sim_grip(pct):
    return np.clip((np.asarray(pct, np.float64) - REAL_OPEN) / (REAL_HOLD - REAL_OPEN) * SIM_HOLD, 0, 1).astype(np.float32)


def load_policy(path, dev):
    ck = torch.load(ROOT / path, map_location='cpu')
    p = Policy(ck['mode'], pretrained=False, flip=ck.get('flip', False)); p.load_state_dict(ck['state']); return p.to(dev).eval(), ck


def predict(pol, mode, img, tool, grip, rule, dev, K=20, move=0.08):
    """Run one policy over a whole episode (frames img (T,72,128,3) RGB). Returns 1 s displacement (T,3) and close cmd (T,)."""
    T = len(tool); fl = flip_since(grip)
    if mode == 'A':
        anc = np.arange(T)
    else:
        anc = anchor_indices(tool, grip, rule, K=K, move_thresh=move)
    uniq = np.unique(anc)
    with torch.no_grad():
        feats = {}
        for k in range(0, len(uniq), 64):
            idx = uniq[k:k + 64]
            f = pol.encode(torch.from_numpy(img[idx]).to(dev))
            feats.update({int(i): f[j] for j, i in enumerate(idx)})
        F = torch.stack([feats[int(a)] for a in anc])
        if mode == 'A':
            pr = proprio(tool, grip, flip=fl)
        else:
            pr = proprio(tool, grip, tool[anc], (grip[anc] > 0.2).astype(np.float32), np.arange(T) - anc, flip=fl)
        po, gl = pol(F, torch.from_numpy(pr).to(dev))
    return po[:, -1].cpu().numpy() * POS_SCALE, (gl[:, 0] > 0).cpu().numpy(), len(uniq)


def metrics(disp, close, tool, target, closed_intent):
    d = target - tool; n = np.linalg.norm(d, axis=1); m = n > 0.01
    cos = np.sum(disp[m] * d[m], 1) / (np.linalg.norm(disp[m], axis=1) * n[m] + 1e-9)
    return dict(n_moving=int(m.sum()), dir_cos_median=float(np.median(cos)), dir_ok_frac=float(np.mean(cos > np.cos(np.radians(30)))),
                grip_agree=float(np.mean(close == closed_intent)))


def real_episodes():
    eps = []
    for d in sorted(REAL_DEMOS.glob('*/')):
        meta = json.load(open(d / 'meta.json'))
        if meta.get('result') != 'success':
            continue
        z = np.load(d / 'episode.npz', allow_pickle=True)
        img = np.stack([cv2.resize(cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB), (128, 72), interpolation=cv2.INTER_AREA)
                        for b in z['color_jpg']])
        lab = z['label']; step = np.array([int(s.split()[0]) if s[0].isdigit() else 0 for s in lab])
        tgt = z['target'].astype(np.float32); tool = z['tool_pose'][:, :3].astype(np.float32)
        tgt = np.where(np.isnan(tgt), tool, tgt)
        eps.append(dict(name=d.name, note=meta.get('note'), img=img, tool=tool, grip=real_to_sim_grip(z['gripper']), target=tgt,
                        closed_intent=(step >= 6) & (step <= 10)))
    return eps


def sim_episodes(n):
    """Validation-like sim episodes from data/v1 (the ResNet training split's last episodes) with the same fields."""
    from expert import PHASES
    eps = []
    for f in sorted(glob.glob(str(ROOT / 'data/v1/ep_*.npz')))[-n:]:
        z = np.load(f, allow_pickle=True)
        img = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB) for b in z['rgb_jpg']])
        sc = json.loads(str(z['scene'])); ph = z['phase']; tool = z['tool']
        # waypoint target per phase: the label stream's endpoint within the phase run
        tgt = np.zeros_like(tool)
        for t in range(len(ph)):
            j = t
            while j + 1 < len(ph) and ph[j + 1] == ph[t]:
                j += 1
            tgt[t] = z['action'][j, :3]
        eps.append(dict(name=Path(f).stem, img=img, tool=tool, grip=z['grip'], target=tgt,
                        closed_intent=np.isin(ph, [PHASES.index(p) for p in ('close', 'lift', 'transfer', 'lower')])))
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--A', default='runs/A2/last.pt')
    ap.add_argument('--B', default='runs/B2/last.pt')
    ap.add_argument('--sim-ref', type=int, default=20)
    ap.add_argument('--out', default='out/real_offline.json')
    a = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pa, _ = load_policy(a.A, dev); pb, _ = load_policy(a.B, dev)
    configs = [('A dense', pa, 'A', 'dense', {}), ('B fixed K=20', pb, 'B', 'fixed', dict(K=20)),
               ('B adaptive 8 cm', pb, 'B', 'adaptive', dict(move=0.08))]
    rep = {}
    for src, eps in [('real', real_episodes()), ('sim', sim_episodes(a.sim_ref) if a.sim_ref else [])]:
        for name, pol, mode, rule, kw in configs:
            rows = []
            for E in eps:
                disp, close, n_enc = predict(pol, mode, E['img'], E['tool'], E['grip'], rule, dev, **kw)
                r = metrics(disp, close, E['tool'], E['target'], E['closed_intent']); r.update(episode=E['name'], encodes=n_enc, frames=len(E['tool']))
                rows.append(r)
            if not rows:
                continue
            agg = {k: float(np.mean([r[k] for r in rows])) for k in ('dir_cos_median', 'dir_ok_frac', 'grip_agree')}
            agg['encodes_per_frame'] = float(sum(r['encodes'] for r in rows) / sum(r['frames'] for r in rows))
            rep[f'{src} | {name}'] = dict(**agg, episodes=rows)
            print(f'{src:4s} | {name:16s}  direction cos (median) {agg["dir_cos_median"]:+.2f}   within 30 deg {agg["dir_ok_frac"]:.0%}'
                  f'   gripper agrees {agg["grip_agree"]:.0%}   encodes/frame {agg["encodes_per_frame"]:.3f}', flush=True)
    json.dump(rep, open(ROOT / a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
