"""Per-phase direction agreement on the recorded real demos (offline, no robot): for each policy, the fraction of
moving frames whose predicted 1 s displacement is within 30 deg of the scripted controller's direction, split by the
scripted step label. Same inputs as real_offline.py.

    .venv/bin/python scripts/real_phase.py A2=runs/A2/last.pt B2=runs/B2/last.pt:fixed:20 A4=runs/A4/last.pt --out out/real_phase_v4.json
spec = NAME=CKPT[:RULE[:K_or_MOVE]]   (A is always dense; B default fixed 20)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from real_offline import REAL_DEMOS, ROOT, load_policy, predict, real_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('specs', nargs='+'); ap.add_argument('--out', default='out/real_phase.json')
    a = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    eps = real_episodes()
    dirs = [d for d in sorted(REAL_DEMOS.glob('*/')) if json.load(open(d / 'meta.json')).get('result') == 'success']
    for E, d in zip(eps, dirs):
        E['lab'] = np.load(d / 'episode.npz', allow_pickle=True)['label'].astype(str)
    labels = sorted({l for E in eps for l in E['lab'] if l[0].isdigit()}, key=lambda s: int(s.split()[0]))
    rep = {}
    for spec in a.specs:
        name, rest = spec.split('=', 1); parts = rest.split(':')
        pol, ck = load_policy(parts[0], dev)
        rule = 'dense' if ck['mode'] == 'A' else (parts[1] if len(parts) > 1 else 'fixed')
        kw = {} if rule == 'dense' else (dict(K=int(parts[2]) if len(parts) > 2 else 20) if rule == 'fixed' else dict(move=float(parts[2]) if len(parts) > 2 else 0.08))
        cs = {l: [] for l in labels}; allc = []
        for E in eps:
            disp, close, _ = predict(pol, ck['mode'], E['img'], E['tool'], E['grip'], rule, dev, **kw)
            d = E['target'] - E['tool']; n = np.linalg.norm(d, axis=1); m = n > 0.01
            cos = np.sum(disp * d, 1) / (np.linalg.norm(disp, axis=1) * n + 1e-9)
            allc += cos[m].tolist()
            for l in labels:
                cs[l] += cos[m & (E['lab'] == l)].tolist()
        ok = lambda v: float(np.mean(np.array(v) > np.cos(np.radians(30)))) if v else None  # noqa: E731
        rep[name] = dict(ckpt=parts[0], rule=rule, **kw, overall=ok(allc), phases={l: ok(v) for l, v in cs.items() if v},
                         n={l: len(v) for l, v in cs.items() if v})
    names = list(rep)
    print(f'{"phase":36s} {"n":>4s} ' + ' '.join(f'{x:>7s}' for x in names))
    for l in labels:
        if l in rep[names[0]]['phases']:
            print(f'{l[:36]:36s} {rep[names[0]]["n"][l]:4d} ' + ' '.join(f'{rep[x]["phases"][l]:7.0%}' for x in names))
    print(f'{"overall":36s} {"":4s} ' + ' '.join(f'{rep[x]["overall"]:7.0%}' for x in names))
    json.dump(rep, open(ROOT / a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
