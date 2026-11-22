"""Fine-tune SmolVLA (A or B) on cached image features, on the Mac (MPS). VLM + vision encoder frozen (SmolVLA default
train_expert_only); trained: action expert, state / action / time projections, and B's delta_proj.

    .venv-vla/bin/python scripts/train_vla.py --mode A --steps 3000 --out runs/vla_A
    .venv-vla/bin/python scripts/train_vla.py --mode B --steps 3000 --out runs/vla_B

B (see vla.py): image features of the anchor frame + current state and anchor delta in the state token. Anchors are
sampled exactly like the v2 ResNet B (train.sample_anchor): 1/3 adaptive, 1/3 fixed K ~ U{1..60}, 1/3 random
age U[0, 80]. Loss = SmolVLA flow-matching MSE on the normalised action chunk.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from policy import POS_SCALE, anchor_indices, flip_since  # noqa: E402
from train import sample_anchor  # noqa: E402
from vla import H_VLA, VLA, anchor_delta, state_vec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(data, n):
    eps = []
    for f in sorted(glob.glob(str(ROOT / data / 'ep_*.npz')))[:n]:
        fe = ROOT / (data + '_feat') / (Path(f).stem + '.npy')
        if not fe.exists():
            continue
        z = np.load(f, allow_pickle=True)
        eps.append(dict(feat=np.load(fe, mmap_mode='r'), tool=z['tool'], grip=z['grip'], action=z['action'], flip=flip_since(z['grip']),
                        anchor_ad={m: anchor_indices(z['tool'], z['grip'], 'adaptive', move_thresh=m) for m in (0.04, 0.08, 0.12)}))
    return eps


def sample(eps, idx, mode, rng):
    feats, states, deltas, acts = [], [], [], []
    for e, t in idx:
        E = eps[e]; T = len(E['tool'])
        j = np.minimum(np.arange(t, t + H_VLA), T - 1)
        acts.append(np.c_[(E['action'][j, :3] - E['tool'][t]) / POS_SCALE, E['action'][j, 3:4] * 2 - 1])
        a = sample_anchor(E, t, rng, 80) if mode == 'B' else t
        feats.append(np.asarray(E['feat'][a], np.float32))            # image of the anchor frame (A: a = t)
        st = state_vec(E['tool'][t], E['grip'][t], E['flip'][t])         # current state for both
        if mode == 'B':
            st = np.r_[st, anchor_delta(E['tool'][t], E['tool'][a], E['grip'][a], t - a)]
        states.append(st)
    f = lambda x: torch.from_numpy(np.stack(x).astype(np.float32))
    return f(feats), f(states), f(acts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['A', 'B'], required=True)
    ap.add_argument('--data', default='data/v2')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log-every', type=int, default=250)
    ap.add_argument('--init', default=None, help='continue from the trainable weights of this checkpoint')
    a = ap.parse_args()
    dev = 'mps'; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    eps = load(a.data, a.n)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps)); n_val = max(5, len(eps) // 20)
    val_e, tr_e = perm[:n_val], perm[n_val:]
    tr_idx = [(e, t) for e in tr_e for t in range(len(eps[e]['tool']))]
    vrng = np.random.default_rng(1)
    va_idx = [(int(e), int(vrng.integers(len(eps[e]['tool'])))) for e in val_e for _ in range(16)]
    print(f'{len(eps)} episodes ({len(tr_e)} train / {n_val} val), {len(tr_idx)} train frames', flush=True)
    v = VLA(a.mode, dev, ckpt=(ROOT / a.init) if a.init else None)
    params = list(v.trainable().values())
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.025 + 0.975 * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps))))
    dummy = torch.zeros(a.bs, 8, 8, 3, dtype=torch.uint8, device=dev)

    def run(idx, g=None):
        fe, st, ac = (x.to(dev) for x in sample(eps, idx, a.mode, rng))
        if g is not None:                                  # fixed noise / time for a comparable validation loss
            v.pol.model.sample_noise = lambda shape, device: torch.randn(shape, generator=g, device='cpu').to(device)
            v.pol.model.sample_time = lambda bsize, device: (torch.distributions.Beta(1.5, 1.0).sample((bsize,)) * 0.999 + 0.001).to(device)
        return v.loss(dummy[:len(idx)], st, ac, None, feat=fe)

    log = []; t0 = time.time()
    for step in range(1, a.steps + 1):
        v.train()
        sel = rng.integers(0, len(tr_idx), a.bs)
        loss = run([tr_idx[i] for i in sel])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 10.0); opt.step(); sched.step()
        if step % a.log_every == 0 or step == a.steps:
            v.eval(); torch.manual_seed(0); g = torch.Generator().manual_seed(0)
            sn, stm = v.pol.model.sample_noise, v.pol.model.sample_time
            with torch.no_grad():
                vl = float(np.mean([run(va_idx[k:k + a.bs], g).item() for k in range(0, len(va_idx), a.bs)]))
            v.pol.model.sample_noise, v.pol.model.sample_time = sn, stm
            rec = dict(step=step, train_loss=float(loss.item()), val_loss=vl, s_per_step=(time.time() - t0) / step)
            log.append(rec); print(json.dumps(rec), flush=True)
            torch.save(dict(state={k: p.detach().cpu() for k, p in v.trainable().items()}, mode=a.mode, step=step), out / 'last.pt')
            json.dump(log, open(out / 'log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
