"""Train policy A (dense image) or B (anchor image + proprio delta) on data/v1, on the Mac (MPS).

    python scripts/train.py --mode A --steps 15000 --out runs/A
    python scripts/train.py --mode B --steps 15000 --out runs/B

B is trained with mixed anchor schedules so one model serves every rule: 50 % anchors from the adaptive rule,
50 % a random anchor age k ~ U[0, 30] steps. Image augmentation (random shift +-6 px, brightness/contrast/
saturation jitter) on the encoded frame. Loss: L1 on chunk offsets + 0.1 BCE on gripper commands.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from policy import H, POS_SCALE, Policy, anchor_indices, flip_since, proprio  # noqa: E402
import dr_aug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _small(bgr):
    """Policy input is 128x72; data stored larger (e.g. 256x144 for SmolVLA) is downscaled like the sim frames."""
    return bgr if bgr.shape[1] == 128 else cv2.resize(bgr, (128, 72), interpolation=cv2.INTER_AREA)


def load(data_dir):
    eps = []
    for f in sorted(glob.glob(str(ROOT / data_dir / 'ep_*.npz'))):
        z = np.load(f, allow_pickle=True)
        img = np.stack([cv2.cvtColor(_small(cv2.imdecode(np.frombuffer(b, np.uint8), 1)), cv2.COLOR_BGR2RGB) for b in z['rgb_jpg']])
        eps.append(dict(img=img, seg=z['seg'] if 'seg' in z.files else None, tool=z['tool'], grip=z['grip'], action=z['action'], phase=z['phase'],
                        flip=flip_since(z['grip']),
                        anchor_ad={m: anchor_indices(z['tool'], z['grip'], 'adaptive', move_thresh=m) for m in (0.04, 0.08, 0.12)}))
    return eps


def sample_anchor(E, t, rng, max_age):
    """Anchor for step t, mixed so that B sees every schedule it may be run with and the anchor age is not tied to
    any phase: 1/3 adaptive (move threshold 4 / 8 / 12 cm), 1/3 fixed K ~ U{1..60} (K = 1 is dense), 1/3 a random age
    U[0, max_age]."""
    u = rng.random()
    if u < 1 / 3:
        return int(E['anchor_ad'][(0.04, 0.08, 0.12)[rng.integers(3)]][t])
    if u < 2 / 3:
        K = int(rng.integers(1, 61)); return t - t % K
    return max(0, t - int(rng.integers(0, max_age + 1)))


def batch(eps, idx, mode, rng, flip=True, max_age=80):
    """idx: list of (episode, t). Returns tensors (numpy) for one batch."""
    imgs, segs, props, offs, grips = [], [], [], [], []
    for e, t in idx:
        E = eps[e]; T = len(E['tool']); fl = E['flip'][t] if flip else None
        if mode == 'B':
            a = sample_anchor(E, t, rng, max_age)
            imgs.append(E['img'][a]); segs.append(E['seg'][a] if E['seg'] is not None else None)
            props.append(proprio(E['tool'][t], E['grip'][t], E['tool'][a], np.float32(E['grip'][a] > 0.2), t - a, flip=fl))
        else:
            imgs.append(E['img'][t]); segs.append(E['seg'][t] if E['seg'] is not None else None)
            props.append(proprio(E['tool'][t], E['grip'][t], flip=fl))
        j = np.minimum(np.arange(t, t + H), T - 1)
        offs.append((E['action'][j, :3] - E['tool'][t]) / POS_SCALE)
        grips.append(E['action'][j, 3])
    seg = np.stack(segs) if all(m is not None for m in segs) else None
    return np.stack(imgs), np.stack(props), np.stack(offs).astype(np.float32), np.stack(grips).astype(np.float32), seg


def augment(x, rng):
    """x: (B, H, W, 3) uint8 torch on device -> augmented uint8."""
    B, Hh, W, _ = x.shape
    x = x.permute(0, 3, 1, 2).float()
    pad = 6
    xp = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    dx, dy = rng.integers(0, 2 * pad + 1, B), rng.integers(0, 2 * pad + 1, B)
    x = torch.stack([xp[i, :, dy[i]:dy[i] + Hh, dx[i]:dx[i] + W] for i in range(B)])
    dev = x.device
    br = torch.tensor(rng.uniform(0.75, 1.25, B), dtype=torch.float32, device=dev).view(B, 1, 1, 1)
    ct = torch.tensor(rng.uniform(0.75, 1.25, B), dtype=torch.float32, device=dev).view(B, 1, 1, 1)
    sa = torch.tensor(rng.uniform(0.7, 1.3, B), dtype=torch.float32, device=dev).view(B, 1, 1, 1)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * ct + m
    g = x.mean(1, keepdim=True)
    x = (x - g) * sa + g
    x = (x * br).clamp(0, 255)
    return x.permute(0, 2, 3, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['A', 'B'], required=True)
    ap.add_argument('--data', default='data/v1')
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-age', type=int, default=80, help='B: largest random anchor age in training (steps)')
    ap.add_argument('--dr', action='store_true', help='v4: class-map driven background / object randomisation (needs gen_data --seg)')
    ap.add_argument('--dr5', action='store_true', help='v5: --dr plus finger erasing (p 0.5) and bottle textures (p 0.5)')
    ap.add_argument('--freeze-enc', action='store_true', help='keep the ImageNet ResNet18 encoder frozen (BN in eval); train proprio + head')
    a = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = load(a.data)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps)); n_val = max(5, len(eps) // 10)
    val_e, tr_e = set(perm[:n_val].tolist()), perm[n_val:]
    tr_idx = [(e, t) for e in tr_e for t in range(len(eps[e]['tool']))]
    va_idx = [(e, t) for e in sorted(val_e) for t in range(0, len(eps[e]['tool']), 3)]
    print(f'{len(eps)} episodes ({len(tr_e)} train / {n_val} val), {len(tr_idx)} train frames, loaded in {time.time() - t0:.0f}s', flush=True)
    pol = Policy(a.mode, flip=True).to(dev)
    if a.freeze_enc:
        pol.enc.requires_grad_(False)
    a.dr = a.dr or a.dr5
    bank = dr_aug.TextureBank(2048, np.random.default_rng(1), dev) if a.dr else None
    if a.dr and any(E['seg'] is None for E in eps):
        sys.exit('--dr needs class maps: generate the data with gen_data.py --seg')
    opt = torch.optim.AdamW([p for p in pol.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    log = []

    def run(idx, train):
        im, pr, of, gr, sg = batch(eps, idx, a.mode, rng, max_age=a.max_age)
        im = torch.from_numpy(im).to(dev)
        if train:
            if a.dr:
                im = dr_aug.randomize(im, torch.from_numpy(sg).to(dev), bank, rng,
                                      **(dict(finger_erase=0.5, bottle_tex=0.5) if a.dr5 else {}))
            im = augment(im, rng)
        feat = pol.encode(im)
        po, gl = pol(feat, torch.from_numpy(pr).to(dev))
        of, gr = torch.from_numpy(of).to(dev), torch.from_numpy(gr).to(dev)
        l_pos = F.l1_loss(po, of); l_g = F.binary_cross_entropy_with_logits(gl, gr)
        return l_pos, l_g, ((torch.sigmoid(gl) > 0.5).float() == gr).float().mean()

    t0 = time.time()
    for step in range(1, a.steps + 1):
        pol.train()
        if a.freeze_enc:
            pol.enc.eval()
        sel = rng.integers(0, len(tr_idx), a.bs)
        l_pos, l_g, _ = run([tr_idx[i] for i in sel], True)
        loss = l_pos + 0.1 * l_g
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps:
            pol.eval()
            with torch.no_grad():
                vs = [run(va_idx[i:i + 256], False) for i in range(0, len(va_idx), 256)]
            vp = float(np.mean([v[0].item() for v in vs])); vg = float(np.mean([v[2].item() for v in vs]))
            rec = dict(step=step, train_pos=float(l_pos.item()), train_grip=float(l_g.item()), val_pos_mm=vp * POS_SCALE * 1000,
                       val_grip_acc=vg, s_per_step=(time.time() - t0) / step)
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(dict(state=pol.state_dict(), mode=a.mode, step=step, flip=True, max_age=a.max_age, dr=a.dr, dr5=a.dr5, freeze_enc=a.freeze_enc), out / 'last.pt')
            json.dump(log, open(out / 'log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
