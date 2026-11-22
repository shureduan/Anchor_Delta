"""Pre-compute the frozen SmolVLA image features (SigLIP + connector, 64 x 960 per frame, fp16) of every frame of the
first N episodes, so fine-tuning only runs the VLM layers and the action expert.

    .venv-vla/bin/python scripts/cache_feats.py --data data/v2 --n 200
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla import VLA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/v2')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--bs', type=int, default=32)
    a = ap.parse_args()
    out = ROOT / (a.data + '_feat'); out.mkdir(exist_ok=True)
    v = VLA('A', 'mps').eval()
    files = sorted(glob.glob(str(ROOT / a.data / 'ep_*.npz')))[:a.n]
    t0 = time.time(); nf = 0
    for i, f in enumerate(files):
        dst = out / (Path(f).stem + '.npy')
        if dst.exists():
            continue
        z = np.load(f, allow_pickle=True)
        img = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB) for b in z['rgb_jpg']])
        feats = [v.image_features(torch.from_numpy(img[k:k + a.bs]).to('mps')).half().cpu().numpy() for k in range(0, len(img), a.bs)]
        np.save(dst, np.concatenate(feats)); nf += len(img)
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(files)} episodes, {nf} frames, {(time.time() - t0) / max(nf, 1) * 1e3:.0f} ms/frame', flush=True)


if __name__ == '__main__':
    main()
