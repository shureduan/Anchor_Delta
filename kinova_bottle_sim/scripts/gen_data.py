"""Generate expert demonstrations with domain randomisation, in parallel.

    python scripts/gen_data.py --n 300 --out data/v1 --workers 6

Each worker renders the wrist camera at 256x144 and stores 128x72 JPEG frames (INTER_AREA, like the real
1280x720 frames will be downscaled). Per episode: data/v1/ep_XXXXX.npz with rgb_jpg (T,), tool (T,3), grip (T,),
action (T,4) = [tool target xyz, gripper cmd], phase (T,), plus meta. Only successful episodes are kept
(failed seeds are listed in meta.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
OUT_W, OUT_H = 128, 72
NOISES = (0.0, 0.005, 0.01)


def work(args):
    seed, out, (OUT_W, OUT_H), look, seg, observable = args
    import cv2
    from expert import run_episode
    from scene import Sim
    global _sim
    if '_sim' not in globals():
        _sim = Sim(256, 144, look=look)
    _sim.record_seg = seg
    rng = np.random.default_rng(seed)
    noise = NOISES[seed % len(NOISES)]
    r = run_episode(_sim, rng, render=True, noise=noise, observable=observable)
    if not r['success']:
        return seed, False, len(r['action'])
    jpg = [cv2.imencode('.jpg', cv2.cvtColor(cv2.resize(im, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for im in r['rgb']]
    np.savez(Path(out) / f'ep_{seed:05d}.npz', rgb_jpg=np.array(jpg, dtype=object), tool=np.array(r['tool'], np.float32),
             grip=np.array(r['grip'], np.float32), action=np.array(r['action'], np.float32), phase=np.array(r['phase'], np.int8),
             scene=json.dumps(r['scene']),
             **(dict(seg=np.stack([cv2.resize(m, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST) for m in r['seg']])) if seg else {}))
    return seed, True, len(r['action'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--out', default='data/v1')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--look', default='v2', help="scene appearance: 'v2' or 'real' (scene.LOOKS)")
    ap.add_argument('--seg', action='store_true', help='also store class maps (T,H,W) uint8 for v4 training-time randomisation')
    ap.add_argument('--observable', action='store_true', help='v5 expert: no hidden waits before gripper changes')
    ap.add_argument('--size', type=int, nargs=2, default=[OUT_W, OUT_H], help='stored frame size W H (render is 256x144)')
    a = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / a.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); ok, fail, steps = 0, [], 0
    with Pool(a.workers) as pool:
        for i, (seed, good, n) in enumerate(pool.imap_unordered(work, [(s, str(out), tuple(a.size), a.look, a.seg, a.observable) for s in range(a.seed0, a.seed0 + a.n)])):
            ok += good; steps += n * good
            if not good:
                fail.append(seed)
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{a.n} episodes, {ok} kept, {time.time() - t0:.0f}s', flush=True)
    meta = dict(n_requested=a.n, n_kept=ok, failed_seeds=fail, steps=steps, image=a.size, look=a.look, seg=a.seg, observable=a.observable, ctrl_hz=10, noises=NOISES,
                seconds=time.time() - t0)
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    print(meta)


if __name__ == '__main__':
    main()
