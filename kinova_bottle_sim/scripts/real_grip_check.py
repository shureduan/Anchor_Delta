"""Gripper decisions on REAL frames from the 2026-11-22 lab runs (offline): for a policy, the mean chunk close-logit
(the value a stationary uniform ensemble converges to) at
  pre-close : arm stationary at the grasp height, gripper open      -> should be clearly > 0 (close now)
  post-close: just closed on the cap (sensor ~54 %), flip 0..8      -> should be clearly > 0 (keep holding)
    .venv/bin/python scripts/real_grip_check.py runs/A4/last.pt runs/A5/last.pt
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from policy import proprio                       # noqa: E402
from real_offline import load_policy, real_to_sim_grip   # noqa: E402
R = Path(__file__).resolve().parents[2] / 'delta_wristcam/real/out/demos'
PRE = [('20261122_095917', 'policy step 70'), ('20261122_101256', 'policy step 64'), ('20261122_103602', 'policy step 100'),
       ('20261122_103602', 'policy step 150')]
POST = [('20261122_105054', 74), ('20261122_105054', 81), ('20261122_105054', 85)]


def rec_frame(d, label):
    z = np.load(R / d / 'episode.npz', allow_pickle=True); k = np.where(z['label'].astype(str) == label)[0][-1]
    im = cv2.imdecode(np.frombuffer(z['color_jpg'][k], np.uint8), 1)
    return im, z['tool_pose'][k, :3].astype(np.float32), float(z['gripper'][k])


def pol_frame(d, t):
    L = json.load(open(R / d / 'meta.json'))['policy_log']; J = np.load(R / d / 'policy_frames.npz', allow_pickle=True)['jpg']
    return cv2.imdecode(np.frombuffer(J[t], np.uint8), 1), np.array(L[t]['tool'], np.float32), float(L[t]['grip_pct'])


def main():
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    for ck in sys.argv[1:]:
        pol, c = load_policy(ck, dev)

        def m(bgr, tool, gpct, fl):
            img = cv2.cvtColor(cv2.resize(bgr, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            g = np.float32(real_to_sim_grip(gpct))
            ex = {} if c['mode'] == 'A' else dict(tool_anchor=tool, grip_anchor=np.float32(g > 0.2), since=np.float32(0))
            with torch.no_grad():
                _, gl = pol(pol.encode(torch.from_numpy(img[None]).to(dev)), torch.from_numpy(proprio(tool, g, flip=np.float32(fl), **ex)[None]).to(dev))
            return float(gl[0].mean())
        pre = [m(*rec_frame(d, l), 50) for d, l in PRE]
        post = [m(*pol_frame(d, t), fl) for d, t in POST for fl in (0, 3, 8)]
        print(f'{ck:22s} pre-close (want >0): {np.round(pre, 1)}   post-close (want >0): {np.round(post, 1)}')


if __name__ == '__main__':
    main()
