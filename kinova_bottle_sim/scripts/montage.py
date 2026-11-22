import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from scene import Sim
from expert import run_episode, PHASES
sim = Sim(320, 180); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
r = run_episode(sim, rng, render=True)
idx = np.linspace(0, len(r['rgb']) - 1, 12).astype(int)
tiles = []
for i in idx:
    im = cv2.cvtColor(r['rgb'][i], cv2.COLOR_RGB2BGR).copy()
    cv2.putText(im, f"t{i} {PHASES[r['phase'][i]]} z{r['tool'][i][2]:.2f}", (5, 15), 0, 0.45, (0, 0, 255), 1)
    tiles.append(im)
cv2.imwrite('out/montage.jpg', np.concatenate([np.concatenate(tiles[k:k + 4], 1) for k in range(0, 12, 4)], 0))
print('success', r['success'], r['info'])
