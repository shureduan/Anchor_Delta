import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from scene import Sim
from expert import run_episode
sim = Sim(160, 90); rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ok = 0; t0 = time.time()
for i in range(n):
    r = run_episode(sim, rng, render=False, noise=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
    ok += r['success']
    print(i, 'success' if r['success'] else 'FAIL', r['info'], 'steps', len(r['action']), 'grip max', round(max(r['grip']), 3))
print(f'{ok}/{n} in {time.time() - t0:.1f}s')
