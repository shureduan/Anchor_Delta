"""Analysis figures (matplotlib) for the report.

    ../kinova_bottle_sim/.venv/bin/python analysis_figures.py   # -> ../kinova_report/media/figures/fig_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM = ROOT / 'kinova_bottle_sim'
OUT = ROOT / 'kinova_report/media/figures'
DATA = ROOT / 'kinova_report/data'
DEMOS = ROOT / 'delta_wristcam/real/out/demos'
sys.path.insert(0, str(SIM / 'scripts'))
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
CA, CB, CC = '#4c72b0', '#dd8452', '#55a868'


def log(run):
    m = json.load(open(DEMOS / run / 'meta.json')); return m, m['policy_log']


def fig_sim_versions():
    rows = [('v2', 'out/eval_A_dense_v2.json', 'out/eval_B_fixed20_v2.json', 'out/eval_B_adaptive0.08_v2.json'),
            ('v4', 'out/eval_A_dense_v4.json', 'out/eval_B_fixed20_v4.json', 'out/eval_B_adaptive0.08_v4.json'),
            ('v5', 'out/eval_A_dense_v5.json', 'out/eval_B_fixed20_v5.json', 'out/eval_B_adaptive0.08_v5.json')]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(rows)); w = 0.26
    for j, (lab, col) in enumerate([('A dense', CA), ('B K=20', CB), ('B adaptive 8 cm', CC)]):
        sr, ms = [], []
        for r in rows:
            p = SIM / r[1 + j]
            d = json.load(open(p)) if p.exists() else None
            sr.append(100 * d['success'] if d else np.nan); ms.append(d['policy_ms_per_step'] if d else np.nan)
        ax[0].bar(x + (j - 1) * w, sr, w, label=lab, color=col)
        ax[1].bar(x + (j - 1) * w, ms, w, label=lab, color=col)
    ax[0].set_xticks(x, [r[0] for r in rows]); ax[0].set_ylabel('sim success (%)'); ax[0].set_ylim(80, 101); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('Sim closed loop (unseen seeds; n=50 v2/v4, 30 v5)')
    ax[1].set_xticks(x, [r[0] for r in rows]); ax[1].set_ylabel('policy compute (ms / step)'); ax[1].set_title('Compute per control step (M4 GPU)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_sim_versions.png'); plt.close(fig)


def fig_real_phase():
    r = json.load(open(DATA / 'real_phase_all_versions.json'))
    phases = list(r['A2']['phases'])
    short = ['above cap', 'down 4 cm', 'down 2 cm', 'grasp h.', 'lift', 'over box', 'rim -2 cm', 'into box', 'lift away']
    fig, ax = plt.subplots(figsize=(11, 3.4))
    names = ['A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5', 'B5']
    cols = ['#9fb7d9', '#efb995', '#7894c2', '#e09a6c', '#4c72b0', '#dd8452', '#253f6e', '#a4511c']
    w = 0.1; x = np.arange(len(phases))
    for j, n in enumerate(names):
        ax.bar(x + (j - 3.5) * w, [100 * r[n]['phases'][p] for p in phases], w, label=f'{n} ({100 * r[n]["overall"]:.0f}%)', color=cols[j])
    ax.set_xticks(x, short); ax.set_ylabel('direction within 30 deg (%)')
    ax.set_title('Offline sim-to-real check on 2 recorded real demos: does the policy point where the scripted controller went?')
    ax.legend(ncol=8, fontsize=7.5, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout(); fig.savefig(OUT / 'fig_real_phase_versions.png'); plt.close(fig)


def fig_post_grasp():
    """v4 B re-opens when the post-grasp real image becomes the anchor; v5 keeps holding (same layout L3)."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[3, 1.1]))
    for i, (run, title) in enumerate([('20261122_114422', 'v4  B K=20  (L3, failed)'), ('20261122_123258', 'v5  B K=20  (L3, success)')]):
        m, L = log(run); t = np.arange(len(L))
        lg = np.array([np.mean(r['chunk_grip_logit']) for r in L]); gc = np.array([r['grip_cmd'] for r in L])
        z = np.array([r['tool'][2] for r in L]); enc = np.array([r['encoded'] for r in L])
        a = ax[i, 0]; a.plot(t, lg, color='k', lw=1.5, label='close logit (chunk mean)'); a.axhline(0, color='grey', lw=0.8)
        a.fill_between(t, -20, 20, where=gc > 0, color=CB, alpha=0.15, label='gripper commanded closed')
        for s in np.where(enc)[0]:
            a.axvline(s, color='red', lw=0.8, alpha=0.6)
        a2 = a.twinx(); a2.plot(t, z * 100, color=CA, lw=1.2); a2.set_ylabel('tool z (cm)', color=CA)
        a.set_ylim(-15, 20); a.set_xlim(40, max(130, len(L))); a.set_ylabel('logit'); a.set_title(title + '   red = image encode (new anchor)', fontsize=9)
        if i == 0:
            a.legend(fontsize=7.5, frameon=False, loc='lower left')
        J = np.load(DEMOS / run / 'policy_frames.npz', allow_pickle=True)['jpg']
        im = cv2.cvtColor(cv2.imdecode(np.frombuffer(J[80], np.uint8), 1), cv2.COLOR_BGR2RGB)
        ax[i, 1].imshow(im); ax[i, 1].axis('off'); ax[i, 1].set_title(f'anchor image @ step 80 (just grasped)\nnext logit: {lg[82]:+.1f}', fontsize=8.5)
    ax[1, 0].set_xlabel('control step (10 Hz)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_post_grasp_reopen_v4_vs_v5.png'); plt.close(fig)


def fig_grasp_timing():
    """Stationary at the grasp pose: per-chunk-index close logits on real frames (A4 vs A5)."""
    import torch
    from policy import proprio
    from real_offline import load_policy, real_to_sim_grip
    frames = [('20261122_095917', 'policy step 70', 'L1'), ('20261122_101256', 'policy step 64', 'L2'), ('20261122_103602', 'policy step 100', 'L3')]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for j, ck in enumerate(['runs/A4/last.pt', 'runs/A5/last.pt']):
        pol, _ = load_policy(ck, 'mps')
        for (d, lab, name), col in zip(frames, [CA, CC, CB]):
            z = np.load(DEMOS / d / 'episode.npz', allow_pickle=True); k = np.where(z['label'].astype(str) == lab)[0][-1]
            img = cv2.cvtColor(cv2.resize(cv2.imdecode(np.frombuffer(z['color_jpg'][k], np.uint8), 1), (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            tool = z['tool_pose'][k, :3].astype(np.float32); g = np.float32(real_to_sim_grip(z['gripper'][k]))
            with torch.no_grad():
                _, gl = pol(pol.encode(torch.from_numpy(img[None]).to('mps')), torch.from_numpy(proprio(tool, g, flip=np.float32(50))[None]).to('mps'))
            c = gl[0].cpu().numpy()
            ax[j].plot(np.arange(10), c, 'o-', color=col, label=f'{name}: mean {c.mean():+.1f}')
        ax[j].axhline(0, color='grey', lw=0.8); ax[j].set_xlabel('chunk index (steps ahead)'); ax[j].legend(frameon=False, fontsize=8)
        ax[j].set_title(['v4 (expert waited 0.3 s before closing):\n"close in ~6 steps", never counts down', 'v5 (close on arrival):\n"close within 1-2 steps", mean clearly > 0'][j], fontsize=9)
    ax[0].set_ylabel('close logit at the grasp pose (open, still)')
    fig.tight_layout(); fig.savefig(OUT / 'fig_grasp_timing_v4_vs_v5.png'); plt.close(fig)


def fig_real_grid():
    s = json.load(open(DATA / 'real_grid_summary.json'))
    cfg = ['A', 'B K20', 'B adaptive']; lay = ['L1', 'L2', 'L3']
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for j, (key, lab) in enumerate([('encodes', 'image encodes per episode'), ('ms_per_step', 'policy compute (ms/step, idle GPU)'),
                                    ('visual_bytes_stored', 'stored visual data (kB)')]):
        for i, c in enumerate(cfg):
            v = [s[f'{l} {c}'][key] / (1e3 if 'bytes' in key else 1) for l in lay]
            ax[j].bar(np.arange(3) + (i - 1) * 0.27, v, 0.27, color=[CA, CB, CC][i], label=c)
        ax[j].set_xticks(np.arange(3), lay); ax[j].set_title(lab, fontsize=9.5)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Real robot, 9 successful runs (L1/L2: v4 models, L3: v5 models)', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig_real_grid_cost.png'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_sim_versions, fig_real_phase, fig_post_grasp, fig_grasp_timing, fig_real_grid):
        f(); print('ok', f.__name__)


if __name__ == '__main__':
    main()
