"""Per-frame visualisation of one real-robot run: phone video + wrist RGB-D + the policy's algorithm state.

    ../kinova_bottle_sim/.venv/bin/python render_run.py b_l3_r1.MOV            # -> out/b_l3_r1_viz.mp4
    ../kinova_bottle_sim/.venv/bin/python render_run.py b_l3_r1.MOV --still 85  # one PNG at policy step 85

Layout (1920x1080):
  top-left     phone (third person)          top-right   wrist RGB + HUD, picture-in-picture = the image the
                                                          policy is using now (B: the cached anchor frame)
  bottom-left  wrist depth + data accounting  bottom-mid  top-down tool path: anchor, delta arrow, 1 s plan
  bottom-right whole-episode timelines (anchor age, delta components, cumulative image encodes) + cursor

Time base: everything is placed on the robot PC's clock (time.time()): recorder frames have 't', policy steps have
t_frame / t_policy / t_cmd / t_done. The phone video is placed by its creation time, refined by cross-correlating
phone motion energy with the measured tool speed (search +-3 s).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEMOS = ROOT / 'delta_wristcam/real/out/demos'
PHONE = ROOT / 'video_from_phone'
W, H = 1920, 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
CFG_NAME = {('A', 'dense'): 'A  dense: encode every frame',
            ('B', 'fixed'): 'B  anchor + delta: re-encode every K=20 steps',
            ('B', 'adaptive'): 'B  anchor + delta: re-encode on gripper flip / 8 cm move'}
C_ANCH = (0, 165, 255)     # orange (BGR)
C_PLAN = (255, 220, 0)     # cyan
C_ENC = (60, 60, 255)      # red


def txt(img, s, xy, scale=0.6, col=(240, 240, 240), th=1, bg=None):
    if bg is not None:
        (tw, tht), b = cv2.getTextSize(s, FONT, scale, th)
        cv2.rectangle(img, (xy[0] - 4, xy[1] - tht - 4), (xy[0] + tw + 4, xy[1] + b + 2), bg, -1)
    cv2.putText(img, s, xy, FONT, scale, col, th, cv2.LINE_AA)


def load_run(run):
    d = DEMOS / run
    meta = json.load(open(d / 'meta.json')); L = meta['policy_log']
    z = np.load(d / 'episode.npz', allow_pickle=True)
    pf = np.load(d / 'policy_frames.npz', allow_pickle=True)
    S = dict(meta=meta, L=L, t_rec=z['t'], jpg=z['color_jpg'], depth=z['depth'], pose=z['tool_pose'], grip=z['gripper'],
             pjpg=pf['jpg'])
    S['t_pol'] = np.array([r['t_policy'] for r in L]); S['tool'] = np.array([r['tool'] for r in L])
    S['enc'] = np.array([r['encoded'] for r in L]); S['age'] = np.array([r['anchor_age'] for r in L])
    S['anc'] = np.array([r['anchor_step'] for r in L]); S['delta'] = np.array([r['delta'] for r in L])
    S['chunk'] = np.array([r['chunk'] for r in L]); S['logit'] = np.array([np.mean(r['chunk_grip_logit']) for r in L])
    S['gcmd'] = np.array([r['grip_cmd'] for r in L]); S['ms'] = np.array([r['ms'] for r in L])
    S['nbytes'] = np.array([len(b) for b in S['pjpg'][:len(L)]])
    return S


def phone_offset(video, S, v0_clock):
    """Video time of robot clock 0 ... returns (offset_s, corr): video_t = robot_t - v0."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf', 'fps=10,scale=160:90,format=gray', '-f', 'rawvideo', '-'],
                         capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.float32)
    mot = np.r_[0, np.abs(np.diff(fr, axis=0)).mean((1, 2))]; tv = np.arange(len(mot)) / 10.0
    tr, p = S['t_rec'], S['pose'][:, :3]
    sp = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(tr), 1e-3)]
    best = (0.0, -1)
    for lag in np.arange(-3, 3.01, 0.1):
        v0 = v0_clock + lag
        rs = np.interp(tv + v0, tr, sp, left=0, right=0)
        m = (tv + v0 > tr[0]) & (tv + v0 < tr[-1])
        if m.sum() < 50:
            continue
        c = np.corrcoef(mot[m], rs[m])[0, 1]
        if c > best[1]:
            best = (lag, c)
    return v0_clock + best[0], best


def plot_series(img, rect, xs, series, cursor, title, ylab, yr=None, ticks=None, hline=None):
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    txt(img, title, (x0 + 6, y0 + 18), 0.5, (200, 200, 200))
    allv = np.concatenate([s for s, _, _ in series])
    lo, hi = (allv.min(), allv.max()) if yr is None else yr
    if hi - lo < 1e-6:
        hi = lo + 1
    X = lambda i: int(x0 + 8 + (w - 16) * i / max(len(xs) - 1, 1))       # noqa: E731
    Y = lambda v: int(y0 + h - 8 - (h - 34) * (v - lo) / (hi - lo))      # noqa: E731
    if hline is not None:
        cv2.line(img, (x0 + 8, Y(hline)), (x0 + w - 8, Y(hline)), (80, 80, 80), 1)
    if ticks is not None:
        for i in np.where(ticks)[0]:
            cv2.line(img, (X(i), y0 + h - 8), (X(i), y0 + h - 20), C_ENC, 2)
    for s, col, lab in series:
        pts = np.array([[X(i), Y(v)] for i, v in enumerate(s)], np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    k = 0
    for s, col, lab in series:
        if lab:
            txt(img, lab, (x0 + w - 230 + k * 75, y0 + 18), 0.45, col); k += 1
    txt(img, f'{hi:.0f}' if abs(hi) >= 10 else f'{hi:.1f}', (x0 + w - 40, y0 + 34), 0.4, (150, 150, 150))
    txt(img, f'{lo:.0f}' if abs(lo) >= 10 else f'{lo:.1f}', (x0 + w - 40, y0 + h - 10), 0.4, (150, 150, 150))
    if cursor is not None:
        cv2.line(img, (X(cursor), y0 + 24), (X(cursor), y0 + h - 6), (255, 255, 255), 1)


def topdown(img_full, rect, S, k):
    X0, Y0, w, h = rect
    img = np.full((h, w, 3), 18, np.uint8); x0 = y0 = 0          # draw on a clipped canvas
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
    txt(img, 'top view (robot base frame) - tool path, anchor, delta, 1 s plan', (x0 + 6, y0 + 18), 0.5, (200, 200, 200))
    P = S['tool']; allp = np.r_[P[:, :2], S['chunk'][:, :, :2].reshape(-1, 2)]
    c = (allp.max(0) + allp.min(0)) / 2; span = max(np.ptp(allp, 0).max() + 0.06, 0.15)
    sc = (min(w, h) - 60) / span
    # screen: up = +x (away from the base), left = +y
    Q = lambda p: (int(x0 + w / 2 - (p[1] - c[1]) * sc), int(y0 + 30 + (h - 30) / 2 - (p[0] - c[0]) * sc))   # noqa: E731
    for g in np.arange(-0.3, 0.31, 0.05):                       # 5 cm grid
        a, b = Q(c + [g, -1]), Q(c + [g, 1]); cv2.line(img, (x0 + 2, a[1]), (x0 + w - 2, a[1]), (40, 40, 40), 1)
        a = Q(c + [-1, g]); cv2.line(img, (a[0], y0 + 26), (a[0], y0 + h - 2), (40, 40, 40), 1)
    txt(img, 'grid 5 cm   up = away from robot', (x0 + 6, y0 + h - 8), 0.4, (130, 130, 130))
    pts = np.array([Q(p) for p in P[:k + 1, :2]], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (150, 150, 150), 2, cv2.LINE_AA)
    a = S['anc'][k]; pa, pn = Q(P[a, :2]), Q(P[k, :2])
    cv2.rectangle(img, (pa[0] - 7, pa[1] - 7), (pa[0] + 7, pa[1] + 7), C_ANCH, 2)
    cv2.arrowedLine(img, pa, pn, C_ANCH, 2, cv2.LINE_AA, tipLength=0.25) if np.hypot(pn[0] - pa[0], pn[1] - pa[1]) > 4 else None
    for j, q in enumerate(S['chunk'][k]):
        cv2.circle(img, Q(q[:2]), 3, tuple(int(v * (1 - 0.06 * j)) for v in C_PLAN), -1, cv2.LINE_AA)
    cv2.circle(img, pn, 7, (255, 255, 255), -1, cv2.LINE_AA)
    # z gauge (side bar)
    zx = x0 + w - 70; zlo, zhi = 0.12, 0.44
    Z = lambda z: int(y0 + h - 30 - (h - 80) * (z - zlo) / (zhi - zlo))   # noqa: E731
    cv2.line(img, (zx, Z(zlo)), (zx, Z(zhi)), (90, 90, 90), 2)
    txt(img, 'z', (zx - 4, Z(zhi) - 8), 0.45, (180, 180, 180))
    cv2.line(img, (zx - 10, Z(S['tool'][a, 2])), (zx + 10, Z(S['tool'][a, 2])), C_ANCH, 2)
    for q in S['chunk'][k]:
        cv2.circle(img, (zx + 18, Z(q[2])), 2, C_PLAN, -1)
    cv2.circle(img, (zx, Z(P[k, 2])), 6, (255, 255, 255), -1)
    txt(img, f'{P[k, 2]:.3f} m', (zx - 60, Z(P[k, 2]) + 5), 0.4, (220, 220, 220))
    # legend
    ly = y0 + 40
    for lab, col in [('current tool', (255, 255, 255)), ('anchor (last image encode)', C_ANCH), ('delta = tool - anchor', C_ANCH),
                     ('policy plan, next 1 s', C_PLAN)]:
        cv2.circle(img, (x0 + 14, ly - 4), 4, col, -1); txt(img, lab, (x0 + 24, ly), 0.42, col); ly += 18
    img_full[Y0:Y0 + h, X0:X0 + w] = img


def depth_vis(d16):
    d = d16.astype(np.float32) / 1000.0
    v = np.clip((d - 0.12) / (0.45 - 0.12), 0, 1)
    c = cv2.applyColorMap((255 * (1 - v)).astype(np.uint8), cv2.COLORMAP_TURBO)
    c[d16 == 0] = 0
    return c


def render(video, run, out, still=None, fps=30, mode=None):
    S = load_run(run); m = S['meta']; L = S['L']
    pol, rule = m['policy'], m['rule']; ck = m.get('policy_server', {}).get('ckpt', '')
    ver = 'v5' if '5' in Path(ck).parent.name else 'v4'
    import datetime
    c = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format_tags=com.apple.quicktime.creationdate', '-of', 'csv=p=0', str(video)],
                       capture_output=True, text=True).stdout.strip()
    v0c = datetime.datetime.strptime(c, '%Y-%m-%dT%H:%M:%S%z').timestamp()
    v0, (lag, corr) = phone_offset(video, S, v0c)
    print(f'{video.name}: phone sync lag {lag:+.1f} s vs creation time (motion corr {corr:.2f})')
    T0, T1 = m['t_start'] - 1.0, S['t_pol'][-1] + 2.0
    n_out = int((T1 - T0) * fps)
    times = T0 + np.arange(n_out) / fps
    if still is not None:
        times = np.array([S['t_pol'][still] + 0.05])
    # phone frames, streamed
    ss = max(times[0] - v0, 0)
    ph = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', f'{ss:.3f}', '-i', str(video), '-vf', f'fps={fps},scale=960:540',
                           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    wr = None
    if still is None:
        wr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'veryfast', str(out)], stdin=subprocess.PIPE)
    steps = len(L); dense_bytes = np.cumsum(S['nbytes']); store_bytes = np.cumsum(np.where(S['enc'], S['nbytes'], 0) + (20 if pol == 'B' else 0))
    enc_cum = np.cumsum(S['enc']); xs = np.arange(steps)
    last_enc_t = -1e9; ph_frame = np.zeros((540, 960, 3), np.uint8)
    for T in times:
        buf = ph.stdout.read(960 * 540 * 3)
        if len(buf) == 960 * 540 * 3:
            ph_frame = np.frombuffer(buf, np.uint8).reshape(540, 960, 3)
        k = int(np.searchsorted(S['t_pol'], T, 'right') - 1)        # policy step whose decision is active at T
        started = k >= 0; k = max(k, 0)
        i = int(np.clip(np.searchsorted(S['t_rec'], T, 'right') - 1, 0, len(S['t_rec']) - 1))
        img = np.full((H, W, 3), 18, np.uint8)
        # ---- phone
        img[0:540, 0:960] = ph_frame
        txt(img, f'{m["note"].split()[0]}   {CFG_NAME[(pol, rule)]}   [{ver}]   result: {"SUCCESS" if m.get("success") else "fail"}',
            (12, 30), 0.62, (255, 255, 255), 2, bg=(0, 0, 0))
        # ---- wrist RGB + HUD
        wrist = cv2.resize(cv2.imdecode(np.frombuffer(S['jpg'][i], np.uint8), 1), (960, 540), interpolation=cv2.INTER_LINEAR)
        img[0:540, 960:1920] = wrist
        if started and S['enc'][k]:
            last_enc_t = S['t_pol'][k]
        flash = T - last_enc_t < 0.35
        txt(img, f'wrist camera   step {k:3d}' if started else 'wrist camera   (waiting for start)', (972, 30), 0.62, (255, 255, 255), 2, bg=(0, 0, 0))
        if started:
            r = L[k]
            txt(img, f'gripper {"CLOSED" if r["grip_cmd"] else "open"}  ({r["grip_pct"]:.0f} %)   close-logit {S["logit"][k]:+.1f}',
                (972, 520), 0.55, (255, 255, 255), 1, bg=(0, 0, 0))
            dl = S['delta'][k] * 100
            txt(img, f'delta = tool - anchor = ({dl[0]:+5.1f}, {dl[1]:+5.1f}, {dl[2]:+5.1f}) cm   |{np.linalg.norm(dl):4.1f}| cm',
                (972, 492), 0.55, C_ANCH, 1, bg=(0, 0, 0))
            txt(img, f'anchor age {S["age"][k]:2d} steps ({S["age"][k] / 10:.1f} s)   image encodes {enc_cum[k]}/{k + 1}',
                (972, 464), 0.55, (255, 255, 255), 1, bg=(0, 0, 0))
            # PiP: what the policy's image encoder output comes from
            a = S['anc'][k]
            pip = cv2.resize(cv2.imdecode(np.frombuffer(S['pjpg'][a], np.uint8), 1), (320, 180))
            age = S['age'][k]; col = C_ENC if flash else tuple(int(v) for v in np.array(C_ANCH) * max(0.35, 1 - age / 40))
            cv2.rectangle(pip, (0, 0), (319, 179), col, 4)
            img[48:228, 1588:1908] = pip
            txt(img, 'policy image now: ' + ('current frame' if pol == 'A' else f'anchor @ step {a}'), (1590, 246), 0.48, col, 1, bg=(0, 0, 0))
            if flash:
                txt(img, 'ENCODE', (1700, 285), 0.9, (255, 255, 255), 2, bg=C_ENC)
        # ---- depth + accounting
        img[540:900, 0:640] = cv2.resize(depth_vis(S['depth'][i]), (640, 360), interpolation=cv2.INTER_NEAREST)
        txt(img, 'wrist depth (red near 0.12 m .. blue far 0.45 m, black = no data)', (10, 562), 0.5, (255, 255, 255), 1, bg=(0, 0, 0))
        if started:
            y = 930
            for s_, col in [(f'image encoder calls : {enc_cum[k]:4d} / {k + 1:4d} steps ({100 * enc_cum[k] / (k + 1):5.1f} %)', (255, 255, 255)),
                            (f'stored visual data   : {store_bytes[k] / 1e3:7.1f} kB  (dense video: {dense_bytes[k] / 1e3:7.1f} kB)', (255, 255, 255)),
                            (f'policy compute (meas.): {S["ms"][:k + 1].mean():5.1f} ms/step', (200, 200, 200))]:
                txt(img, s_, (10, y), 0.52, col); y += 30
            txt(img, 'B stores only anchor frames + 5 floats/step (delta from robot FK + gripper sensor)' if pol == 'B'
                else 'A stores and encodes every frame', (10, y + 4), 0.45, (150, 150, 150))
        # ---- top view
        topdown(img, (650, 545, 620, 530), S, k)
        # ---- timelines
        cur = k if started else None
        plot_series(img, (1280, 545, 632, 170), xs, [(S['age'].astype(float), C_ANCH, 'anchor age')], cur, 'anchor age (steps) - red ticks = image encodes',
                    '', ticks=S['enc'])
        plot_series(img, (1280, 720, 632, 175), xs, [(S['delta'][:, 0] * 100, (90, 90, 255), 'dx'), (S['delta'][:, 1] * 100, (90, 255, 90), 'dy'),
                                                    (S['delta'][:, 2] * 100, (255, 160, 90), 'dz')], cur, 'delta = tool - anchor (cm), from robot FK', '', hline=0)
        plot_series(img, (1280, 900, 632, 175), xs, [(np.arange(1, steps + 1, dtype=float), (120, 120, 120), 'dense A'),
                                                    (enc_cum.astype(float), C_ENC, 'this run')], cur, 'cumulative image encodes', '')
        if still is not None:
            cv2.imwrite(str(out), img); break
        wr.stdin.write(img.tobytes())
    ph.stdout.close(); ph.wait()
    if wr is not None:
        wr.stdin.close(); wr.wait()
    print('->', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('--still', type=int, default=None); ap.add_argument('--fps', type=int, default=30)
    a = ap.parse_args()
    man = json.load(open(HERE / 'sync_manifest.json'))
    run = man[a.video]['run']
    (HERE / 'out').mkdir(exist_ok=True)
    stem = Path(a.video).stem
    out = HERE / 'out' / (f'{stem}_step{a.still}.png' if a.still is not None else f'{stem}_viz.mp4')
    render(PHONE / a.video, run, out, still=a.still, fps=a.fps)


if __name__ == '__main__':
    main()
