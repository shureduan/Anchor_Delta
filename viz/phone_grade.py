"""Uniform colour grading of the phone videos (some were shot through a smudged lens).

Per video (one fixed set of parameters for the whole clip, so nothing flickers):
  1. Reinhard colour transfer in LAB to a reference clip (the cleanest one, a_l3_r1): match the mean and std of
     L, a and b. This equalises brightness, contrast and white balance, and removes most of the haze veil.
  2. Mild local contrast (CLAHE on L), saturation boost, and an unsharp mask that is stronger for softer clips.

    ../kinova_bottle_sim/.venv/bin/python phone_grade.py            # out/<v>_phone.mp4 -> out/<v>_phone_graded.mp4 + GIF
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
VIDS = ['a_l1_r1', 'b_l1_r1', 'c_l1_r1', 'a_l2_r1', 'b_l2_r1', 'c_l2_r1', 'a_l3_r1', 'b_l3_r1', 'c_l3_r1']
REF = 'a_l3_r1'
SAT = 1.18                 # global saturation boost
CLAHE = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))


def sample(path, n=24):
    cap = cv2.VideoCapture(str(path)); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for k in np.linspace(0, N - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if ok:
            fr.append(f)
    return fr


def stats(frames):
    lab = np.concatenate([cv2.cvtColor(f, cv2.COLOR_BGR2LAB).reshape(-1, 3) for f in frames]).astype(np.float32)
    sharp = float(np.median([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for f in frames]))
    return dict(mean=lab.mean(0).tolist(), std=lab.std(0).tolist(), sharp=sharp)


def params_all():
    p = {v: stats(sample(HERE / 'out' / f'{v}_phone.mp4')) for v in VIDS}
    ref = p[REF]
    for v in VIDS:
        p[v]['ref_mean'], p[v]['ref_std'] = ref['mean'], ref['std']
        p[v]['sharpen'] = float(np.clip(0.55 * np.sqrt(ref['sharp'] / p[v]['sharp']), 0.55, 1.1))
    json.dump(p, open(HERE / 'phone_grade_params.json', 'w'), indent=1)
    return p


def grade(bgr, q):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    m, s, rm, rs = (np.array(q[k], np.float32) for k in ('mean', 'std', 'ref_mean', 'ref_std'))
    lab = (lab - m) / np.maximum(s, 1e-3) * rs + rm
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    lab[..., 0] = CLAHE.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * SAT, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.4)
    return cv2.addWeighted(out, 1 + q['sharpen'], blur, -q['sharpen'], 0)


def process(v, q):
    src = HERE / 'out' / f'{v}_phone.mp4'; dst = HERE / 'out' / f'{v}_phone_graded.mp4'
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS); w, h = int(cap.get(3)), int(cap.get(4))
    wr = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps}', '-i', '-',
                           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'veryfast', str(dst)], stdin=subprocess.PIPE)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        wr.stdin.write(grade(f, q).tobytes())
    wr.stdin.close(); wr.wait()
    gif = HERE.parent / 'kinova_report/media/gif' / f'{v}_phone_15x.gif'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(dst), '-vf',
                    'setpts=PTS/15,fps=12,scale=360:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192:stats_mode=diff[p];'
                    '[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle', str(gif)])
    print(v, 'sharpen', round(q['sharpen'], 2), '->', gif.name, flush=True)


def main():
    p = params_all()
    if 'preview' in sys.argv:
        tiles = []
        for v in VIDS:
            f = sample(HERE / 'out' / f'{v}_phone.mp4', 3)[1]
            tiles.append(np.vstack([cv2.resize(f, (270, 480)), cv2.resize(grade(f, p[v]), (270, 480))]))
        cv2.imwrite(sys.argv[-1], np.hstack(tiles)); return
    for v in VIDS:
        process(v, p[v])


if __name__ == '__main__':
    main()
