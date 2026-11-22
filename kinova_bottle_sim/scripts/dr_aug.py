"""v4 training-time appearance randomisation, driven by the sim class maps (scene.Sim.render_seg; never a policy input).

Why: the real-demo region-swap test (2026-11-17) showed the sim->real gap is NOT the box colour but the table /
background (white glossy table, glare, a yellow cable) and the held bottle's look; each alone breaks the transfer
phase. So per training sample (on the device, after the batch is gathered, before the usual shift / colour jitter):

  background (table, floor, nothing): replaced by a random texture from a bank (plain / tinted / any colour, low-
      frequency mottling, fine grain, scratches, glare blobs, cables, spots, a dark half-plane like a table edge),
      multiplied by the sim's own smoothed shading so the gripper / bottle shadows stay;
  bottle: per-channel gain, brightness, optional specular blob;   box: milder gain;   gripper: brightness;
  then (whole image) blur, Gaussian noise and small random cut-outs.

    bank = TextureBank(1024, rng); x = randomize(x_u8, seg_u8, bank, rng)        # (B, H, W, 3) / (B, H, W) on device
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

H, W = 72, 128


def _texture(rng):
    u = rng.random()
    if u < 0.65:                                             # neutral / slightly tinted light tables (most real labs)
        base = rng.uniform(140, 245) + rng.normal(0, 6, 3)
    elif u < 0.85:                                           # wood / grey / dark tables
        base = np.array([rng.uniform(60, 200)] * 3) * np.array([1.0, rng.uniform(0.8, 1.0), rng.uniform(0.6, 1.0)])
    else:                                                    # anything
        base = rng.uniform(30, 245, 3)
    t = np.ones((H, W, 3), np.float32) * base
    gh, gw = rng.integers(2, 10), rng.integers(3, 16)        # low-frequency mottling
    mot = rng.normal(0, 1, (gh, gw, 1)) + 0.25 * rng.normal(0, 1, (gh, gw, 3))     # mostly brightness, a little colour
    t += cv2.resize(mot.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC) * rng.uniform(0, 25)
    g = rng.normal(0, rng.uniform(0, 10), (H, W)).astype(np.float32)      # fine grain (correlated across channels)
    t += cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.0))[..., None]
    for _ in range(rng.integers(0, 6)):                      # scratches / marks
        p = rng.integers(0, [W, H], 2); q = p + rng.normal(0, 25, 2).astype(int)
        cv2.line(t, tuple(int(v) for v in p), tuple(int(v) for v in q), tuple(float(v) for v in base + rng.normal(0, 25, 3)), 1)
    if rng.random() < 0.25:                                  # dark half-plane (table edge / other furniture)
        a = rng.uniform(0, 2 * np.pi); c = rng.uniform([0, 0], [W, H]); yy, xx = np.mgrid[0:H, 0:W]
        side = (xx - c[0]) * np.cos(a) + (yy - c[1]) * np.sin(a) > rng.uniform(20, 60)
        t[side] = rng.uniform(10, 120, 3) + rng.normal(0, 5, (int(side.sum()), 3))
    for _ in range(rng.integers(0, 3)):                      # glare (soft bright blobs, sometimes elongated)
        c = rng.uniform([-10, -10], [W + 10, H + 10]); sx, sy = rng.uniform(4, 40, 2); yy, xx = np.mgrid[0:H, 0:W]
        t += (rng.uniform(15, 90) * np.exp(-0.5 * (((xx - c[0]) / sx) ** 2 + ((yy - c[1]) / sy) ** 2)))[..., None]
    if rng.random() < 0.45:                                  # cables: random Bezier curves, mostly yellow / black / white
        for _ in range(rng.integers(1, 3)):
            col = [(230, 200, 40), (20, 20, 20), (235, 235, 235), (200, 60, 40), (40, 80, 200)][rng.integers(5)]   # RGB
            P = rng.uniform([-20, -20], [W + 20, H + 20], (4, 2)); s = np.linspace(0, 1, 40)[:, None]
            B = (1 - s) ** 3 * P[0] + 3 * (1 - s) ** 2 * s * P[1] + 3 * (1 - s) * s ** 2 * P[2] + s ** 3 * P[3]
            cv2.polylines(t, [B.astype(np.int32)], False, tuple(float(v) for v in np.array(col) + rng.normal(0, 15, 3)), int(rng.integers(1, 4)))
    for _ in range(rng.integers(0, 4) if rng.random() < 0.3 else 0):   # small spots / stains
        c = tuple(int(v) for v in rng.integers(0, [W, H])); ax = tuple(int(v) for v in rng.integers(1, 5, 2))
        cv2.ellipse(t, c, ax, float(rng.uniform(0, 180)), 0, 360, tuple(float(v) for v in base * rng.uniform(0.3, 0.9)), -1)
    return np.clip(t, 0, 255).astype(np.float32)


class TextureBank:
    def __init__(self, n, rng, device='cpu'):
        self.t = torch.from_numpy(np.stack([_texture(rng) for _ in range(n)])).permute(0, 3, 1, 2).contiguous().to(device)  # (N,3,H,W)

    def sample(self, B, rng):
        x = self.t[torch.from_numpy(rng.integers(0, len(self.t), B))]
        if rng.random() < 0.5:
            x = x.flip(3)
        return x


def _u(rng, lo, hi, shape, dev):
    return torch.tensor(rng.uniform(lo, hi, shape), dtype=torch.float32, device=dev)


def randomize(x_u8, seg, bank, rng, p=0.85, finger_erase=0.0, bottle_tex=0.0):
    """v5 extras: finger_erase = prob. of replacing the gripper pixels by the background texture (the real wrist view
    shows the fingers much less than the sim one, and v4 judged 'holding' from finger appearance instead of the gripper
    sensor); bottle_tex = prob. of blending a bank texture into the bottle (labels / translucency variety)."""
    """x_u8: (B, H, W, 3) uint8, seg: (B, H, W) uint8 on the same device -> (B, H, W, 3) float in 0..255."""
    dev = x_u8.device; B = x_u8.shape[0]
    x = x_u8.permute(0, 3, 1, 2).float()
    on = torch.from_numpy(rng.random(B) < p).to(dev).view(B, 1, 1, 1)
    s = seg.unsqueeze(1)
    bg, bo, bx, gr = (s == 0).float(), (s == 1).float(), (s == 2).float(), (s == 3).float()
    # background: texture x the sim's smoothed shading (masked blur keeps shadows, drops the sim speckle)
    lum = x.mean(1, keepdim=True)
    k = 7
    num = F.avg_pool2d(lum * bg, k, 1, k // 2); den = F.avg_pool2d(bg, k, 1, k // 2).clamp_min(1e-3)
    sm = num / den
    ref = (sm * bg).flatten(1).sum(1) / bg.flatten(1).sum(1).clamp_min(1)
    shade = (sm / ref.view(B, 1, 1, 1).clamp_min(1)).clamp(0.25, 1.4) ** _u(rng, 0.6, 1.0, (B, 1, 1, 1), dev)
    new_bg = bank.sample(B, rng).to(dev) * shade
    # bottle: per-channel gain + brightness + optional specular blob
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
    cx, cy, sg = _u(rng, 30, 100, (B, 1, 1), dev), _u(rng, 20, 72, (B, 1, 1), dev), _u(rng, 2, 10, (B, 1, 1), dev)
    blob = torch.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sg ** 2).unsqueeze(1) * _u(rng, 0, 90, (B, 1, 1, 1), dev) \
        * torch.from_numpy(rng.random(B) < 0.5).to(dev).view(B, 1, 1, 1)
    new_bo = x * _u(rng, 0.6, 1.4, (B, 3, 1, 1), dev) * _u(rng, 0.55, 1.25, (B, 1, 1, 1), dev) + blob
    new_bx = x * _u(rng, 0.85, 1.15, (B, 3, 1, 1), dev) * _u(rng, 0.7, 1.3, (B, 1, 1, 1), dev)
    new_gr = x * _u(rng, 0.5, 1.4, (B, 1, 1, 1), dev)
    if finger_erase > 0:
        er = torch.from_numpy(rng.random(B) < finger_erase).to(dev).view(B, 1, 1, 1)
        new_gr = torch.where(er, new_bg, new_gr)
    if bottle_tex > 0:
        bt = torch.from_numpy(rng.random(B) < bottle_tex).to(dev).view(B, 1, 1, 1).float() * _u(rng, 0.2, 0.6, (B, 1, 1, 1), dev)
        new_bo = new_bo * (1 - bt) + bank.sample(B, rng).to(dev) * _u(rng, 0.3, 1.0, (B, 1, 1, 1), dev) * bt
    y = bg * new_bg + bo * new_bo + bx * new_bx + gr * new_gr
    x = torch.where(on, y, x)
    # whole image: blur (30 %), noise, cut-outs (25 %)
    nb = torch.from_numpy(rng.random(B) < 0.3).to(dev).view(B, 1, 1, 1)
    sig = float(rng.uniform(0.5, 1.3)); r = 2; g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=dev, dtype=torch.float32) / sig) ** 2); g = g / g.sum()
    xb = F.conv2d(F.pad(x, (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
    xb = F.conv2d(F.pad(xb, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3)
    x = torch.where(nb, xb, x)
    x = x + torch.randn_like(x) * _u(rng, 0, 7, (B, 1, 1, 1), dev)
    for i in np.where(rng.random(B) < 0.25)[0]:
        for _ in range(rng.integers(1, 3)):
            h, w = rng.integers(4, 16, 2); y0, x0 = rng.integers(0, H - h), rng.integers(0, W - w)
            x[i, :, y0:y0 + h, x0:x0 + w] = torch.tensor(rng.uniform(0, 255, 3), dtype=torch.float32, device=dev).view(3, 1, 1)
    return x.clamp(0, 255).permute(0, 2, 3, 1)
