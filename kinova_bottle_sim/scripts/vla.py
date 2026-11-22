"""SmolVLA (lerobot/smolvla_base, 450M) for bottle -> box, in the A / B split of DELTA.

SmolVLA = SmolVLM2 prefix (wrist image + task text + state token -> KV cache) + a flow-matching action expert that
cross-attends to that cache for 10 denoising steps. The prefix is the expensive part (vision encoder + 16 VLM layers).

A (dense):  every planning call re-encodes the prefix from the CURRENT image and state (standard SmolVLA).
B (anchor): the IMAGE is encoded (SigLIP + connector, the dominant cost) only at anchor frames; the 64 image tokens are
            cached and reused. Every plan re-runs the VLM layers over [cached image tokens, task, state token] with the
            CURRENT state plus what changed since the anchor, through SmolVLA's own state pathway:
            state_B = [state now (5), (tool - tool_anchor) / 0.05 (3), gripper closed at the anchor, steps since anchor / 50]
            (all from robot measurements, never sim state).
            (The earlier variant, now runs/vla_Bold, kept the anchor's whole prefix KV cache and added the delta to the
            action tokens through a zero-initialised linear layer; it did not learn to use the delta in 3000 steps.)

State (prefix token, A): [(tool - mean) / std, gripper opening, steps since gripper change / 50]  (policy.proprio).
Actions (chunk H_VLA at 10 Hz): [(target - tool_now) / 0.05 (3), gripper cmd * 2 - 1].
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from policy import POS_SCALE, proprio  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import (SmolVLAPolicy, create_sinusoidal_pos_embedding,  # noqa: E402
                                                       make_att_2d_masks)

BASE = 'lerobot/smolvla_base'
TASK = 'put the bottle into the green box\n'
H_VLA = 20                     # chunk length (2 s at 10 Hz)
IMG_W, IMG_H = 256, 144
STATE_DIM, DELTA_DIM, ACT_DIM = 5, 6, 4
IMG_KEY = 'observation.images.wrist'


def make_config(device):
    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.input_features = {'observation.state': PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
                          IMG_KEY: PolicyFeature(FeatureType.VISUAL, (3, IMG_H, IMG_W))}
    cfg.output_features = {'action': PolicyFeature(FeatureType.ACTION, (ACT_DIM,))}
    cfg.normalization_mapping = {'VISUAL': NormalizationMode.IDENTITY, 'STATE': NormalizationMode.IDENTITY,
                                 'ACTION': NormalizationMode.IDENTITY}      # we normalise ourselves
    cfg.chunk_size = H_VLA; cfg.n_action_steps = H_VLA; cfg.empty_cameras = 0; cfg.device = device
    return cfg


class VLA(nn.Module):
    """Thin wrapper: tokenised task, delta injection, split prefix / denoise calls."""

    def __init__(self, mode='A', device='mps', ckpt=None):
        super().__init__()
        self.mode = mode; self.device = device
        self.pol = SmolVLAPolicy.from_pretrained(BASE, config=make_config(device))
        m = self.pol.model
        m.delta_proj = nn.Linear(DELTA_DIM, m.vlm_with_expert.expert_hidden_size)
        nn.init.zeros_(m.delta_proj.weight); nn.init.zeros_(m.delta_proj.bias)
        m._delta = None
        m.embed_suffix = self._embed_suffix.__get__(m)              # delta-aware suffix
        # cached image features (frozen SigLIP + connector) can be injected for training: embed_image returns them
        ve = m.vlm_with_expert; ve._feat = None; orig = ve.embed_image
        ve.embed_image = lambda image: ve._feat.to(image.dtype) if ve._feat is not None else orig(image)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.pol.config.vlm_model_name)
        t = tok(TASK, padding='max_length', max_length=self.pol.config.tokenizer_max_length, padding_side='right',
                return_tensors='pt', truncation=True)
        self.register_buffer('lang', t['input_ids']); self.register_buffer('lang_mask', t['attention_mask'].bool())
        if ckpt is not None:
            self.load_trainable(torch.load(ckpt, map_location='cpu')['state'])
        self.to(device)

    @staticmethod
    def _embed_suffix(m, noisy_actions, timestep):
        action_emb = m.action_in_proj(noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(timestep, m.vlm_with_expert.expert_hidden_size, m.config.min_period,
                                                   m.config.max_period, device=action_emb.device).type(action_emb.dtype)
        x = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], 2)
        x = m.action_time_mlp_out(F.silu(m.action_time_mlp_in(x)))
        if m._delta is not None:
            x = x + m.delta_proj(m._delta.to(x.dtype))[:, None, :]
        B, L = x.shape[:2]
        pad = torch.ones(B, L, dtype=torch.bool, device=x.device)
        att = torch.ones(B, L, dtype=x.dtype, device=x.device)
        return x, pad, att

    # ---- trainable parameters (the VLM and vision encoder stay frozen, as in SmolVLA fine-tuning) ----
    def trainable(self):
        return {k: p for k, p in self.pol.model.named_parameters() if p.requires_grad}

    def load_trainable(self, sd):
        missing = set(self.trainable()) - set(sd)
        assert not missing, f'missing trainable weights: {sorted(missing)[:5]}'
        self.pol.model.load_state_dict(sd, strict=False)

    # ---- inputs ----
    def batch(self, img_u8, state, action=None):
        """img_u8: (B, H, W, 3) uint8 (or float 0..255) torch; state (B, 5); action (B, H_VLA, 4) normalised."""
        img = img_u8.permute(0, 3, 1, 2).float() / 255.0
        B = img.shape[0]
        b = {IMG_KEY: img, 'observation.state': state.float(),
             'observation.language.tokens': self.lang.expand(B, -1), 'observation.language.attention_mask': self.lang_mask.expand(B, -1)}
        if action is not None:
            b['action'] = action.float()
        return b

    def loss(self, img_u8, state, action, delta=None, feat=None):
        """feat: (B, 64, 960) cached image features (then img_u8 may be a tiny dummy)."""
        m = self.pol.model; m._delta = delta.float() if delta is not None else None
        m.vlm_with_expert._feat = feat
        try:
            loss, _ = self.pol.forward(self.batch(img_u8, state, action))
        finally:
            m.vlm_with_expert._feat = None
        return loss

    @torch.no_grad()
    def image_features(self, img_u8):
        """Frozen SigLIP + connector features (B, 64, 960) of uint8 frames, exactly as the prefix computes them."""
        images, _ = self.pol.prepare_images(self.batch(img_u8, torch.zeros(len(img_u8), STATE_DIM, device=img_u8.device)))
        return self.pol.model.vlm_with_expert.embed_image(images[0])

    # ---- inference, split so B can reuse the prefix ----
    @torch.no_grad()
    def prefix(self, img_u8, state, feat=None):
        """Vision encoder (skipped when cached image features `feat` are given) + VLM over image, text and state
        -> (KV cache, prefix pad mask)."""
        ve = self.pol.model.vlm_with_expert; ve._feat = feat
        try:
            return self._prefix(img_u8, state)
        finally:
            ve._feat = None

    def _prefix(self, img_u8, state):
        b = self.batch(img_u8, state); m = self.pol.model
        images, img_masks = self.pol.prepare_images(b)
        st = self.pol.prepare_state(b)
        embs, pad, att = m.embed_prefix(images, img_masks, b['observation.language.tokens'], b['observation.language.attention_mask'], state=st)
        att2d = make_att_2d_masks(pad, att); pos = torch.cumsum(pad, 1) - 1
        _, pkv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                           inputs_embeds=[embs, None], use_cache=True, fill_kv_cache=True)
        return pkv, pad

    def denoise(self, cache, delta=None, noise=None, rtc=None):
        """Cheap part: 10 flow-matching steps of the action expert on a cached prefix. Returns (H_VLA, 4) normalised.
        rtc: None, or dict(prev_chunk_left_over=(1, H, 32) tensor, inference_delay=int, execution_horizon=int) for RTC
        guided inpainting (needs autograd, so it is not wrapped in no_grad)."""
        m = self.pol.model; pkv, pad = cache
        m._delta = delta.float() if delta is not None else None
        x = noise if noise is not None else m.sample_noise((1, H_VLA, m.config.max_action_dim), pad.device)
        n = m.config.num_steps; dt = -1.0 / n
        for s in range(n):
            tt = torch.full((1,), 1.0 + s * dt, dtype=torch.float32, device=pad.device)

            def call(xx, tt=tt):
                return m.denoise_step(x_t=xx, prefix_pad_masks=pad, past_key_values=pkv, timestep=tt)
            if rtc is not None and rtc.get('prev_chunk_left_over') is not None:
                v = m.rtc_processor.denoise_step(x_t=x, time=1.0 + s * dt, original_denoise_step_partial=call, **rtc)
            else:
                with torch.no_grad():
                    v = call(x)
            x = (x + dt * v).detach()
        return x


def state_vec(tool, grip, flip):
    return proprio(np.asarray(tool, np.float32), np.float32(grip), flip=flip)


def anchor_delta(tool, tool_a, grip_a, since):
    """B's extra state entries: displacement since the anchor, gripper closed at the anchor, anchor age."""
    return np.r_[(np.asarray(tool) - tool_a) / POS_SCALE, float(grip_a > 0.2), since / 50.0].astype(np.float32)


def delta_vec(tool, tool_a, grip, since, flip):
    return np.r_[(np.asarray(tool) - tool_a) / POS_SCALE, grip, since / 50.0, min(flip, 50) / 50.0].astype(np.float32)


def bench(n=10):
    dev = 'mps'
    v = VLA('B', dev).eval()
    print('trainable params %.1fM / total %.1fM' % (sum(p.numel() for p in v.trainable().values()) / 1e6,
                                                     sum(p.numel() for p in v.parameters()) / 1e6))
    img = torch.randint(0, 255, (1, IMG_H, IMG_W, 3), dtype=torch.uint8, device=dev)
    st = torch.zeros(1, STATE_DIM, device=dev); de = torch.zeros(1, DELTA_DIM, device=dev)
    tp, td = [], []
    for i in range(n + 2):
        t0 = time.perf_counter(); c = v.prefix(img, st); torch.mps.synchronize(); t1 = time.perf_counter()
        v.denoise(c, de); torch.mps.synchronize(); t2 = time.perf_counter()
        if i >= 2:
            tp.append(t1 - t0); td.append(t2 - t1)
    print(f'prefix (image+VLM) {np.median(tp) * 1e3:.1f} ms   denoise x10 (action expert) {np.median(td) * 1e3:.1f} ms')
    # one training step at batch 16
    opt = torch.optim.AdamW(list(v.trainable().values()), lr=1e-4)
    v.train()
    for bs in (8, 16):
        im = torch.randint(0, 255, (bs, IMG_H, IMG_W, 3), dtype=torch.uint8, device=dev)
        ts = []
        for i in range(4):
            t0 = time.perf_counter()
            l = v.loss(im, torch.zeros(bs, STATE_DIM, device=dev), torch.zeros(bs, H_VLA, ACT_DIM, device=dev), torch.zeros(bs, DELTA_DIM, device=dev))
            opt.zero_grad(); l.backward(); opt.step(); torch.mps.synchronize()
            if i:
                ts.append(time.perf_counter() - t0)
        print(f'train step bs {bs}: {np.median(ts):.2f} s')


if __name__ == '__main__':
    bench()
