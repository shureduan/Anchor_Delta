"""Policies for bottle -> box, sharing one architecture so A and B differ only in their inputs.

A (dense):   image of the CURRENT frame + proprio (tool xyz, gripper opening)
B (anchor):  image of the latest ANCHOR frame (encoded once, cached) + proprio + delta since the anchor:
             (tool_now - tool_anchor, gripper at the anchor, steps since the anchor / 50)
             Delta comes from the robot's own measurements (FK tool position, gripper sensor), never sim state.

Output: an action chunk of H steps: tool target offsets relative to the current tool position (metres / 0.05) and
gripper-close logits. ResNet18 (ImageNet weights) encoder, MLP head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

H = 10                        # chunk length (1 s)
POS_SCALE = 0.05              # metres per unit of the normalised offsets
TOOL_MEAN = np.array([0.35, 0.10, 0.25], np.float32)
TOOL_STD = np.array([0.10, 0.12, 0.10], np.float32)
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


FLIP_CAP = 50                 # steps; "time since the gripper last changed state" saturates here


def flip_since(grip):
    """Steps since the gripper (opening sensor, closed = grip > 0.2) last changed state, per step of one episode;
    counted from the episode start before the first change. Available on the real robot from the gripper sensor."""
    closed = np.asarray(grip) > 0.2; out = np.zeros(len(closed), np.float32); last = 0
    for t in range(len(closed)):
        if t > 0 and closed[t] != closed[t - 1]:
            last = t
        out[t] = t - last
    return out


def anchor_indices(tool, grip, rule, K=20, move_thresh=0.08):
    """Anchor frame index for every step of one episode, from proprio only.
    fixed:    every K steps
    adaptive: at t = 0, when the gripper state flips (closing / opening detected from the opening sensor), and when
              the tool has moved more than move_thresh since the anchor (the anchor view is stale)."""
    T = len(tool); a = np.zeros(T, int); last = 0
    closed = grip > 0.2
    for t in range(T):
        if rule == 'dense':
            last = t
        elif rule == 'fixed':
            if t % K == 0:
                last = t
        elif rule == 'adaptive':
            if t == 0 or (t > 0 and closed[t] != closed[t - 1]) or np.linalg.norm(tool[t] - tool[last]) > move_thresh:
                last = t
        a[t] = last
    return a


class Policy(nn.Module):
    def __init__(self, mode='A', pretrained=True, flip=False):
        super().__init__()
        self.mode = mode; self.flip = flip
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        r.fc = nn.Identity()
        self.enc = r
        n_prop = 4 + int(flip) + (5 if mode == 'B' else 0)
        self.prop = nn.Sequential(nn.Linear(n_prop, 128), nn.ReLU(), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(512 + 128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, H * 4))

    def encode(self, img_u8):
        """img_u8: (B, H, W, 3) uint8 RGB tensor -> (B, 512)."""
        x = img_u8.permute(0, 3, 1, 2).float() / 255.0
        x = (x - IMG_MEAN.to(x.device)) / IMG_STD.to(x.device)
        return self.enc(x)

    def forward(self, feat, prop):
        out = self.head(torch.cat([feat, self.prop(prop)], 1)).view(-1, H, 4)
        return out[..., :3], out[..., 3]                      # offsets (normalised), gripper logits


def proprio(tool, grip, tool_anchor=None, grip_anchor=None, since=None, flip=None):
    """flip: steps since the last gripper state change (None for the v1 models that do not take it)."""
    p = [(tool - TOOL_MEAN) / TOOL_STD, grip[..., None]]
    if flip is not None:
        p += [np.minimum(np.asarray(flip, np.float32), FLIP_CAP)[..., None] / FLIP_CAP]
    if tool_anchor is not None:
        p += [(tool - tool_anchor) / POS_SCALE, grip_anchor[..., None], np.asarray(since, np.float32)[..., None] / 50.0]
    return np.concatenate(p, -1).astype(np.float32)
