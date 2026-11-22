# 1. Algorithm: dense encoding (A) vs. anchor + delta (B)

The question: a visuomotor policy normally encodes a camera image at every control step. **Does it need to?** Line B encodes an image only at sparse *anchors* and, in between, tells the policy how the robot has moved since the anchor. That motion is the *delta*, computed from the robot's own sensors. Lines A and B share the same network, data and training recipe; only the inputs differ.

## 1.1 Network (identical for A and B)

```
 image 128x72 RGB                                   proprioception
      |                                                   |
 ResNet-18 (ImageNet init, fine-tuned)          MLP 2 layers -> 128
 11.2 M params -> 512-d feature                           |
      +-------------------------+-------------------------+
                                |
              MLP head 640 -> 512 -> 512 -> 10 x 4   (0.61 M params)
                                |
   action chunk, H = 10 steps (1 s at 10 Hz); per step:
   tool-position offset xyz (relative to the current tool position, unit 5 cm) + gripper-close logit
```

| | **A: dense** | **B: anchor + delta** |
|---|---|---|
| image input | the **current** frame | the latest **anchor** frame; its 512-d feature is computed once and cached |
| proprio | tool xyz (normalised), gripper opening, steps since the gripper last changed state (/50, capped) = **5** | the same 5 + **delta 5**: (tool_now - tool_anchor) / 5 cm (3), gripper closed at the anchor (1), anchor age / 50 (1) = **10** |
| compute per step | ResNet + head | anchor steps: ResNet + head; all other steps: **head only** |

**Delta source.** The tool position comes from Kortex forward kinematics and the gripper opening from the gripper's position sensor. Simulator ground truth is **never** used, so the same inputs exist on the real robot.

## 1.2 Anchor rules (B), decided from proprioception only

| rule | re-encode when |
|---|---|
| `dense` | every step (same information as A; diagnostic only) |
| `fixed K=20` | t mod 20 = 0 (every 2 s) |
| `adaptive 8 cm` | t = 0; the gripper state flips (opening sensor, closed = reading > 0.2); or the tool has moved > 8 cm since the anchor |

## 1.3 From a chunk to a command: temporal ensembling (A and B)

Every 0.1 s the policy outputs a 10-step chunk. The command for the current step is the **mean of what all still-valid chunks predicted for this step** (ACT-style, uniform weights). The gripper closes when the mean logit is > 0. A single frame cannot tell how long the robot has been waiting; without ensembling the policy deadlocked at the grasp pose in v1.

## 1.4 Training (behaviour cloning, `kinova_bottle_sim/scripts/train.py`)

| item | setting |
|---|---|
| data | scripted-expert demonstrations in simulation (Section 2); ~490 episodes, ~100 k frames; split by episode, 90 / 10 |
| labels | the expert's next 10 tool targets minus the current tool position (/5 cm), and the expert's next 10 gripper commands |
| loss | L1 on offsets + 0.1 x BCE on gripper commands |
| optimiser | AdamW, lr 3e-4, weight decay 1e-4, OneCycle (5 % warm-up), 8000 steps, batch 128 (~45 min on an Apple M4 GPU) |
| B anchors during training | per sample: 1/3 adaptive rule (4 / 8 / 12 cm), 1/3 fixed K ~ U{1..60}, 1/3 random age U[0, 80]. One model serves every rule, and the anchor age cannot be used as a clock. v1 learned exactly that shortcut: fixed-60 succeeded 2/50 |
| image augmentation | v4: class-map-driven appearance randomisation (`dr_aug.py`): random table textures with glare, cables, stains and table edges; bottle, box and gripper colour gains; blur, noise and cut-outs; plus shift and colour jitter. v5 adds: erase the gripper fingers (p = 0.5) and random bottle textures (p = 0.5) |

## 1.5 What B saves

- **Compute:** most steps skip the image encoder. On the real robot B used 10–12 encoder calls per ~190-step episode (5 %), and its policy compute is 1.6 ms/step against 9.0 ms for A (idle M4 GPU).
- **Storage:** only anchor frames plus 5 floats per step need to be kept. That is about 1/18 of the visual data of a dense log on the real runs.
- **What B does not save:** reading the camera itself, and the robot's motion time.

## 1.6 Version history (what changed and why)

| version | change | reason |
|---|---|---|
| v1 | A / B as above (no gripper timer) | B used the anchor age as a clock ("lift N steps after the anchor") |
| v2 | both get *steps since the gripper changed*; B uses mixed anchor sampling | removes the shortcut: B fixed-60 rose from 2/50 to 47/50 |
| v3 | sim colours matched to the real scene | **worse** on real frames (see failure analysis) |
| v4 | class-map-driven appearance randomisation | real-frame agreement 97–98 %, up from 77–86 % |
| v5 | expert closes and releases **on arrival** (no hidden waits); finger erasing; bottle textures | v4 gripper decisions came from appearance and an unobservable timer (Section 5) |

## 1.7 Second line: SmolVLA (supplementary)

The same A / B idea was tried on `lerobot/smolvla_base`: the action expert is fine-tuned, re-planning happens every 10 steps with RTC, and B caches the image tokens at anchors and puts the delta into the state token. After 200 episodes and 8 k more steps, A reached 24/30 and B 13–15/30. A plan costs 452 ms (image 148 + VLM 36 + denoising 265), so **B can save at most ~1.24x** there. When action generation dominates the compute, the anchor idea pays off much less. The main line is therefore the ResNet policy.
