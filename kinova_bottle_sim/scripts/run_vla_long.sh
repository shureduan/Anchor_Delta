#!/bin/zsh
# Longer SmolVLA training: 200 episodes, continue A and B for 8000 steps each, then the eval table.
cd "$(dirname $0)/.."
export HF_HUB_OFFLINE=1
V=.venv-vla/bin/python
set -e
$V scripts/cache_feats.py --data data/v2 --n 200
$V scripts/train_vla.py --mode A --n 200 --steps 8000 --init runs/vla_A/last.pt --out runs/vla_A_long
$V scripts/train_vla.py --mode B --n 200 --steps 8000 --init runs/vla_B/last.pt --out runs/vla_B_long
E() { $V scripts/eval_vla.py --n 30 "$@" | grep -vE "^ep " ; }
E --ckpt runs/vla_A_long/last.pt --rule dense --out out/vla_long_A_dense.json
E --ckpt runs/vla_B_long/last.pt --rule dense --out out/vla_long_B_dense.json
E --ckpt runs/vla_B_long/last.pt --rule adaptive --move 0.08 --out out/vla_long_B_adaptive0.08.json
for K in 20 40; do E --ckpt runs/vla_B_long/last.pt --rule fixed --K $K --out out/vla_long_B_fixed$K.json; done
