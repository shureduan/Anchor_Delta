#!/bin/zsh
# SmolVLA B (cached image tokens + state-token delta): fine-tune, then the A / B closed-loop table.
cd "$(dirname $0)/.."
export HF_HUB_OFFLINE=1
V=.venv-vla/bin/python
set -e
$V scripts/train_vla.py --mode B --n 80 --steps 3000 --out runs/vla_B
E() { $V scripts/eval_vla.py --n 30 "$@" | grep -vE "^ep " ; }
E --ckpt runs/vla_A/last.pt --rule dense --out out/vla_A_dense.json
E --ckpt runs/vla_A/last.pt --rule dense --delay 0 --out out/vla_A_dense_d0.json
E --ckpt runs/vla_B/last.pt --rule dense --out out/vla_B_dense.json
for M in 0.04 0.08; do E --ckpt runs/vla_B/last.pt --rule adaptive --move $M --out out/vla_B_adaptive$M.json; done
for K in 20 40; do E --ckpt runs/vla_B/last.pt --rule fixed --K $K --out out/vla_B_fixed$K.json; done
E --ckpt runs/vla_B/last.pt --rule fixed --K 20 --delay 0 --out out/vla_B_fixed20_d0.json
