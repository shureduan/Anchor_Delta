#!/bin/zsh
# SmolVLA pipeline: cached image features -> clean timing -> fine-tune A and B -> closed-loop evals.
cd "$(dirname $0)/.."
export HF_HUB_OFFLINE=1                 # everything is in the local HF cache; do not stall on the network
P=.venv/bin/python; V=.venv-vla/bin/python
set -e
$P scripts/eval.py --ckpt runs/A2/last.pt --rule dense --n 50 --out out/eval_A_dense_v2.json | grep -vE "^ep "   # clean re-timing
$V scripts/cache_feats.py --data data/v2 --n 80
$V scripts/vla.py > out/vla_bench.log 2>&1
$V scripts/train_vla.py --mode A --n 80 --steps 3000 --out runs/vla_A
$V scripts/train_vla.py --mode B --n 80 --steps 3000 --out runs/vla_B
$V scripts/eval_vla.py --ckpt runs/vla_A/last.pt --rule dense --n 30 --out out/vla_A_dense.json
$V scripts/eval_vla.py --ckpt runs/vla_B/last.pt --rule dense --n 30 --out out/vla_B_dense.json
for M in 0.04 0.08; do $V scripts/eval_vla.py --ckpt runs/vla_B/last.pt --rule adaptive --move $M --n 30 --out out/vla_B_adaptive$M.json; done
for K in 20 40; do $V scripts/eval_vla.py --ckpt runs/vla_B/last.pt --rule fixed --K $K --n 30 --out out/vla_B_fixed$K.json; done
