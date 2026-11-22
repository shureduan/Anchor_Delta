#!/bin/zsh
# Full A vs B comparison, run sequentially so timings are not contaminated.
#   scripts/sweep.sh runs/A runs/B ""      (v1)
#   scripts/sweep.sh runs/A2 runs/B2 _v2
cd "$(dirname $0)/.."
P=.venv/bin/python; A=${1:-runs/A}; B=${2:-runs/B}; S=${3:-}
$P scripts/eval.py --ckpt $A/last.pt --rule dense --n 50 --out out/eval_A_dense$S.json
$P scripts/eval.py --ckpt $B/last.pt --rule dense --n 50 --out out/eval_B_dense$S.json
for K in 5 10 20 30 60 400; do
  $P scripts/eval.py --ckpt $B/last.pt --rule fixed --K $K --n 50 --out out/eval_B_fixed$K$S.json
done
for M in 0.04 0.08 0.12; do
  $P scripts/eval.py --ckpt $B/last.pt --rule adaptive --move $M --n 50 --out out/eval_B_adaptive$M$S.json
done
