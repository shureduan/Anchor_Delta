#!/bin/zsh
# v4: class-map driven appearance randomisation (scripts/dr_aug.py) -> A4 / B4 (fine-tuned encoder) and A4f (frozen
# ImageNet encoder, control) -> per-phase offline check on the real demos right after each model -> sim evals.
cd "$(dirname $0)/.."
P=.venv/bin/python
set -e
[ -f data/v4/meta.json ] || $P scripts/gen_data.py --n 500 --out data/v4 --workers 6 --look real --seg
PH() { $P scripts/real_phase.py A2=runs/A2/last.pt B2=runs/B2/last.pt:fixed:20 "$@" ; }
$P scripts/train.py --mode A --data data/v4 --steps 8000 --dr --out runs/A4
PH A4=runs/A4/last.pt --out out/real_phase_v4_A.json
$P scripts/train.py --mode B --data data/v4 --steps 8000 --dr --out runs/B4
PH A4=runs/A4/last.pt B4=runs/B4/last.pt:fixed:20 B4ad=runs/B4/last.pt:adaptive:0.08 --out out/real_phase_v4_AB.json
$P scripts/train.py --mode A --data data/v4 --steps 8000 --dr --freeze-enc --out runs/A4f
PH A4=runs/A4/last.pt B4=runs/B4/last.pt:fixed:20 B4ad=runs/B4/last.pt:adaptive:0.08 A4f=runs/A4f/last.pt --out out/real_phase_v4.json
E() { $P scripts/eval.py --n 50 --look real "$@" | grep -vE "^ep " ; }
E --ckpt runs/A4/last.pt --rule dense --out out/eval_A_dense_v4.json
E --ckpt runs/B4/last.pt --rule fixed --K 20 --out out/eval_B_fixed20_v4.json
E --ckpt runs/B4/last.pt --rule adaptive --move 0.08 --out out/eval_B_adaptive0.08_v4.json
E --ckpt runs/A4f/last.pt --rule dense --out out/eval_A4f_dense_v4.json
$P scripts/real_offline.py --A runs/A4/last.pt --B runs/B4/last.pt --sim-ref 0 --out out/real_offline_v4.json
