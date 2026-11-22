#!/bin/zsh
# v5: observable gripper timing (expert --observable) + finger erasing / bottle textures (train --dr5).
cd "$(dirname $0)/.."
P=.venv/bin/python
set -e
[ -f data/v5/meta.json ] || $P scripts/gen_data.py --n 500 --out data/v5 --workers 4 --look real --seg --observable
$P scripts/train.py --mode A --data data/v5 --steps 8000 --dr5 --out runs/A5
$P scripts/real_grip_check.py runs/A4/last.pt runs/A5/last.pt
$P scripts/train.py --mode B --data data/v5 --steps 8000 --dr5 --out runs/B5
$P scripts/real_grip_check.py runs/B4/last.pt runs/B5/last.pt
$P scripts/real_phase.py A4=runs/A4/last.pt B4=runs/B4/last.pt:fixed:20 A5=runs/A5/last.pt B5=runs/B5/last.pt:fixed:20 B5ad=runs/B5/last.pt:adaptive:0.08 --out out/real_phase_v5.json
E() { $P scripts/eval.py --n 30 --look real "$@" | grep -vE "^ep " ; }
E --ckpt runs/A5/last.pt --rule dense --out out/eval_A_dense_v5.json
E --ckpt runs/B5/last.pt --rule fixed --K 20 --out out/eval_B_fixed20_v5.json
E --ckpt runs/B5/last.pt --rule adaptive --move 0.08 --out out/eval_B_adaptive0.08_v5.json
