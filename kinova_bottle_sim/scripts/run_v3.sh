#!/bin/zsh
# v3: real-matched appearance (scene look 'real') -> ResNet A3 / B3 -> sim evals -> offline check on the real demos.
cd "$(dirname $0)/.."
P=.venv/bin/python
set -e
[ -f data/v3/meta.json ] || $P scripts/gen_data.py --n 500 --out data/v3 --workers 6 --size 256 144 --look real
$P scripts/train.py --mode A --data data/v3 --steps 8000 --out runs/A3
$P scripts/train.py --mode B --data data/v3 --steps 8000 --out runs/B3
E() { $P scripts/eval.py --n 50 --look real "$@" | grep -vE "^ep " ; }
E --ckpt runs/A3/last.pt --rule dense --out out/eval_A_dense_v3.json
E --ckpt runs/B3/last.pt --rule fixed --K 20 --out out/eval_B_fixed20_v3.json
E --ckpt runs/B3/last.pt --rule adaptive --move 0.08 --out out/eval_B_adaptive0.08_v3.json
$P scripts/real_offline.py --A runs/A3/last.pt --B runs/B3/last.pt --sim-ref 0 --out out/real_offline_v3.json
