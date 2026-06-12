#!/usr/bin/env bash
# Full pipeline: prepare → train → merge → evaluate
set -euo pipefail
CONFIG="${1:-configs/training.yaml}"

python -m lahja.data.prepare --config "$CONFIG"
python -m lahja.train.qlora  --config "$CONFIG"
python -m lahja.train.merge  --config "$CONFIG"
python -m lahja.eval.evaluate --config "$CONFIG"
