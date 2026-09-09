#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python experiments/run_all.py
python experiments/run_learned_model.py --outdir results/real_model
