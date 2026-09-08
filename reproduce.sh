#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python experiments/run_all.py
