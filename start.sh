#!/usr/bin/env bash
# Reproduce all results of the paper with the settings they were published with.
# Any argument given here is passed straight through to main.py, so for example
#   ./start.sh --steps tables
#   ./start.sh --no-usetex
set -euo pipefail

cd "$(dirname "$0")"

# Override with e.g. PYTHON=/path/to/conda/env/bin/python3 ./start.sh
"${PYTHON:-python3}" main.py \
  --steps analysis tables local \
  --result-dir ./results \
  --plot-dir ./plots \
  --model-dir ./models \
  --pathologies AVB ISCH RBBB LBBB \
  --n-bootstrap 1000 \
  "$@"
