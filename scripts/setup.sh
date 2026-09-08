#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
if [[ -f requirements-lock.txt ]]; then
  .venv/bin/python -m pip install -r requirements-lock.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m pip check
