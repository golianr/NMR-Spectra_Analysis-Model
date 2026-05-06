#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d ".venv" ]; then
  echo "Creating new virtual environment .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f "nmr_artifacts_fusion.zip" ]; then
  echo ""
  echo "ERROR: nmr_artifacts_fusion.zip is missing."
  echo ""
fi

python app.py --server-name 0.0.0.0 --server-port 7860
