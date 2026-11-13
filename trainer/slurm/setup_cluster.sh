#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
python -m compileall -q 3dgs_trainer.py importance_prune.py scripts shared
python scripts/check_setup.py

echo "Cluster environment is ready at ${repository_root}/.venv"
