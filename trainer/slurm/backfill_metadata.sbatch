#!/usr/bin/env bash
#SBATCH --job-name=cp4282-metadata
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/project/logs/metadata-%j.out
#SBATCH --error=outputs/project/logs/metadata-%j.err

set -euo pipefail

cd /home/d/dingzhe/CP4282/project
mapfile -t configs < <(find config/project -maxdepth 1 -name '*.yaml' -type f | sort)
for task_id in 0 1 2 3 4 5 6 7 9 10 11 12 13 14; do
    .venv/bin/python scripts/postprocess_model.py "${configs[task_id]}" --skip-existing
done
