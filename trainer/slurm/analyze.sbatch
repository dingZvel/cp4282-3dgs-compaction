#!/bin/bash
#SBATCH --job-name=cp4282-analyze
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/project/logs/analyze-%j.out
#SBATCH --error=outputs/project/logs/analyze-%j.err

set -euo pipefail

cd /home/d/dingzhe/CP4282/project
.venv/bin/python scripts/analyze_project_results.py
