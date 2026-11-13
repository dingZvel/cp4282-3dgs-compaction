#!/usr/bin/env bash
#SBATCH --job-name=cp4282-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/project/logs/smoke-%j.out
#SBATCH --error=outputs/project/logs/smoke-%j.err

set -euo pipefail
repository_root="${CP4282_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}}"
cd "${repository_root}"
mkdir -p outputs/project/logs
source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m compileall -q 3dgs_trainer.py importance_prune.py scripts shared
python scripts/check_setup.py
python 3dgs_gradient_check_gpu.py --device cuda:0

# One dense learned-mask step verifies CUDA compilation and mask gradients without launching the
# full grid.
python 3dgs_trainer.py config/3dgs_smoke_mask_gpu.yaml
