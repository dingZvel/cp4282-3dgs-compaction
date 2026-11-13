#!/usr/bin/env bash
#SBATCH --job-name=cp4282-train
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=24:00:00
#SBATCH --output=outputs/project/logs/train-%A_%a.out
#SBATCH --error=outputs/project/logs/train-%A_%a.err

set -euo pipefail
repository_root="${CP4282_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}}"
cd "${repository_root}"
mkdir -p outputs/project/logs
source .venv/bin/activate

mapfile -t configs < <(find config/project -maxdepth 1 -name '*.yaml' -type f | sort)
task_id="${SLURM_ARRAY_TASK_ID:?Submit this file as an array job.}"
if (( task_id < 0 || task_id >= ${#configs[@]} )); then
    echo "Array task ${task_id} is outside the ${#configs[@]} generated configs." >&2
    exit 2
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${task_id} config=${configs[task_id]}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python 3dgs_trainer.py "${configs[task_id]}"
