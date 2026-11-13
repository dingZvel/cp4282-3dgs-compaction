#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
download_root="${1:-${repository_root}/data/downloads}"

mkdir -p "${download_root}"
python "${repository_root}/scripts/fetch_nerf_synthetic.py" "${download_root}"

python "${repository_root}/scripts/prepare_project_data.py" \
    "${download_root}" "${repository_root}/data/project_split" \
    --scenes lego ficus drums --validation-count 10
python "${repository_root}/scripts/generate_project_configs.py" \
    --data-root "${repository_root}/data/project_split" \
    --output-root "${repository_root}/outputs/project" \
    --config-root "${repository_root}/config/project"

echo "Prepared project datasets in ${repository_root}/data/project_split"
