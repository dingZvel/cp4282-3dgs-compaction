<!-- markdownlint-disable-file MD013 -->
# Scene-Dependent Compaction of 3D Gaussian Splatting

This repository contains the code, experiment artifacts, report, and presentation for a CP4282
course project on compacting 3D Gaussian Splatting (3DGS) models. The project compares two pruning
directions:

1. **Learned-mask pruning:** mask variables are optimized jointly with the rendering objective
   during training.
2. **Importance pruning:** after training, Gaussians are ranked by their accumulated alpha-blending
   contribution and the least important Gaussians are removed.

The initial hypothesis was that learned masks should generally outperform post-training importance
pruning because they are trained with the representation and can learn which Gaussians are
dispensable. The experiments instead show that this advantage is scene-dependent and uncommon in
the tested set. At equal Gaussian counts, Ficus benefits from a mild learned mask, while Drums and
Lego achieve better quality by skipping the learned-mask stage. Because every candidate mask
penalty requires another training run and the useful penalty range is scene-specific, importance
pruning is the recommended default.

## Main results

Every eligible method is compared at exact common budgets derived from the 300,000-Gaussian
unmasked reference:

- **40% retention:** 120,000 Gaussians;
- **20% retention:** 60,000 Gaussians.

| Scene | Retention | Best direction | Mask penalty | Count | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|
| Drums | 20% | Importance only | 0 | 60,000 | 21.735 | 0.89094 | 0.09262 |
| Drums | 40% | Importance only | 0 | 120,000 | 22.128 | 0.89649 | 0.08432 |
| Ficus | 20% | Mask then importance | 0.004 | 60,000 | 22.311 | 0.90839 | 0.06691 |
| Ficus | 40% | Mask then importance | 0.004 | 120,000 | 22.446 | 0.91097 | 0.06592 |
| Lego | 20% | Importance only | 0 | 60,000 | 26.169 | 0.93626 | 0.04853 |
| Lego | 40% | Importance only | 0 | 120,000 | 26.799 | 0.94257 | 0.04032 |

The strong penalties are intentionally retained as negative results. With
\(\lambda_{\mathrm{mask}}=0.016\), Drums ends with 16 Gaussians, Ficus collapses to zero, and Lego
ends with 84. Ficus already collapses to 39 Gaussians at \(\lambda_{\mathrm{mask}}=0.008\).

For the complete interpretation, limitations, and failure analysis, read
[project_report.md](project_report.md).

## Repository layout

```text
.
├── README.md                         # This overview and reproduction guide
├── project_ideas.md                  # Initial project notes
├── project_proposal.md               # Proposal and evaluation plan
├── project_report.md                 # Full final report
├── presentation/                     # Static interactive presentation
├── presentation_script.md            # Slide-by-slide speaker notes
├── presentation_script_print.pdf     # Printable speaker notes
└── trainer/                           # Trainer, pruning code, Slurm jobs, data tools, and outputs
    ├── 3dgs_trainer.py
    ├── 3dgs_evaluator.py
    ├── importance_prune.py
    ├── requirements.txt
    ├── config/
    ├── scripts/
    ├── shared/
    └── slurm/
```

The experiment code lives under `trainer/`. Unless a command says otherwise, run the reproduction
commands below from that directory:

```bash
cd trainer
```

Generated datasets, models, renders, metrics, and logs are excluded from Git by
`trainer/.gitignore`. A clean reproduction therefore creates `trainer/outputs/project/` locally or
on the cluster.

## Experiment design

The fixed grid contains:

- three NeRF Synthetic scenes: Drums, Ficus, and Lego;
- five mask settings per scene: \(0\), \(0.002\), \(0.004\), \(0.008\), and \(0.016\);
- 20,000 training iterations per configuration;
- seed 7 for every configuration;
- a 300,000-Gaussian model capacity;
- 256 x 256 training and evaluation resolution;
- 90 optimization views and 10 validation views selected deterministically from the original
  training set;
- the untouched 200-view NeRF Synthetic test set for final metrics;
- four relative importance-pruning fractions: 0%, 25%, 50%, and 75%;
- two exact matched budgets: 40% and 20% retention;
- PSNR, SSIM, and LPIPS evaluation.

The completed experiment contains 15 training configurations and 78 successfully evaluated
models. Six exact-budget arms are structured exclusions because learned-mask training had already
reduced them below the requested budget, and one training configuration is a complete collapse.

## Requirements

### Cluster hardware

The full experiment was run on the NUS Computing Cluster with one **H100 96 GB GPU** per training
or post-processing task. The supplied Slurm files request:

- partition `gpu-long`;
- `--gres=gpu:h100-96:1`;
- 8 CPU cores and 32 GB host memory;
- up to 24 hours for training and 12 hours for post-processing.

The smoke test requests an A100 80 GB GPU on the `gpu` partition. If another cluster uses
different partition or generic-resource names, edit only the `#SBATCH` resource lines; the Python
commands and experiment configuration can remain unchanged.

### Software

- Linux cluster node with a CUDA-capable NVIDIA GPU and driver;
- Slurm for the supplied batch workflow;
- Python 3 with `venv` support;
- network access during setup to install packages and download NeRF Synthetic;
- enough project or scratch storage for checkpoints, 78 sets of 200 test renders, metrics, and
  diagnostic sidecars.

The Python dependencies are declared in `trainer/requirements.txt` and include NumPy, Pillow,
PyYAML, Warp, PyTorch, LPIPS, `gdown`, and `huggingface-hub`.

## Reproduce the results on Slurm

### 1. Transfer or check out the repository

Place the complete repository in cluster storage, then enter the trainer directory. The original
implementation repository is [dingZvel/cp4282-as2](https://github.com/dingZvel/cp4282-as2), but
reproduction requires the project files present in this snapshot, including `importance_prune.py`,
`scripts/`, and `slurm/`.

```bash
cd /path/to/Project/trainer
pwd
```

Do not run the full experiment from a login node. Login nodes should only be used for environment
setup, data preparation, job submission, and lightweight analysis.

### 2. Create the Python environment

The setup script creates `trainer/.venv`, installs all requirements, compiles the Python sources,
and runs the repository setup check:

```bash
bash slurm/setup_cluster.sh
source .venv/bin/activate
python --version
```

If package installation must use a cluster proxy, wheelhouse, or module-provided Python, configure
that first and then run the same script. The requirement ranges are not a fully pinned lock file,
so exact bitwise reproduction may require recording `pip freeze` and the GPU driver from the
original environment. The scientific comparisons should be numerically close, but CUDA kernels
are not guaranteed to be bitwise deterministic across Warp, driver, or GPU versions.

### 3. Download and prepare the dataset

This command performs all three data-preparation steps:

1. download Drums, Ficus, and Lego from the original NeRF Synthetic archive, falling back to a
   Hugging Face mirror if the Google Drive archive is unavailable;
2. create deterministic 90-train/10-validation manifests while preserving all 200 test views;
3. generate the 15 experiment YAML files under `config/project/`.

```bash
bash scripts/download_project_data.sh
```

The resulting data layout is:

```text
data/
├── downloads/nerf_synthetic/{drums,ficus,lego}/
└── project_split/{drums,ficus,lego}/
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    └── project_split.json
```

The prepared image directories are symbolic links to the downloaded dataset, so do not move
`data/downloads/` after preparing the split.

Confirm that 15 configurations were generated and inspect their exact Slurm array mapping:

```bash
find config/project -maxdepth 1 -name '*.yaml' -type f | sort | nl -v0 -ba
```

With the current naming scheme, task IDs 0-4 are Drums, 5-9 are Ficus, and 10-14 are Lego. Within
each scene, the sorted order is `0p002`, `0p004`, `0p008`, `0p016`, and `none`. Therefore task 8 is
Ficus at penalty `0.016`, the known zero-Gaussian collapse.

### 4. Run the GPU smoke test

Create the log directory before the first submission:

```bash
mkdir -p outputs/project/logs
SMOKE_JOB=$(sbatch --parsable slurm/smoke.sbatch)
echo "Smoke job: ${SMOKE_JOB}"
squeue -j "${SMOKE_JOB}"
```

The smoke job performs a CUDA gradient check and one learned-mask training step. Check both log
files after it finishes:

```bash
cat "outputs/project/logs/smoke-${SMOKE_JOB}.out"
cat "outputs/project/logs/smoke-${SMOKE_JOB}.err"
```

Do not submit the full grid until the gradient check and smoke configuration finish successfully.

### 5. Submit the 15 training configurations

The `%2` limit runs at most two H100 jobs concurrently. Reduce it if your allocation or fair-share
policy requires lower concurrency.

```bash
TRAIN_JOB=$(sbatch --parsable --array='0-14%2' slurm/train_array.sbatch)
echo "Training array: ${TRAIN_JOB}"
```

Training outputs are written under:

```text
outputs/project/train/<scene>/mask_<penalty>.*
outputs/project/logs/train-<array-job>_<task>.{out,err}
```

The Ficus `0.016` task is expected to close every Gaussian at approximately iteration 12,300 and
may finish without a PLY. This is a scientific negative result, not a reason to discard the rest of
the array. Its structured record is retained in `config/project_failures.json`.

Monitor the array with:

```bash
squeue -j "${TRAIN_JOB}"
sacct -j "${TRAIN_JOB}" --format=JobID,State,Elapsed,MaxRSS,AllocTRES%60
```

### 6. Submit importance pruning and evaluation

Post-processing computes accumulated \(T_i\alpha_i\) importance, writes the four relative-pruning
arms, constructs the 120,000- and 60,000-Gaussian matched-budget arms when possible, renders all
200 test views, and calculates PSNR, SSIM, and LPIPS.

Use an `afterany` dependency because task 8 is an expected collapse. The post-processing array
omits only task 8 and starts after the entire training array has reached a terminal state:

```bash
POST_JOB=$(sbatch --parsable \
  --dependency="afterany:${TRAIN_JOB}" \
  --array='0-7,9-14%2' \
  slurm/postprocess_array.sbatch)
echo "Post-processing array: ${POST_JOB}"
```

`postprocess_array.sbatch` passes `--skip-existing`, so a resubmission preserves completed PLY,
JSON, diagnostic, render, and metric outputs and reruns only missing work.

Monitor the array with:

```bash
squeue -j "${POST_JOB}"
sacct -j "${POST_JOB}" --format=JobID,State,Elapsed,MaxRSS,AllocTRES%60
```

Post-processing creates:

```text
outputs/project/pruned/<scene>/<mask>/<arm>.ply
outputs/project/pruned/<scene>/<mask>/<arm>.json
outputs/project/pruned/<scene>/<mask>/*.importance.npz
outputs/project/eval/<scene>/<mask>/<arm>/metrics.csv
outputs/project/eval/<scene>/<mask>/<arm>/renders/*.png
```

### 7. Run aggregate analysis

Submit analysis only after every non-collapsed post-processing task succeeds. The following command
is portable across repository locations and avoids any author-specific working directory:

```bash
ANALYZE_JOB=$(sbatch --parsable \
  --dependency="afterok:${POST_JOB}" \
  --chdir="${PWD}" \
  --job-name=cp4282-analyze \
  --partition=normal \
  --cpus-per-task=2 \
  --mem=16G \
  --time=01:00:00 \
  --output='outputs/project/logs/analyze-%j.out' \
  --error='outputs/project/logs/analyze-%j.err' \
  --wrap='.venv/bin/python scripts/analyze_project_results.py')
echo "Analysis job: ${ANALYZE_JOB}"
```

Alternatively, after post-processing has completed, analysis is light enough to run in an
appropriate CPU allocation:

```bash
.venv/bin/python scripts/analyze_project_results.py
```

The analysis writes:

```text
outputs/project/analysis/
├── all_results.csv
├── matched_comparisons.csv
├── excluded_settings.csv
├── gaussian_diagnostics.csv
└── report.md
```

### 8. Verify the reproduced experiment

First confirm the expected number of completed evaluation CSV files:

```bash
find outputs/project/eval -name metrics.csv -type f | wc -l
```

The expected value is:

```text
78
```

Then inspect the automatic result summary and structured exclusions:

```bash
cat outputs/project/analysis/report.md
sed -n '1,20p' outputs/project/analysis/excluded_settings.csv
```

The reproduced result should have the same qualitative ordering even if small floating-point
differences occur:

- Ficus with mask penalty `0.004` wins PSNR, SSIM, and LPIPS at both 40% and 20% retention;
- importance-only pruning wins all three metrics for Drums and Lego at both budgets;
- Drums `0.016`, Ficus `0.008`, and Lego `0.016` overshoot below the common budgets;
- Ficus `0.016` is recorded as a complete collapse;
- the exact retained counts are 120,000 and 60,000 for every eligible matched-budget arm.

For exact numerical comparison, compare the generated files against the tables in
[project_report.md](project_report.md) or [trainer/README.md](trainer/README.md).

## Resume or rerun part of the experiment

### Training

Training jobs do not use `--skip-existing`. Before resubmitting a task, inspect its log and existing
checkpoint/PLY files. Resubmit only the required task IDs:

```bash
sbatch --array='3,11' slurm/train_array.sbatch
```

Do not delete successful outputs merely because another array task failed.

### Post-processing and evaluation

Post-processing is designed to resume safely:

```bash
sbatch --array='0-7,9-14%2' slurm/postprocess_array.sbatch
```

Existing completed metrics are skipped, while missing PLY or JSON artifacts are checked
independently and regenerated as required.

### Analysis

Analysis can be rerun at any time after the expected metadata and metrics exist. It overwrites only
the aggregate CSV and Markdown summaries under `outputs/project/analysis/`.

## Run one configuration manually

For debugging, train and post-process a single generated configuration on an allocated GPU node:

```bash
source .venv/bin/activate
python 3dgs_trainer.py config/project/ficus_mask_0p004.yaml
python scripts/postprocess_model.py config/project/ficus_mask_0p004.yaml --skip-existing
```

The second command performs both importance scoring and all configured evaluation arms for that
training result. It is not suitable for a login node because it uses the GPU and renders the full
test set.

## Lightweight local checks

The full grid requires GPUs, but syntax, setup, and CPU gradient checks can be run locally:

```bash
cd trainer
source .venv/bin/activate
python -m compileall -q 3dgs_trainer.py importance_prune.py scripts shared
python scripts/check_setup.py
python 3dgs_gradient_check_gpu.py --device cpu
python 3dgs_trainer.py config/3dgs_smoke_mask_cpu.yaml
```

The CPU smoke configuration is intentionally tiny and does not reproduce the reported scientific
results; it only checks that the training and mask paths execute.

## View the presentation

From the repository root—not `trainer/`—serve the static presentation:

```bash
cd ..
python3 -m http.server 8000 --directory presentation
```

Open <http://localhost:8000>. Use the arrow keys, Page Up/Page Down, or Space to navigate. The full
report, speaker notes, and printable notes are available at:

- [project_report.md](project_report.md)
- [presentation_script.md](presentation_script.md)
- [presentation_script_print.pdf](presentation_script_print.pdf)

## Reproducibility notes and limitations

- **One seed:** every configuration uses seed 7. The experiment compares arms under a common seed
  but does not estimate variation across independent training runs.
- **No recovery fine-tuning:** matched models are evaluated immediately after importance pruning.
  The reported result measures immediate pruning safety, not recovery capacity.
- **Three scenes:** the results demonstrate scene dependence but do not establish a universal scene
  classifier.
- **Penalty search cost:** a learned-mask setting requires a separate training run for each
  \(\lambda_{\mathrm{mask}}\). Importance pruning is therefore the lower-cost default unless a
  short mask pilot provides evidence of a gain.
- **Expected collapse:** Ficus at `0.016` intentionally has no matched-budget evaluation because the
  training population reaches zero.
- **Version sensitivity:** dependency ranges are specified, but the environment is not fully
  locked. Preserve `pip freeze`, `nvidia-smi`, and Slurm accounting output when exact environment
  provenance is required.

## Key implementation files

- `trainer/3dgs_trainer.py`: Warp 3DGS training, learned masks, straight-through mask gradients,
  densification, cleanup, and diagnostics.
- `trainer/importance_prune.py`: accumulated \(T_i\alpha_i\) importance, exact-count pruning, and
  per-Gaussian diagnostics.
- `trainer/scripts/download_project_data.sh`: complete dataset/config preparation entry point.
- `trainer/scripts/prepare_project_data.py`: deterministic train/validation split.
- `trainer/scripts/generate_project_configs.py`: fixed 15-arm experiment grid.
- `trainer/scripts/postprocess_model.py`: importance scoring, relative pruning, exact-budget
  pruning, test rendering, and metric evaluation.
- `trainer/scripts/analyze_project_results.py`: aggregate rankings, exclusions, and diagnostics.
- `trainer/slurm/train_array.sbatch`: H100 training array.
- `trainer/slurm/postprocess_array.sbatch`: H100 importance/evaluation array.
- `trainer/slurm/smoke.sbatch`: A100 CUDA smoke test.

## Citation and course context

This project builds on the CP4282 Gaussian Splatting course trainer and studies learned-mask and
importance-based compaction. The implementation repository originated from
[dingZvel/cp4282-as2](https://github.com/dingZvel/cp4282-as2). The report cites the relevant
compaction literature and describes which components were implemented specifically for this
project.
