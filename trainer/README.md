<!-- markdownlint-disable-file MD013 -->
# CP4282 3D Gaussian Splatting — Compaction Project

This repository extends the CP4282 Warp 3D Gaussian Splatting trainer to study scene-dependent
combinations of learned-mask pruning and post-training importance pruning. It retains the original
Assignment 2 documentation below, but the backward kernels, compaction methods, experiment
automation, and evaluation pipeline are implemented.

The complete experiment has finished. The main conclusion is:

- **Ficus:** a mild learned-mask penalty of `0.004` followed by importance pruning performs best at
  both 40% and 20% retention.
- **Drums and Lego:** importance pruning alone performs best at both budgets.
- **Strong mask penalties are unstable:** `0.016` leaves 16 Gaussians on Drums, collapses Ficus to
  zero, and leaves 84 on Lego. Ficus already overshoots to 39 Gaussians at `0.008`.

See the [full project report](../project_report.md) for the quantitative analysis, failure modes,
implemented fixes, limitations, and proposed follow-up experiments.

## Presentation

The static presentation is available at [presentation/index.html](../presentation/index.html). It
contains slide-style navigation, interactive matched-budget charts, representative test-view
comparisons at both 40% and 20% retention, penalty-collapse and pruning-robustness visuals, keyboard controls, fullscreen mode,
and a print layout suitable for PDF export. The matching 13–15 minute speaker notes are in
[presentation_script.md](../presentation_script.md).

From the project root, serve it locally with:

```bash
python3 -m http.server 8000 --directory presentation
```

Then open `http://localhost:8000`. Use the arrow keys, Page Up/Page Down, or Space to move between
slides. The page has no external runtime dependencies and can also be opened directly from disk.

## Project results

| Scene | Retention | Best direction | Penalty | Count | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|---:|---:|
| Drums | 20% | Importance only | 0 | 60,000 | 21.735 | 0.89094 | 0.09262 |
| Drums | 40% | Importance only | 0 | 120,000 | 22.128 | 0.89649 | 0.08432 |
| Ficus | 20% | Mask then importance | 0.004 | 60,000 | 22.311 | 0.90839 | 0.06691 |
| Ficus | 40% | Mask then importance | 0.004 | 120,000 | 22.446 | 0.91097 | 0.06592 |
| Lego | 20% | Importance only | 0 | 60,000 | 26.169 | 0.93626 | 0.04853 |
| Lego | 40% | Importance only | 0 | 120,000 | 26.799 | 0.94257 | 0.04032 |

The final analysis contains 78 evaluation arms over 200 test views, six structured budget
overshoots, and one complete training collapse. Results are in `outputs/project/analysis/`:

- `all_results.csv`
- `matched_comparisons.csv`
- `excluded_settings.csv`
- `gaussian_diagnostics.csv`
- `report.md`

## Project pipeline

The relevant implementation files are:

- `3dgs_trainer.py`: dense/sparse training, learned masks, straight-through gradients,
  densification inheritance, final cleanup, and diagnostic sidecars;
- `importance_prune.py`: accumulated \(T_i\alpha_i\) importance, exact-count pruning, and
  per-Gaussian diagnostics;
- `scripts/prepare_project_data.py`: deterministic 90-train/10-validation split while preserving
  the original 200-view test set;
- `scripts/generate_project_configs.py`: 15 controlled scene/penalty configurations;
- `scripts/postprocess_model.py`: percentage and exact-budget pruning plus test evaluation;
- `scripts/analyze_project_results.py`: aggregate rankings, exclusions, Gaussian diagnostics, and
  automatic Markdown summary;
- `slurm/`: cluster setup, training, post-processing, metadata repair, and analysis jobs.

Generate the dataset split and configurations with:

```bash
bash scripts/download_project_data.sh
python scripts/prepare_project_data.py
python scripts/generate_project_configs.py
```

The principal Slurm entry points are:

```bash
sbatch --array=0-14%2 slurm/train_array.sbatch
sbatch --array=0-7,9-14%2 slurm/postprocess_array.sbatch
sbatch --dependency=afterok:<post-job-id> slurm/analyze.sbatch
```

The post-processing array above reproduces the completed experiment and skips configuration 8,
Ficus at penalty `0.016`, because that training arm is intentionally recorded as a zero-Gaussian
collapse and therefore has no PLY to evaluate.

The experiment used H100 GPUs on the NUS Computing Cluster. The completed cluster result tree was
`/home/d/dingzhe/CP4282/project/outputs/project/`.

## Original assignment context

The repository began as starter code for the CP4282 Gaussian Splatting **training** assignment.

The missing sections described below were the original assignment tasks. This project version
implements the dense and sparse trainer paths needed by the compaction experiments; the historical
instructions are retained to document where the code came from.

## Setup

```bash
git clone https://github.com/weitsang/cp4282-as2.git
cd cp4282-as2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`torch` and `lpips` are in that list only for the evaluator's LPIPS metric. If you would rather not
pull PyTorch down, install everything above them and run the evaluator with `--metrics psnr ssim`.

The repository contains a small Lego dataset under `data/lego/`. The three manifests sit at the
top of that directory and name the images in the split directories beside them:

- `train/` — 100 posed training images, listed by `transforms_train.json`.
- `val/` — 100 validation images, listed by `transforms_val.json`, for training-time held-out checks.
- `test/` — 200 held-out images, listed by `transforms_test.json`, for final evaluation.
- `init.ply` — the 45,733-splat point cloud training starts from.

The three splits share no images.

The trainer reads **both** manifests. It optimises only against `train/`; it renders eight
held-out views from `val/` every `reporting.eval_every` iterations to report `fixed_eval`, so
that number measures generalisation rather than a second reading of the training loss. The
validation views are never used to compute a gradient. The trainer prints exactly which frames it
picked on its first line, and `reporting.eval_manifest: null` reverts to evaluating on training
views if you want the old behaviour.

## The assignment

The original assignment asked students to implement `render_backward` for the dense path and
`render_sparse_backward` for the sampled path. Both are implemented in this project version.

```bash
python 3dgs_trainer.py config/3dgs_training_gpu.yaml
```

Check your gradients against finite differences before judging a training run:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

In this project version the check passes on CPU and CUDA. Run it after changing a rendering,
masking, or optimization kernel so a training run is not used as the first gradient test.

The annotated walkthrough explains the whole file without giving the TODO solution:

- `3dgs_trainer_annotated.md`

Shared and support files:

- `shared/trainable_gaussian.py`: trainable splat parameter storage used by the trainer
  (documented in `shared_annotated.md`)
- `shared/training_config.py`: the YAML schema, defaults and validation
- `gaussian_first_tile_workspace_gpu.py`: tile record construction used by the trainer
- `config/3dgs_training_gpu.yaml`: the starter YAML for training runs. Its paths are resolved
  relative to the config file, and `shared/training_config.py` lists every key it may contain.
- `3dgs_renderer_v1.py`, `3dgs_1_syn_trainer.py`, `3dgs_k_syn_trainer.py`: Assignment 1's renderer
  and the two small Unit 8 synthetic trainers. **These carry Assignment 1's own unfinished
  `render` loop.** The synthetic trainers build their ground truth by calling it, so until you
  paste your Assignment 1 solution into `3dgs_renderer_v1.py` they train against an all-black
  target and their PSNR figures are meaningless. Assignment 2's task does not depend on any of
  these three files.

## Evaluating your output

`3dgs_evaluator.py` renders every camera in a manifest, saves the images, and reports the metrics
you ask for:

```bash
python 3dgs_evaluator.py outputs/warp-training.ply data/lego
```

It decides three things for you rather than making you pick a script:

- **Device.** `--arch auto` (the default) uses CUDA when Warp reports it and CPU otherwise. Force
  it with `--arch cpu` or `--arch gpu`; asking for gpu without CUDA is an error, not a slow
  fallback.
- **Appearance.** If a `<ply-stem>.sh.npz` sidecar sits beside the PLY, the render uses degree-2
  view-dependent colour. `--sh-degree 0` ignores it, `--sh-degree 2` requires it, and `--sh-npz`
  points at one stored elsewhere. This assignment produces no sidecar, so this stays off.
- **Metrics.** `--metrics` takes any of `psnr`, `ssim`, `lpips`; all three run by default. LPIPS
  needs PyTorch, so drop it with `--metrics psnr ssim` if you have not installed the optional
  requirements.

Use `--manifest transforms_train.json` or `--manifest transforms_val.json` to score against those
splits instead of the held-out test split, and `--width`/`--height` to render at a different
resolution than the default. The test manifest has 200 views; on CPU, evaluate a smaller split
first.

## Running checks

```bash
python -m compileall src scripts shared 3dgs_trainer.py 3dgs_gradient_check_gpu.py gaussian_first_tile_workspace_gpu.py 3dgs_1_syn_trainer.py 3dgs_k_syn_trainer.py image_metrics.py 3dgs_evaluator.py 3dgs_renderer_v1.py
python scripts/check_setup.py
```

Verify the implemented backward pass with a finite-difference gradient check:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
python 3dgs_gradient_check_gpu.py --device cuda:0
```

Use `--help` on each script for its command-line arguments. Start with low resolution and a small
iteration count while debugging.
