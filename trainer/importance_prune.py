"""Prune a trained 3DGS PLY by accumulated front-to-back blending importance.

The score matches the project proposal:

    U_i = sum_{v,p} T_{i,v}(p) * alpha_{i,v}(p)

Only training cameras are used.  The command writes an exact-count pruned PLY, a compressed NPZ
containing per-Gaussian diagnostics, and a JSON summary suitable for experiment aggregation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import warp as wp
import yaml


ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from gaussian_set import GaussianSet
from training_config import TrainingConfig


def load_trainer_module():
    """Load 3dgs_trainer.py despite its filename beginning with a digit."""
    path = ROOT / "3dgs_trainer.py"
    spec = importlib.util.spec_from_file_location("warp_3dgs_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trainer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def subset(scene: GaussianSet, keep: np.ndarray) -> GaussianSet:
    return GaussianSet(
        means=scene.means[keep],
        scales=scene.scales[keep],
        rotations=scene.rotations[keep],
        opacities=scene.opacities[keep],
        colors=scene.colors[keep],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prune a trained PLY using accumulated T*alpha importance."
    )
    parser.add_argument("config", type=Path, help="Training YAML that defines data/rendering.")
    parser.add_argument("input_ply", type=Path, help="Trained PLY to rank and prune.")
    parser.add_argument("output_ply", type=Path, help="Destination for the pruned PLY.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--prune-fraction", type=float,
        help="Fraction of input Gaussians to remove; the retained count is rounded down.",
    )
    target.add_argument("--target-count", type=int, help="Exact number of Gaussians to retain.")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="Override runtime.arch in the YAML (default: use the YAML).",
    )
    parser.add_argument(
        "--max-views", type=int, default=None,
        help="Use only the first N training views (debugging only; omit for experiments).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = TrainingConfig.load(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise SystemExit(f"Invalid training configuration: {error}") from error

    input_ply = args.input_ply.expanduser().resolve()
    output_ply = args.output_ply.expanduser().resolve()
    if output_ply.suffix.lower() != ".ply":
        raise SystemExit("output_ply must have a .ply suffix.")
    scene = GaussianSet.from_ply(str(input_ply))
    original_count = len(scene.means)
    if original_count < 1:
        raise SystemExit("The input PLY contains no Gaussians.")

    if args.prune_fraction is not None:
        if not 0.0 <= args.prune_fraction < 1.0:
            raise SystemExit("--prune-fraction must be in [0, 1).")
        target_count = int(np.floor(original_count * (1.0 - args.prune_fraction)))
    else:
        target_count = int(args.target_count)
    if not 1 <= target_count <= original_count:
        raise SystemExit(
            f"Target count must be between 1 and {original_count:,}; got {target_count:,}."
        )

    trainer_module = load_trainer_module()
    model = config["model"]
    background = (
        np.ones(3, dtype=np.float32)
        if model["background"] == "white"
        else np.zeros(3, dtype=np.float32)
    )
    targets, cameras, focal, distance, frame_paths = trainer_module.load_views(
        config["paths"]["data"], model["width"], model["height"], background,
    )
    if args.max_views is not None:
        if args.max_views < 1:
            raise SystemExit("--max-views must be positive.")
        targets = targets[: args.max_views]
        cameras = cameras[: args.max_views]
        frame_paths = frame_paths[: args.max_views]

    wp.init()
    requested = config["runtime"]["arch"] if args.device == "auto" else args.device
    device = "cpu" if requested == "cpu" else "cuda:0"
    if requested == "gpu" and not wp.is_cuda_available():
        raise SystemExit("GPU importance pruning requested, but Warp cannot find CUDA.")
    scene_radius = model["scene_radius"] or distance * 0.35
    training_options = {
        "compact_box": config["compact_box"],
        "learned_mask": {
            **config["learned_mask"],
            # Closed learned masks have already been physically removed from the saved PLY.
            "enabled": False,
            "penalty": 0.0,
        },
        "multiview_adc": {**config["multiview_adc"], "enabled": False},
        "sparse": {**config["sparse"], "enabled": False},
    }
    trainer = trainer_module.WarpImageTrainer(
        targets=targets,
        cameras=cameras,
        focal=focal,
        scene_radius=scene_radius,
        capacity=original_count,
        initial_splats=original_count,
        device=device,
        seed=config["runtime"]["seed"],
        init_scene=scene,
        background=background,
        learning_rates=config["learning_rates"],
        tile_pair_capacity=model["tile_pair_capacity"],
        training_options=training_options,
    )
    view_ids = np.arange(len(targets), dtype=np.int32)
    importance, alpha_sum, pixel_count, visible_views = trainer.accumulate_importance(view_ids)

    # Stable ordering makes tied zero-importance Gaussians deterministic by original PLY index.
    ranking = np.argsort(importance, kind="stable")
    keep = np.ones(original_count, dtype=bool)
    keep[ranking[: original_count - target_count]] = False
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    written = subset(scene, keep).to_ply(str(output_ply))

    diagnostics_path = output_ply.with_suffix(".importance.npz")
    mean_transmittance = np.divide(
        importance,
        alpha_sum,
        out=np.zeros_like(importance),
        where=alpha_sum > 0.0,
    )
    np.savez_compressed(
        diagnostics_path,
        importance=importance,
        accumulated_alpha=alpha_sum,
        alpha_weighted_mean_transmittance=mean_transmittance,
        contributing_pixels=pixel_count,
        visible_views=visible_views,
        retained=keep,
        means=scene.means,
        scales=scene.scales,
        opacities=scene.opacities,
    )
    summary = {
        "input_ply": str(input_ply),
        "output_ply": str(output_ply),
        "diagnostics": str(diagnostics_path),
        "data": str(config["paths"]["data"]),
        "device": str(trainer.device),
        "resolution": [int(model["width"]), int(model["height"])],
        "training_views": len(view_ids),
        "frame_paths": frame_paths,
        "original_count": original_count,
        "retained_count": written,
        "removed_count": original_count - written,
        "retained_fraction": written / original_count,
        "importance_threshold": (
            float(np.min(importance[keep])) if written else None
        ),
        "zero_importance_count": int(np.count_nonzero(importance == 0.0)),
    }
    summary_path = output_ply.with_suffix(".importance.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
