"""Score one model once, create fraction and matched-count pruning arms, and evaluate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from gaussian_set import GaussianSet
from training_config import TrainingConfig


FRACTIONS = (0.0, 0.25, 0.50, 0.75)
MATCHED_RETENTIONS = (0.40, 0.20)


def fraction_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def subset(scene: GaussianSet, keep: np.ndarray) -> GaussianSet:
    return GaussianSet(
        means=scene.means[keep],
        scales=scene.scales[keep],
        rotations=scene.rotations[keep],
        opacities=scene.opacities[keep],
        colors=scene.colors[keep],
    )


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def materialize_and_evaluate(
    *, scene: GaussianSet, importance: np.ndarray, ranking: np.ndarray, target_count: int,
    label: str, metadata: dict, destination: Path, evaluation_root: Path,
    trained_ply: Path, scored_npz: Path, config: TrainingConfig, metrics: list[str],
    skip_existing: bool,
) -> None:
    """Write one exact-count arm and evaluate it on the untouched test manifest."""
    if not 1 <= target_count <= len(scene.means):
        raise RuntimeError(
            f"Arm {label} requests {target_count:,} Gaussians from a model containing "
            f"{len(scene.means):,}. A learned-mask run below the shared budget cannot be "
            "compared fairly by adding Gaussians back."
        )
    output_ply = destination / f"{label}.ply"
    metadata_path = destination / f"{label}.json"
    keep = np.ones(len(scene.means), dtype=bool)
    keep[ranking[: len(scene.means) - target_count]] = False
    if not (skip_existing and output_ply.exists()):
        subset(scene, keep).to_ply(str(output_ply))
    if not (skip_existing and metadata_path.exists()):
        metadata_path.write_text(json.dumps({
            "status": "completed",
            "input_ply": str(trained_ply),
            "output_ply": str(output_ply),
            "original_count": len(scene.means),
            "retained_count": target_count,
            "importance_threshold": float(np.min(importance[keep])),
            "shared_diagnostics": str(scored_npz),
            **metadata,
        }, indent=2) + "\n")

    evaluation = evaluation_root / label
    csv_path = evaluation / "metrics.csv"
    if skip_existing and csv_path.exists():
        return
    run([
        sys.executable,
        str(ROOT / "3dgs_evaluator.py"),
        str(output_ply),
        str(config["paths"]["data"]),
        "--manifest", "transforms_test.json",
        "--metrics", *metrics,
        "--arch", "gpu",
        "--width", str(config["model"]["width"]),
        "--height", str(config["model"]["height"]),
        "--output-dir", str(evaluation / "renders"),
        "--csv", str(csv_path),
    ])


def record_overshot_budget(
    *, label: str, destination: Path, trained_ply: Path, original_count: int,
    reference_ply: Path, reference_count: int, retention: float, target_count: int,
) -> None:
    """Record a learned-mask model that is already smaller than the matched target."""
    metadata_path = destination / f"{label}.json"
    metadata_path.write_text(json.dumps({
        "status": "overshot_budget",
        "arm_type": "matched_budget",
        "input_ply": str(trained_ply),
        "original_count": original_count,
        "retained_count": original_count,
        "reference_ply": str(reference_ply),
        "reference_count": reference_count,
        "target_retention": retention,
        "nominal_target_count": target_count,
        "reason": (
            f"Learned masking retained only {original_count:,} Gaussians, below the exact "
            f"{retention:.0%} target of {target_count:,}; Gaussians cannot be added back."
        ),
    }, indent=2) + "\n")
    print(f"Skipping {label}: {json.loads(metadata_path.read_text())['reason']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    trained_ply = config["paths"]["output"].with_suffix(".ply")
    if not trained_ply.exists():
        raise SystemExit(f"Trained model does not exist: {trained_ply}")

    scene_name = config["paths"]["data"].name
    mask_name = trained_ply.stem
    destination = ROOT / "outputs" / "project" / "pruned" / scene_name / mask_name
    destination.mkdir(parents=True, exist_ok=True)
    scored_ply = destination / "importance_0p00.ply"
    scored_npz = scored_ply.with_suffix(".importance.npz")
    if not (args.skip_existing and scored_npz.exists()):
        run([
            sys.executable,
            str(ROOT / "importance_prune.py"),
            str(args.config),
            str(trained_ply),
            str(scored_ply),
            "--prune-fraction", "0",
            "--device", "gpu",
        ])

    scene = GaussianSet.from_ply(str(trained_ply))
    diagnostics = np.load(scored_npz)
    importance = diagnostics["importance"]
    ranking = np.argsort(importance, kind="stable")
    evaluation_root = ROOT / "outputs" / "project" / "eval" / scene_name / mask_name
    for fraction in FRACTIONS:
        tag = fraction_tag(fraction)
        target_count = int(np.floor(len(scene.means) * (1.0 - fraction)))
        materialize_and_evaluate(
            scene=scene, importance=importance, ranking=ranking, target_count=target_count,
            label=f"importance_{tag}",
            metadata={"arm_type": "fraction", "importance_fraction": fraction},
            destination=destination, evaluation_root=evaluation_root,
            trained_ply=trained_ply, scored_npz=scored_npz, config=config,
            metrics=args.metrics, skip_existing=args.skip_existing,
        )

    unpruned_ply = trained_ply.parent / "mask_none.ply"
    if not unpruned_ply.exists():
        raise RuntimeError(f"Matched-count reference does not exist: {unpruned_ply}")
    unpruned_count = len(GaussianSet.from_ply(str(unpruned_ply)).means)
    for retention in MATCHED_RETENTIONS:
        nominal_target_count = int(np.floor(unpruned_count * retention))
        label = f"budget_{fraction_tag(retention)}"
        if len(scene.means) < nominal_target_count:
            record_overshot_budget(
                label=label, destination=destination, trained_ply=trained_ply,
                original_count=len(scene.means), reference_ply=unpruned_ply,
                reference_count=unpruned_count, retention=retention,
                target_count=nominal_target_count,
            )
            continue
        materialize_and_evaluate(
            scene=scene, importance=importance, ranking=ranking,
            target_count=nominal_target_count, label=label,
            metadata={
                "arm_type": "matched_budget",
                "reference_ply": str(unpruned_ply),
                "reference_count": unpruned_count,
                "target_retention": retention,
                "nominal_target_count": nominal_target_count,
                "actual_target_count": nominal_target_count,
                "actual_retention": retention,
            },
            destination=destination, evaluation_root=evaluation_root,
            trained_ply=trained_ply, scored_npz=scored_npz, config=config,
            metrics=args.metrics, skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
