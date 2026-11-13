"""Aggregate project metrics and diagnose which scene suits each pruning combination."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs" / "project"
METRIC_DIRECTIONS = {"psnr_db": -1.0, "ssim": -1.0, "lpips": 1.0}


def mean_metrics(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No evaluation rows in {path}")
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRIC_DIRECTIONS
        if metric in rows[0]
    }


def mask_penalty(mask_name: str) -> float:
    if mask_name == "mask_none":
        return 0.0
    return float(mask_name.removeprefix("mask_").replace("p", "."))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(outputs: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted((outputs / "eval").glob("*/mask_*/*/metrics.csv")):
        scene, mask, arm = metrics_path.parts[-4:-1]
        metadata_path = outputs / "pruned" / scene / mask / f"{arm}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing arm metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        row = {
            "scene": scene,
            "mask": mask,
            "mask_penalty": mask_penalty(mask),
            "arm": arm,
            "arm_type": metadata["arm_type"],
            "original_count": metadata["original_count"],
            "retained_count": metadata["retained_count"],
            "retained_fraction": (
                metadata["retained_count"] / metadata["original_count"]
            ),
            **mean_metrics(metrics_path),
        }
        for key in ("importance_fraction", "target_retention", "actual_retention"):
            if key in metadata:
                row[key] = metadata[key]
        rows.append(row)
    return rows


def collect_exclusions(outputs: Path) -> list[dict]:
    """Collect overshot budgets and training collapses excluded from matched comparisons."""
    rows = []
    for metadata_path in sorted((outputs / "pruned").glob("*/mask_*/budget_*.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("status") == "completed":
            continue
        scene, mask = metadata_path.parts[-3:-1]
        rows.append({
            "scene": scene,
            "mask": mask,
            "mask_penalty": mask_penalty(mask),
            "arm": metadata_path.stem,
            "status": metadata.get("status", "excluded"),
            "available_count": metadata.get("original_count", ""),
            "target_count": metadata.get("nominal_target_count", ""),
            "reason": metadata.get("reason", ""),
        })
    failure_path = ROOT / "config" / "project_failures.json"
    if failure_path.exists():
        for failure in json.loads(failure_path.read_text()):
            rows.append(failure)
    return rows


def ranked_matched_rows(rows: list[dict]) -> list[dict]:
    matched = [row.copy() for row in rows if row["arm_type"] == "matched_budget"]
    groups = sorted({(row["scene"], row["arm"]) for row in matched})
    output = []
    for scene, arm in groups:
        group = [row for row in matched if row["scene"] == scene and row["arm"] == arm]
        baseline = next((row for row in group if row["mask"] == "mask_none"), None)
        if baseline is None:
            raise RuntimeError(f"Importance-only baseline is missing for {scene}/{arm}")
        ranks = {row["mask"]: [] for row in group}
        for metric, direction in METRIC_DIRECTIONS.items():
            ordered = sorted(group, key=lambda row: direction * row[metric])
            for rank, row in enumerate(ordered, 1):
                ranks[row["mask"]].append(rank)
        for row in group:
            row["delta_psnr_db_vs_importance_only"] = row["psnr_db"] - baseline["psnr_db"]
            row["delta_ssim_vs_importance_only"] = row["ssim"] - baseline["ssim"]
            row["delta_lpips_vs_importance_only"] = row["lpips"] - baseline["lpips"]
            row["mean_metric_rank"] = float(np.mean(ranks[row["mask"]]))
        output.extend(sorted(group, key=lambda row: (row["mean_metric_rank"], row["mask_penalty"])))
    return output


def distribution(prefix: dict, values: dict[str, np.ndarray]) -> dict:
    row = dict(prefix)
    for name, array in values.items():
        array = np.asarray(array)
        if array.ndim > 1:
            array = np.linalg.norm(array, axis=1)
        if len(array):
            row.update({
                f"{name}_mean": float(np.mean(array)),
                f"{name}_median": float(np.median(array)),
                f"{name}_q10": float(np.quantile(array, 0.10)),
                f"{name}_q90": float(np.quantile(array, 0.90)),
            })
    return row


def diagnostic_rows(outputs: Path, matched: list[dict]) -> list[dict]:
    rows = []
    # Learned-mask records describe Gaussians at the moment the mask deleted them, before slots
    # could be reused by later densification. They are intentionally kept separate from the
    # final importance-pruned population rather than pretending the two are contemporaneous.
    learned_mask_sidecars = sorted((outputs / "train").glob("*/mask_*.ply.mask-pruned.npz"))
    # Runs launched before the final-sidecar naming fix used the preview PNG as their base. Keep
    # those diagnostics usable; a model has only one of these two final-sidecar names.
    learned_mask_sidecars += sorted(
        (outputs / "train").glob("*/mask_*.png.mask-pruned.npz")
    )
    for sidecar in learned_mask_sidecars:
        scene = sidecar.parent.name
        mask = sidecar.name.split(".", 1)[0]
        with np.load(sidecar) as data:
            rows.append(distribution(
                {"scene": scene, "mask": mask, "arm": "learned_mask_events",
                 "method": "learned_mask", "removed_count": len(data["opacities"])},
                {"opacity": data["opacities"], "scale_norm": data["scales"],
                 "soft_mask": data["soft_masks"]},
            ))

    for arm in matched:
        metadata_path = outputs / "pruned" / arm["scene"] / arm["mask"] / f"{arm['arm']}.json"
        metadata = json.loads(metadata_path.read_text())
        with np.load(metadata["shared_diagnostics"]) as data:
            remove_count = int(metadata["original_count"] - metadata["retained_count"])
            removed = np.argsort(data["importance"], kind="stable")[:remove_count]
            rows.append(distribution(
                {"scene": arm["scene"], "mask": arm["mask"], "arm": arm["arm"],
                 "method": "importance", "removed_count": remove_count},
                {"opacity": data["opacities"][removed],
                 "scale_norm": data["scales"][removed],
                 "importance": data["importance"][removed],
                 "visible_views": data["visible_views"][removed],
                 "contributing_pixels": data["contributing_pixels"][removed],
                 "mean_transmittance": data["alpha_weighted_mean_transmittance"][removed]},
            ))
    return rows


def report_markdown(matched: list[dict], exclusions: list[dict]) -> str:
    lines = [
        "# Project Result Summary", "",
        "Matched-count winners use the mean of their PSNR, SSIM, and LPIPS ranks. PSNR and ",
        "SSIM are ranked high-to-low; LPIPS is ranked low-to-high. Metric-specific deltas remain ",
        "in `matched_comparisons.csv` so a rank tie or disagreement is not hidden.", "",
        "| Scene | Budget | Best direction | Penalty | Count | PSNR | SSIM | LPIPS | Mean rank |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, arm in sorted({(row["scene"], row["arm"]) for row in matched}):
        group = [row for row in matched if row["scene"] == scene and row["arm"] == arm]
        winner = min(group, key=lambda row: (row["mean_metric_rank"], row["mask_penalty"]))
        direction = "importance only" if winner["mask"] == "mask_none" else "mask then importance"
        budget = float(winner.get("target_retention", winner["retained_fraction"]))
        lines.append(
            f"| {scene} | {budget:.0%} | {direction} | {winner['mask_penalty']:.4g} | "
            f"{int(winner['retained_count']):,} | {winner['psnr_db']:.3f} | "
            f"{winner['ssim']:.5f} | {winner['lpips']:.5f} | "
            f"{winner['mean_metric_rank']:.2f} |"
        )
    lines.extend([
        "", "## Excluded or degenerate settings", "",
        "These settings are retained as negative results but excluded from matched-count ranking ",
        "because they did not contain enough Gaussians to reach the requested budget.", "",
        "| Scene | Setting | Arm | Status | Available | Target | Reason |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for row in exclusions:
        lines.append(
            f"| {row['scene']} | {row['mask']} | {row.get('arm', '')} | "
            f"{row['status']} | {row.get('available_count', '')} | "
            f"{row.get('target_count', '')} | {row.get('reason', '')} |"
        )
    lines.extend([
        "", "## Interpretation guide", "",
        "A combined-method winner supports mask-first pruning for that scene and budget; an ",
        "importance-only winner indicates that learned masking did not improve the final ",
        "quality/count trade-off. Compare the Gaussian diagnostic table across scenes to relate ",
        "that outcome to opacity, scale, visibility, and transmittance. These are associations, ",
        "not causal claims, because the project uses one training seed per setting.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    outputs = args.outputs.resolve()
    destination = (args.destination or outputs / "analysis").resolve()
    rows = collect(outputs)
    if not rows:
        raise SystemExit(f"No completed evaluations found below {outputs / 'eval'}")
    matched = ranked_matched_rows(rows)
    exclusions = collect_exclusions(outputs)
    write_csv(destination / "all_results.csv", rows)
    write_csv(destination / "matched_comparisons.csv", matched)
    write_csv(destination / "excluded_settings.csv", exclusions)
    write_csv(destination / "gaussian_diagnostics.csv", diagnostic_rows(outputs, matched))
    (destination / "report.md").write_text(report_markdown(matched, exclusions))
    print(f"Wrote aggregate results and report to {destination}")


if __name__ == "__main__":
    main()
