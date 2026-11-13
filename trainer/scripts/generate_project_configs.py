"""Generate the fixed project training grid for lego, ficus, and drums."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SCENES = ("lego", "ficus", "drums")
MASK_PENALTIES = (0.0, 0.002, 0.004, 0.008, 0.016)


def penalty_tag(value: float) -> str:
    return "none" if value == 0.0 else f"{value:.3f}".replace(".", "p")


def config_for(scene: str, penalty: float, data_root: Path, output_root: Path) -> dict:
    scene_data = (data_root / scene).resolve()
    init_ply = scene_data / "init.ply"
    return {
        "paths": {
            "data": str(scene_data),
            "output": str((output_root / "train" / scene / f"mask_{penalty_tag(penalty)}.png").resolve()),
            "init_ply": str(init_ply) if init_ply.exists() else None,
        },
        "runtime": {"arch": "gpu", "seed": 7},
        "model": {
            "width": 256,
            "height": 256,
            "capacity": 300_000,
            "initial_splats": 50_000,
            "background": "white",
        },
        "training": {"iterations": 20_000},
        "learned_mask": {
            "enabled": penalty > 0.0,
            "penalty": penalty,
            "threshold": 0.01,
            "initial_logit": 1.0,
        },
        "densification": {
            "interval": 100,
            "fraction": 0.2,
            "percent_dense": 0.01,
            "opacity_reset_interval": 0,
        },
        # Disable the trainer's separate multi-view ADC pruning so the only experimental
        # compaction variables are learned masks and the post-training importance fraction.
        "multiview_adc": {"enabled": False},
        "sparse": {"enabled": True, "samples_per_tile": 48},
        "convergence": {"enabled": False},
        "reporting": {
            "log_every": 100,
            "eval_every": 500,
            "eval_manifest": "transforms_val.json",
            "eval_views": 10,
            "eval_view_ids": list(range(10)),
            "snapshot_every": 2_000,
            "save_ply": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/project_split"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/project"))
    parser.add_argument("--config-root", type=Path, default=Path("config/project"))
    args = parser.parse_args()
    config_root = args.config_root.expanduser().resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        for penalty in MASK_PENALTIES:
            path = config_root / f"{scene}_mask_{penalty_tag(penalty)}.yaml"
            path.write_text(
                yaml.safe_dump(
                    config_for(scene, penalty, args.data_root, args.output_root),
                    sort_keys=False,
                )
            )
            print(path)


if __name__ == "__main__":
    main()
