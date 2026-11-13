"""Create the project's deterministic 90/10 train/validation splits.

The original NeRF Synthetic test manifest remains untouched.  Ten cameras are selected from the
original training set by farthest-point sampling on camera direction; those frames become the
validation manifest and are therefore excluded from both optimisation and importance scoring.
Image directories are symlinked instead of copied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np


def farthest_camera_ids(frames: list[dict], count: int) -> list[int]:
    positions = np.asarray(
        [np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames]
    )
    directions = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12)
    chosen = [0]
    distance = np.sum((directions - directions[0]) ** 2, axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.sum((directions - directions[nxt]) ** 2, axis=1))
    return sorted(chosen)


def locate_scene(root: Path, name: str) -> Path:
    candidates = (root / name, root / "nerf_synthetic" / name)
    for candidate in candidates:
        if (candidate / "transforms_train.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find scene {name!r} under {root}")


def write_manifest(template: dict, frames: list[dict], path: Path) -> None:
    manifest = {**template, "frames": frames}
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def prepare_scene(source: Path, destination: Path, validation_count: int) -> None:
    train = json.loads((source / "transforms_train.json").read_text())
    frames = train["frames"]
    if not 1 <= validation_count < len(frames):
        raise ValueError(
            f"validation_count must be in [1, {len(frames) - 1}] for {source.name}"
        )
    validation_ids = set(farthest_camera_ids(frames, validation_count))
    training_frames = [frame for index, frame in enumerate(frames) if index not in validation_ids]
    validation_frames = [frame for index, frame in enumerate(frames) if index in validation_ids]

    destination.mkdir(parents=True, exist_ok=True)
    write_manifest(train, training_frames, destination / "transforms_train.json")
    write_manifest(train, validation_frames, destination / "transforms_val.json")
    shutil.copy2(source / "transforms_test.json", destination / "transforms_test.json")

    for directory in ("train", "val", "test"):
        source_directory = source / directory
        if not source_directory.exists():
            continue
        link = destination / directory
        if link.is_symlink():
            if link.resolve() != source_directory.resolve():
                raise FileExistsError(f"{link} points to {link.resolve()}, not {source_directory}")
        elif link.exists():
            raise FileExistsError(f"Refusing to replace existing non-symlink {link}")
        else:
            os.symlink(source_directory, link, target_is_directory=True)

    init_ply = source / "init.ply"
    if init_ply.exists():
        shutil.copy2(init_ply, destination / "init.ply")
    split = {
        "source": str(source),
        "training_count": len(training_frames),
        "validation_count": len(validation_frames),
        "validation_source_indices": sorted(validation_ids),
        "test_count": len(json.loads((source / "transforms_test.json").read_text())["frames"]),
    }
    (destination / "project_split.json").write_text(json.dumps(split, indent=2) + "\n")
    print(f"{source.name}: {len(training_frames)} train, {len(validation_frames)} validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_root", type=Path)
    parser.add_argument("destination_root", type=Path)
    parser.add_argument("--scenes", nargs="+", default=["lego", "ficus", "drums"])
    parser.add_argument("--validation-count", type=int, default=10)
    args = parser.parse_args()
    for scene in args.scenes:
        prepare_scene(
            locate_scene(args.source_root.expanduser().resolve(), scene),
            args.destination_root.expanduser().resolve() / scene,
            args.validation_count,
        )


if __name__ == "__main__":
    main()
