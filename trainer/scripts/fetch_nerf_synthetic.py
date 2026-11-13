"""Fetch the three project scenes, preferring the original NeRF Google Drive archive."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

import gdown
# The cluster login nodes enforce a low thread limit.  Hugging Face's Xet client otherwise starts
# a large native thread pool before `snapshot_download(max_workers=...)` can constrain Python's
# own workers, so select the ordinary HTTP backend before importing the hub package.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download


SCENES = ("lego", "ficus", "drums")
ORIGINAL_FILE_ID = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
MIRROR_REPOSITORY = "pablovela5620/nerf-synthetic-mirror"


def complete(root: Path) -> bool:
    return all(
        (root / scene / "transforms_train.json").exists()
        and (root / scene / "transforms_test.json").exists()
        and any((root / scene / "train").glob("*.png"))
        and any((root / scene / "test").glob("*.png"))
        for scene in SCENES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = destination / "nerf_synthetic"
    if complete(extracted):
        print(f"NeRF Synthetic scenes already complete at {extracted}")
        return

    archive = destination / "nerf_synthetic.zip"
    original_error = None
    try:
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            archive.unlink(missing_ok=True)
            result = gdown.download(id=ORIGINAL_FILE_ID, output=str(archive), quiet=False)
            if result is None:
                raise RuntimeError("Google Drive did not provide the public archive.")
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    except Exception as error:  # The original link is frequently quota-limited or disabled.
        original_error = error

    if not complete(extracted):
        print(f"Original archive unavailable ({original_error}); using {MIRROR_REPOSITORY}.")
        snapshot_download(
            repo_id=MIRROR_REPOSITORY,
            repo_type="dataset",
            local_dir=extracted,
            allow_patterns=[
                pattern
                for scene in SCENES
                for pattern in (
                    f"{scene}/transforms_train.json",
                    f"{scene}/transforms_test.json",
                    f"{scene}/train/**",
                    f"{scene}/test/**",
                )
            ],
            ignore_patterns=[
                pattern
                for scene in SCENES
                for pattern in (
                    f"{scene}/**/*_depth_*.png",
                    f"{scene}/**/*_normal_*.png",
                )
            ],
            max_workers=2,
        )
    if not complete(extracted):
        raise RuntimeError(f"Downloaded data is incomplete under {extracted}")
    print(f"Fetched {', '.join(SCENES)} into {extracted}")


if __name__ == "__main__":
    main()
