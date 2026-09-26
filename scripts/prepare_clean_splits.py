#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from preactir.utils.io import ensure_dir, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split an image directory into train/val/test folders.")
    parser.add_argument("--input", required=True, help="Directory containing clean images.")
    parser.add_argument("--output", required=True, help="Output root containing train/val/test.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(args.input)
    if not images:
        raise RuntimeError("No images found")
    if not 0 < args.train_ratio < 1 or not 0 <= args.val_ratio < 1:
        raise ValueError("Invalid split ratios")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = random.Random(args.seed)
    rng.shuffle(images)
    train_end = int(len(images) * args.train_ratio)
    val_end = train_end + int(len(images) * args.val_ratio)
    assignments = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    output_root = Path(args.output)
    for split, split_images in assignments.items():
        split_dir = ensure_dir(output_root / split)
        for index, source in enumerate(split_images):
            destination = split_dir / f"{index:07d}_{source.stem}{source.suffix.lower()}"
            if destination.exists():
                continue
            if args.mode == "copy":
                shutil.copy2(source, destination)
            else:
                destination.symlink_to(source.resolve())
        print(f"{split}: {len(split_images)} images")


if __name__ == "__main__":
    main()
