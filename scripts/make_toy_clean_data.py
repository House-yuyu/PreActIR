#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from preactir.utils.image import save_image
from preactir.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small procedural clean-image splits for a smoke test.")
    parser.add_argument("--output", default="data/toy_clean")
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_image(size: int, rng: np.random.Generator, index: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = 0.3 + 0.35 * np.sin(xx / (7.0 + index % 5))
    image[..., 1] = 0.4 + 0.3 * np.cos(yy / (9.0 + index % 7))
    image[..., 2] = 0.5 + 0.25 * np.sin((xx + yy) / (11.0 + index % 3))
    image = np.clip(image, 0.0, 1.0)
    for _ in range(8):
        color = tuple(float(v) for v in rng.uniform(0.05, 0.95, size=3))
        if rng.random() < 0.5:
            p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
            p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
            cv2.rectangle(image, p1, p2, color, thickness=int(rng.integers(1, 5)))
        else:
            center = tuple(int(v) for v in rng.integers(0, size, size=2))
            radius = int(rng.integers(max(3, size // 25), max(4, size // 6)))
            cv2.circle(image, center, radius, color, thickness=-1)
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        split_dir = ensure_dir(Path(args.output) / split)
        for index in range(count):
            save_image(split_dir / f"toy_{index:04d}.png", make_image(args.size, rng, index))
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
