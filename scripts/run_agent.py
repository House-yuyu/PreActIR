#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from preactir.agent.factory import build_agent
from preactir.config import load_config
from preactir.utils.image import load_image, save_image
from preactir.utils.io import ensure_dir, list_images, write_jsonl
from preactir.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore arbitrary images with the closed-loop PreActIR agent.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", default=None)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Single degraded image.")
    source.add_argument("--input-dir", help="Directory of degraded images.")
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--native-resolution", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    agent = build_agent(
        cfg,
        belief_checkpoint=args.belief_checkpoint,
        world_checkpoint=args.world_checkpoint,
        verifier_checkpoint=args.verifier_checkpoint,
        device=device,
    )

    paths = [Path(args.input)] if args.input else list_images(args.input_dir)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("No input images found")

    output_root = ensure_dir(args.output)
    restored_root = ensure_dir(output_root / "restored")
    image_size = None if args.native_resolution else int(cfg.data.image_size)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(tqdm(paths, desc="PreActIR inference")):
        image = load_image(path, image_size)
        restored, trace = agent.restore(image)
        output_path = restored_root / f"{index:06d}_{path.stem}.png"
        save_image(output_path, restored)
        rows.append(
            {
                "id": f"{index:06d}_{path.stem}",
                "input_path": str(path.resolve()),
                "output_path": str(output_path.resolve()),
                **trace,
            }
        )
    write_jsonl(output_root / "traces.jsonl", rows)
    summary = {
        "num_images": len(rows),
        "mean_tool_calls": sum(row["tool_calls"] for row in rows) / len(rows),
        "mean_accepted_calls": sum(row["accepted_calls"] for row in rows) / len(rows),
        "mean_rejected_calls": sum(row["rejected_calls"] for row in rows) / len(rows),
        "termination_counts": {
            reason: sum(int(row["termination"] == reason) for row in rows)
            for reason in sorted({row["termination"] for row in rows})
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
