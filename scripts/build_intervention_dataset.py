#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preactir.config import load_config
from preactir.data.builder import BuilderOptions, InterventionDatasetBuilder
from preactir.tools.registry import build_registry_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build InterveneIR-style state and action-transition manifests from clean images."
    )
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument(
        "--clean-root",
        required=True,
        help="Root containing train/val/test clean-image folders.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to data.root in the YAML config.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    registry = build_registry_from_config(cfg)
    options = BuilderOptions(
        image_size=int(cfg.data.image_size),
        max_degradations=int(cfg.data.max_degradations),
        states_per_image=int(cfg.data.states_per_image),
        max_actions_per_state=int(cfg.data.max_actions_per_state),
        spatial_probability=float(cfg.data.spatial_probability),
        distractor_action_probability=float(cfg.data.distractor_action_probability),
        acceptance_target_gain=float(cfg.data.acceptance_target_gain),
        acceptance_max_damage=float(cfg.data.acceptance_max_damage),
        strengths=tuple(float(value) for value in cfg.tools.strengths),
        seed=int(args.seed if args.seed is not None else cfg.seed),
    )
    builder = InterventionDatasetBuilder(
        clean_root=Path(args.clean_root),
        output_root=Path(args.output_root or cfg.data.root),
        degradation_names=list(cfg.degradations.names),
        registry=registry,
        options=options,
    )
    builder.build(splits=list(args.splits), limit_per_split=args.limit_per_split)
    print(f"Dataset written to: {Path(args.output_root or cfg.data.root).resolve()}")


if __name__ == "__main__":
    main()
