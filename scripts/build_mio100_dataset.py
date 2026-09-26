#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from preactir.config import load_config
from preactir.data.builder import BuilderOptions
from preactir.data.mio100 import MiO100DatasetBuilder
from preactir.tools.paper_registry import inspect_paper_backend
from preactir.tools.registry import build_registry_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PreActIR manifests and intervention rollouts from AgenticIR's MiO100 release."
    )
    parser.add_argument("--config", default="configs/preactir_mio100.yaml")
    parser.add_argument("--mio-root", required=True, help="Extracted root containing HQ/train/test.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--transition-splits",
        nargs="*",
        default=["train", "val"],
        help="Splits for which tool rollout images are generated. Pass the flag with no values for manifests only.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.transition_splits and str(cfg.tools.get("backend", "classical")) == "agenticir_4kagent" and bool(
        cfg.tools.get("require_cached_assets", False)
    ):
        report = inspect_paper_backend(
            str(cfg.tools.external_root),
            str(cfg.tools.get("profile", "agenticir_default")),
        )
        if report["missing_lazy_assets"]:
            missing = [
                report["lazy_assets"][name]["path"] for name in report["missing_lazy_assets"]
            ]
            raise RuntimeError(
                "Paper rollout is blocked because required lazy-download weights are absent:\n- "
                + "\n- ".join(missing)
                + "\nRun scripts/check_paper_tools.py after copying/downloading them."
            )
    registry = build_registry_from_config(cfg, gpu_id=args.gpu_id)
    options = BuilderOptions(
        image_size=int(cfg.data.image_size),
        rollout_native_resolution=bool(cfg.data.get("rollout_native_resolution", False)),
        max_degradations=int(cfg.data.max_degradations),
        states_per_image=1,
        max_actions_per_state=int(cfg.data.max_actions_per_state),
        spatial_probability=0.0,
        distractor_action_probability=float(cfg.data.distractor_action_probability),
        acceptance_target_gain=float(cfg.data.acceptance_target_gain),
        acceptance_max_damage=float(cfg.data.acceptance_max_damage),
        strengths=tuple(float(value) for value in cfg.tools.strengths),
        seed=int(args.seed if args.seed is not None else cfg.seed),
    )
    output_root = Path(args.output_root or cfg.data.root)
    builder = MiO100DatasetBuilder(
        dataset_root=args.mio_root,
        output_root=output_root,
        degradation_names=list(cfg.degradations.names),
        registry=registry,
        options=options,
        val_fraction=args.val_fraction,
    )
    builder.build(
        splits=list(args.splits),
        transition_splits=set(args.transition_splits),
        limit_per_split=args.limit_per_split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(f"MiO100 PreActIR dataset written to: {output_root.resolve()}")


if __name__ == "__main__":
    main()
