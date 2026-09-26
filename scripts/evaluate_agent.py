#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from preactir.agent.factory import build_agent
from preactir.config import load_config
from preactir.utils.image import align_to_reference, load_image, save_image
from preactir.utils.io import ensure_dir, read_jsonl, write_jsonl
from preactir.utils.metrics import aggregate_metric_dicts, psnr, quality_vector, ssim
from preactir.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate closed-loop restoration on a generated InterveneIR split.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--belief-checkpoint", required=True)
    parser.add_argument(
        "--belief-base-channels",
        type=int,
        default=None,
        help="Belief encoder width used to create the supplied checkpoint.",
    )
    parser.add_argument(
        "--belief-global-backbone",
        choices=["none", "clip_vit_b32"],
        default=None,
    )
    parser.add_argument("--belief-clip-checkpoint", default=None)
    parser.add_argument(
        "--belief-clip-fusion-mode",
        choices=["concat", "replace", "linear"],
        default=None,
    )
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--verifier-checkpoint", default=None)
    parser.add_argument(
        "--learned-verifier-threshold",
        type=float,
        default=None,
        help="Override the learned commit threshold with a frozen inner-validation value.",
    )
    parser.add_argument(
        "--learned-verifier-mode",
        choices=("weighted", "and", "learned", "learned_progress_guard"),
        default=None,
        help="Evaluation-only override for the post-action verifier composition rule.",
    )
    parser.add_argument(
        "--learned-min-relative-target-gain",
        type=float,
        default=None,
        help=(
            "Minimum observed relative target-belief reduction required when "
            "learned_progress_guard overrides heuristic progress."
        ),
    )
    parser.add_argument(
        "--block-target-on-progress-guard-reject",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "After a learned-progress-guard rejection, suppress other tools for "
            "that target until an accepted transition changes the image state."
        ),
    )
    parser.add_argument(
        "--belief-thresholds-file",
        default=None,
        help=(
            "Frozen evaluate_belief.py calibration JSON. Per-degradation presence/severity "
            "gates are extracted from calibration.per_degradation."
        ),
    )
    parser.add_argument(
        "--belief-presence-threshold",
        type=float,
        default=None,
        help="Scalar belief presence gate; mutually exclusive with --belief-thresholds-file.",
    )
    parser.add_argument(
        "--belief-severity-threshold",
        type=float,
        default=None,
        help="Scalar belief severity gate; mutually exclusive with --belief-thresholds-file.",
    )
    parser.add_argument(
        "--disable-diagnostic-probes",
        action="store_true",
        help="Disable handcrafted uncertainty probes for this evaluation.",
    )
    parser.add_argument("--gamma-guard", default=None)
    parser.add_argument(
        "--risk-calibration",
        default=None,
        help="Frozen v5 pre-action harmful-risk calibration JSON.",
    )
    parser.add_argument(
        "--enabled-tools",
        nargs="+",
        default=None,
        help="Restrict closed-loop candidates to this predeclared safe action subset.",
    )
    parser.add_argument(
        "--enabled-tools-file",
        default=None,
        help="Frozen JSON artifact containing an enabled_tools list.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/agent_eval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--harm-weight",
        type=float,
        default=None,
        help="Evaluation-only override for the planner harmful-risk penalty.",
    )
    parser.add_argument(
        "--min-planning-utility",
        type=float,
        default=None,
        help="Evaluation-only override for the no-op/execute utility boundary.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state-id", default=None, help="Evaluate one exact state for diagnosis.")
    parser.add_argument(
        "--source-ids",
        nargs="+",
        default=None,
        help="Evaluate only these source identities (for leakage-free protocol subsets).",
    )
    parser.add_argument("--order-seed", type=int, default=0, help="Shuffle test order when non-negative.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the shard trace, which is checkpointed after every image.",
    )
    parser.add_argument(
        "--save-step-images",
        action="store_true",
        help=(
            "Save every executed candidate image and record its path in the trace. "
            "Enable this for train-only on-policy data collection."
        ),
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_summary(
    trace_rows: list[dict[str, Any]],
    *,
    split: str,
    order_seed: int,
    learned_verifier: bool,
    total_split_images: int,
    expected_images: int | None = None,
    shard: dict[str, int] | None = None,
    enabled_tools: list[str] | None = None,
    enabled_tools_file: str | None = None,
    risk_calibration: str | None = None,
    source_ids: list[str] | None = None,
    harm_weight: float | None = None,
    min_planning_utility: float | None = None,
    learned_verifier_threshold: float | None = None,
    learned_verifier_mode: str | None = None,
    learned_min_relative_target_gain: float | None = None,
    block_target_on_progress_guard_reject: bool | None = None,
    belief_thresholds_file: str | None = None,
    belief_base_channels: int | None = None,
    belief_global_backbone: str | None = None,
    belief_presence_threshold: float | None = None,
    belief_severity_threshold: float | None = None,
    diagnostic_probes_enabled: bool | None = None,
    maxim_conda_env: str | None = None,
) -> dict[str, Any]:
    metric_rows = [row["metrics"] for row in trace_rows]
    group_metric_rows: dict[str, list[dict[str, float]]] = {}
    combination_metric_rows: dict[str, list[dict[str, float]]] = {}
    for row in trace_rows:
        metrics = row["metrics"]
        group = str(row.get("group", "unspecified"))
        combination = str(row.get("combination", "unspecified"))
        group_metric_rows.setdefault(group, []).append(metrics)
        combination_metric_rows.setdefault(f"{group}/{combination}", []).append(metrics)
    summary: dict[str, Any] = {
        "num_images": len(trace_rows),
        "expected_images": expected_images if expected_images is not None else len(trace_rows),
        "total_split_images": total_split_images,
        "split": split,
        "order_seed": order_seed,
        "learned_verifier": learned_verifier,
        "enabled_tools": enabled_tools,
        "enabled_tools_file": enabled_tools_file,
        "risk_calibration": risk_calibration,
        "source_ids": source_ids,
        "harm_weight": harm_weight,
        "min_planning_utility": min_planning_utility,
        "learned_verifier_threshold": learned_verifier_threshold,
        "learned_verifier_mode": learned_verifier_mode,
        "learned_min_relative_target_gain": learned_min_relative_target_gain,
        "block_target_on_progress_guard_reject": (
            block_target_on_progress_guard_reject
        ),
        "belief_thresholds_file": belief_thresholds_file,
        "belief_base_channels": belief_base_channels,
        "belief_global_backbone": belief_global_backbone,
        "belief_presence_threshold": belief_presence_threshold,
        "belief_severity_threshold": belief_severity_threshold,
        "diagnostic_probes_enabled": diagnostic_probes_enabled,
        "maxim_conda_env": maxim_conda_env,
        "aggregate": aggregate_metric_dicts(metric_rows),
        "by_group": {
            group: {"num_images": len(values), "aggregate": aggregate_metric_dicts(values)}
            for group, values in sorted(group_metric_rows.items())
        },
        "by_combination": {
            name: {"num_images": len(values), "aggregate": aggregate_metric_dicts(values)}
            for name, values in sorted(combination_metric_rows.items())
        },
        "termination_counts": {
            reason: sum(int(row["termination"] == reason) for row in trace_rows)
            for reason in sorted({row["termination"] for row in trace_rows})
        },
    }
    if shard is not None:
        summary["shard"] = shard
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.belief_base_channels is not None:
        cfg.belief_model.base_channels = int(args.belief_base_channels)
    if args.belief_global_backbone is not None:
        cfg.belief_model["global_backbone"] = args.belief_global_backbone
    if args.belief_clip_checkpoint is not None:
        cfg.belief_model["clip_checkpoint"] = args.belief_clip_checkpoint
    if args.belief_clip_fusion_mode is not None:
        cfg.belief_model["clip_fusion_mode"] = args.belief_clip_fusion_mode
    if args.disable_diagnostic_probes:
        cfg.agent["use_diagnostic_probes"] = False
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    data_root = Path(args.data_root or cfg.data.root)
    if args.enabled_tools is not None and args.enabled_tools_file is not None:
        raise ValueError("Use only one of --enabled-tools or --enabled-tools-file")
    enabled_tools = args.enabled_tools
    if args.enabled_tools_file is not None:
        with Path(args.enabled_tools_file).open("r", encoding="utf-8") as handle:
            enabled_payload = json.load(handle)
        enabled_tools = [str(name) for name in enabled_payload["enabled_tools"]]
    rows = read_jsonl(data_root / f"states_{args.split}.jsonl")
    total_split_images = len(rows)
    if args.source_ids is not None:
        requested_source_ids = {str(source_id) for source_id in args.source_ids}
        rows = [row for row in rows if str(row.get("source_id")) in requested_source_ids]
        found_source_ids = {str(row.get("source_id")) for row in rows}
        missing_source_ids = sorted(requested_source_ids - found_source_ids)
        if missing_source_ids:
            raise KeyError(f"Source IDs not found in {args.split}: {missing_source_ids}")
    if args.state_id is not None:
        rows = [row for row in rows if row["state_id"] == args.state_id]
        if not rows:
            raise KeyError(f"State ID not found in {args.split}: {args.state_id}")
    if args.order_seed >= 0:
        random.Random(args.order_seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.belief_thresholds_file is not None and (
        args.belief_presence_threshold is not None
        or args.belief_severity_threshold is not None
    ):
        raise ValueError(
            "Use either --belief-thresholds-file or scalar belief thresholds, not both"
        )
    presence_threshold_override = args.belief_presence_threshold
    severity_threshold_override = args.belief_severity_threshold
    if args.belief_thresholds_file is not None:
        with Path(args.belief_thresholds_file).open("r", encoding="utf-8") as handle:
            belief_calibration = json.load(handle)
        per_degradation = belief_calibration["calibration"]["per_degradation"]
        presence_threshold_override = {
            str(name): float(payload["selected"]["presence_threshold"])
            for name, payload in per_degradation.items()
        }
        severity_threshold_override = {
            str(name): float(payload["selected"]["severity_threshold"])
            for name, payload in per_degradation.items()
        }
    agent = build_agent(
        cfg,
        belief_checkpoint=args.belief_checkpoint,
        world_checkpoint=args.world_checkpoint,
        verifier_checkpoint=args.verifier_checkpoint,
        gamma_guard_path=args.gamma_guard,
        risk_calibration_path=args.risk_calibration,
        learned_verifier_threshold_override=args.learned_verifier_threshold,
        learned_verifier_mode_override=args.learned_verifier_mode,
        learned_min_relative_target_gain_override=(
            args.learned_min_relative_target_gain
        ),
        block_target_on_progress_guard_reject_override=(
            args.block_target_on_progress_guard_reject
        ),
        presence_threshold_override=presence_threshold_override,
        severity_threshold_override=severity_threshold_override,
        enabled_tools=enabled_tools,
        device=device,
    )
    if args.harm_weight is not None:
        agent.planner.weights["harm"] = float(args.harm_weight)
    if args.min_planning_utility is not None:
        agent.min_planning_utility = float(args.min_planning_utility)

    output_root = ensure_dir(args.output)
    restored_root = ensure_dir(output_root / "restored")
    step_images_root = (
        ensure_dir(output_root / "step_candidates") if args.save_step_images else None
    )
    shard_suffix = "" if args.num_shards == 1 else f".shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    trace_path = output_root / f"traces{shard_suffix}.jsonl"
    summary_path = output_root / f"summary{shard_suffix}.json"
    expected_ids = {row["state_id"] for row in rows}
    trace_rows: list[dict[str, Any]] = []
    if args.resume and trace_path.is_file():
        trace_rows = read_jsonl(trace_path)
        completed_ids = [row["state_id"] for row in trace_rows]
        if len(completed_ids) != len(set(completed_ids)):
            raise RuntimeError(f"Duplicate state IDs in resume trace: {trace_path}")
        unexpected = sorted(set(completed_ids) - expected_ids)
        if unexpected:
            raise RuntimeError(f"Resume trace does not match this shard: {unexpected[:5]}")
    else:
        trace_path.write_text("", encoding="utf-8")
    completed_ids = {row["state_id"] for row in trace_rows}
    pending_rows = [row for row in rows if row["state_id"] not in completed_ids]
    for row in tqdm(pending_rows, desc=f"Agent evaluation shard {args.shard_index}/{args.num_shards}"):
        evaluation_size = None if bool(cfg.data.get("rollout_native_resolution", False)) else int(
            cfg.data.image_size
        )
        image = load_image(resolve(data_root, row["image_path"]), evaluation_size)
        clean = load_image(resolve(data_root, row["clean_path"]), evaluation_size)
        candidate_image_writer = None
        if step_images_root is not None:
            state_step_root = ensure_dir(step_images_root / str(row["state_id"]))

            def candidate_image_writer(
                step_index: int,
                candidate_image: np.ndarray,
                accepted: bool,
                *,
                state_step_root: Path = state_step_root,
            ) -> str:
                decision = "commit" if accepted else "rollback"
                step_path = state_step_root / f"step-{step_index:03d}-{decision}.png"
                save_image(step_path, candidate_image)
                return str(step_path.resolve())

        restored, trace = agent.restore(
            image,
            max_output_shape=clean.shape[:2],
            candidate_image_writer=candidate_image_writer,
        )
        output_path = restored_root / f"{row['state_id']}.png"
        save_image(output_path, restored)
        metric_image = align_to_reference(image, clean)
        metric_restored = align_to_reference(restored, clean)
        before_quality = quality_vector(metric_image, clean)
        after_quality = quality_vector(metric_restored, clean)
        metrics = {
            "before_psnr": psnr(metric_image, clean),
            "after_psnr": psnr(metric_restored, clean),
            "psnr_gain": psnr(metric_restored, clean) - psnr(metric_image, clean),
            "before_ssim": ssim(metric_image, clean),
            "after_ssim": ssim(metric_restored, clean),
            "ssim_gain": ssim(metric_restored, clean) - ssim(metric_image, clean),
            "mean_quality_gain": float(np.mean(after_quality - before_quality)),
            "tool_calls": float(trace["tool_calls"]),
            "accepted_calls": float(trace["accepted_calls"]),
            "rejected_calls": float(trace["rejected_calls"]),
            "transition_prediction_error": float(trace["mean_transition_prediction_error"]),
        }
        group = str(row.get("group", "unspecified"))
        combination = str(row.get("combination", "unspecified"))
        trace_row = {
            "state_id": row["state_id"],
            "group": group,
            "combination": combination,
            "source_id": row.get("source_id"),
            "output_path": str(output_path),
            "metrics": metrics,
            **trace,
        }
        trace_rows.append(trace_row)
        # Checkpoint each completed image. A later external-tool failure can be
        # resumed without discarding hours of successful closed-loop rollouts.
        write_jsonl(trace_path, [trace_row], append=True)

    summary = build_summary(
        trace_rows,
        split=args.split,
        order_seed=args.order_seed,
        learned_verifier=args.verifier_checkpoint is not None,
        total_split_images=total_split_images,
        expected_images=len(rows),
        shard={"index": args.shard_index, "count": args.num_shards},
        enabled_tools=enabled_tools,
        enabled_tools_file=(
            str(Path(args.enabled_tools_file).resolve())
            if args.enabled_tools_file is not None
            else None
        ),
        risk_calibration=(
            str(Path(args.risk_calibration).resolve())
            if args.risk_calibration is not None
            else None
        ),
        source_ids=(
            sorted(str(source_id) for source_id in args.source_ids)
            if args.source_ids is not None
            else None
        ),
        harm_weight=float(agent.planner.weights.get("harm", 0.0)),
        min_planning_utility=float(agent.min_planning_utility),
        learned_verifier_threshold=args.learned_verifier_threshold,
        learned_verifier_mode=(
            str(args.learned_verifier_mode or cfg.agent.learned_verifier_mode)
            if args.verifier_checkpoint is not None
            else None
        ),
        learned_min_relative_target_gain=(
            float(
                args.learned_min_relative_target_gain
                if args.learned_min_relative_target_gain is not None
                else cfg.agent.get("learned_min_relative_target_gain", 0.01)
            )
            if args.verifier_checkpoint is not None
            else None
        ),
        block_target_on_progress_guard_reject=(
            bool(agent.block_target_on_progress_guard_reject)
            if args.verifier_checkpoint is not None
            else None
        ),
        belief_thresholds_file=(
            str(Path(args.belief_thresholds_file).resolve())
            if args.belief_thresholds_file is not None
            else None
        ),
        belief_base_channels=(
            int(cfg.belief_model.base_channels)
            if args.belief_base_channels is not None
            else None
        ),
        belief_global_backbone=(
            str(cfg.belief_model.get("global_backbone", "none"))
            if args.belief_global_backbone is not None
            else None
        ),
        belief_presence_threshold=(
            float(args.belief_presence_threshold)
            if args.belief_presence_threshold is not None
            else None
        ),
        belief_severity_threshold=(
            float(args.belief_severity_threshold)
            if args.belief_severity_threshold is not None
            else None
        ),
        diagnostic_probes_enabled=bool(cfg.agent.use_diagnostic_probes),
        maxim_conda_env=os.environ.get("AGENTICIR_MAXIM_CONDA_ENV"),
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
