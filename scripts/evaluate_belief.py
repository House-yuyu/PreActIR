#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import BeliefDataset
from preactir.models.factory import build_belief_model
from preactir.utils.checkpoint import load_checkpoint
from preactir.utils.io import write_jsonl
from preactir.utils.seed import seed_everything
from preactir.utils.train import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the spatial belief encoder.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Belief encoder width used to create the checkpoint.",
    )
    parser.add_argument(
        "--global-backbone",
        choices=["none", "clip_vit_b32"],
        default=None,
    )
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument(
        "--clip-fusion-mode", choices=["concat", "replace", "linear"], default=None
    )
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--severity-threshold", type=float, default=0.08)
    parser.add_argument(
        "--thresholds-file",
        default=None,
        help=(
            "Frozen evaluate_belief.py calibration JSON. When supplied, apply its "
            "per-degradation presence/severity gates instead of the scalar gates."
        ),
    )
    parser.add_argument(
        "--calibrate-agent-thresholds",
        action="store_true",
        help="Select global presence/severity gates on this split. Use validation only.",
    )
    parser.add_argument("--calibration-beta", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Optional JSONL output with one prediction row per evaluated state.",
    )
    return parser.parse_args()


def load_per_degradation_thresholds(
    path: str | None,
    degradation_names: list[str],
    default_presence: float,
    default_severity: float,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.full(len(degradation_names), float(default_presence), dtype=np.float32)
    severity = np.full(len(degradation_names), float(default_severity), dtype=np.float32)
    if path is None:
        return presence, severity
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_degradation = payload.get("calibration", {}).get("per_degradation")
    if not isinstance(per_degradation, dict):
        raise ValueError(f"{path} does not contain calibration.per_degradation")
    missing = [name for name in degradation_names if name not in per_degradation]
    if missing:
        raise ValueError(f"{path} lacks calibrated degradations: {missing}")
    for index, name in enumerate(degradation_names):
        selected = per_degradation[name].get("selected", per_degradation[name])
        presence[index] = float(selected["presence_threshold"])
        severity[index] = float(selected["severity_threshold"])
    return presence, severity


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probability >= lower) & (probability < upper if index < bins - 1 else probability <= upper)
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return float(result)


def binary_stats(prediction: np.ndarray, target: np.ndarray, beta: float = 1.0) -> dict[str, float | int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta_sq = float(beta) ** 2
    f_beta = (1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_agent_thresholds(
    probability: np.ndarray,
    severity: np.ndarray,
    target: np.ndarray,
    beta: float,
    degradation_names: list[str],
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for presence_threshold in np.linspace(0.05, 0.75, 29):
        for severity_threshold in np.linspace(0.0, 0.20, 21):
            prediction = (probability >= presence_threshold) & (severity >= severity_threshold)
            per_class = [
                binary_stats(prediction[:, index], target[:, index], beta=beta)
                for index in range(target.shape[1])
                if bool(target[:, index].any())
            ]
            macro_f_beta = float(np.mean([row["f_beta"] for row in per_class]))
            micro = binary_stats(prediction, target, beta=beta)
            state_coverage = float(prediction.any(axis=1).mean())
            candidates.append(
                {
                    "presence_threshold": float(presence_threshold),
                    "severity_threshold": float(severity_threshold),
                    "macro_f_beta": macro_f_beta,
                    "micro_f_beta": micro["f_beta"],
                    "precision": micro["precision"],
                    "recall": micro["recall"],
                    "state_coverage": state_coverage,
                }
            )
    candidates.sort(
        key=lambda row: (
            float(row["macro_f_beta"]),
            float(row["micro_f_beta"]),
            float(row["precision"]),
        ),
        reverse=True,
    )
    per_degradation = {}
    for index, name in enumerate(degradation_names):
        class_candidates = []
        for presence_threshold in np.linspace(0.05, 0.95, 181):
            for severity_threshold in np.linspace(0.0, 0.20, 21):
                prediction = (probability[:, index] >= presence_threshold) & (
                    severity[:, index] >= severity_threshold
                )
                stats = binary_stats(prediction, target[:, index], beta=beta)
                class_candidates.append(
                    {
                        "presence_threshold": float(presence_threshold),
                        "severity_threshold": float(severity_threshold),
                        **stats,
                    }
                )
        class_candidates.sort(
            key=lambda row: (
                float(row["f_beta"]),
                float(row["precision"]),
                float(row["recall"]),
            ),
            reverse=True,
        )
        per_degradation[str(name)] = {
            "selected": class_candidates[0],
            "top_candidates": class_candidates[:5],
        }
    return {
        "beta": float(beta),
        "selected": candidates[0],
        "top_candidates": candidates[:10],
        "per_degradation": per_degradation,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.base_channels is not None:
        cfg.belief_model.base_channels = int(args.base_channels)
    if args.global_backbone is not None:
        cfg.belief_model["global_backbone"] = args.global_backbone
    if args.clip_checkpoint is not None:
        cfg.belief_model["clip_checkpoint"] = args.clip_checkpoint
    if args.clip_fusion_mode is not None:
        cfg.belief_model["clip_fusion_mode"] = args.clip_fusion_mode
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    dataset = BeliefDataset(
        args.data_root or cfg.data.root,
        args.split,
        list(cfg.degradations.names),
        image_size=int(cfg.data.image_size),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_belief_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    all_prob: list[np.ndarray] = []
    all_presence: list[np.ndarray] = []
    all_severity_mu: list[np.ndarray] = []
    all_severity_true: list[np.ndarray] = []
    all_severity_std: list[np.ndarray] = []
    all_state_ids: list[str] = []
    mask_intersection = np.zeros(len(cfg.degradations.names), dtype=np.float64)
    mask_union = np.zeros(len(cfg.degradations.names), dtype=np.float64)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Belief evaluation"):
            batch = move_to_device(batch, device)
            output = model.predict_belief(batch["image"])
            probability = output["presence_prob"].cpu().numpy()
            severity_mu = output["severity_mu"].cpu().numpy()
            severity_std = output["severity_std"].cpu().numpy()
            presence_true = batch["presence"].cpu().numpy()
            severity_true = batch["severity"].cpu().numpy()
            mask_pred = (output["mask_prob"] >= 0.5).cpu().numpy()
            mask_true = (batch["masks"] >= 0.5).cpu().numpy()
            active = presence_true >= 0.5
            for degradation_index in range(active.shape[1]):
                selected = active[:, degradation_index]
                if not selected.any():
                    continue
                pred = mask_pred[selected, degradation_index]
                true = mask_true[selected, degradation_index]
                mask_intersection[degradation_index] += np.logical_and(pred, true).sum()
                mask_union[degradation_index] += np.logical_or(pred, true).sum()
            all_prob.append(probability)
            all_presence.append(presence_true)
            all_severity_mu.append(severity_mu)
            all_severity_true.append(severity_true)
            all_severity_std.append(severity_std)
            all_state_ids.extend(str(value) for value in batch["state_id"])

    probability = np.concatenate(all_prob, axis=0)
    presence_true = np.concatenate(all_presence, axis=0)
    severity_mu = np.concatenate(all_severity_mu, axis=0)
    severity_true = np.concatenate(all_severity_true, axis=0)
    severity_std = np.concatenate(all_severity_std, axis=0)
    prediction = probability >= args.presence_threshold
    target = presence_true >= 0.5
    presence_stats = binary_stats(prediction, target)
    gate_presence, gate_severity = load_per_degradation_thresholds(
        args.thresholds_file,
        list(cfg.degradations.names),
        args.presence_threshold,
        args.severity_threshold,
    )
    agent_prediction = (probability >= gate_presence[None, :]) & (
        severity_mu >= gate_severity[None, :]
    )
    agent_stats = binary_stats(agent_prediction, target)
    active = target
    severity_error = np.abs(severity_mu - severity_true)
    normalized_error = severity_error / np.maximum(severity_std, 1e-4)
    per_degradation_iou = {
        str(name): float(mask_intersection[index] / max(mask_union[index], 1.0))
        for index, name in enumerate(cfg.degradations.names)
    }
    per_degradation = {}
    for index, name in enumerate(cfg.degradations.names):
        per_degradation[str(name)] = {
            "presence": binary_stats(prediction[:, index], target[:, index]),
            "agent_gate": binary_stats(agent_prediction[:, index], target[:, index]),
            "probability_quantiles": np.quantile(
                probability[:, index], [0.0, 0.1, 0.5, 0.9, 1.0]
            ).astype(float).tolist(),
            "active_severity_quantiles": (
                np.quantile(severity_mu[target[:, index], index], [0.0, 0.1, 0.5, 0.9, 1.0])
                .astype(float)
                .tolist()
                if bool(target[:, index].any())
                else []
            ),
        }
    summary = {
        "num_states": int(len(dataset)),
        "presence_accuracy": float((prediction == target).mean()),
        "presence_precision": presence_stats["precision"],
        "presence_recall": presence_stats["recall"],
        "presence_f1": presence_stats["f_beta"],
        "presence_brier": float(np.mean((probability - presence_true) ** 2)),
        "presence_ece": expected_calibration_error(probability.reshape(-1), presence_true.reshape(-1)),
        "severity_mae_active": float(severity_error[active].mean()) if active.any() else 0.0,
        "severity_nll_proxy_active": float((0.5 * normalized_error[active] ** 2 + np.log(np.maximum(severity_std[active], 1e-4))).mean()) if active.any() else 0.0,
        "mean_mask_iou_active": float(np.mean(list(per_degradation_iou.values()))),
        "mask_iou_by_degradation": per_degradation_iou,
        "confusion": {
            key: presence_stats[key] for key in ("tp", "fp", "fn", "tn")
        },
        "agent_gate": {
            "presence_thresholds": gate_presence.astype(float).tolist(),
            "severity_thresholds": gate_severity.astype(float).tolist(),
            "thresholds_file": str(Path(args.thresholds_file).resolve())
            if args.thresholds_file
            else None,
            "state_coverage": float(agent_prediction.any(axis=1).mean()),
            **agent_stats,
        },
        "per_degradation": per_degradation,
    }
    if args.calibrate_agent_thresholds:
        summary["calibration"] = calibrate_agent_thresholds(
            probability,
            severity_mu,
            target,
            beta=float(args.calibration_beta),
            degradation_names=list(cfg.degradations.names),
        )
    print(json.dumps(summary, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.predictions_output:
        if len(all_state_ids) != probability.shape[0]:
            raise RuntimeError("State ID and prediction counts do not match")
        rows = []
        for index, state_id in enumerate(all_state_ids):
            rows.append(
                {
                    "state_id": state_id,
                    "presence_probability": probability[index].astype(float).tolist(),
                    "severity_mu": severity_mu[index].astype(float).tolist(),
                    "severity_std": severity_std[index].astype(float).tolist(),
                    "presence_true": presence_true[index].astype(float).tolist(),
                    "severity_true": severity_true[index].astype(float).tolist(),
                    "agent_prediction": agent_prediction[index].astype(int).tolist(),
                }
            )
        write_jsonl(args.predictions_output, rows)


if __name__ == "__main__":
    main()
