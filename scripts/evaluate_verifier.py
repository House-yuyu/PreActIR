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
from preactir.data.datasets import STATUS_NAMES, TransitionDataset
from preactir.models.factory import build_verifier_model
from preactir.utils.checkpoint import load_checkpoint
from preactir.utils.seed import seed_everything
from preactir.utils.train import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the learned before/after transition verifier.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--calibration-beta", type=float, default=0.5)
    parser.add_argument("--exclude-tools", nargs="*", default=[])
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions-output", default=None)
    return parser.parse_args()


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        selected = (probability >= lo) & (probability < hi if index < bins - 1 else probability <= hi)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return float(ece)


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
    balanced_accuracy = 0.5 * (
        tp / max(tp + fn, 1) + tn / max(tn + fp, 1)
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibrate_threshold(probability: np.ndarray, target: np.ndarray, beta: float) -> dict[str, object]:
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        stats = binary_stats(probability >= threshold, target >= 0.5, beta=beta)
        candidates.append({"threshold": float(threshold), **stats})
    candidates.sort(
        key=lambda row: (
            float(row["f_beta"]),
            float(row["balanced_accuracy"]),
            float(row["precision"]),
        ),
        reverse=True,
    )
    return {"beta": float(beta), "selected": candidates[0], "top_candidates": candidates[:10]}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    dataset = TransitionDataset(
        args.data_root or cfg.data.root,
        args.split,
        list(cfg.degradations.names),
        list(cfg.tools.names),
        image_size=int(cfg.data.image_size),
        augment=False,
        exclude_tools=set(args.exclude_tools),
        manifest_root=args.manifest_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_verifier_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    accept_probabilities: list[np.ndarray] = []
    accept_targets: list[np.ndarray] = []
    status_predictions: list[np.ndarray] = []
    status_targets: list[np.ndarray] = []
    target_gain_predictions: list[np.ndarray] = []
    target_gain_targets: list[np.ndarray] = []
    side_effect_predictions: list[np.ndarray] = []
    side_effect_targets: list[np.ndarray] = []
    outside_predictions: list[np.ndarray] = []
    outside_targets: list[np.ndarray] = []
    tool_names: list[str] = []
    state_ids: list[str] = []
    transition_ids: list[str] = []
    paper_quality_predictions: list[np.ndarray] = []
    paper_quality_stds: list[np.ndarray] = []
    paper_quality_targets: list[np.ndarray] = []
    paper_accept_probabilities: list[np.ndarray] = []
    paper_accept_targets: list[np.ndarray] = []
    paper_quality_before: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Verifier evaluation"):
            batch = move_to_device(batch, device)
            output = model(
                before=batch["image"],
                after=batch["next_image"],
                action_mask=batch["action_mask"],
                tool_id=batch["tool_id"],
                target_index=batch["target_index"],
                strength=batch["strength"],
                mask_area=batch["mask_area"],
                cost_prior=batch["cost_prior"],
                trajectory_context=batch.get("trajectory_context"),
            )
            accept_probabilities.append(torch.sigmoid(output["accept_logit"]).cpu().numpy())
            accept_targets.append(batch["accepted"].cpu().numpy())
            status_predictions.append(output["status_logits"].argmax(dim=1).cpu().numpy())
            status_targets.append(batch["status"].cpu().numpy())
            target_gain_predictions.append(output["target_gain"].cpu().numpy())
            target_gain_targets.append(batch["target_gain"].cpu().numpy())
            side_effect_predictions.append(output["side_effect"].cpu().numpy())
            side_effect_targets.append(batch["max_side_effect"].cpu().numpy())
            outside_predictions.append(output["outside_change"].cpu().numpy())
            outside_targets.append(batch["outside_change"].cpu().numpy())
            tool_names.extend(str(value) for value in batch["tool_name"])
            state_ids.extend(str(value) for value in batch["state_id"])
            transition_ids.extend(str(value) for value in batch["transition_id"])
            if "paper_quality_mu" in output and "paper_accept_logit" in output:
                paper_quality_predictions.append(
                    output["paper_quality_mu"].cpu().numpy()
                )
                paper_quality_stds.append(
                    torch.exp(0.5 * output["paper_quality_logvar"]).cpu().numpy()
                )
                paper_quality_targets.append(
                    batch["paper_quality_gain"].cpu().numpy()
                )
                paper_accept_probabilities.append(
                    torch.sigmoid(output["paper_accept_logit"]).cpu().numpy()
                )
                paper_accept_targets.append(batch["paper_accept"].cpu().numpy())
                paper_quality_before.append(
                    batch["paper_quality_before"].cpu().numpy()
                )

    probability = np.concatenate(accept_probabilities)
    target = np.concatenate(accept_targets)
    status_pred = np.concatenate(status_predictions)
    status_true = np.concatenate(status_targets)
    target_gain_pred = np.concatenate(target_gain_predictions)
    target_gain_true = np.concatenate(target_gain_targets)
    side_pred = np.concatenate(side_effect_predictions)
    side_true = np.concatenate(side_effect_targets)
    outside_pred = np.concatenate(outside_predictions)
    outside_true = np.concatenate(outside_targets)

    confusion = np.zeros((len(STATUS_NAMES), len(STATUS_NAMES)), dtype=np.int64)
    for truth, pred in zip(status_true, status_pred):
        confusion[int(truth), int(pred)] += 1
    summary = {
        "num_transitions": int(len(dataset)),
        "accept_accuracy": float(((probability >= args.threshold) == (target >= 0.5)).mean()),
        "accept_brier": float(np.mean((probability - target) ** 2)),
        "accept_ece": expected_calibration_error(probability, target),
        "status_accuracy": float((status_pred == status_true).mean()),
        "target_gain_mae": float(np.mean(np.abs(target_gain_pred - target_gain_true))),
        "side_effect_mae": float(np.mean(np.abs(side_pred - side_true))),
        "outside_change_mae": float(np.mean(np.abs(outside_pred - outside_true))),
        "status_names": STATUS_NAMES,
        "status_confusion_rows_true_columns_pred": confusion.tolist(),
        "accept_rate": float(target.mean()),
        "probability_quantiles": np.quantile(
            probability, [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        ).astype(float).tolist(),
        "accept_classification": binary_stats(
            probability >= args.threshold,
            target >= 0.5,
        ),
        "per_tool": {},
    }
    paper_quality_pred = (
        np.concatenate(paper_quality_predictions)
        if paper_quality_predictions
        else None
    )
    paper_quality_true = (
        np.concatenate(paper_quality_targets) if paper_quality_targets else None
    )
    paper_quality_std = (
        np.concatenate(paper_quality_stds) if paper_quality_stds else None
    )
    paper_probability = (
        np.concatenate(paper_accept_probabilities)
        if paper_accept_probabilities
        else None
    )
    paper_target = (
        np.concatenate(paper_accept_targets) if paper_accept_targets else None
    )
    paper_before = (
        np.concatenate(paper_quality_before) if paper_quality_before else None
    )
    if (
        paper_quality_pred is not None
        and paper_quality_true is not None
        and paper_probability is not None
        and paper_target is not None
    ):
        paper_error = np.abs(paper_quality_pred - paper_quality_true)
        summary["paper_quality"] = {
            "mae": float(paper_error.mean()),
            "psnr_mae_db": float(paper_error[:, 0].mean() * 10.0),
            "ssim_mae": float(paper_error[:, 1].mean()),
            "accept_rate": float(paper_target.mean()),
            "accept_accuracy": float(
                ((paper_probability >= args.threshold) == (paper_target >= 0.5)).mean()
            ),
            "accept_brier": float(
                np.mean((paper_probability - paper_target) ** 2)
            ),
            "accept_ece": expected_calibration_error(
                paper_probability, paper_target
            ),
            "accept_classification": binary_stats(
                paper_probability >= args.threshold,
                paper_target >= 0.5,
                beta=float(args.calibration_beta),
            ),
            "calibration": calibrate_threshold(
                paper_probability,
                paper_target,
                beta=float(args.calibration_beta),
            ),
        }
    tool_array = np.asarray(tool_names, dtype=object)
    for tool_name in sorted(set(tool_names)):
        selected = tool_array == tool_name
        tool_probability = probability[selected]
        tool_target = target[selected]
        summary["per_tool"][tool_name] = {
            "num_transitions": int(selected.sum()),
            "accept_rate": float(tool_target.mean()),
            "probability_quantiles": np.quantile(
                tool_probability, [0.0, 0.25, 0.5, 0.75, 1.0]
            ).astype(float).tolist(),
            "accept_classification": binary_stats(
                tool_probability >= args.threshold,
                tool_target >= 0.5,
            ),
            "calibration": calibrate_threshold(
                tool_probability,
                tool_target,
                beta=float(args.calibration_beta),
            ),
        }
    if args.calibrate_threshold:
        summary["calibration"] = calibrate_threshold(
            probability,
            target,
            beta=float(args.calibration_beta),
        )
    print(json.dumps(summary, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.predictions_output:
        rows = [
            {
                "transition_id": transition_ids[index],
                "state_id": state_ids[index],
                "tool_name": tool_names[index],
                "accept_probability": float(probability[index]),
                "accepted": bool(target[index] >= 0.5),
                "status_prediction": int(status_pred[index]),
                "status_target": int(status_true[index]),
                "target_gain_prediction": float(target_gain_pred[index]),
                "target_gain_target": float(target_gain_true[index]),
                "side_effect_prediction": float(side_pred[index]),
                "side_effect_target": float(side_true[index]),
                "outside_change_prediction": float(outside_pred[index]),
                "outside_change_target": float(outside_true[index]),
                "paper_accept_probability": (
                    float(paper_probability[index])
                    if paper_probability is not None
                    else None
                ),
                "paper_accept_target": (
                    bool(paper_target[index] >= 0.5)
                    if paper_target is not None
                    else None
                ),
                "predicted_paper_psnr_gain_db": (
                    float(paper_quality_pred[index, 0] * 10.0)
                    if paper_quality_pred is not None
                    else None
                ),
                "predicted_paper_ssim_gain": (
                    float(paper_quality_pred[index, 1])
                    if paper_quality_pred is not None
                    else None
                ),
                "predicted_paper_psnr_std_db": (
                    float(paper_quality_std[index, 0] * 10.0)
                    if paper_quality_std is not None
                    else None
                ),
                "predicted_paper_ssim_std": (
                    float(paper_quality_std[index, 1])
                    if paper_quality_std is not None
                    else None
                ),
                "actual_paper_psnr_gain_db": (
                    float(paper_quality_true[index, 0] * 10.0)
                    if paper_quality_true is not None
                    else None
                ),
                "actual_paper_ssim_gain": (
                    float(paper_quality_true[index, 1])
                    if paper_quality_true is not None
                    else None
                ),
                "paper_psnr_before": (
                    float(paper_before[index, 0])
                    if paper_before is not None
                    else None
                ),
                "paper_ssim_before": (
                    float(paper_before[index, 1])
                    if paper_before is not None
                    else None
                ),
            }
            for index in range(len(dataset))
        ]
        path = Path(args.predictions_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
