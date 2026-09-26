#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import TransitionDataset
from preactir.models.factory import build_world_model
from preactir.utils.checkpoint import load_checkpoint
from preactir.utils.seed import seed_everything
from preactir.utils.train import move_to_device
from preactir.utils.risk_metrics import binary_roc_auc, risk_coverage_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate transition prediction and calibration.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--include-tools", nargs="*", default=None)
    return parser.parse_args()


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if not selected.any():
            continue
        confidence = float(probability[selected].mean())
        accuracy = float(target[selected].mean())
        ece += float(selected.mean()) * abs(confidence - accuracy)
    return ece


def kendall_tau(x: list[float], y: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return float((concordant - discordant) / total) if total else 0.0


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
        manifest_root=args.manifest_root,
        include_tools=set(args.include_tools) if args.include_tools is not None else None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(cfg.data.num_workers))
    model = build_world_model(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)

    all_pred_deg: list[np.ndarray] = []
    all_true_deg: list[np.ndarray] = []
    all_pred_quality: list[np.ndarray] = []
    all_true_quality: list[np.ndarray] = []
    all_pred_damage: list[np.ndarray] = []
    all_true_damage: list[np.ndarray] = []
    all_accept_prob: list[np.ndarray] = []
    all_accept_true: list[np.ndarray] = []
    all_harm_prob: list[np.ndarray] = []
    all_harm_true: list[np.ndarray] = []
    all_severe_harm_prob: list[np.ndarray] = []
    all_severe_harm_true: list[np.ndarray] = []
    all_pred_paper_quality: list[np.ndarray] = []
    all_true_paper_quality: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []
    ranking_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pred": [], "true": []})

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = move_to_device(batch, device)
            output = model(
                image=batch["image"],
                belief=batch["belief"],
                action_mask=batch["action_mask"],
                tool_id=batch["tool_id"],
                target_index=batch["target_index"],
                strength=batch["strength"],
                mask_area=batch["mask_area"],
                cost_prior=batch["cost_prior"],
                trajectory_context=batch.get("trajectory_context"),
            )
            pred_deg = output["delta_degradation_mu"].cpu().numpy()
            true_deg = batch["delta_degradation"].cpu().numpy()
            pred_quality = output["delta_quality_mu"].cpu().numpy()
            true_quality = batch["delta_quality"].cpu().numpy()
            pred_damage = output["damage_mu"].cpu().numpy()
            true_damage = batch["damage"].cpu().numpy()
            accept_prob = torch.sigmoid(output["accept_logit"]).cpu().numpy()
            accept_true = batch["accepted"].cpu().numpy()
            harm_probability = (
                torch.sigmoid(output["harm_logit"]).cpu().numpy()
                if "harm_logit" in output
                else 1.0 - accept_prob
            )
            harm_true = batch["harmful"].cpu().numpy()
            severe_harm_probability = (
                torch.sigmoid(output["severe_harm_logit"]).cpu().numpy()
                if "severe_harm_logit" in output
                else harm_probability
            )
            severe_harm_true = batch["severe_harm"].cpu().numpy()
            pred_paper_quality = (
                output["paper_quality_mu"].cpu().numpy()
                if "paper_quality_mu" in output
                else None
            )
            true_paper_quality = batch["paper_quality_gain"].cpu().numpy()
            target_indices = batch["target_index"].cpu().numpy()

            all_pred_deg.append(pred_deg)
            all_true_deg.append(true_deg)
            all_pred_quality.append(pred_quality)
            all_true_quality.append(true_quality)
            all_pred_damage.append(pred_damage)
            all_true_damage.append(true_damage)
            all_accept_prob.append(accept_prob)
            all_accept_true.append(accept_true)
            all_harm_prob.append(harm_probability)
            all_harm_true.append(harm_true)
            all_severe_harm_prob.append(severe_harm_probability)
            all_severe_harm_true.append(severe_harm_true)
            if pred_paper_quality is not None:
                all_pred_paper_quality.append(pred_paper_quality)
                all_true_paper_quality.append(true_paper_quality)

            for index, state_id in enumerate(batch["state_id"]):
                target = int(target_indices[index])
                legacy_pred_utility = float(pred_deg[index, target] + 0.5 * pred_quality[index].mean() - 2.0 * pred_damage[index])
                legacy_true_utility = float(true_deg[index, target] + 0.5 * true_quality[index].mean() - 2.0 * true_damage[index])
                pred_utility = (
                    float(pred_paper_quality[index].sum())
                    if pred_paper_quality is not None
                    else legacy_pred_utility
                )
                true_utility = float(true_paper_quality[index].sum())
                ranking_group_id = str(batch["ranking_group_id"][index])
                ranking_groups[ranking_group_id]["pred"].append(pred_utility)
                ranking_groups[ranking_group_id]["true"].append(true_utility)
                prediction_rows.append(
                    {
                        "split": args.split,
                        "transition_id": str(batch["transition_id"][index]),
                        "state_id": str(state_id),
                        "tool_name": str(batch["tool_name"][index]),
                        "target_index": target,
                        "accept_probability": float(accept_prob[index]),
                        "accepted": bool(accept_true[index] >= 0.5),
                        "harm_probability": float(harm_probability[index]),
                        "harmful": bool(harm_true[index] >= 0.5),
                        "severe_harm_probability": float(
                            severe_harm_probability[index]
                        ),
                        "severe_harm": bool(severe_harm_true[index] >= 0.5),
                        "target_name": str(
                            cfg.degradations.names[target]
                        ),
                        "predicted_utility": pred_utility,
                        "actual_utility": true_utility,
                        "legacy_predicted_utility": legacy_pred_utility,
                        "legacy_actual_utility": legacy_true_utility,
                        "actual_reference_utility": float(
                            batch["reference_utility"][index].cpu()
                        ),
                        "predicted_target_gain": float(pred_deg[index, target]),
                        "actual_target_gain": float(true_deg[index, target]),
                        "predicted_quality_gain": float(pred_quality[index].mean()),
                        "actual_quality_gain": float(true_quality[index].mean()),
                        "predicted_damage": float(pred_damage[index]),
                        "actual_damage": float(true_damage[index]),
                        "predicted_paper_psnr_gain_db": (
                            float(pred_paper_quality[index, 0] * 10.0)
                            if pred_paper_quality is not None
                            else None
                        ),
                        "predicted_paper_ssim_gain": (
                            float(pred_paper_quality[index, 1])
                            if pred_paper_quality is not None
                            else None
                        ),
                        "actual_paper_psnr_gain_db": float(
                            true_paper_quality[index, 0] * 10.0
                        ),
                        "actual_paper_ssim_gain": float(true_paper_quality[index, 1]),
                        "paper_psnr_before": float(
                            batch["paper_quality_before"][index, 0].cpu()
                        ),
                        "paper_ssim_before": float(
                            batch["paper_quality_before"][index, 1].cpu()
                        ),
                        "ranking_group_id": ranking_group_id,
                    }
                )

    pred_deg = np.concatenate(all_pred_deg)
    true_deg = np.concatenate(all_true_deg)
    pred_quality = np.concatenate(all_pred_quality)
    true_quality = np.concatenate(all_true_quality)
    pred_damage = np.concatenate(all_pred_damage)
    true_damage = np.concatenate(all_true_damage)
    accept_prob = np.concatenate(all_accept_prob)
    accept_true = np.concatenate(all_accept_true)
    harm_probability = np.concatenate(all_harm_prob)
    harm_true = np.concatenate(all_harm_true)
    severe_harm_probability = np.concatenate(all_severe_harm_prob)
    severe_harm_true = np.concatenate(all_severe_harm_true)
    pred_paper_quality = (
        np.concatenate(all_pred_paper_quality) if all_pred_paper_quality else None
    )
    true_paper_quality = (
        np.concatenate(all_true_paper_quality) if all_true_paper_quality else None
    )
    taus = [kendall_tau(group["pred"], group["true"]) for group in ranking_groups.values() if len(group["pred"]) > 1]
    summary = {
        "degradation_mae": float(np.mean(np.abs(pred_deg - true_deg))),
        "quality_mae": float(np.mean(np.abs(pred_quality - true_quality))),
        "damage_mae": float(np.mean(np.abs(pred_damage - true_damage))),
        "accept_accuracy": float(np.mean((accept_prob >= 0.5) == (accept_true >= 0.5))),
        "accept_brier": float(np.mean((accept_prob - accept_true) ** 2)),
        "accept_ece": expected_calibration_error(accept_prob, accept_true),
        "harm_accuracy": float(np.mean((harm_probability >= 0.5) == (harm_true >= 0.5))),
        "harm_brier": float(np.mean((harm_probability - harm_true) ** 2)),
        "harm_ece": expected_calibration_error(harm_probability, harm_true),
        "harm_roc_auc": binary_roc_auc(harm_probability, harm_true),
        "harm_rate": float(harm_true.mean()),
        "risk_coverage_curve": risk_coverage_curve(harm_probability, harm_true),
        "severe_harm_accuracy": float(
            np.mean(
                (severe_harm_probability >= 0.5)
                == (severe_harm_true >= 0.5)
            )
        ),
        "severe_harm_brier": float(
            np.mean((severe_harm_probability - severe_harm_true) ** 2)
        ),
        "severe_harm_ece": expected_calibration_error(
            severe_harm_probability, severe_harm_true
        ),
        "severe_harm_roc_auc": binary_roc_auc(
            severe_harm_probability, severe_harm_true
        ),
        "severe_harm_rate": float(severe_harm_true.mean()),
        "severe_risk_coverage_curve": risk_coverage_curve(
            severe_harm_probability, severe_harm_true
        ),
        "mean_state_kendall_tau": float(np.mean(taus)) if taus else 0.0,
        "num_transitions": int(len(accept_true)),
        "num_states_ranked": int(len(taus)),
    }
    if pred_paper_quality is not None and true_paper_quality is not None:
        paper_error = np.abs(pred_paper_quality - true_paper_quality)
        summary.update(
            {
                "paper_quality_mae": float(paper_error.mean()),
                "paper_psnr_mae_db": float(paper_error[:, 0].mean() * 10.0),
                "paper_ssim_mae": float(paper_error[:, 1].mean()),
            }
        )
    print(json.dumps(summary, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.predictions_output:
        path = Path(args.predictions_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
