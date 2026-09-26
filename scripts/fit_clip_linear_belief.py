#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import BeliefDataset
from preactir.models.factory import build_belief_model
from preactir.utils.checkpoint import save_checkpoint
from preactir.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a frozen-CLIP linear degradation belief head."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--severity-alpha", type=float, default=1.0)
    return parser.parse_args()


def binary_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    cfg.belief_model.base_channels = int(args.base_channels)
    cfg.belief_model["global_backbone"] = "clip_vit_b32"
    cfg.belief_model["clip_checkpoint"] = str(Path(args.clip_checkpoint).resolve())
    cfg.belief_model["clip_fusion_mode"] = "linear"
    device = torch.device(args.device)

    dataset = BeliefDataset(
        args.data_root,
        args.split,
        list(cfg.degradations.names),
        image_size=int(cfg.data.image_size),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    model = build_belief_model(cfg).to(device).eval()
    if model.clip_visual is None or model.clip_fusion_mode != "linear":
        raise RuntimeError("Expected a frozen CLIP linear belief model")

    feature_rows: list[np.ndarray] = []
    presence_rows: list[np.ndarray] = []
    severity_rows: list[np.ndarray] = []
    state_ids: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="CLIP belief features"):
            image = batch["image"].to(device, non_blocking=True)
            clip_image = F.interpolate(
                image,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            clip_image = (clip_image - model.clip_mean) / model.clip_std
            feature = model.clip_visual(clip_image).float()
            feature = F.normalize(feature, dim=1)
            feature_rows.append(feature.cpu().numpy())
            presence_rows.append(batch["presence"].numpy())
            severity_rows.append(batch["severity"].numpy())
            state_ids.extend(str(value) for value in batch["state_id"])

    features = np.concatenate(feature_rows, axis=0)
    presence = np.concatenate(presence_rows, axis=0)
    severity = np.concatenate(severity_rows, axis=0)
    if len(state_ids) != features.shape[0]:
        raise RuntimeError("Feature and state counts do not match")

    presence_probability = []
    with torch.no_grad():
        for index, name in enumerate(cfg.degradations.names):
            target = presence[:, index].astype(int)
            if np.unique(target).size != 2:
                raise RuntimeError(f"Training split lacks both labels for {name}")
            classifier = LogisticRegression(
                C=float(args.logistic_c),
                class_weight="balanced",
                max_iter=2000,
                random_state=int(cfg.seed),
            ).fit(features, target)
            model.presence_head.weight[index].copy_(
                torch.from_numpy(classifier.coef_[0]).to(
                    model.presence_head.weight, non_blocking=True
                )
            )
            model.presence_head.bias[index].copy_(
                torch.as_tensor(classifier.intercept_[0]).to(model.presence_head.bias)
            )
            presence_probability.append(classifier.predict_proba(features)[:, 1])

            active = target.astype(bool)
            active_severity = np.clip(severity[active, index], 1e-4, 1.0 - 1e-4)
            severity_logit = np.log(active_severity / (1.0 - active_severity))
            regressor = Ridge(alpha=float(args.severity_alpha)).fit(
                features[active], severity_logit
            )
            model.severity_mu_head.weight[index].copy_(
                torch.from_numpy(regressor.coef_).to(
                    model.severity_mu_head.weight, non_blocking=True
                )
            )
            model.severity_mu_head.bias[index].copy_(
                torch.as_tensor(regressor.intercept_).to(model.severity_mu_head.bias)
            )
            residual = severity_logit - regressor.predict(features[active])
            logvar = float(np.log(max(float(np.var(residual)), 1e-4)))
            model.severity_logvar_head.weight[index].zero_()
            model.severity_logvar_head.bias[index].fill_(logvar)

    probability = np.stack(presence_probability, axis=1)
    stats = binary_stats(probability >= 0.5, presence >= 0.5)
    manifest_hash = hashlib.sha256("\n".join(state_ids).encode("utf-8")).hexdigest()
    extra = {
        "degradation_names": list(cfg.degradations.names),
        "selection_mode": "frozen_clip_linear_fit",
        "global_backbone": "clip_vit_b32",
        "clip_checkpoint": str(Path(args.clip_checkpoint).resolve()),
        "clip_fusion_mode": "linear",
        "base_channels": int(args.base_channels),
        "logistic_c": float(args.logistic_c),
        "severity_alpha": float(args.severity_alpha),
        "num_states": int(len(dataset)),
        "state_id_sha256": manifest_hash,
    }
    output = Path(args.output)
    save_checkpoint(output, model, epoch=0, metrics=stats, extra=extra)
    summary = {
        **extra,
        "checkpoint": str(output.resolve()),
        "checkpoint_bytes": output.stat().st_size,
        "train_presence": stats,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
