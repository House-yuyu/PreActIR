#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from preactir.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a verifier commit threshold from identity-level OOF predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", type=float, default=0.5)
    return parser.parse_args()


def stats(prediction: np.ndarray, target: np.ndarray, beta: float) -> dict[str, float | int]:
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta_sq = float(beta) ** 2
    f_beta = (1.0 + beta_sq) * precision * recall / max(
        beta_sq * precision + recall, 1e-12
    )
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


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    rows = read_jsonl(prediction_path)
    if not rows:
        raise RuntimeError("No verifier predictions found")
    transition_ids = [str(row["transition_id"]) for row in rows]
    if len(transition_ids) != len(set(transition_ids)):
        raise RuntimeError("Verifier calibration predictions contain duplicate transition IDs")
    probability = np.asarray(
        [float(row["accept_probability"]) for row in rows], dtype=np.float64
    )
    target = np.asarray([bool(row["accepted"]) for row in rows], dtype=bool)
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 197):
        candidates.append(
            {
                "threshold": float(threshold),
                **stats(probability >= threshold, target, float(args.beta)),
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["f_beta"]),
            float(row["balanced_accuracy"]),
            float(row["precision"]),
            float(row["threshold"]),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "protocol": "identity-level out-of-fold verifier threshold calibration",
        "predictions": str(prediction_path.resolve()),
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "num_predictions": len(rows),
        "num_source_ids": len({str(row["source_id"]) for row in rows}),
        "beta": float(args.beta),
        "selected": candidates[0],
        "top_candidates": candidates[:10],
        "test_read": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
