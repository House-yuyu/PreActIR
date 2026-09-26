#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from preactir.utils.io import read_jsonl
from preactir.utils.risk_metrics import select_risk_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze finite-sample harmful-action admission thresholds from validation predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-selected", type=int, default=10)
    parser.add_argument("--min-tool-calibration", type=int, default=20)
    parser.add_argument("--fallback", choices=["global", "reject"], default="global")
    parser.add_argument("--calibration-split", default="cal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = Path(args.predictions)
    rows = read_jsonl(prediction_path)
    if not rows:
        raise RuntimeError("No prediction rows found")
    if {str(row.get("split")) for row in rows} != {str(args.calibration_split)}:
        raise ValueError(
            f"Risk calibration expected split={args.calibration_split!r} predictions only"
        )
    probability = np.asarray([row["harm_probability"] for row in rows], dtype=np.float64)
    harmful = np.asarray([row["harmful"] for row in rows], dtype=bool)
    global_rule = select_risk_threshold(
        probability,
        harmful,
        alpha=float(args.alpha),
        delta=float(args.delta),
        min_selected=int(args.min_selected),
    )
    by_tool: dict[str, object] = {}
    tool_names = np.asarray([str(row["tool_name"]) for row in rows], dtype=object)
    for tool_name in sorted(set(tool_names.tolist())):
        selected = tool_names == tool_name
        if int(selected.sum()) < int(args.min_tool_calibration):
            by_tool[tool_name] = {
                "enabled": False,
                "threshold": 0.0,
                "num_calibration": int(selected.sum()),
                "reason": "insufficient_tool_calibration",
            }
            continue
        by_tool[tool_name] = select_risk_threshold(
            probability[selected],
            harmful[selected],
            alpha=float(args.alpha),
            delta=float(args.delta),
            min_selected=int(args.min_selected),
        )
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "method": "clopper_pearson_selective_risk_control",
        "calibration_split": str(args.calibration_split),
        "prediction_file": str(prediction_path.resolve()),
        "prediction_sha256": digest,
        "alpha": float(args.alpha),
        "delta": float(args.delta),
        "min_selected": int(args.min_selected),
        "min_tool_calibration": int(args.min_tool_calibration),
        "fallback": args.fallback,
        "global": global_rule,
        "by_tool": by_tool,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
