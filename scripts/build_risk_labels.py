#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from preactir.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create explicit beneficial/neutral/harmful action labels without touching test."
    )
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--harm-psnr-gain-db", type=float, default=-0.05)
    parser.add_argument("--harm-ssim-gain", type=float, default=-0.005)
    parser.add_argument("--harm-mean-quality-gain", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.manifest_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, object] = {}
    for split in args.splits:
        source_path = source_root / f"transitions_{split}.jsonl"
        rows = read_jsonl(source_path)
        output_rows = []
        by_tool: dict[str, dict[str, int]] = {}
        for row in rows:
            delta_quality = np.asarray(row["delta_quality"], dtype=np.float32)
            psnr_gain_db = float(delta_quality[0] * 50.0)
            ssim_gain = float(delta_quality[1])
            mean_quality_gain = float(delta_quality.mean())
            criteria = {
                "psnr": bool(psnr_gain_db < args.harm_psnr_gain_db),
                "ssim": bool(ssim_gain < args.harm_ssim_gain),
                "mean_quality": bool(mean_quality_gain < args.harm_mean_quality_gain),
            }
            harmful = bool(any(criteria.values()))
            beneficial = bool(row["accepted"])
            category = "harmful" if harmful else ("beneficial" if beneficial else "neutral")
            updated = dict(row)
            updated["beneficial"] = beneficial
            updated["harmful"] = harmful
            updated["risk_label_v5"] = {
                "category": category,
                "harmful": harmful,
                "beneficial": beneficial,
                "psnr_gain_db": psnr_gain_db,
                "ssim_gain": ssim_gain,
                "mean_quality_gain": mean_quality_gain,
                "triggered_harm_criteria": criteria,
            }
            output_rows.append(updated)
            name = str(row["action"]["tool_name"])
            stats = by_tool.setdefault(
                name, {"total": 0, "beneficial": 0, "neutral": 0, "harmful": 0}
            )
            stats["total"] += 1
            stats[category] += 1
        write_jsonl(output_root / source_path.name, output_rows)
        split_summaries[split] = {
            "num_transitions": len(output_rows),
            "beneficial": sum(row["risk_label_v5"]["category"] == "beneficial" for row in output_rows),
            "neutral": sum(row["risk_label_v5"]["category"] == "neutral" for row in output_rows),
            "harmful": sum(row["risk_label_v5"]["category"] == "harmful" for row in output_rows),
            "by_tool": dict(sorted(by_tool.items())),
        }
    summary = {
        "schema_version": 1,
        "source_manifest_root": str(source_root.resolve()),
        "splits_written": list(args.splits),
        "test_touched": "test" in args.splits,
        "harm_rule": {
            "any_of": {
                "psnr_gain_db_below": float(args.harm_psnr_gain_db),
                "ssim_gain_below": float(args.harm_ssim_gain),
                "mean_quality_gain_below": float(args.harm_mean_quality_gain),
            }
        },
        "splits": split_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
