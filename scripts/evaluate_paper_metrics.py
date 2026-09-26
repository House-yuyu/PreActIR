#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from preactir.utils.image import align_to_reference, image_to_tensor, load_image
from preactir.utils.io import read_jsonl, write_jsonl
from preactir.utils.metrics import aggregate_metric_dicts
from preactir.utils.paper_report import write_paper_metrics_markdown
from preactir.utils.benchmark_protocol import (
    audit_mio100_protocol,
    audit_mio100_training_subset,
    load_benchmark_protocol,
    sha256_file,
)
from preactir.tools.paper_registry import inspect_paper_metric_assets


EXPECTED_METRIC_NAMES = ("psnr", "ssim", "lpips", "maniqa", "clipiqa", "musiq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the six MiO100 paper metrics with pyiqa and Group A/B/C summaries."
    )
    parser.add_argument("--data-root", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--restored-root")
    source.add_argument(
        "--use-degraded-input",
        action="store_true",
        help="Evaluate each manifest image_path as the no-restoration baseline.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Explicit state manifest. With a frozen protocol this is audited as a "
            "training-only diagnostic subset and can never be marked main-table eligible."
        ),
    )
    parser.add_argument(
        "--output", default="outputs/mio100/paper_metrics_agenticir_official"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--protocol-config",
        help=(
            "Frozen benchmark protocol JSON. MiO100 paper runs should use "
            "configs/agenticir_mio100_protocol.json."
        ),
    )
    parser.add_argument(
        "--size-mismatch-policy",
        choices=("auto", "error", "bicubic", "agenticir_matlab_x4"),
        default="auto",
        help=(
            "Geometry policy. 'auto' uses the protocol setting when supplied; without a protocol it "
            "retains the legacy bicubic alignment behavior. AgenticIR uses "
            "BasicSR's MATLAB-style imresize only for exact x4 mismatches."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help=(
            "Evaluate only these groups after auditing the complete frozen split; "
            "repeat for multiple groups."
        ),
    )
    parser.add_argument(
        "--state-id",
        action="append",
        default=None,
        help="Evaluate one exact state ID; repeat the option for multiple states.",
    )
    parser.add_argument(
        "--report-title",
        default="Image Restoration Paper Metrics",
        help="Title written to summary.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("pyiqa is required; run this script in the 4kagent environment") from exc

    device = torch.device(args.device)
    asset_report = inspect_paper_metric_assets()
    if asset_report["missing"]:
        missing_paths = [asset_report["assets"][name]["path"] for name in asset_report["missing"]]
        raise RuntimeError(
            "Paper IQA weights are not fully cached:\n- "
            + "\n- ".join(missing_paths)
            + "\nDownload/copy them before running the 1,440-image evaluation."
        )
    data_root = Path(args.data_root)
    restored_root = Path(args.restored_root) if args.restored_root is not None else None
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest is not None
        else data_root / f"states_{args.split}.jsonl"
    )
    rows = read_jsonl(manifest_path)
    protocol = load_benchmark_protocol(args.protocol_config) if args.protocol_config else None
    protocol_audit = None
    runtime_audit = None
    if protocol is not None:
        _validate_metric_protocol(protocol)
        protocol_audit = (
            audit_mio100_training_subset(data_root, manifest_path, protocol)
            if args.manifest is not None
            else audit_mio100_protocol(data_root, args.split, protocol)
        )
        if not protocol_audit["passed"]:
            raise RuntimeError(
                f"{protocol['protocol']} MiO100 data audit failed:\n- "
                + "\n- ".join(protocol_audit["mismatches"])
            )
        runtime_audit = _audit_metric_runtime(protocol, asset_report, pyiqa)
        if not runtime_audit["passed"]:
            raise RuntimeError(
                f"{protocol['protocol']} metric runtime audit failed:\n- "
                + "\n- ".join(runtime_audit["mismatches"])
            )
    if args.group is not None:
        requested_groups = set(map(str, args.group))
        available_groups = {str(row.get("group", "unspecified")) for row in rows}
        missing_groups = sorted(requested_groups - available_groups)
        if missing_groups:
            raise KeyError(f"Groups not found in split={args.split}: {missing_groups}")
        rows = [row for row in rows if str(row.get("group")) in requested_groups]
    if args.state_id is not None:
        requested_ids = set(args.state_id)
        rows = [row for row in rows if str(row["state_id"]) in requested_ids]
        found_ids = {str(row["state_id"]) for row in rows}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise KeyError(f"State IDs not found in split={args.split}: {missing_ids}")
    if args.limit is not None:
        rows = rows[: args.limit]

    source_kind = "degraded_input" if args.use_degraded_input else "restored_output"
    expected_geometry_policy = (
        str(protocol["geometry"][source_kind]) if protocol is not None else "bicubic"
    )
    geometry_policy = (
        expected_geometry_policy if args.size_mismatch_policy == "auto" else args.size_mismatch_policy
    )
    if protocol is not None and geometry_policy != expected_geometry_policy:
        raise ValueError(
            f"Protocol {protocol['protocol']} requires geometry policy "
            f"{expected_geometry_policy!r} for {source_kind}, found {geometry_policy!r}"
        )
    crop_border = int(protocol["geometry"]["crop_border"]) if protocol is not None else 0
    geometry_audit = _audit_geometry(
        rows,
        data_root=data_root,
        restored_root=restored_root,
        use_degraded_input=args.use_degraded_input,
        policy=geometry_policy,
    )
    if not geometry_audit["passed"]:
        examples = geometry_audit["mismatch_examples"]
        rendered_examples = ", ".join(
            f"{item['state_id']} ({item['source_size']} vs HQ {item['clean_size']})"
            for item in examples
        )
        raise RuntimeError(
            f"Evaluated images violate the {protocol['protocol'] if protocol else 'local'} "
            f"geometry requirement: "
            f"{geometry_audit['num_size_mismatches']}/{geometry_audit['num_images']} mismatches. "
            f"Examples: {rendered_examples}"
        )

    metric_definitions = (
        protocol["metrics"] if protocol is not None else _legacy_metric_definitions()
    )
    metrics = {
        name: pyiqa.create_metric(
            definition["pyiqa_name"],
            device=device,
            **definition.get("constructor_kwargs", {}),
        )
        for name, definition in metric_definitions.items()
    }
    no_reference = {
        name for name, definition in metric_definitions.items() if not definition["reference"]
    }
    image_loader = (
        str(protocol["image_processing"]["loader"])
        if protocol is not None
        else "pil_rgb_float"
    )
    per_image: list[dict[str, Any]] = []
    overall_values: list[dict[str, float]] = []
    by_group: dict[str, list[dict[str, float]]] = {}
    by_combination: dict[str, list[dict[str, float]]] = {}

    with torch.inference_mode():
        for row in tqdm(rows, desc="Paper IQA metrics"):
            if args.use_degraded_input:
                restored_path = _resolve(data_root, row["image_path"])
            else:
                assert restored_root is not None
                restored_path = restored_root / f"{row['state_id']}.png"
                if not restored_path.is_file():
                    raise FileNotFoundError(f"Missing restored image: {restored_path}")
            clean_path = Path(row["clean_path"])
            if not clean_path.is_absolute():
                clean_path = data_root / clean_path
            clean_tensor = _load_metric_tensor(clean_path, image_loader)
            restored_tensor = _load_metric_tensor(restored_path, image_loader)
            if restored_tensor.shape != clean_tensor.shape:
                restored_tensor = _align_metric_tensor(
                    restored_tensor,
                    clean_tensor,
                    geometry_policy,
                    state_id=str(row["state_id"]),
                )
            if crop_border > 0:
                if min(clean_tensor.shape[-2:]) <= 2 * crop_border:
                    raise ValueError(
                        f"crop_border={crop_border} is invalid for image shape "
                        f"{tuple(clean_tensor.shape[-2:])}"
                    )
                restored_tensor = restored_tensor[
                    ..., crop_border:-crop_border, crop_border:-crop_border
                ]
                clean_tensor = clean_tensor[
                    ..., crop_border:-crop_border, crop_border:-crop_border
                ]
            clean_tensor = clean_tensor.to(device)
            restored_tensor = restored_tensor.to(device)
            values: dict[str, float] = {}
            for name, metric in metrics.items():
                score = (
                    metric(restored_tensor)
                    if name in no_reference
                    else metric(restored_tensor, clean_tensor)
                )
                values[name] = float(score.item())
            group = str(row.get("group", "unspecified"))
            combination = str(row.get("combination", "unspecified"))
            per_image.append(
                {
                    "state_id": row["state_id"],
                    "group": group,
                    "combination": combination,
                    "metrics": values,
                }
            )
            overall_values.append(values)
            by_group.setdefault(group, []).append(values)
            by_combination.setdefault(f"{group}/{combination}", []).append(values)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "per_image.jsonl", per_image)
    by_combination_summary = {
        key: {"num_images": len(values), "aggregate": aggregate_metric_dicts(values)}
        for key, values in sorted(by_combination.items())
    }
    aggregation = (
        str(protocol.get("aggregation", {}).get("group", "image_macro"))
        if protocol is not None
        else "image_macro"
    )
    if aggregation == "combination_macro":
        grouped_combination_means: dict[str, list[dict[str, float]]] = {}
        for key, payload in by_combination_summary.items():
            group, _ = key.split("/", 1)
            grouped_combination_means.setdefault(group, []).append(
                {
                    metric_name: float(payload["aggregate"][metric_name])
                    for metric_name in EXPECTED_METRIC_NAMES
                }
            )
        by_group_summary = {
            group: {
                "num_images": len(by_group[group]),
                "num_combinations": len(combination_means),
                "aggregate": aggregate_metric_dicts(combination_means),
            }
            for group, combination_means in sorted(grouped_combination_means.items())
        }
        overall_aggregate = aggregate_metric_dicts(
            [
                {
                    metric_name: float(payload["aggregate"][metric_name])
                    for metric_name in EXPECTED_METRIC_NAMES
                }
                for payload in by_combination_summary.values()
            ]
        )
    elif aggregation == "image_macro":
        by_group_summary = {
            key: {"num_images": len(values), "aggregate": aggregate_metric_dicts(values)}
            for key, values in sorted(by_group.items())
        }
        overall_aggregate = aggregate_metric_dicts(overall_values)
    else:
        raise ValueError(f"Unsupported group aggregation: {aggregation!r}")

    full_protocol_run = bool(
        protocol is not None
        and args.manifest is None
        and source_kind == "restored_output"
        and len(per_image) == int(protocol["dataset"]["num_images"])
        and geometry_audit["passed"]
        and protocol_audit is not None
        and protocol_audit["passed"]
        and runtime_audit is not None
        and runtime_audit["passed"]
    )
    summary = {
        "num_images": len(per_image),
        "requested_groups": sorted(args.group) if args.group is not None else None,
        "source": source_kind,
        "manifest": str(manifest_path.resolve()),
        "benchmark_protocol": (
            {
                "name": protocol["protocol"],
                "config": protocol["_config_path"],
                "source": protocol["source"],
                "data_audit": protocol_audit,
                "runtime_audit": runtime_audit,
                "metric_definitions": protocol["metrics"],
                "image_processing": protocol["image_processing"],
                "aggregation": protocol.get("aggregation"),
                "data_metric_lock_passed": True,
                "main_table_eligible": full_protocol_run,
                "comparison_role": (
                    str(protocol.get("main_table_role", "protocol_main_table_candidate"))
                    if full_protocol_run
                    else "diagnostic_only"
                ),
            }
            if protocol is not None
            else None
        ),
        "geometry": {
            **geometry_audit,
            "crop_border": crop_border,
        },
        "metric_directions": {
            "psnr": "higher",
            "ssim": "higher",
            "lpips": "lower",
            "maniqa": "higher",
            "clipiqa": "higher",
            "musiq": "higher",
        },
        "aggregation": {
            "group": aggregation,
            "overall": (
                "combination_macro_extension"
                if aggregation == "combination_macro"
                else "image_macro"
            ),
        },
        "aggregate": overall_aggregate,
        "by_group": by_group_summary,
        "by_combination": by_combination_summary,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_paper_metrics_markdown(
        summary,
        output_root / "summary.md",
        title=str(args.report_title),
    )
    print(json.dumps(summary, indent=2))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _legacy_metric_definitions() -> dict[str, dict[str, Any]]:
    return {
        "psnr": {
            "pyiqa_name": "psnr",
            "reference": True,
            "constructor_kwargs": {"test_y_channel": True, "color_space": "ycbcr"},
        },
        "ssim": {
            "pyiqa_name": "ssim",
            "reference": True,
            "constructor_kwargs": {"test_y_channel": True, "color_space": "ycbcr"},
        },
        "lpips": {"pyiqa_name": "lpips", "reference": True},
        "maniqa": {"pyiqa_name": "maniqa", "reference": False},
        "clipiqa": {"pyiqa_name": "clipiqa", "reference": False},
        "musiq": {"pyiqa_name": "musiq", "reference": False},
    }


def _validate_metric_protocol(protocol: dict[str, Any]) -> None:
    metrics = protocol.get("metrics")
    if not isinstance(metrics, dict) or tuple(metrics) != EXPECTED_METRIC_NAMES:
        raise RuntimeError(
            f"Protocol metrics must be ordered as {EXPECTED_METRIC_NAMES}, found "
            f"{tuple(metrics) if isinstance(metrics, dict) else type(metrics).__name__}"
        )
    for name, definition in metrics.items():
        if definition.get("pyiqa_name") != name:
            raise RuntimeError(
                f"Metric {name!r} must use the identically named pyiqa model; found "
                f"{definition.get('pyiqa_name')!r}"
            )
        if not isinstance(definition.get("reference"), bool):
            raise RuntimeError(f"Metric {name!r} is missing a Boolean reference field")


def _load_metric_tensor(path: Path, loader: str) -> torch.Tensor:
    if loader == "opencv_bgr_to_rgb_uint8":
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    if loader == "pil_rgb_float":
        image = load_image(path, image_size=None)
        return image_to_tensor(image).unsqueeze(0)
    raise ValueError(f"Unsupported image loader: {loader!r}")


def _align_metric_tensor(
    image: torch.Tensor,
    reference: torch.Tensor,
    policy: str,
    *,
    state_id: str,
) -> torch.Tensor:
    if policy == "bicubic":
        image_np = image[0].permute(1, 2, 0).numpy()
        reference_np = reference[0].permute(1, 2, 0).numpy()
        aligned = align_to_reference(image_np, reference_np)
        return image_to_tensor(aligned).unsqueeze(0)
    if policy == "agenticir_matlab_x4":
        image_h, image_w = image.shape[-2:]
        reference_h, reference_w = reference.shape[-2:]
        if image_h * 4 != reference_h or image_w * 4 != reference_w:
            raise ValueError(
                f"AgenticIR only aligns exact x4 mismatches for {state_id}: "
                f"image={(image_w, image_h)}, reference={(reference_w, reference_h)}"
            )
        from basicsr.utils.matlab_functions import imresize

        return torch.clamp(imresize(image[0], scale=4).unsqueeze(0), 0, 1)
    raise ValueError(
        f"Image shapes do not match for {state_id}: image={tuple(image.shape)}, "
        f"reference={tuple(reference.shape)}, policy={policy!r}"
    )


def _audit_geometry(
    rows: list[dict[str, Any]],
    *,
    data_root: Path,
    restored_root: Path | None,
    use_degraded_input: bool,
    policy: str,
) -> dict[str, Any]:
    mismatch_by_group: Counter[str] = Counter()
    mismatch_examples: list[dict[str, Any]] = []
    mismatch_count = 0
    exact_x4_mismatch_count = 0
    invalid_mismatch_count = 0
    for row in rows:
        if use_degraded_input:
            source_path = _resolve(data_root, str(row["image_path"]))
        else:
            assert restored_root is not None
            source_path = restored_root / f"{row['state_id']}.png"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing evaluated image: {source_path}")
        clean_path = _resolve(data_root, str(row["clean_path"]))
        with Image.open(source_path) as source_image:
            source_size = source_image.size
        with Image.open(clean_path) as clean_image:
            clean_size = clean_image.size
        if source_size != clean_size:
            mismatch_count += 1
            is_exact_x4 = (
                source_size[0] * 4 == clean_size[0]
                and source_size[1] * 4 == clean_size[1]
            )
            if is_exact_x4:
                exact_x4_mismatch_count += 1
            else:
                invalid_mismatch_count += 1
            mismatch_by_group[str(row.get("group", "unspecified"))] += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "state_id": str(row["state_id"]),
                        "source_size": list(source_size),
                        "clean_size": list(clean_size),
                        "exact_x4": is_exact_x4,
                    }
                )
    return {
        "policy": policy,
        "num_images": len(rows),
        "num_size_mismatches": mismatch_count,
        "mismatch_by_group": dict(sorted(mismatch_by_group.items())),
        "mismatch_examples": mismatch_examples,
        "num_bicubic_aligned": mismatch_count if policy == "bicubic" else 0,
        "num_agenticir_matlab_x4_aligned": (
            exact_x4_mismatch_count if policy == "agenticir_matlab_x4" else 0
        ),
        "num_invalid_mismatches": invalid_mismatch_count,
        "passed": (
            policy == "bicubic"
            or mismatch_count == 0
            or (policy == "agenticir_matlab_x4" and invalid_mismatch_count == 0)
        ),
    }


def _audit_metric_runtime(
    protocol: dict[str, Any],
    asset_report: dict[str, Any],
    pyiqa_module: Any,
) -> dict[str, Any]:
    import torchvision

    expected = protocol["locked_local_runtime"]
    actual_packages = {}
    for name in expected["packages"]:
        if name == "torch":
            actual_packages[name] = str(torch.__version__)
        elif name == "torchvision":
            actual_packages[name] = str(torchvision.__version__)
        elif name == "pyiqa":
            actual_packages[name] = str(pyiqa_module.__version__)
        else:
            try:
                actual_packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                actual_packages[name] = None
    actual_assets = {
        name: sha256_file(details["path"])
        for name, details in asset_report["assets"].items()
        if details["exists"]
    }
    mismatches: list[str] = []
    for name, expected_version in expected["packages"].items():
        if actual_packages.get(name) != expected_version:
            mismatches.append(
                f"package {name}: expected {expected_version}, found {actual_packages.get(name)}"
            )
    for name, expected_hash in expected["asset_sha256"].items():
        if actual_assets.get(name) != expected_hash:
            mismatches.append(
                f"asset {name}: expected {expected_hash}, found {actual_assets.get(name)}"
            )
    pyiqa_defaults = _resolved_pyiqa_defaults(pyiqa_module, protocol["metrics"])
    expected_defaults = {}
    for name, definition in protocol["metrics"].items():
        expected_defaults[name] = {
            **definition.get("resolved_options", {}),
            **definition.get("architecture_defaults", {}),
        }
    for metric_name, expected_options in expected_defaults.items():
        actual_options = pyiqa_defaults.get(metric_name, {})
        for option, expected_value in expected_options.items():
            if actual_options.get(option) != expected_value:
                mismatches.append(
                    f"pyiqa default {metric_name}.{option}: expected {expected_value!r}, "
                    f"found {actual_options.get(option)!r}"
                )
    return {
        "environment": expected["environment"],
        "packages": actual_packages,
        "asset_sha256": actual_assets,
        "pyiqa_resolved_defaults": pyiqa_defaults,
        "mismatches": mismatches,
        "passed": not mismatches,
        "paper_version_note": expected["note"],
    }


def _resolved_pyiqa_defaults(
    pyiqa_module: Any,
    metric_definitions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    from pyiqa.default_model_configs import DEFAULT_CONFIGS
    from pyiqa.utils.registry import ARCH_REGISTRY

    resolved = {}
    for name, definition in metric_definitions.items():
        pyiqa_name = str(definition["pyiqa_name"])
        options = dict(DEFAULT_CONFIGS[pyiqa_name]["metric_opts"])
        options.update(definition.get("constructor_kwargs", {}))
        architecture_defaults = definition.get("architecture_defaults", {})
        if architecture_defaults:
            architecture_type = str(options["type"])
            signature = inspect.signature(ARCH_REGISTRY.get(architecture_type).__init__)
            for option in architecture_defaults:
                options[option] = signature.parameters[option].default
        resolved[name] = options
    return resolved


if __name__ == "__main__":
    main()
