from __future__ import annotations

from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    ("psnr", "PSNR ↑", 4),
    ("ssim", "SSIM ↑", 5),
    ("lpips", "LPIPS ↓", 5),
    ("maniqa", "MANIQA ↑", 5),
    ("clipiqa", "CLIP-IQA ↑", 5),
    ("musiq", "MUSIQ ↑", 2),
)


def _format_metrics(aggregate: dict[str, Any]) -> list[str]:
    values = []
    for key, _, precision in METRIC_COLUMNS:
        value = aggregate.get(key)
        values.append("—" if value is None else f"{float(value):.{precision}f}")
    return values


def _table_header(first_column: str) -> list[str]:
    labels = [first_column, "Images", *[label for _, label, _ in METRIC_COLUMNS]]
    return [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---", "---:", *(["---:"] * len(METRIC_COLUMNS))]) + " |",
    ]


def paper_metrics_markdown(
    summary: dict[str, Any],
    *,
    title: str = "Image Restoration Paper Metrics",
) -> str:
    protocol = summary.get("benchmark_protocol")
    geometry = summary.get("geometry")
    aggregation = summary.get("aggregation")
    definitions = protocol.get("metric_definitions", {}) if isinstance(protocol, dict) else {}
    psnr_color = definitions.get("psnr", {}).get("effective_color_space")
    ssim_color = definitions.get("ssim", {}).get("effective_color_space")
    if psnr_color == "RGB":
        metric_note = (
            "PSNR: RGB; SSIM: Y channel (AgenticIR/pyiqa defaults); LPIPS is "
            "lower-is-better; MANIQA, CLIP-IQA, and MUSIQ are higher-is-better."
        )
    elif psnr_color == "Y channel in YCbCr" and ssim_color == "Y channel in YCbCr":
        metric_note = (
            "PSNR/SSIM: Y channel in YCbCr; LPIPS is lower-is-better; MANIQA, "
            "CLIP-IQA, and MUSIQ are higher-is-better."
        )
    else:
        metric_note = (
            "PSNR/SSIM: Y channel; LPIPS is lower-is-better; MANIQA, CLIP-IQA, "
            "and MUSIQ are higher-is-better."
        )
    lines = [
        f"# {title}",
        "",
        f"- Images: {int(summary.get('num_images', 0))}",
        f"- Source: `{summary.get('source', 'unspecified')}`",
        f"- {metric_note}",
        "- Full-precision means and standard deviations remain in `summary.json`; per-image values remain in `per_image.jsonl`.",
    ]
    if isinstance(protocol, dict):
        lines.extend(
            [
                f"- Benchmark protocol: `{protocol.get('name', 'unspecified')}`",
                f"- Comparison role: `{protocol.get('comparison_role', 'unspecified')}`; "
                f"main-table eligible: `{bool(protocol.get('main_table_eligible', False))}`",
            ]
        )
    if isinstance(geometry, dict):
        lines.append(
            f"- Geometry: policy `{geometry.get('policy', 'unspecified')}`, "
            f"size mismatches {int(geometry.get('num_size_mismatches', 0))}, "
            f"bicubic-aligned {int(geometry.get('num_bicubic_aligned', 0))}, "
            f"AgenticIR MATLAB-x4-aligned "
            f"{int(geometry.get('num_agenticir_matlab_x4_aligned', 0))}, "
            f"crop border {int(geometry.get('crop_border', 0))}."
        )
    if isinstance(aggregation, dict):
        lines.append(
            f"- Aggregation: groups use `{aggregation.get('group', 'unspecified')}`; "
            f"overall uses `{aggregation.get('overall', 'unspecified')}`."
        )
    lines.extend(["", "## Overall and groups", ""])
    lines.extend(_table_header("Split"))
    aggregate = summary.get("aggregate", {})
    lines.append(
        "| Overall | "
        + str(int(summary.get("num_images", 0)))
        + " | "
        + " | ".join(_format_metrics(aggregate))
        + " |"
    )
    for name, payload in sorted(summary.get("by_group", {}).items()):
        lines.append(
            f"| {name} | {int(payload.get('num_images', 0))} | "
            + " | ".join(_format_metrics(payload.get("aggregate", {})))
            + " |"
        )

    combinations = summary.get("by_combination", {})
    if combinations:
        lines.extend(["", "## Degradation combinations", ""])
        lines.extend(_table_header("Group / combination"))
        for name, payload in sorted(combinations.items()):
            safe_name = str(name).replace("|", "\\|")
            lines.append(
                f"| {safe_name} | {int(payload.get('num_images', 0))} | "
                + " | ".join(_format_metrics(payload.get("aggregate", {})))
                + " |"
            )
    return "\n".join(lines) + "\n"


def write_paper_metrics_markdown(
    summary: dict[str, Any],
    output: str | Path,
    *,
    title: str = "Image Restoration Paper Metrics",
) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(paper_metrics_markdown(summary, title=title), encoding="utf-8")
