from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from preactir.tools.external_executor import ExternalExecutorTool, _load_4kagent_executor
from preactir.tools.registry import ToolRegistry


@dataclass(frozen=True)
class PaperToolSpec:
    name: str
    target: str
    subtask: str
    backend_name: str
    cost_prior: float
    output_scale: float


def _spec(
    target: str,
    subtask: str,
    backend: str,
    cost: float = 1.0,
    output_scale: float = 1.0,
) -> PaperToolSpec:
    return PaperToolSpec(f"{target}__{backend}", target, subtask, backend, cost, output_scale)


# Exact task/model layout released by AgenticIR . 4KAgent uses
# ``hat_psnr`` for the deterministic HAT checkpoint that corresponds to HAT.
AGENTICIR_DEFAULT_SPECS: tuple[PaperToolSpec, ...] = (
    _spec("rain", "deraining", "maxim", 1.2),
    _spec("rain", "deraining", "xrestormer", 0.8),
    _spec("rain", "deraining", "restormer", 0.8),
    _spec("rain", "deraining", "mprnet", 0.7),
    _spec("haze", "dehazing", "xrestormer", 0.8),
    _spec("haze", "dehazing", "ridcp", 1.3),
    _spec("haze", "dehazing", "dehazeformer", 0.6),
    _spec("haze", "dehazing", "maxim", 1.2),
    _spec("motion_blur", "motion deblurring", "restormer", 0.8),
    _spec("motion_blur", "motion deblurring", "mprnet", 0.7),
    _spec("motion_blur", "motion deblurring", "maxim", 1.2),
    _spec("motion_blur", "motion deblurring", "xrestormer", 0.8),
    _spec("low_resolution", "super-resolution", "diffbir", 4.0, output_scale=4.0),
    _spec("low_resolution", "super-resolution", "xrestormer", 0.8, output_scale=4.0),
    _spec("low_resolution", "super-resolution", "swinir_gan", 1.0, output_scale=4.0),
    _spec("low_resolution", "super-resolution", "swinir_psnr", 1.0, output_scale=4.0),
    _spec("low_resolution", "super-resolution", "hat_psnr", 1.1, output_scale=4.0),
    _spec("dark", "brightening", "histogram_equalization", 0.05),
    _spec("dark", "brightening", "gamma_correction", 0.02),
    _spec("dark", "brightening", "constant_shift", 0.02),
    _spec("noise", "denoising", "xrestormer", 0.8),
    _spec("noise", "denoising", "swinir_15", 1.0),
    _spec("noise", "denoising", "swinir_50", 1.0),
    _spec("noise", "denoising", "mprnet", 0.7),
    _spec("noise", "denoising", "maxim", 1.2),
    _spec("noise", "denoising", "restormer", 0.8),
    _spec("defocus_blur", "defocus deblurring", "drbnet", 0.7),
    _spec("defocus_blur", "defocus deblurring", "restormer", 0.8),
    _spec("defocus_blur", "defocus deblurring", "ifan", 1.0),
    _spec("jpeg", "jpeg compression artifact removal", "fbcnn_blind", 0.6),
    _spec("jpeg", "jpeg compression artifact removal", "fbcnn_5", 0.6),
    _spec("jpeg", "jpeg compression artifact removal", "fbcnn_90", 0.6),
    _spec("jpeg", "jpeg compression artifact removal", "swinir_40", 1.0),
)


PAPER_PROFILES = {"agenticir_default": AGENTICIR_DEFAULT_SPECS}


def paper_tool_names(profile: str = "agenticir_default") -> list[str]:
    try:
        return [spec.name for spec in PAPER_PROFILES[profile]]
    except KeyError as exc:
        raise KeyError(f"Unknown paper tool profile {profile!r}; available={sorted(PAPER_PROFILES)}") from exc


def build_paper_registry(
    *,
    external_root: str | Path,
    profile: str = "agenticir_default",
    tool_names: list[str] | None = None,
    conda_env: str = "4kagent",
    execution_mode: str = "auto",
    gpu_id: int | None = None,
) -> ToolRegistry:
    try:
        specs = PAPER_PROFILES[profile]
    except KeyError as exc:
        raise KeyError(f"Unknown paper tool profile {profile!r}; available={sorted(PAPER_PROFILES)}") from exc
    by_name = {spec.name: spec for spec in specs}
    selected = tool_names or list(by_name)
    missing = sorted(set(selected) - set(by_name))
    if missing:
        raise KeyError(f"Tools absent from paper profile {profile}: {missing}")
    return ToolRegistry(
        ExternalExecutorTool(
            name=by_name[name].name,
            target=by_name[name].target,
            subtask=by_name[name].subtask,
            backend_name=by_name[name].backend_name,
            external_root=external_root,
            conda_env=conda_env,
            execution_mode=execution_mode,
            cost_prior=by_name[name].cost_prior,
            output_scale=by_name[name].output_scale,
            gpu_id=gpu_id,
        )
        for name in selected
    )


def inspect_paper_backend(
    external_root: str | Path,
    profile: str = "agenticir_default",
) -> dict[str, object]:
    root = Path(external_root).resolve()
    executor = _load_4kagent_executor(root)
    specs = PAPER_PROFILES[profile]
    missing_tools: list[str] = []
    for spec in specs:
        available = [tool.tool_name for tool in executor.toolbox_router.get(spec.subtask, [])]
        if spec.backend_name not in available:
            missing_tools.append(spec.name)
    diffbir_weights = (
        root / "executor/super_resolution/tools/DiffBIR/weights"
    )
    lazy_assets = {
        "jpeg_fbcnn": root
        / "executor/jpeg_compression_artifact_removal/tools/FBCNN/FBCNN/model_zoo/fbcnn_color.pth",
        # DiffBIR's loader resolves URL basenames relative to its working
        # directory. Checking Torch Hub alone can therefore report a false
        # positive while inference still attempts a network download.
        "diffbir_bsrnet": diffbir_weights / "BSRNet.pth",
        "diffbir_stable_diffusion": diffbir_weights / "v2-1_512-ema-pruned.ckpt",
        "diffbir_controlnet_v2": diffbir_weights / "v2.pth",
    }
    lazy_report = {
        name: {"path": str(path), "exists": path.is_file() and path.stat().st_size > 0}
        for name, path in lazy_assets.items()
    }
    return {
        "external_root": str(root),
        "profile": profile,
        "num_tools": len(specs),
        "missing_tools": missing_tools,
        "lazy_assets": lazy_report,
        "missing_lazy_assets": [name for name, item in lazy_report.items() if not item["exists"]],
    }


def inspect_paper_metric_assets() -> dict[str, object]:
    cache_root = Path.home() / ".cache"
    maniqa_backbone_root = (
        cache_root
        / "huggingface/hub/models--timm--vit_base_patch8_224.augreg2_in21k_ft_in1k"
    )
    maniqa_backbone_ref = maniqa_backbone_root / "refs/main"
    maniqa_backbone_revision = (
        maniqa_backbone_ref.read_text(encoding="utf-8").strip()
        if maniqa_backbone_ref.is_file()
        else "missing"
    )
    assets = {
        # AgenticIR pins pyiqa 0.1.10. That release resolves URL-based model
        # files under torch.hub/checkpoints rather than the newer pyiqa/
        # subdirectory. Formal preflight must inspect the path the locked
        # runtime will actually open.
        "lpips": cache_root / "torch/hub/checkpoints/LPIPS_v0.1_alex-df73285e.pth",
        "maniqa": cache_root / "torch/hub/checkpoints/ckpt_koniq10k.pt",
        "maniqa_backbone":
            maniqa_backbone_root
            / "snapshots"
            / maniqa_backbone_revision
            / "model.safetensors",
        "musiq": cache_root / "torch/hub/checkpoints/musiq_koniq_ckpt-e95806b9.pth",
        # pyiqa 0.1.10's CLIPIQA constructor defaults to OpenAI CLIP RN50.
        "clipiqa": cache_root / "clip/RN50.pt",
    }
    report = {
        name: {"path": str(path), "exists": path.is_file() and path.stat().st_size > 0}
        for name, path in assets.items()
    }
    return {
        "assets": report,
        "missing": [name for name, item in report.items() if not item["exists"]],
    }
