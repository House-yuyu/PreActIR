from __future__ import annotations

from collections.abc import Iterable

from preactir.tools.base import RestorationTool
from preactir.tools.classical import (
    BrightenTool,
    DarkEnhanceTool,
    DeblurTool,
    DefocusDeblurTool,
    DehazeTool,
    DejpegTool,
    DenoiseTool,
    DerainTool,
    MotionDeblurTool,
    SuperResolutionTool,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[RestorationTool] | None = None) -> None:
        self._tools: dict[str, RestorationTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: RestorationTool, overwrite: bool = False) -> None:
        if tool.name in self._tools and not overwrite:
            raise KeyError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RestorationTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(self._tools)}") from exc

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def items(self):
        return self._tools.items()

    def target_map(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for name, tool in self._tools.items():
            for target in tool.targets:
                mapping.setdefault(target, []).append(name)
        return mapping


def build_default_registry(tool_names: list[str] | None = None) -> ToolRegistry:
    all_tools: dict[str, RestorationTool] = {
        "denoise": DenoiseTool(),
        "deblur": DeblurTool(),
        "brighten": BrightenTool(),
        "dehaze": DehazeTool(),
        "dejpeg": DejpegTool(),
        "derain": DerainTool(),
        "motion_deblur": MotionDeblurTool(),
        "defocus_deblur": DefocusDeblurTool(),
        "dark_enhance": DarkEnhanceTool(),
        "super_resolve": SuperResolutionTool(),
    }
    selected = tool_names or list(all_tools.keys())
    missing = sorted(set(selected) - set(all_tools))
    if missing:
        raise KeyError(f"No built-in implementation for tools: {missing}")
    return ToolRegistry(all_tools[name] for name in selected)


def build_registry_from_config(cfg, gpu_id: int | None = None) -> ToolRegistry:
    """Build the lightweight or paper tool registry selected by configuration."""

    backend = str(cfg.tools.get("backend", "classical"))
    tool_names = list(cfg.tools.names)
    if backend == "classical":
        return build_default_registry(tool_names)
    if backend == "agenticir_4kagent":
        from preactir.tools.paper_registry import build_paper_registry

        return build_paper_registry(
            external_root=str(cfg.tools.external_root),
            profile=str(cfg.tools.get("profile", "agenticir_default")),
            tool_names=tool_names,
            conda_env=str(cfg.tools.get("conda_env", "4kagent")),
            execution_mode=str(cfg.tools.get("execution_mode", "auto")),
            gpu_id=gpu_id,
        )
    raise ValueError(f"Unknown tools.backend={backend!r}")
