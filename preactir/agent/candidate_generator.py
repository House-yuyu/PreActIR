from __future__ import annotations

import numpy as np

from preactir.agent.actions import CandidateAction
from preactir.tools.registry import ToolRegistry


class CandidateGenerator:
    def __init__(
        self,
        degradation_names: list[str],
        tool_names: list[str],
        registry: ToolRegistry,
        strengths: list[float],
        presence_threshold: float | list[float] | np.ndarray = 0.45,
        severity_threshold: float | list[float] | np.ndarray = 0.0,
        top_active_degradations: int = 3,
        use_spatial_masks: bool = True,
        region_threshold: float = 0.45,
        enabled_tools: list[str] | None = None,
    ) -> None:
        self.degradation_names = degradation_names
        self.tool_names = tool_names
        self.tool_to_id = {name: index for index, name in enumerate(tool_names)}
        self.registry = registry
        self.strengths = [float(value) for value in strengths]
        presence_threshold_array = np.asarray(presence_threshold, dtype=np.float32)
        severity_threshold_array = np.asarray(severity_threshold, dtype=np.float32)
        if presence_threshold_array.ndim == 0:
            presence_threshold_array = np.full(
                len(degradation_names), float(presence_threshold_array), dtype=np.float32
            )
        if severity_threshold_array.ndim == 0:
            severity_threshold_array = np.full(
                len(degradation_names), float(severity_threshold_array), dtype=np.float32
            )
        if presence_threshold_array.shape != (len(degradation_names),):
            raise ValueError("presence_threshold must be scalar or match degradation_names")
        if severity_threshold_array.shape != (len(degradation_names),):
            raise ValueError("severity_threshold must be scalar or match degradation_names")
        self.presence_threshold = presence_threshold_array
        self.severity_threshold = severity_threshold_array
        self.top_active_degradations = int(top_active_degradations)
        self.use_spatial_masks = bool(use_spatial_masks)
        self.region_threshold = float(region_threshold)
        unknown_enabled_tools = sorted(set(enabled_tools or []) - set(tool_names))
        if unknown_enabled_tools:
            raise ValueError(f"Unknown enabled tools: {unknown_enabled_tools}")
        self.enabled_tools = None if enabled_tools is None else frozenset(enabled_tools)
        self.target_map = registry.target_map()

    def generate(
        self,
        presence: np.ndarray,
        severity: np.ndarray,
        mask_probability: np.ndarray,
        rejected_signatures: set[str] | None = None,
    ) -> list[CandidateAction]:
        rejected_signatures = rejected_signatures or set()
        priority = presence * np.maximum(severity, 0.05)
        active_indices = [
            int(index)
            for index in np.argsort(-priority)
            if presence[index] >= self.presence_threshold[index]
            and severity[index] >= self.severity_threshold[index]
        ][: self.top_active_degradations]
        candidates: list[CandidateAction] = []
        height, width = mask_probability.shape[-2:]
        global_mask = np.ones((height, width), dtype=np.float32)
        for target_index in active_indices:
            target_name = self.degradation_names[target_index]
            for tool_name in self.target_map.get(target_name, []):
                if tool_name not in self.tool_to_id:
                    continue
                if self.enabled_tools is not None and tool_name not in self.enabled_tools:
                    continue
                tool = self.registry.get(tool_name)
                masks: list[tuple[str, np.ndarray]] = [("global", global_mask)]
                if self.use_spatial_masks:
                    local = np.clip(mask_probability[target_index], 0.0, 1.0)
                    hard = (local >= self.region_threshold).astype(np.float32)
                    area = float(hard.mean())
                    if 0.03 <= area <= 0.95:
                        # Keep a soft boundary while zeroing very low-probability pixels.
                        local = local * hard
                        maximum = float(local.max())
                        if maximum > 0:
                            local = local / maximum
                        masks.append(("local", local.astype(np.float32)))
                for strength in self.strengths:
                    for region_kind, region_mask in masks:
                        action = CandidateAction(
                            tool_name=tool_name,
                            tool_id=self.tool_to_id[tool_name],
                            target_name=target_name,
                            target_index=target_index,
                            strength=float(strength),
                            region_kind=region_kind,
                            region_mask=region_mask.copy(),
                            cost_prior=float(tool.cost_prior),
                        )
                        if action.signature not in rejected_signatures:
                            candidates.append(action)
        return candidates
