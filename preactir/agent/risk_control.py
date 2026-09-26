from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskDecision:
    admissible: bool
    probability: float
    threshold: float
    calibration_source: str


class RiskControlPolicy:
    """Frozen pre-action selective-risk policy produced on a calibration split."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.schema_version = int(payload.get("schema_version", 0))
        if self.schema_version not in {1, 2}:
            raise ValueError("Unsupported risk-calibration schema")
        self.payload = payload
        self.global_rule = dict(payload["global"])
        self.by_tool = {
            str(name): dict(rule) for name, rule in payload.get("by_tool", {}).items()
        }
        self.by_target = {
            str(name): dict(rule)
            for name, rule in payload.get("by_target", {}).items()
        }
        self.fallback = str(payload.get("fallback", "global"))
        if self.fallback not in {"global", "reject"}:
            raise ValueError("Risk fallback must be 'global' or 'reject'")

    @classmethod
    def from_json(cls, path: str | Path) -> "RiskControlPolicy":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def decision(
        self,
        tool_name: str,
        harmful_probability: float,
        target_name: str | None = None,
    ) -> RiskDecision:
        probability = float(min(max(harmful_probability, 0.0), 1.0))
        if self.schema_version == 1:
            rule = self.by_tool.get(tool_name)
            source = f"tool:{tool_name}"
            if rule is None or not bool(rule.get("enabled", False)):
                if self.fallback == "reject":
                    return RiskDecision(
                        False, probability, 0.0, "fallback:reject"
                    )
                rule = self.global_rule
                source = "global"
            enabled = bool(rule.get("enabled", False))
            threshold = float(rule.get("threshold", 0.0))
            return RiskDecision(
                enabled and probability <= threshold,
                probability,
                threshold,
                source,
            )
        candidates = [(f"tool:{tool_name}", self.by_tool.get(tool_name))]
        if target_name is not None:
            candidates.append(
                (f"target:{target_name}", self.by_target.get(target_name))
            )
        candidates.append(("global", self.global_rule))
        rule = None
        source = "fallback:reject"
        for candidate_source, candidate_rule in candidates:
            if candidate_rule is not None and bool(
                candidate_rule.get("terminal_reject", False)
            ):
                return RiskDecision(
                    False, probability, 0.0, f"{candidate_source}:reject"
                )
            if candidate_rule is not None and bool(candidate_rule.get("enabled", False)):
                rule = candidate_rule
                source = candidate_source
                break
        if rule is None:
            return RiskDecision(False, probability, 0.0, source)
        enabled = bool(rule.get("enabled", False))
        threshold = float(rule.get("threshold", 0.0))
        return RiskDecision(enabled and probability <= threshold, probability, threshold, source)
