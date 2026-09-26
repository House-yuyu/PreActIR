from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CandidateAction:
    tool_name: str
    tool_id: int
    target_name: str
    target_index: int
    strength: float
    region_kind: str
    region_mask: np.ndarray
    cost_prior: float

    @property
    def signature(self) -> str:
        return f"{self.tool_name}:{self.target_index}:{self.strength:.3f}:{self.region_kind}"


@dataclass
class ScoredAction:
    action: CandidateAction
    utility: float
    one_step_utility: float
    target_gain: float
    quality_gain: float
    side_effect: float
    damage: float
    uncertainty: float
    accept_probability: float
    paper_psnr_gain: float = 0.0
    paper_ssim_gain: float = 0.0
    paper_utility_gain: float = 0.0
    harm_probability: float = 0.0
    severe_harm_probability: float = 0.0
    risk_threshold: float = 1.0
    risk_admissible: bool = True
    risk_calibration_source: str = "disabled"
    predicted_delta_degradation: list[float] = field(default_factory=list)
    predicted_delta_quality: list[float] = field(default_factory=list)
    lookahead_value: float = 0.0
