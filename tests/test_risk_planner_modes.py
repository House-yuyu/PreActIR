from __future__ import annotations

import numpy as np
import pytest
import torch

from preactir.agent.actions import CandidateAction, ScoredAction
from preactir.agent.planner import RiskAwarePlanner
from preactir.agent.risk_control import RiskControlPolicy


def scored(name: str, utility: float, admissible: bool) -> ScoredAction:
    action = CandidateAction(
        tool_name=name,
        tool_id=0,
        target_name="haze",
        target_index=0,
        strength=1.0,
        region_kind="global",
        region_mask=np.ones((4, 4), dtype=np.float32),
        cost_prior=1.0,
    )
    return ScoredAction(
        action=action,
        utility=utility,
        one_step_utility=utility,
        target_gain=0.0,
        quality_gain=0.0,
        side_effect=0.0,
        damage=0.0,
        uncertainty=0.0,
        accept_probability=0.5,
        harm_probability=0.8,
        risk_threshold=0.2,
        risk_admissible=admissible,
    )


def planner(mode: str) -> RiskAwarePlanner:
    return RiskAwarePlanner(
        torch.nn.Identity(),
        torch.device("cpu"),
        utility_weights={},
        risk_control_mode=mode,
    )


def test_hard_filter_preserves_v5_risk_veto() -> None:
    safe = scored("safe", 0.1, True)
    risky = scored("risky", 0.5, False)
    assert planner("hard_filter")._eligible_scores([safe, risky]) == [safe]


def test_rank_mode_keeps_reversible_risky_candidate() -> None:
    safe = scored("safe", 0.1, True)
    risky = scored("risky", 0.5, False)
    assert planner("rank")._eligible_scores([safe, risky]) == [safe, risky]


def test_unknown_risk_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="risk_control_mode"):
        planner("unknown")


def test_v2_risk_policy_prefers_enabled_tool_rule() -> None:
    policy = RiskControlPolicy(
        {
            "schema_version": 2,
            "fallback": "global",
            "global": {"enabled": True, "threshold": 0.8},
            "by_target": {"haze": {"enabled": True, "threshold": 0.4}},
            "by_tool": {"haze__maxim": {"enabled": True, "threshold": 0.2}},
        }
    )
    decision = policy.decision("haze__maxim", 0.3, "haze")
    assert not decision.admissible
    assert decision.threshold == pytest.approx(0.2)
    assert decision.calibration_source == "tool:haze__maxim"


def test_v2_risk_policy_falls_through_insufficient_tool_to_target() -> None:
    policy = RiskControlPolicy(
        {
            "schema_version": 2,
            "fallback": "global",
            "global": {"enabled": True, "threshold": 0.8},
            "by_target": {"haze": {"enabled": True, "threshold": 0.4}},
            "by_tool": {
                "haze__maxim": {
                    "enabled": False,
                    "terminal_reject": False,
                    "threshold": 0.0,
                }
            },
        }
    )
    decision = policy.decision("haze__maxim", 0.3, "haze")
    assert decision.admissible
    assert decision.threshold == pytest.approx(0.4)
    assert decision.calibration_source == "target:haze"


def test_v2_terminal_tool_reject_cannot_fall_back_to_permissive_target() -> None:
    policy = RiskControlPolicy(
        {
            "schema_version": 2,
            "fallback": "global",
            "global": {"enabled": True, "threshold": 0.8},
            "by_target": {"haze": {"enabled": True, "threshold": 0.4}},
            "by_tool": {
                "haze__maxim": {
                    "enabled": False,
                    "terminal_reject": True,
                    "threshold": 0.0,
                }
            },
        }
    )
    decision = policy.decision("haze__maxim", 0.1, "haze")
    assert not decision.admissible
    assert decision.calibration_source == "tool:haze__maxim:reject"


def test_v1_risk_policy_keeps_legacy_tool_then_global_behavior() -> None:
    policy = RiskControlPolicy(
        {
            "schema_version": 1,
            "fallback": "global",
            "global": {"enabled": True, "threshold": 0.6},
            "by_tool": {"haze__maxim": {"enabled": False, "threshold": 0.1}},
        }
    )
    decision = policy.decision("haze__maxim", 0.5, "haze")
    assert decision.admissible
    assert decision.threshold == pytest.approx(0.6)
    assert decision.calibration_source == "global"
