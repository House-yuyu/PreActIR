from __future__ import annotations

from pathlib import Path

import torch

from preactir.agent.candidate_generator import CandidateGenerator
from preactir.agent.diagnostics import DiagnosticProbes
from preactir.agent.gamma_guard import GammaSafetyGuard
from preactir.agent.planner import RiskAwarePlanner
from preactir.agent.risk_control import RiskControlPolicy
from preactir.agent.runner import PreActIRAgent
from preactir.agent.verifier import HybridTransitionVerifier, TransitionVerifier
from preactir.config import Config
from preactir.models.factory import build_belief_model, build_verifier_model, build_world_model
from preactir.tools.registry import ToolRegistry, build_registry_from_config
from preactir.utils.checkpoint import load_checkpoint


def _validate_checkpoint_vocabulary(
    checkpoint: dict,
    *,
    degradation_names: list[str],
    tool_names: list[str] | None = None,
) -> None:
    extra = checkpoint.get("extra", {})
    saved_degradations = extra.get("degradation_names")
    if saved_degradations is not None and list(saved_degradations) != degradation_names:
        raise ValueError(
            "Checkpoint degradation order does not match config: "
            f"checkpoint={list(saved_degradations)}, config={degradation_names}"
        )
    saved_tools = extra.get("tool_names")
    if tool_names is not None and saved_tools is not None and list(saved_tools) != tool_names:
        raise ValueError(
            "Checkpoint tool order does not match config; tool embeddings would be invalid: "
            f"checkpoint={list(saved_tools)}, config={tool_names}"
        )


def build_agent(
    cfg: Config,
    belief_checkpoint: str | Path,
    world_checkpoint: str | Path,
    device: torch.device,
    verifier_checkpoint: str | Path | None = None,
    gamma_guard_path: str | Path | None = None,
    risk_calibration_path: str | Path | None = None,
    learned_verifier_threshold_override: float | None = None,
    learned_verifier_mode_override: str | None = None,
    learned_min_relative_target_gain_override: float | None = None,
    block_target_on_progress_guard_reject_override: bool | None = None,
    presence_threshold_override: float | dict[str, float] | None = None,
    severity_threshold_override: float | dict[str, float] | None = None,
    enabled_tools: list[str] | None = None,
    registry: ToolRegistry | None = None,
) -> PreActIRAgent:
    """Build a complete PreActIR inference agent from checkpoints.

    A custom registry can be supplied to replace the lightweight built-in tools
    with pretrained restoration networks while keeping the planner unchanged.
    """

    degradation_names = list(cfg.degradations.names)
    tool_names = list(cfg.tools.names)
    registry = registry or build_registry_from_config(cfg)
    agent_cfg = cfg.agent
    presence_threshold_config = (
        presence_threshold_override
        if presence_threshold_override is not None
        else agent_cfg.get("presence_thresholds", agent_cfg.presence_threshold)
    )
    severity_threshold_config = (
        severity_threshold_override
        if severity_threshold_override is not None
        else agent_cfg.get(
            "severity_thresholds",
            agent_cfg.get("severity_threshold", float(agent_cfg.resolved_threshold) * 0.45),
        )
    )
    presence_threshold = (
        [float(presence_threshold_config[name]) for name in degradation_names]
        if isinstance(presence_threshold_config, dict)
        else float(presence_threshold_config)
    )
    severity_threshold = (
        [float(severity_threshold_config[name]) for name in degradation_names]
        if isinstance(severity_threshold_config, dict)
        else float(severity_threshold_config)
    )

    belief_model = build_belief_model(cfg).to(device)
    world_model = build_world_model(cfg).to(device)
    belief_payload = load_checkpoint(belief_checkpoint, belief_model, map_location=device)
    world_payload = load_checkpoint(world_checkpoint, world_model, map_location=device)
    _validate_checkpoint_vocabulary(
        belief_payload,
        degradation_names=degradation_names,
    )
    _validate_checkpoint_vocabulary(
        world_payload,
        degradation_names=degradation_names,
        tool_names=tool_names,
    )
    belief_model.eval()
    world_model.eval()

    candidate_generator = CandidateGenerator(
        degradation_names=degradation_names,
        tool_names=tool_names,
        registry=registry,
        strengths=[float(value) for value in cfg.tools.strengths],
        presence_threshold=presence_threshold,
        severity_threshold=severity_threshold,
        top_active_degradations=int(agent_cfg.top_active_degradations),
        use_spatial_masks=bool(agent_cfg.use_spatial_masks),
        region_threshold=float(agent_cfg.region_threshold),
        enabled_tools=enabled_tools,
    )
    planner = RiskAwarePlanner(
        world_model,
        device,
        utility_weights={key: float(value) for key, value in agent_cfg.utility_weights.items()},
        horizon=int(agent_cfg.horizon),
        discount=float(agent_cfg.discount),
        beam_size=int(agent_cfg.beam_size),
        image_size=int(cfg.data.image_size),
        risk_control=(
            RiskControlPolicy.from_json(risk_calibration_path)
            if risk_calibration_path is not None
            else None
        ),
        risk_control_mode=str(agent_cfg.get("risk_control_mode", "hard_filter")),
    )
    heuristic_verifier = TransitionVerifier(
        min_target_gain=float(agent_cfg.min_target_gain),
        max_side_effect=float(agent_cfg.max_side_effect),
        max_outside_change=float(agent_cfg.max_outside_change),
        resolved_threshold=float(agent_cfg.resolved_threshold),
        presence_threshold=float(agent_cfg.presence_threshold),
    )
    verifier = heuristic_verifier
    if verifier_checkpoint is not None:
        learned_verifier = build_verifier_model(cfg).to(device)
        verifier_payload = load_checkpoint(verifier_checkpoint, learned_verifier, map_location=device)
        _validate_checkpoint_vocabulary(
            verifier_payload,
            degradation_names=degradation_names,
            tool_names=tool_names,
        )
        verifier_threshold_config = (
            float(learned_verifier_threshold_override)
            if learned_verifier_threshold_override is not None
            else agent_cfg.get(
                "learned_verifier_thresholds", agent_cfg.learned_verifier_threshold
            )
        )
        verifier_threshold = (
            [
                float(
                    verifier_threshold_config.get(
                        name, agent_cfg.learned_verifier_threshold
                    )
                )
                for name in tool_names
            ]
            if isinstance(verifier_threshold_config, dict)
            else float(verifier_threshold_config)
        )
        verifier = HybridTransitionVerifier(
            heuristic=heuristic_verifier,
            model=learned_verifier,
            device=device,
            tool_names=tool_names,
            threshold=verifier_threshold,
            mode=(
                str(learned_verifier_mode_override)
                if learned_verifier_mode_override is not None
                else str(agent_cfg.learned_verifier_mode)
            ),
            learned_weight=float(agent_cfg.learned_verifier_weight),
            learned_min_relative_target_gain=(
                float(learned_min_relative_target_gain_override)
                if learned_min_relative_target_gain_override is not None
                else float(agent_cfg.get("learned_min_relative_target_gain", 0.01))
            ),
            image_size=int(cfg.data.image_size),
            gamma_guard=(
                GammaSafetyGuard.from_json(gamma_guard_path)
                if gamma_guard_path is not None
                else None
            ),
            paper_lcb_z=float(agent_cfg.get("learned_paper_lcb_z", 0.0)),
        )

    diagnostic_probes = None
    if bool(agent_cfg.use_diagnostic_probes):
        diagnostic_probes = DiagnosticProbes(degradation_names)

    return PreActIRAgent(
        belief_model=belief_model,
        planner=planner,
        candidate_generator=candidate_generator,
        verifier=verifier,
        registry=registry,
        device=device,
        max_steps=int(agent_cfg.max_steps),
        presence_threshold=presence_threshold,
        severity_threshold=severity_threshold,
        min_planning_utility=float(agent_cfg.min_planning_utility),
        diagnostic_probes=diagnostic_probes,
        diagnostic_uncertainty_threshold=float(agent_cfg.diagnostic_uncertainty_threshold),
        diagnostic_blend=float(agent_cfg.diagnostic_blend),
        perception_image_size=int(cfg.data.image_size),
        allow_action_repetition=bool(agent_cfg.get("allow_action_repetition", True)),
        max_accepted_steps=(
            int(agent_cfg.max_accepted_steps)
            if agent_cfg.get("max_accepted_steps") is not None
            else None
        ),
        block_target_on_progress_guard_reject=(
            bool(block_target_on_progress_guard_reject_override)
            if block_target_on_progress_guard_reject_override is not None
            else bool(agent_cfg.get("block_target_on_progress_guard_reject", False))
        ),
    )
