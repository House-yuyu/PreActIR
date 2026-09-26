from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from preactir.agent.factory import build_agent
from preactir.agent.gamma_guard import GAMMA_FEATURE_NAMES, GammaSafetyGuard
from preactir.agent.risk_control import RiskControlPolicy
from preactir.agent.verifier import (
    HybridTransitionVerifier,
    TransitionVerifier,
    VerificationResult,
)
from preactir.config import load_config
from preactir.data.builder import BuilderOptions, InterventionDatasetBuilder
from preactir.data.datasets import BeliefDataset, GroupedBatchSampler, TransitionDataset
from preactir.models.factory import build_belief_model, build_verifier_model, build_world_model
from preactir.models.losses import belief_loss, verifier_model_loss, world_model_loss
from preactir.tools.registry import build_default_registry
from preactir.utils.checkpoint import save_checkpoint
from preactir.utils.image import load_image, resize_and_center_crop_image, save_image
from preactir.utils.risk_metrics import clopper_pearson_upper, select_risk_threshold


def _make_clean_images(root: Path, size: int = 64) -> None:
    rng = np.random.default_rng(123)
    for split, count in (("train", 3), ("val", 2), ("test", 2)):
        split_root = root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
            image = np.stack(
                [
                    0.45 + 0.35 * np.sin((xx + index) / 7.0),
                    0.45 + 0.35 * np.cos((yy + index) / 9.0),
                    0.45 + 0.25 * np.sin((xx + yy) / 11.0),
                ],
                axis=-1,
            )
            image += rng.normal(0.0, 0.005, image.shape).astype(np.float32)
            save_image(split_root / f"{index:03d}.png", np.clip(image, 0.0, 1.0))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / "configs/preactir_debug.yaml")
    work_root = project_root / "outputs/smoke"
    if work_root.exists():
        shutil.rmtree(work_root)
    clean_root = work_root / "clean"
    data_root = work_root / "data"
    _make_clean_images(clean_root, int(cfg.data.image_size))
    wide = np.linspace(0.0, 1.0, 40 * 80 * 3, dtype=np.float32).reshape(40, 80, 3)
    wide_path = work_root / "wide.png"
    save_image(wide_path, wide)
    offline_preprocessed = load_image(wide_path, int(cfg.data.image_size))
    online_preprocessed = resize_and_center_crop_image(
        load_image(wide_path), int(cfg.data.image_size)
    )
    assert np.array_equal(offline_preprocessed, online_preprocessed)

    registry = build_default_registry(list(cfg.tools.names))
    builder = InterventionDatasetBuilder(
        clean_root=clean_root,
        output_root=data_root,
        degradation_names=list(cfg.degradations.names),
        registry=registry,
        options=BuilderOptions(
            image_size=int(cfg.data.image_size),
            max_degradations=2,
            states_per_image=1,
            max_actions_per_state=3,
            spatial_probability=0.5,
            strengths=(0.4, 0.8),
            seed=int(cfg.seed),
        ),
    )
    builder.build()

    belief_data = BeliefDataset(data_root, "train", list(cfg.degradations.names), image_size=64)
    transition_data = TransitionDataset(
        data_root,
        "train",
        list(cfg.degradations.names),
        list(cfg.tools.names),
        image_size=64,
    )

    belief_batch = belief_data[0]
    belief_model = build_belief_model(cfg)
    belief_output = belief_model(belief_batch["image"].unsqueeze(0))
    belief_targets = {
        "presence": belief_batch["presence"].unsqueeze(0),
        "severity": belief_batch["severity"].unsqueeze(0),
        "masks": belief_batch["masks"].unsqueeze(0),
    }
    belief_total, _ = belief_loss(
        belief_output,
        belief_targets,
        {key: float(value) for key, value in cfg.train_belief.loss_weights.items()},
    )
    assert torch.isfinite(belief_total)

    item = transition_data[0]
    transition_batch = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in item.items()
    }
    world_model = build_world_model(cfg)
    world_output = world_model(
        image=transition_batch["image"],
        belief=transition_batch["belief"],
        action_mask=transition_batch["action_mask"],
        tool_id=transition_batch["tool_id"],
        target_index=transition_batch["target_index"],
        strength=transition_batch["strength"],
        mask_area=transition_batch["mask_area"],
        cost_prior=transition_batch["cost_prior"],
    )
    world_total, _ = world_model_loss(
        world_output,
        transition_batch,
        {key: float(value) for key, value in cfg.train_world.loss_weights.items()},
    )
    assert torch.isfinite(world_total)

    risk_cfg = load_config(project_root / "configs/preactir_debug.yaml")
    risk_cfg.world_model["predict_harm"] = True
    risk_cfg.world_model["predict_severe_harm"] = True
    risk_world_model = build_world_model(risk_cfg)
    risk_output = risk_world_model(
        image=transition_batch["image"],
        belief=transition_batch["belief"],
        action_mask=transition_batch["action_mask"],
        tool_id=transition_batch["tool_id"],
        target_index=transition_batch["target_index"],
        strength=transition_batch["strength"],
        mask_area=transition_batch["mask_area"],
        cost_prior=transition_batch["cost_prior"],
    )
    assert risk_output["harm_logit"].shape == (1,)
    assert risk_output["severe_harm_logit"].shape == (1,)
    risk_total, risk_terms = world_model_loss(
        risk_output,
        transition_batch,
        {"harm": 1.0, "severe_harm": 1.0},
    )
    assert torch.isfinite(risk_total)
    assert torch.isfinite(risk_terms["severe_harm"])

    paper_cfg = load_config(project_root / "configs/preactir_debug.yaml")
    paper_cfg.world_model["predict_paper_quality"] = True
    paper_cfg.world_model["trajectory_context_dim"] = 4
    paper_world_model = build_world_model(paper_cfg)
    paper_output = paper_world_model(
        image=transition_batch["image"],
        belief=transition_batch["belief"],
        action_mask=transition_batch["action_mask"],
        tool_id=transition_batch["tool_id"],
        target_index=transition_batch["target_index"],
        strength=transition_batch["strength"],
        mask_area=transition_batch["mask_area"],
        cost_prior=transition_batch["cost_prior"],
        trajectory_context=transition_batch["trajectory_context"],
    )
    assert paper_output["paper_quality_mu"].shape == (1, 2)
    assert paper_output["paper_quality_logvar"].shape == (1, 2)
    paper_total, paper_terms = world_model_loss(
        paper_output,
        transition_batch,
        {"paper_quality": 1.0, "paper_ranking": 1.0},
    )
    assert torch.isfinite(paper_total)
    assert torch.isfinite(paper_terms["paper_quality"])

    grouped_sampler = GroupedBatchSampler(
        ["a", "a", "b", "b", "c"], batch_size=3, shuffle=False, seed=0
    )
    grouped_batches = list(grouped_sampler)
    assert grouped_batches == [[0, 1], [2, 3, 4]]
    batch_lookup = {
        index: batch_index
        for batch_index, indices in enumerate(grouped_batches)
        for index in indices
    }
    assert batch_lookup[0] == batch_lookup[1]
    assert batch_lookup[2] == batch_lookup[3]
    assert clopper_pearson_upper(0, 20, 0.05) < 0.15
    calibrated = select_risk_threshold(
        np.asarray([0.05] * 20 + [0.9] * 5),
        np.asarray([0] * 20 + [1] * 5),
        alpha=0.2,
        delta=0.05,
        min_selected=10,
    )
    assert calibrated["enabled"] and calibrated["selected"] == 20
    risk_policy = RiskControlPolicy(
        {
            "schema_version": 1,
            "fallback": "global",
            "global": calibrated,
            "by_tool": {},
        }
    )
    assert risk_policy.decision("unseen", 0.05).admissible
    assert not risk_policy.decision("unseen", 0.9).admissible

    verifier_model = build_verifier_model(cfg)
    verifier_output = verifier_model(
        before=transition_batch["image"],
        after=transition_batch["next_image"],
        action_mask=transition_batch["action_mask"],
        tool_id=transition_batch["tool_id"],
        target_index=transition_batch["target_index"],
        strength=transition_batch["strength"],
        mask_area=transition_batch["mask_area"],
        cost_prior=transition_batch["cost_prior"],
    )
    verifier_total, _ = verifier_model_loss(
        verifier_output,
        transition_batch,
        {key: float(value) for key, value in cfg.train_verifier.loss_weights.items()},
    )
    assert torch.isfinite(verifier_total)
    paper_cfg.verifier_model["predict_paper_quality"] = True
    paper_cfg.verifier_model["trajectory_context_dim"] = 4
    paper_verifier_model = build_verifier_model(paper_cfg)
    paper_verifier_output = paper_verifier_model(
        before=transition_batch["image"],
        after=transition_batch["next_image"],
        action_mask=transition_batch["action_mask"],
        tool_id=transition_batch["tool_id"],
        target_index=transition_batch["target_index"],
        strength=transition_batch["strength"],
        mask_area=transition_batch["mask_area"],
        cost_prior=transition_batch["cost_prior"],
        trajectory_context=transition_batch["trajectory_context"],
    )
    assert paper_verifier_output["paper_quality_mu"].shape == (1, 2)
    assert paper_verifier_output["paper_accept_logit"].shape == (1,)
    paper_verifier_total, paper_verifier_terms = verifier_model_loss(
        paper_verifier_output,
        transition_batch,
        {"paper_quality": 1.0, "paper_accept": 1.0},
    )
    assert torch.isfinite(paper_verifier_total)
    assert torch.isfinite(paper_verifier_terms["paper_quality"])
    assert torch.isfinite(paper_verifier_terms["paper_accept"])
    threshold_probe = HybridTransitionVerifier(
        heuristic=TransitionVerifier(),
        model=verifier_model,
        device=torch.device("cpu"),
        tool_names=["tool_a", "tool_b"],
        threshold=[0.4, 0.8],
        image_size=64,
    )
    assert np.isclose(threshold_probe.threshold_for_tool("tool_a"), 0.4)
    assert np.isclose(threshold_probe.threshold_for_tool("tool_b"), 0.8)
    assert threshold_probe.threshold_for_tool("missing") == 1.0
    guard_base = VerificationResult(
        accepted=False,
        status="Rejected",
        target_gain=0.02,
        max_side_effect=0.01,
        outside_change=0.01,
        reason="insufficient_target_reduction",
    )
    guard_passed, relative_gain, required_gain = (
        threshold_probe._passes_learned_progress_guard(guard_base, before_target=1.0)
    )
    assert guard_passed and np.isclose(relative_gain, 0.02)
    assert np.isclose(required_gain, 0.01)
    guard_base.target_gain = 0.005
    assert not threshold_probe._passes_learned_progress_guard(
        guard_base, before_target=1.0
    )[0]
    guard_base.target_gain = 0.02
    guard_base.max_side_effect = 0.09
    assert not threshold_probe._passes_learned_progress_guard(
        guard_base, before_target=1.0
    )[0]
    feature_count = len(GAMMA_FEATURE_NAMES)
    gamma_guard = GammaSafetyGuard(
        mean=np.zeros(feature_count, dtype=np.float32),
        scale=np.ones(feature_count, dtype=np.float32),
        coefficients=np.zeros(feature_count, dtype=np.float32),
        intercept=0.0,
        threshold=0.4,
    )
    before_probe = transition_batch["image"][0].permute(1, 2, 0).numpy()
    after_probe = transition_batch["next_image"][0].permute(1, 2, 0).numpy()
    guard_accepted, guard_probability = gamma_guard.accepts(before_probe, after_probe)
    assert guard_accepted and np.isclose(guard_probability, 0.5)

    checkpoint_root = work_root / "checkpoints"
    save_checkpoint(checkpoint_root / "belief.pt", belief_model)
    save_checkpoint(checkpoint_root / "world.pt", world_model)
    save_checkpoint(checkpoint_root / "verifier.pt", verifier_model)
    agent = build_agent(
        cfg,
        belief_checkpoint=checkpoint_root / "belief.pt",
        world_checkpoint=checkpoint_root / "world.pt",
        verifier_checkpoint=checkpoint_root / "verifier.pt",
        block_target_on_progress_guard_reject_override=True,
        device=torch.device("cpu"),
    )
    assert agent.block_target_on_progress_guard_reject
    presence = np.zeros(len(cfg.degradations.names), dtype=np.float32)
    severity = np.zeros_like(presence)
    presence[0] = agent.candidate_generator.presence_threshold[0] + 0.01
    severity[0] = max(0.0, agent.candidate_generator.severity_threshold[0] - 0.01)
    masks = np.ones((len(presence), 64, 64), dtype=np.float32)
    assert not agent.candidate_generator.generate(presence, severity, masks)

    safe_agent = build_agent(
        cfg,
        belief_checkpoint=checkpoint_root / "belief.pt",
        world_checkpoint=checkpoint_root / "world.pt",
        device=torch.device("cpu"),
        enabled_tools=[str(cfg.tools.names[0])],
    )
    safe_agent.candidate_generator.presence_threshold[:] = 0.0
    safe_agent.candidate_generator.severity_threshold[:] = 0.0
    safe_candidates = safe_agent.candidate_generator.generate(
        np.ones_like(presence), np.ones_like(severity), masks
    )
    assert safe_candidates
    assert {candidate.tool_name for candidate in safe_candidates} == {str(cfg.tools.names[0])}

    image = transition_batch["image"][0].permute(1, 2, 0).numpy()
    restored, trace = agent.restore(image, max_output_shape=image.shape[:2])
    assert restored.shape == image.shape
    assert np.isfinite(restored).all()
    assert "termination" in trace and "steps" in trace
    assert trace["max_output_shape"] == list(image.shape[:2])
    print("Smoke test passed:", {"states": len(belief_data), "transitions": len(transition_data), "termination": trace["termination"]})


if __name__ == "__main__":
    main()
