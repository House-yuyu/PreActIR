from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Protocol

import numpy as np
import torch

from preactir.agent.candidate_generator import CandidateGenerator
from preactir.agent.diagnostics import DiagnosticProbes
from preactir.agent.planner import RiskAwarePlanner
from preactir.agent.verifier import VerificationResult
from preactir.models.belief_encoder import SpatialBeliefEncoder
from preactir.tools.registry import ToolRegistry
from preactir.utils.image import image_to_tensor, resize_and_center_crop_image


class VerifierProtocol(Protocol):
    def verify(self, *args, **kwargs) -> VerificationResult: ...


class PreActIRAgent:
    def __init__(
        self,
        belief_model: SpatialBeliefEncoder,
        planner: RiskAwarePlanner,
        candidate_generator: CandidateGenerator,
        verifier: VerifierProtocol,
        registry: ToolRegistry,
        device: torch.device,
        max_steps: int = 8,
        presence_threshold: float | list[float] | np.ndarray = 0.45,
        severity_threshold: float | list[float] | np.ndarray = 0.08,
        min_planning_utility: float = -0.05,
        diagnostic_probes: DiagnosticProbes | None = None,
        diagnostic_uncertainty_threshold: float = 0.30,
        diagnostic_blend: float = 0.30,
        perception_image_size: int | None = 256,
        allow_action_repetition: bool = True,
        max_accepted_steps: int | None = None,
        block_target_on_progress_guard_reject: bool = False,
    ) -> None:
        self.belief_model = belief_model.to(device).eval()
        self.planner = planner
        self.candidate_generator = candidate_generator
        self.verifier = verifier
        self.registry = registry
        self.device = device
        self.max_steps = int(max_steps)
        num_degradations = len(candidate_generator.degradation_names)
        presence_threshold_array = np.asarray(presence_threshold, dtype=np.float32)
        severity_threshold_array = np.asarray(severity_threshold, dtype=np.float32)
        if presence_threshold_array.ndim == 0:
            presence_threshold_array = np.full(
                num_degradations, float(presence_threshold_array), dtype=np.float32
            )
        if severity_threshold_array.ndim == 0:
            severity_threshold_array = np.full(
                num_degradations, float(severity_threshold_array), dtype=np.float32
            )
        if presence_threshold_array.shape != (num_degradations,):
            raise ValueError("presence_threshold must be scalar or match degradation count")
        if severity_threshold_array.shape != (num_degradations,):
            raise ValueError("severity_threshold must be scalar or match degradation count")
        self.presence_threshold = presence_threshold_array
        self.severity_threshold = severity_threshold_array
        self.min_planning_utility = float(min_planning_utility)
        self.diagnostic_probes = diagnostic_probes
        self.diagnostic_uncertainty_threshold = float(diagnostic_uncertainty_threshold)
        self.diagnostic_blend = float(diagnostic_blend)
        self.perception_image_size = (
            None if perception_image_size is None else int(perception_image_size)
        )
        self.allow_action_repetition = bool(allow_action_repetition)
        self.max_accepted_steps = (
            None if max_accepted_steps is None else int(max_accepted_steps)
        )
        self.block_target_on_progress_guard_reject = bool(
            block_target_on_progress_guard_reject
        )
        if self.max_accepted_steps is not None and self.max_accepted_steps < 1:
            raise ValueError("max_accepted_steps must be positive or None")

    @torch.inference_mode()
    def perceive(self, image: np.ndarray) -> dict[str, np.ndarray | dict[str, float] | None]:
        model_image = image
        if self.perception_image_size is not None:
            model_image = resize_and_center_crop_image(image, self.perception_image_size)
        tensor = image_to_tensor(model_image).unsqueeze(0).to(self.device)
        output = self.belief_model.predict_belief(tensor)
        presence = output["presence_prob"][0].cpu().numpy().astype(np.float32)
        severity = output["severity_mu"][0].cpu().numpy().astype(np.float32)
        uncertainty = output["severity_std"][0].cpu().numpy().astype(np.float32)
        diagnostics: dict[str, float] | None = None
        if self.diagnostic_probes is not None and float(uncertainty.max(initial=0.0)) >= self.diagnostic_uncertainty_threshold:
            probe_output = self.diagnostic_probes.run(model_image)
            presence, severity = self.diagnostic_probes.refine(
                presence,
                severity,
                uncertainty,
                probe_output,
                uncertainty_threshold=self.diagnostic_uncertainty_threshold,
                blend=self.diagnostic_blend,
            )
            diagnostics = probe_output.scores
        return {
            "presence": presence,
            "severity": severity,
            "uncertainty": uncertainty,
            "masks": output["mask_prob"][0].cpu().numpy().astype(np.float32),
            "diagnostics": diagnostics,
        }

    def restore(
        self,
        image: np.ndarray,
        max_output_shape: tuple[int, int] | None = None,
        candidate_image_writer: Callable[[int, np.ndarray, bool], str | None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Restore an image under an optional ``(height, width)`` budget.

        The budget is especially important for x4 super-resolution actions:
        once the target geometry has been reached, another x4 action must not
        be proposed merely because the belief model still predicts blur/LR.
        """
        current = image.copy().astype(np.float32)
        if max_output_shape is not None:
            max_output_shape = (int(max_output_shape[0]), int(max_output_shape[1]))
            if min(max_output_shape) <= 0:
                raise ValueError(f"Invalid max_output_shape={max_output_shape}")
            if current.shape[0] > max_output_shape[0] or current.shape[1] > max_output_shape[1]:
                raise ValueError(
                    f"Input shape {current.shape[:2]} exceeds output budget {max_output_shape}"
                )
        rejected_signatures: set[str] = set()
        rejected_target_indices: set[int] = set()
        executed_signatures: set[str] = set()
        steps: list[dict[str, Any]] = []
        termination = "budget_exhausted"
        resolution_guard_rejections = 0
        accepted_steps = 0
        self.planner.last_plan_diagnostics = {}

        for step_index in range(self.max_steps):
            # Do not leak a previous image/step's planner diagnostics into a
            # trace when candidate generation terminates before plan() runs.
            self.planner.last_plan_diagnostics = {}
            before_belief = self.perceive(current)
            presence = np.asarray(before_belief["presence"], dtype=np.float32)
            severity = np.asarray(before_belief["severity"], dtype=np.float32)
            masks = np.asarray(before_belief["masks"], dtype=np.float32)
            active = (presence >= self.presence_threshold) & (severity >= self.severity_threshold)
            if not bool(active.any()):
                termination = "no_active_degradation"
                break

            candidates = self.candidate_generator.generate(
                presence,
                severity,
                masks,
                rejected_signatures | executed_signatures,
            )
            if self.block_target_on_progress_guard_reject:
                candidates = [
                    candidate
                    for candidate in candidates
                    if candidate.target_index not in rejected_target_indices
                ]
            if max_output_shape is not None:
                admissible = []
                for candidate in candidates:
                    scale = float(self.registry.get(candidate.tool_name).output_scale)
                    predicted_height = int(round(current.shape[0] * scale))
                    predicted_width = int(round(current.shape[1] * scale))
                    if (
                        predicted_height > max_output_shape[0]
                        or predicted_width > max_output_shape[1]
                    ):
                        resolution_guard_rejections += 1
                        continue
                    admissible.append(candidate)
                candidates = admissible
            if not candidates:
                termination = (
                    "resolution_budget_reached"
                    if resolution_guard_rejections
                    else "no_admissible_action"
                )
                break
            trajectory_context = np.asarray(
                [
                    min(accepted_steps / 3.0, 1.0),
                    min(step_index / 6.0, 1.0),
                    min(len(rejected_signatures) / 6.0, 1.0),
                    1.0,
                ],
                dtype=np.float32,
            )
            selected = self.planner.plan(
                current,
                presence,
                severity,
                candidates,
                trajectory_context=trajectory_context,
            )
            if selected is None:
                termination = (
                    "pre_action_risk_abstention"
                    if self.planner.last_plan_diagnostics.get("reason")
                    == "risk_control_rejected_all"
                    else "planner_returned_none"
                )
                break
            if selected.utility < self.min_planning_utility:
                termination = "risk_aware_abstention"
                break

            tool_result = self.registry.get(selected.action.tool_name).run(
                current,
                strength=selected.action.strength,
                region_mask=selected.action.region_mask,
            )
            if not self.allow_action_repetition:
                executed_signatures.add(selected.action.signature)
            candidate_image = tool_result.image
            if max_output_shape is not None and (
                candidate_image.shape[0] > max_output_shape[0]
                or candidate_image.shape[1] > max_output_shape[1]
            ):
                raise RuntimeError(
                    f"Tool {selected.action.tool_name} violated the output geometry budget: "
                    f"produced {candidate_image.shape[:2]}, budget={max_output_shape}"
                )
            after_belief = self.perceive(candidate_image)
            after_presence = np.asarray(after_belief["presence"], dtype=np.float32)
            after_severity = np.asarray(after_belief["severity"], dtype=np.float32)
            verification = self.verifier.verify(
                current,
                candidate_image,
                presence,
                severity,
                after_presence,
                after_severity,
                selected.action,
                trajectory_context,
            )
            candidate_image_path = None
            if candidate_image_writer is not None:
                candidate_image_path = candidate_image_writer(
                    step_index, candidate_image, bool(verification.accepted)
                )

            observed_delta = severity - after_severity
            predicted_delta = np.asarray(selected.predicted_delta_degradation, dtype=np.float32)
            prediction_error = float(np.mean(np.abs(observed_delta - predicted_delta)))
            step_record = {
                "step": step_index,
                "action": {
                    "tool_name": selected.action.tool_name,
                    "target_name": selected.action.target_name,
                    "target_index": selected.action.target_index,
                    "strength": selected.action.strength,
                    "region_kind": selected.action.region_kind,
                    "mask_area": float(selected.action.region_mask.mean()),
                    "signature": selected.action.signature,
                },
                "planning": {
                    "utility": selected.utility,
                    "one_step_utility": selected.one_step_utility,
                    "lookahead_value": selected.lookahead_value,
                    "target_gain": selected.target_gain,
                    "quality_gain": selected.quality_gain,
                    "paper_psnr_gain": selected.paper_psnr_gain,
                    "paper_ssim_gain": selected.paper_ssim_gain,
                    "paper_utility_gain": selected.paper_utility_gain,
                    "side_effect": selected.side_effect,
                    "damage": selected.damage,
                    "uncertainty": selected.uncertainty,
                    "accept_probability": selected.accept_probability,
                    "harm_probability": selected.harm_probability,
                    "severe_harm_probability": selected.severe_harm_probability,
                    "risk_threshold": selected.risk_threshold,
                    "risk_admissible": selected.risk_admissible,
                    "risk_calibration_source": selected.risk_calibration_source,
                    "predicted_delta_degradation": selected.predicted_delta_degradation,
                    "predicted_delta_quality": selected.predicted_delta_quality,
                },
                "verification": asdict(verification),
                "transition_prediction_error": prediction_error,
                "elapsed_ms": tool_result.elapsed_ms,
                "before_presence": presence.tolist(),
                "before_severity": severity.tolist(),
                "after_presence": after_presence.tolist(),
                "after_severity": after_severity.tolist(),
                "before_diagnostics": before_belief.get("diagnostics"),
                "after_diagnostics": after_belief.get("diagnostics"),
                "candidate_image_path": candidate_image_path,
                "trajectory_context": trajectory_context.tolist(),
            }
            steps.append(step_record)

            if verification.accepted:
                current = candidate_image
                accepted_steps += 1
                if (
                    self.max_accepted_steps is not None
                    and accepted_steps >= self.max_accepted_steps
                ):
                    termination = "accepted_step_budget_reached"
                    break
                # Rejected actions are state-conditional. Once a verified action
                # changes the state, reconsider them under the updated belief.
                rejected_signatures.clear()
                rejected_target_indices.clear()
            else:
                rejected_signatures.add(selected.action.signature)
                if (
                    self.block_target_on_progress_guard_reject
                    and verification.reason == "learned_progress_guard_reject"
                ):
                    # No observed target progress is evidence that the current
                    # target hypothesis is not actionable in this state. Avoid
                    # cycling through every tool for that same target; a later
                    # accepted transition clears this state-conditional block.
                    rejected_target_indices.add(selected.action.target_index)

        trace = {
            "termination": termination,
            "steps": steps,
            "tool_calls": len(steps),
            "accepted_calls": sum(int(step["verification"]["accepted"]) for step in steps),
            "rejected_calls": sum(int(not step["verification"]["accepted"]) for step in steps),
            "mean_transition_prediction_error": float(
                np.mean([step["transition_prediction_error"] for step in steps]) if steps else 0.0
            ),
            "max_output_shape": list(max_output_shape) if max_output_shape is not None else None,
            "resolution_guard_rejections": resolution_guard_rejections,
            "action_repetition_blocked": not self.allow_action_repetition,
            "target_rejection_blocked": self.block_target_on_progress_guard_reject,
            "blocked_target_indices": sorted(rejected_target_indices),
            "max_accepted_steps": self.max_accepted_steps,
            "risk_control_enabled": self.planner.risk_control is not None,
            "risk_control_mode": self.planner.risk_control_mode,
            "candidate_images_saved": candidate_image_writer is not None,
            "last_plan_diagnostics": self.planner.last_plan_diagnostics,
        }
        return current, trace
