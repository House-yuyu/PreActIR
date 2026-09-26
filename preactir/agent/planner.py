from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from preactir.agent.actions import CandidateAction, ScoredAction
from preactir.agent.risk_control import RiskControlPolicy
from preactir.models.world_model import ToolConditionedWorldModel
from preactir.utils.image import image_to_tensor, mask_to_tensor, resize_and_center_crop_image


class RiskAwarePlanner:
    def __init__(
        self,
        model: ToolConditionedWorldModel,
        device: torch.device,
        utility_weights: dict[str, float],
        horizon: int = 2,
        discount: float = 0.85,
        beam_size: int = 4,
        image_size: int | None = 256,
        risk_control: RiskControlPolicy | None = None,
        risk_control_mode: str = "hard_filter",
    ) -> None:
        self.model = model
        self.device = device
        self.weights = {key: float(value) for key, value in utility_weights.items()}
        self.horizon = max(1, int(horizon))
        self.discount = float(discount)
        self.beam_size = max(1, int(beam_size))
        self.image_size = None if image_size is None else int(image_size)
        self.risk_control = risk_control
        self.risk_control_mode = str(risk_control_mode)
        if self.risk_control_mode not in {"hard_filter", "rank"}:
            raise ValueError(
                "risk_control_mode must be 'hard_filter' or 'rank', got "
                f"{self.risk_control_mode!r}"
            )
        self.last_plan_diagnostics: dict[str, object] = {}

    def _eligible_scores(self, scores: list[ScoredAction]) -> list[ScoredAction]:
        """Apply the configured pre-action risk semantics.

        ``hard_filter`` preserves the v5 selective-risk behavior. ``rank`` is
        intended for reversible restoration trials: calibrated risk remains in
        the score and trace, but the post-action verifier decides whether the
        candidate is committed.
        """

        if self.risk_control_mode == "hard_filter":
            return [item for item in scores if item.risk_admissible]
        return scores

    @torch.inference_mode()
    def score_candidates(
        self,
        image: np.ndarray,
        presence: np.ndarray,
        severity: np.ndarray,
        candidates: list[CandidateAction],
        trajectory_context: np.ndarray | None = None,
    ) -> list[ScoredAction]:
        if not candidates:
            return []
        self.model.eval()
        count = len(candidates)
        model_image = image
        if self.image_size is not None:
            model_image = resize_and_center_crop_image(image, self.image_size)
        image_tensor = image_to_tensor(model_image).unsqueeze(0).to(self.device).repeat(count, 1, 1, 1)
        belief_np = np.concatenate([presence, severity], axis=0).astype(np.float32)
        belief = torch.from_numpy(belief_np).to(self.device).unsqueeze(0).repeat(count, 1)
        masks = torch.stack([mask_to_tensor(action.region_mask) for action in candidates], dim=0).to(self.device)
        tool_ids = torch.tensor([action.tool_id for action in candidates], dtype=torch.long, device=self.device)
        target_indices = torch.tensor(
            [action.target_index for action in candidates], dtype=torch.long, device=self.device
        )
        strengths = torch.tensor([action.strength for action in candidates], dtype=torch.float32, device=self.device)
        mask_areas = torch.tensor(
            [float(action.region_mask.mean()) for action in candidates], dtype=torch.float32, device=self.device
        )
        costs = torch.tensor([action.cost_prior for action in candidates], dtype=torch.float32, device=self.device)
        trajectory_context_tensor = None
        if int(getattr(self.model, "trajectory_context_dim", 0)):
            expected_dim = int(self.model.trajectory_context_dim)
            if trajectory_context is None:
                trajectory_context = np.zeros(expected_dim, dtype=np.float32)
            trajectory_context = np.asarray(trajectory_context, dtype=np.float32)
            if trajectory_context.shape != (expected_dim,):
                raise ValueError(
                    f"trajectory_context must have shape ({expected_dim},), got "
                    f"{trajectory_context.shape}"
                )
            trajectory_context_tensor = (
                torch.from_numpy(trajectory_context)
                .to(self.device)
                .unsqueeze(0)
                .repeat(count, 1)
            )
        output = self.model(
            image=image_tensor,
            belief=belief,
            action_mask=masks,
            tool_id=tool_ids,
            target_index=target_indices,
            strength=strengths,
            mask_area=mask_areas,
            cost_prior=costs,
            trajectory_context=trajectory_context_tensor,
        )

        delta_deg = output["delta_degradation_mu"].cpu().numpy()
        delta_quality = output["delta_quality_mu"].cpu().numpy()
        damage = output["damage_mu"].cpu().numpy()
        accept_prob = torch.sigmoid(output["accept_logit"]).cpu().numpy()
        harm_prob = (
            torch.sigmoid(output["harm_logit"]).cpu().numpy()
            if "harm_logit" in output
            else 1.0 - accept_prob
        )
        severe_harm_prob = (
            torch.sigmoid(output["severe_harm_logit"]).cpu().numpy()
            if "severe_harm_logit" in output
            else harm_prob
        )
        deg_std = torch.exp(0.5 * output["delta_degradation_logvar"]).mean(dim=1).cpu().numpy()
        quality_std = torch.exp(0.5 * output["delta_quality_logvar"]).mean(dim=1).cpu().numpy()
        damage_std = torch.exp(0.5 * output["damage_logvar"]).cpu().numpy()
        paper_quality = (
            output["paper_quality_mu"].cpu().numpy()
            if "paper_quality_mu" in output
            else np.zeros((count, 2), dtype=np.float32)
        )
        paper_std = (
            torch.exp(0.5 * output["paper_quality_logvar"])
            .mean(dim=1)
            .cpu()
            .numpy()
            if "paper_quality_logvar" in output
            else np.zeros(count, dtype=np.float32)
        )

        scored: list[ScoredAction] = []
        for index, action in enumerate(candidates):
            target_gain = float(delta_deg[index, action.target_index])
            other = np.delete(delta_deg[index], action.target_index)
            side_effect = float(np.maximum(-other, 0.0).sum())
            quality_gain = float(delta_quality[index].mean())
            paper_psnr_gain = float(paper_quality[index, 0] * 10.0)
            paper_ssim_gain = float(paper_quality[index, 1])
            paper_utility_gain = float(paper_psnr_gain + 10.0 * paper_ssim_gain)
            uncertainty_terms = [deg_std[index], quality_std[index], damage_std[index]]
            if "paper_quality_logvar" in output:
                uncertainty_terms.append(paper_std[index])
            uncertainty = float(np.mean(uncertainty_terms))
            if self.risk_control is None:
                risk_admissible = True
                risk_threshold = 1.0
                risk_source = "disabled"
            else:
                risk_decision = self.risk_control.decision(
                    action.tool_name,
                    float(severe_harm_prob[index]),
                    action.target_name,
                )
                risk_admissible = risk_decision.admissible
                risk_threshold = risk_decision.threshold
                risk_source = risk_decision.calibration_source
            one_step = (
                self.weights.get("target_gain", 2.0) * target_gain
                + self.weights.get("quality_gain", 0.8) * quality_gain
                + self.weights.get("paper_utility", 0.0) * paper_utility_gain
                - self.weights.get("side_effect", 1.5) * side_effect
                - self.weights.get("damage", 2.0) * float(damage[index])
                - self.weights.get("uncertainty", 0.35) * uncertainty
                - self.weights.get("harm", 0.0) * float(harm_prob[index])
                - self.weights.get("severe_harm", 0.0)
                * float(severe_harm_prob[index])
                - self.weights.get("risk_excess", 0.0)
                * max(
                    0.0,
                    float(severe_harm_prob[index]) - float(risk_threshold),
                )
                - self.weights.get("cost", 0.05) * float(action.cost_prior)
            )
            # A low predicted commit probability should directly reduce expected utility.
            expected = float(accept_prob[index]) * one_step - (1.0 - float(accept_prob[index])) * 0.05
            scored.append(
                ScoredAction(
                    action=action,
                    utility=expected,
                    one_step_utility=expected,
                    target_gain=target_gain,
                    quality_gain=quality_gain,
                    paper_psnr_gain=paper_psnr_gain,
                    paper_ssim_gain=paper_ssim_gain,
                    paper_utility_gain=paper_utility_gain,
                    side_effect=side_effect,
                    damage=float(damage[index]),
                    uncertainty=uncertainty,
                    accept_probability=float(accept_prob[index]),
                    harm_probability=float(harm_prob[index]),
                    severe_harm_probability=float(severe_harm_prob[index]),
                    risk_threshold=float(risk_threshold),
                    risk_admissible=bool(risk_admissible),
                    risk_calibration_source=risk_source,
                    predicted_delta_degradation=delta_deg[index].astype(float).tolist(),
                    predicted_delta_quality=delta_quality[index].astype(float).tolist(),
                )
            )
        return scored

    def plan(
        self,
        image: np.ndarray,
        presence: np.ndarray,
        severity: np.ndarray,
        candidates: list[CandidateAction],
        trajectory_context: np.ndarray | None = None,
    ) -> ScoredAction | None:
        first_scores = self.score_candidates(
            image, presence, severity, candidates, trajectory_context
        )
        if not first_scores:
            self.last_plan_diagnostics = {"reason": "no_candidates", "num_candidates": 0}
            return None
        eligible_scores = self._eligible_scores(first_scores)
        num_risk_admissible = sum(int(item.risk_admissible) for item in first_scores)
        self.last_plan_diagnostics = {
            "reason": "planned" if eligible_scores else "risk_control_rejected_all",
            "num_candidates": len(first_scores),
            "num_risk_admissible": num_risk_admissible,
            "risk_control_enabled": self.risk_control is not None,
            "risk_control_mode": self.risk_control_mode,
        }
        if not eligible_scores:
            return None
        if self.horizon <= 1:
            best = max(eligible_scores, key=lambda item: item.utility)
            self.last_plan_diagnostics["selected"] = self._diagnostic_score(best)
            return best

        # Short-horizon model-predictive lookahead. The current image feature is
        # held fixed while the predicted belief is advanced; the real image is
        # re-encoded after the selected first action is actually executed.
        ranked_first = sorted(eligible_scores, key=lambda item: item.utility, reverse=True)[: self.beam_size]
        best: ScoredAction | None = None
        for first in ranked_first:
            next_severity = np.clip(
                severity - np.asarray(first.predicted_delta_degradation, dtype=np.float32), 0.0, 1.0
            )
            next_presence = np.where(next_severity > 0.08, presence, 0.0).astype(np.float32)
            remaining = [action for action in candidates if action.signature != first.action.signature]
            future = self._future_value(
                image,
                next_presence,
                next_severity,
                remaining,
                depth=self.horizon - 1,
                trajectory_context=trajectory_context,
            )
            total = first.one_step_utility + self.discount * future
            candidate = replace(first, utility=float(total), lookahead_value=float(future))
            if best is None or candidate.utility > best.utility:
                best = candidate
        if best is not None:
            self.last_plan_diagnostics["selected"] = self._diagnostic_score(best)
        return best

    @staticmethod
    def _diagnostic_score(item: ScoredAction) -> dict[str, object]:
        """Return a compact pre-execution audit record, including abstained plans."""

        return {
            "tool_name": item.action.tool_name,
            "target_name": item.action.target_name,
            "utility": float(item.utility),
            "one_step_utility": float(item.one_step_utility),
            "lookahead_value": float(item.lookahead_value),
            "accept_probability": float(item.accept_probability),
            "harm_probability": float(item.harm_probability),
            "severe_harm_probability": float(item.severe_harm_probability),
            "paper_psnr_gain": float(item.paper_psnr_gain),
            "paper_ssim_gain": float(item.paper_ssim_gain),
            "paper_utility_gain": float(item.paper_utility_gain),
            "risk_threshold": float(item.risk_threshold),
            "risk_admissible": bool(item.risk_admissible),
            "risk_calibration_source": item.risk_calibration_source,
        }

    def _future_value(
        self,
        image: np.ndarray,
        presence: np.ndarray,
        severity: np.ndarray,
        candidates: list[CandidateAction],
        depth: int,
        trajectory_context: np.ndarray | None = None,
    ) -> float:
        if depth <= 0 or not candidates or float(presence.max(initial=0.0)) <= 0:
            return 0.0
        active_candidates = [
            action for action in candidates if presence[action.target_index] > 0.0 and severity[action.target_index] > 0.05
        ]
        if not active_candidates:
            return 0.0
        scores = self._eligible_scores(
            self.score_candidates(
                image,
                presence,
                severity,
                active_candidates,
                trajectory_context,
            )
        )
        if not scores:
            return 0.0
        top = sorted(scores, key=lambda item: item.utility, reverse=True)[: self.beam_size]
        if depth == 1:
            return float(max(0.0, top[0].utility))
        best = 0.0
        for item in top:
            next_severity = np.clip(
                severity - np.asarray(item.predicted_delta_degradation, dtype=np.float32), 0.0, 1.0
            )
            next_presence = np.where(next_severity > 0.08, presence, 0.0).astype(np.float32)
            remaining = [action for action in active_candidates if action.signature != item.action.signature]
            value = item.one_step_utility + self.discount * self._future_value(
                image,
                next_presence,
                next_severity,
                remaining,
                depth - 1,
                trajectory_context,
            )
            best = max(best, float(value))
        return best
