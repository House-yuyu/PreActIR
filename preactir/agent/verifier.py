from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from preactir.agent.actions import CandidateAction
from preactir.agent.gamma_guard import GammaSafetyGuard
from preactir.models.verifier_model import LearnedTransitionVerifier
from preactir.utils.image import (
    image_to_tensor,
    mask_to_tensor,
    resize_and_center_crop_image,
    resize_image,
)


@dataclass
class VerificationResult:
    accepted: bool
    status: str
    target_gain: float
    max_side_effect: float
    outside_change: float
    reason: str
    heuristic_accepted: bool | None = None
    learned_probability: float | None = None
    learned_threshold: float | None = None
    learned_status: str | None = None
    combined_score: float | None = None
    gamma_guard_accepted: bool | None = None
    gamma_guard_probability: float | None = None
    gamma_guard_threshold: float | None = None
    learned_progress_guard_passed: bool | None = None
    learned_observed_relative_target_gain: float | None = None
    learned_required_target_gain: float | None = None
    learned_paper_accept_probability: float | None = None
    learned_paper_psnr_gain_db: float | None = None
    learned_paper_ssim_gain: float | None = None
    learned_paper_psnr_std_db: float | None = None
    learned_paper_ssim_std: float | None = None
    learned_paper_psnr_lcb_db: float | None = None
    learned_paper_ssim_lcb: float | None = None
    learned_paper_gate_passed: bool | None = None


class TransitionVerifier:
    """Belief-change and content-change verifier used without additional training."""

    def __init__(
        self,
        min_target_gain: float = 0.015,
        max_side_effect: float = 0.08,
        max_outside_change: float = 0.08,
        resolved_threshold: float = 0.18,
        presence_threshold: float = 0.45,
    ) -> None:
        self.min_target_gain = float(min_target_gain)
        self.max_side_effect = float(max_side_effect)
        self.max_outside_change = float(max_outside_change)
        self.resolved_threshold = float(resolved_threshold)
        self.presence_threshold = float(presence_threshold)

    def verify(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        before_presence: np.ndarray,
        before_severity: np.ndarray,
        after_presence: np.ndarray,
        after_severity: np.ndarray,
        action: CandidateAction,
        trajectory_context: np.ndarray | None = None,
    ) -> VerificationResult:
        del trajectory_context
        target = action.target_index
        before_target = float(before_presence[target] * before_severity[target])
        after_target = float(after_presence[target] * after_severity[target])
        target_gain = before_target - after_target

        change = after_presence * after_severity - before_presence * before_severity
        other_change = np.delete(change, target)
        max_side_effect = float(max(0.0, other_change.max(initial=0.0)))

        comparison_size = after_image.shape[:2]
        before_aligned = resize_image(before_image, comparison_size)
        mask = action.region_mask
        if mask.shape != comparison_size:
            mask = cv2.resize(
                mask.astype(np.float32),
                (comparison_size[1], comparison_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        outside = np.clip(1.0 - mask, 0.0, 1.0)
        difference = np.mean(np.abs(after_image - before_aligned), axis=2)
        if float(outside.sum()) > 1e-6:
            outside_change = float((difference * outside).sum() / (outside.sum() + 1e-6))
        else:
            outside_change = 0.0

        resolved = bool(
            after_presence[target] < self.presence_threshold
            or after_severity[target] < self.resolved_threshold
        )
        progress = target_gain >= self.min_target_gain or resolved
        safe = max_side_effect <= self.max_side_effect and outside_change <= self.max_outside_change
        accepted = bool(progress and safe)
        if not progress:
            reason = "insufficient_target_reduction"
        elif max_side_effect > self.max_side_effect:
            reason = "cross_degradation_side_effect"
        elif outside_change > self.max_outside_change:
            reason = "outside_region_content_change"
        else:
            reason = "verified"
        status = "Rejected" if not accepted else ("Resolved" if resolved else "Progress")
        return VerificationResult(
            accepted=accepted,
            status=status,
            target_gain=float(target_gain),
            max_side_effect=max_side_effect,
            outside_change=outside_change,
            reason=reason,
            heuristic_accepted=accepted,
        )


class HybridTransitionVerifier:
    """Combine an independently trained pairwise verifier with heuristic safeguards."""

    STATUS_NAMES = ["Rejected", "Progress", "Resolved"]

    def __init__(
        self,
        heuristic: TransitionVerifier,
        model: LearnedTransitionVerifier,
        device: torch.device,
        tool_names: list[str],
        threshold: float | list[float] | np.ndarray = 0.55,
        mode: str = "weighted",
        learned_weight: float = 0.55,
        learned_min_relative_target_gain: float = 0.01,
        image_size: int = 256,
        gamma_guard: GammaSafetyGuard | None = None,
        paper_lcb_z: float = 0.0,
    ) -> None:
        if mode not in {
            "weighted",
            "and",
            "learned",
            "learned_progress_guard",
            "paper",
        }:
            raise ValueError(
                "mode must be one of: weighted, and, learned, "
                "learned_progress_guard, paper"
            )
        if learned_min_relative_target_gain < 0.0:
            raise ValueError("learned_min_relative_target_gain must be non-negative")
        self.heuristic = heuristic
        self.model = model.to(device).eval()
        self.device = device
        self.tool_to_id = {name: index for index, name in enumerate(tool_names)}
        threshold_array = np.asarray(threshold, dtype=np.float32)
        if threshold_array.ndim == 0:
            threshold_array = np.full(len(tool_names), float(threshold_array), dtype=np.float32)
        if threshold_array.shape != (len(tool_names),):
            raise ValueError("threshold must be scalar or match tool_names")
        if np.any((threshold_array < 0.0) | (threshold_array > 1.0)):
            raise ValueError("threshold values must be in [0, 1]")
        self.threshold = threshold_array
        self.mode = mode
        self.learned_weight = float(np.clip(learned_weight, 0.0, 1.0))
        self.learned_min_relative_target_gain = float(learned_min_relative_target_gain)
        self.image_size = int(image_size)
        self.gamma_guard = gamma_guard
        self.paper_lcb_z = float(paper_lcb_z)
        if self.paper_lcb_z < 0.0:
            raise ValueError("paper_lcb_z must be non-negative")

    def threshold_for_tool(self, tool_name: str) -> float:
        tool_id = self.tool_to_id.get(tool_name)
        if tool_id is None:
            return 1.0
        return float(self.threshold[tool_id])

    def _passes_learned_progress_guard(
        self,
        base: VerificationResult,
        before_target: float,
    ) -> tuple[bool, float, float]:
        denominator = max(abs(float(before_target)), 1e-6)
        relative_gain = float(base.target_gain / denominator)
        required_gain = float(self.learned_min_relative_target_gain * denominator)
        observed_progress = bool(
            base.target_gain > 0.0 and base.target_gain >= required_gain
        )
        hard_safe = bool(
            base.max_side_effect <= self.heuristic.max_side_effect
            and base.outside_change <= self.heuristic.max_outside_change
        )
        return bool((base.accepted or observed_progress) and hard_safe), relative_gain, required_gain

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.image_size, self.image_size):
            return image
        return resize_and_center_crop_image(image, self.image_size)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.shape == (self.image_size, self.image_size):
            return mask
        return cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

    @torch.inference_mode()
    def _predict(
        self,
        before: np.ndarray,
        after: np.ndarray,
        action: CandidateAction,
        trajectory_context: np.ndarray | None = None,
    ) -> dict[str, float | str | None]:
        if action.tool_name not in self.tool_to_id:
            return {
                "probability": 0.5,
                "status": "Rejected",
                "paper_accept_probability": None,
                "paper_psnr_gain_db": None,
                "paper_ssim_gain": None,
                "paper_psnr_std_db": None,
                "paper_ssim_std": None,
            }
        before_tensor = image_to_tensor(self._resize_image(before)).unsqueeze(0).to(self.device)
        after_tensor = image_to_tensor(self._resize_image(after)).unsqueeze(0).to(self.device)
        mask = mask_to_tensor(self._resize_mask(action.region_mask)).unsqueeze(0).to(self.device)
        output = self.model(
            before=before_tensor,
            after=after_tensor,
            action_mask=mask,
            tool_id=torch.tensor([self.tool_to_id[action.tool_name]], device=self.device),
            target_index=torch.tensor([action.target_index], device=self.device),
            strength=torch.tensor([action.strength], dtype=torch.float32, device=self.device),
            mask_area=torch.tensor([float(action.region_mask.mean())], dtype=torch.float32, device=self.device),
            cost_prior=torch.tensor([action.cost_prior], dtype=torch.float32, device=self.device),
            trajectory_context=(
                torch.as_tensor(
                    trajectory_context,
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(1, -1)
                if trajectory_context is not None
                else None
            ),
        )
        probability = float(torch.sigmoid(output["accept_logit"])[0].cpu())
        status_index = int(output["status_logits"].argmax(dim=1)[0].cpu())
        result: dict[str, float | str | None] = {
            "probability": probability,
            "status": self.STATUS_NAMES[status_index],
            "paper_accept_probability": None,
            "paper_psnr_gain_db": None,
            "paper_ssim_gain": None,
            "paper_psnr_std_db": None,
            "paper_ssim_std": None,
        }
        if "paper_quality_mu" in output and "paper_accept_logit" in output:
            paper_mu = output["paper_quality_mu"][0]
            paper_std = torch.exp(0.5 * output["paper_quality_logvar"][0])
            result.update(
                {
                    "paper_accept_probability": float(
                        torch.sigmoid(output["paper_accept_logit"])[0].cpu()
                    ),
                    "paper_psnr_gain_db": float(paper_mu[0].cpu() * 10.0),
                    "paper_ssim_gain": float(paper_mu[1].cpu()),
                    "paper_psnr_std_db": float(paper_std[0].cpu() * 10.0),
                    "paper_ssim_std": float(paper_std[1].cpu()),
                }
            )
        return result

    def verify(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        before_presence: np.ndarray,
        before_severity: np.ndarray,
        after_presence: np.ndarray,
        after_severity: np.ndarray,
        action: CandidateAction,
        trajectory_context: np.ndarray | None = None,
    ) -> VerificationResult:
        base = self.heuristic.verify(
            before_image,
            after_image,
            before_presence,
            before_severity,
            after_presence,
            after_severity,
            action,
            trajectory_context,
        )
        learned = self._predict(
            before_image, after_image, action, trajectory_context
        )
        probability = float(learned["probability"])
        learned_status = str(learned["status"])
        learned_threshold = self.threshold_for_tool(action.tool_name)
        gamma_guard_accepted: bool | None = None
        gamma_guard_probability: float | None = None
        if self.gamma_guard is not None and action.tool_name == "dark__gamma_correction":
            gamma_guard_accepted, gamma_guard_probability = self.gamma_guard.accepts(
                before_image, after_image
            )

        gain_score = float(np.clip(base.target_gain / max(self.heuristic.min_target_gain, 1e-6), 0.0, 1.0))
        side_score = float(
            1.0 - np.clip(base.max_side_effect / max(self.heuristic.max_side_effect, 1e-6), 0.0, 1.0)
        )
        outside_score = float(
            1.0 - np.clip(base.outside_change / max(self.heuristic.max_outside_change, 1e-6), 0.0, 1.0)
        )
        heuristic_score = (gain_score + side_score + outside_score) / 3.0
        combined = self.learned_weight * probability + (1.0 - self.learned_weight) * heuristic_score

        catastrophic = (
            base.max_side_effect > 1.5 * self.heuristic.max_side_effect
            or base.outside_change > 1.5 * self.heuristic.max_outside_change
        )
        before_target = float(
            before_presence[action.target_index] * before_severity[action.target_index]
        )
        progress_guard_passed: bool | None = None
        observed_relative_target_gain: float | None = None
        required_target_gain: float | None = None
        paper_accept_probability = learned["paper_accept_probability"]
        paper_psnr_gain_db = learned["paper_psnr_gain_db"]
        paper_ssim_gain = learned["paper_ssim_gain"]
        paper_psnr_std_db = learned["paper_psnr_std_db"]
        paper_ssim_std = learned["paper_ssim_std"]
        paper_psnr_lcb_db: float | None = None
        paper_ssim_lcb: float | None = None
        paper_gate_passed: bool | None = None
        if (
            paper_accept_probability is not None
            and paper_psnr_gain_db is not None
            and paper_ssim_gain is not None
            and paper_psnr_std_db is not None
            and paper_ssim_std is not None
        ):
            paper_psnr_lcb_db = float(
                paper_psnr_gain_db - self.paper_lcb_z * paper_psnr_std_db
            )
            paper_ssim_lcb = float(
                paper_ssim_gain - self.paper_lcb_z * paper_ssim_std
            )
            paper_gate_passed = bool(
                paper_accept_probability >= learned_threshold
                and paper_psnr_lcb_db > 0.0
                and paper_ssim_lcb > -0.005
            )
        if gamma_guard_accepted is not None:
            accepted = bool(gamma_guard_accepted and not catastrophic)
        elif self.mode == "and":
            accepted = bool(base.accepted and probability >= learned_threshold)
        elif self.mode == "learned_progress_guard":
            (
                progress_guard_passed,
                observed_relative_target_gain,
                required_target_gain,
            ) = self._passes_learned_progress_guard(base, before_target)
            accepted = bool(
                probability >= learned_threshold
                and progress_guard_passed
                and not catastrophic
            )
        elif self.mode == "learned":
            accepted = bool(probability >= learned_threshold and not catastrophic)
        elif self.mode == "paper":
            accepted = bool(paper_gate_passed and not catastrophic)
        else:
            accepted = bool(combined >= learned_threshold and not catastrophic)

        if not accepted:
            status = "Rejected"
            if gamma_guard_accepted is not None:
                reason = "gamma_guard_reject"
            elif self.mode == "learned_progress_guard" and not progress_guard_passed:
                reason = "learned_progress_guard_reject"
            elif self.mode == "paper":
                reason = "learned_paper_gate_reject"
            else:
                reason = f"hybrid_reject:{base.reason}"
        else:
            status = learned_status if learned_status in {"Progress", "Resolved"} else base.status
            reason = "gamma_guard_verified" if gamma_guard_accepted is not None else "hybrid_verified"
        return VerificationResult(
            accepted=accepted,
            status=status,
            target_gain=base.target_gain,
            max_side_effect=base.max_side_effect,
            outside_change=base.outside_change,
            reason=reason,
            heuristic_accepted=base.accepted,
            learned_probability=probability,
            learned_threshold=learned_threshold,
            learned_status=learned_status,
            combined_score=combined,
            gamma_guard_accepted=gamma_guard_accepted,
            gamma_guard_probability=gamma_guard_probability,
            gamma_guard_threshold=(
                self.gamma_guard.threshold if gamma_guard_accepted is not None else None
            ),
            learned_progress_guard_passed=progress_guard_passed,
            learned_observed_relative_target_gain=observed_relative_target_gain,
            learned_required_target_gain=required_target_gain,
            learned_paper_accept_probability=(
                float(paper_accept_probability)
                if paper_accept_probability is not None
                else None
            ),
            learned_paper_psnr_gain_db=(
                float(paper_psnr_gain_db)
                if paper_psnr_gain_db is not None
                else None
            ),
            learned_paper_ssim_gain=(
                float(paper_ssim_gain) if paper_ssim_gain is not None else None
            ),
            learned_paper_psnr_std_db=(
                float(paper_psnr_std_db)
                if paper_psnr_std_db is not None
                else None
            ),
            learned_paper_ssim_std=(
                float(paper_ssim_std) if paper_ssim_std is not None else None
            ),
            learned_paper_psnr_lcb_db=paper_psnr_lcb_db,
            learned_paper_ssim_lcb=paper_ssim_lcb,
            learned_paper_gate_passed=paper_gate_passed,
        )
