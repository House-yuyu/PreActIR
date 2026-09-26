from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar))


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(2, 3))
    denominator = probability.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice


def belief_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    presence_target = batch["presence"]
    severity_target = batch["severity"]
    mask_target = batch["masks"]

    pos_weight = torch.full(
        (presence_target.shape[1],),
        float(weights.get("presence_pos_weight", 1.0)),
        dtype=presence_target.dtype,
        device=presence_target.device,
    )
    presence_values = F.binary_cross_entropy_with_logits(
        output["presence_logits"],
        presence_target,
        pos_weight=pos_weight,
        reduction="none",
    )
    focal_gamma = float(weights.get("presence_focal_gamma", 0.0))
    if focal_gamma > 0.0:
        probability = torch.sigmoid(output["presence_logits"])
        target_probability = probability * presence_target + (1.0 - probability) * (
            1.0 - presence_target
        )
        presence_values = presence_values * (1.0 - target_probability).pow(focal_gamma)
    presence = presence_values.mean()
    severity_nll = gaussian_nll(output["severity_mu"], output["severity_logvar"], severity_target)
    severity = (severity_nll * presence_target).sum() / presence_target.sum().clamp_min(1.0)

    pixel_weights = 0.25 + 0.75 * presence_target[:, :, None, None]
    mask_bce = F.binary_cross_entropy_with_logits(output["mask_logits"], mask_target, reduction="none")
    mask_bce = (mask_bce * pixel_weights).mean()
    mask_dice_values = dice_loss(output["mask_logits"], mask_target)
    mask_dice = (mask_dice_values * presence_target).sum() / presence_target.sum().clamp_min(1.0)

    terms = {
        "presence": presence,
        "severity": severity,
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
    }
    total = sum(float(weights.get(name, 1.0)) * value for name, value in terms.items())
    return total, terms


def _pairwise_ranking_loss(
    predicted_utility: torch.Tensor,
    actual_utility: torch.Tensor,
    state_ids: Sequence[str],
    min_gap: float = 1e-3,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    batch_size = predicted_utility.shape[0]
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if state_ids[i] != state_ids[j]:
                continue
            gap = actual_utility[i] - actual_utility[j]
            if torch.abs(gap).item() < min_gap:
                continue
            sign = torch.sign(gap).detach()
            losses.append(F.softplus(-sign * (predicted_utility[i] - predicted_utility[j])))
    if not losses:
        return predicted_utility.sum() * 0.0
    return torch.stack(losses).mean()


def world_model_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | Sequence[str]],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    delta_degradation = batch["delta_degradation"]
    delta_quality = batch["delta_quality"]
    damage = batch["damage"]
    accepted = batch["accepted"]
    target_index = batch["target_index"]

    assert isinstance(delta_degradation, torch.Tensor)
    assert isinstance(delta_quality, torch.Tensor)
    assert isinstance(damage, torch.Tensor)
    assert isinstance(accepted, torch.Tensor)
    assert isinstance(target_index, torch.Tensor)

    degradation = gaussian_nll(
        output["delta_degradation_mu"], output["delta_degradation_logvar"], delta_degradation
    ).mean()
    quality = gaussian_nll(output["delta_quality_mu"], output["delta_quality_logvar"], delta_quality).mean()
    damage_loss = gaussian_nll(output["damage_mu"], output["damage_logvar"], damage).mean()
    accept = F.binary_cross_entropy_with_logits(output["accept_logit"], accepted)
    harmful = batch.get("harmful")
    if "harm_logit" in output and isinstance(harmful, torch.Tensor):
        harm_pos_weight = torch.as_tensor(
            float(weights.get("harm_pos_weight", 1.0)),
            dtype=output["harm_logit"].dtype,
            device=output["harm_logit"].device,
        )
        harm = F.binary_cross_entropy_with_logits(
            output["harm_logit"], harmful, pos_weight=harm_pos_weight
        )
    else:
        harm = output["accept_logit"].sum() * 0.0
    severe_harmful = batch.get("severe_harm")
    if "severe_harm_logit" in output and isinstance(severe_harmful, torch.Tensor):
        severe_pos_weight = torch.as_tensor(
            float(weights.get("severe_harm_pos_weight", 1.0)),
            dtype=output["severe_harm_logit"].dtype,
            device=output["severe_harm_logit"].device,
        )
        severe_harm = F.binary_cross_entropy_with_logits(
            output["severe_harm_logit"],
            severe_harmful,
            pos_weight=severe_pos_weight,
        )
    else:
        severe_harm = output["accept_logit"].sum() * 0.0
    paper_quality_target = batch.get("paper_quality_gain")
    if "paper_quality_mu" in output and isinstance(paper_quality_target, torch.Tensor):
        paper_quality = gaussian_nll(
            output["paper_quality_mu"],
            output["paper_quality_logvar"],
            paper_quality_target,
        ).mean()
        paper_predicted_utility = output["paper_quality_mu"].sum(dim=1)
        paper_actual_utility = paper_quality_target.sum(dim=1)
        ranking_group_ids = batch.get("ranking_group_id", batch["state_id"])
        assert isinstance(ranking_group_ids, Sequence)
        paper_ranking = _pairwise_ranking_loss(
            paper_predicted_utility,
            paper_actual_utility,
            ranking_group_ids,
            min_gap=1e-4,
        )
    else:
        paper_quality = output["accept_logit"].sum() * 0.0
        paper_ranking = output["accept_logit"].sum() * 0.0

    target_gain_pred = output["delta_degradation_mu"].gather(1, target_index[:, None]).squeeze(1)
    target_gain_true = delta_degradation.gather(1, target_index[:, None]).squeeze(1)
    predicted_utility = target_gain_pred + 0.5 * output["delta_quality_mu"].mean(dim=1) - 2.0 * output["damage_mu"]
    actual_utility = target_gain_true + 0.5 * delta_quality.mean(dim=1) - 2.0 * damage
    state_ids = batch["state_id"]
    assert isinstance(state_ids, Sequence)
    ranking = _pairwise_ranking_loss(predicted_utility, actual_utility, state_ids)

    terms = {
        "degradation": degradation,
        "quality": quality,
        "damage": damage_loss,
        "accept": accept,
        "harm": harm,
        "severe_harm": severe_harm,
        "paper_quality": paper_quality,
        "paper_ranking": paper_ranking,
        "ranking": ranking,
    }
    total = sum(float(weights.get(name, 1.0)) * value for name, value in terms.items())
    return total, terms


def verifier_model_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | Sequence[str]],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    accepted = batch["accepted"]
    status = batch["status"]
    target_gain = batch["target_gain"]
    max_side_effect = batch["max_side_effect"]
    outside_change = batch["outside_change"]
    assert isinstance(accepted, torch.Tensor)
    assert isinstance(status, torch.Tensor)
    assert isinstance(target_gain, torch.Tensor)
    assert isinstance(max_side_effect, torch.Tensor)
    assert isinstance(outside_change, torch.Tensor)

    accept = F.binary_cross_entropy_with_logits(output["accept_logit"], accepted)
    status_loss = F.cross_entropy(output["status_logits"], status)
    target_gain_loss = F.smooth_l1_loss(output["target_gain"], target_gain)
    side_effect_loss = F.smooth_l1_loss(output["side_effect"], max_side_effect)
    outside_loss = F.smooth_l1_loss(output["outside_change"], outside_change)
    paper_quality_target = batch.get("paper_quality_gain")
    paper_accept_target = batch.get("paper_accept")
    if (
        "paper_quality_mu" in output
        and "paper_accept_logit" in output
        and isinstance(paper_quality_target, torch.Tensor)
        and isinstance(paper_accept_target, torch.Tensor)
    ):
        paper_quality = gaussian_nll(
            output["paper_quality_mu"],
            output["paper_quality_logvar"],
            paper_quality_target,
        ).mean()
        paper_accept_pos_weight = torch.as_tensor(
            float(weights.get("paper_accept_pos_weight", 1.0)),
            dtype=output["paper_accept_logit"].dtype,
            device=output["paper_accept_logit"].device,
        )
        paper_accept = F.binary_cross_entropy_with_logits(
            output["paper_accept_logit"],
            paper_accept_target,
            pos_weight=paper_accept_pos_weight,
        )
    else:
        paper_quality = output["accept_logit"].sum() * 0.0
        paper_accept = output["accept_logit"].sum() * 0.0
    terms = {
        "accept": accept,
        "status": status_loss,
        "target_gain": target_gain_loss,
        "side_effect": side_effect_loss,
        "outside_change": outside_loss,
        "paper_quality": paper_quality,
        "paper_accept": paper_accept,
    }
    total = sum(float(weights.get(name, 1.0)) * value for name, value in terms.items())
    return total, terms
