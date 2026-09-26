from __future__ import annotations

import math
from typing import Any

import numpy as np


def expected_calibration_error(
    probability: np.ndarray, target: np.ndarray, bins: int = 10
) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probability >= lower) & (
            probability < upper if index < bins - 1 else probability <= upper
        )
        if selected.any():
            value += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return float(value)


def binary_roc_auc(probability: np.ndarray, target: np.ndarray) -> float | None:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    positive = int(target.sum())
    negative = int((~target).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(probability), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probability[order[end]] == probability[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[target].sum())
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    logs = [
        math.lgamma(n + 1)
        - math.lgamma(index + 1)
        - math.lgamma(n - index + 1)
        + index * math.log(probability)
        + (n - index) * math.log1p(-probability)
        for index in range(k + 1)
    ]
    maximum = max(logs)
    return float(math.exp(maximum) * sum(math.exp(value - maximum) for value in logs))


def clopper_pearson_upper(harmful: int, selected: int, delta: float) -> float:
    """One-sided exact binomial upper confidence bound.

    The returned value is the smallest risk ``p`` whose lower-tail probability
    of observing at most ``harmful`` events is ``delta``.
    """

    if selected <= 0:
        return 1.0
    if harmful >= selected:
        return 1.0
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    lower = harmful / selected
    upper = 1.0
    for _ in range(70):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(harmful, selected, midpoint) > delta:
            lower = midpoint
        else:
            upper = midpoint
    return float(upper)


def select_risk_threshold(
    probability: np.ndarray,
    harmful: np.ndarray,
    *,
    alpha: float,
    delta: float,
    min_selected: int = 1,
    max_threshold: float | None = None,
) -> dict[str, Any]:
    """Select maximum-coverage ``P(harm)`` threshold under an exact risk UCB."""

    probability = np.asarray(probability, dtype=np.float64)
    harmful = np.asarray(harmful, dtype=bool)
    if probability.shape != harmful.shape:
        raise ValueError("probability and harmful must have matching shapes")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    candidates: list[dict[str, Any]] = []
    thresholds = np.unique(np.clip(probability, 0.0, 1.0))
    if max_threshold is not None:
        thresholds = thresholds[thresholds <= float(max_threshold)]
    for threshold in thresholds:
        selected_mask = probability <= threshold
        selected = int(selected_mask.sum())
        if selected < min_selected:
            continue
        harmful_selected = int(harmful[selected_mask].sum())
        empirical_risk = harmful_selected / selected
        risk_ucb = clopper_pearson_upper(harmful_selected, selected, delta)
        candidates.append(
            {
                "threshold": float(threshold),
                "selected": selected,
                "coverage": float(selected / max(len(probability), 1)),
                "harmful_selected": harmful_selected,
                "empirical_risk": float(empirical_risk),
                "risk_ucb": float(risk_ucb),
                "valid": bool(risk_ucb <= alpha),
            }
        )
    valid = [row for row in candidates if row["valid"]]
    if not valid:
        return {
            "enabled": False,
            "threshold": 0.0,
            "selected": 0,
            "coverage": 0.0,
            "harmful_selected": 0,
            "empirical_risk": 0.0,
            "risk_ucb": 1.0,
            "num_calibration": int(len(probability)),
        }
    selected = max(valid, key=lambda row: (row["coverage"], -row["risk_ucb"]))
    return {"enabled": True, "num_calibration": int(len(probability)), **selected}


def risk_coverage_curve(
    probability: np.ndarray, harmful: np.ndarray, points: int = 21
) -> list[dict[str, float | int]]:
    probability = np.asarray(probability, dtype=np.float64)
    harmful = np.asarray(harmful, dtype=bool)
    rows: list[dict[str, float | int]] = []
    for threshold in np.linspace(0.0, 1.0, points):
        selected = probability <= threshold
        count = int(selected.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(count / max(len(probability), 1)),
                "selected": count,
                "harmful_selected": int(harmful[selected].sum()),
                "selective_risk": float(harmful[selected].mean()) if count else 0.0,
            }
        )
    return rows
