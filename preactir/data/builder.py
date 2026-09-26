from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2
from tqdm import tqdm

from preactir.data.degradations import (
    apply_degradation,
    compute_degradation_scores,
    random_spatial_mask,
)
from preactir.tools.registry import ToolRegistry
from preactir.utils.image import align_to_reference, load_image, save_image, save_mask
from preactir.utils.io import ensure_dir, list_images, write_jsonl
from preactir.utils.metrics import outside_damage, quality_vector


@dataclass
class BuilderOptions:
    image_size: int = 256
    rollout_native_resolution: bool = False
    max_degradations: int = 3
    states_per_image: int = 2
    max_actions_per_state: int = 12
    spatial_probability: float = 0.45
    distractor_action_probability: float = 0.25
    acceptance_target_gain: float = 0.01
    acceptance_max_damage: float = 0.02
    strengths: tuple[float, ...] = (0.35, 0.65, 1.0)
    seed: int = 42


class InterventionDatasetBuilder:
    """Construct synthetic state and intervention-transition manifests."""

    def __init__(
        self,
        clean_root: str | Path,
        output_root: str | Path,
        degradation_names: list[str],
        registry: ToolRegistry,
        options: BuilderOptions,
    ) -> None:
        self.clean_root = Path(clean_root)
        self.output_root = Path(output_root)
        self.degradation_names = list(degradation_names)
        self.registry = registry
        self.options = options
        self.target_map = registry.target_map()
        self.name_to_index = {name: index for index, name in enumerate(self.degradation_names)}

        unsupported = sorted(set(self.target_map) - set(self.degradation_names))
        if unsupported:
            raise ValueError(f"Tool targets absent from degradation list: {unsupported}")

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.output_root).as_posix())
        except ValueError:
            return str(path.resolve())

    def _sample_id(self, split: str, source: Path, state_index: int) -> str:
        text = f"{split}:{source.resolve()}:{state_index}:{self.options.seed}"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
        return f"{split}_{digest}_{state_index:02d}"

    def build(self, splits: list[str] | None = None, limit_per_split: int | None = None) -> None:
        splits = splits or ["train", "val", "test"]
        ensure_dir(self.output_root)
        metadata = {
            "degradation_names": self.degradation_names,
            "tool_names": self.registry.names(),
            "quality_names": ["psnr_scaled", "ssim", "l1_similarity", "edge_similarity"],
            "options": vars(self.options),
        }
        with (self.output_root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        for split in splits:
            source_dir = self.clean_root / split
            if not source_dir.exists():
                if split == "train" and self.clean_root.exists():
                    source_dir = self.clean_root
                else:
                    continue
            images = list_images(source_dir)
            if limit_per_split is not None:
                images = images[:limit_per_split]
            if not images:
                continue
            state_rows: list[dict[str, Any]] = []
            transition_rows: list[dict[str, Any]] = []
            for image_index, source_path in enumerate(tqdm(images, desc=f"Building {split}")):
                clean = load_image(source_path, self.options.image_size)
                clean_key = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:16]
                clean_path = self.output_root / "clean" / split / f"{clean_key}.png"
                if not clean_path.exists():
                    save_image(clean_path, clean)

                for state_index in range(self.options.states_per_image):
                    seed = self.options.seed + image_index * 1009 + state_index * 9173 + len(split) * 101
                    rng = np.random.default_rng(seed)
                    state_row, transitions = self._build_state(
                        split=split,
                        source_path=source_path,
                        clean=clean,
                        clean_path=clean_path,
                        state_index=state_index,
                        rng=rng,
                    )
                    state_rows.append(state_row)
                    transition_rows.extend(transitions)

            write_jsonl(self.output_root / f"states_{split}.jsonl", state_rows)
            write_jsonl(self.output_root / f"transitions_{split}.jsonl", transition_rows)

    def _build_state(
        self,
        split: str,
        source_path: Path,
        clean: np.ndarray,
        clean_path: Path,
        state_index: int,
        rng: np.random.Generator,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        state_id = self._sample_id(split, source_path, state_index)
        height, width = clean.shape[:2]
        max_deg = min(self.options.max_degradations, len(self.degradation_names))
        count = int(rng.integers(1, max_deg + 1))
        active_names = list(rng.choice(self.degradation_names, size=count, replace=False))
        rng.shuffle(active_names)

        current = clean.copy()
        severity_map = {name: 0.0 for name in self.degradation_names}
        masks_by_name: dict[str, np.ndarray] = {
            name: np.ones((height, width), dtype=np.float32) for name in self.degradation_names
        }
        mask_paths: dict[str, str] = {}
        degradation_order: list[str] = []

        for name in active_names:
            severity = float(rng.uniform(0.25, 1.0))
            mask = random_spatial_mask(
                height,
                width,
                rng,
                spatial_probability=self.options.spatial_probability,
            )
            current = apply_degradation(current, name, severity, mask, rng)
            severity_map[name] = severity
            masks_by_name[name] = mask
            degradation_order.append(name)
            mask_path = self.output_root / "masks" / split / state_id / f"{name}.png"
            save_mask(mask_path, mask)
            mask_paths[name] = self._relative(mask_path)

        state_path = self.output_root / "states" / split / f"{state_id}.png"
        save_image(state_path, current)
        presence = [1.0 if name in active_names else 0.0 for name in self.degradation_names]
        severity = [severity_map[name] for name in self.degradation_names]
        proxy_scores = compute_degradation_scores(
            current,
            clean,
            masks_by_name,
            self.degradation_names,
        )
        q_before = quality_vector(current, clean)

        state_row: dict[str, Any] = {
            "state_id": state_id,
            "split": split,
            "source_path": str(source_path),
            "clean_path": self._relative(clean_path),
            "image_path": self._relative(state_path),
            "degradation_order": degradation_order,
            "presence": presence,
            "severity": severity,
            "proxy_scores": proxy_scores.tolist(),
            "mask_paths": mask_paths,
            "quality": q_before.tolist(),
        }

        transitions = self._build_transitions(
            split=split,
            state_id=state_id,
            current=current,
            clean=clean,
            current_path=state_path,
            clean_path=clean_path,
            active_names=active_names,
            masks_by_name=masks_by_name,
            presence=presence,
            severity=severity,
            proxy_scores=proxy_scores,
            quality_before=q_before,
            rng=rng,
        )
        return state_row, transitions

    def _build_transitions(
        self,
        *,
        split: str,
        state_id: str,
        current: np.ndarray,
        clean: np.ndarray,
        current_path: Path,
        clean_path: Path,
        active_names: list[str],
        masks_by_name: dict[str, np.ndarray],
        presence: list[float],
        severity: list[float],
        proxy_scores: np.ndarray,
        quality_before: np.ndarray,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        """Execute candidate tools and derive clean-reference transition labels."""

        actions = self._candidate_actions(active_names, masks_by_name, rng)
        transitions: list[dict[str, Any]] = []
        for action_index, action in enumerate(actions):
            tool = self.registry.get(action["tool_name"])
            target_name = str(action["target_name"])
            target_index = self.name_to_index[target_name]
            region_mask = action["region_mask"]
            result = tool.run(current, strength=float(action["strength"]), region_mask=region_mask)
            candidate = result.image
            candidate_path = (
                self.output_root
                / "transitions"
                / split
                / state_id
                / f"a{action_index:03d}_{tool.name}_{action['region_kind']}_{float(action['strength']):.2f}.png"
            )
            save_image(candidate_path, candidate)

            metric_current = align_to_reference(current, clean)
            metric_candidate = align_to_reference(candidate, clean)
            metric_masks = {
                name: (
                    mask
                    if mask.shape == clean.shape[:2]
                    else cv2.resize(
                        mask.astype(np.float32),
                        (clean.shape[1], clean.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                for name, mask in masks_by_name.items()
            }
            after_scores = compute_degradation_scores(
                metric_candidate,
                clean,
                metric_masks,
                self.degradation_names,
            )
            delta_degradation = proxy_scores - after_scores
            q_after = quality_vector(metric_candidate, clean)
            delta_quality = q_after - quality_before
            target_mask = metric_masks.get(
                target_name, np.ones(clean.shape[:2], dtype=np.float32)
            )
            damage = outside_damage(metric_current, metric_candidate, clean, target_mask)
            target_gain = float(delta_degradation[target_index])
            accepted = bool(
                target_gain >= self.options.acceptance_target_gain
                and damage <= self.options.acceptance_max_damage
                and float(np.mean(delta_quality)) >= -0.025
            )

            region_path: str | None = None
            if action["region_kind"] == "local":
                region_file = self.output_root / "action_masks" / split / state_id / f"a{action_index:03d}.png"
                save_mask(region_file, region_mask)
                region_path = self._relative(region_file)

            transitions.append(
                {
                    "transition_id": f"{state_id}_a{action_index:03d}",
                    "state_id": state_id,
                    "split": split,
                    "clean_path": self._relative(clean_path),
                    "current_path": self._relative(current_path),
                    "next_path": self._relative(candidate_path),
                    "presence": presence,
                    "severity": severity,
                    "proxy_before": proxy_scores.tolist(),
                    "proxy_after": after_scores.tolist(),
                    "delta_degradation": delta_degradation.tolist(),
                    "quality_before": quality_before.tolist(),
                    "quality_after": q_after.tolist(),
                    "delta_quality": delta_quality.tolist(),
                    "damage": float(damage),
                    "accepted": accepted,
                    "action": {
                        "tool_name": tool.name,
                        "target_name": target_name,
                        "target_index": target_index,
                        "strength": float(action["strength"]),
                        "region_kind": action["region_kind"],
                        "region_mask_path": region_path,
                        "mask_area": float(region_mask.mean()),
                        "elapsed_ms": float(result.elapsed_ms),
                        "cost_prior": float(tool.cost_prior),
                    },
                }
            )
        return transitions

    def _candidate_actions(
        self,
        active_names: list[str],
        masks_by_name: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for target_name in active_names:
            for tool_name in self.target_map.get(target_name, []):
                for strength in self.options.strengths:
                    candidates.append(
                        {
                            "tool_name": tool_name,
                            "target_name": target_name,
                            "strength": float(strength),
                            "region_kind": "global",
                            "region_mask": np.ones_like(masks_by_name[target_name]),
                        }
                    )
                    local_mask = masks_by_name[target_name]
                    if float(local_mask.mean()) < 0.98:
                        candidates.append(
                            {
                                "tool_name": tool_name,
                                "target_name": target_name,
                                "strength": float(strength),
                                "region_kind": "local",
                                "region_mask": local_mask,
                            }
                        )

        inactive_tools = [
            (tool_name, tool.targets[0])
            for tool_name, tool in self.registry.items()
            if tool.targets and tool.targets[0] not in active_names
        ]
        if inactive_tools and rng.random() < self.options.distractor_action_probability:
            rng.shuffle(inactive_tools)
            for tool_name, target_name in inactive_tools[:2]:
                candidates.append(
                    {
                        "tool_name": tool_name,
                        "target_name": target_name,
                        "strength": float(rng.choice(self.options.strengths)),
                        "region_kind": "global",
                        "region_mask": np.ones_like(next(iter(masks_by_name.values()))),
                    }
                )

        rng.shuffle(candidates)
        return candidates[: self.options.max_actions_per_state]
