from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from preactir.utils.image import image_to_tensor, load_image, load_mask, mask_to_tensor
from preactir.utils.io import read_jsonl

STATUS_NAMES = ["Rejected", "Progress", "Resolved"]
_ONPOLICY_STEP_PATTERN = re.compile(r"_onpolicy_s(\d+)$")


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _flip(array: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    if horizontal:
        array = np.flip(array, axis=1)
    if vertical:
        array = np.flip(array, axis=0)
    return np.ascontiguousarray(array)


class GroupedBatchSampler(Sampler[list[int]]):
    """Pack same-state candidates together so pairwise ranking receives pairs."""

    def __init__(
        self,
        group_ids: list[str],
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            grouped[str(group_id)].append(index)
        self.groups = list(grouped.values())
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.groups)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=generator).tolist()
        batch: list[int] = []
        for group_index in order:
            indices = list(self.groups[group_index])
            if self.shuffle and len(indices) > 1:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]
            # A ranking group should never be split merely to fill the tail of
            # the preceding batch.  Groups larger than a whole batch are the
            # only unavoidable exception.
            if batch and len(indices) <= self.batch_size and (
                len(batch) + len(indices) > self.batch_size
            ):
                yield batch
                batch = []
            while indices:
                capacity = self.batch_size - len(batch)
                if capacity == 0:
                    yield batch
                    batch = []
                    capacity = self.batch_size
                take = min(capacity, len(indices))
                batch.extend(indices[:take])
                indices = indices[take:]
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = list(range(len(self.groups)))
        if self.shuffle:
            order = torch.randperm(len(order), generator=generator).tolist()
        batch_count = 0
        filled = 0
        for group_index in order:
            remaining = len(self.groups[group_index])
            if filled and remaining <= self.batch_size and (
                filled + remaining > self.batch_size
            ):
                batch_count += 1
                filled = 0
            while remaining:
                take = min(self.batch_size - filled, remaining)
                filled += take
                remaining -= take
                if filled == self.batch_size:
                    batch_count += 1
                    filled = 0
        return batch_count + int(filled > 0)


class BeliefDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        degradation_names: list[str],
        image_size: int | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.degradation_names = list(degradation_names)
        self.image_size = image_size
        self.augment = augment
        self.rows = read_jsonl(self.root / f"states_{split}.jsonl")
        if not self.rows:
            raise RuntimeError(f"No state rows found for split={split}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = load_image(_resolve(self.root, row["image_path"]), self.image_size)
        height, width = image.shape[:2]
        masks: list[np.ndarray] = []
        for name, present in zip(self.degradation_names, row["presence"]):
            path = row.get("mask_paths", {}).get(name)
            if float(present) > 0 and path is not None:
                mask = load_mask(_resolve(self.root, path), self.image_size)
            elif float(present) > 0:
                # Official MiO100 degradations are global and do not ship masks.
                mask = np.ones((height, width), dtype=np.float32)
            else:
                mask = np.zeros((height, width), dtype=np.float32)
            masks.append(mask)

        if self.augment:
            horizontal = random.random() < 0.5
            vertical = random.random() < 0.1
            image = _flip(image, horizontal, vertical)
            masks = [_flip(mask, horizontal, vertical) for mask in masks]

        mask_array = np.stack(masks, axis=0).astype(np.float32)
        return {
            "image": image_to_tensor(image),
            "presence": torch.tensor(row["presence"], dtype=torch.float32),
            "severity": torch.tensor(row["severity"], dtype=torch.float32),
            "masks": torch.from_numpy(mask_array),
            "proxy_scores": torch.tensor(row.get("proxy_scores", row["severity"]), dtype=torch.float32),
            "state_id": row["state_id"],
        }


class TransitionDataset(Dataset[dict[str, Any]]):
    """Action-level transitions for world-model and verifier training."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        degradation_names: list[str],
        tool_names: list[str],
        image_size: int | None = None,
        augment: bool = False,
        exclude_tools: set[str] | None = None,
        include_tools: set[str] | None = None,
        manifest_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.root
        self.split = split
        self.degradation_names = list(degradation_names)
        self.tool_names = list(tool_names)
        self.tool_to_index = {name: index for index, name in enumerate(self.tool_names)}
        self.image_size = image_size
        self.augment = augment
        rows = read_jsonl(self.manifest_root / f"transitions_{split}.jsonl")
        exclude_tools = exclude_tools or set()
        self.rows = [
            row
            for row in rows
            if row["action"]["tool_name"] not in exclude_tools
            and (include_tools is None or row["action"]["tool_name"] in include_tools)
        ]
        if not self.rows:
            raise RuntimeError(f"No transition rows found for split={split} after filtering")
        self.ranking_group_ids = [
            f"{row.get('parent_state_id', row['state_id'])}|{row['current_path']}"
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = load_image(_resolve(self.root, row["current_path"]), self.image_size)
        next_image = load_image(_resolve(self.root, row["next_path"]), self.image_size)
        height, width = image.shape[:2]
        mask_path = row["action"].get("region_mask_path")
        if mask_path:
            action_mask = load_mask(_resolve(self.root, mask_path), self.image_size)
        else:
            action_mask = np.ones((height, width), dtype=np.float32)

        if self.augment:
            horizontal = random.random() < 0.5
            vertical = random.random() < 0.1
            image = _flip(image, horizontal, vertical)
            next_image = _flip(next_image, horizontal, vertical)
            action_mask = _flip(action_mask, horizontal, vertical)

        tool_name = row["action"]["tool_name"]
        if tool_name not in self.tool_to_index:
            raise KeyError(f"Tool '{tool_name}' is not in configured tool list")
        target_index = int(row["action"]["target_index"])
        belief = np.concatenate(
            [np.asarray(row["presence"], dtype=np.float32), np.asarray(row["severity"], dtype=np.float32)],
            axis=0,
        )
        delta_degradation = np.asarray(row["delta_degradation"], dtype=np.float32)
        target_gain = float(delta_degradation[target_index])
        non_target = np.delete(delta_degradation, target_index)
        max_side_effect = float(np.maximum(-non_target, 0.0).max(initial=0.0))
        accepted = bool(row["accepted"])
        delta_quality = np.asarray(row["delta_quality"], dtype=np.float32)
        risk_label = row.get("risk_label_v5", {})
        psnr_gain_db = float(
            risk_label.get("psnr_gain_db", float(delta_quality[0] * 50.0))
        )
        ssim_gain = float(
            risk_label.get(
                "ssim_gain", float(delta_quality[1]) if delta_quality.size > 1 else 0.0
            )
        )
        paper_quality_gain = np.asarray(
            [psnr_gain_db / 10.0, ssim_gain], dtype=np.float32
        )
        paper_y_before = risk_label.get("paper_y_before", {})
        paper_quality_before = np.asarray(
            [
                float(paper_y_before.get("psnr", float("nan"))),
                float(paper_y_before.get("ssim", float("nan"))),
            ],
            dtype=np.float32,
        )
        on_policy = bool(row.get("on_policy", False))
        stage = max(0, int(row.get("stage", 0) or 0))
        step_index_value = row.get("step_index")
        if step_index_value is None:
            match = _ONPOLICY_STEP_PATTERN.search(str(row["state_id"]))
            step_index = int(match.group(1)) if match is not None else 0
        else:
            step_index = max(0, int(step_index_value))
        prior_rejections = max(0, step_index - stage)
        trajectory_context = np.asarray(
            [
                min(stage / 3.0, 1.0),
                min(step_index / 6.0, 1.0),
                min(prior_rejections / 6.0, 1.0),
                float(on_policy),
            ],
            dtype=np.float32,
        )
        if "harmful" in row:
            harmful = bool(row["harmful"])
        elif "harmful" in risk_label:
            harmful = bool(risk_label["harmful"])
        else:
            # Conservative backward-compatible fallback used by synthetic smoke
            # data and legacy manifests.  Formal v5 runs use explicit labels.
            mean_quality_gain = float(delta_quality.mean())
            harmful = bool(
                psnr_gain_db < -0.05
                or ssim_gain < -0.005
                or mean_quality_gain < 0.0
            )
        if "severe_harm" in row:
            severe_harm = bool(row["severe_harm"])
        else:
            severe_harm = bool(psnr_gain_db <= -1.0 or ssim_gain <= -0.05)
        proxy_after = np.asarray(row.get("proxy_after", []), dtype=np.float32)
        resolved_threshold = float(row.get("resolved_threshold", 0.18))
        if not accepted:
            status = 0
        elif proxy_after.size > target_index and float(proxy_after[target_index]) <= resolved_threshold:
            status = 2
        else:
            status = 1

        return {
            "image": image_to_tensor(image),
            "next_image": image_to_tensor(next_image),
            "belief": torch.from_numpy(belief),
            "action_mask": mask_to_tensor(action_mask),
            "tool_id": torch.tensor(self.tool_to_index[tool_name], dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "strength": torch.tensor(float(row["action"]["strength"]), dtype=torch.float32),
            "mask_area": torch.tensor(float(row["action"].get("mask_area", action_mask.mean())), dtype=torch.float32),
            "cost_prior": torch.tensor(float(row["action"].get("cost_prior", 1.0)), dtype=torch.float32),
            "delta_degradation": torch.from_numpy(delta_degradation),
            "delta_quality": torch.from_numpy(delta_quality),
            "damage": torch.tensor(float(row["damage"]), dtype=torch.float32),
            "accepted": torch.tensor(float(accepted), dtype=torch.float32),
            "harmful": torch.tensor(float(harmful), dtype=torch.float32),
            "severe_harm": torch.tensor(float(severe_harm), dtype=torch.float32),
            "paper_quality_gain": torch.from_numpy(paper_quality_gain),
            "paper_quality_before": torch.from_numpy(paper_quality_before),
            "paper_reference_utility": torch.tensor(
                float(paper_quality_gain.sum()), dtype=torch.float32
            ),
            "paper_accept": torch.tensor(
                float(psnr_gain_db > 0.0 and ssim_gain > -0.005),
                dtype=torch.float32,
            ),
            "trajectory_context": torch.from_numpy(trajectory_context),
            "reference_utility": torch.tensor(
                psnr_gain_db + 10.0 * ssim_gain, dtype=torch.float32
            ),
            "status": torch.tensor(status, dtype=torch.long),
            "target_gain": torch.tensor(target_gain, dtype=torch.float32),
            "max_side_effect": torch.tensor(max_side_effect, dtype=torch.float32),
            "outside_change": torch.tensor(float(row["damage"]), dtype=torch.float32),
            "state_id": row["state_id"],
            "ranking_group_id": self.ranking_group_ids[index],
            "transition_id": row["transition_id"],
            "tool_name": tool_name,
        }
