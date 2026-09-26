from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from preactir.data.builder import BuilderOptions, InterventionDatasetBuilder
from preactir.data.degradations import compute_degradation_scores
from preactir.tools.registry import ToolRegistry
from preactir.utils.image import align_to_reference, load_image
from preactir.utils.io import ensure_dir, list_images, write_jsonl
from preactir.utils.metrics import quality_vector


MIO100_NAME_MAP = {
    "rain": "rain",
    "haze": "haze",
    "motion blur": "motion_blur",
    "low resolution": "low_resolution",
    "dark": "dark",
    "noise": "noise",
    "defocus blur": "defocus_blur",
    "jpeg compression artifact": "jpeg",
}


@dataclass(frozen=True)
class MiO100StateSource:
    image_path: Path
    clean_path: Path
    source_id: str
    degradation_order: tuple[str, ...]
    combination: str
    group: str


def parse_mio100_combination(name: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in name.split("+") if token.strip()]
    try:
        return tuple(MIO100_NAME_MAP[token] for token in tokens)
    except KeyError as exc:
        raise ValueError(f"Unknown MiO100 degradation token in '{name}': {exc.args[0]}") from exc


class MiO100DatasetBuilder(InterventionDatasetBuilder):
    """Build PreActIR manifests and tool rollouts from the official AgenticIR MiO100 release."""

    def __init__(
        self,
        dataset_root: str | Path,
        output_root: str | Path,
        degradation_names: list[str],
        registry: ToolRegistry,
        options: BuilderOptions,
        val_fraction: float = 0.2,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.hq_root = self.dataset_root / "HQ"
        self.train_root = self.dataset_root / "train"
        self.test_root = self.dataset_root / "test"
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be between 0 and 1")
        self.val_fraction = float(val_fraction)
        super().__init__(
            clean_root=self.hq_root,
            output_root=output_root,
            degradation_names=degradation_names,
            registry=registry,
            options=options,
        )
        missing = sorted(set(MIO100_NAME_MAP.values()) - set(self.degradation_names))
        if missing:
            raise ValueError(f"MiO100 config is missing degradation labels: {missing}")
        for required in (self.hq_root, self.train_root, self.test_root):
            if not required.is_dir():
                raise FileNotFoundError(f"Missing extracted MiO100 directory: {required}")

    def _source(self, image_path: Path, combination: str, group: str) -> MiO100StateSource:
        clean_path = self.hq_root / image_path.name
        if not clean_path.is_file():
            raise FileNotFoundError(f"No HQ pair for {image_path}: expected {clean_path}")
        return MiO100StateSource(
            image_path=image_path.resolve(),
            clean_path=clean_path.resolve(),
            source_id=image_path.stem,
            degradation_order=parse_mio100_combination(combination),
            combination=combination,
            group=group,
        )

    def _discover_train(self) -> list[MiO100StateSource]:
        rows: list[MiO100StateSource] = []
        for combination_dir in sorted(path for path in self.train_root.iterdir() if path.is_dir()):
            for image_path in list_images(combination_dir):
                rows.append(self._source(image_path, combination_dir.name, "Group A"))
        return rows

    def _discover_test(self) -> list[MiO100StateSource]:
        rows: list[MiO100StateSource] = []
        for group_dir in sorted(path for path in self.test_root.iterdir() if path.is_dir()):
            for combination_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
                for image_path in list_images(combination_dir):
                    rows.append(self._source(image_path, combination_dir.name, group_dir.name))
        return rows

    def _split_sources(self) -> tuple[dict[str, list[MiO100StateSource]], list[str], list[str]]:
        official_train = self._discover_train()
        source_ids = sorted({row.source_id for row in official_train})
        if len(source_ids) < 2:
            raise RuntimeError("MiO100 training data must contain at least two distinct HQ identities")
        rng = np.random.default_rng(self.options.seed)
        shuffled = list(source_ids)
        rng.shuffle(shuffled)
        val_count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * self.val_fraction))))
        val_ids = set(shuffled[:val_count])
        train_ids = set(shuffled[val_count:])
        split_rows = {
            "train": [row for row in official_train if row.source_id in train_ids],
            "val": [row for row in official_train if row.source_id in val_ids],
            "test": self._discover_test(),
        }
        return split_rows, sorted(train_ids), sorted(val_ids)

    def _state_id(self, split: str, source: MiO100StateSource) -> str:
        text = f"{split}:{source.group}:{source.combination}:{source.source_id}:{self.options.seed}"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return f"mio100_{split}_{digest}"

    def build(
        self,
        splits: list[str] | None = None,
        transition_splits: set[str] | None = None,
        limit_per_split: int | None = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        splits = splits or ["train", "val", "test"]
        unknown_splits = sorted(set(splits) - {"train", "val", "test"})
        if unknown_splits:
            raise ValueError(f"Unknown MiO100 splits: {unknown_splits}")
        transition_splits = {"train", "val"} if transition_splits is None else set(transition_splits)
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError(
                f"Invalid shard selection: shard_index={shard_index}, num_shards={num_shards}"
            )
        split_sources, train_ids, val_ids = self._split_sources()
        ensure_dir(self.output_root)
        metadata = {
            "dataset": "MiO100-AgenticIR",
            "dataset_root": str(self.dataset_root),
            "protocol": "official Group A/B/C; train/val divided only within the 20 official training IDs",
            "degradation_names": self.degradation_names,
            "tool_names": self.registry.names(),
            "quality_names": ["psnr_scaled", "ssim", "l1_similarity", "edge_similarity"],
            "train_source_ids": train_ids,
            "val_source_ids": val_ids,
            "transition_splits": sorted(transition_splits),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "options": vars(self.options),
        }
        metadata_name = (
            "metadata.json"
            if num_shards == 1
            else f"metadata.shard-{shard_index:05d}-of-{num_shards:05d}.json"
        )
        with (self.output_root / metadata_name).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        for split in splits:
            sources = split_sources[split]
            if limit_per_split is not None:
                sources = sources[:limit_per_split]
            sources = [
                source for index, source in enumerate(sources) if index % num_shards == shard_index
            ]
            state_rows: list[dict[str, Any]] = []
            transition_rows: list[dict[str, Any]] = []
            for source in tqdm(sources, desc=f"Building MiO100 {split}"):
                state_row, transitions = self._build_official_state(
                    split=split,
                    source=source,
                    build_transitions=split in transition_splits,
                )
                state_rows.append(state_row)
                transition_rows.extend(transitions)
            suffix = "" if num_shards == 1 else f".shard-{shard_index:05d}-of-{num_shards:05d}"
            write_jsonl(self.output_root / f"states_{split}{suffix}.jsonl", state_rows)
            write_jsonl(self.output_root / f"transitions_{split}{suffix}.jsonl", transition_rows)

    def _build_official_state(
        self,
        *,
        split: str,
        source: MiO100StateSource,
        build_transitions: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        rollout_size = None if self.options.rollout_native_resolution else self.options.image_size
        current = load_image(source.image_path, rollout_size)
        clean = load_image(source.clean_path, rollout_size)
        has_low_resolution = "low_resolution" in source.degradation_order
        if current.shape != clean.shape and not has_low_resolution:
            raise ValueError(f"LQ/HQ shape mismatch for {source.image_path}: {current.shape} vs {clean.shape}")
        height, width = clean.shape[:2]
        metric_current = align_to_reference(current, clean)
        active_names = list(source.degradation_order)
        masks_by_name = {
            name: np.ones((height, width), dtype=np.float32) for name in self.degradation_names
        }
        presence = [1.0 if name in active_names else 0.0 for name in self.degradation_names]
        proxy_scores = compute_degradation_scores(
            metric_current, clean, masks_by_name, self.degradation_names
        )
        severity = [
            float(proxy_scores[index]) if presence[index] > 0.5 else 0.0
            for index in range(len(self.degradation_names))
        ]
        q_before = quality_vector(metric_current, clean)
        state_id = self._state_id(split, source)
        state_row: dict[str, Any] = {
            "state_id": state_id,
            "split": split,
            "dataset": "MiO100-AgenticIR",
            "group": source.group,
            "combination": source.combination,
            "source_id": source.source_id,
            "source_path": str(source.image_path),
            "clean_path": str(source.clean_path),
            "image_path": str(source.image_path),
            "degradation_order": active_names,
            "presence": presence,
            "severity": severity,
            "proxy_scores": proxy_scores.tolist(),
            "mask_paths": {},
            "mask_kind": "global",
            "quality": q_before.tolist(),
        }
        if not build_transitions:
            return state_row, []

        seed_text = f"{state_id}:{self.options.seed}"
        state_seed = int(hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        transitions = self._build_transitions(
            split=split,
            state_id=state_id,
            current=current,
            clean=clean,
            current_path=source.image_path,
            clean_path=source.clean_path,
            active_names=active_names,
            masks_by_name=masks_by_name,
            presence=presence,
            severity=severity,
            proxy_scores=proxy_scores,
            quality_before=q_before,
            rng=np.random.default_rng(state_seed),
        )
        for transition in transitions:
            transition.update(
                {
                    "dataset": "MiO100-AgenticIR",
                    "group": source.group,
                    "combination": source.combination,
                    "source_id": source.source_id,
                }
            )
        return state_row, transitions
