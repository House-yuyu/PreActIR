from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from preactir.utils.io import read_jsonl


def load_benchmark_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    payload = _load_benchmark_protocol(protocol_path, stack=())
    payload["_config_path"] = str(protocol_path)
    return payload


def _load_benchmark_protocol(
    protocol_path: Path,
    *,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    if protocol_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, protocol_path))
        raise ValueError(f"Benchmark protocol inheritance cycle: {chain}")
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Benchmark protocol must be a JSON object: {protocol_path}")
    base_reference = payload.pop("extends", None)
    expected_base_hash = payload.pop("extends_sha256", None)
    if base_reference is None:
        if expected_base_hash is not None:
            raise ValueError("extends_sha256 requires an extends path")
        return payload

    base_path = Path(str(base_reference))
    if not base_path.is_absolute():
        base_path = (protocol_path.parent / base_path).resolve()
    if expected_base_hash is None:
        raise ValueError(f"Inherited protocol must lock extends_sha256: {protocol_path}")
    actual_base_hash = sha256_file(base_path)
    if actual_base_hash != str(expected_base_hash):
        raise ValueError(
            f"Inherited protocol base hash mismatch for {base_path}: "
            f"expected {expected_base_hash}, found {actual_base_hash}"
        )
    base = _load_benchmark_protocol(base_path, stack=(*stack, protocol_path))
    merged = _json_merge_patch(base, payload)
    merged["_base_protocol"] = {
        "path": str(base_path),
        "sha256": actual_base_hash,
    }
    return merged


def _json_merge_patch(target: Any, patch: Any) -> Any:
    """Apply RFC 7396-style merge semantics to compact protocol overrides."""

    if not isinstance(patch, dict):
        return deepcopy(patch)
    result = deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _json_merge_patch(result.get(key), value)
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_multiset_sha256(paths: Iterable[Path]) -> tuple[str, int, int]:
    file_hashes = sorted(sha256_file(path) for path in paths)
    digest = hashlib.sha256()
    for file_hash in file_hashes:
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(file_hashes), len(set(file_hashes))


def audit_mio100_protocol(
    data_root: str | Path,
    split: str,
    protocol: dict[str, Any],
    *,
    verify_content: bool = True,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    manifest_path = root / f"states_{split}.jsonl"
    rows = read_jsonl(manifest_path)
    expected = protocol["dataset"]
    mismatches: list[str] = []

    state_ids = [str(row["state_id"]) for row in rows]
    input_paths = [_resolve(root, str(row["image_path"])) for row in rows]
    clean_paths = [_resolve(root, str(row["clean_path"])) for row in rows]
    unique_clean_paths = sorted(set(clean_paths), key=str)
    group_counts = dict(sorted(Counter(str(row.get("group")) for row in rows).items()))
    combinations = {
        (str(row.get("group")), str(row.get("combination"))) for row in rows
    }
    combination_counts = Counter(
        (str(row.get("group")), str(row.get("combination"))) for row in rows
    )
    actual: dict[str, Any] = {
        "split": split,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "num_images": len(rows),
        "num_unique_state_ids": len(set(state_ids)),
        "num_unique_input_paths": len(set(input_paths)),
        "num_unique_clean_paths": len(unique_clean_paths),
        "num_combinations": len(combinations),
        "group_counts": group_counts,
        "combination_counts": {
            f"{group}/{combination}": count
            for (group, combination), count in sorted(combination_counts.items())
        },
    }

    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    missing_cleans = [str(path) for path in unique_clean_paths if not path.is_file()]
    actual["missing_input_files"] = len(missing_inputs)
    actual["missing_clean_files"] = len(missing_cleans)
    if missing_inputs:
        mismatches.append(f"missing input files: {len(missing_inputs)}")
    if missing_cleans:
        mismatches.append(f"missing clean files: {len(missing_cleans)}")

    scalar_keys = (
        "split",
        "manifest_sha256",
        "num_images",
        "num_unique_state_ids",
        "num_unique_input_paths",
        "num_unique_clean_paths",
        "num_combinations",
    )
    for key in scalar_keys:
        if actual[key] != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, found {actual[key]!r}")
    if group_counts != expected["group_counts"]:
        mismatches.append(
            f"group_counts: expected {expected['group_counts']!r}, found {group_counts!r}"
        )

    expected_combination_order = expected.get("combination_order")
    if expected_combination_order is not None:
        expected_combinations = {
            (group, combination)
            for group, group_combinations in expected_combination_order.items()
            for combination in group_combinations
        }
        if combinations != expected_combinations:
            missing = sorted(expected_combinations - combinations)
            extra = sorted(combinations - expected_combinations)
            mismatches.append(
                f"combination membership differs: missing={missing!r}, extra={extra!r}"
            )
        for group, group_combinations in expected_combination_order.items():
            expected_group_count = int(expected["group_counts"][group])
            if expected_group_count % len(group_combinations) != 0:
                mismatches.append(
                    f"group {group} count {expected_group_count} is not divisible by "
                    f"{len(group_combinations)} combinations"
                )
                continue
            expected_cell_count = expected_group_count // len(group_combinations)
            for combination in group_combinations:
                actual_cell_count = combination_counts[(group, combination)]
                if actual_cell_count != expected_cell_count:
                    mismatches.append(
                        f"combination {group}/{combination}: expected {expected_cell_count}, "
                        f"found {actual_cell_count}"
                    )

    if verify_content and not missing_inputs and not missing_cleans:
        input_digest, input_count, unique_input_contents = content_multiset_sha256(input_paths)
        clean_digest, clean_count, unique_clean_contents = content_multiset_sha256(
            unique_clean_paths
        )
        actual.update(
            {
                "input_content_multiset_sha256": input_digest,
                "clean_content_set_sha256": clean_digest,
                "num_hashed_inputs": input_count,
                "num_hashed_cleans": clean_count,
                "num_unique_input_contents": unique_input_contents,
                "num_unique_clean_contents": unique_clean_contents,
            }
        )
        for key in (
            "input_content_multiset_sha256",
            "clean_content_set_sha256",
            "num_unique_input_contents",
            "num_unique_clean_contents",
        ):
            if actual[key] != expected[key]:
                mismatches.append(f"{key}: expected {expected[key]!r}, found {actual[key]!r}")

    return {
        "protocol": protocol["protocol"],
        "config": protocol["_config_path"],
        "verify_content": verify_content,
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def audit_mio100_training_subset(
    data_root: str | Path,
    manifest_path: str | Path,
    protocol: dict[str, Any],
    *,
    verify_content: bool = True,
) -> dict[str, Any]:
    """Audit a diagnostic OOF subset against the frozen MiO100 train release.

    This deliberately does not grant main-table eligibility. It verifies that
    every selected state belongs to the protocol's official training identity
    and degradation-combination sets and that no test path is present.
    """

    root = Path(data_root).resolve()
    manifest = Path(manifest_path).resolve()
    rows = read_jsonl(manifest)
    expected = protocol["training_dataset"]
    expected_ids = {str(value) for value in expected["hq_identities"]}
    expected_combinations = {str(value) for value in expected["combinations"]}
    mismatches: list[str] = []
    state_ids = [str(row["state_id"]) for row in rows]
    source_ids = [str(row.get("source_id")) for row in rows]
    combinations = [str(row.get("combination")) for row in rows]
    input_paths = [_resolve(root, str(row["image_path"])) for row in rows]
    clean_paths = [_resolve(root, str(row["clean_path"])) for row in rows]

    if not rows:
        mismatches.append("diagnostic subset is empty")
    if len(state_ids) != len(set(state_ids)):
        mismatches.append("diagnostic subset contains duplicate state IDs")
    if len(input_paths) != len(set(input_paths)):
        mismatches.append("diagnostic subset contains duplicate input paths")
    unknown_ids = sorted(set(source_ids) - expected_ids)
    if unknown_ids:
        mismatches.append(f"source identities are outside the training release: {unknown_ids}")
    unknown_combinations = sorted(set(combinations) - expected_combinations)
    if unknown_combinations:
        mismatches.append(
            f"degradation combinations are outside the training release: {unknown_combinations}"
        )

    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    missing_cleans = [str(path) for path in clean_paths if not path.is_file()]
    if missing_inputs:
        mismatches.append(f"missing input files: {len(missing_inputs)}")
    if missing_cleans:
        mismatches.append(f"missing clean files: {len(missing_cleans)}")
    test_paths = [
        str(path)
        for path in (*input_paths, *clean_paths)
        if "test" in {part.lower() for part in path.parts}
    ]
    if test_paths:
        mismatches.append(f"diagnostic training subset references test paths: {len(test_paths)}")

    for row, input_path, clean_path in zip(rows, input_paths, clean_paths, strict=True):
        source_id = str(row.get("source_id"))
        combination = str(row.get("combination"))
        if input_path.stem != source_id or input_path.parent.name != combination:
            mismatches.append(
                f"state {row['state_id']} input path does not match source/combination metadata"
            )
        if clean_path.stem != source_id:
            mismatches.append(
                f"state {row['state_id']} clean path does not match source identity"
            )

    actual: dict[str, Any] = {
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "num_images": len(rows),
        "num_unique_state_ids": len(set(state_ids)),
        "num_unique_input_paths": len(set(input_paths)),
        "num_unique_clean_paths": len(set(clean_paths)),
        "source_ids": sorted(set(source_ids)),
        "combinations": sorted(set(combinations)),
        "missing_input_files": len(missing_inputs),
        "missing_clean_files": len(missing_cleans),
        "test_path_references": len(test_paths),
    }
    if verify_content and not missing_inputs and not missing_cleans:
        input_digest, input_count, unique_input_contents = content_multiset_sha256(input_paths)
        clean_digest, clean_count, unique_clean_contents = content_multiset_sha256(
            sorted(set(clean_paths), key=str)
        )
        actual.update(
            {
                "input_content_multiset_sha256": input_digest,
                "clean_content_set_sha256": clean_digest,
                "num_hashed_inputs": input_count,
                "num_hashed_cleans": clean_count,
                "num_unique_input_contents": unique_input_contents,
                "num_unique_clean_contents": unique_clean_contents,
            }
        )
    return {
        "protocol": protocol["protocol"],
        "config": protocol["_config_path"],
        "audit_kind": "training_subset_diagnostic",
        "verify_content": verify_content,
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
        "main_table_eligible": False,
    }


def audit_mio100_archives(
    dataset_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    expected = protocol["dataset"]["archive_sha256"]
    actual: dict[str, str | None] = {}
    mismatches: list[str] = []
    for name, expected_hash in expected.items():
        path = root / name
        actual_hash = sha256_file(path) if path.is_file() else None
        actual[name] = actual_hash
        if actual_hash != expected_hash:
            mismatches.append(f"{name}: expected {expected_hash}, found {actual_hash}")
    return {
        "dataset_root": str(root),
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def audit_mio100_training_release(
    dataset_root: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    expected = protocol["training_dataset"]
    train_root = root / str(expected["release_subdirectory"])
    paths = sorted(train_root.glob("*/*.png"))
    identities = sorted({path.stem for path in paths})
    combination_counts = Counter(path.parent.name for path in paths)
    content_digest, content_count, unique_contents = content_multiset_sha256(paths)
    actual = {
        "train_root": str(train_root),
        "num_images": len(paths),
        "num_unique_contents": unique_contents,
        "num_hq_identities": len(identities),
        "hq_identities": identities,
        "num_combinations": len(combination_counts),
        "combination_counts": dict(sorted(combination_counts.items())),
        "content_multiset_sha256": content_digest,
        "num_hashed_images": content_count,
    }
    mismatches: list[str] = []
    for key in (
        "num_images",
        "num_unique_contents",
        "num_hq_identities",
        "hq_identities",
        "num_combinations",
        "content_multiset_sha256",
    ):
        if actual[key] != expected[key]:
            mismatches.append(f"{key}: expected {expected[key]!r}, found {actual[key]!r}")
    expected_counts = {
        combination: int(expected["images_per_combination"])
        for combination in expected["combinations"]
    }
    if actual["combination_counts"] != expected_counts:
        mismatches.append(
            f"combination_counts: expected {expected_counts!r}, "
            f"found {actual['combination_counts']!r}"
        )
    return {
        "dataset_root": str(root),
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
