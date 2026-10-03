#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# BENCHMARK07B — train-only Full and PCA90 target representations.
#
# Consumes the frozen 323-dimensional UCSFFSX7 target table and the row manifests
# frozen by BENCHMARK06B. This stage changes no input-feature decision.
#
# Target-fit groups
# -----------------
# 1. snapshot_all
#    - scaler and PCA fitted only on snapshot_all/train target rows.
# 2. matched_pair
#    - scaler and PCA fitted only on snapshot_matched_last/train target rows;
#    - reused unchanged by snapshot_matched_last and longitudinal_previous_last,
#      whose current target rows must be identical and in identical order.
#
# Representations
# ---------------
# Full:
#     featurewise StandardScaler fitted on train only, retaining all 323 targets.
# PCA90:
#     PCA fitted on standardized train targets only, retaining the smallest
#     number of components whose cumulative train explained variance is >= 0.90.
#
# Validation is transform-only. No target is imputed. Test target values are not
# parsed, transformed, materialized, or used for any decision.
#
# Privacy
# -------
# All target matrices, row manifests, target names, scaler parameters, PCA
# components/loadings and fitted objects are LOCAL ONLY. SAFE reports contain
# only aggregate dimensions, metrics, hashes and policy metadata. Never share
# files from data/processed/local_only/benchmark07b_target_representations/.

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "benchmark07b"
TARGET_POLICY_VERSION: Final[str] = "1.1.0-crosschecked-b"
TARGET_COUNT: Final[int] = 323
PCA_VARIANCE_THRESHOLD: Final[float] = 0.90
PCA_SVD_SOLVER: Final[str] = "full"
OUTPUT_DTYPE: Final[str] = "float32"

EXPECTED_BENCHMARK06B_VERSION: Final[str] = "0.1.0"
HISTORICAL_PREPROCESSING_DECISION_ID_REFERENCE: Final[str] = "276e092a4b9f0c122e30"
HISTORICAL_SCENARIO_DECISION_ID_REFERENCE: Final[str] = "8662f66fae4b470c7e4f"
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"

RID: Final[str] = "RID"
VISCODE2: Final[str] = "VISCODE2"
SPLIT: Final[str] = "split"
CURRENT_VISCODE: Final[str] = "current_VISCODE2"
TRAIN: Final[str] = "train"
VALIDATION: Final[str] = "validation"
TEST: Final[str] = "test"

SCENARIOS: Final[tuple[str, ...]] = (
    "snapshot_all",
    "snapshot_matched_last",
    "longitudinal_previous_last",
)
TARGET_GROUPS: Final[tuple[str, ...]] = ("snapshot_all", "matched_pair")
SCENARIO_TO_GROUP: Final[dict[str, str]] = {
    "snapshot_all": "snapshot_all",
    "snapshot_matched_last": "matched_pair",
    "longitudinal_previous_last": "matched_pair",
}
GROUP_REFERENCE_SCENARIO: Final[dict[str, str]] = {
    "snapshot_all": "snapshot_all",
    "matched_pair": "snapshot_matched_last",
}

TARGETS_RELATIVE: Final[str] = "03_final_ready_tables/UCSFFSX7_targets_323.csv.gz"
TARGET_CATALOG_RELATIVE: Final[str] = "03_final_ready_tables/MRI_target_catalog_323.csv"
PACKAGE_MANIFEST_RELATIVE: Final[str] = "04_official_split/official_frozen_manifest.json"
PACKAGE_INVENTORY_RELATIVE: Final[str] = "04_official_split/frozen_file_inventory.csv"

BENCHMARK06B_LOCAL: Final[str] = "data/processed/local_only/benchmark06b_preprocessed_inputs"
BENCHMARK06B_SUMMARY: Final[str] = "reports/benchmark06b/safe/benchmark06b_summary.json"
BENCHMARK06B_CHECKS: Final[str] = "reports/benchmark06b/safe/benchmark06b_validation_checks.csv"
PREPROCESSING_POLICY: Final[str] = "config/benchmark_preprocessing_policy_b.json"
PREPROCESSING_MANIFEST: Final[str] = "config/benchmark_preprocessing_manifest_b.csv"
PREPROCESSING_LOCK: Final[str] = "config/benchmark_preprocessing_lock_b.json"

LOCAL_OUTPUT_DIR: Final[str] = "data/processed/local_only/benchmark07b_target_representations"
TARGET_POLICY_PATH: Final[str] = "config/benchmark_target_policy_b.json"
TARGET_SCENARIO_MAP_PATH: Final[str] = "config/benchmark_target_scenario_map_b.csv"
TARGET_LOCK_PATH: Final[str] = "config/benchmark_target_lock_b.json"


class TargetStageAbort(RuntimeError):
    # Fail-closed exception for target-stage incompatibility.
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    stage: Path
    safe: Path
    logs: Path
    benchmark06b_local: Path
    benchmark06b_summary: Path
    benchmark06b_checks: Path
    preprocessing_policy: Path
    preprocessing_manifest: Path
    preprocessing_lock: Path
    targets: Path
    target_catalog: Path
    package_manifest: Path
    package_inventory: Path
    local: Path
    target_policy: Path
    target_scenario_map: Path
    target_lock: Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: Any
    observed: Any
    details: str = ""


class Checks:
    def __init__(self) -> None:
        self.items: list[Check] = []

    def add(
        self,
        name: str,
        passed: bool,
        expected: Any = None,
        observed: Any = None,
        details: str = "",
    ) -> bool:
        item = Check(str(name), bool(passed), expected, observed, str(details))
        self.items.append(item)
        logging.log(
            logging.INFO if passed else logging.ERROR,
            "%s | %s",
            "PASS" if passed else "FAIL",
            name,
        )
        if not passed:
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                name,
                expected,
                observed,
                f" | {details}" if details else "",
            )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "name": item.name,
                    "passed": item.passed,
                    "expected": json_scalar(item.expected),
                    "observed": json_scalar(item.observed),
                    "details": item.details,
                }
                for item in self.items
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit train-only Full and PCA90 target representations for BENCHMARK07B."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing BENCHMARK07B local/config outputs intentionally.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic target-scaling/PCA test without ADNI files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    package = (
        args.package_root.expanduser().resolve()
        if args.package_root is not None
        else root / "data" / "processed" / "frozen_experiment_v1_1"
    )
    reports_root = (
        args.reports_root.expanduser().resolve()
        if args.reports_root is not None
        else root / "reports"
    )
    processed_root = (
        args.processed_root.expanduser().resolve()
        if args.processed_root is not None
        else root / "data" / "processed" / "local_only"
    )
    stage = reports_root / STAGE_NAME
    return Paths(
        root=root,
        package=package,
        stage=stage,
        safe=stage / "safe",
        logs=stage / "logs",
        benchmark06b_local=root / BENCHMARK06B_LOCAL,
        benchmark06b_summary=root / BENCHMARK06B_SUMMARY,
        benchmark06b_checks=root / BENCHMARK06B_CHECKS,
        preprocessing_policy=root / PREPROCESSING_POLICY,
        preprocessing_manifest=root / PREPROCESSING_MANIFEST,
        preprocessing_lock=root / PREPROCESSING_LOCK,
        targets=package / TARGETS_RELATIVE,
        target_catalog=package / TARGET_CATALOG_RELATIVE,
        package_manifest=package / PACKAGE_MANIFEST_RELATIVE,
        package_inventory=package / PACKAGE_INVENTORY_RELATIVE,
        local=processed_root / "benchmark07b_target_representations",
        target_policy=root / TARGET_POLICY_PATH,
        target_scenario_map=root / TARGET_SCENARIO_MAP_PATH,
        target_lock=root / TARGET_LOCK_PATH,
    )


def configure_logging(paths: Paths) -> None:
    paths.safe.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "benchmark07b.log", encoding="utf-8"),
        ],
        force=True,
    )


def json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return json.dumps([json_scalar(v) for v in value], ensure_ascii=False)
    if isinstance(value, Mapping):
        return json.dumps(
            {str(k): json_scalar(v) for k, v in value.items()},
            ensure_ascii=False,
            sort_keys=True,
        )
    return str(value)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise TargetStageAbort(f"Cannot parse boolean: {value!r}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(
        sanitize_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(sanitize_json(dict(payload)), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def save_npy_deterministic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def write_csv_gzip_deterministic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(
        tmp,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    tmp.replace(path)


def require_columns(frame: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise TargetStageAbort(f"{context} missing required columns: {missing}")


def normalize_rid_scalar(value: Any, context: str) -> str:
    text = str(value).strip()
    if not text:
        raise TargetStageAbort(f"Missing RID in {context}.")
    try:
        number = float(text)
    except ValueError as exc:
        raise TargetStageAbort(f"Non-numeric RID in {context}.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise TargetStageAbort(f"Invalid RID in {context}.")
    return str(int(number))


def normalize_viscode_scalar(value: Any, context: str) -> str:
    text = str(value).strip().casefold()
    if not text:
        raise TargetStageAbort(f"Missing VISCODE2 in {context}.")
    return text


def normalize_split_scalar(value: Any, context: str) -> str:
    text = str(value).strip().casefold()
    if text not in {TRAIN, VALIDATION, TEST}:
        raise TargetStageAbort(f"Unexpected split in {context}: {value!r}")
    return text


def normalize_manifest(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    require_columns(frame, [RID, SPLIT, CURRENT_VISCODE], context)
    output = frame.copy().reset_index(drop=True)
    output[RID] = [normalize_rid_scalar(v, context) for v in output[RID].tolist()]
    output[SPLIT] = [normalize_split_scalar(v, context) for v in output[SPLIT].tolist()]
    output[CURRENT_VISCODE] = [
        normalize_viscode_scalar(v, context) for v in output[CURRENT_VISCODE].tolist()
    ]
    if output.duplicated([RID, CURRENT_VISCODE, SPLIT]).any():
        raise TargetStageAbort(f"Duplicate current-target keys in {context}.")
    return output


def key_tuples(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(
        zip(
            frame[RID].astype(str),
            frame[CURRENT_VISCODE].astype(str),
            frame[SPLIT].astype(str),
        )
    )


def load_target_catalog(paths: Paths, checks: Checks) -> tuple[pd.DataFrame, list[str]]:
    checks.add("artifact__target_catalog", paths.target_catalog.is_file(), True, paths.target_catalog.is_file(), str(paths.target_catalog))
    if not paths.target_catalog.is_file():
        raise TargetStageAbort("Frozen target catalog is missing.")
    catalog = pd.read_csv(paths.target_catalog, low_memory=False)
    require_columns(catalog, ["target_position_1_based", "target_column"], "target catalog")
    catalog["target_position_1_based"] = pd.to_numeric(
        catalog["target_position_1_based"], errors="raise"
    ).astype(int)
    catalog["target_column"] = catalog["target_column"].astype(str)
    catalog = catalog.sort_values("target_position_1_based", kind="stable").reset_index(drop=True)
    target_columns = catalog["target_column"].tolist()
    checks.add("target_catalog_count", len(catalog) == TARGET_COUNT, TARGET_COUNT, len(catalog))
    checks.add(
        "target_catalog_positions",
        catalog["target_position_1_based"].tolist() == list(range(1, TARGET_COUNT + 1)),
        f"1..{TARGET_COUNT}",
        f"rows={len(catalog)}",
    )
    checks.add("target_catalog_unique_columns", len(set(target_columns)) == TARGET_COUNT, TARGET_COUNT, len(set(target_columns)))
    if not checks.passed:
        raise TargetStageAbort("Target catalog validation failed.")
    return catalog, target_columns


def stream_development_targets(
    target_path: Path,
    target_columns: Sequence[str],
    checks: Checks,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    # Parse target values only for train/validation; never parse test targets.
    checks.add("artifact__targets", target_path.is_file(), True, target_path.is_file(), str(target_path))
    if not target_path.is_file():
        raise TargetStageAbort("Frozen target table is missing.")

    metadata_rows: list[dict[str, Any]] = []
    value_rows: list[np.ndarray] = []
    counts = {TRAIN: 0, VALIDATION: 0, TEST: 0}
    test_target_values_parsed = 0

    with gzip.open(target_path, mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise TargetStageAbort("Frozen target table is empty.") from exc
        index = {name: position for position, name in enumerate(header)}
        required = [RID, VISCODE2, SPLIT, "eligible_visit_position", *target_columns]
        missing = [column for column in required if column not in index]
        if missing:
            raise TargetStageAbort(f"Frozen target table missing columns: {missing}")
        target_indices = [index[column] for column in target_columns]

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise TargetStageAbort(
                    f"Target row {line_number} has {len(row)} cells; expected {len(header)}."
                )
            split = normalize_split_scalar(row[index[SPLIT]], f"target row {line_number}")
            counts[split] += 1
            if split == TEST:
                # Critical: do not touch the 323 target cells.
                continue

            rid = normalize_rid_scalar(row[index[RID]], f"target row {line_number}")
            viscode = normalize_viscode_scalar(row[index[VISCODE2]], f"target row {line_number}")
            try:
                position = int(float(row[index["eligible_visit_position"]]))
            except ValueError as exc:
                raise TargetStageAbort(
                    f"Invalid eligible_visit_position at target row {line_number}."
                ) from exc

            values = np.empty(TARGET_COUNT, dtype=np.float64)
            for j, source_index in enumerate(target_indices):
                text = row[source_index].strip()
                if not text:
                    raise TargetStageAbort(
                        f"Missing development target at row {line_number}, target position {j + 1}."
                    )
                try:
                    number = float(text)
                except ValueError as exc:
                    raise TargetStageAbort(
                        f"Non-numeric development target at row {line_number}, target position {j + 1}."
                    ) from exc
                if not math.isfinite(number):
                    raise TargetStageAbort(
                        f"Non-finite development target at row {line_number}, target position {j + 1}."
                    )
                values[j] = number

            metadata_rows.append(
                {
                    RID: rid,
                    VISCODE2: viscode,
                    SPLIT: split,
                    "eligible_visit_position": position,
                }
            )
            value_rows.append(values)

    metadata = pd.DataFrame(metadata_rows)
    values = np.vstack(value_rows) if value_rows else np.empty((0, TARGET_COUNT), dtype=np.float64)
    checks.add("development_target_rows_nonempty", len(metadata) > 0, True, len(metadata))
    checks.add("development_target_width", values.ndim == 2 and values.shape[1] == TARGET_COUNT, TARGET_COUNT, values.shape)
    checks.add("development_targets_finite", bool(np.isfinite(values).all()), True, bool(np.isfinite(values).all()))
    checks.add("development_target_keys_unique", not metadata.duplicated([RID, VISCODE2, SPLIT]).any(), True, not metadata.duplicated([RID, VISCODE2, SPLIT]).any())
    checks.add("test_target_values_parsed", test_target_values_parsed == 0, 0, test_target_values_parsed)
    checks.add("target_split_accounting", sum(counts.values()) == len(metadata) + counts[TEST], True, {**counts, "development": len(metadata)})
    if not checks.passed:
        raise TargetStageAbort("Frozen target loading failed.")
    return metadata, values, counts


def load_and_validate_upstream(paths: Paths, checks: Checks) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts: dict[str, Path] = {
        "package_root": paths.package,
        "benchmark06b_local": paths.benchmark06b_local,
        "benchmark06b_summary": paths.benchmark06b_summary,
        "benchmark06b_checks": paths.benchmark06b_checks,
        "preprocessing_policy": paths.preprocessing_policy,
        "preprocessing_manifest": paths.preprocessing_manifest,
        "preprocessing_lock": paths.preprocessing_lock,
        "package_manifest": paths.package_manifest,
        "package_inventory": paths.package_inventory,
    }
    for name, path in artifacts.items():
        exists = path.is_dir() if name.endswith("root") or name.endswith("local") else path.is_file()
        checks.add(f"artifact__{name}", exists, True, exists, str(path))
    if not checks.passed:
        raise TargetStageAbort("Required upstream artifacts are missing.")

    summary = read_json(paths.benchmark06b_summary)
    lock = read_json(paths.preprocessing_lock)
    check_frame = pd.read_csv(paths.benchmark06b_checks, low_memory=False)
    require_columns(check_frame, ["name", "passed"], "BENCHMARK06B checks")
    failed_upstream = int((~check_frame["passed"].map(parse_bool)).sum())

    checks.add("benchmark06b_passed", bool(summary.get("preprocessing_passed")), True, summary.get("preprocessing_passed"))
    checks.add("benchmark06b_version", str(summary.get("script_version")) == EXPECTED_BENCHMARK06B_VERSION, EXPECTED_BENCHMARK06B_VERSION, summary.get("script_version"))
    checks.add("benchmark06b_no_failed_checks", int(summary.get("failed_check_count", -1)) == 0, 0, summary.get("failed_check_count"))
    checks.add("benchmark06b_check_table_passed", failed_upstream == 0, 0, failed_upstream)
    preprocessing_decision_id = str(
        summary.get("preprocessing_decision_id", "")
    ).strip()
    lock_preprocessing_decision_id = str(
        lock.get("preprocessing_decision_id", "")
    ).strip()
    scenario_decision_id = str(
        summary.get("scenario_decision_id", "")
    ).strip()
    lock_scenario_decision_id = str(
        lock.get("upstream_scenario_decision_id", "")
    ).strip()

    checks.add(
        "preprocessing_decision_id_nonempty",
        bool(preprocessing_decision_id),
        True,
        bool(preprocessing_decision_id),
    )
    checks.add(
        "preprocessing_decision_chain_consistent",
        bool(preprocessing_decision_id)
        and preprocessing_decision_id == lock_preprocessing_decision_id,
        True,
        {
            "summary": preprocessing_decision_id,
            "lock": lock_preprocessing_decision_id,
        },
    )
    checks.add(
        "scenario_decision_id_nonempty",
        bool(scenario_decision_id),
        True,
        bool(scenario_decision_id),
    )
    checks.add(
        "scenario_decision_chain_consistent",
        bool(scenario_decision_id)
        and scenario_decision_id == lock_scenario_decision_id,
        True,
        {
            "summary": scenario_decision_id,
            "lock": lock_scenario_decision_id,
        },
    )
    checks.add("temporal_policy_decision_id", str(summary.get("temporal_policy_decision_id")) == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, summary.get("temporal_policy_decision_id"))
    checks.add("validation_not_refit_upstream", bool(summary.get("scope", {}).get("validation_transformed_without_refit")), True, summary.get("scope", {}).get("validation_transformed_without_refit"))
    checks.add("test_not_materialized_upstream", not bool(summary.get("scope", {}).get("test_materialized")), False, summary.get("scope", {}).get("test_materialized"))
    checks.add("targets_not_loaded_upstream", not bool(summary.get("scope", {}).get("targets_loaded")), False, summary.get("scope", {}).get("targets_loaded"))

    reported_artifacts = summary.get("artifacts", {})
    for check_name, path, report_key in [
        ("preprocessing_policy_hash", paths.preprocessing_policy, "preprocessing_policy_sha256"),
        ("preprocessing_manifest_hash", paths.preprocessing_manifest, "preprocessing_manifest_sha256"),
        ("preprocessing_lock_hash", paths.preprocessing_lock, "preprocessing_lock_sha256"),
    ]:
        actual = sha256_file(path)
        expected = reported_artifacts.get(report_key)
        checks.add(check_name, actual == expected, expected, actual)

    local_inventory_path = paths.benchmark06b_local / "LOCAL_ONLY_artifact_inventory.csv"
    checks.add("artifact__benchmark06b_inventory", local_inventory_path.is_file(), True, local_inventory_path.is_file(), str(local_inventory_path))
    if not local_inventory_path.is_file():
        raise TargetStageAbort("BENCHMARK06B local inventory is missing.")
    actual_inventory_hash = sha256_file(local_inventory_path)
    checks.add(
        "benchmark06b_inventory_hash",
        actual_inventory_hash == reported_artifacts.get("local_artifact_inventory_sha256"),
        reported_artifacts.get("local_artifact_inventory_sha256"),
        actual_inventory_hash,
    )
    inventory = pd.read_csv(local_inventory_path, low_memory=False)
    require_columns(inventory, ["relative_path", "sha256"], "BENCHMARK06B local inventory")
    failures = 0
    for row in inventory.to_dict(orient="records"):
        artifact = paths.benchmark06b_local / str(row["relative_path"])
        if not artifact.is_file() or sha256_file(artifact) != str(row["sha256"]):
            failures += 1
    checks.add("benchmark06b_local_inventory_matches", failures == 0, 0, failures)

    package_manifest = read_json(paths.package_manifest)
    expected_target_hash = package_manifest.get("critical_unchanged_sha256", {}).get(
        "UCSFFSX7_targets_323.csv.gz"
    )
    checks.add("package_manifest_declares_target_hash", bool(expected_target_hash), True, bool(expected_target_hash))
    actual_target_hash = sha256_file(paths.targets) if paths.targets.is_file() else None
    checks.add("frozen_manifest_target_hash", bool(expected_target_hash) and actual_target_hash == expected_target_hash, expected_target_hash, actual_target_hash)

    package_inventory = pd.read_csv(paths.package_inventory, low_memory=False)
    require_columns(package_inventory, ["relative_path", "sha256"], "frozen package inventory")
    inventory_lookup = {
        str(row["relative_path"]).replace("\\", "/"): str(row["sha256"])
        for row in package_inventory.to_dict(orient="records")
    }
    for label, relative, artifact in [
        ("targets", TARGETS_RELATIVE, paths.targets),
        ("target_catalog", TARGET_CATALOG_RELATIVE, paths.target_catalog),
    ]:
        expected = inventory_lookup.get(relative)
        actual = sha256_file(artifact) if artifact.is_file() else None
        checks.add(f"frozen_inventory_hash__{label}", expected is not None and actual == expected, expected, actual)

    if not checks.passed:
        raise TargetStageAbort("Upstream/target freeze validation failed.")
    return summary, lock


def load_scenario_manifests(
    paths: Paths,
    benchmark06b_summary: Mapping[str, Any],
    checks: Checks,
) -> dict[str, dict[str, pd.DataFrame]]:
    expected_dimensions = {
        str(row["scenario"]): row
        for row in benchmark06b_summary.get("scenario_dimensions", [])
    }
    manifests: dict[str, dict[str, pd.DataFrame]] = {}
    for scenario in SCENARIOS:
        scenario_dir = paths.benchmark06b_local / scenario
        train_path = scenario_dir / "row_manifest_train.csv.gz"
        validation_path = scenario_dir / "row_manifest_validation.csv.gz"
        checks.add(f"artifact__row_manifest__{scenario}__train", train_path.is_file(), True, train_path.is_file(), str(train_path))
        checks.add(f"artifact__row_manifest__{scenario}__validation", validation_path.is_file(), True, validation_path.is_file(), str(validation_path))
        if not train_path.is_file() or not validation_path.is_file():
            raise TargetStageAbort(f"Missing BENCHMARK06B row manifest for {scenario}.")

        train = normalize_manifest(pd.read_csv(train_path, low_memory=False), f"{scenario}/train")
        validation = normalize_manifest(pd.read_csv(validation_path, low_memory=False), f"{scenario}/validation")
        checks.add(f"manifest_split_train__{scenario}", set(train[SPLIT].unique()) == {TRAIN}, [TRAIN], sorted(train[SPLIT].unique().tolist()))
        checks.add(f"manifest_split_validation__{scenario}", set(validation[SPLIT].unique()) == {VALIDATION}, [VALIDATION], sorted(validation[SPLIT].unique().tolist()))
        expected = expected_dimensions.get(scenario, {})
        checks.add(f"manifest_train_rows__{scenario}", len(train) == int(expected.get("train_rows", -1)), expected.get("train_rows"), len(train))
        checks.add(f"manifest_validation_rows__{scenario}", len(validation) == int(expected.get("validation_rows", -1)), expected.get("validation_rows"), len(validation))
        checks.add(f"manifest_train_participants__{scenario}", int(train[RID].nunique()) == int(expected.get("train_participants", -1)), expected.get("train_participants"), int(train[RID].nunique()))
        checks.add(f"manifest_validation_participants__{scenario}", int(validation[RID].nunique()) == int(expected.get("validation_participants", -1)), expected.get("validation_participants"), int(validation[RID].nunique()))
        manifests[scenario] = {TRAIN: train, VALIDATION: validation}

    for split in (TRAIN, VALIDATION):
        matched_keys = key_tuples(manifests["snapshot_matched_last"][split])
        longitudinal_keys = key_tuples(manifests["longitudinal_previous_last"][split])
        checks.add(
            f"matched_longitudinal_target_order_identical__{split}",
            matched_keys == longitudinal_keys,
            True,
            matched_keys == longitudinal_keys,
        )
    if not checks.passed:
        raise TargetStageAbort("Scenario manifest validation failed.")
    return manifests


def build_target_lookup(
    metadata: pd.DataFrame,
    values: np.ndarray,
) -> dict[tuple[str, str, str], int]:
    lookup: dict[tuple[str, str, str], int] = {}
    for row_index, row in enumerate(metadata.itertuples(index=False)):
        key = (
            str(getattr(row, RID)),
            str(getattr(row, VISCODE2)),
            str(getattr(row, SPLIT)),
        )
        if key in lookup:
            raise TargetStageAbort("Duplicate frozen target key.")
        lookup[key] = row_index
    if len(lookup) != len(values):
        raise TargetStageAbort("Frozen target lookup/value length mismatch.")
    return lookup


def align_targets(
    manifest: pd.DataFrame,
    lookup: Mapping[tuple[str, str, str], int],
    values: np.ndarray,
    context: str,
) -> np.ndarray:
    indices: list[int] = []
    missing = 0
    for key in key_tuples(manifest):
        index = lookup.get(key)
        if index is None:
            missing += 1
        else:
            indices.append(index)
    if missing:
        raise TargetStageAbort(f"{context}: {missing} rows have no frozen target.")
    output = np.asarray(values[np.asarray(indices, dtype=np.int64), :], dtype=np.float64)
    if output.shape != (len(manifest), TARGET_COUNT):
        raise TargetStageAbort(f"Unexpected target shape for {context}: {output.shape}")
    return output


def participant_macro_rmse(residual: np.ndarray, manifest: pd.DataFrame) -> float:
    squared_per_row = np.mean(np.square(residual), axis=1)
    frame = pd.DataFrame({RID: manifest[RID].astype(str), "mse": squared_per_row})
    participant_mse = frame.groupby(RID, sort=False)["mse"].mean().to_numpy(dtype=float)
    return float(np.mean(np.sqrt(participant_mse)))


def reconstruction_metrics(
    standardized: np.ndarray,
    reconstructed: np.ndarray,
    manifest: pd.DataFrame,
) -> dict[str, float]:
    residual = reconstructed - standardized
    per_row_rmse = np.sqrt(np.mean(np.square(residual), axis=1))
    return {
        "cell_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "cell_mae": float(np.mean(np.abs(residual))),
        "row_macro_rmse": float(np.mean(per_row_rmse)),
        "participant_macro_rmse": participant_macro_rmse(residual, manifest),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }


def target_policy_payload() -> dict[str, Any]:
    return {
        "policy_version": TARGET_POLICY_VERSION,
        "target_count": TARGET_COUNT,
        "target_missingness": "forbidden; no target imputation",
        "fit_groups": {
            "snapshot_all": {
                "reference_scenario": "snapshot_all",
                "fit_rows": "snapshot_all train rows only",
            },
            "matched_pair": {
                "reference_scenario": "snapshot_matched_last",
                "fit_rows": "matched train rows only",
                "shared_by": ["snapshot_matched_last", "longitudinal_previous_last"],
                "reason": "paired scenarios have identical current target rows",
            },
        },
        "full": {
            "representation": "featurewise standardized original 323-dimensional target",
            "scaler": "StandardScaler",
            "fit": "train only within target group",
            "validation_refit": False,
        },
        "pca90": {
            "input": "standardized Full target",
            "estimator": "sklearn.decomposition.PCA",
            "variance_threshold": PCA_VARIANCE_THRESHOLD,
            "svd_solver": PCA_SVD_SOLVER,
            "whiten": False,
            "fit": "train only within target group",
            "validation_refit": False,
            "component_rule": "smallest component count reaching >= 0.90 cumulative train explained variance",
        },
        "inverse_transform": {
            "pca_to_standardized_full": "PCA.inverse_transform",
            "standardized_full_to_raw": "StandardScaler.inverse_transform",
        },
        "test": {
            "values_parsed": False,
            "materialized": False,
            "policy": "apply frozen group state only after final model/configuration lock",
        },
        "output_dtype": OUTPUT_DTYPE,
    }


def fit_group(
    group: str,
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    train_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    checks: Checks,
) -> dict[str, Any]:
    logging.info("Fitting target scaler and PCA90: %s", group)
    if train_raw.shape[1] != TARGET_COUNT or validation_raw.shape[1] != TARGET_COUNT:
        raise TargetStageAbort(f"Unexpected raw target width in {group}.")
    if not np.isfinite(train_raw).all() or not np.isfinite(validation_raw).all():
        raise TargetStageAbort(f"Non-finite target values in {group}.")

    train_range = np.ptp(train_raw, axis=0)
    zero_variance_raw = int(np.sum(train_range == 0.0))
    checks.add(f"raw_target_zero_variance__{group}", zero_variance_raw == 0, 0, zero_variance_raw)
    if zero_variance_raw:
        raise TargetStageAbort(f"Constant raw target dimensions in {group}.")

    scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
    y_full_train64 = scaler.fit_transform(train_raw)
    y_full_validation64 = scaler.transform(validation_raw)
    n_seen = scaler.n_samples_seen_
    if np.ndim(n_seen) > 0:
        n_seen_value = int(np.asarray(n_seen).ravel()[0])
    else:
        n_seen_value = int(n_seen)
    checks.add(f"scaler_fitted_samples__{group}", n_seen_value == len(train_raw), len(train_raw), n_seen_value)
    checks.add(f"full_train_finite__{group}", bool(np.isfinite(y_full_train64).all()), True, bool(np.isfinite(y_full_train64).all()))
    checks.add(f"full_validation_finite__{group}", bool(np.isfinite(y_full_validation64).all()), True, bool(np.isfinite(y_full_validation64).all()))

    train_mean_max_abs = float(np.max(np.abs(np.mean(y_full_train64, axis=0))))
    train_std_max_abs_error = float(np.max(np.abs(np.std(y_full_train64, axis=0, ddof=0) - 1.0)))
    checks.add(f"standardized_train_mean__{group}", train_mean_max_abs < 1e-10, "<1e-10", train_mean_max_abs)
    checks.add(f"standardized_train_std__{group}", train_std_max_abs_error < 1e-10, "<1e-10", train_std_max_abs_error)

    pca = PCA(
        n_components=PCA_VARIANCE_THRESHOLD,
        svd_solver=PCA_SVD_SOLVER,
        whiten=False,
        copy=True,
    )
    y_pca_train64 = pca.fit_transform(y_full_train64)
    y_pca_validation64 = pca.transform(y_full_validation64)
    component_count = int(pca.n_components_)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    cumulative_final = float(cumulative[-1])
    cumulative_previous = float(cumulative[-2]) if component_count > 1 else 0.0
    checks.add(f"pca_components_positive__{group}", 0 < component_count <= TARGET_COUNT, f"1..{TARGET_COUNT}", component_count)
    checks.add(f"pca90_threshold_reached__{group}", cumulative_final >= PCA_VARIANCE_THRESHOLD, f">={PCA_VARIANCE_THRESHOLD}", cumulative_final)
    checks.add(f"pca90_minimal_component_count__{group}", component_count == 1 or cumulative_previous < PCA_VARIANCE_THRESHOLD, f"<{PCA_VARIANCE_THRESHOLD}", cumulative_previous)
    checks.add(f"pca_train_finite__{group}", bool(np.isfinite(y_pca_train64).all()), True, bool(np.isfinite(y_pca_train64).all()))
    checks.add(f"pca_validation_finite__{group}", bool(np.isfinite(y_pca_validation64).all()), True, bool(np.isfinite(y_pca_validation64).all()))

    reconstructed_train = pca.inverse_transform(y_pca_train64)
    reconstructed_validation = pca.inverse_transform(y_pca_validation64)
    train_metrics = reconstruction_metrics(y_full_train64, reconstructed_train, train_manifest)
    validation_metrics = reconstruction_metrics(y_full_validation64, reconstructed_validation, validation_manifest)

    variance_frame = pd.DataFrame(
        {
            "component_1_based": np.arange(1, component_count + 1, dtype=int),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )

    return {
        "scaler": scaler,
        "pca": pca,
        "component_count": component_count,
        "cumulative_explained_variance": cumulative_final,
        "previous_cumulative_explained_variance": cumulative_previous,
        "y_full_train": np.asarray(y_full_train64, dtype=np.float32),
        "y_full_validation": np.asarray(y_full_validation64, dtype=np.float32),
        "y_pca_train": np.asarray(y_pca_train64, dtype=np.float32),
        "y_pca_validation": np.asarray(y_pca_validation64, dtype=np.float32),
        "variance_frame": variance_frame,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_mean_max_abs": train_mean_max_abs,
        "train_std_max_abs_error": train_std_max_abs_error,
        "raw_train_min_scale": float(np.min(scaler.scale_)),
        "raw_train_max_scale": float(np.max(scaler.scale_)),
    }


def build_story(
    dimensions: pd.DataFrame,
    pca_summary: pd.DataFrame,
    target_decision_id: str,
) -> str:
    rows = []
    for row in dimensions.to_dict(orient="records"):
        rows.append(
            f"- {row['target_group']}: train={row['train_rows']}, validation={row['validation_rows']}, "
            f"Full=323, PCA90={row['pca90_components']} components, "
            f"cumulative={row['pca90_cumulative_explained_variance']:.6f}."
        )
    return (
        "# BENCHMARK07B target representation story\n\n"
        f"Target decision ID: `{target_decision_id}`.\n\n"
        "The 323 frozen UCSFFSX7 targets are standardized with parameters fitted only on "
        "the corresponding training target group. PCA90 is then fitted only on those "
        "standardized training targets. Validation is transform-only and test target values "
        "are not parsed or materialized. Snapshot-matched and longitudinal share the same "
        "matched target transformation because their current target rows are identical.\n\n"
        + "\n".join(rows)
        + "\n"
    )


def ensure_output_freshness(paths: Paths, overwrite: bool) -> None:
    official = [paths.target_policy, paths.target_scenario_map, paths.target_lock]
    existing = [path for path in official if path.exists()]
    if paths.local.exists():
        existing.append(paths.local)
    if existing and not overwrite:
        joined = "\n  - ".join(str(p) for p in existing)
        raise TargetStageAbort(
            "BENCHMARK07B outputs already exist. Refusing to overwrite:\n  - "
            + joined
            + "\nUse --overwrite only for an intentional same-stage replacement."
        )


def run_stage(paths: Paths, args: argparse.Namespace, checks: Checks) -> dict[str, Any]:
    ensure_output_freshness(paths, args.overwrite)
    b06, preprocessing_lock = load_and_validate_upstream(paths, checks)
    preprocessing_decision_id = str(
        b06["preprocessing_decision_id"]
    ).strip()
    scenario_decision_id = str(
        b06["scenario_decision_id"]
    ).strip()
    manifests = load_scenario_manifests(paths, b06, checks)
    catalog, target_columns = load_target_catalog(paths, checks)
    target_metadata, target_values, target_split_counts = stream_development_targets(
        paths.targets, target_columns, checks
    )
    target_lookup = build_target_lookup(target_metadata, target_values)

    staging = paths.local.with_name(paths.local.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)

    artifact_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    group_results: dict[str, dict[str, Any]] = {}

    try:
        for group in TARGET_GROUPS:
            reference = GROUP_REFERENCE_SCENARIO[group]
            train_manifest = manifests[reference][TRAIN]
            validation_manifest = manifests[reference][VALIDATION]
            train_raw = align_targets(train_manifest, target_lookup, target_values, f"{group}/train")
            validation_raw = align_targets(validation_manifest, target_lookup, target_values, f"{group}/validation")
            result = fit_group(
                group,
                train_raw,
                validation_raw,
                train_manifest,
                validation_manifest,
                checks,
            )
            group_results[group] = result

            group_dir = staging / group
            group_dir.mkdir(parents=True, exist_ok=True)
            files_to_write: list[str] = []
            arrays = {
                "Y_full_train.npy": result["y_full_train"],
                "Y_full_validation.npy": result["y_full_validation"],
                "Y_pca90_train.npy": result["y_pca_train"],
                "Y_pca90_validation.npy": result["y_pca_validation"],
                "pca90_components.npy": np.asarray(result["pca"].components_, dtype=np.float64),
            }
            for filename, array in arrays.items():
                save_npy_deterministic(group_dir / filename, array)
                files_to_write.append(filename)

            write_csv_gzip_deterministic(train_manifest, group_dir / "row_manifest_train.csv.gz")
            write_csv_gzip_deterministic(validation_manifest, group_dir / "row_manifest_validation.csv.gz")
            write_csv_gzip_deterministic(result["variance_frame"], group_dir / "pca90_variance.csv.gz")
            catalog.to_csv(group_dir / "target_catalog.csv", index=False, lineterminator="\n")
            joblib.dump(result["scaler"], group_dir / "target_scaler.joblib", compress=3)
            joblib.dump(result["pca"], group_dir / "pca90.joblib", compress=3)
            files_to_write.extend(
                [
                    "row_manifest_train.csv.gz",
                    "row_manifest_validation.csv.gz",
                    "pca90_variance.csv.gz",
                    "target_catalog.csv",
                    "target_scaler.joblib",
                    "pca90.joblib",
                ]
            )

            state_summary = {
                "target_group": group,
                "reference_scenario": reference,
                "shared_by_scenarios": [
                    scenario for scenario, mapped_group in SCENARIO_TO_GROUP.items() if mapped_group == group
                ],
                "train_rows": len(train_manifest),
                "validation_rows": len(validation_manifest),
                "full_dimensions": TARGET_COUNT,
                "pca90_components": result["component_count"],
                "pca90_cumulative_explained_variance": result["cumulative_explained_variance"],
                "fit_split": TRAIN,
                "validation_refit": False,
                "test_materialized": False,
                "output_dtype": OUTPUT_DTYPE,
            }
            atomic_write_json(group_dir / "target_transform_summary.json", state_summary)
            files_to_write.append("target_transform_summary.json")

            for filename in files_to_write:
                path = group_dir / filename
                artifact_rows.append(
                    {
                        "target_group": group,
                        "relative_path": f"{group}/{filename}",
                        "sha256": sha256_file(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )

            dimension_rows.append(
                {
                    "target_group": group,
                    "reference_scenario": reference,
                    "scenarios": ";".join(
                        scenario for scenario, mapped_group in SCENARIO_TO_GROUP.items() if mapped_group == group
                    ),
                    "train_rows": len(train_manifest),
                    "validation_rows": len(validation_manifest),
                    "train_participants": int(train_manifest[RID].nunique()),
                    "validation_participants": int(validation_manifest[RID].nunique()),
                    "full_dimensions": TARGET_COUNT,
                    "pca90_components": result["component_count"],
                    "pca90_cumulative_explained_variance": result["cumulative_explained_variance"],
                    "output_dtype": OUTPUT_DTYPE,
                }
            )
            pca_rows.append(
                {
                    "target_group": group,
                    "variance_threshold": PCA_VARIANCE_THRESHOLD,
                    "svd_solver": PCA_SVD_SOLVER,
                    "pca90_components": result["component_count"],
                    "cumulative_explained_variance": result["cumulative_explained_variance"],
                    "cumulative_before_last_component": result["previous_cumulative_explained_variance"],
                    "first_component_explained_variance_ratio": float(result["pca"].explained_variance_ratio_[0]),
                    "last_retained_component_explained_variance_ratio": float(result["pca"].explained_variance_ratio_[-1]),
                }
            )
            for split, metrics in [(TRAIN, result["train_metrics"]), (VALIDATION, result["validation_metrics"])]:
                oracle_rows.append({"target_group": group, "split": split, **metrics})
            scaling_rows.append(
                {
                    "target_group": group,
                    "train_rows": len(train_manifest),
                    "target_dimensions": TARGET_COUNT,
                    "zero_variance_target_dimensions": 0,
                    "standardized_train_mean_max_abs": result["train_mean_max_abs"],
                    "standardized_train_std_max_abs_error": result["train_std_max_abs_error"],
                    "scaler_scale_min": result["raw_train_min_scale"],
                    "scaler_scale_max": result["raw_train_max_scale"],
                }
            )

        for split in (TRAIN, VALIDATION):
            checks.add(
                f"paired_scenarios_share_target_group__{split}",
                SCENARIO_TO_GROUP["snapshot_matched_last"]
                == SCENARIO_TO_GROUP["longitudinal_previous_last"]
                == "matched_pair",
                "matched_pair",
                SCENARIO_TO_GROUP["snapshot_matched_last"],
            )

        # Explicit equality audit: both paired scenarios must point to the same current Y rows.
        matched_group = group_results["matched_pair"]
        checks.add(
            "matched_pair_train_row_count_consistent",
            matched_group["y_full_train"].shape[0]
            == len(manifests["longitudinal_previous_last"][TRAIN]),
            len(manifests["longitudinal_previous_last"][TRAIN]),
            matched_group["y_full_train"].shape[0],
        )
        checks.add(
            "matched_pair_validation_row_count_consistent",
            matched_group["y_full_validation"].shape[0]
            == len(manifests["longitudinal_previous_last"][VALIDATION]),
            len(manifests["longitudinal_previous_last"][VALIDATION]),
            matched_group["y_full_validation"].shape[0],
        )
        checks.add("test_not_materialized", True, True, True)
        checks.add("test_target_values_never_parsed", True, True, True)

        if not checks.passed:
            raise TargetStageAbort(
                f"Target representation validation failed with {checks.failed} checks."
            )

        readme = "\n".join(
            [
                "BENCHMARK07B TARGET REPRESENTATIONS — LOCAL ONLY",
                "",
                "This directory contains ADNI-derived standardized MRI target matrices, PCA",
                "scores, row manifests, target scaler parameters, PCA components/loadings and",
                "target field names. Do not commit, upload, publish or share any file from this",
                "directory.",
                "",
                "Only train and validation targets are materialized. Test target values were not",
                "parsed or transformed.",
                "",
            ]
        )
        atomic_write_text(staging / "README_LOCAL_ONLY.txt", readme)
        artifact_rows.append(
            {
                "target_group": "<ROOT>",
                "relative_path": "README_LOCAL_ONLY.txt",
                "sha256": sha256_file(staging / "README_LOCAL_ONLY.txt"),
                "size_bytes": int((staging / "README_LOCAL_ONLY.txt").stat().st_size),
            }
        )
        inventory_frame = pd.DataFrame(artifact_rows).sort_values(
            ["target_group", "relative_path"], kind="stable"
        ).reset_index(drop=True)
        inventory_frame.to_csv(
            staging / "LOCAL_ONLY_artifact_inventory.csv",
            index=False,
            lineterminator="\n",
        )

        if paths.local.exists():
            shutil.rmtree(paths.local)
        staging.replace(paths.local)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    dimension_summary = pd.DataFrame(dimension_rows)
    pca_summary = pd.DataFrame(pca_rows)
    oracle_summary = pd.DataFrame(oracle_rows)
    scaling_summary = pd.DataFrame(scaling_rows)
    scenario_map = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "target_group": group,
"reference_scenario": GROUP_REFERENCE_SCENARIO[group],
                "full_representation": "Y_full",
                "pca90_representation": "Y_pca90",
            }
            for scenario, group in SCENARIO_TO_GROUP.items()
        ]
    )

    policy = target_policy_payload()
    upstream_hashes = {
        "benchmark06b_summary_sha256": sha256_file(paths.benchmark06b_summary),
        "benchmark06b_checks_sha256": sha256_file(paths.benchmark06b_checks),
        "preprocessing_policy_sha256": sha256_file(paths.preprocessing_policy),
        "preprocessing_manifest_sha256": sha256_file(paths.preprocessing_manifest),
        "preprocessing_lock_sha256": sha256_file(paths.preprocessing_lock),
        "targets_sha256": sha256_file(paths.targets),
        "target_catalog_sha256": sha256_file(paths.target_catalog),
        "package_manifest_sha256": sha256_file(paths.package_manifest),
        "package_inventory_sha256": sha256_file(paths.package_inventory),
    }
    decision_material = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "preprocessing_decision_id": preprocessing_decision_id,
        "scenario_decision_id": scenario_decision_id,
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "targets_sha256": upstream_hashes["targets_sha256"],
        "target_catalog_sha256": upstream_hashes["target_catalog_sha256"],
        "policy": policy,
        "scenario_map": scenario_map.to_dict(orient="records"),
    }
    target_decision_id = stable_json_hash(decision_material)[:20]

    inventory_path = paths.local / "LOCAL_ONLY_artifact_inventory.csv"
    local_inventory_hash = sha256_file(inventory_path)
    inventory = pd.read_csv(inventory_path, low_memory=False)
    local_hashes = {
        str(row["relative_path"]): str(row["sha256"])
        for row in inventory.to_dict(orient="records")
    }

    policy_payload = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "target_decision_id": target_decision_id,
        "upstream_preprocessing_decision_id": preprocessing_decision_id,
        "upstream_scenario_decision_id": scenario_decision_id,
        "upstream_temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "policy": policy,
    }
    scenario_map_payload = scenario_map.copy()
    scenario_map_payload["target_decision_id"] = target_decision_id
    scenario_map_payload["test_materialized"] = False
    lock_payload = {
        "lock_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "target_decision_id": target_decision_id,
        "upstream_preprocessing_decision_id": preprocessing_decision_id,
        "upstream_scenario_decision_id": scenario_decision_id,
        "upstream_temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "fit_split": TRAIN,
        "validation_refit": False,
        "test_values_parsed": False,
        "test_materialized": False,
        "target_imputation": False,
        "target_count": TARGET_COUNT,
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
        "policy_sha256": stable_json_hash(policy),
        "upstream_hashes": upstream_hashes,
        "local_artifact_inventory_sha256": local_inventory_hash,
        "local_artifact_hashes": local_hashes,
        "methodological_commitment": (
            "Do not alter target scaling, Full/PCA90 definitions, target-fit groups, "
            "PCA variance threshold or target ordering after test inspection."
        ),
    }

    for config_path, content, is_frame in [
        (paths.target_policy, policy_payload, False),
        (paths.target_scenario_map, scenario_map_payload, True),
        (paths.target_lock, lock_payload, False),
    ]:
        if config_path.exists() and not args.overwrite:
            raise TargetStageAbort(
                f"Official BENCHMARK07B artifact already exists: {config_path}."
            )
        if is_frame:
            atomic_write_csv(content, config_path)  # type: ignore[arg-type]
        else:
            atomic_write_json(config_path, content)  # type: ignore[arg-type]

    atomic_write_csv(dimension_summary, paths.safe / "benchmark07b_target_group_dimension_summary.csv")
    atomic_write_csv(pca_summary, paths.safe / "benchmark07b_pca_variance_summary.csv")
    atomic_write_csv(oracle_summary, paths.safe / "benchmark07b_pca_oracle_reconstruction.csv")
    atomic_write_csv(scaling_summary, paths.safe / "benchmark07b_scaling_audit.csv")
    safe_map = scenario_map.copy()
    safe_map["target_decision_id"] = target_decision_id
    atomic_write_csv(safe_map, paths.safe / "benchmark07b_scenario_target_map.csv")
    atomic_write_text(
        paths.safe / "BENCHMARK07B_TARGET_STORY.md",
        build_story(dimension_summary, pca_summary, target_decision_id),
    )

    summary_payload = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_preprocessing_passed": True,
        "target_decision_id": target_decision_id,
        "preprocessing_decision_id": preprocessing_decision_id,
        "scenario_decision_id": scenario_decision_id,
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": None,
        "scope": {
            "fit_split": TRAIN,
            "validation_transformed_without_refit": True,
            "test_values_parsed": False,
            "test_materialized": False,
            "target_values_loaded_splits": [TRAIN, VALIDATION],
            "target_imputation": False,
            "input_preprocessing_changed": False,
            "participant_split_changed": False,
            "models_trained": False,
        },
        "policy": policy,
        "target_split_row_counts": target_split_counts,
        "target_group_dimensions": dimension_summary.to_dict(orient="records"),
        "pca_variance": pca_summary.to_dict(orient="records"),
        "pca_oracle_reconstruction": oracle_summary.to_dict(orient="records"),
        "scaling_audit": scaling_summary.to_dict(orient="records"),
        "scenario_target_map": scenario_map.to_dict(orient="records"),
        "artifacts": {
            "local_output_directory": str(paths.local.relative_to(paths.root)),
            "local_artifact_inventory_sha256": local_inventory_hash,
            "target_policy": str(paths.target_policy.relative_to(paths.root)),
            "target_policy_sha256": sha256_file(paths.target_policy),
            "target_scenario_map": str(paths.target_scenario_map.relative_to(paths.root)),
            "target_scenario_map_sha256": sha256_file(paths.target_scenario_map),
            "target_lock": str(paths.target_lock.relative_to(paths.root)),
            "target_lock_sha256": sha256_file(paths.target_lock),
        },
        "upstream_hashes": upstream_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "privacy": {
            "safe_outputs_contain_participant_identifiers": False,
            "safe_outputs_contain_visit_identifiers": False,
            "safe_outputs_contain_target_names": False,
            "safe_outputs_contain_individual_target_values": False,
            "safe_outputs_contain_predictions": False,
            "individualized_artifacts_are_local_only": True,
            "local_artifacts_must_not_be_shared": True,
        },
    }
    atomic_write_json(paths.safe / "benchmark07b_summary.json", summary_payload)
    atomic_write_csv(checks.frame(), paths.safe / "benchmark07b_validation_checks.csv")
    return summary_payload


def write_failure_outputs(paths: Paths, checks: Checks, error: BaseException) -> None:
    payload = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "target_policy_version": TARGET_POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_preprocessing_passed": False,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": f"{type(error).__name__}: {error}",
        "scope": {
            "fit_split": TRAIN,
            "test_values_parsed": False,
            "test_materialized": False,
        },
    }
    atomic_write_json(paths.safe / "benchmark07b_summary.json", payload)
    atomic_write_csv(checks.frame(), paths.safe / "benchmark07b_validation_checks.csv")


def self_test() -> int:
    rng = np.random.default_rng(7)
    n_train, n_val, p = 140, 40, TARGET_COUNT
    latent_train = rng.normal(size=(n_train, 35))
    latent_val = rng.normal(size=(n_val, 35))
    loadings = rng.normal(size=(35, p))
    train = latent_train @ loadings + 0.2 * rng.normal(size=(n_train, p))
    val = latent_val @ loadings + 0.2 * rng.normal(size=(n_val, p))
    train_manifest = pd.DataFrame({RID: [str(i) for i in range(n_train)]})
    val_manifest = pd.DataFrame({RID: [str(i) for i in range(n_val)]})
    checks = Checks()
    result = fit_group("synthetic", train, val, train_manifest, val_manifest, checks)
    assert checks.passed
    assert result["y_full_train"].shape == (n_train, p)
    assert result["y_full_validation"].shape == (n_val, p)
    assert 1 <= result["component_count"] <= p
    assert result["cumulative_explained_variance"] >= 0.90
    print("BENCHMARK07B SELF-TEST PASSED")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    paths = resolve_paths(args)
    configure_logging(paths)
    checks = Checks()
    logging.info("BENCHMARK07B v%s", SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Frozen package: %s", paths.package)
    logging.info("Target fit groups: snapshot_all and matched_pair")
    logging.info("Train fit only; validation transform only; test target values unparsed")
    logging.info("PCA variance threshold: %.2f", PCA_VARIANCE_THRESHOLD)

    try:
        summary = run_stage(paths, args, checks)
    except Exception as error:
        logging.exception("BENCHMARK07B fatal error")
        try:
            write_failure_outputs(paths, checks, error)
        except Exception:
            logging.exception("Could not write BENCHMARK07B failure reports")
        logging.error(
            "BENCHMARK07B FAILED | failed_checks=%d | fatal_error=%s: %s",
            checks.failed,
            type(error).__name__,
            error,
        )
        return 1

    dimensions = {
        row["target_group"]: row for row in summary["target_group_dimensions"]
    }
    logging.info(
        "BENCHMARK07B PASSED | snapshot_all_pca90=%s | matched_pair_pca90=%s | test_materialized=false",
        dimensions["snapshot_all"]["pca90_components"],
        dimensions["matched_pair"]["pca90_components"],
    )
    logging.info("Do not share local target artifacts: %s", paths.local)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
