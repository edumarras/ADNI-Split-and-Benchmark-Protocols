#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK01B validates the frozen v1.1 package without training models.
# Safe outputs contain aggregate metadata and hashes only.



from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION: Final[str] = "0.2.0"

# Physical directory used by the current project. The authoritative package
# identity is validated from official_frozen_manifest.json, not from this name.
DEFAULT_PACKAGE_DIRNAME: Final[str] = "frozen_experiment_v1_1"
EXPECTED_PACKAGE_NAME: Final[str] = "frozen_experiment_v1_1"
EXPECTED_BASE_PACKAGE_NAME: Final[str] = "frozen_experiment_v1"
EXPECTED_EXTENSION_SCRIPT_VERSION: Final[str] = "0.1.0"
EXPECTED_BASE_SCRIPT_VERSION: Final[str] = "1.0.1"
EXPECTED_TEMPORAL_POLICY_NAME: Final[str] = (
    "strict_date_interval_same_wave_previous_visit_v1"
)
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"
EXPECTED_EXTENSION_STAGE: Final[str] = "split07_freeze_strict_date_temporal_extension"

SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")
INPUT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS", "CDR", "FAQ", "MMSE", "MOCA", "NEUROBAT", "PTDEMOG", "APOERES"
)
SOURCE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "ADAS": ("RID", "VISCODE2"),
    "CDR": ("RID", "VISCODE2"),
    "FAQ": ("RID", "VISCODE2"),
    "MMSE": ("RID", "VISCODE2"),
    "MOCA": ("RID", "VISCODE2"),
    "NEUROBAT": ("RID", "VISCODE2"),
    "PTDEMOG": ("RID",),
    "APOERES": ("RID",),
}

# The original v1 inventory is preserved inside v1.1 and remains a useful
# lineage invariant. The v1.1 inventory itself is intentionally NOT hardcoded
# by row count because SPLIT07 adds files; exactness is checked against the
# filesystem instead.
EXPECTED_BASE_INVENTORY_ROWS: Final[int] = 63
IGNORABLE_METADATA_NAMES: Final[frozenset[str]] = frozenset({
    "desktop.ini", "Thumbs.db", ".DS_Store"
})
EXPECTED_TARGETS: Final[int] = 323
EXPECTED_VISITS: Final[int] = 9_600
EXPECTED_PARTICIPANTS: Final[int] = 2_839
EXPECTED_PARTICIPANTS_BY_SPLIT: Final[dict[str, int]] = {
    "train": 1_987, "validation": 426, "test": 426
}
EXPECTED_VISITS_BY_SPLIT: Final[dict[str, int]] = {
    "train": 6_707, "validation": 1_445, "test": 1_448
}
EXPECTED_CONFIGURATION: Final[str] = "uniform__raw__mean_max_equal"
EXPECTED_CANDIDATE_ID: Final[int] = 10_185
EXPECTED_CANDIDATE_SEED: Final[int] = 42_010_184
EXPECTED_MAPPING_SHA256: Final[str] = (
    "6e5d6b401e3a66f2bf03f5d278fb3db174f794313b215a3bbed4a4ce0803658d"
)
EXPECTED_FINAL_ELIGIBLE: Final[dict[str, int]] = {
    "train": 1_426, "validation": 301
}


ARTIFACTS: Final[dict[str, str]] = {
    "manifest": "04_official_split/official_frozen_manifest.json",
    "inventory": "04_official_split/frozen_file_inventory.csv",
    "base_manifest": "04_official_split/official_frozen_manifest_v1_base.json",
    "base_inventory": "04_official_split/frozen_file_inventory_v1_base.csv",
    "temporal_policy_lock": "04_official_split/strict_date_temporal_policy_lock.json",
    "temporal_date_profiles": "03_final_ready_tables/development_visit_date_profiles.csv.gz",
    "temporal_pairing_reference": "02_cuts_and_adjustments/development_strict_date_pairing_reference.csv.gz",
    "temporal_readme": "README_V1_1_TEMPORAL_EXTENSION_LOCAL_ONLY.txt",
    "split_validation": "04_official_split/official_split_validation.txt",
    "participant_split": "04_official_split/official_participant_split.csv",
    "visit_split": "04_official_split/official_visit_split.csv.gz",
    "anchor": "03_final_ready_tables/supervised_anchor_visits.csv.gz",
    "targets": "03_final_ready_tables/UCSFFSX7_targets_323.csv.gz",
    "target_catalog": "03_final_ready_tables/MRI_target_catalog_323.csv",
    "feature_catalog": "03_final_ready_tables/input_feature_catalog.csv",
}


PACKAGE_DIRS: Final[tuple[str, ...]] = (
    "01_postprocessed_sources",
    "02_cuts_and_adjustments",
    "03_final_ready_tables",
    "04_official_split",
)

TARGET_META: Final[tuple[str, ...]] = (
    "RID", "VISCODE2", "split", "eligible_visit_position"
)


class ValidationAbort(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    safe: Path
    logs: Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: Any
    observed: Any
    details: str = ""


@dataclass(frozen=True)
class InventoryResult:
    relative_path: str
    exists: bool
    size_matches: bool
    sha256_matches: bool
    expected_size_bytes: int | None
    observed_size_bytes: int | None
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True)
class SourceResult:
    source_name: str
    rows: int
    columns: int
    features: int
    missing_key_rows: int
    duplicate_key_rows: int
    invalid_split_rows: int
    unofficial_participant_rows: int
    split_mismatch_rows: int
    catalog_matches_table: bool
    passed: bool


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failed_check_count: int
    fatal_error: str | None
    aggregate: dict[str, Any]
    checks: tuple[Check, ...]
    inventory: tuple[InventoryResult, ...]
    sources: tuple[SourceResult, ...]
    safe_dir: Path


class Checks:
    def __init__(self) -> None:
        self.items: list[Check] = []

    def add(
        self,
        name: str,
        passed: bool,
        expected: Any,
        observed: Any,
        details: str = "",
    ) -> bool:
        item = Check(name, bool(passed), expected, observed, details)
        self.items.append(item)
        logging.log(
            logging.INFO if passed else logging.ERROR,
            "%s | %s",
            "PASS" if passed else "FAIL",
            name,
        )
        if not passed:
            expected_text = str(expected)
            observed_text = str(observed)
            if len(expected_text) > 600:
                expected_text = expected_text[:597] + "..."
            if len(observed_text) > 600:
                observed_text = observed_text[:597] + "..."
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                name,
                expected_text,
                observed_text,
                f" | {details}" if details else "",
            )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path.cwd().resolve()
    package = (
        args.package_root.expanduser().resolve()
        if args.package_root
        else root / "data" / "processed" / DEFAULT_PACKAGE_DIRNAME
    )
    reports = (
        args.reports_root.expanduser().resolve()
        if args.reports_root
        else root / "reports"
    )
    stage = reports / "benchmark01b_validate_frozen_package"
    safe = stage / "safe"
    logs = stage / "logs"
    safe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return Paths(root=root, package=package, safe=safe, logs=logs)


def configure_logging(paths: Paths) -> None:
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(paths.logs / "benchmark01b.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(text)


def catalog_sha256(columns: Sequence[str]) -> str:
    canonical = [str(column).strip().upper() for column in columns]
    return sha256_text("\n".join(canonical))


def read_text_csv(
    path: Path,
    usecols: Sequence[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        nrows=nrows,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    return frame


def read_header(path: Path) -> list[str]:
    return list(read_text_csv(path, nrows=0).columns)


def clean_text(series: pd.Series, lower: bool = False) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.str.lower() if lower else result


def parse_validation_text(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def json_cell(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
    checks: Checks,
) -> None:
    required_list = list(required)
    missing = [column for column in required_list if column not in frame.columns]
    checks.add(f"{label}__required_columns", not missing, required_list, list(frame.columns))
    if missing:
        raise ValidationAbort(f"Invalid schema in {label}; missing-column count={len(missing)}")


def mapping_sha256(participant_split: pd.DataFrame) -> str:
    work = participant_split.loc[:, ["RID", "split"]].copy()
    work["RID"] = clean_text(work["RID"])
    work["split"] = clean_text(work["split"], lower=True)
    text = "\n".join(f"{row.RID},{row.split}" for row in work.itertuples(index=False)) + "\n"
    return sha256_text(text)


def visit_key_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            clean_text(frame["RID"]).astype(str),
            clean_text(frame["VISCODE2"], lower=True).astype(str),
            strict=True,
        )
    )


def locate_artifacts(paths: Paths, checks: Checks) -> dict[str, Path]:
    checks.add("package_root_exists", paths.package.is_dir(), True, paths.package.is_dir())
    if not paths.package.is_dir():
        raise ValidationAbort(f"Frozen package directory was not found: {paths.package}")

    for name in PACKAGE_DIRS:
        checks.add(f"package_directory__{name}", (paths.package / name).is_dir(), True, (paths.package / name).is_dir())

    artifacts = {name: paths.package / rel for name, rel in ARTIFACTS.items()}
    missing = 0
    for name, path in artifacts.items():
        exists = path.is_file()
        checks.add(f"artifact__{name}", exists, True, exists, ARTIFACTS[name])
        missing += int(not exists)

    for source in INPUT_SOURCES:
        rel = f"03_final_ready_tables/input_sources/{source}_model_ready.csv.gz"
        path = paths.package / rel
        artifacts[f"input__{source}"] = path
        exists = path.is_file()
        checks.add(f"artifact__input__{source}", exists, True, exists, rel)
        missing += int(not exists)

    if missing:
        raise ValidationAbort(f"Missing critical-artifact count={missing}")
    return artifacts


def validate_inventory(
    paths: Paths,
    artifacts: dict[str, Path],
    checks: Checks,
) -> tuple[list[InventoryResult], dict[str, Any]]:
    """Validate the v1.1 inventory and v1 lineage without hardcoding v1.1 file counts."""
    inventory = read_text_csv(artifacts["inventory"])
    require_columns(
        inventory,
        (
            "category", "relative_path", "size_bytes", "sha256",
            "origin", "base_v1_sha256", "unchanged_from_v1",
        ),
        "inventory",
        checks,
    )
    inventory = inventory.loc[:, [
        "category", "relative_path", "size_bytes", "sha256",
        "origin", "base_v1_sha256", "unchanged_from_v1",
    ]].copy()
    inventory["relative_path"] = clean_text(inventory["relative_path"])
    inventory["sha256"] = clean_text(inventory["sha256"], lower=True)
    inventory["base_v1_sha256"] = clean_text(inventory["base_v1_sha256"], lower=True)
    inventory["origin"] = clean_text(inventory["origin"])
    inventory["unchanged_from_v1"] = clean_text(inventory["unchanged_from_v1"], lower=True)
    inventory["size_bytes"] = pd.to_numeric(inventory["size_bytes"], errors="coerce").astype("Int64")

    duplicate_paths = int(inventory["relative_path"].duplicated().sum())
    unsafe_paths = sum(
        PurePosixPath(str(value)).is_absolute() or ".." in PurePosixPath(str(value)).parts
        for value in inventory["relative_path"]
    )
    invalid_hashes = int((~inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}", na=False)).sum())
    invalid_sizes = int(inventory["size_bytes"].isna().sum())
    invalid_origin = int((~inventory["origin"].isin(("unchanged_from_v1", "v1_1_temporal_extension"))).sum())
    invalid_unchanged_flag = int((~inventory["unchanged_from_v1"].isin(("true", "false"))).sum())
    checks.add("inventory_unique_paths", duplicate_paths == 0, 0, duplicate_paths)
    checks.add("inventory_safe_relative_paths", unsafe_paths == 0, 0, unsafe_paths)
    checks.add("inventory_sha256_format", invalid_hashes == 0, 0, invalid_hashes)
    checks.add("inventory_size_format", invalid_sizes == 0, 0, invalid_sizes)
    checks.add("inventory_origin_domain", invalid_origin == 0, 0, invalid_origin)
    checks.add("inventory_unchanged_flag_domain", invalid_unchanged_flag == 0, 0, invalid_unchanged_flag)

    results: list[InventoryResult] = []
    for row in inventory.itertuples(index=False):
        relative = str(row.relative_path)
        path = paths.package / PurePosixPath(relative)
        exists = path.is_file()
        observed_size = int(path.stat().st_size) if exists else None
        observed_sha = sha256_file(path) if exists else ""
        expected_size = int(row.size_bytes) if not pd.isna(row.size_bytes) else None
        expected_sha = str(row.sha256)
        results.append(
            InventoryResult(
                relative_path=relative,
                exists=exists,
                size_matches=exists and observed_size == expected_size,
                sha256_matches=exists and observed_sha == expected_sha,
                expected_size_bytes=expected_size,
                observed_size_bytes=observed_size,
                expected_sha256=expected_sha,
                observed_sha256=observed_sha,
            )
        )

    missing = sum(not item.exists for item in results)
    size_errors = sum(not item.size_matches for item in results)
    hash_errors = sum(not item.sha256_matches for item in results)
    checks.add("inventory_all_files_exist", missing == 0, 0, missing)
    checks.add("inventory_all_sizes_match", size_errors == 0, 0, size_errors)
    checks.add("inventory_all_hashes_match", hash_errors == 0, 0, hash_errors)

    # Exact filesystem contract: SPLIT07 intentionally excludes only the current
    # inventory and current manifest from the inventory it writes.
    all_actual_files = [path for path in paths.package.rglob("*") if path.is_file()]
    ignored_metadata_files = [
        path for path in all_actual_files
        if path.name in IGNORABLE_METADATA_NAMES
        or "__MACOSX" in path.relative_to(paths.package).parts
    ]
    ignored_metadata_set = set(ignored_metadata_files)
    actual_files = [path for path in all_actual_files if path not in ignored_metadata_set]
    actual_rel = {path.relative_to(paths.package).as_posix() for path in actual_files}
    ignored_rel = sorted(path.relative_to(paths.package).as_posix() for path in ignored_metadata_files)
    listed_rel = set(inventory["relative_path"].astype(str))
    expected_unlisted = {
        artifacts["inventory"].relative_to(paths.package).as_posix(),
        artifacts["manifest"].relative_to(paths.package).as_posix(),
    }
    unlisted = actual_rel - listed_rel
    missing_listed = listed_rel - actual_rel
    checks.add(
        "inventory_expected_unlisted_files",
        unlisted == expected_unlisted,
        sorted(expected_unlisted),
        sorted(unlisted),
        "v1.1 inventory intentionally excludes only itself and the current manifest.",
    )
    checks.add("inventory_no_missing_listed_files", not missing_listed, [], sorted(missing_listed))

    # Validate preserved v1 inventory and continuity of every v1 content file.
    base_inventory = read_text_csv(artifacts["base_inventory"])
    require_columns(base_inventory, ("category", "relative_path", "size_bytes", "sha256"), "base_inventory", checks)
    base_inventory = base_inventory.loc[:, ["category", "relative_path", "size_bytes", "sha256"]].copy()
    base_inventory["relative_path"] = clean_text(base_inventory["relative_path"])
    base_inventory["sha256"] = clean_text(base_inventory["sha256"], lower=True)
    base_inventory["size_bytes"] = pd.to_numeric(base_inventory["size_bytes"], errors="coerce").astype("Int64")
    checks.add(
        "base_inventory_exact_row_count",
        len(base_inventory) == EXPECTED_BASE_INVENTORY_ROWS,
        EXPECTED_BASE_INVENTORY_ROWS,
        len(base_inventory),
    )
    checks.add(
        "base_inventory_unique_paths",
        not base_inventory["relative_path"].duplicated().any(),
        0,
        int(base_inventory["relative_path"].duplicated().sum()),
    )

    current_by_path = inventory.set_index("relative_path", drop=False)
    base_missing = 0
    base_hash_mismatch = 0
    lineage_metadata_mismatch = 0
    for row in base_inventory.itertuples(index=False):
        relative = str(row.relative_path)
        expected_sha = str(row.sha256)
        current_path = paths.package / PurePosixPath(relative)
        if not current_path.is_file():
            base_missing += 1
            continue
        actual_sha = sha256_file(current_path)
        if actual_sha != expected_sha:
            base_hash_mismatch += 1
        if relative not in current_by_path.index:
            lineage_metadata_mismatch += 1
            continue
        current_row = current_by_path.loc[relative]
        # Paths are unique, so loc returns a Series.
        base_sha_cell = str(current_row["base_v1_sha256"]).strip().lower()
        unchanged_cell = str(current_row["unchanged_from_v1"]).strip().lower()
        origin_cell = str(current_row["origin"]).strip()
        if not (
            base_sha_cell == expected_sha
            and unchanged_cell == "true"
            and origin_cell == "unchanged_from_v1"
            and str(current_row["sha256"]).strip().lower() == expected_sha
        ):
            lineage_metadata_mismatch += 1

    checks.add("base_v1_listed_files_present", base_missing == 0, 0, base_missing)
    checks.add("base_v1_listed_files_byte_identical", base_hash_mismatch == 0, 0, base_hash_mismatch)
    checks.add("base_v1_lineage_metadata_consistent", lineage_metadata_mismatch == 0, 0, lineage_metadata_mismatch)

    if ignored_rel:
        logging.warning("Ignored operating-system metadata inside frozen package: %s", ignored_rel)

    return results, {
        "actual_file_count_including_ignored_metadata": len(all_actual_files),
        "content_file_count": len(actual_files),
        "ignored_metadata_file_count": len(ignored_rel),
        "ignored_metadata_relative_paths": ignored_rel,
        "inventory_row_count": int(len(inventory)),
        "base_inventory_row_count": int(len(base_inventory)),
        "unlisted_content_file_count": len(unlisted),
        "unlisted_content_relative_paths": sorted(unlisted),
        "missing_listed_file_count": len(missing_listed),
        "base_v1_missing_file_count": base_missing,
        "base_v1_hash_mismatch_count": base_hash_mismatch,
        "base_v1_lineage_metadata_mismatch_count": lineage_metadata_mismatch,
        "inventory_sha256": sha256_file(artifacts["inventory"]),
        "manifest_sha256": sha256_file(artifacts["manifest"]),
        "base_inventory_sha256": sha256_file(artifacts["base_inventory"]),
        "base_manifest_sha256": sha256_file(artifacts["base_manifest"]),
    }


def validate_base_manifest(
    base_manifest: dict[str, Any],
    split_text: dict[str, str],
    checks: Checks,
) -> dict[str, Any]:
    checks.add("base_manifest_package_name", base_manifest.get("frozen_package_name") == EXPECTED_BASE_PACKAGE_NAME, EXPECTED_BASE_PACKAGE_NAME, base_manifest.get("frozen_package_name"))
    checks.add("base_manifest_source_script_version", base_manifest.get("script_version") == EXPECTED_BASE_SCRIPT_VERSION, EXPECTED_BASE_SCRIPT_VERSION, base_manifest.get("script_version"))
    checks.add("base_manifest_split_frozen", base_manifest.get("official_split_frozen") is True, True, base_manifest.get("official_split_frozen"))

    privacy = base_manifest.get("privacy", {})
    checks.add("base_manifest_local_only", privacy.get("package_is_local_only") is True, True, privacy.get("package_is_local_only"))
    checks.add("base_manifest_contains_individualized_records", privacy.get("contains_individualized_records") is True, True, privacy.get("contains_individualized_records"))
    checks.add("base_manifest_must_not_be_shared", privacy.get("must_not_be_committed_or_shared") is True, True, privacy.get("must_not_be_committed_or_shared"))

    population = base_manifest.get("population", {})
    checks.add("base_manifest_targets", population.get("supervised_target_count") == EXPECTED_TARGETS, EXPECTED_TARGETS, population.get("supervised_target_count"))
    checks.add("base_manifest_visits", population.get("supervised_visits") == EXPECTED_VISITS, EXPECTED_VISITS, population.get("supervised_visits"))
    checks.add("base_manifest_participants", population.get("participants") == EXPECTED_PARTICIPANTS, EXPECTED_PARTICIPANTS, population.get("participants"))

    config = base_manifest.get("official_configuration", {})
    checks.add("base_manifest_weight_scheme", config.get("weight_scheme") == "uniform", "uniform", config.get("weight_scheme"))
    checks.add("base_manifest_distance_scale", config.get("distance_scale") == "raw", "raw", config.get("distance_scale"))
    checks.add("base_manifest_candidate_pool", config.get("candidate_pool") == 100_000, 100_000, config.get("candidate_pool"))

    selection_root = base_manifest.get("selection", {})
    selection = selection_root.get("selection", {})
    validation = selection_root.get("validation", {})
    checks.add("base_manifest_candidate_id", selection.get("candidate_id") == EXPECTED_CANDIDATE_ID, EXPECTED_CANDIDATE_ID, selection.get("candidate_id"))
    checks.add("base_manifest_candidate_seed", selection.get("seed") == EXPECTED_CANDIDATE_SEED, EXPECTED_CANDIDATE_SEED, selection.get("seed"))
    checks.add("base_manifest_mapping_sha256", selection.get("split_mapping_sha256") == EXPECTED_MAPPING_SHA256, EXPECTED_MAPPING_SHA256, selection.get("split_mapping_sha256"))
    checks.add("base_manifest_participant_counts", validation.get("participant_counts") == EXPECTED_PARTICIPANTS_BY_SPLIT, EXPECTED_PARTICIPANTS_BY_SPLIT, validation.get("participant_counts"))
    checks.add("base_manifest_visit_counts", validation.get("visit_counts") == EXPECTED_VISITS_BY_SPLIT, EXPECTED_VISITS_BY_SPLIT, validation.get("visit_counts"))
    expected_overlap = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    checks.add("base_manifest_participant_overlap", validation.get("participant_overlaps") == expected_overlap, expected_overlap, validation.get("participant_overlaps"))

    checks.add("split_text_frozen", split_text.get("official_split_frozen") == "true", "true", split_text.get("official_split_frozen"))
    checks.add("split_text_configuration", split_text.get("official_configuration") == EXPECTED_CONFIGURATION, EXPECTED_CONFIGURATION, split_text.get("official_configuration"))
    checks.add("split_text_candidate", split_text.get("candidate_id") == str(EXPECTED_CANDIDATE_ID), str(EXPECTED_CANDIDATE_ID), split_text.get("candidate_id"))
    checks.add("split_text_seed", split_text.get("seed") == str(EXPECTED_CANDIDATE_SEED), str(EXPECTED_CANDIDATE_SEED), split_text.get("seed"))
    checks.add("split_text_mapping_sha256", split_text.get("split_mapping_sha256") == EXPECTED_MAPPING_SHA256, EXPECTED_MAPPING_SHA256, split_text.get("split_mapping_sha256"))
    checks.add("split_text_target_count", split_text.get("target_count") == str(EXPECTED_TARGETS), str(EXPECTED_TARGETS), split_text.get("target_count"))
    checks.add("split_text_allocation_unit", split_text.get("allocation_unit") == "participant", "participant", split_text.get("allocation_unit"))
    checks.add("split_text_targets_not_used", split_text.get("target_values_used_for_split_selection") == "false", "false", split_text.get("target_values_used_for_split_selection"))
    checks.add("split_text_models_not_used", split_text.get("model_performance_used_for_split_selection") == "false", "false", split_text.get("model_performance_used_for_split_selection"))

    return {
        "source_script_version": base_manifest.get("script_version"),
        "candidate_id": selection.get("candidate_id"),
        "candidate_seed": selection.get("seed"),
        "target_catalog_sha256": base_manifest.get("target_catalog_sha256"),
    }


def validate_extension_manifest(
    manifest: dict[str, Any],
    base_manifest: dict[str, Any],
    artifacts: dict[str, Path],
    checks: Checks,
) -> dict[str, Any]:
    checks.add("manifest_package_name", manifest.get("frozen_package_name") == EXPECTED_PACKAGE_NAME, EXPECTED_PACKAGE_NAME, manifest.get("frozen_package_name"))
    checks.add("manifest_base_package_name", manifest.get("base_frozen_package_name") == EXPECTED_BASE_PACKAGE_NAME, EXPECTED_BASE_PACKAGE_NAME, manifest.get("base_frozen_package_name"))
    checks.add("manifest_source_script_version", manifest.get("script_version") == EXPECTED_EXTENSION_SCRIPT_VERSION, EXPECTED_EXTENSION_SCRIPT_VERSION, manifest.get("script_version"))
    checks.add("manifest_split_frozen", manifest.get("official_split_frozen") is True, True, manifest.get("official_split_frozen"))
    checks.add("manifest_participant_split_unchanged", manifest.get("participant_split_changed") is False, False, manifest.get("participant_split_changed"))
    checks.add("manifest_population_unchanged", manifest.get("supervised_population_changed") is False, False, manifest.get("supervised_population_changed"))
    checks.add("manifest_targets_unchanged", manifest.get("mri_targets_changed") is False, False, manifest.get("mri_targets_changed"))
    checks.add("manifest_model_ready_sources_unchanged", manifest.get("model_ready_sources_changed") is False, False, manifest.get("model_ready_sources_changed"))

    actual_base_manifest_sha = sha256_file(artifacts["base_manifest"])
    checks.add("manifest_base_manifest_sha256", manifest.get("base_manifest_sha256") == actual_base_manifest_sha, actual_base_manifest_sha, manifest.get("base_manifest_sha256"))
    checks.add("manifest_base_population_matches", manifest.get("base_population") == base_manifest.get("population"), base_manifest.get("population"), manifest.get("base_population"))
    checks.add("manifest_base_selection_matches", manifest.get("base_selection") == base_manifest.get("selection"), "preserved base selection", "match" if manifest.get("base_selection") == base_manifest.get("selection") else "mismatch")

    privacy = manifest.get("privacy", {})
    checks.add("manifest_local_only", privacy.get("package_is_local_only") is True, True, privacy.get("package_is_local_only"))
    checks.add("manifest_contains_individualized_records", privacy.get("contains_individualized_records") is True, True, privacy.get("contains_individualized_records"))
    checks.add("manifest_contains_clinical_dates", privacy.get("contains_clinical_dates") is True, True, privacy.get("contains_clinical_dates"))
    checks.add("manifest_must_not_be_shared", privacy.get("must_not_be_committed_or_shared") is True, True, privacy.get("must_not_be_committed_or_shared"))

    temporal = manifest.get("temporal_extension", {})
    checks.add("manifest_temporal_policy_id", temporal.get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, temporal.get("temporal_policy_decision_id"))
    checks.add("manifest_temporal_policy_name", temporal.get("policy") == EXPECTED_TEMPORAL_POLICY_NAME, EXPECTED_TEMPORAL_POLICY_NAME, temporal.get("policy"))
    checks.add("manifest_temporal_development_splits", temporal.get("development_splits_used") == ["train", "validation"], ["train", "validation"], temporal.get("development_splits_used"))
    checks.add("manifest_temporal_test_values_unused", temporal.get("test_values_used") is False, False, temporal.get("test_values_used"))
    checks.add("manifest_temporal_test_pairings_unmaterialized", temporal.get("test_pairings_materialized") is False, False, temporal.get("test_pairings_materialized"))
    checks.add("manifest_temporal_mri_targets_unloaded", temporal.get("mri_target_values_loaded") is False, False, temporal.get("mri_target_values_loaded"))
    checks.add("manifest_temporal_visits_not_concatenated", temporal.get("visits_concatenated") is False, False, temporal.get("visits_concatenated"))
    checks.add("manifest_temporal_final_eligible", temporal.get("development_final_eligible_participants") == EXPECTED_FINAL_ELIGIBLE, EXPECTED_FINAL_ELIGIBLE, temporal.get("development_final_eligible_participants"))

    critical = manifest.get("critical_unchanged_sha256", {})
    critical_paths = {
        "official_participant_split.csv": artifacts["participant_split"],
        "supervised_anchor_visits.csv.gz": artifacts["anchor"],
        "UCSFFSX7_targets_323.csv.gz": artifacts["targets"],
    }
    critical_mismatch = 0
    for label, path in critical_paths.items():
        expected = critical.get(label)
        observed = sha256_file(path)
        ok = expected == observed
        critical_mismatch += int(not ok)
        checks.add(f"manifest_critical_hash__{label}", ok, expected, observed)

    artifact_hashes = temporal.get("artifact_sha256", {})
    temporal_artifact_paths = {
        "development_visit_date_profiles": artifacts["temporal_date_profiles"],
        "development_strict_date_pairing_reference": artifacts["temporal_pairing_reference"],
        "strict_date_temporal_policy_lock": artifacts["temporal_policy_lock"],
        "readme_v1_1_extension": artifacts["temporal_readme"],
    }
    temporal_hash_mismatch = 0
    for label, path in temporal_artifact_paths.items():
        expected = artifact_hashes.get(label)
        observed = sha256_file(path)
        ok = expected == observed
        temporal_hash_mismatch += int(not ok)
        checks.add(f"manifest_temporal_artifact_hash__{label}", ok, expected, observed)

    return {
        "logical_package_name": manifest.get("frozen_package_name"),
        "base_package_name": manifest.get("base_frozen_package_name"),
        "source_script_version": manifest.get("script_version"),
        "temporal_policy_decision_id": temporal.get("temporal_policy_decision_id"),
        "temporal_policy_name": temporal.get("policy"),
        "critical_hash_mismatch_count": critical_mismatch,
        "temporal_artifact_hash_mismatch_count": temporal_hash_mismatch,
    }


def validate_temporal_policy_lock(
    policy: dict[str, Any],
    manifest: dict[str, Any],
    artifacts: dict[str, Path],
    checks: Checks,
) -> dict[str, Any]:
    checks.add("policy_stage", policy.get("stage") == EXPECTED_EXTENSION_STAGE, EXPECTED_EXTENSION_STAGE, policy.get("stage"))
    checks.add("policy_script_version", policy.get("script_version") == EXPECTED_EXTENSION_SCRIPT_VERSION, EXPECTED_EXTENSION_SCRIPT_VERSION, policy.get("script_version"))
    checks.add("policy_name", policy.get("policy_name") == EXPECTED_TEMPORAL_POLICY_NAME, EXPECTED_TEMPORAL_POLICY_NAME, policy.get("policy_name"))
    checks.add("policy_decision_id", policy.get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, policy.get("temporal_policy_decision_id"))

    observed_id = policy.get("temporal_policy_decision_id")
    reconstruction = policy.get("reconstruction_provenance", {})
    canonical_payload = dict(policy)
    canonical_payload.pop("temporal_policy_decision_id", None)
    canonical_payload.pop("reconstruction_provenance", None)
    recomputed_payload_sha256 = canonical_json_sha256(canonical_payload)
    declared_payload_sha256 = reconstruction.get("current_policy_payload_sha256")
    checks.add(
        "policy_reconstruction_payload_sha256",
        recomputed_payload_sha256 == declared_payload_sha256,
        declared_payload_sha256,
        recomputed_payload_sha256,
    )
    checks.add(
        "policy_historical_decision_id_preserved",
        reconstruction.get("historical_decision_id_preserved") is True,
        True,
        reconstruction.get("historical_decision_id_preserved"),
    )
    manifest_reconstruction_sha = manifest.get("temporal_extension", {}).get(
        "reconstruction_policy_payload_sha256"
    )
    checks.add(
        "policy_manifest_reconstruction_sha256",
        declared_payload_sha256 == manifest_reconstruction_sha,
        manifest_reconstruction_sha,
        declared_payload_sha256,
    )

    scope = policy.get("policy_scope", {})
    checks.add("policy_scope_decision_splits", scope.get("decision_splits") == ["train", "validation"], ["train", "validation"], scope.get("decision_splits"))
    checks.add("policy_scope_test_values_unused", scope.get("test_values_used") is False, False, scope.get("test_values_used"))
    checks.add("policy_scope_mri_targets_unloaded", scope.get("mri_target_values_loaded") is False, False, scope.get("mri_target_values_loaded"))
    checks.add("policy_scope_split_unchanged", scope.get("participant_split_changed") is False, False, scope.get("participant_split_changed"))
    checks.add("policy_scope_population_unchanged", scope.get("supervised_population_changed") is False, False, scope.get("supervised_population_changed"))

    rules = policy.get("rules", {})
    checks.add("policy_rules_exact_visit_key", rules.get("exact_visit_key") == ["RID", "VISCODE2"], ["RID", "VISCODE2"], rules.get("exact_visit_key"))
    checks.add("policy_rules_visits_not_concatenated", rules.get("visits_concatenated") is False, False, rules.get("visits_concatenated"))
    checks.add("policy_rules_no_median_fallback", rules.get("median_date_fallback") is False, False, rules.get("median_date_fallback"))
    checks.add("policy_rules_no_viscode_hierarchy", rules.get("viscode_hierarchy_fallback") is False, False, rules.get("viscode_hierarchy_fallback"))
    checks.add("policy_rules_no_lexical_fallback", rules.get("lexical_fallback") is False, False, rules.get("lexical_fallback"))
    checks.add("policy_rules_previous_mri_not_required", rules.get("previous_mri_required") is False, False, rules.get("previous_mri_required"))
    checks.add("policy_rules_previous_target_complete_not_required", rules.get("previous_target_complete_required") is False, False, rules.get("previous_target_complete_required"))

    evidence = policy.get("frozen_development_evidence", {})
    checks.add("policy_final_eligible", evidence.get("final_eligible") == EXPECTED_FINAL_ELIGIBLE, EXPECTED_FINAL_ELIGIBLE, evidence.get("final_eligible"))

    upstream = policy.get("upstream", {})
    checks.add("policy_upstream_base_package", upstream.get("base_package") == EXPECTED_BASE_PACKAGE_NAME, EXPECTED_BASE_PACKAGE_NAME, upstream.get("base_package"))
    split_sha = sha256_file(artifacts["participant_split"])
    anchor_sha = sha256_file(artifacts["anchor"])
    checks.add("policy_upstream_split_hash", upstream.get("official_participant_split_sha256") == split_sha, split_sha, upstream.get("official_participant_split_sha256"))
    checks.add("policy_upstream_anchor_hash", upstream.get("supervised_anchor_sha256") == anchor_sha, anchor_sha, upstream.get("supervised_anchor_sha256"))

    manifest_temporal = manifest.get("temporal_extension", {})
    checks.add("policy_manifest_id_agreement", policy.get("temporal_policy_decision_id") == manifest_temporal.get("temporal_policy_decision_id"), manifest_temporal.get("temporal_policy_decision_id"), policy.get("temporal_policy_decision_id"))
    checks.add("policy_manifest_name_agreement", policy.get("policy_name") == manifest_temporal.get("policy"), manifest_temporal.get("policy"), policy.get("policy_name"))

    return {
        "temporal_policy_decision_id": policy.get("temporal_policy_decision_id"),
        "policy_name": policy.get("policy_name"),
        "recomputed_policy_payload_sha256": recomputed_payload_sha256,
        "final_eligible": evidence.get("final_eligible"),
    }


def validate_core(
    artifacts: dict[str, Path],
    base_manifest: dict[str, Any],
    checks: Checks,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    participants = read_text_csv(artifacts["participant_split"])
    visits = read_text_csv(artifacts["visit_split"])
    anchor = read_text_csv(artifacts["anchor"], usecols=("RID", "VISCODE2", "split", "eligible_visit_position"))
    target_catalog = read_text_csv(artifacts["target_catalog"])
    feature_catalog = read_text_csv(artifacts["feature_catalog"])

    require_columns(participants, ("RID", "split"), "participant_split", checks)
    require_columns(visits, ("RID", "VISCODE2", "split", "eligible_visit_position"), "visit_split", checks)
    require_columns(anchor, ("RID", "VISCODE2", "split", "eligible_visit_position"), "anchor", checks)
    require_columns(target_catalog, ("target_position_1_based", "target_column"), "target_catalog", checks)
    require_columns(feature_catalog, ("source_name", "feature_position_within_source", "column_name", "role", "final_train_only_removal_pending"), "feature_catalog", checks)

    for frame in (participants, visits, anchor):
        frame["RID"] = clean_text(frame["RID"])
        frame["split"] = clean_text(frame["split"], lower=True)
        if "VISCODE2" in frame:
            frame["VISCODE2"] = clean_text(frame["VISCODE2"], lower=True)

    missing_rid = int(participants["RID"].eq("").sum())
    duplicate_rid = int(participants["RID"].duplicated().sum())
    invalid_participant_split = int((~participants["split"].isin(SPLITS)).sum())
    participant_counts = {name: int(participants["split"].eq(name).sum()) for name in SPLITS}
    checks.add("participant_missing_rid", missing_rid == 0, 0, missing_rid)
    checks.add("participant_unique_rid", duplicate_rid == 0, 0, duplicate_rid)
    checks.add("participant_valid_split", invalid_participant_split == 0, 0, invalid_participant_split)
    checks.add("participant_total", len(participants) == EXPECTED_PARTICIPANTS, EXPECTED_PARTICIPANTS, len(participants))
    checks.add("participant_counts", participant_counts == EXPECTED_PARTICIPANTS_BY_SPLIT, EXPECTED_PARTICIPANTS_BY_SPLIT, participant_counts)

    computed_mapping_sha = mapping_sha256(participants)
    manifest_mapping_sha = base_manifest.get("selection", {}).get("selection", {}).get("split_mapping_sha256")
    checks.add("participant_mapping_sha256_exact", computed_mapping_sha == EXPECTED_MAPPING_SHA256, EXPECTED_MAPPING_SHA256, computed_mapping_sha)
    checks.add("participant_mapping_sha256_manifest", computed_mapping_sha == manifest_mapping_sha, manifest_mapping_sha, computed_mapping_sha)

    participant_map = participants.set_index("RID")["split"]
    for label, frame in (("visit_split", visits), ("anchor", anchor)):
        missing_key = int((frame["RID"].eq("") | frame["VISCODE2"].eq("")).sum())
        duplicate_key = int(frame.duplicated(["RID", "VISCODE2"]).sum())
        invalid_split = int((~frame["split"].isin(SPLITS)).sum())
        mapped = frame["RID"].map(participant_map)
        unofficial = int(mapped.isna().sum())
        mismatch = int((mapped.notna() & mapped.ne(frame["split"])).sum())
        checks.add(f"{label}_missing_key", missing_key == 0, 0, missing_key)
        checks.add(f"{label}_duplicate_key", duplicate_key == 0, 0, duplicate_key)
        checks.add(f"{label}_valid_split", invalid_split == 0, 0, invalid_split)
        checks.add(f"{label}_total", len(frame) == EXPECTED_VISITS, EXPECTED_VISITS, len(frame))
        checks.add(f"{label}_official_participants", unofficial == 0, 0, unofficial)
        checks.add(f"{label}_split_consistency", mismatch == 0, 0, mismatch)
        numeric_position = pd.to_numeric(frame["eligible_visit_position"], errors="coerce")
        checks.add(f"{label}_numeric_position", int(numeric_position.isna().sum()) == 0, 0, int(numeric_position.isna().sum()))
        frame["eligible_visit_position"] = numeric_position.astype("Int64")

    position_failures = 0
    for _, group in anchor.groupby("RID", sort=False):
        values = sorted(int(value) for value in group["eligible_visit_position"].dropna())
        position_failures += int(values != list(range(1, len(group) + 1)))
    checks.add("anchor_consecutive_positions", position_failures == 0, 0, position_failures)

    anchor_keys = visit_key_set(anchor)
    visit_keys = visit_key_set(visits)
    checks.add("anchor_visit_split_same_keys", anchor_keys == visit_keys, EXPECTED_VISITS, len(anchor_keys & visit_keys), "Observed is common-key count")

    anchor_indexed = anchor.set_index(["RID", "VISCODE2"]).sort_index()
    visits_indexed = visits.set_index(["RID", "VISCODE2"]).sort_index()
    common = anchor_indexed.index.intersection(visits_indexed.index)
    split_mismatch = int(anchor_indexed.loc[common, "split"].ne(visits_indexed.loc[common, "split"]).sum())
    position_mismatch = int(anchor_indexed.loc[common, "eligible_visit_position"].ne(visits_indexed.loc[common, "eligible_visit_position"]).sum())
    checks.add("anchor_visit_split_same_labels", split_mismatch == 0, 0, split_mismatch)
    checks.add("anchor_visit_split_same_positions", position_mismatch == 0, 0, position_mismatch)

    target_catalog["target_position_1_based"] = pd.to_numeric(target_catalog["target_position_1_based"], errors="coerce").astype("Int64")
    target_catalog["target_column"] = clean_text(target_catalog["target_column"])
    positions = target_catalog["target_position_1_based"].dropna().astype(int).tolist()
    target_columns = target_catalog["target_column"].astype(str).tolist()
    checks.add("target_catalog_total", len(target_catalog) == EXPECTED_TARGETS, EXPECTED_TARGETS, len(target_catalog))
    checks.add("target_catalog_positions", positions == list(range(1, EXPECTED_TARGETS + 1)), f"1..{EXPECTED_TARGETS}", f"{min(positions) if positions else 'NA'}..{max(positions) if positions else 'NA'}")
    checks.add("target_catalog_unique_names", len(set(target_columns)) == EXPECTED_TARGETS, EXPECTED_TARGETS, len(set(target_columns)))
    target_sha = catalog_sha256(target_columns)
    manifest_target_sha = base_manifest.get("target_catalog_sha256")
    checks.add("target_catalog_sha256", target_sha == manifest_target_sha, manifest_target_sha, target_sha)

    target_header = read_header(artifacts["targets"])
    checks.add("target_table_metadata_prefix", tuple(target_header[:4]) == TARGET_META, list(TARGET_META), target_header[:4])
    checks.add("target_table_target_count", len(target_header[4:]) == EXPECTED_TARGETS, EXPECTED_TARGETS, len(target_header[4:]))
    order_matches = sum(left == right for left, right in zip(target_header[4:], target_columns, strict=False))
    checks.add("target_table_catalog_order", target_header[4:] == target_columns, EXPECTED_TARGETS, order_matches, "Observed is order-matched names")

    targets = pd.read_csv(
        artifacts["targets"],
        dtype={"RID": "string", "VISCODE2": "string", "split": "string"},
        low_memory=False,
    )
    targets["RID"] = clean_text(targets["RID"])
    targets["VISCODE2"] = clean_text(targets["VISCODE2"], lower=True)
    targets["split"] = clean_text(targets["split"], lower=True)
    target_missing_key = int((targets["RID"].isna() | targets["RID"].eq("") | targets["VISCODE2"].isna() | targets["VISCODE2"].eq("")).sum())
    target_duplicate_key = int(targets.duplicated(["RID", "VISCODE2"]).sum())
    target_invalid_split = int((~targets["split"].isin(SPLITS)).sum())
    checks.add("target_table_total", len(targets) == EXPECTED_VISITS, EXPECTED_VISITS, len(targets))
    checks.add("target_table_missing_key", target_missing_key == 0, 0, target_missing_key)
    checks.add("target_table_duplicate_key", target_duplicate_key == 0, 0, target_duplicate_key)
    checks.add("target_table_valid_split", target_invalid_split == 0, 0, target_invalid_split)

    numeric_targets = targets.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    missing_targets = int(numeric_targets.isna().sum().sum())
    nonfinite_targets = int((~np.isfinite(numeric_targets.to_numpy(dtype=float))).sum())
    checks.add("target_values_numeric_complete", missing_targets == 0, 0, missing_targets)
    checks.add("target_values_finite", nonfinite_targets == 0, 0, nonfinite_targets)

    target_position = pd.to_numeric(targets["eligible_visit_position"], errors="coerce")
    checks.add("target_numeric_position", int(target_position.isna().sum()) == 0, 0, int(target_position.isna().sum()))
    targets["eligible_visit_position"] = target_position.astype("Int64")

    target_keys = visit_key_set(targets)
    checks.add("anchor_target_same_keys", anchor_keys == target_keys, EXPECTED_VISITS, len(anchor_keys & target_keys), "Observed is common-key count")
    targets_indexed = targets.set_index(["RID", "VISCODE2"]).sort_index()
    target_common = anchor_indexed.index.intersection(targets_indexed.index)
    target_split_mismatch = int(anchor_indexed.loc[target_common, "split"].ne(targets_indexed.loc[target_common, "split"]).sum())
    target_position_mismatch = int(anchor_indexed.loc[target_common, "eligible_visit_position"].ne(targets_indexed.loc[target_common, "eligible_visit_position"]).sum())
    checks.add("anchor_target_same_split", target_split_mismatch == 0, 0, target_split_mismatch)
    checks.add("anchor_target_same_position", target_position_mismatch == 0, 0, target_position_mismatch)

    visit_counts = {name: int(anchor["split"].eq(name).sum()) for name in SPLITS}
    target_visit_counts = {name: int(targets["split"].eq(name).sum()) for name in SPLITS}
    checks.add("anchor_split_counts", visit_counts == EXPECTED_VISITS_BY_SPLIT, EXPECTED_VISITS_BY_SPLIT, visit_counts)
    checks.add("target_split_counts", target_visit_counts == EXPECTED_VISITS_BY_SPLIT, EXPECTED_VISITS_BY_SPLIT, target_visit_counts)

    feature_catalog["source_name"] = clean_text(feature_catalog["source_name"]).str.upper()
    feature_catalog["column_name"] = clean_text(feature_catalog["column_name"])
    feature_catalog["feature_position_within_source"] = pd.to_numeric(feature_catalog["feature_position_within_source"], errors="coerce").astype("Int64")
    source_set = sorted(set(feature_catalog["source_name"].astype(str)))
    duplicate_features = int(feature_catalog.duplicated(["source_name", "column_name"]).sum())
    invalid_roles = int(clean_text(feature_catalog["role"]).ne("candidate_input_feature").sum())
    checks.add("feature_catalog_sources", set(source_set) == set(INPUT_SOURCES), sorted(INPUT_SOURCES), source_set)
    checks.add("feature_catalog_unique_source_column", duplicate_features == 0, 0, duplicate_features)
    checks.add("feature_catalog_roles", invalid_roles == 0, 0, invalid_roles)

    return (
        {
            "population": {"participants": len(participants), "visits": len(anchor)},
            "split": {
                "participant_counts": participant_counts,
                "visit_counts": visit_counts,
                "participant_overlaps": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                "split_mapping_sha256": computed_mapping_sha,
            },
            "targets": {
                "target_count": len(target_columns),
                "target_catalog_sha256": target_sha,
                "missing_target_values": missing_targets,
                "nonfinite_target_values": nonfinite_targets,
            },
            "feature_catalog": {
                "candidate_feature_count": len(feature_catalog),
                "source_count": feature_catalog["source_name"].nunique(),
            },
        },
        participants,
        feature_catalog,
    )


def validate_input_sources(
    artifacts: dict[str, Path],
    participants: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    checks: Checks,
) -> list[SourceResult]:
    participant_map = participants.set_index("RID")["split"]
    official = set(participants["RID"].astype(str))
    results: list[SourceResult] = []

    for source in INPUT_SOURCES:
        path = artifacts[f"input__{source}"]
        header = read_header(path)
        required = [*SOURCE_KEYS[source], "split"]
        missing_required = [column for column in required if column not in header]
        checks.add(f"source__{source}__required_columns", not missing_required, required, header)
        if missing_required:
            results.append(SourceResult(source, 0, len(header), 0, 0, 0, 0, 0, 0, False, False))
            continue

        keys = read_text_csv(path, usecols=required)
        keys["RID"] = clean_text(keys["RID"])
        keys["split"] = clean_text(keys["split"], lower=True)
        for column in SOURCE_KEYS[source]:
            if column != "RID":
                keys[column] = clean_text(keys[column], lower=True)

        missing_key = int(keys.loc[:, list(SOURCE_KEYS[source])].eq("").any(axis=1).sum())
        duplicate_key = int(keys.duplicated(list(SOURCE_KEYS[source])).sum())
        invalid_split = int((~keys["split"].isin(SPLITS)).sum())
        unofficial = int((~keys["RID"].isin(official)).sum())
        mapped = keys["RID"].map(participant_map)
        mismatch = int((mapped.notna() & mapped.ne(keys["split"])).sum())

        table_features = [column for column in header if column not in required]
        catalog_features = (
            feature_catalog.loc[feature_catalog["source_name"].eq(source)]
            .sort_values("feature_position_within_source", kind="stable")["column_name"]
            .astype(str)
            .tolist()
        )
        catalog_matches = table_features == catalog_features
        order_matches = sum(left == right for left, right in zip(table_features, catalog_features, strict=False))

        checks.add(f"source__{source}__missing_key", missing_key == 0, 0, missing_key)
        checks.add(f"source__{source}__duplicate_key", duplicate_key == 0, 0, duplicate_key)
        checks.add(f"source__{source}__valid_split", invalid_split == 0, 0, invalid_split)
        checks.add(f"source__{source}__official_participants", unofficial == 0, 0, unofficial)
        checks.add(f"source__{source}__split_consistency", mismatch == 0, 0, mismatch)
        checks.add(f"source__{source}__catalog_order", catalog_matches, len(catalog_features), order_matches, "Observed is order-matched feature names")

        passed = all((missing_key == 0, duplicate_key == 0, invalid_split == 0, unofficial == 0, mismatch == 0, catalog_matches))
        results.append(SourceResult(source, len(keys), len(header), len(table_features), missing_key, duplicate_key, invalid_split, unofficial, mismatch, catalog_matches, passed))

    checks.add("all_input_sources_pass", all(result.passed for result in results), len(INPUT_SOURCES), sum(result.passed for result in results), "Expected and observed are source counts")
    return results


def write_outputs(
    paths: Paths,
    checks: Checks,
    inventory: Sequence[InventoryResult],
    sources: Sequence[SourceResult],
    aggregate: dict[str, Any],
    fatal_error: str | None,
) -> None:
    pd.DataFrame(
        [
            {
                "check_name": item.name,
                "passed": item.passed,
                "expected": json_cell(item.expected),
                "observed": json_cell(item.observed),
                "details": item.details,
            }
            for item in checks.items
        ]
    ).to_csv(paths.safe / "benchmark01b_validation_checks.csv", index=False, encoding="utf-8")

    pd.DataFrame([asdict(item) for item in inventory]).to_csv(
        paths.safe / "benchmark01b_inventory_validation.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame([asdict(item) for item in sources]).to_csv(
        paths.safe / "benchmark01b_input_source_validation.csv", index=False, encoding="utf-8"
    )

    passed = checks.passed and fatal_error is None
    summary = {
        "benchmark01_script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "benchmark01b_validate_frozen_package_v1_1",
        "validation_passed": passed,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": fatal_error,
        "aggregate_summary": aggregate,
        "privacy": {
            "contains_individual_records": False,
            "contains_participant_identifiers": False,
            "contains_visit_identifiers": False,
            "contains_medical_values": False,
            "contains_only_aggregate_counts_file_metadata_and_global_hashes": True,
        },
    }
    (paths.safe / "benchmark01b_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    population = aggregate.get("population", {})
    split = aggregate.get("split", {})
    targets = aggregate.get("targets", {})
    inventory_summary = aggregate.get("inventory", {})
    story = [
        "# BENCHMARK01B — Frozen package v1.1 validation",
        "",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        f"- Logical package: {aggregate.get('manifest', {}).get('logical_package_name', 'NA')}",
        f"- Base package: {aggregate.get('manifest', {}).get('base_package_name', 'NA')}",
        f"- Temporal policy ID: {aggregate.get('temporal_policy', {}).get('temporal_policy_decision_id', 'NA')}",
        f"- Checks executed: {len(checks.items)}",
        f"- Failed checks: {checks.failed}",
        f"- Package content files: {inventory_summary.get('content_file_count', 'NA')}",
        f"- Ignored operating-system metadata files: {inventory_summary.get('ignored_metadata_file_count', 0)}",
        f"- Participants: {population.get('participants', 'NA')}",
        f"- Supervised visits: {population.get('visits', 'NA')}",
        f"- MRI targets: {targets.get('target_count', 'NA')}",
        f"- Participants by split: {split.get('participant_counts', 'NA')}",
        f"- Visits by split: {split.get('visit_counts', 'NA')}",
        f"- Split mapping SHA-256: {split.get('split_mapping_sha256', 'NA')}",
        "",
        "No model was trained. No raw ADNI source was read. The report contains only aggregate counts, file metadata, and global hashes.",
    ]
    if fatal_error:
        story.extend(["", "## Fatal error", "", fatal_error])
    (paths.safe / "BENCHMARK01B_VALIDATION_STORY.md").write_text("\n".join(story) + "\n", encoding="utf-8")


def validate_frozen_package(
    package_root: Path,
    reports_root: Path,
) -> ValidationResult:
    package_root = package_root.expanduser().resolve()
    reports_root = reports_root.expanduser().resolve()
    stage = reports_root / "benchmark01b_validate_frozen_package"
    safe = stage / "safe"
    logs = stage / "logs"
    safe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    project_root = package_root
    for _ in range(3):
        project_root = project_root.parent

    paths = Paths(
        root=project_root,
        package=package_root,
        safe=safe,
        logs=logs,
    )
    configure_logging(paths)
    logging.info("benchmark01b_validate_frozen_package v%s", SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Frozen package directory: %s", paths.package)
    logging.info("Expected logical package: %s", EXPECTED_PACKAGE_NAME)

    checks = Checks()
    inventory_results: list[InventoryResult] = []
    source_results: list[SourceResult] = []
    aggregate: dict[str, Any] = {}
    fatal_error: str | None = None

    try:
        artifacts = locate_artifacts(paths, checks)

        manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
        base_manifest = json.loads(artifacts["base_manifest"].read_text(encoding="utf-8"))
        policy = json.loads(artifacts["temporal_policy_lock"].read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValidationAbort("Current manifest root is not a JSON object")
        if not isinstance(base_manifest, dict):
            raise ValidationAbort("Preserved v1 manifest root is not a JSON object")
        if not isinstance(policy, dict):
            raise ValidationAbort("Temporal policy lock root is not a JSON object")

        split_text = parse_validation_text(artifacts["split_validation"])

        inventory_results, inventory_summary = validate_inventory(paths, artifacts, checks)
        aggregate["inventory"] = inventory_summary
        aggregate["base_manifest"] = validate_base_manifest(base_manifest, split_text, checks)
        aggregate["manifest"] = validate_extension_manifest(manifest, base_manifest, artifacts, checks)
        aggregate["temporal_policy"] = validate_temporal_policy_lock(policy, manifest, artifacts, checks)

        core_summary, participants, feature_catalog = validate_core(artifacts, base_manifest, checks)
        aggregate.update(core_summary)

        source_results = validate_input_sources(artifacts, participants, feature_catalog, checks)
        aggregate["input_sources"] = {
            "source_count": len(source_results),
            "passed_source_count": sum(result.passed for result in source_results),
            "row_counts": {result.source_name: result.rows for result in source_results},
            "feature_counts": {result.source_name: result.features for result in source_results},
        }
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("BENCHMARK01B aborted")

    write_outputs(paths, checks, inventory_results, source_results, aggregate, fatal_error)
    passed = checks.passed and fatal_error is None

    if passed:
        logging.info(
            "BENCHMARK01B PASSED | package=%s | temporal_policy=%s | participants=%s | visits=%s | targets=%s",
            aggregate.get("manifest", {}).get("logical_package_name"),
            aggregate.get("temporal_policy", {}).get("temporal_policy_decision_id"),
            aggregate.get("population", {}).get("participants"),
            aggregate.get("population", {}).get("visits"),
            aggregate.get("targets", {}).get("target_count"),
        )
        logging.info("No model was trained. No frozen artifact was modified.")
    else:
        logging.error(
            "BENCHMARK01B FAILED | failed_checks=%d | fatal_error=%s",
            checks.failed,
            fatal_error,
        )

    return ValidationResult(
        passed=passed,
        failed_check_count=checks.failed,
        fatal_error=fatal_error,
        aggregate=aggregate,
        checks=tuple(checks.items),
        inventory=tuple(inventory_results),
        sources=tuple(source_results),
        safe_dir=safe,
    )


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    result = validate_frozen_package(
        package_root=paths.package,
        reports_root=paths.safe.parent.parent,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
