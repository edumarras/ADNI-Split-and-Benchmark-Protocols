#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK04B — freeze the corrected cross-checked feature schema.
#
# Purpose
# -------
# Turn the already-frozen semantic CrossCheck (A06) plus the aggregate BENCHMARK02
# profile into the authoritative schema for the corrected rerun.
#
# Key design rule
# ---------------
# A06 is authoritative for every feature it explicitly changed. For current
# cross-checked features that A06 did not change, BENCHMARK04B preserves the
# previously frozen/validated semantic type, encoding and transform from the
# legacy official schema. BENCHMARK02 heuristics remain diagnostic only and are
# never allowed to silently overwrite a prior semantic decision.
#
# This stage:
# - validates BENCHMARK01, A06 v0.1.2 and BENCHMARK02 v0.2.0 PASS;
# - validates the A06 catalog/directive hashes;
# - reads the legacy official schema/transform registry only as a semantic
#   baseline for unchanged features;
# - freezes the 50-participant support rule without applying it yet;
# - writes a new B-suffixed official schema/transform registry/support policy;
# - writes a deterministic schema decision ID and lock;
# - emits SAFE aggregate/schema-only reports.
#
# This stage does NOT:
# - read individualized source rows;
# - read MRI target values;
# - alter the participant split or supervised population;
# - apply support filtering;
# - construct scenarios;
# - impute/encode/scale;
# - train any model or inspect test performance.
#
# Outputs
# -------
# config/benchmark_feature_schema_b.csv
# config/benchmark_feature_transform_registry_b.csv
# config/benchmark_support_policy_b.json
# config/benchmark_schema_lock_b.json
#
# reports/benchmark04b/safe/
#     benchmark04b_summary.json
#     benchmark04b_validation_checks.csv
#     benchmark04b_source_schema_summary.csv
#     benchmark04b_feature_type_summary.csv
#     benchmark04b_decision_origin_summary.csv
#     benchmark04b_schema_diff_vs_legacy.csv
#     benchmark04b_official_schema_safe.csv
#     benchmark04b_transform_registry_safe.csv
#     BENCHMARK04B_SCHEMA_STORY.md

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION: Final[str] = "0.1.2"
STAGE_NAME: Final[str] = "benchmark04b"
EXPECTED_LOGICAL_PACKAGE: Final[str] = "frozen_experiment_v1_1"
EXPECTED_TEMPORAL_POLICY_ID: Final[str] = "90fefe5ec34d49743612"
EXPECTED_LEDGER_VERSION: Final[str] = "1.0.0"
EXPECTED_LEDGER_SHA256: Final[str] = (
    "0bfe87ca42e9d9c74c804878a7c09372bc4eb856cf2d60200c315dd301b04fa5"
)
EXPECTED_A06_VERSION: Final[str] = "0.1.2"
EXPECTED_B02_VERSION: Final[str] = "0.2.1"
EXPECTED_CURRENT_FEATURES: Final[int] = 218
EXPECTED_EXCLUSION_DIRECTIVES: Final[int] = 143
EXPECTED_DERIVE_LATER: Final[int] = 1

EXPECTED_LEGACY_EXCLUDED_CURRENT: Final[frozenset[tuple[str, str]]] = frozenset({
    ("ADAS", "DONE"),
    ("ADAS", "Q1TR1"),
    ("ADAS", "Q1TR2"),
    ("ADAS", "Q1TR3"),
    ("ADAS", "Q3TASK5"),
    ("ADAS", "Q4TASK"),
    ("MMSE", "DONE"),
})
EXPECTED_SUPPORT_THRESHOLD: Final[int] = 50
EXPECTED_LEGACY_SCHEMA_SHA256: Final[str] = (
    "0e43dbf3640e8e872f06ab35f5e8be56ab9f963ec9b73d3c275778c9db2a8bd3"
)
EXPECTED_LEGACY_TRANSFORMS_SHA256: Final[str] = (
    "830956f08536e54a6ed9642c994ce649d630d63008afa1bb907aa87e038f9e52"
)
INPUT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS", "CDR", "FAQ", "MMSE", "MOCA", "NEUROBAT", "PTDEMOG", "APOERES"
)

PACKAGE_DIR_NAME: Final[str] = "frozen_experiment_v1_1"
CROSSCHECK_DIR_NAME: Final[str] = "crosschecked_inputs_v1"

BENCHMARK01_SUMMARY: Final[str] = (
    "reports/benchmark01b_validate_frozen_package/safe/benchmark01b_validation_summary.json"
)
A06_REPORT: Final[str] = (
    "reports/auditA06_freeze_crosschecked_inputs/safe/a06_verification_report.json"
)
B02_SUMMARY: Final[str] = (
    "reports/benchmark02b/safe/benchmark02b_summary.json"
)
B02_PROFILE: Final[str] = (
    "reports/benchmark02b/safe/benchmark02b_feature_profile.csv"
)
B02_SCHEMA_DRAFT: Final[str] = (
    "reports/benchmark02b/safe/benchmark02b_feature_schema_draft.csv"
)

A06_CATALOG: Final[str] = "crosschecked_input_feature_catalog.csv"
A06_DIRECTIVES: Final[str] = "crosscheck_feature_directives.csv"
A06_MANIFEST: Final[str] = "crosschecked_input_manifest.json"

# Legacy semantic baseline. These are never overwritten by this B stage.
LEGACY_SCHEMA: Final[str] = "config/benchmark_feature_schema.csv"
LEGACY_TRANSFORMS: Final[str] = "config/benchmark_feature_transform_registry_final.csv"

# Corrected B-stage outputs.
OFFICIAL_SCHEMA_B: Final[str] = "config/benchmark_feature_schema_b.csv"
OFFICIAL_TRANSFORMS_B: Final[str] = "config/benchmark_feature_transform_registry_b.csv"
SUPPORT_POLICY_B: Final[str] = "config/benchmark_support_policy_b.json"
SCHEMA_LOCK_B: Final[str] = "config/benchmark_schema_lock_b.json"

SUPPORT_UNIT: Final[str] = "distinct_fit_participants_with_observed_original_feature"
SUPPORT_APPLICATION: Final[str] = (
    "recomputed_per_scenario_and_per_temporal_block_on_fit_data"
)
SUPPORT_REMOVAL_RULE: Final[str] = (
    "remove_column_if_support_lt_50_or_all_missing_or_effectively_constant_in_fit_never_remove_visits"
)

IDENTITY_TRANSFORMS: Final[frozenset[str]] = frozenset(
    {"identity", "numeric_identity", "categorical_identity", "binary_zero_one_identity", "binary_identity", "none", ""}
)

ALLOWED_TYPES: Final[frozenset[str]] = frozenset(
    {"numeric", "binary", "categorical", "multi_response", "excluded"}
)
TYPE_TO_ENCODING: Final[dict[str, str]] = {
    "numeric": "numeric_standardized",
    "binary": "binary_identity",
    "categorical": "categorical_one_hot",
    "multi_response": "multi_hot",
    "excluded": "excluded",
}

A06_INCLUDE_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "retain", "retain_harmonized", "retain_retype", "retain_transform_retype",
        "derive_retain", "derive_later",
    }
)


class SchemaAbort(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    crosschecked: Path
    safe: Path
    logs: Path
    benchmark01_summary: Path
    a06_report: Path
    b02_summary: Path
    b02_profile: Path
    b02_schema_draft: Path
    a06_catalog: Path
    a06_directives: Path
    a06_manifest: Path
    legacy_schema: Path
    legacy_transforms: Path
    official_schema_b: Path
    official_transforms_b: Path
    support_policy_b: Path
    schema_lock_b: Path


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

    def add(self, name: str, passed: bool, expected: Any, observed: Any, details: str = "") -> bool:
        item = Check(name, bool(passed), expected, observed, details)
        self.items.append(item)
        logging.log(logging.INFO if passed else logging.ERROR, "%s | %s", "PASS" if passed else "FAIL", name)
        if not passed:
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                name, short_text(expected), short_text(observed), f" | {details}" if details else "",
            )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)


def short_text(value: Any, limit: int = 700) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    raise SchemaAbort(f"Cannot interpret boolean value: {value!r}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaAbort(f"{label} missing required columns: {missing}")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n", na_rep="")
    temp.replace(path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def resolve_paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    stage = root / "reports" / STAGE_NAME
    safe = stage / "safe"
    logs = stage / "logs"
    safe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    crosschecked = root / "data" / "processed" / CROSSCHECK_DIR_NAME
    return Paths(
        root=root,
        package=root / "data" / "processed" / PACKAGE_DIR_NAME,
        crosschecked=crosschecked,
        safe=safe,
        logs=logs,
        benchmark01_summary=root / BENCHMARK01_SUMMARY,
        a06_report=root / A06_REPORT,
        b02_summary=root / B02_SUMMARY,
        b02_profile=root / B02_PROFILE,
        b02_schema_draft=root / B02_SCHEMA_DRAFT,
        a06_catalog=crosschecked / A06_CATALOG,
        a06_directives=crosschecked / A06_DIRECTIVES,
        a06_manifest=crosschecked / A06_MANIFEST,
        legacy_schema=root / LEGACY_SCHEMA,
        legacy_transforms=root / LEGACY_TRANSFORMS,
        official_schema_b=root / OFFICIAL_SCHEMA_B,
        official_transforms_b=root / OFFICIAL_TRANSFORMS_B,
        support_policy_b=root / SUPPORT_POLICY_B,
        schema_lock_b=root / SCHEMA_LOCK_B,
    )


def configure_logging(paths: Paths) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "benchmark04b.log", encoding="utf-8"),
        ],
        force=True,
    )


def validate_upstream(paths: Paths, checks: Checks) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {
        "benchmark01_summary": paths.benchmark01_summary,
        "a06_report": paths.a06_report,
        "b02_summary": paths.b02_summary,
        "b02_profile": paths.b02_profile,
        "b02_schema_draft": paths.b02_schema_draft,
        "a06_catalog": paths.a06_catalog,
        "a06_directives": paths.a06_directives,
        "a06_manifest": paths.a06_manifest,
        "legacy_schema": paths.legacy_schema,
        "legacy_transforms": paths.legacy_transforms,
    }
    for label, path in required.items():
        checks.add(f"artifact__{label}", path.is_file(), True, path.is_file(), str(path))
    if not all(path.is_file() for path in required.values()):
        raise SchemaAbort("Required upstream artifacts are missing.")

    checks.add(
        "legacy_schema_exact_historical_sha256",
        sha256_file(paths.legacy_schema) == EXPECTED_LEGACY_SCHEMA_SHA256,
        EXPECTED_LEGACY_SCHEMA_SHA256,
        sha256_file(paths.legacy_schema),
    )
    checks.add(
        "legacy_transforms_exact_historical_sha256",
        sha256_file(paths.legacy_transforms) == EXPECTED_LEGACY_TRANSFORMS_SHA256,
        EXPECTED_LEGACY_TRANSFORMS_SHA256,
        sha256_file(paths.legacy_transforms),
    )

    b01 = read_json(paths.benchmark01_summary)
    a06 = read_json(paths.a06_report)
    b02 = read_json(paths.b02_summary)
    a06_manifest = read_json(paths.a06_manifest)

    b01_pass = bool(b01.get("validation_passed", b01.get("package_validation_passed", False)))
    checks.add("benchmark01_passed", b01_pass, True, b01_pass)

    checks.add("a06_passed", a06.get("status") == "PASS", "PASS", a06.get("status"))
    checks.add("a06_version", str(a06.get("script_version")) == EXPECTED_A06_VERSION, EXPECTED_A06_VERSION, a06.get("script_version"))
    checks.add("a06_failed_checks", int(a06.get("failed_check_count", -1)) == 0, 0, a06.get("failed_check_count"))
    checks.add("a06_package", a06.get("upstream", {}).get("logical_package_name") == EXPECTED_LOGICAL_PACKAGE, EXPECTED_LOGICAL_PACKAGE, a06.get("upstream", {}).get("logical_package_name"))
    checks.add("a06_temporal_policy", a06.get("upstream", {}).get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_ID, EXPECTED_TEMPORAL_POLICY_ID, a06.get("upstream", {}).get("temporal_policy_decision_id"))
    checks.add("a06_ledger_version", a06.get("upstream", {}).get("crosscheck_ledger_version") == EXPECTED_LEDGER_VERSION, EXPECTED_LEDGER_VERSION, a06.get("upstream", {}).get("crosscheck_ledger_version"))
    checks.add("a06_ledger_hash", a06.get("upstream", {}).get("crosscheck_ledger_sha256") == EXPECTED_LEDGER_SHA256, EXPECTED_LEDGER_SHA256, a06.get("upstream", {}).get("crosscheck_ledger_sha256"))
    checks.add("a06_support_threshold", int(a06.get("frozen_invariants", {}).get("support_threshold", -1)) == EXPECTED_SUPPORT_THRESHOLD, EXPECTED_SUPPORT_THRESHOLD, a06.get("frozen_invariants", {}).get("support_threshold"))
    checks.add("a06_split_unchanged", a06.get("frozen_invariants", {}).get("participant_mapping_changed") is False, False, a06.get("frozen_invariants", {}).get("participant_mapping_changed"))
    checks.add("a06_population_unchanged", a06.get("frozen_invariants", {}).get("supervised_population_changed") is False, False, a06.get("frozen_invariants", {}).get("supervised_population_changed"))
    checks.add("a06_targets_unchanged", a06.get("frozen_invariants", {}).get("target_catalog_changed") is False and a06.get("frozen_invariants", {}).get("target_values_changed") is False, False, bool(a06.get("frozen_invariants", {}).get("target_catalog_changed")) or bool(a06.get("frozen_invariants", {}).get("target_values_changed")))

    checks.add("b02_passed", b02.get("profiling_passed") is True, True, b02.get("profiling_passed"))
    b02_version_observed = b02.get("benchmark02b_script_version", b02.get("benchmark02_script_version"))
    checks.add("b02_version", str(b02_version_observed) == EXPECTED_B02_VERSION, EXPECTED_B02_VERSION, b02_version_observed)
    checks.add("b02_failed_checks", int(b02.get("failed_check_count", -1)) == 0, 0, b02.get("failed_check_count"))
    checks.add("b02_feature_count", int(b02.get("aggregate_summary", {}).get("feature_count", -1)) == EXPECTED_CURRENT_FEATURES, EXPECTED_CURRENT_FEATURES, b02.get("aggregate_summary", {}).get("feature_count"))
    checks.add("b02_source_count", int(b02.get("aggregate_summary", {}).get("source_count", -1)) == len(INPUT_SOURCES), len(INPUT_SOURCES), b02.get("aggregate_summary", {}).get("source_count"))
    checks.add("b02_no_all_missing", int(b02.get("advisory_flags", {}).get("snapshot_train_all_missing_features", -1)) == 0, 0, b02.get("advisory_flags", {}).get("snapshot_train_all_missing_features"))
    checks.add("b02_no_effective_constants", int(b02.get("advisory_flags", {}).get("snapshot_train_effective_constant_features", -1)) == 0, 0, b02.get("advisory_flags", {}).get("snapshot_train_effective_constant_features"))

    # Crosschecked artifact hashes are frozen by A06.
    checks.add("a06_catalog_hash", sha256_file(paths.a06_catalog) == a06.get("feature_catalog", {}).get("sha256"), a06.get("feature_catalog", {}).get("sha256"), sha256_file(paths.a06_catalog))
    checks.add("a06_directives_hash", sha256_file(paths.a06_directives) == a06.get("directives", {}).get("sha256"), a06.get("directives", {}).get("sha256"), sha256_file(paths.a06_directives))
    checks.add("a06_manifest_output_name", a06_manifest.get("output_name") == CROSSCHECK_DIR_NAME, CROSSCHECK_DIR_NAME, a06_manifest.get("output_name"))

    if not checks.passed:
        raise SchemaAbort("Upstream contract validation failed.")
    return a06, b02, a06_manifest


def load_tables(paths: Paths, checks: Checks) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(paths.a06_catalog, low_memory=False)
    directives = pd.read_csv(paths.a06_directives, low_memory=False)
    profile = pd.read_csv(paths.b02_profile, low_memory=False)
    legacy = pd.read_csv(paths.legacy_schema, low_memory=False)
    legacy_transforms = pd.read_csv(paths.legacy_transforms, low_memory=False)

    for frame in (catalog, directives, profile, legacy, legacy_transforms):
        frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]

    require_columns(catalog, ["source_name", "feature_position_within_source", "column_name"], "A06 catalog")
    require_columns(directives, ["source_name", "column_name", "crosscheck_action", "final_feature_type", "value_transform_id", "semantic_role", "reason", "execution_stage"], "A06 directives")
    require_columns(profile, ["source_name", "column_name", "grain", "feature_position_within_source", "a06_directive_present", "suggested_type", "suggested_encoding", "snapshot_train_observed_participants", "snapshot_train_all_missing", "snapshot_train_effective_constant"], "BENCHMARK02 profile")
    require_columns(legacy, ["source_name", "column_name", "full_column_name", "official_semantic_include", "official_feature_type", "official_encoding", "official_semantic_role", "official_value_transform_id", "official_decision_id"], "legacy official schema")
    require_columns(legacy_transforms, ["source_name", "column_name", "full_column_name", "transform_id", "phase_required_for_transform", "phase_exposed_to_model", "parameters_json"], "legacy transform registry")

    for frame in (catalog, directives, profile, legacy, legacy_transforms):
        for column in [c for c in ("source_name", "column_name", "full_column_name") if c in frame.columns]:
            frame[column] = frame[column].astype("string").str.strip()

    checks.add("catalog_feature_count", len(catalog) == EXPECTED_CURRENT_FEATURES, EXPECTED_CURRENT_FEATURES, len(catalog))
    checks.add("catalog_unique", not catalog.duplicated(["source_name", "column_name"]).any(), True, not catalog.duplicated(["source_name", "column_name"]).any())
    checks.add("profile_feature_count", len(profile) == EXPECTED_CURRENT_FEATURES, EXPECTED_CURRENT_FEATURES, len(profile))
    checks.add("profile_unique", not profile.duplicated(["source_name", "column_name"]).any(), True, not profile.duplicated(["source_name", "column_name"]).any())
    checks.add("directives_unique", not directives.duplicated(["source_name", "column_name"]).any(), True, not directives.duplicated(["source_name", "column_name"]).any())
    checks.add("legacy_unique", not legacy.duplicated(["source_name", "column_name"]).any(), True, not legacy.duplicated(["source_name", "column_name"]).any())
    checks.add("legacy_transform_unique", not legacy_transforms.duplicated(["source_name", "column_name"]).any(), True, not legacy_transforms.duplicated(["source_name", "column_name"]).any())

    catalog_keys = set(zip(catalog["source_name"].astype(str), catalog["column_name"].astype(str)))
    profile_keys = set(zip(profile["source_name"].astype(str), profile["column_name"].astype(str)))
    checks.add("profile_matches_catalog", profile_keys == catalog_keys, len(catalog_keys), len(profile_keys), f"missing={sorted(catalog_keys-profile_keys)[:10]}, extra={sorted(profile_keys-catalog_keys)[:10]}")

    exclusion_count = int(directives["crosscheck_action"].astype(str).eq("exclude").sum())
    derive_later_count = int(directives["crosscheck_action"].astype(str).eq("derive_later").sum())
    checks.add("exclusion_directive_count", exclusion_count == EXPECTED_EXCLUSION_DIRECTIVES, EXPECTED_EXCLUSION_DIRECTIVES, exclusion_count)
    checks.add("derive_later_directive_count", derive_later_count == EXPECTED_DERIVE_LATER, EXPECTED_DERIVE_LATER, derive_later_count)

    if not checks.passed:
        raise SchemaAbort("Structural input validation failed.")
    return catalog, directives, profile, legacy, legacy_transforms


def build_schema(
    catalog: pd.DataFrame,
    directives: pd.DataFrame,
    profile: pd.DataFrame,
    legacy: pd.DataFrame,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_map = {(str(r.source_name), str(r.column_name)): r for r in catalog.itertuples(index=False)}
    profile_map = {(str(r.source_name), str(r.column_name)): r for r in profile.itertuples(index=False)}
    directive_map = {(str(r.source_name), str(r.column_name)): r for r in directives.itertuples(index=False)}
    legacy_map = {(str(r.source_name), str(r.column_name)): r for r in legacy.itertuples(index=False)}

    current_keys = set(catalog_map)
    excluded_keys = {
        key for key, row in directive_map.items() if str(row.crosscheck_action) == "exclude"
    }
    derived_keys = {
        key for key, row in directive_map.items() if str(row.crosscheck_action) == "derive_later"
    }

    # Excluded columns must no longer be present in crosschecked current catalog.
    overlap_excluded = sorted(current_keys & excluded_keys)
    checks.add("excluded_not_in_crosschecked_catalog", not overlap_excluded, [], overlap_excluded[:20])

    rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    inherited_missing: list[str] = []
    inherited_was_excluded: list[str] = []
    directed_current_count = 0
    inherited_current_count = 0

    # Current crosschecked features.
    for key in sorted(current_keys, key=lambda x: (INPUT_SOURCES.index(x[0]), int(catalog_map[x].feature_position_within_source))):
        source, column = key
        cat = catalog_map[key]
        prof = profile_map[key]
        directive = directive_map.get(key)
        legacy_row = legacy_map.get(key)

        if directive is not None and str(directive.crosscheck_action) in A06_INCLUDE_ACTIONS:
            directed_current_count += 1
            feature_type = str(directive.final_feature_type).strip()
            transform_id = str(directive.value_transform_id).strip() or "identity"
            role = str(directive.semantic_role).strip()
            reason = str(directive.reason).strip()
            execution_stage = str(directive.execution_stage).strip()
            origin = "A06_frozen_crosscheck_directive"
            include = True
        else:
            inherited_current_count += 1
            if legacy_row is None:
                inherited_missing.append(f"{source}::{column}")
                continue
            legacy_include = parse_bool(legacy_row.official_semantic_include)
            if not legacy_include:
                # Some columns remain physically present in the crosschecked tables because A06
                # only materializes the *new* CrossCheck changes. Their older semantic exclusion
                # is still authoritative and must be preserved in the official B schema.
                inherited_was_excluded.append(f"{source}::{column}")
                feature_type = "excluded"
                transform_id = "none"
                role = str(legacy_row.official_semantic_role).strip()
                reason = "unchanged_feature_preserves_legacy_frozen_exclusion"
                execution_stage = "preserved_from_legacy_schema"
                origin = "legacy_schema_preserved_exclusion_for_unchanged_feature"
                include = False
            else:
                feature_type = str(legacy_row.official_feature_type).strip()
                transform_id = str(legacy_row.official_value_transform_id).strip() or "identity"
                role = str(legacy_row.official_semantic_role).strip()
                reason = "unchanged_feature_preserves_legacy_frozen_semantic_decision"
                execution_stage = "preserved_from_legacy_schema"
                origin = "legacy_schema_preserved_for_unchanged_feature"
                include = True

        if feature_type not in ALLOWED_TYPES:
            raise SchemaAbort(f"Unsupported final type for {source}::{column}: {feature_type}")
        encoding = TYPE_TO_ENCODING[feature_type]
        full_name = f"{source}__{column}"
        legacy_type = str(legacy_row.official_feature_type).strip() if legacy_row is not None else ""
        legacy_encoding = str(legacy_row.official_encoding).strip() if legacy_row is not None else ""
        legacy_transform = str(legacy_row.official_value_transform_id).strip() if legacy_row is not None else ""

        rows.append(
            {
                "source_name": source,
                "column_name": column,
                "full_column_name": full_name,
                "grain": str(prof.grain),
                "current_in_crosschecked_inputs": True,
                "derived_later": False,
                "feature_position_within_source": int(cat.feature_position_within_source),
                "official_semantic_include": include,
                "official_feature_type": feature_type,
                "official_encoding": encoding,
                "official_semantic_role": role,
                "official_value_transform_id": transform_id,
                "official_support_threshold_participants": EXPECTED_SUPPORT_THRESHOLD,
                "official_support_unit": SUPPORT_UNIT,
                "official_support_application": SUPPORT_APPLICATION,
                "official_support_filter_removes_visits": False,
                "official_support_evaluated_before_encoding": True,
                "official_support_recomputed_at_final_refit": True,
                "official_statistical_removal_rule": SUPPORT_REMOVAL_RULE,
                "official_decision_origin": origin,
                "official_decision_reason": reason,
                "official_execution_stage": execution_stage,
                "b02_snapshot_train_observed_participants_reference": int(prof.snapshot_train_observed_participants),
                "b02_snapshot_train_support_reference_only": True,
                "b02_suggested_type_diagnostic": str(prof.suggested_type),
                "legacy_feature_type_reference": legacy_type,
                "legacy_encoding_reference": legacy_encoding,
                "legacy_transform_reference": legacy_transform,
            }
        )
        diff_rows.append(
            {
                "source_name": source,
                "column_name": column,
                "status": "current_crosschecked_feature",
                "decision_origin": origin,
                "legacy_feature_type": legacy_type,
                "new_feature_type": feature_type,
                "legacy_encoding": legacy_encoding,
                "new_encoding": encoding,
                "legacy_transform": legacy_transform,
                "new_transform": transform_id,
                "semantic_changed_vs_legacy": bool(
                    legacy_row is None
                    or legacy_type != feature_type
                    or legacy_transform != transform_id
                ),
                "encoding_label_changed_vs_legacy": bool(legacy_row is not None and legacy_encoding != encoding),
            }
        )

    checks.add("unchanged_current_features_have_legacy_semantics", not inherited_missing, [], inherited_missing[:20])
    observed_legacy_excluded_current = frozenset(
        tuple(x.split("::", 1)) for x in inherited_was_excluded
    )
    checks.add(
        "unchanged_current_features_preserve_expected_legacy_exclusions",
        observed_legacy_excluded_current == EXPECTED_LEGACY_EXCLUDED_CURRENT,
        sorted(f"{s}::{c}" for s, c in EXPECTED_LEGACY_EXCLUDED_CURRENT),
        sorted(inherited_was_excluded),
    )

    # Frozen A06 exclusions stay represented in the official B schema for auditability.
    for key in sorted(excluded_keys, key=lambda x: (INPUT_SOURCES.index(x[0]), x[1])):
        source, column = key
        d = directive_map[key]
        legacy_row = legacy_map.get(key)
        rows.append(
            {
                "source_name": source,
                "column_name": column,
                "full_column_name": f"{source}__{column}",
                "grain": "participant" if source in {"PTDEMOG", "APOERES"} else "visit",
                "current_in_crosschecked_inputs": False,
                "derived_later": False,
                "feature_position_within_source": pd.NA,
                "official_semantic_include": False,
                "official_feature_type": "excluded",
                "official_encoding": "excluded",
                "official_semantic_role": str(d.semantic_role),
                "official_value_transform_id": "none",
                "official_support_threshold_participants": EXPECTED_SUPPORT_THRESHOLD,
                "official_support_unit": SUPPORT_UNIT,
                "official_support_application": SUPPORT_APPLICATION,
                "official_support_filter_removes_visits": False,
                "official_support_evaluated_before_encoding": True,
                "official_support_recomputed_at_final_refit": True,
                "official_statistical_removal_rule": SUPPORT_REMOVAL_RULE,
                "official_decision_origin": "A06_frozen_crosscheck_exclusion",
                "official_decision_reason": str(d.reason),
                "official_execution_stage": str(d.execution_stage),
                "b02_snapshot_train_observed_participants_reference": pd.NA,
                "b02_snapshot_train_support_reference_only": True,
                "b02_suggested_type_diagnostic": "not_profiled_removed_upstream",
                "legacy_feature_type_reference": str(legacy_row.official_feature_type).strip() if legacy_row is not None else "",
                "legacy_encoding_reference": str(legacy_row.official_encoding).strip() if legacy_row is not None else "",
                "legacy_transform_reference": str(legacy_row.official_value_transform_id).strip() if legacy_row is not None else "",
            }
        )
        diff_rows.append(
            {
                "source_name": source,
                "column_name": column,
                "status": "crosscheck_excluded",
                "decision_origin": "A06_frozen_crosscheck_exclusion",
                "legacy_feature_type": str(legacy_row.official_feature_type).strip() if legacy_row is not None else "",
                "new_feature_type": "excluded",
                "legacy_encoding": str(legacy_row.official_encoding).strip() if legacy_row is not None else "",
                "new_encoding": "excluded",
                "legacy_transform": str(legacy_row.official_value_transform_id).strip() if legacy_row is not None else "",
                "new_transform": "none",
                "semantic_changed_vs_legacy": True,
                "encoding_label_changed_vs_legacy": bool(legacy_row is not None and str(legacy_row.official_encoding).strip() != "excluded"),
            }
        )

    # Visit-specific AGE_AT_TARGET is frozen as a B05b-stage derivation.
    for key in sorted(derived_keys):
        source, column = key
        d = directive_map[key]
        if key in current_keys:
            raise SchemaAbort(f"derive_later feature unexpectedly exists in current A06 catalog: {source}::{column}")
        rows.append(
            {
                "source_name": source,
                "column_name": column,
                "full_column_name": f"{source}__{column}",
                "grain": "visit_derived",
                "current_in_crosschecked_inputs": False,
                "derived_later": True,
                "feature_position_within_source": pd.NA,
                "official_semantic_include": True,
                "official_feature_type": str(d.final_feature_type),
                "official_encoding": TYPE_TO_ENCODING[str(d.final_feature_type)],
                "official_semantic_role": str(d.semantic_role),
                "official_value_transform_id": str(d.value_transform_id),
                "official_support_threshold_participants": EXPECTED_SUPPORT_THRESHOLD,
                "official_support_unit": "distinct_fit_participants_with_valid_derived_feature",
                "official_support_application": SUPPORT_APPLICATION,
                "official_support_filter_removes_visits": False,
                "official_support_evaluated_before_encoding": True,
                "official_support_recomputed_at_final_refit": True,
                "official_statistical_removal_rule": SUPPORT_REMOVAL_RULE,
                "official_decision_origin": "A06_frozen_derive_later_directive",
                "official_decision_reason": str(d.reason),
                "official_execution_stage": "BENCHMARK05b",
                "b02_snapshot_train_observed_participants_reference": pd.NA,
                "b02_snapshot_train_support_reference_only": True,
                "b02_suggested_type_diagnostic": "not_profiled_derived_in_BENCHMARK05b",
                "legacy_feature_type_reference": "",
                "legacy_encoding_reference": "",
                "legacy_transform_reference": "",
            }
        )
        diff_rows.append(
            {
                "source_name": source,
                "column_name": column,
                "status": "derive_later_new_feature",
                "decision_origin": "A06_frozen_derive_later_directive",
                "legacy_feature_type": "",
                "new_feature_type": str(d.final_feature_type),
                "legacy_encoding": "",
                "new_encoding": TYPE_TO_ENCODING[str(d.final_feature_type)],
                "legacy_transform": "",
                "new_transform": str(d.value_transform_id),
                "semantic_changed_vs_legacy": True,
                "encoding_label_changed_vs_legacy": True,
            }
        )

    schema = pd.DataFrame(rows)
    diff = pd.DataFrame(diff_rows)

    checks.add("schema_unique_source_column", not schema.duplicated(["source_name", "column_name"]).any(), True, not schema.duplicated(["source_name", "column_name"]).any())
    checks.add("schema_unique_full_name", not schema["full_column_name"].duplicated().any(), True, not schema["full_column_name"].duplicated().any())
    expected_total = EXPECTED_CURRENT_FEATURES + EXPECTED_EXCLUSION_DIRECTIVES + EXPECTED_DERIVE_LATER
    checks.add("schema_total_rows", len(schema) == expected_total, expected_total, len(schema))
    included = int(schema["official_semantic_include"].map(parse_bool).sum())
    excluded = int((~schema["official_semantic_include"].map(parse_bool)).sum())
    expected_included = EXPECTED_CURRENT_FEATURES - len(EXPECTED_LEGACY_EXCLUDED_CURRENT) + EXPECTED_DERIVE_LATER
    expected_excluded = EXPECTED_EXCLUSION_DIRECTIVES + len(EXPECTED_LEGACY_EXCLUDED_CURRENT)
    checks.add("schema_included_rows", included == expected_included, expected_included, included)
    checks.add("schema_excluded_rows", excluded == expected_excluded, expected_excluded, excluded)
    checks.add("schema_sources", set(schema["source_name"].astype(str).unique()) == set(INPUT_SOURCES), list(INPUT_SOURCES), sorted(schema["source_name"].astype(str).unique().tolist()))
    checks.add("directed_current_feature_count_positive", directed_current_count > 0, ">0", directed_current_count)
    checks.add("inherited_current_feature_count_positive", inherited_current_count > 0, ">0", inherited_current_count)

    if not checks.passed:
        raise SchemaAbort("Feature schema construction failed.")

    schema = schema.sort_values(
        ["source_name", "official_semantic_include", "current_in_crosschecked_inputs", "feature_position_within_source", "column_name"],
        ascending=[True, False, False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    return schema, diff


def canonical_schema_payload(schema: pd.DataFrame, upstream_hashes: Mapping[str, str]) -> str:
    decision_columns = [
        "source_name", "column_name", "full_column_name", "grain",
        "current_in_crosschecked_inputs", "derived_later", "official_semantic_include",
        "official_feature_type", "official_encoding", "official_semantic_role",
        "official_value_transform_id", "official_support_threshold_participants",
        "official_support_unit", "official_support_application",
        "official_statistical_removal_rule", "official_decision_origin",
        "official_decision_reason", "official_execution_stage",
    ]
    records = schema.loc[:, decision_columns].fillna("").to_dict("records")
    payload = {
        "schema_version": "2.0.0-crosschecked-b",
        "support_threshold": EXPECTED_SUPPORT_THRESHOLD,
        "ledger_sha256": EXPECTED_LEDGER_SHA256,
        "upstream_hashes": dict(sorted(upstream_hashes.items())),
        "rows": records,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_transform_registry(
    schema: pd.DataFrame,
    legacy_transforms: pd.DataFrame,
    decision_id: str,
    checks: Checks,
) -> pd.DataFrame:
    legacy_map = {
        (str(r.source_name), str(r.column_name)): r
        for r in legacy_transforms.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    missing_parameters: list[str] = []

    included = schema.loc[schema["official_semantic_include"].map(parse_bool)].copy()
    for row in included.itertuples(index=False):
        transform_id = str(row.official_value_transform_id or "identity").strip()
        if transform_id in IDENTITY_TRANSFORMS:
            continue

        key = (str(row.source_name), str(row.column_name))
        old = legacy_map.get(key)
        parameters: dict[str, Any]
        input_semantics = ""
        output_semantics = ""
        phase_required = False

        if transform_id == "derive_age_at_target_from_birth_year_and_current_target_date":
            parameters = {
                "execution_stage": "BENCHMARK05b",
                "birth_metadata_artifact": "data/crosschecked_inputs_v1/local_only/LOCAL_ONLY_ptdemog_age_derivation_metadata.csv.gz",
                "birth_metadata_field": "BIRTH_YEAR_FOR_AGE",
                "birth_metadata_precision": "calendar_year",
                "current_target_date_source": "strict_date_temporal_reference_or_current_target_visit_date",
                "output_unit": "years",
                "privacy": "derivation_is_local_only; birth metadata and dates are never exported to SAFE reports",
            }
            input_semantics = "participant_birth_year_metadata_plus_current_target_visit_date"
            output_semantics = "visit_specific_age_covariate"
        elif old is not None and str(old.transform_id).strip() == transform_id:
            try:
                parameters = json.loads(str(old.parameters_json)) if str(old.parameters_json).strip() else {}
            except json.JSONDecodeError as exc:
                raise SchemaAbort(f"Legacy transform parameters invalid for {key}: {exc}") from exc
            input_semantics = str(getattr(old, "input_semantics", ""))
            output_semantics = str(getattr(old, "output_semantics", ""))
            phase_required = parse_bool(getattr(old, "phase_required_for_transform", False))
        elif transform_id == "faq_official_item_score_map":
            parameters = {"0": 0, "1": 0, "2": 1, "3": 1, "4": 2, "5": 3}
            input_semantics = "official_raw_FAQ_item_codes"
            output_semantics = "official_FAQ_item_score_0_to_3"
        elif transform_id == "ptdemog_pthome_harmonization":
            parameters = {
                "1": "house", "9": "house", "10": "house", "2": "condo_or_coop",
                "3": "apartment", "4": "mobile_home", "5": "retirement_community",
                "6": "assisted_living", "7": "skilled_nursing", "8": "other",
            }
            input_semantics = "phase_specific_residence_code"
            output_semantics = "phase_invariant_residence_category"
        elif transform_id == "dictionary_multi_response_tokens":
            parameters = {
                "token_separators": ["|", ";"],
                "encoding": "multi_hot",
                "missing_indicator": True,
                "special_none_correct_code_must_not_equal_missing": True,
            }
            input_semantics = "documented_task_component_code_set"
            output_semantics = "component_indicator_vector"
        else:
            missing_parameters.append(f"{row.full_column_name}:{transform_id}")
            continue

        rows.append(
            {
                "source_name": row.source_name,
                "column_name": row.column_name,
                "full_column_name": row.full_column_name,
                "transform_id": transform_id,
                "input_semantics": input_semantics,
                "output_semantics": output_semantics,
                "phase_required_for_transform": bool(phase_required),
                "phase_exposed_to_model": False,
                "parameters_json": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                "execution_stage": row.official_execution_stage,
                "official_decision_id": decision_id,
            }
        )

    checks.add("transform_parameters_resolved", not missing_parameters, [], missing_parameters[:20])
    registry = pd.DataFrame(rows)
    if registry.empty:
        raise SchemaAbort("Transform registry unexpectedly empty.")
    registry = registry.sort_values(["source_name", "column_name"], kind="stable").reset_index(drop=True)
    checks.add("transform_registry_unique", not registry.duplicated(["source_name", "column_name"]).any(), True, not registry.duplicated(["source_name", "column_name"]).any())
    checks.add("transform_registry_phase_not_exposed", not registry["phase_exposed_to_model"].any(), False, bool(registry["phase_exposed_to_model"].any()))
    checks.add("transform_registry_phase_not_required", not registry["phase_required_for_transform"].any(), False, bool(registry["phase_required_for_transform"].any()))

    expected_nonidentity = set(
        schema.loc[
            schema["official_semantic_include"].map(parse_bool)
            & ~schema["official_value_transform_id"].astype(str).isin(IDENTITY_TRANSFORMS),
            "full_column_name",
        ].astype(str)
    )
    observed_nonidentity = set(registry["full_column_name"].astype(str))
    checks.add("transform_registry_covers_all_nonidentity", observed_nonidentity == expected_nonidentity, sorted(expected_nonidentity), sorted(observed_nonidentity))

    if not checks.passed:
        raise SchemaAbort("Transform registry construction failed.")
    return registry


def summarize_schema(schema: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = (
        schema.groupby("source_name", dropna=False)
        .agg(
            total_schema_rows=("column_name", "size"),
            included_features=("official_semantic_include", lambda s: int(pd.Series(s).map(parse_bool).sum())),
            excluded_features=("official_semantic_include", lambda s: int((~pd.Series(s).map(parse_bool)).sum())),
            current_crosschecked_features=("current_in_crosschecked_inputs", lambda s: int(pd.Series(s).map(parse_bool).sum())),
            derived_later_features=("derived_later", lambda s: int(pd.Series(s).map(parse_bool).sum())),
        )
        .reset_index()
        .sort_values("source_name", kind="stable")
        .reset_index(drop=True)
    )
    type_summary = (
        schema.loc[schema["official_semantic_include"].map(parse_bool)]
        .groupby(["official_feature_type", "official_encoding"], dropna=False)
        .size().rename("feature_count").reset_index()
        .sort_values(["feature_count", "official_feature_type"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    origin = (
        schema.groupby(["official_decision_origin", "official_semantic_include"], dropna=False)
        .size().rename("feature_count").reset_index()
        .sort_values(["official_decision_origin", "official_semantic_include"], kind="stable")
        .reset_index(drop=True)
    )
    return source, type_summary, origin


def write_reports(
    paths: Paths,
    checks: Checks,
    summary: Mapping[str, Any],
    schema: pd.DataFrame,
    registry: pd.DataFrame,
    diff: pd.DataFrame,
) -> None:
    checks_frame = pd.DataFrame(
        [
            {
                "check_name": item.name,
                "passed": item.passed,
                "expected": json.dumps(item.expected, ensure_ascii=False, default=str),
                "observed": json.dumps(item.observed, ensure_ascii=False, default=str),
                "details": item.details,
            }
            for item in checks.items
        ]
    )
    atomic_write_csv(checks_frame, paths.safe / "benchmark04b_validation_checks.csv")
    if schema.empty:
        source_summary = pd.DataFrame(columns=["source_name", "total_schema_rows", "included_features", "excluded_features", "current_crosschecked_features", "derived_later_features"])
        type_summary = pd.DataFrame(columns=["official_feature_type", "official_encoding", "feature_count"])
        origin_summary = pd.DataFrame(columns=["official_decision_origin", "official_semantic_include", "feature_count"])
    else:
        source_summary, type_summary, origin_summary = summarize_schema(schema)
    atomic_write_csv(source_summary, paths.safe / "benchmark04b_source_schema_summary.csv")
    atomic_write_csv(type_summary, paths.safe / "benchmark04b_feature_type_summary.csv")
    atomic_write_csv(origin_summary, paths.safe / "benchmark04b_decision_origin_summary.csv")
    if diff.empty:
        diff = pd.DataFrame(columns=["source_name", "column_name", "status", "decision_origin", "legacy_feature_type", "new_feature_type", "legacy_encoding", "new_encoding", "legacy_transform", "new_transform", "semantic_changed_vs_legacy", "encoding_label_changed_vs_legacy"])
    atomic_write_csv(diff.sort_values(["source_name", "column_name"], kind="stable"), paths.safe / "benchmark04b_schema_diff_vs_legacy.csv")
    atomic_write_csv(schema, paths.safe / "benchmark04b_official_schema_safe.csv")
    atomic_write_csv(registry, paths.safe / "benchmark04b_transform_registry_safe.csv")
    atomic_write_json(summary, paths.safe / "benchmark04b_summary.json")

    story = [
        "# BENCHMARK04B — Frozen corrected feature schema",
        "",
        f"- Status: **{'PASS' if summary['status'] == 'PASS' else 'FAIL'}**",
        f"- Schema decision ID: `{summary.get('schema_decision_id', '')}`",
        f"- Current crosschecked features: {summary.get('counts', {}).get('current_crosschecked_features', 'NA')}",
        f"- Derived later: {summary.get('counts', {}).get('derived_later_features', 'NA')}",
        f"- Frozen exclusions represented in schema: {summary.get('counts', {}).get('excluded_features', 'NA')}",
        f"- Semantically included features before scenario-specific support: {summary.get('counts', {}).get('included_features', 'NA')}",
        f"- Support threshold frozen: {EXPECTED_SUPPORT_THRESHOLD} distinct fit participants",
        "",
        "## Interpretation",
        "",
        "A06 directives override legacy semantics only where the CrossCheck explicitly changed a feature. Unchanged crosschecked features inherit their previously frozen semantic decision, including prior exclusions. Physical presence in crosschecked_inputs_v1 does not by itself imply semantic inclusion. BENCHMARK02 heuristic suggestions remain diagnostic only.",
        "",
        "Support 50 is frozen here but is **not applied here**. BENCHMARK05b must recompute support independently for each scenario and temporal block using fit participants only; low-support columns are removed, never visits.",
        "",
        "`AGE_AT_TARGET` is frozen as a numeric visit-derived covariate but is materialized only in BENCHMARK05b from LOCAL_ONLY birth metadata plus the current target visit date.",
        "",
        "No individualized records or MRI values were read by this stage.",
    ]
    (paths.safe / "BENCHMARK04B_SCHEMA_STORY.md").write_text("\n".join(story) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze corrected crosschecked feature schema.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing B-suffixed config outputs.")
    parser.add_argument("--self-test", action="store_true", help="Run lightweight internal contract tests and exit.")
    args = parser.parse_args()

    if args.self_test:
        assert TYPE_TO_ENCODING["categorical"] == "categorical_one_hot"
        assert EXPECTED_SUPPORT_THRESHOLD == 50
        assert "derive_later" in A06_INCLUDE_ACTIONS
        print("BENCHMARK04B self-test PASS")
        return 0

    paths = resolve_paths()
    configure_logging(paths)
    logging.info("BENCHMARK04B v%s", SCRIPT_VERSION)
    checks = Checks()
    fatal_error: str | None = None
    schema = pd.DataFrame()
    registry = pd.DataFrame()
    diff = pd.DataFrame()
    summary: dict[str, Any] = {}

    output_paths = [paths.official_schema_b, paths.official_transforms_b, paths.support_policy_b, paths.schema_lock_b]
    existing = [str(p) for p in output_paths if p.exists()]
    if existing and not args.overwrite:
        logging.error("Refusing to overwrite existing B config outputs without --overwrite: %s", existing)
        return 2

    try:
        a06, b02, a06_manifest = validate_upstream(paths, checks)
        catalog, directives, profile, legacy, legacy_transforms = load_tables(paths, checks)
        schema, diff = build_schema(catalog, directives, profile, legacy, checks)

        upstream_hashes = {
            "a06_catalog_sha256": sha256_file(paths.a06_catalog),
            "a06_directives_sha256": sha256_file(paths.a06_directives),
            "a06_manifest_sha256": sha256_file(paths.a06_manifest),
            "benchmark02_profile_sha256": sha256_file(paths.b02_profile),
            "benchmark02_schema_draft_sha256": sha256_file(paths.b02_schema_draft),
            "legacy_schema_sha256": sha256_file(paths.legacy_schema),
            "legacy_transforms_sha256": sha256_file(paths.legacy_transforms),
        }
        decision_payload = canonical_schema_payload(schema, upstream_hashes)
        decision_id = sha256_text(decision_payload)[:24]

        schema = schema.copy()
        schema.insert(0, "official_schema_version", "2.0.0-crosschecked-b")
        schema.insert(1, "official_schema_frozen", True)
        schema.insert(2, "official_freeze_stage", STAGE_NAME)
        schema.insert(3, "official_decision_id", decision_id)

        registry = build_transform_registry(schema, legacy_transforms, decision_id, checks)
        registry.insert(0, "official_transform_registry_version", "2.0.0-crosschecked-b")
        registry.insert(1, "official_transform_registry_frozen", True)
        registry.insert(2, "official_freeze_stage", STAGE_NAME)

        # Final schema structural checks.
        checks.add("schema_decision_id_single", schema["official_decision_id"].nunique() == 1 and str(schema["official_decision_id"].iloc[0]) == decision_id, decision_id, sorted(schema["official_decision_id"].astype(str).unique().tolist()))
        checks.add("schema_types_allowed", set(schema["official_feature_type"].astype(str).unique()).issubset(ALLOWED_TYPES), sorted(ALLOWED_TYPES), sorted(schema["official_feature_type"].astype(str).unique().tolist()))
        checks.add("schema_support_threshold_50", set(pd.to_numeric(schema["official_support_threshold_participants"], errors="raise").astype(int).unique()) == {50}, [50], sorted(pd.to_numeric(schema["official_support_threshold_participants"], errors="raise").astype(int).unique().tolist()))
        checks.add("all_current_catalog_features_included", int(schema["current_in_crosschecked_inputs"].map(parse_bool).sum()) == EXPECTED_CURRENT_FEATURES, EXPECTED_CURRENT_FEATURES, int(schema["current_in_crosschecked_inputs"].map(parse_bool).sum()))
        checks.add("age_at_target_frozen", bool(((schema["source_name"] == "PTDEMOG") & (schema["column_name"] == "AGE_AT_TARGET") & schema["official_semantic_include"].map(parse_bool) & schema["derived_later"].map(parse_bool)).any()), True, bool(((schema["source_name"] == "PTDEMOG") & (schema["column_name"] == "AGE_AT_TARGET") & schema["official_semantic_include"].map(parse_bool) & schema["derived_later"].map(parse_bool)).any()))

        if not checks.passed:
            raise SchemaAbort("Final schema validation failed.")

        support_policy = {
            "policy_version": "2.0.0-crosschecked-b",
            "frozen": True,
            "freeze_stage": STAGE_NAME,
            "official_decision_id": decision_id,
            "selected_support_threshold_participants": EXPECTED_SUPPORT_THRESHOLD,
            "support_unit": SUPPORT_UNIT,
            "support_application": SUPPORT_APPLICATION,
            "support_filter_removes_visits": False,
            "support_evaluated_before_encoding": True,
            "support_recomputed_at_final_refit": True,
            "statistical_removal_rule": SUPPORT_REMOVAL_RULE,
            "derived_feature_exception": {
                "PTDEMOG__AGE_AT_TARGET": "count distinct fit participants with valid derived age after local derivation and before encoding"
            },
            "methodological_note": "Threshold 50 was frozen before this corrected rerun and is not re-selected in BENCHMARK04B.",
        }

        # Write configs atomically only after all in-memory checks pass.
        atomic_write_csv(schema, paths.official_schema_b)
        atomic_write_csv(registry, paths.official_transforms_b)
        atomic_write_json(support_policy, paths.support_policy_b)

        lock = {
            "schema_lock_version": "2.0.0-crosschecked-b",
            "frozen": True,
            "freeze_stage": STAGE_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "official_decision_id": decision_id,
            "logical_frozen_package": EXPECTED_LOGICAL_PACKAGE,
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_ID,
            "crosscheck_ledger_version": EXPECTED_LEDGER_VERSION,
            "crosscheck_ledger_sha256": EXPECTED_LEDGER_SHA256,
            "a06_script_version": EXPECTED_A06_VERSION,
            "benchmark02b_script_version": EXPECTED_B02_VERSION,
            "selected_support_threshold_participants": EXPECTED_SUPPORT_THRESHOLD,
            "artifacts": {
                "benchmark_feature_schema_b.csv": sha256_file(paths.official_schema_b),
                "benchmark_feature_transform_registry_b.csv": sha256_file(paths.official_transforms_b),
                "benchmark_support_policy_b.json": sha256_file(paths.support_policy_b),
            },
            "upstream_hashes": upstream_hashes,
            "invariants": {
                "participant_split_changed": False,
                "supervised_population_changed": False,
                "mri_targets_changed": False,
                "temporal_policy_changed": False,
                "support_threshold_changed": False,
            },
        }
        atomic_write_json(lock, paths.schema_lock_b)

        # Verify output hashes after write.
        checks.add("output_schema_written", paths.official_schema_b.is_file(), True, paths.official_schema_b.is_file())
        checks.add("output_transforms_written", paths.official_transforms_b.is_file(), True, paths.official_transforms_b.is_file())
        checks.add("output_support_policy_written", paths.support_policy_b.is_file(), True, paths.support_policy_b.is_file())
        checks.add("output_schema_lock_written", paths.schema_lock_b.is_file(), True, paths.schema_lock_b.is_file())
        checks.add("schema_lock_schema_hash", read_json(paths.schema_lock_b)["artifacts"]["benchmark_feature_schema_b.csv"] == sha256_file(paths.official_schema_b), sha256_file(paths.official_schema_b), read_json(paths.schema_lock_b)["artifacts"]["benchmark_feature_schema_b.csv"])

        included_count = int(schema["official_semantic_include"].map(parse_bool).sum())
        excluded_count = int((~schema["official_semantic_include"].map(parse_bool)).sum())
        current_count = int(schema["current_in_crosschecked_inputs"].map(parse_bool).sum())
        derived_count = int(schema["derived_later"].map(parse_bool).sum())

        summary = {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if checks.passed else "FAIL",
            "fatal_error": None,
            "check_count": len(checks.items),
            "failed_check_count": checks.failed,
            "schema_decision_id": decision_id,
            "upstream": {
                "logical_frozen_package": EXPECTED_LOGICAL_PACKAGE,
                "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_ID,
                "crosscheck_ledger_version": EXPECTED_LEDGER_VERSION,
                "crosscheck_ledger_sha256": EXPECTED_LEDGER_SHA256,
                "a06_script_version": EXPECTED_A06_VERSION,
                "benchmark02b_script_version": EXPECTED_B02_VERSION,
            },
            "counts": {
                "schema_rows_total": int(len(schema)),
                "current_crosschecked_features": current_count,
                "derived_later_features": derived_count,
                "included_features": included_count,
                "excluded_features": excluded_count,
                "transform_registry_rows": int(len(registry)),
                "features_with_A06_override_or_directive": int((schema["official_decision_origin"].astype(str).str.startswith("A06_")).sum()),
                "features_preserving_legacy_semantics": int(schema["official_decision_origin"].astype(str).eq("legacy_schema_preserved_for_unchanged_feature").sum()),
                "semantic_changes_vs_legacy_current_features": int(diff.loc[diff["status"].eq("current_crosschecked_feature"), "semantic_changed_vs_legacy"].map(parse_bool).sum()),
            },
            "support_policy": {
                "threshold": EXPECTED_SUPPORT_THRESHOLD,
                "applied_here": False,
                "next_application_stage": "BENCHMARK05b",
                "removes_visits": False,
            },
            "age_at_target": {
                "frozen_in_schema": True,
                "materialized_here": False,
                "materialization_stage": "BENCHMARK05b",
            },
            "outputs": {
                "schema": str(paths.official_schema_b.relative_to(paths.root)),
                "schema_sha256": sha256_file(paths.official_schema_b),
                "transform_registry": str(paths.official_transforms_b.relative_to(paths.root)),
                "transform_registry_sha256": sha256_file(paths.official_transforms_b),
                "support_policy": str(paths.support_policy_b.relative_to(paths.root)),
                "support_policy_sha256": sha256_file(paths.support_policy_b),
                "schema_lock": str(paths.schema_lock_b.relative_to(paths.root)),
                "schema_lock_sha256": sha256_file(paths.schema_lock_b),
            },
            "privacy": {
                "contains_individual_records": False,
                "contains_participant_identifiers": False,
                "contains_visit_identifiers": False,
                "contains_medical_values": False,
                "contains_only_schema_column_names_aggregate_counts_hashes_and_rules": True,
                "SAFE_TO_SHARE": True,
            },
            "methodological_note": (
                "A06 frozen directives have precedence. Unchanged crosschecked features preserve the legacy frozen semantic schema. "
                "BENCHMARK02 heuristics are diagnostic only. Support 50 is frozen but applied later per scenario/block in BENCHMARK05b."
            ),
        }

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("BENCHMARK04B aborted")
        if summary == {}:
            summary = {
                "stage": STAGE_NAME,
                "script_version": SCRIPT_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "FAIL",
                "fatal_error": fatal_error,
                "check_count": len(checks.items),
                "failed_check_count": checks.failed,
                "privacy": {"SAFE_TO_SHARE": True, "contains_individual_records": False},
            }

    write_reports(paths, checks, summary, schema, registry, diff)

    if summary.get("status") == "PASS" and checks.passed:
        logging.info(
            "BENCHMARK04B PASSED | decision_id=%s | schema_rows=%s | included=%s | excluded=%s | support=50",
            summary.get("schema_decision_id"),
            summary.get("counts", {}).get("schema_rows_total"),
            summary.get("counts", {}).get("included_features"),
            summary.get("counts", {}).get("excluded_features"),
        )
        logging.info("No individualized records were read. No model was trained.")
        return 0

    logging.error("BENCHMARK04B FAILED | failed_checks=%s | fatal_error=%s", checks.failed, fatal_error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
