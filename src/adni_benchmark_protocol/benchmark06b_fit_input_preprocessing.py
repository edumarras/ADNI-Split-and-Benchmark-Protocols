#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK06B — ajuste e materialização auditada do pré-processamento das entradas.
#
# Objetivo
# --------
# Consumir exclusivamente os cenários e contratos congelados pelo BENCHMARK05B e
# ajustar, usando somente o split de treino, um pipeline de entradas para cada um
# dos três cenários oficiais:
#
# - snapshot_all
# - snapshot_matched_last
# - longitudinal_previous_last
#
# Política
# --------
# 1. Transformações semânticas: contrato oficial BENCHMARK04B/A06.
# 2. Suporte: manifesto BENCHMARK05B, threshold 50, por cenário/bloco.
# 3. Numéricas: mediana do treino + StandardScaler + indicador de ausência.
# 4. Binárias: mediana do treino, sem scaling + indicador de ausência.
# 5. Categóricas: categoria explícita __MISSING__ + one-hot; unknown ignorado.
# 6. Multi-response: multi-hot por vocabulário do treino; missing explícito;
#    tokens desconhecidos na validação são ignorados e auditados.
# 7. Modalidade: um indicador de presença por fonte/bloco antes da imputação.
# 8. Variância zero: remoção somente após encoding e somente com base no treino.
# 9. Validação: transformada sem reaprender mediana, escala, categorias ou tokens.
# 10. Teste: não materializado, não usado e não consultado para decisões.
#
# Privacidade
# -----------
# Matrizes, preprocessadores, listas de categorias/tokens e row manifests são
# LOCAL ONLY. Relatórios safe contêm somente contagens, dimensões, hashes e
# estatísticas agregadas, sem RID, VISCODE2, datas, categorias observadas ou
# valores médicos.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "benchmark06b"
PREPROCESSING_VERSION: Final[str] = "2.0.0-crosschecked-b"
EXPECTED_BENCHMARK05B_VERSION: Final[str] = "0.1.0"
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"
SUPPORT_THRESHOLD: Final[int] = 50

RID: Final[str] = "RID"
VISCODE2: Final[str] = "VISCODE2"
SPLIT: Final[str] = "split"
TRAIN: Final[str] = "train"
VALIDATION: Final[str] = "validation"
DEVELOPMENT_SPLITS: Final[tuple[str, ...]] = (TRAIN, VALIDATION)

SCENARIOS: Final[tuple[str, ...]] = (
    "snapshot_all",
    "snapshot_matched_last",
    "longitudinal_previous_last",
)
VISIT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
)
PARTICIPANT_SOURCES: Final[tuple[str, ...]] = ("PTDEMOG", "APOERES")
INPUT_SOURCES: Final[tuple[str, ...]] = (*VISIT_SOURCES, *PARTICIPANT_SOURCES)

BENCHMARK05B_SCRIPT: Final[str] = "src/adni_benchmark_protocol/benchmark05b_build_scenarios.py"
BENCHMARK05_SUMMARY: Final[str] = (
    "reports/benchmark05b/safe/benchmark05b_summary.json"
)
BENCHMARK05_CHECKS: Final[str] = (
    "reports/benchmark05b/safe/benchmark05b_validation_checks.csv"
)
OFFICIAL_SCHEMA: Final[str] = "config/benchmark_feature_schema_b.csv"
OFFICIAL_TRANSFORMS: Final[str] = (
    "config/benchmark_feature_transform_registry_b.csv"
)
SUPPORT_POLICY: Final[str] = "config/benchmark_support_policy_b.json"
SCHEMA_LOCK: Final[str] = "config/benchmark_schema_lock_b.json"
SCENARIO_DEFINITIONS: Final[str] = "config/benchmark_scenario_definitions_b.json"
SCENARIO_SUPPORT_MANIFEST: Final[str] = (
    "config/benchmark_scenario_support_manifest_b.csv"
)
SCENARIO_LOCK: Final[str] = "config/benchmark_scenario_lock_b.json"
BENCHMARK05B_LOCAL: Final[str] = "data/processed/local_only/benchmark05b_scenarios"
AUDIT_A05_SUMMARY: Final[str] = (
    "reports/auditA05_temporal_modality_exclusions/safe/auditA05_summary.json"
)
AUDIT_A05_LOCAL_MANIFEST: Final[str] = (
    "reports/auditA05_temporal_modality_exclusions/local_only/"
    "LOCAL_ONLY_temporal_modality_exclusions.csv"
)
TEMPORAL_EXCLUSION_FLAG_PREFIX: Final[str] = "exclude_previous__"

PREPROCESSING_POLICY: Final[str] = "config/benchmark_preprocessing_policy_b.json"
PREPROCESSING_MANIFEST: Final[str] = (
    "config/benchmark_preprocessing_manifest_b.csv"
)
PREPROCESSING_LOCK: Final[str] = "config/benchmark_preprocessing_lock_b.json"
LOCAL_OUTPUT_DIR: Final[str] = "data/processed/local_only/benchmark06b_preprocessed_inputs"

SCENARIO_ARTIFACTS: Final[dict[str, str]] = {
    "snapshot_all": "snapshot_all_development_anchor.csv.gz",
    "snapshot_matched_last": "snapshot_matched_last_development_anchor.csv.gz",
    "longitudinal_previous_last": (
        "longitudinal_previous_last_development_pairs.csv.gz"
    ),
}

DERIVED_FEATURES: Final[frozenset[str]] = frozenset({"PTDEMOG__AGE_AT_TARGET"})

class PreprocessingAbort(RuntimeError):
    # Fail-closed exception for preprocessing incompatibility.
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    safe: Path
    logs: Path
    local: Path
    benchmark05b_script: Path
    benchmark05b_summary: Path
    benchmark05b_checks: Path
    official_schema: Path
    official_transforms: Path
    support_policy: Path
    schema_lock: Path
    scenario_definitions: Path
    scenario_support_manifest: Path
    scenario_lock: Path
    benchmark05b_local: Path
    temporal_exclusion_summary: Path
    temporal_exclusion_manifest: Path
    preprocessing_policy: Path
    preprocessing_manifest: Path
    preprocessing_lock: Path


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
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                name,
                short_text(expected),
                short_text(observed),
                f" | {details}" if details else "",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit train-only input preprocessing for the three frozen "
            "BENCHMARK05B scenarios."
        )
    )
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing BENCHMARK06B local output after validation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path(__file__).resolve().parents[2]
    package = (
        args.package_root.resolve()
        if args.package_root is not None
        else root / "data" / "processed" / "frozen_experiment_v1_1"
    )
    reports_root = (
        args.reports_root.resolve()
        if args.reports_root is not None
        else root / "reports"
    )
    processed_root = (
        args.processed_root.resolve()
        if args.processed_root is not None
        else root / "data" / "processed" / "local_only"
    )
    stage = reports_root / STAGE_NAME
    return Paths(
        root=root,
        package=package,
        safe=stage / "safe",
        logs=stage / "logs",
        local=processed_root / "benchmark06b_preprocessed_inputs",
        benchmark05b_script=root / BENCHMARK05B_SCRIPT,
        benchmark05b_summary=root / BENCHMARK05_SUMMARY,
        benchmark05b_checks=root / BENCHMARK05_CHECKS,
        official_schema=root / OFFICIAL_SCHEMA,
        official_transforms=root / OFFICIAL_TRANSFORMS,
        support_policy=root / SUPPORT_POLICY,
        schema_lock=root / SCHEMA_LOCK,
        scenario_definitions=root / SCENARIO_DEFINITIONS,
        scenario_support_manifest=root / SCENARIO_SUPPORT_MANIFEST,
        scenario_lock=root / SCENARIO_LOCK,
        benchmark05b_local=root / BENCHMARK05B_LOCAL,
        temporal_exclusion_summary=root / AUDIT_A05_SUMMARY,
        temporal_exclusion_manifest=root / AUDIT_A05_LOCAL_MANIFEST,
        preprocessing_policy=root / PREPROCESSING_POLICY,
        preprocessing_manifest=root / PREPROCESSING_MANIFEST,
        preprocessing_lock=root / PREPROCESSING_LOCK,
    )


def configure_logging(paths: Paths) -> None:
    paths.safe.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)

    # pandas 3.x may warn when a wide DataFrame is assembled incrementally.
    # This is a performance-only warning and does not affect values or column
    # order. Suppress it here so the audit log remains readable.
    warnings.filterwarnings(
        "ignore",
        category=pd.errors.PerformanceWarning,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "benchmark06b.log", encoding="utf-8"),
        ],
        force=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    raise PreprocessingAbort(f"Cannot interpret boolean value: {value!r}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PreprocessingAbort(f"{label} is missing required columns: {missing}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n", na_rep="")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_benchmark05b_module(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location(
        "benchmark05b_contract_module",
        path,
    )
    if specification is None or specification.loader is None:
        raise PreprocessingAbort(f"Cannot import BENCHMARK05B from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def make_one_hot_encoder() -> OneHotEncoder:
    parameters = inspect.signature(OneHotEncoder).parameters
    kwargs: dict[str, Any] = {
        "handle_unknown": "ignore",
        "dtype": np.float32,
    }
    if "sparse_output" in parameters:
        kwargs["sparse_output"] = True
    else:  # pragma: no cover - old scikit-learn compatibility
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def canonical_value_series(series: pd.Series) -> pd.Series:
    output = series.astype("string")
    return output.mask(output.str.strip().eq(""), pd.NA)


def normalize_rid_key(series: pd.Series, context: str) -> pd.Series:
    # Normalize RID as an integer identifier represented by pandas string.
    # CSV inference may load the same RID column as int64 in a scenario manifest
    # and as string in a model-ready source. This gives both sides the same merge
    # representation without changing identifier semantics.
    raw = series.astype("string").str.strip()
    raw = raw.mask(raw.eq(""), pd.NA)
    numeric = pd.to_numeric(raw, errors="coerce")

    invalid = raw.notna() & numeric.isna()
    if invalid.any():
        raise PreprocessingAbort(
            f"{context} contains {int(invalid.sum())} observed non-numeric RID values."
        )

    non_integer = numeric.notna() & numeric.ne(numeric.round())
    if non_integer.any():
        raise PreprocessingAbort(
            f"{context} contains {int(non_integer.sum())} non-integer RID values."
        )

    return numeric.round().astype("Int64").astype("string")


def normalize_viscode_key(series: pd.Series) -> pd.Series:
    output = series.astype("string").str.strip().str.casefold()
    return output.mask(output.eq(""), pd.NA)


def normalize_split_key(series: pd.Series) -> pd.Series:
    output = series.astype("string").str.strip().str.casefold()
    return output.mask(output.eq(""), pd.NA)


def temporal_exclusion_flag(source_name: str) -> str:
    return f"{TEMPORAL_EXCLUSION_FLAG_PREFIX}{source_name}"


def expected_rows_from_benchmark05(
    benchmark05b_summary: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {
        scenario: {} for scenario in SCENARIOS
    }
    for row in benchmark05b_summary.get("scenario_counts", []):
        scenario = str(row.get("scenario", ""))
        split_label = str(row.get("split", ""))
        if scenario in output and split_label in DEVELOPMENT_SPLITS:
            output[scenario][split_label] = int(row.get("rows", -1))
    missing = {
        scenario: [
            split_label
            for split_label in DEVELOPMENT_SPLITS
            if split_label not in output[scenario]
        ]
        for scenario in SCENARIOS
    }
    missing = {key: value for key, value in missing.items() if value}
    if missing:
        raise PreprocessingAbort(
            f"BENCHMARK05B summary lacks scenario row counts: {missing}"
        )
    return output


def validate_temporal_exclusion_flags(
    manifest: pd.DataFrame,
    paths: Paths,
    benchmark05b_summary: Mapping[str, Any],
    checks: Checks,
) -> None:
    required_flags = [
        temporal_exclusion_flag(source_name)
        for source_name in VISIT_SOURCES
    ]
    require_columns(
        manifest,
        [
            RID,
            SPLIT,
            "current_VISCODE2",
            "previous_VISCODE2",
            "temporal_source_exclusion_applied",
            "temporal_source_exclusion_action_count",
            *required_flags,
        ],
        "longitudinal temporal exclusion flags",
    )

    summary = read_json(paths.temporal_exclusion_summary)
    expected_hash = str(
        summary.get("local_only_manifest", {}).get("sha256", "")
    )
    actual_hash = sha256_file(paths.temporal_exclusion_manifest)
    upstream_hash = str(
        benchmark05b_summary.get("artifacts", {}).get(
            "temporal_exclusion_local_manifest_sha256", ""
        )
    )
    hashes_match = bool(
        expected_hash
        and actual_hash == expected_hash
        and actual_hash == upstream_hash
    )
    checks.add(
        "temporal_exclusion_manifest_hash_chain",
        hashes_match,
        True,
        hashes_match,
    )
    audit_passed = bool(summary.get("audit_passed"))
    checks.add(
        "temporal_exclusion_audit_passed",
        audit_passed,
        True,
        audit_passed,
    )

    actions = pd.read_csv(
        paths.temporal_exclusion_manifest,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )
    require_columns(
        actions,
        [
            RID,
            SPLIT,
            "current_VISCODE2",
            "previous_VISCODE2",
            "source_name",
        ],
        "AUDIT A05 local manifest",
    )
    actions[RID] = normalize_rid_key(actions[RID], "AUDIT A05")
    actions[SPLIT] = normalize_split_key(actions[SPLIT])
    actions["current_VISCODE2"] = normalize_viscode_key(
        actions["current_VISCODE2"]
    )
    actions["previous_VISCODE2"] = normalize_viscode_key(
        actions["previous_VISCODE2"]
    )
    actions["source_name"] = (
        actions["source_name"].astype("string").str.strip().str.upper()
    )

    base = manifest[
        [RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"]
    ].copy()
    base[RID] = normalize_rid_key(base[RID], "BENCHMARK05B longitudinal manifest")
    base[SPLIT] = normalize_split_key(base[SPLIT])
    base["current_VISCODE2"] = normalize_viscode_key(
        base["current_VISCODE2"]
    )
    base["previous_VISCODE2"] = normalize_viscode_key(
        base["previous_VISCODE2"]
    )
    base["__row_order"] = np.arange(len(base), dtype=np.int64)

    all_flags_match = True
    for source_name in VISIT_SOURCES:
        flag_column = temporal_exclusion_flag(source_name)
        expected = actions.loc[
            actions["source_name"].eq(source_name),
            [RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"],
        ].copy()
        expected["__expected_flag"] = True
        merged = base.merge(
            expected,
            on=[RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("__row_order", kind="stable")
        expected_flag = (
            merged["__expected_flag"].map(parse_bool).to_numpy(dtype=bool)
        )
        observed_flag = manifest[flag_column].map(parse_bool).to_numpy(dtype=bool)
        all_flags_match = all_flags_match and bool(
            np.array_equal(expected_flag, observed_flag)
        )

    observed_flags = manifest[required_flags].apply(
        lambda column: column.map(parse_bool)
    )
    observed_any = observed_flags.any(axis=1)
    observed_count = observed_flags.sum(axis=1).astype(int)
    declared_any = manifest[
        "temporal_source_exclusion_applied"
    ].map(parse_bool)
    declared_count = pd.to_numeric(
        manifest["temporal_source_exclusion_action_count"],
        errors="coerce",
    ).fillna(-1).astype(int)
    internal_flags_consistent = bool(
        observed_any.eq(declared_any).all()
        and observed_count.eq(declared_count).all()
    )
    checks.add(
        "temporal_exclusion_flags_match_audit_manifest",
        all_flags_match,
        True,
        all_flags_match,
    )
    checks.add(
        "temporal_exclusion_flag_columns_internally_consistent",
        internal_flags_consistent,
        True,
        internal_flags_consistent,
    )
    if not checks.passed:
        raise PreprocessingAbort(
            "BENCHMARK05B temporal exclusion flags failed cross-stage validation."
        )


def aligned_source_rows(
    scenario: str,
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    source_name: str,
    block: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Align one source and apply frozen temporal block exclusions.

    For longitudinal previous-visit blocks, all source features are set to
    missing before modality presence, semantic transformation, imputation or
    encoding whenever BENCHMARK05B marked that source as temporally unsafe.
    """
    manifest_work = manifest.reset_index(drop=True).copy()
    manifest_work["__row_order"] = np.arange(len(manifest_work), dtype=np.int64)

    require_columns(manifest_work, [RID, SPLIT], f"{source_name}/{block} manifest")
    require_columns(source, [RID, SPLIT, *columns], f"{source_name}/{block} source")

    manifest_work[RID] = normalize_rid_key(
        manifest_work[RID], f"{source_name}/{block} manifest"
    )
    manifest_work[SPLIT] = normalize_split_key(manifest_work[SPLIT])

    source_work = source[[RID, SPLIT, *columns]].copy()
    source_work[RID] = normalize_rid_key(
        source_work[RID], f"{source_name}/{block} source"
    )
    source_work[SPLIT] = normalize_split_key(source_work[SPLIT])

    temporal_flag_column: str | None = None
    if (
        scenario == "longitudinal_previous_last"
        and block == "previous_visit"
        and source_name in VISIT_SOURCES
    ):
        temporal_flag_column = temporal_exclusion_flag(source_name)
        require_columns(
            manifest_work,
            [temporal_flag_column],
            f"{source_name}/{block} temporal exclusion flag",
        )

    if source_name in VISIT_SOURCES:
        visit_column = (
            "previous_VISCODE2" if block == "previous_visit" else "current_VISCODE2"
        )
        require_columns(
            manifest_work,
            [RID, SPLIT, visit_column],
            f"{source_name}/{block} manifest",
        )
        require_columns(source, [VISCODE2], f"{source_name}/{block} source")

        manifest_work[visit_column] = normalize_viscode_key(
            manifest_work[visit_column]
        )
        source_work[VISCODE2] = normalize_viscode_key(source[VISCODE2])

        key_columns = [RID, SPLIT, visit_column, "__row_order"]
        if temporal_flag_column is not None:
            key_columns.append(temporal_flag_column)
        keys = manifest_work[key_columns].rename(columns={visit_column: VISCODE2})
        if keys.duplicated([RID, VISCODE2]).any():
            raise PreprocessingAbort(
                f"Duplicate visit keys in {source_name}/{block} manifest."
            )
        if source_work.duplicated([RID, VISCODE2, SPLIT]).any():
            raise PreprocessingAbort(
                f"Duplicate normalized source keys in {source_name}/{block}."
            )

        aligned = keys.merge(
            source_work[[RID, VISCODE2, SPLIT, *columns]],
            on=[RID, VISCODE2, SPLIT],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    else:
        keys = manifest_work[[RID, SPLIT, "__row_order"]]
        if source_work.duplicated([RID, SPLIT]).any():
            raise PreprocessingAbort(
                f"Duplicate normalized participant keys in {source_name}/{block}."
            )
        aligned = keys.merge(
            source_work[[RID, SPLIT, *columns]],
            on=[RID, SPLIT],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    aligned = aligned.sort_values("__row_order", kind="stable").reset_index(drop=True)
    if len(aligned) != len(manifest_work):
        raise PreprocessingAbort(
            f"Alignment changed row count for {source_name}/{block}: "
            f"{len(manifest_work)} -> {len(aligned)}"
        )

    if temporal_flag_column is not None:
        unsafe = aligned[temporal_flag_column].map(parse_bool).to_numpy(dtype=bool)
        if unsafe.any():
            aligned.loc[unsafe, list(columns)] = pd.NA
        aligned = aligned.drop(columns=[temporal_flag_column])

    return aligned.drop(columns=["__row_order"])


def model_feature_name(
    scenario: str,
    block: str,
    source_name: str,
    column_name: str,
) -> str:
    base = f"{source_name}__{column_name}"
    if scenario == "longitudinal_previous_last":
        if block == "current_visit":
            return f"current__{base}"
        if block == "previous_visit":
            return f"previous__{base}"
        if block == "current_derived":
            return f"current__{base}"
    return base


def modality_presence_name(
    scenario: str,
    block: str,
    source_name: str,
) -> str:
    if block == "current_derived":
        raise PreprocessingAbort(
            "Derived visit covariates do not receive modality-presence indicators."
        )
    if scenario == "longitudinal_previous_last":
        temporal = {
            "current_visit": "current",
            "previous_visit": "previous",
            "participant_static": "static",
        }[block]
        return f"modality_present__{temporal}__{source_name}"
    return f"modality_present__{source_name}"


def build_raw_scenario(
    scenario: str,
    manifest: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
    schema: pd.DataFrame,
    registry: pd.DataFrame,
    support_manifest: pd.DataFrame,
    expected_rows: Mapping[str, int],
    b5: Any,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create transformed-but-not-imputed model inputs for one scenario."""
    scenario_support = support_manifest.loc[
support_manifest["scenario"].eq(scenario)
        & support_manifest["retained_at_support_threshold"]
    ].copy()
    if scenario_support.empty:
        raise PreprocessingAbort(f"No retained support rows for {scenario}.")

    raw = pd.DataFrame(index=np.arange(len(manifest)))
    feature_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    registry_map = b5.registry_lookup(registry)

    standard_support = scenario_support.loc[
        ~scenario_support["block"].astype(str).eq("current_derived")
    ].copy()
    derived_support = scenario_support.loc[
        scenario_support["block"].astype(str).eq("current_derived")
    ].copy()

    grouped = standard_support.groupby(
        ["block", "source_name"],
        sort=True,
        dropna=False,
    )
    for (block_raw, source_raw), retained_rows in grouped:
        block = str(block_raw)
        source_name = str(source_raw)
        if source_name not in source_frames:
            raise PreprocessingAbort(f"Source frame missing: {source_name}")

        source_schema_all = schema.loc[
            schema["source_name"].eq(source_name)
            & schema["official_semantic_include"]
            & schema["current_in_crosschecked_inputs"]
            & ~schema["derived_later"]
        ].copy()
        all_authorized_columns = source_schema_all["column_name"].astype(str).tolist()
        retained_columns = retained_rows["column_name"].astype(str).tolist()
        aligned = aligned_source_rows(
            scenario,
            manifest,
            source_frames[source_name],
            source_name,
            block,
            all_authorized_columns,
        )

        # Source presence is measured before support filtering and before imputation.
        present_name = modality_presence_name(scenario, block, source_name)
        presence = aligned[all_authorized_columns].notna().any(axis=1).astype(np.float32)
        if present_name in raw.columns:
            raise PreprocessingAbort(f"Duplicate modality indicator: {present_name}")
        raw[present_name] = presence.to_numpy(dtype=np.float32)
        feature_rows.append(
            {
                "scenario": scenario,
                "block": block,
                "source_name": source_name,
                "column_name": "<MODALITY_PRESENT>",
                "full_column_name": "<MODALITY_PRESENT>",
                "model_feature_name": present_name,
                "feature_type": "presence",
                "encoding": "binary_identity",
                "transform_id": "identity",
                "train_observed_participants": int(
                    retained_rows["train_participant_count"].max()
                ),
                "support_threshold_participants": SUPPORT_THRESHOLD,
            }
        )

        for split_label in DEVELOPMENT_SPLITS:
            split_mask = manifest[SPLIT].astype(str).eq(split_label)
            split_presence = presence.loc[split_mask.to_numpy()]
            modality_rows.append(
                {
                    "scenario": scenario,
                    "block": block,
                    "source_name": source_name,
                    "split": split_label,
                    "rows": int(split_mask.sum()),
                    "modality_present_count": int(split_presence.sum()),
                    "modality_present_pct": (
                        100.0 * float(split_presence.mean())
                        if len(split_presence)
                        else np.nan
                    ),
                }
            )

        source_schema_by_column = {
            str(row["column_name"]): row
            for row in source_schema_all.to_dict(orient="records")
        }
        support_by_column = {
            str(row["column_name"]): row
            for row in retained_rows.to_dict(orient="records")
        }

        for column_name in retained_columns:
            if column_name not in source_schema_by_column:
                raise PreprocessingAbort(
                    f"Retained feature absent from schema: {source_name}__{column_name}"
                )
            schema_row = source_schema_by_column[column_name]
            support_row = support_by_column[column_name]
            full_name = str(schema_row["full_column_name"])
            feature_type = str(schema_row["official_feature_type"])
            transform_id = str(schema_row["official_value_transform_id"] or "identity")
            parameters: Mapping[str, Any] = {}
            if full_name in registry_map:
                transform_id = str(registry_map[full_name]["transform_id"])
                parameters = registry_map[full_name]["parameters"]

            transformed, transform_failures, type_failures = b5.transform_and_validate_type(
                aligned[column_name],
                feature_type,
                transform_id,
                parameters,
                full_name,
            )
            checks.add(
                f"transform_valid__{scenario}__{block}__{full_name}",
                transform_failures == 0 and type_failures == 0,
                {"transform_failures": 0, "type_failures": 0},
                {
                    "transform_failures": int(transform_failures),
                    "type_failures": int(type_failures),
                },
            )
            model_name = model_feature_name(
                scenario,
                block,
                source_name,
                column_name,
            )
            if model_name in raw.columns:
                raise PreprocessingAbort(f"Duplicate model feature name: {model_name}")
            raw[model_name] = transformed.reset_index(drop=True)
            feature_rows.append(
                {
                    "scenario": scenario,
                    "block": block,
                    "source_name": source_name,
                    "column_name": column_name,
                    "full_column_name": full_name,
                    "model_feature_name": model_name,
                    "feature_type": feature_type,
                    "encoding": str(schema_row["official_encoding"]),
                    "transform_id": transform_id,
                    "train_observed_participants": int(
                        support_row["train_observed_participants"]
                    ),
                    "support_threshold_participants": int(
                        support_row["support_threshold_participants"]
                    ),
                }
            )

    # Visit-specific covariates already derived and audited by BENCHMARK05B
    # (currently AGE_AT_TARGET) are read directly from the scenario manifest.
    # They must never be looked up in participant-static PTDEMOG.
    for support_row in derived_support.to_dict(orient="records"):
        block = str(support_row["block"])
        source_name = str(support_row["source_name"])
        column_name = str(support_row["column_name"])
        full_name = str(support_row["full_column_name"])
        if full_name not in DERIVED_FEATURES:
            raise PreprocessingAbort(
                f"Unexpected derived feature in support manifest: {full_name}"
            )
        if block != "current_derived":
            raise PreprocessingAbort(
                f"Derived feature has unexpected block {block!r}: {full_name}"
            )
        require_columns(manifest, [column_name], f"{scenario} derived manifest")
        schema_rows = schema.loc[
            schema["full_column_name"].eq(full_name)
            & schema["official_semantic_include"]
            & schema["derived_later"]
        ].copy()
        if len(schema_rows) != 1:
            raise PreprocessingAbort(
                f"Expected one derived schema row for {full_name}; found {len(schema_rows)}."
            )
        schema_row = schema_rows.iloc[0]
        feature_type = str(schema_row["official_feature_type"])
        if feature_type != "numeric":
            raise PreprocessingAbort(
                f"Derived feature {full_name} must be numeric, got {feature_type!r}."
            )
        values = pd.to_numeric(manifest[column_name], errors="coerce")
        original_nonmissing = int(manifest[column_name].notna().sum())
        converted_nonmissing = int(values.notna().sum())
        checks.add(
            f"derived_numeric_valid__{scenario}__{full_name}",
            original_nonmissing == converted_nonmissing,
            original_nonmissing,
            converted_nonmissing,
        )
        model_name = model_feature_name(
            scenario, block, source_name, column_name
        )
        if model_name in raw.columns:
            raise PreprocessingAbort(f"Duplicate model feature name: {model_name}")
        raw[model_name] = values.reset_index(drop=True).astype("Float64")
        feature_rows.append(
            {
                "scenario": scenario,
                "block": block,
                "source_name": source_name,
                "column_name": column_name,
                "full_column_name": full_name,
                "model_feature_name": model_name,
                "feature_type": feature_type,
                "encoding": str(schema_row["official_encoding"]),
                "transform_id": str(schema_row["official_value_transform_id"]),
                "train_observed_participants": int(
                    support_row["train_observed_participants"]
                ),
                "support_threshold_participants": int(
                    support_row["support_threshold_participants"]
                ),
            }
        )

    if scenario == "longitudinal_previous_last":
        require_columns(manifest, ["gap_months"], "longitudinal manifest")
        gap = pd.to_numeric(manifest["gap_months"], errors="coerce")
        if not np.isfinite(gap.to_numpy(dtype=float)).all() or not gap.gt(0).all():
            raise PreprocessingAbort("Longitudinal gap_months is not finite and positive.")
        raw["gap_months"] = gap.to_numpy(dtype=np.float64)
        feature_rows.append(
            {
                "scenario": scenario,
                "block": "temporal",
                "source_name": "TEMPORAL",
                "column_name": "gap_months",
                "full_column_name": "TEMPORAL__gap_months",
                "model_feature_name": "gap_months",
                "feature_type": "numeric",
                "encoding": "numeric_standardized",
                "transform_id": "identity",
                "train_observed_participants": int(
                    manifest.loc[manifest[SPLIT].eq(TRAIN), RID].nunique()
                ),
                "support_threshold_participants": SUPPORT_THRESHOLD,
            }
        )

    feature_manifest = pd.DataFrame(feature_rows)
    if feature_manifest["model_feature_name"].duplicated().any():
        duplicates = feature_manifest.loc[
            feature_manifest["model_feature_name"].duplicated(),
            "model_feature_name",
        ].tolist()
        raise PreprocessingAbort(f"Duplicate raw model features: {duplicates[:20]}")
    if set(raw.columns) != set(feature_manifest["model_feature_name"]):
        raise PreprocessingAbort("Raw matrix columns do not match feature manifest.")

    row_manifest = manifest.copy().reset_index(drop=True)
    split_values = row_manifest[SPLIT].astype("string").str.strip()
    checks.add(
        f"development_only__{scenario}",
        set(split_values.dropna().astype(str).unique()) == set(DEVELOPMENT_SPLITS),
        list(DEVELOPMENT_SPLITS),
        sorted(split_values.dropna().astype(str).unique().tolist()),
    )
    row_manifest[SPLIT] = split_values
    raw[SPLIT] = split_values.to_numpy()

    train = raw.loc[raw[SPLIT].eq(TRAIN)].drop(columns=[SPLIT]).reset_index(drop=True)
    validation = raw.loc[raw[SPLIT].eq(VALIDATION)].drop(columns=[SPLIT]).reset_index(drop=True)
    train_rows = row_manifest.loc[row_manifest[SPLIT].eq(TRAIN)].reset_index(drop=True)
    validation_rows = row_manifest.loc[
        row_manifest[SPLIT].eq(VALIDATION)
    ].reset_index(drop=True)

    expected = expected_rows
    checks.add(
        f"train_row_count__{scenario}",
        len(train) == expected[TRAIN],
        expected[TRAIN],
        len(train),
    )
    checks.add(
        f"validation_row_count__{scenario}",
        len(validation) == expected[VALIDATION],
        expected[VALIDATION],
        len(validation),
    )

    return (
        train,
        validation,
        feature_manifest,
        pd.DataFrame(modality_rows),
    )


def numeric_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float64)

    converted = frame[list(columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # pandas 3.x can expose a read-only NumPy view. The preprocessing stage
    # normalizes non-finite values in place, so force an independent writable
    # C-contiguous array at this boundary.
    values = np.array(
        converted.to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        dtype=np.float64,
        copy=True,
        order="C",
    )

    values[~np.isfinite(values)] = np.nan
    return values


def categorical_array(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    converted = frame[list(columns)].astype("object")
    converted = converted.where(pd.notna(converted), "__MISSING__")
    for column in converted.columns:
        converted[column] = converted[column].map(
            lambda value: (
                "__MISSING__"
                if value is None or not str(value).strip()
                else str(value).strip()
            )
        )
    return converted.to_numpy(dtype=object)


def fit_medians(values: np.ndarray, label: str) -> np.ndarray:
    if values.shape[1] == 0:
        return np.empty((0,), dtype=np.float64)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=0)
    if not np.isfinite(medians).all():
        bad = int((~np.isfinite(medians)).sum())
        raise PreprocessingAbort(
            f"{label} contains {bad} retained columns without a finite median."
        )
    return medians


def impute_with_medians(
    values: np.ndarray,
    medians: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    missing = np.isnan(values)
    imputed = values.copy()
    if imputed.size:
        rows, columns = np.where(missing)
        imputed[rows, columns] = medians[columns]
    return imputed, missing.astype(np.float32)


def token_lists_for_series(series: pd.Series, b5: Any) -> list[list[str]]:
    return [b5.parse_multi_tokens(value) for value in series.tolist()]


def multi_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    vocabularies: Mapping[str, Sequence[str]],
    b5: Any,
) -> tuple[sparse.csr_matrix, dict[str, int], dict[str, int]]:
    blocks: list[sparse.csr_matrix] = []
    unknown_token_occurrences: dict[str, int] = {}
    unknown_rows: dict[str, int] = {}

    for column in columns:
        vocabulary = list(vocabularies[column])
        token_to_index = {token: index for index, token in enumerate(vocabulary)}
        missing_index = len(vocabulary)
        unknown_index = len(vocabulary) + 1
        width = len(vocabulary) + 2
        row_indices: list[int] = []
        column_indices: list[int] = []
        unknown_occurrences = 0
        unknown_row_count = 0

        token_lists = token_lists_for_series(frame[column], b5)
        for row_index, tokens in enumerate(token_lists):
            if not tokens:
                row_indices.append(row_index)
                column_indices.append(missing_index)
                continue
            unknown_found = False
            for token in sorted(set(tokens)):
                token_index = token_to_index.get(token)
                if token_index is None:
                    unknown_occurrences += 1
                    unknown_found = True
                else:
                    row_indices.append(row_index)
                    column_indices.append(token_index)
            if unknown_found:
                unknown_row_count += 1
                row_indices.append(row_index)
                column_indices.append(unknown_index)

        data = np.ones(len(row_indices), dtype=np.float32)
        blocks.append(
            sparse.csr_matrix(
                (data, (row_indices, column_indices)),
                shape=(len(frame), width),
                dtype=np.float32,
            )
        )
        unknown_token_occurrences[column] = int(unknown_occurrences)
        unknown_rows[column] = int(unknown_row_count)

    matrix = (
        sparse.hstack(blocks, format="csr")
        if blocks
        else sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    )
    return matrix, unknown_token_occurrences, unknown_rows


def expanded_feature_names_before_variance(
    state: Mapping[str, Any],
) -> list[str]:
    names: list[str] = []
    names.extend(f"numeric_scaled::{column}" for column in state["numeric_columns"])
    names.extend(f"missing_numeric::{column}" for column in state["numeric_columns"])
    names.extend(f"binary::{column}" for column in state["binary_columns"])
    names.extend(f"missing_binary::{column}" for column in state["binary_columns"])

    encoder = state.get("categorical_encoder")
    if encoder is not None:
        names.extend(
            f"categorical::{name}"
            for name in encoder.get_feature_names_out(
                state["categorical_columns"]
            ).astype(str).tolist()
        )

    for column in state["multi_columns"]:
        vocabulary = state["multi_vocabularies"][column]
        names.extend(f"multi::{column}::{token}" for token in vocabulary)
        names.append(f"multi::{column}::__MISSING__")
        names.append(f"multi::{column}::__UNKNOWN__")

    names.extend(f"presence::{column}" for column in state["presence_columns"])
    return names


def build_encoded_parts(
    frame: pd.DataFrame,
    state: Mapping[str, Any],
    b5: Any,
    fit: bool,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    blocks: list[sparse.csr_matrix] = []
    audit: dict[str, Any] = {
        "categorical_unknown_counts": {},
        "multi_unknown_token_occurrences": {},
        "multi_unknown_rows": {},
    }

    numeric_columns = state["numeric_columns"]
    numeric_raw = numeric_array(frame, numeric_columns)
    if numeric_columns:
        numeric_imputed, numeric_missing = impute_with_medians(
            numeric_raw,
            state["numeric_medians"],
        )
        numeric_scaled = state["numeric_scaler"].transform(numeric_imputed)
        blocks.append(sparse.csr_matrix(numeric_scaled.astype(np.float32)))
        blocks.append(sparse.csr_matrix(numeric_missing, dtype=np.float32))

    binary_columns = state["binary_columns"]
    binary_raw = numeric_array(frame, binary_columns)
    if binary_columns:
        observed = binary_raw[np.isfinite(binary_raw)]
        invalid = int((~np.isin(observed, [0.0, 1.0])).sum())
        if invalid:
            raise PreprocessingAbort(
                f"Binary block contains {invalid} values outside 0/1."
            )
        binary_imputed, binary_missing = impute_with_medians(
            binary_raw,
            state["binary_medians"],
        )
        blocks.append(sparse.csr_matrix(binary_imputed.astype(np.float32)))
        blocks.append(sparse.csr_matrix(binary_missing, dtype=np.float32))

    categorical_columns = state["categorical_columns"]
    if categorical_columns:
        categorical = categorical_array(frame, categorical_columns)
        encoder: OneHotEncoder = state["categorical_encoder"]
        encoded = encoder.transform(categorical)
        blocks.append(sparse.csr_matrix(encoded, dtype=np.float32))
        if not fit:
            for index, column in enumerate(categorical_columns):
                known = set(str(value) for value in encoder.categories_[index])
                values = pd.Series(categorical[:, index], dtype="string")
                audit["categorical_unknown_counts"][column] = int(
                    (~values.isin(known)).sum()
                )

    multi_encoded, unknown_occurrences, unknown_rows = multi_matrix(
        frame,
        state["multi_columns"],
        state["multi_vocabularies"],
        b5,
    )
    blocks.append(multi_encoded)
    audit["multi_unknown_token_occurrences"] = unknown_occurrences
    audit["multi_unknown_rows"] = unknown_rows

    presence_columns = state["presence_columns"]
    if presence_columns:
        presence = (
            frame[presence_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        invalid_presence = int((~np.isin(presence, [0.0, 1.0])).sum())
        if invalid_presence:
            raise PreprocessingAbort(
                f"Presence block contains {invalid_presence} values outside 0/1."
            )
        blocks.append(sparse.csr_matrix(presence, dtype=np.float32))

    matrix = (
        sparse.hstack(blocks, format="csr")
        if blocks
        else sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    )
    return matrix, audit


def fit_preprocessor_state(
    train: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    b5: Any,
) -> tuple[dict[str, Any], sparse.csr_matrix, dict[str, Any]]:
    type_map = feature_manifest.set_index("model_feature_name")["feature_type"].to_dict()
    numeric_columns = [
        column for column in train.columns if type_map.get(column) == "numeric"
    ]
    binary_columns = [
        column for column in train.columns if type_map.get(column) == "binary"
    ]
    categorical_columns = [
        column for column in train.columns if type_map.get(column) == "categorical"
    ]
    multi_columns = [
        column for column in train.columns if type_map.get(column) == "multi_response"
    ]
    presence_columns = [
        column for column in train.columns if type_map.get(column) == "presence"
    ]

    numeric_raw = numeric_array(train, numeric_columns)
    numeric_medians = fit_medians(numeric_raw, "Numeric block")
    numeric_scaler: StandardScaler | None = None
    if numeric_columns:
        numeric_imputed, _ = impute_with_medians(numeric_raw, numeric_medians)
        numeric_scaler = StandardScaler(with_mean=True, with_std=True)
        numeric_scaler.fit(numeric_imputed)

    binary_raw = numeric_array(train, binary_columns)
    binary_medians = fit_medians(binary_raw, "Binary block")
    if binary_columns:
        observed_binary = binary_raw[np.isfinite(binary_raw)]
        invalid_binary = int((~np.isin(observed_binary, [0.0, 1.0])).sum())
        if invalid_binary:
            raise PreprocessingAbort(
                f"Binary training block has {invalid_binary} values outside 0/1."
            )

    categorical_encoder: OneHotEncoder | None = None
    if categorical_columns:
        categorical_encoder = make_one_hot_encoder()
        categorical_encoder.fit(categorical_array(train, categorical_columns))

    multi_vocabularies: dict[str, list[str]] = {}
    for column in multi_columns:
        token_lists = token_lists_for_series(train[column], b5)
        multi_vocabularies[column] = sorted(
            {token for tokens in token_lists for token in tokens}
        )

    state: dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "numeric_columns": numeric_columns,
        "binary_columns": binary_columns,
        "categorical_columns": categorical_columns,
        "multi_columns": multi_columns,
        "presence_columns": presence_columns,
        "numeric_medians": numeric_medians,
        "binary_medians": binary_medians,
        "numeric_scaler": numeric_scaler,
        "categorical_encoder": categorical_encoder,
        "multi_vocabularies": multi_vocabularies,
        "keep_encoded_mask": None,
        "feature_names_before_variance": [],
        "feature_names": [],
    }

    matrix_before, train_audit = build_encoded_parts(train, state, b5, fit=True)
    names_before = expanded_feature_names_before_variance(state)
    if matrix_before.shape[1] != len(names_before):
        raise PreprocessingAbort(
            "Encoded width/name mismatch before variance filtering: "
            f"{matrix_before.shape[1]} vs {len(names_before)}"
        )
    if matrix_before.shape[1] == 0:
        raise PreprocessingAbort("Preprocessor produced zero encoded columns.")

    maximum = np.asarray(matrix_before.max(axis=0).toarray()).ravel()
    minimum = np.asarray(matrix_before.min(axis=0).toarray()).ravel()
    keep = np.isfinite(maximum) & np.isfinite(minimum) & (maximum != minimum)
    if not keep.any():
        raise PreprocessingAbort("All encoded columns are constant in training.")

    state["keep_encoded_mask"] = keep
    state["feature_names_before_variance"] = names_before
    state["feature_names"] = [
        name for name, retained in zip(names_before, keep, strict=True) if retained
    ]
    matrix = matrix_before[:, keep].tocsr().astype(np.float32)
    if not np.isfinite(matrix.data).all():
        raise PreprocessingAbort("Training X contains NaN or infinity.")
    return state, matrix, train_audit


def transform_with_state(
    frame: pd.DataFrame,
    state: Mapping[str, Any],
    b5: Any,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    matrix_before, audit = build_encoded_parts(frame, state, b5, fit=False)
    keep = np.asarray(state["keep_encoded_mask"], dtype=bool)
    if matrix_before.shape[1] != len(keep):
        raise PreprocessingAbort(
            "Validation encoded width differs from training width: "
            f"{matrix_before.shape[1]} vs {len(keep)}"
        )
    matrix = matrix_before[:, keep].tocsr().astype(np.float32)
    if not np.isfinite(matrix.data).all():
        raise PreprocessingAbort("Validation X contains NaN or infinity.")
    return matrix, audit


def sparse_density(matrix: sparse.csr_matrix) -> float:
    denominator = matrix.shape[0] * matrix.shape[1]
    return float(matrix.nnz / denominator) if denominator else 0.0


def aggregate_missingness(
    scenario: str,
    split_label: str,
    frame: pd.DataFrame,
    feature_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_type_map = feature_manifest.set_index("model_feature_name")[
        "feature_type"
    ].to_dict()
    for feature_type in ["numeric", "binary", "categorical", "multi_response"]:
        columns = [
            column for column in frame.columns if feature_type_map.get(column) == feature_type
        ]
        if not columns:
            continue
        missing = frame[columns].isna()
        rows.append(
            {
                "scenario": scenario,
                "split": split_label,
                "feature_type": feature_type,
                "original_feature_count": len(columns),
                "row_count": len(frame),
                "missing_cells": int(missing.to_numpy().sum()),
                "total_cells": int(missing.size),
                "missing_cell_pct": (
                    100.0 * float(missing.to_numpy().mean()) if missing.size else 0.0
                ),
                "features_with_any_missing": int(missing.any(axis=0).sum()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_unknowns(
    scenario: str,
    audit: Mapping[str, Any],
) -> pd.DataFrame:
    categorical_counts = audit.get("categorical_unknown_counts", {})
    multi_occurrences = audit.get("multi_unknown_token_occurrences", {})
    multi_rows = audit.get("multi_unknown_rows", {})
    return pd.DataFrame(
        [
            {
                "scenario": scenario,
                "unknown_kind": "categorical_value",
                "features_with_unknown": int(
                    sum(int(value) > 0 for value in categorical_counts.values())
                ),
                "unknown_occurrences": int(sum(map(int, categorical_counts.values()))),
                "rows_with_unknown": np.nan,
            },
            {
                "scenario": scenario,
                "unknown_kind": "multi_response_token",
                "features_with_unknown": int(
                    sum(int(value) > 0 for value in multi_occurrences.values())
                ),
                "unknown_occurrences": int(sum(map(int, multi_occurrences.values()))),
                "rows_with_unknown": int(sum(map(int, multi_rows.values()))),
            },
        ]
    )


def encoded_manifest(state: Mapping[str, Any]) -> pd.DataFrame:
    before_names = list(state["feature_names_before_variance"])
    keep = np.asarray(state["keep_encoded_mask"], dtype=bool)
    records: list[dict[str, Any]] = []
    output_position = 0
    for before_position, (name, retained) in enumerate(
        zip(before_names, keep, strict=True),
        start=1,
    ):
        if retained:
            output_position += 1
        records.append(
            {
                "encoded_position_before_variance_1_based": before_position,
                "encoded_feature_name": name,
                "retained_after_train_variance_filter": bool(retained),
                "encoded_position_final_1_based": output_position if retained else pd.NA,
            }
        )
    return pd.DataFrame(records)


def compare_matched_longitudinal_raw(
    matched_train: pd.DataFrame,
    matched_validation: pd.DataFrame,
    longitudinal_train: pd.DataFrame,
    longitudinal_validation: pd.DataFrame,
    checks: Checks,
) -> None:
    for split_label, matched, longitudinal in [
        (TRAIN, matched_train, longitudinal_train),
        (VALIDATION, matched_validation, longitudinal_validation),
    ]:
        comparable: list[tuple[str, str]] = []
        for matched_column in matched.columns:
            if matched_column.startswith("modality_present__"):
                source = matched_column.removeprefix("modality_present__")
                long_column = (
                    f"modality_present__current__{source}"
                    if source in VISIT_SOURCES
                    else f"modality_present__static__{source}"
                )
            elif matched_column.startswith(tuple(f"{s}__" for s in VISIT_SOURCES)):
                long_column = f"current__{matched_column}"
            elif matched_column == "PTDEMOG__AGE_AT_TARGET":
                long_column = "current__PTDEMOG__AGE_AT_TARGET"
            else:
                long_column = matched_column
            if long_column in longitudinal.columns:
                comparable.append((matched_column, long_column))

        differences = 0
        for matched_column, long_column in comparable:
            left = matched[matched_column].astype("string").fillna("<NA>")
            right = longitudinal[long_column].astype("string").fillna("<NA>")
            differences += int((~left.eq(right)).sum())
        checks.add(
            f"matched_longitudinal_current_static_raw_equal__{split_label}",
            differences == 0,
            0,
            differences,
            f"compared_feature_pairs={len(comparable)}",
        )


def preprocessing_policy_payload() -> dict[str, Any]:
    return {
        "policy_version": PREPROCESSING_VERSION,
        "semantic_transforms": "frozen BENCHMARK04B/A06 transform registry",
        "support": {
"threshold_participants": SUPPORT_THRESHOLD,
            "application": "per scenario and temporal block on train",
            "unit": "distinct train participants with original feature observed",
            "performed_before_encoding": True,
        },
        "numeric": {
            "imputation": "train median",
            "scaling": "StandardScaler fit on train after imputation",
            "missing_indicator": "one indicator for every retained numeric feature; train-zero constants removed later",
        },
        "binary": {
            "allowed_values": [0, 1],
            "imputation": "train median",
            "scaling": "none",
            "missing_indicator": "one indicator for every retained binary feature; train-zero constants removed later",
        },
        "categorical": {
            "missing": "explicit __MISSING__ category",
            "encoding": "OneHotEncoder fit on train",
            "validation_unknown": "ignored and counted",
        },
        "multi_response": {
            "encoding": "multi-hot vocabulary fit on train",
            "missing": "explicit __MISSING__ column",
            "validation_unknown": "ignored after audit; __UNKNOWN__ train-constant column removed by zero-variance filter",
        },
        "temporal_source_exclusions": {
            "scope": "longitudinal previous visit-level source blocks",
            "rule": (
                "BENCHMARK05B flags collected on or after current target MRI "
                "are set missing"
            ),
            "application_order": (
                "before modality presence, semantic transform, imputation and encoding"
            ),
            "cross_stage_hash_and_flag_validation": True,
        },
        "modality": {
            "indicator": "one modality_present indicator per source-backed block; no presence indicator for derived covariates",
            "computed": "after temporal source exclusions and before support imputation",
        },
        "variance_filter": {
            "rule": "remove non-finite or zero-range encoded columns using train only",
            "validation_refit": False,
        },
        "test": {
            "materialized": False,
            "used": False,
            "policy": "same frozen state will be applied only after final model lock",
        },
    }


def build_story(
    dimension_summary: pd.DataFrame,
    unknown_summary: pd.DataFrame,
    zero_variance_summary: pd.DataFrame,
) -> str:
    lines = [
        "# BENCHMARK06B — Pré-processamento das entradas",
        "",
        f"- Versão do script: `{SCRIPT_VERSION}`",
        f"- Política: `{PREPROCESSING_VERSION}`",
        "- Fit: treino somente",
        "- Validação reaprendida: **não**",
        "- Teste usado/materializado: **não**",
        "- Targets MRI carregados: **não**",
        "",
        "## Política",
        "",
        "- transformações semânticas e suporte são consumidos dos locks anteriores;",
        "- numéricos usam mediana, standardization e indicador de ausência;",
        "- binários usam mediana, sem scaling, e indicador de ausência;",
        "- categóricos usam `__MISSING__` e one-hot com unknown ignorado;",
        "- multi-response usa vocabulário do treino e multi-hot;",
        "- blocos históricos temporalmente inseguros são anulados antes de qualquer transformação;",
        "- cada fonte/bloco recebe `modality_present` após essa anulação e antes da imputação;",
        "- variância zero é removida somente pelo treino.",
        "",
        "## Dimensões",
        "",
        "| Cenário | Treino | Validação | Originais | Antes var. | Finais | Densidade treino |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dimension_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scenario}` | {row.train_rows} | {row.validation_rows} | "
            f"{row.original_model_features} | {row.encoded_columns_before_variance} | "
            f"{row.encoded_columns_final} | {row.train_density:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Unknowns na validação",
            "",
            "Unknowns são contabilizados, mas valores/tokens não são escritos nos relatórios safe.",
            "",
            "| Cenário | Tipo | Features afetadas | Ocorrências |",
            "|---|---|---:|---:|",
        ]
    )
    for row in unknown_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scenario}` | `{row.unknown_kind}` | "
            f"{row.features_with_unknown} | {row.unknown_occurrences} |"
        )
    lines.extend(
        [
            "",
            "## Variância zero",
            "",
            "| Cenário | Antes | Removidas | Finais |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in zero_variance_summary.itertuples(index=False):
        lines.append(
            f"| `{row.scenario}` | {row.encoded_columns_before_variance} | "
            f"{row.zero_variance_columns_removed} | {row.encoded_columns_final} |"
        )
    lines.extend(
        [
            "",
            "## Artefatos",
            "",
            "Matrizes, preprocessadores, row manifests, medianas, categorias e tokens foram escritos somente em `data/processed/local_only/benchmark06b_preprocessed_inputs/`.",
            "Essa pasta é individualizada/local-only e não pode ser compartilhada.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_upstream(
    paths: Paths,
    b5: Any,
    checks: Checks,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    artifacts = {
        "package_root": paths.package,
        "benchmark05b_script": paths.benchmark05b_script,
        "benchmark05b_summary": paths.benchmark05b_summary,
        "benchmark05b_checks": paths.benchmark05b_checks,
        "official_schema": paths.official_schema,
        "official_transforms": paths.official_transforms,
        "support_policy": paths.support_policy,
        "schema_lock": paths.schema_lock,
        "scenario_definitions": paths.scenario_definitions,
        "scenario_support_manifest": paths.scenario_support_manifest,
        "scenario_lock": paths.scenario_lock,
        "benchmark05b_local": paths.benchmark05b_local,
        "temporal_exclusion_summary": paths.temporal_exclusion_summary,
        "temporal_exclusion_manifest": paths.temporal_exclusion_manifest,
    }
    for label, path in artifacts.items():
        exists = path.is_dir() if label in {"package_root", "benchmark05b_local"} else path.is_file()
        checks.add(f"artifact__{label}", exists, True, exists, str(path))
    for scenario, filename in SCENARIO_ARTIFACTS.items():
        path = paths.benchmark05b_local / filename
        checks.add(f"artifact__scenario__{scenario}", path.is_file(), True, path.is_file(), str(path))
    if not checks.passed:
        raise PreprocessingAbort("Required BENCHMARK06B upstream artifacts are missing.")

    benchmark05b_summary = read_json(paths.benchmark05b_summary)
    benchmark05b_checks = pd.read_csv(paths.benchmark05b_checks, low_memory=False)
    scenario_lock = read_json(paths.scenario_lock)
    scenario_definitions = read_json(paths.scenario_definitions)
    support_policy = read_json(paths.support_policy)

    checks.add(
        "benchmark05b_passed",
        bool(benchmark05b_summary.get("scenario_build_passed")),
        True,
        benchmark05b_summary.get("scenario_build_passed"),
    )
    checks.add(
        "benchmark05b_version",
        str(benchmark05b_summary.get("benchmark05b_script_version"))
        == EXPECTED_BENCHMARK05B_VERSION,
        EXPECTED_BENCHMARK05B_VERSION,
        benchmark05b_summary.get("benchmark05b_script_version"),
    )
    scenario_decision_id = str(
        benchmark05b_summary.get("scenario_decision_id", "")
    )
    scenario_ids_consistent = bool(
        scenario_decision_id
        and str(scenario_lock.get("scenario_decision_id", ""))
        == scenario_decision_id
        and str(scenario_definitions.get("scenario_decision_id", ""))
        == scenario_decision_id
    )
    checks.add(
        "scenario_decision_id_chain_consistent",
        scenario_ids_consistent,
        True,
        scenario_ids_consistent,
    )
    temporal_correction_declared = bool(
        benchmark05b_summary.get("scope", {}).get(
            "temporal_source_exclusions_applied"
        )
        and scenario_lock.get("temporal_source_exclusions", {}).get(
            "applied_before_support_and_transform_audit"
        )
    )
    checks.add(
        "benchmark05b_temporal_source_correction_declared",
        temporal_correction_declared,
        True,
        temporal_correction_declared,
    )
    checks.add(
        "temporal_policy_decision_id",
        str(benchmark05b_summary.get("scope", {}).get("temporal_policy_decision_id"))
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        benchmark05b_summary.get("scope", {}).get("temporal_policy_decision_id"),
    )
    checks.add(
        "benchmark05b_no_failed_checks",
        int(benchmark05b_summary.get("failed_check_count", -1)) == 0,
        0,
        benchmark05b_summary.get("failed_check_count"),
    )
    if "passed" in benchmark05b_checks.columns:
        failed_upstream = int((~benchmark05b_checks["passed"].map(parse_bool)).sum())
    else:
        failed_upstream = -1
    checks.add("benchmark05b_check_table_passed", failed_upstream == 0, 0, failed_upstream)
    checks.add(
        "support_threshold_50",
        int(support_policy.get("selected_support_threshold_participants", -1))
        == SUPPORT_THRESHOLD,
        SUPPORT_THRESHOLD,
        support_policy.get("selected_support_threshold_participants"),
    )
    checks.add(
        "test_not_materialized_upstream",
        not bool(benchmark05b_summary.get("scope", {}).get("test_scenarios_materialized")),
        False,
        benchmark05b_summary.get("scope", {}).get("test_scenarios_materialized"),
    )

    summary_hashes = benchmark05b_summary.get("artifacts", {}).get(
        "local_artifact_hashes", {}
    )
    for scenario, filename in SCENARIO_ARTIFACTS.items():
        actual = sha256_file(paths.benchmark05b_local / filename)
        expected = summary_hashes.get(filename)
        checks.add(
            f"scenario_artifact_hash__{scenario}",
            actual == expected,
            expected,
            actual,
        )
    checks.add(
        "support_manifest_hash",
        sha256_file(paths.scenario_support_manifest)
        == benchmark05b_summary.get("artifacts", {}).get(
            "scenario_support_manifest_sha256"
        ),
        benchmark05b_summary.get("artifacts", {}).get(
            "scenario_support_manifest_sha256"
        ),
        sha256_file(paths.scenario_support_manifest),
    )
    for label, path, summary_key in [
        ("scenario_definitions", paths.scenario_definitions, "scenario_definitions_sha256"),
        ("scenario_lock", paths.scenario_lock, "scenario_lock_sha256"),
    ]:
        expected_hash = benchmark05b_summary.get("artifacts", {}).get(summary_key)
        actual_hash = sha256_file(path)
        checks.add(
            f"{label}_hash",
            bool(expected_hash) and actual_hash == expected_hash,
            expected_hash,
            actual_hash,
        )
    if not checks.passed:
        raise PreprocessingAbort("BENCHMARK05B lock validation failed.")

    # Use BENCHMARK05B's already audited schema/transform/source loaders.
    b5_args = argparse.Namespace(
        package_root=paths.package,
        reports_root=paths.safe.parents[1],
        processed_root=paths.local.parent,
        supersede_existing=False,
    )
    b5_paths = b5.resolve_paths(b5_args)
    b5_checks = b5.Checks()
    schema = b5.load_official_schema(b5_paths, b5_checks)
    registry = b5.load_transform_registry(b5_paths, schema, b5_checks)
    source_frames = b5.load_source_frames(b5_paths, schema, b5_checks)
    checks.add(
        "benchmark05b_contract_loaders_passed",
        b5_checks.passed,
        True,
        b5_checks.passed,
        f"failed={b5_checks.failed}",
    )

    support_manifest = pd.read_csv(paths.scenario_support_manifest, low_memory=False)
    require_columns(
        support_manifest,
        [
            "scenario",
            "block",
            "source_name",
            "column_name",
            "full_column_name",
            "official_feature_type",
            "train_participant_count",
            "train_observed_participants",
            "support_threshold_participants",
            "retained_at_support_threshold",
        ],
        "scenario support manifest",
    )
    support_manifest["retained_at_support_threshold"] = support_manifest[
        "retained_at_support_threshold"
    ].map(parse_bool)
    for column in [
        "train_participant_count",
        "train_observed_participants",
        "support_threshold_participants",
    ]:
        support_manifest[column] = pd.to_numeric(
            support_manifest[column], errors="raise"
        ).astype(int)
    checks.add(
        "support_manifest_scenarios",
        set(support_manifest["scenario"].astype(str).unique()) == set(SCENARIOS),
        list(SCENARIOS),
        sorted(support_manifest["scenario"].astype(str).unique().tolist()),
    )
    checks.add(
        "support_manifest_threshold",
        set(support_manifest["support_threshold_participants"].unique())
        == {SUPPORT_THRESHOLD},
        [SUPPORT_THRESHOLD],
        sorted(support_manifest["support_threshold_participants"].unique().tolist()),
    )
    age_support = support_manifest.loc[
        support_manifest["full_column_name"].astype(str).eq("PTDEMOG__AGE_AT_TARGET")
    ].copy()
    age_contract_ok = bool(
        len(age_support) == len(SCENARIOS)
        and set(age_support["scenario"].astype(str)) == set(SCENARIOS)
        and age_support["block"].astype(str).eq("current_derived").all()
        and age_support["official_feature_type"].astype(str).eq("numeric").all()
        and age_support["retained_at_support_threshold"].map(parse_bool).all()
    )
    checks.add(
        "age_at_target_support_contract",
        age_contract_ok,
        True,
        age_contract_ok,
        f"rows={len(age_support)}",
    )
    previous_age_rows = int(
        support_manifest["full_column_name"].astype(str).str.contains(
            "AGE_AT_TARGET", regex=False
        )
        .where(support_manifest["block"].astype(str).eq("previous_visit"), False)
        .sum()
    )
    checks.add(
        "no_previous_age_feature",
        previous_age_rows == 0,
        0,
        previous_age_rows,
    )
    if not checks.passed:
        raise PreprocessingAbort("Official schema/source/support loading failed.")

    return (
        benchmark05b_summary,
        scenario_lock,
        scenario_definitions,
        support_manifest,
        schema,
        registry,
        source_frames,
    )


def run_stage(paths: Paths, args: argparse.Namespace, checks: Checks) -> dict[str, Any]:
    b5 = load_benchmark05b_module(paths.benchmark05b_script)
    checks.add(
        "benchmark05b_module_version",
        str(getattr(b5, "SCRIPT_VERSION", "")) == EXPECTED_BENCHMARK05B_VERSION,
        EXPECTED_BENCHMARK05B_VERSION,
        getattr(b5, "SCRIPT_VERSION", None),
    )
    if not checks.passed:
        raise PreprocessingAbort("Unexpected BENCHMARK05B module version.")

    (
        benchmark05b_summary,
        scenario_lock,
        scenario_definitions,
        support_manifest,
        schema,
        registry,
        source_frames,
    ) = validate_upstream(paths, b5, checks)
    scenario_decision_id = str(benchmark05b_summary["scenario_decision_id"])
    expected_rows = expected_rows_from_benchmark05(benchmark05b_summary)

    if paths.local.exists() and not args.overwrite:
        raise PreprocessingAbort(
            f"Local output already exists: {paths.local}. Use --overwrite only "
            "after confirming that replacing it is intended."
        )

    staging_parent = paths.local.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="benchmark06b_staging_", dir=str(staging_parent))
    )

    dimension_rows: list[dict[str, Any]] = []
    zero_variance_rows: list[dict[str, Any]] = []
    unknown_frames: list[pd.DataFrame] = []
    missing_frames: list[pd.DataFrame] = []
    modality_frames: list[pd.DataFrame] = []
    original_type_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    raw_by_scenario: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    try:
        for scenario in SCENARIOS:
            logging.info("Building and fitting preprocessing: %s", scenario)
            manifest_path = paths.benchmark05b_local / SCENARIO_ARTIFACTS[scenario]
            manifest = pd.read_csv(manifest_path, low_memory=False)
            require_columns(manifest, [RID, SPLIT, "current_VISCODE2"], scenario)
            manifest[SPLIT] = manifest[SPLIT].astype("string").str.strip()
            if scenario == "longitudinal_previous_last":
                validate_temporal_exclusion_flags(
                    manifest,
                    paths,
                    benchmark05b_summary,
                    checks,
                )

            train_raw, validation_raw, feature_manifest, modality_summary = (
                build_raw_scenario(
                    scenario,
                    manifest,
                    source_frames,
                    schema,
                    registry,
                    support_manifest,
                    expected_rows[scenario],
                    b5,
                    checks,
                )
            )
            raw_by_scenario[scenario] = (train_raw, validation_raw)

            state, x_train, _ = fit_preprocessor_state(
                train_raw,
                feature_manifest,
                b5,
            )
            x_validation, validation_audit = transform_with_state(
                validation_raw,
                state,
                b5,
            )

            checks.add(
                f"matrix_width_alignment__{scenario}",
                x_train.shape[1] == x_validation.shape[1],
                x_train.shape[1],
                x_validation.shape[1],
            )
            checks.add(
                f"matrix_finite_train__{scenario}",
                bool(np.isfinite(x_train.data).all()),
                True,
                bool(np.isfinite(x_train.data).all()),
            )
            checks.add(
                f"matrix_finite_validation__{scenario}",
                bool(np.isfinite(x_validation.data).all()),
                True,
                bool(np.isfinite(x_validation.data).all()),
            )
            if x_train.shape[1]:
                train_max = np.asarray(x_train.max(axis=0).toarray()).ravel()
                train_min = np.asarray(x_train.min(axis=0).toarray()).ravel()
                zero_final = int((train_max == train_min).sum())
            else:
                zero_final = x_train.shape[1]
            checks.add(
                f"zero_variance_removed__{scenario}",
                zero_final == 0,
                0,
                zero_final,
            )

            scenario_dir = staging / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            sparse.save_npz(scenario_dir / "X_train.npz", x_train, compressed=True)
            sparse.save_npz(
                scenario_dir / "X_validation.npz",
                x_validation,
                compressed=True,
            )

            train_rows = manifest.loc[manifest[SPLIT].eq(TRAIN)].reset_index(drop=True)
            validation_rows = manifest.loc[
                manifest[SPLIT].eq(VALIDATION)
            ].reset_index(drop=True)
            train_rows.to_csv(
                scenario_dir / "row_manifest_train.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
                lineterminator="\n",
            )
            validation_rows.to_csv(
                scenario_dir / "row_manifest_validation.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
                lineterminator="\n",
            )
            feature_manifest.to_csv(
                scenario_dir / "raw_feature_manifest.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
                lineterminator="\n",
            )
            expanded = encoded_manifest(state)
            expanded.to_csv(
                scenario_dir / "encoded_feature_manifest.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
                lineterminator="\n",
            )
            joblib.dump(state, scenario_dir / "preprocessor.joblib", compress=3)

            state_summary = {
                "scenario": scenario,
                "preprocessing_version": PREPROCESSING_VERSION,
                "numeric_features": len(state["numeric_columns"]),
                "binary_features": len(state["binary_columns"]),
                "categorical_features": len(state["categorical_columns"]),
                "multi_response_features": len(state["multi_columns"]),
                "modality_presence_features": len(state["presence_columns"]),
                "encoded_columns_before_variance": len(
                    state["feature_names_before_variance"]
                ),
                "encoded_columns_final": len(state["feature_names"]),
                "test_materialized": False,
            }
            atomic_write_json(scenario_dir / "preprocessor_summary.json", state_summary)

            files = [
                "X_train.npz",
                "X_validation.npz",
                "row_manifest_train.csv.gz",
                "row_manifest_validation.csv.gz",
                "raw_feature_manifest.csv.gz",
                "encoded_feature_manifest.csv.gz",
                "preprocessor.joblib",
                "preprocessor_summary.json",
            ]
            for filename in files:
                path = scenario_dir / filename
                artifact_rows.append(
                    {
                        "scenario": scenario,
                        "relative_path": f"{scenario}/{filename}",
                        "sha256": sha256_file(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )

            before_width = len(state["feature_names_before_variance"])
            final_width = len(state["feature_names"])
            dimension_rows.append(
                {
                    "scenario": scenario,
                    "train_rows": x_train.shape[0],
                    "validation_rows": x_validation.shape[0],
                    "train_participants": int(
                        train_rows[RID].astype(str).nunique()
                    ),
                    "validation_participants": int(
                        validation_rows[RID].astype(str).nunique()
                    ),
                    "original_model_features": int(len(feature_manifest)),
                    "encoded_columns_before_variance": before_width,
                    "encoded_columns_final": final_width,
                    "train_nnz": int(x_train.nnz),
                    "validation_nnz": int(x_validation.nnz),
                    "train_density": sparse_density(x_train),
                    "validation_density": sparse_density(x_validation),
                }
            )
            zero_variance_rows.append(
                {
                    "scenario": scenario,
                    "encoded_columns_before_variance": before_width,
                    "zero_variance_columns_removed": before_width - final_width,
                    "encoded_columns_final": final_width,
                }
            )
            unknown_frames.append(aggregate_unknowns(scenario, validation_audit))
            missing_frames.append(
                aggregate_missingness(scenario, TRAIN, train_raw, feature_manifest)
            )
            missing_frames.append(
                aggregate_missingness(
                    scenario,
                    VALIDATION,
                    validation_raw,
                    feature_manifest,
                )
            )
            modality_frames.append(modality_summary)
            for feature_type, count in (
                feature_manifest["feature_type"].value_counts().sort_index().items()
            ):
                original_type_rows.append(
                    {
                        "scenario": scenario,
                        "feature_type": str(feature_type),
                        "original_model_feature_count": int(count),
                    }
                )

        compare_matched_longitudinal_raw(
            *raw_by_scenario["snapshot_matched_last"],
            *raw_by_scenario["longitudinal_previous_last"],
            checks,
        )

        if not checks.passed:
            raise PreprocessingAbort(
                f"Preprocessing validation failed with {checks.failed} failed checks."
            )

        readme = """BENCHMARK06B PREPROCESSED INPUTS — LOCAL ONLY

This directory contains ADNI-derived matrices, RID/VISCODE2 row manifests,
train-fitted medians/scalers/categories/token vocabularies and expanded feature
names. Do not commit, upload, publish or share any file from this directory.

Only train and validation are materialized. Test remains closed.
"""
        atomic_write_text(staging / "README_LOCAL_ONLY.txt", readme)
        artifact_rows.append(
            {
                "scenario": "<ROOT>",
                "relative_path": "README_LOCAL_ONLY.txt",
                "sha256": sha256_file(staging / "README_LOCAL_ONLY.txt"),
                "size_bytes": int((staging / "README_LOCAL_ONLY.txt").stat().st_size),
            }
        )

        artifact_manifest = pd.DataFrame(artifact_rows).sort_values(
            ["scenario", "relative_path"], kind="stable"
        ).reset_index(drop=True)
        artifact_manifest.to_csv(
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
    zero_variance_summary = pd.DataFrame(zero_variance_rows)
    unknown_summary = pd.concat(unknown_frames, ignore_index=True)
    missingness_summary = pd.concat(missing_frames, ignore_index=True)
    modality_summary = pd.concat(modality_frames, ignore_index=True)
    original_type_summary = pd.DataFrame(original_type_rows)

    policy = preprocessing_policy_payload()
    upstream_hashes = {
        "benchmark05b_script_sha256": sha256_file(paths.benchmark05b_script),
        "benchmark05b_summary_sha256": sha256_file(paths.benchmark05b_summary),
        "official_schema_sha256": sha256_file(paths.official_schema),
        "official_transforms_sha256": sha256_file(paths.official_transforms),
        "support_policy_sha256": sha256_file(paths.support_policy),
        "schema_lock_sha256": sha256_file(paths.schema_lock),
        "scenario_definitions_sha256": sha256_file(paths.scenario_definitions),
        "scenario_support_manifest_sha256": sha256_file(
            paths.scenario_support_manifest
        ),
        "scenario_lock_sha256": sha256_file(paths.scenario_lock),
        "auditA05_summary_sha256": sha256_file(
            paths.temporal_exclusion_summary
        ),
        "auditA05_local_manifest_sha256": sha256_file(
            paths.temporal_exclusion_manifest
        ),
    }
    decision_material = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "scenario_decision_id": scenario_decision_id,
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "policy": policy,
        "upstream_hashes": upstream_hashes,
    }
    preprocessing_decision_id = stable_json_hash(decision_material)[:20]

    final_inventory_path = paths.local / "LOCAL_ONLY_artifact_inventory.csv"
    inventory = pd.read_csv(final_inventory_path, low_memory=False)
    local_hashes = {
        str(row["relative_path"]): str(row["sha256"])
        for row in inventory.to_dict(orient="records")
    }

    policy_payload = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "preprocessing_decision_id": preprocessing_decision_id,
        "scenario_decision_id": scenario_decision_id,
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "policy": policy,
    }
    manifest_payload = dimension_summary.copy()
    manifest_payload["preprocessing_decision_id"] = preprocessing_decision_id
    manifest_payload["scenario_decision_id"] = scenario_decision_id
    manifest_payload["test_materialized"] = False

    lock_payload = {
        "lock_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_decision_id": preprocessing_decision_id,
        "upstream_scenario_decision_id": scenario_decision_id,
        "upstream_temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "support_threshold_participants": SUPPORT_THRESHOLD,
        "fit_split": TRAIN,
        "validation_refit": False,
        "test_materialized": False,
        "test_used": False,
        "policy_sha256": stable_json_hash(policy),
        "upstream_hashes": upstream_hashes,
        "local_artifact_hashes": local_hashes,
        "methodological_commitment": (
            "Do not change semantic transforms, support, imputation, encoding, "
            "missingness representation, temporal source exclusions, modality "
            "indicators or train-only fit after test inspection."
        ),
    }

    for config_path, content, is_frame in [
        (paths.preprocessing_policy, policy_payload, False),
        (paths.preprocessing_manifest, manifest_payload, True),
        (paths.preprocessing_lock, lock_payload, False),
    ]:
        if config_path.exists() and not args.overwrite:
            raise PreprocessingAbort(
                f"Official preprocessing artifact already exists: {config_path}. "
                "Use --overwrite only for an intentional same-stage replacement."
            )
        if is_frame:
            atomic_write_csv(content, config_path)  # type: ignore[arg-type]
        else:
            atomic_write_json(config_path, content)  # type: ignore[arg-type]

    atomic_write_csv(
        dimension_summary,
        paths.safe / "benchmark06b_scenario_dimension_summary.csv",
    )
    atomic_write_csv(
        original_type_summary,
        paths.safe / "benchmark06b_original_feature_type_summary.csv",
    )
    atomic_write_csv(
        zero_variance_summary,
        paths.safe / "benchmark06b_zero_variance_summary.csv",
    )
    atomic_write_csv(
        unknown_summary,
        paths.safe / "benchmark06b_unknown_validation_summary.csv",
)
    atomic_write_csv(
        missingness_summary,
        paths.safe / "benchmark06b_missingness_summary.csv",
    )
    atomic_write_csv(
        modality_summary.sort_values(
            ["scenario", "block", "source_name", "split"], kind="stable"
        ).reset_index(drop=True),
        paths.safe / "benchmark06b_modality_presence_summary.csv",
    )

    story = build_story(dimension_summary, unknown_summary, zero_variance_summary)
    atomic_write_text(
        paths.safe / "BENCHMARK06B_PREPROCESSING_STORY.md",
        story,
    )

    summary_payload = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_passed": True,
        "preprocessing_decision_id": preprocessing_decision_id,
        "scenario_decision_id": scenario_decision_id,
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": None,
        "scope": {
            "fit_split": TRAIN,
            "validation_transformed_without_refit": True,
            "test_values_used": False,
            "test_materialized": False,
            "targets_loaded": False,
            "models_trained": False,
            "support_threshold_participants": SUPPORT_THRESHOLD,
            "temporal_source_exclusions_applied_before_preprocessing": True,
            "temporal_exclusion_flags_cross_validated": True,
            "participant_split_changed": False,
            "age_at_target_materialized_upstream": True,
            "age_at_target_preprocessed_as_current_numeric_covariate": True,
            "previous_age_feature_created": False,
            "raw_birth_fields_exposed": False,
        },
        "policy": policy,
        "scenario_dimensions": dimension_summary.to_dict(orient="records"),
        "zero_variance": zero_variance_summary.to_dict(orient="records"),
        "unknown_validation": unknown_summary.to_dict(orient="records"),
        "artifacts": {
            "local_output_directory": str(paths.local.relative_to(paths.root)),
            "local_artifact_inventory_sha256": sha256_file(final_inventory_path),
            "preprocessing_policy": str(
                paths.preprocessing_policy.relative_to(paths.root)
            ),
            "preprocessing_policy_sha256": sha256_file(paths.preprocessing_policy),
            "preprocessing_manifest": str(
                paths.preprocessing_manifest.relative_to(paths.root)
            ),
            "preprocessing_manifest_sha256": sha256_file(
                paths.preprocessing_manifest
            ),
            "preprocessing_lock": str(
                paths.preprocessing_lock.relative_to(paths.root)
            ),
            "preprocessing_lock_sha256": sha256_file(paths.preprocessing_lock),
        },
        "upstream_hashes": upstream_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "privacy": {
            "safe_outputs_contain_participant_identifiers": False,
            "safe_outputs_contain_visit_identifiers": False,
            "safe_outputs_contain_category_or_token_values": False,
            "safe_outputs_contain_medical_values": False,
            "safe_outputs_contain_predictions": False,
            "individualized_artifacts_are_local_only": True,
            "local_artifacts_must_not_be_shared": True,
        },
    }
    atomic_write_json(paths.safe / "benchmark06b_summary.json", summary_payload)
    return summary_payload


def write_checks(paths: Paths, checks: Checks) -> None:
    frame = pd.DataFrame([asdict(item) for item in checks.items])
    atomic_write_csv(frame, paths.safe / "benchmark06b_validation_checks.csv")


def write_failure(paths: Paths, checks: Checks, error: str) -> None:
    write_checks(paths, checks)
    atomic_write_json(
        paths.safe / "benchmark06b_summary.json",
        {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing_passed": False,
            "check_count": len(checks.items),
            "failed_check_count": checks.failed,
            "fatal_error": error,
            "scope": {
                "test_values_used": False,
                "test_materialized": False,
                "targets_loaded": False,
                "models_trained": False,
            },
        },
    )


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    configure_logging(paths)
    checks = Checks()

    logging.info("%s v%s", STAGE_NAME, SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Frozen package: %s", paths.package)
    logging.info("Fit split: train; validation transform only; test closed.")
    logging.info("Support threshold frozen at %d participants.", SUPPORT_THRESHOLD)
    logging.info(
        "Temporal source exclusions are cross-validated and applied before preprocessing."
    )

    try:
        summary = run_stage(paths, args, checks)
        write_checks(paths, checks)
        dimensions = {
            row["scenario"]: row["encoded_columns_final"]
            for row in summary["scenario_dimensions"]
        }
        logging.info(
            "BENCHMARK06B PASSED | snapshot_all=%s | matched=%s | longitudinal=%s | test_materialized=false",
            dimensions.get("snapshot_all"),
            dimensions.get("snapshot_matched_last"),
            dimensions.get("longitudinal_previous_last"),
        )
        logging.info("Do not share %s.", paths.local)
        logging.info("Safe reports: %s", paths.safe)
        return 0
    except Exception as exc:
        logging.exception("BENCHMARK06B fatal error")
        error = f"{type(exc).__name__}: {exc}"
        write_failure(paths, checks, error)
        logging.error(
            "BENCHMARK06B FAILED | failed_checks=%d | fatal_error=%s",
            checks.failed,
            error,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
