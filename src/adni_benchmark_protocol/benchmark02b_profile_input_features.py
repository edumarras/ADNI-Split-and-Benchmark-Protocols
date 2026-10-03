#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK02B — perfil agregado das features cross-checked de entrada.
#
# Versão corrigida para o rerun pós-cross-check.
#
# O script:
# - exige BENCHMARK01 PASS sobre o pacote lógico frozen_experiment_v1_1;
# - exige AUDIT A06 v0.1.2 PASS e o ledger cross-check v1.0.0;
# - mantém anchor/split no pacote frozen v1.1;
# - perfila exclusivamente as oito tabelas de data/crosschecked_inputs_v1/;
# - usa o catálogo cross-checked e carrega as diretivas semânticas do A06;
# - usa somente participantes/visitas de TREINO para estatísticas de tipo/suporte;
# - não lê targets MRI, não imputa, não aplica suporte, não treina modelos.
#
# As diretivas A06 são registradas no perfil/schema draft como contrato congelado.
# As heurísticas deste stage continuam apenas diagnósticas e NÃO podem substituir
# uma decisão semântica congelada pelo cross-check.
#
# Todos os relatórios são SAFE e agregados por coluna/fonte. Nenhum RID,
# VISCODE2, valor médico/cognitivo, categoria observada ou valor genético
# individual é escrito.

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION: Final[str] = "0.2.1"
PACKAGE_DIR_NAME: Final[str] = "frozen_experiment_v1_1"
EXPECTED_LOGICAL_PACKAGE_NAME: Final[str] = "frozen_experiment_v1_1"
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"
CROSSCHECKED_INPUTS_NAME: Final[str] = "crosschecked_inputs_v1"
EXPECTED_A06_SCRIPT_VERSION: Final[str] = "0.1.2"
EXPECTED_LEDGER_VERSION: Final[str] = "1.0.0"
EXPECTED_LEDGER_SHA256: Final[str] = "0bfe87ca42e9d9c74c804878a7c09372bc4eb856cf2d60200c315dd301b04fa5"
EXPECTED_CROSSCHECK_FEATURE_COUNT: Final[int] = 218
EXPECTED_A06_DIRECTIVE_COUNT: Final[int] = 198
SPLITS: Final[tuple[str, ...]] = ("train", "validation", "test")
INPUT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS", "CDR", "FAQ", "MMSE", "MOCA", "NEUROBAT", "PTDEMOG", "APOERES"
)
SOURCE_GRAIN: Final[dict[str, str]] = {
    "ADAS": "visit",
    "CDR": "visit",
    "FAQ": "visit",
    "MMSE": "visit",
    "MOCA": "visit",
    "NEUROBAT": "visit",
    "PTDEMOG": "participant",
    "APOERES": "participant",
}
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

PACKAGE_ARTIFACTS: Final[dict[str, str]] = {
    "manifest": "04_official_split/official_frozen_manifest.json",
    "policy_lock": "04_official_split/strict_date_temporal_policy_lock.json",
    "anchor": "03_final_ready_tables/supervised_anchor_visits.csv.gz",
}

CROSSCHECK_ARTIFACTS: Final[dict[str, str]] = {
    "crosschecked_manifest": "crosschecked_input_manifest.json",
    "feature_catalog": "crosschecked_input_feature_catalog.csv",
    "directives": "crosscheck_feature_directives.csv",
}

BENCHMARK01_SUMMARY: Final[str] = (
    "reports/benchmark01b_validate_frozen_package/safe/"
    "benchmark01b_validation_summary.json"
)
A06_REPORT: Final[str] = (
    "reports/auditA06_freeze_crosschecked_inputs/safe/"
    "a06_verification_report.json"
)

MULTI_VALUE_DELIMITERS: Final[tuple[str, ...]] = ("|", ";")
NUMERIC_PARSE_THRESHOLD: Final[float] = 0.98
INTEGER_LIKE_THRESHOLD: Final[float] = 0.98
LOW_CARDINALITY_MAX: Final[int] = 20
EXTREME_MISSINGNESS_THRESHOLD: Final[float] = 0.99
HIGH_MISSINGNESS_THRESHOLD: Final[float] = 0.95

IDENTIFIER_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(^|_)(RID|ID|UID|IMAGEUID|PTID|SITEID|ROWID|RECORDID)($|_)", re.IGNORECASE
)
DATE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(DATE|TIME|TIMESTAMP|MONTH|YEAR|DAY)", re.IGNORECASE
)
ADMIN_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(PHASE|VISCODE|VERSION|STATUS|SOURCE|SITE|PROTOCOL|UPDATE|USER|COMMENT)",
    re.IGNORECASE,
)


class ProfileAbort(RuntimeError):
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
                _short_text(expected),
                _short_text(observed),
                f" | {details}" if details else "",
            )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)


def _short_text(value: Any, limit: int = 600) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile candidate non-MRI input features using aggregate-only outputs."
    )
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--crosschecked-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--benchmark01-summary", type=Path, default=None)
    parser.add_argument("--a06-report", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path(__file__).resolve().parents[2]
    package = (
        args.package_root.expanduser().resolve()
        if args.package_root
        else root / "data" / "processed" / PACKAGE_DIR_NAME
    )
    crosschecked = (
        args.crosschecked_root.expanduser().resolve()
        if args.crosschecked_root
        else root / "data" / "processed" / CROSSCHECKED_INPUTS_NAME
    )
    reports = (
        args.reports_root.expanduser().resolve()
        if args.reports_root
        else root / "reports"
    )
    stage = reports / "benchmark02b"
    safe = stage / "safe"
    logs = stage / "logs"
    safe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    benchmark01_summary = (
        args.benchmark01_summary.expanduser().resolve()
        if args.benchmark01_summary
        else root / BENCHMARK01_SUMMARY
    )
    a06_report = (
        args.a06_report.expanduser().resolve()
        if args.a06_report
        else root / A06_REPORT
    )
    return Paths(
        root=root,
        package=package,
        crosschecked=crosschecked,
        safe=safe,
        logs=logs,
        benchmark01_summary=benchmark01_summary,
        a06_report=a06_report,
    )


def configure_logging(paths: Paths) -> None:
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(paths.logs / "benchmark02b.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_text(path: Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    for column in frame.columns:
        series = frame[column].astype("string").str.strip()
        frame[column] = series.mask(series.eq(""), pd.NA)
    return frame


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ProfileAbort(f"{label} is missing required columns: {missing}")


def safe_div(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def normalize_observed(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.loc[text.notna() & text.ne("")]


def numeric_characteristics(values: pd.Series) -> dict[str, float | int | None]:
    observed = normalize_observed(values)
    n_observed = int(len(observed))
    if n_observed == 0:
        return {
            "numeric_parse_count": 0,
            "numeric_parse_fraction": None,
            "integer_like_count": 0,
            "integer_like_fraction_among_numeric": None,
            "finite_numeric_count": 0,
        }

    parsed = pd.to_numeric(observed, errors="coerce")
    numeric_mask = parsed.notna()
    numeric_count = int(numeric_mask.sum())
    finite_count = int(np.isfinite(parsed.loc[numeric_mask].to_numpy(dtype=float)).sum())

    integer_like_count = 0
    if numeric_count:
        numeric_values = parsed.loc[numeric_mask].to_numpy(dtype=float)
        integer_like_count = int(np.isclose(numeric_values, np.rint(numeric_values)).sum())

    return {
        "numeric_parse_count": numeric_count,
        "numeric_parse_fraction": finite_or_none(safe_div(numeric_count, n_observed)),
        "integer_like_count": integer_like_count,
        "integer_like_fraction_among_numeric": finite_or_none(
            safe_div(integer_like_count, numeric_count)
        ),
        "finite_numeric_count": finite_count,
    }


def multi_value_characteristics(values: pd.Series) -> dict[str, Any]:
    observed = normalize_observed(values)
    n_observed = int(len(observed))
    delimiter_counts: dict[str, int] = {}
    any_multi_mask = pd.Series(False, index=observed.index, dtype=bool)

    for delimiter in MULTI_VALUE_DELIMITERS:
        mask = observed.str.contains(re.escape(delimiter), regex=True, na=False)
        delimiter_counts[delimiter] = int(mask.sum())
        any_multi_mask = any_multi_mask | mask

    multi_count = int(any_multi_mask.sum())
    token_counts: list[int] = []
    distinct_tokens: set[str] = set()
    if multi_count:
        splitter = re.compile(r"[|;]")
        for value in observed.loc[any_multi_mask].tolist():
            tokens = [token.strip() for token in splitter.split(str(value)) if token.strip()]
            token_counts.append(len(tokens))
            distinct_tokens.update(tokens)

    return {
        "multi_value_count": multi_count,
        "multi_value_fraction": finite_or_none(safe_div(multi_count, n_observed)),
        "pipe_count": delimiter_counts.get("|", 0),
        "semicolon_count": delimiter_counts.get(";", 0),
        "max_tokens_in_observed_cell": max(token_counts, default=0),
        "distinct_token_count": len(distinct_tokens),
    }


def support_band(observed_participants: int) -> str:
    if observed_participants <= 0:
        return "000_all_missing"
    if observed_participants <= 5:
        return "001_1_to_5"
    if observed_participants <= 10:
        return "002_6_to_10"
    if observed_participants <= 20:
        return "003_11_to_20"
    if observed_participants <= 30:
        return "004_21_to_30"
    if observed_participants <= 50:
        return "005_31_to_50"
    if observed_participants <= 100:
        return "006_51_to_100"
    if observed_participants <= 250:
        return "007_101_to_250"
    if observed_participants <= 500:
        return "008_251_to_500"
    return "009_over_500"


def infer_type(
    native_train_values: pd.Series,
    snapshot_train_values: pd.Series,
    column_name: str,
) -> dict[str, Any]:
    native_train_observed = normalize_observed(native_train_values)
    snapshot_train_observed = normalize_observed(snapshot_train_values)
    basis = (
        native_train_observed
        if len(native_train_observed)
        else snapshot_train_observed
    )
    distinct_count = int(basis.nunique(dropna=True))

    numeric = numeric_characteristics(basis)
    multi = multi_value_characteristics(basis)
    numeric_fraction = numeric["numeric_parse_fraction"]
    integer_fraction = numeric["integer_like_fraction_among_numeric"]
    multi_fraction = multi["multi_value_fraction"]

    if len(basis) == 0:
        suggested_type = "undetermined_all_missing"
        suggested_encoding = "manual_review"
    elif multi_fraction is not None and multi_fraction > 0:
        suggested_type = "multi_response_candidate"
        suggested_encoding = "multi_hot"
    elif numeric_fraction is not None and numeric_fraction >= NUMERIC_PARSE_THRESHOLD:
        if distinct_count <= 2:
            suggested_type = "binary_numeric_candidate"
            suggested_encoding = "binary_or_one_hot_review"
        elif (
            distinct_count <= LOW_CARDINALITY_MAX
            and integer_fraction is not None
            and integer_fraction >= INTEGER_LIKE_THRESHOLD
        ):
            suggested_type = "categorical_numeric_code_candidate"
            suggested_encoding = "one_hot_or_numeric_review"
        else:
            suggested_type = "numeric_candidate"
            suggested_encoding = "numeric_median_imputation"
    else:
        if distinct_count <= 2:
            suggested_type = "binary_categorical_candidate"
            suggested_encoding = "one_hot"
        else:
            suggested_type = "categorical_candidate"
            suggested_encoding = "one_hot"

    flags: list[str] = []
    if IDENTIFIER_NAME_PATTERN.search(column_name):
        flags.append("identifier_name_pattern")
    if DATE_NAME_PATTERN.search(column_name):
        flags.append("date_or_time_name_pattern")
    if ADMIN_NAME_PATTERN.search(column_name):
        flags.append("administrative_name_pattern")
    if suggested_type in {
        "categorical_numeric_code_candidate",
        "binary_numeric_candidate",
    }:
        flags.append("numeric_looking_code_requires_semantic_review")
    if suggested_type == "multi_response_candidate":
        flags.append("multi_response_requires_token_semantics_review")
    if suggested_type == "categorical_candidate" and distinct_count > 100:
        flags.append("high_cardinality_categorical")

    return {
        "suggested_type": suggested_type,
        "suggested_encoding": suggested_encoding,
        "semantic_review_flags": "|".join(flags),
        **{f"type_basis_{key}": value for key, value in numeric.items()},
        **{f"type_basis_{key}": value for key, value in multi.items()},
        "type_basis_distinct_count": distinct_count,
        "type_basis_observed_count": int(len(basis)),
    }


def profile_scope(
    values: pd.Series,
    rids: pd.Series,
    prefix: str,
) -> dict[str, Any]:
    observed_mask = values.notna() & values.astype("string").str.strip().ne("")
    observed_values = values.loc[observed_mask].astype("string").str.strip()
    total_rows = int(len(values))
    observed_rows = int(observed_mask.sum())
    missing_rows = total_rows - observed_rows
    total_participants = int(rids.astype("string").nunique(dropna=True))
    observed_participants = int(
        rids.loc[observed_mask].astype("string").nunique(dropna=True)
    )
    distinct_observed = int(observed_values.nunique(dropna=True))
    effective = values.astype("string").fillna("__MISSING__").str.strip()
    effective = effective.mask(effective.eq(""), "__MISSING__")
    effective_distinct = int(effective.nunique(dropna=False))

    return {
        f"{prefix}_rows": total_rows,
        f"{prefix}_participants": total_participants,
        f"{prefix}_observed_rows": observed_rows,
        f"{prefix}_missing_rows": missing_rows,
        f"{prefix}_observed_fraction": finite_or_none(safe_div(observed_rows, total_rows)),
        f"{prefix}_missing_fraction": finite_or_none(safe_div(missing_rows, total_rows)),
        f"{prefix}_observed_participants": observed_participants,
        f"{prefix}_observed_participant_fraction": finite_or_none(
            safe_div(observed_participants, total_participants)
        ),
        f"{prefix}_distinct_observed_values": distinct_observed,
        f"{prefix}_distinct_including_missing": effective_distinct,
        f"{prefix}_all_missing": observed_rows == 0,
        f"{prefix}_observed_constant": observed_rows > 0 and distinct_observed == 1,
        f"{prefix}_effective_constant": effective_distinct <= 1,
    }


def align_source_to_anchor(
    source_name: str,
    source: pd.DataFrame,
    anchor: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    keys = list(SOURCE_KEYS[source_name])
    source_subset = source.loc[:, [*keys, *feature_columns]].copy()
    anchor_subset = anchor.loc[:, ["RID", "VISCODE2", "split"]].copy()

    if SOURCE_GRAIN[source_name] == "visit":
        aligned = anchor_subset.merge(
            source_subset,
            on=["RID", "VISCODE2"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    else:
        aligned = anchor_subset.merge(
            source_subset,
            on=["RID"],
            how="left",
            validate="many_to_one",
            sort=False,
        )

    if len(aligned) != len(anchor_subset):
        raise ProfileAbort(
            f"{source_name} alignment changed the supervised anchor row count."
        )
    return aligned


def validate_prerequisites(
    paths: Paths,
    checks: Checks,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path], pd.DataFrame]:
    # BENCHMARK01 is the immutable upstream package validator.
    checks.add(
        "benchmark01_summary_exists",
        paths.benchmark01_summary.is_file(),
        True,
        paths.benchmark01_summary.is_file(),
    )
    checks.add(
        "a06_report_exists",
        paths.a06_report.is_file(),
        True,
        paths.a06_report.is_file(),
    )
    if not paths.benchmark01_summary.is_file() or not paths.a06_report.is_file():
        raise ProfileAbort("BENCHMARK01 and A06 SAFE summaries are required.")

    benchmark01 = read_json(paths.benchmark01_summary)
    a06_report = read_json(paths.a06_report)

    b01_pass = bool(benchmark01.get("validation_passed"))
    checks.add("benchmark01_validation_passed", b01_pass, True, b01_pass)

    b01_aggregate = benchmark01.get("aggregate_summary", {})
    b01_package = b01_aggregate.get("manifest", {}).get("logical_package_name")
    b01_policy = b01_aggregate.get("temporal_policy", {}).get(
        "temporal_policy_decision_id"
    )
    checks.add(
        "benchmark01_logical_package",
        b01_package == EXPECTED_LOGICAL_PACKAGE_NAME,
        EXPECTED_LOGICAL_PACKAGE_NAME,
        b01_package,
    )
    checks.add(
        "benchmark01_temporal_policy_id",
        b01_policy == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        b01_policy,
    )

    checks.add(
        "a06_status",
        a06_report.get("status") == "PASS",
        "PASS",
        a06_report.get("status"),
    )
    checks.add(
        "a06_failed_checks",
        int(a06_report.get("failed_check_count", -1)) == 0,
        0,
        a06_report.get("failed_check_count"),
    )
    checks.add(
        "a06_script_version",
        a06_report.get("script_version") == EXPECTED_A06_SCRIPT_VERSION,
        EXPECTED_A06_SCRIPT_VERSION,
        a06_report.get("script_version"),
    )

    a06_upstream = a06_report.get("upstream", {})
    checks.add(
        "a06_upstream_package",
        a06_upstream.get("logical_package_name") == EXPECTED_LOGICAL_PACKAGE_NAME,
        EXPECTED_LOGICAL_PACKAGE_NAME,
        a06_upstream.get("logical_package_name"),
    )
    checks.add(
        "a06_temporal_policy_id",
        a06_upstream.get("temporal_policy_decision_id")
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        a06_upstream.get("temporal_policy_decision_id"),
    )
    checks.add(
        "a06_ledger_version",
        a06_upstream.get("crosscheck_ledger_version") == EXPECTED_LEDGER_VERSION,
        EXPECTED_LEDGER_VERSION,
        a06_upstream.get("crosscheck_ledger_version"),
    )
    checks.add(
        "a06_ledger_sha256",
        a06_upstream.get("crosscheck_ledger_sha256") == EXPECTED_LEDGER_SHA256,
        EXPECTED_LEDGER_SHA256,
        a06_upstream.get("crosscheck_ledger_sha256"),
    )

    invariants = a06_report.get("frozen_invariants", {})
    frozen_bool_checks = {
        "participant_mapping_changed": False,
        "supervised_population_changed": False,
        "target_catalog_changed": False,
        "target_values_changed": False,
        "temporal_policy_changed": False,
        "support_threshold_changed": False,
    }
    for key, expected in frozen_bool_checks.items():
        checks.add(
            f"a06_invariant__{key}",
            invariants.get(key) is expected,
            expected,
            invariants.get(key),
        )
    checks.add(
        "a06_support_threshold",
        int(invariants.get("support_threshold", -1)) == 50,
        50,
        invariants.get("support_threshold"),
    )

    directive_checks = a06_report.get("directive_policy_checks", {})
    directive_failures = sorted(
        key
        for key, payload in directive_checks.items()
        if not isinstance(payload, dict) or payload.get("status") != "PASS"
    )
    checks.add(
        "a06_all_directive_policy_checks_pass",
        bool(directive_checks) and not directive_failures,
        [],
        directive_failures,
    )

    privacy = a06_report.get("privacy", {})
    checks.add(
        "a06_safe_report_privacy",
        privacy.get("SAFE_TO_SHARE") is True
        and privacy.get("contains_only_aggregate_counts_column_names_booleans_and_hashes")
        is True,
        True,
        {
            "SAFE_TO_SHARE": privacy.get("SAFE_TO_SHARE"),
            "aggregate_only": privacy.get(
                "contains_only_aggregate_counts_column_names_booleans_and_hashes"
            ),
        },
    )

    if not checks.passed:
        raise ProfileAbort("BENCHMARK01/A06 contract validation failed.")

    checks.add("package_root_exists", paths.package.is_dir(), True, paths.package.is_dir())
    checks.add(
        "crosschecked_root_exists",
        paths.crosschecked.is_dir(),
        True,
        paths.crosschecked.is_dir(),
    )
    if not paths.package.is_dir() or not paths.crosschecked.is_dir():
        raise ProfileAbort("Frozen package or crosschecked input overlay was not found.")

    artifacts: dict[str, Path] = {
        name: paths.package / relative
        for name, relative in PACKAGE_ARTIFACTS.items()
    }
    artifacts.update(
        {
            name: paths.crosschecked / relative
            for name, relative in CROSSCHECK_ARTIFACTS.items()
        }
    )
    for name, path in artifacts.items():
        checks.add(f"artifact__{name}", path.is_file(), True, path.is_file())
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        raise ProfileAbort(f"Required artifacts are missing: {missing}")

    crosschecked_manifest = read_json(artifacts["crosschecked_manifest"])
    checks.add(
        "crosschecked_manifest_stage",
        crosschecked_manifest.get("stage") == "auditA06_freeze_crosschecked_inputs",
        "auditA06_freeze_crosschecked_inputs",
        crosschecked_manifest.get("stage"),
    )
    checks.add(
        "crosschecked_manifest_output_name",
        crosschecked_manifest.get("output_name") == CROSSCHECKED_INPUTS_NAME,
        CROSSCHECKED_INPUTS_NAME,
        crosschecked_manifest.get("output_name"),
    )
    checks.add(
        "crosschecked_manifest_upstream_package",
        crosschecked_manifest.get("upstream_frozen_package_name")
        == EXPECTED_LOGICAL_PACKAGE_NAME,
        EXPECTED_LOGICAL_PACKAGE_NAME,
        crosschecked_manifest.get("upstream_frozen_package_name"),
    )
    checks.add(
        "crosschecked_manifest_temporal_policy",
        crosschecked_manifest.get("upstream_temporal_policy_decision_id")
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        crosschecked_manifest.get("upstream_temporal_policy_decision_id"),
    )
    checks.add(
        "crosschecked_manifest_ledger_sha256",
        crosschecked_manifest.get("crosscheck_ledger_sha256")
        == EXPECTED_LEDGER_SHA256,
        EXPECTED_LEDGER_SHA256,
        crosschecked_manifest.get("crosscheck_ledger_sha256"),
    )

    # Verify the A06 overlay itself byte-for-byte against its committed manifest.
    expected_catalog_hash = crosschecked_manifest.get("feature_catalog_sha256")
    observed_catalog_hash = sha256_file(artifacts["feature_catalog"])
    checks.add(
        "crosschecked_feature_catalog_sha256",
        observed_catalog_hash == expected_catalog_hash,
        expected_catalog_hash,
        observed_catalog_hash,
    )
    expected_directives_hash = crosschecked_manifest.get("directives_sha256")
    observed_directives_hash = sha256_file(artifacts["directives"])
    checks.add(
        "crosschecked_directives_sha256",
        observed_directives_hash == expected_directives_hash,
        expected_directives_hash,
        observed_directives_hash,
    )

    expected_source_hashes = crosschecked_manifest.get("input_source_sha256", {})
    for source_name in INPUT_SOURCES:
        path = paths.crosschecked / "input_sources" / f"{source_name}_model_ready.csv.gz"
        artifacts[f"input__{source_name}"] = path
        exists = path.is_file()
        checks.add(f"artifact__input__{source_name}", exists, True, exists)
        if exists:
            observed = sha256_file(path)
            expected = expected_source_hashes.get(source_name)
            checks.add(
                f"crosschecked_source_sha256__{source_name}",
                observed == expected,
                expected,
                observed,
            )

    feature_catalog = read_csv_text(artifacts["feature_catalog"])
    require_columns(
        feature_catalog,
        ["source_name", "feature_position_within_source", "column_name", "role"],
        "crosschecked input feature catalog",
    )
    checks.add(
        "crosschecked_feature_catalog_count",
        len(feature_catalog) == EXPECTED_CROSSCHECK_FEATURE_COUNT,
        EXPECTED_CROSSCHECK_FEATURE_COUNT,
        len(feature_catalog),
    )
    checks.add(
        "crosschecked_feature_catalog_unique",
        not feature_catalog.duplicated(["source_name", "column_name"]).any(),
        True,
        not feature_catalog.duplicated(["source_name", "column_name"]).any(),
    )
    checks.add(
        "crosschecked_feature_catalog_sources",
        set(feature_catalog["source_name"].dropna().astype(str)) == set(INPUT_SOURCES),
        sorted(INPUT_SOURCES),
        sorted(feature_catalog["source_name"].dropna().astype(str).unique().tolist()),
    )

    directives = read_csv_text(artifacts["directives"])
    require_columns(
        directives,
        [
            "source_name",
            "column_name",
            "crosscheck_action",
            "final_feature_type",
            "value_transform_id",
            "semantic_role",
            "reason",
            "execution_stage",
        ],
        "A06 crosscheck directives",
    )
    checks.add(
        "crosschecked_directive_count",
        len(directives) == EXPECTED_A06_DIRECTIVE_COUNT,
        EXPECTED_A06_DIRECTIVE_COUNT,
        len(directives),
    )
    checks.add(
        "crosschecked_directives_unique",
        not directives.duplicated(["source_name", "column_name"]).any(),
        True,
        not directives.duplicated(["source_name", "column_name"]).any(),
    )

    if not checks.passed:
        raise ProfileAbort("Crosschecked overlay validation failed.")

    return benchmark01, a06_report, artifacts, directives


def profile_features(
    artifacts: dict[str, Path],
    directives: pd.DataFrame,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    anchor = read_csv_text(
        artifacts["anchor"],
        usecols=["RID", "VISCODE2", "split"],
    )
    require_columns(anchor, ["RID", "VISCODE2", "split"], "supervised anchor")
    checks.add("anchor_valid_splits", set(anchor["split"].dropna().tolist()) == set(SPLITS), list(SPLITS), sorted(anchor["split"].dropna().unique().tolist()))
    checks.add("anchor_unique_visit_keys", not anchor.duplicated(["RID", "VISCODE2"]).any(), True, not anchor.duplicated(["RID", "VISCODE2"]).any())

    feature_catalog = read_csv_text(artifacts["feature_catalog"])
    require_columns(
        feature_catalog,
        ["source_name", "feature_position_within_source", "column_name", "role"],
        "input feature catalog",
    )
    checks.add(
        "feature_catalog_sources",
        set(feature_catalog["source_name"].dropna().tolist()) == set(INPUT_SOURCES),
        list(INPUT_SOURCES),
        sorted(feature_catalog["source_name"].dropna().unique().tolist()),
    )

    records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for source_name in INPUT_SOURCES:
        logging.info("Profiling %s.", source_name)
        catalog_source = feature_catalog.loc[
            feature_catalog["source_name"].eq(source_name)
        ].copy()
        catalog_source["feature_position_within_source"] = pd.to_numeric(
            catalog_source["feature_position_within_source"], errors="raise"
        ).astype(int)
        catalog_source = catalog_source.sort_values(
            "feature_position_within_source", kind="stable"
        )
        feature_columns = catalog_source["column_name"].astype(str).tolist()

        source = read_csv_text(artifacts[f"input__{source_name}"])
        require_columns(
            source,
            [*SOURCE_KEYS[source_name], "split", *feature_columns],
            f"{source_name} model-ready source",
        )
        checks.add(
            f"source__{source_name}__unique_keys",
            not source.duplicated(list(SOURCE_KEYS[source_name])).any(),
            True,
            not source.duplicated(list(SOURCE_KEYS[source_name])).any(),
        )
        checks.add(
            f"source__{source_name}__catalog_order",
            [column for column in source.columns if column not in [*SOURCE_KEYS[source_name], "split"]] == feature_columns,
            feature_columns,
            [column for column in source.columns if column not in [*SOURCE_KEYS[source_name], "split"]],
        )

        aligned = align_source_to_anchor(
            source_name=source_name,
            source=source,
            anchor=anchor,
            feature_columns=feature_columns,
        )
        source_split_counts = source["split"].value_counts(dropna=False).to_dict()
        source_records.append(
            {
                "source_name": source_name,
                "grain": SOURCE_GRAIN[source_name],
                "source_rows": int(len(source)),
                "source_participants": int(source["RID"].nunique(dropna=True)),
                "feature_count": len(feature_columns),
                "source_train_rows": int(source_split_counts.get("train", 0)),
                "snapshot_all_rows": int(len(aligned)),
                "snapshot_all_participants": int(aligned["RID"].nunique(dropna=True)),
                "snapshot_train_rows": int(aligned["split"].eq("train").sum()),
                "snapshot_train_participants": int(
                    aligned.loc[aligned["split"].eq("train"), "RID"].nunique(dropna=True)
                ),
            }
        )

        for _, catalog_row in catalog_source.iterrows():
            column_name = str(catalog_row["column_name"])
            native_train_mask = source["split"].eq("train")
            native_train_values = source.loc[native_train_mask, column_name]
            snapshot_train_mask = aligned["split"].eq("train")
            snapshot_train_values = aligned.loc[snapshot_train_mask, column_name]

            directive_match = directives.loc[
                directives["source_name"].eq(source_name)
                & directives["column_name"].eq(column_name)
            ]
            if len(directive_match) > 1:
                raise ProfileAbort(
                    f"Duplicate A06 directive for {source_name}::{column_name}."
                )
            if directive_match.empty:
                directive_payload = {
                    "a06_directive_present": False,
                    "a06_crosscheck_action": "",
                    "a06_final_feature_type": "",
                    "a06_value_transform_id": "",
                    "a06_semantic_role": "",
                    "a06_reason": "",
                    "a06_execution_stage": "",
                }
            else:
                drow = directive_match.iloc[0]
                directive_payload = {
                    "a06_directive_present": True,
                    "a06_crosscheck_action": str(drow["crosscheck_action"]),
                    "a06_final_feature_type": str(drow["final_feature_type"]),
                    "a06_value_transform_id": str(drow["value_transform_id"]),
                    "a06_semantic_role": str(drow["semantic_role"]),
                    "a06_reason": str(drow["reason"]),
                    "a06_execution_stage": str(drow["execution_stage"]),
                }

            record: dict[str, Any] = {
                "source_name": source_name,
                "grain": SOURCE_GRAIN[source_name],
                "feature_position_within_source": int(
                    catalog_row["feature_position_within_source"]
                ),
                "column_name": column_name,
                "catalog_role": str(catalog_row["role"]),
                "catalog_train_only_removal_note": str(
                    catalog_row.get("final_train_only_removal_pending", "") or ""
                ),
                **directive_payload,
            }
            record.update(
                profile_scope(
                    native_train_values,
                    source.loc[native_train_mask, "RID"],
                    "native_train",
                )
            )
            record.update(
                profile_scope(
                    snapshot_train_values,
                    aligned.loc[snapshot_train_mask, "RID"],
                    "snapshot_train",
                )
            )

            record.update(
                infer_type(
                    native_train_values=native_train_values,
                    snapshot_train_values=snapshot_train_values,
                    column_name=column_name,
                )
            )

            train_missing_fraction = record.get("snapshot_train_missing_fraction")
            train_observed_participants = int(
                record["snapshot_train_observed_participants"]
            )
            review_flags = [
                flag
                for flag in str(record.get("semantic_review_flags", "")).split("|")
                if flag
            ]
            if train_missing_fraction is not None:
                if train_missing_fraction >= EXTREME_MISSINGNESS_THRESHOLD:
                    review_flags.append("train_missingness_at_least_99_percent")
                elif train_missing_fraction >= HIGH_MISSINGNESS_THRESHOLD:
                    review_flags.append("train_missingness_at_least_95_percent")
            if train_observed_participants <= 30:
                review_flags.append("low_train_participant_support_review")
            if bool(record["snapshot_train_all_missing"]):
                review_flags.append("snapshot_train_all_missing")
            elif bool(record["snapshot_train_effective_constant"]):
                review_flags.append("snapshot_train_effective_constant")
            elif bool(record["snapshot_train_observed_constant"]):
                review_flags.append(
                    "snapshot_train_observed_value_constant_missingness_may_vary"
                )

            # Heuristic typing remains diagnostic. A frozen A06 directive has precedence.
            if record.get("a06_directive_present"):
                review_flags.append("a06_semantic_directive_locked")

            record["snapshot_train_participant_support_band"] = support_band(
                train_observed_participants
            )
            record["review_flags"] = "|".join(dict.fromkeys(review_flags))
            record["automatic_decision"] = "none_profile_only"
            records.append(record)

    profile = pd.DataFrame(records).sort_values(
        ["source_name", "feature_position_within_source"], kind="stable"
    ).reset_index(drop=True)
    source_summary = pd.DataFrame(source_records).sort_values(
        "source_name", kind="stable"
    ).reset_index(drop=True)

    checks.add("profile_feature_count_matches_catalog", len(profile) == len(feature_catalog), len(feature_catalog), len(profile))
    checks.add("profile_unique_source_column", not profile.duplicated(["source_name", "column_name"]).any(), True, not profile.duplicated(["source_name", "column_name"]).any())

    aggregate = {
        "anchor_rows": int(len(anchor)),
        "anchor_participants": int(anchor["RID"].nunique(dropna=True)),
        "feature_count": int(len(profile)),
        "source_count": int(len(source_summary)),
        "features_with_a06_directive": int(profile["a06_directive_present"].sum()),
        "features_without_a06_directive": int((~profile["a06_directive_present"]).sum()),
    }
    return profile, source_summary, aggregate


def build_schema_draft(profile: pd.DataFrame) -> pd.DataFrame:
    draft = profile.loc[
        :,
        [
            "source_name",
            "grain",
            "feature_position_within_source",
            "column_name",
            "a06_directive_present",
            "a06_crosscheck_action",
            "a06_final_feature_type",
            "a06_value_transform_id",
            "a06_semantic_role",
            "a06_reason",
            "a06_execution_stage",
            "suggested_type",
            "suggested_encoding",
            "native_train_observed_rows",
            "native_train_observed_participants",
            "native_train_observed_fraction",
            "snapshot_train_observed_rows",
            "snapshot_train_observed_participants",
            "snapshot_train_observed_fraction",
            "snapshot_train_observed_participant_fraction",
            "snapshot_train_distinct_observed_values",
            "snapshot_train_participant_support_band",
            "review_flags",
        ],
    ].copy()
    draft["final_include"] = ""
    draft["final_feature_type"] = ""
    draft["final_encoding"] = ""
    draft["semantic_role"] = ""
    draft["decision_reason"] = ""
    draft["review_notes"] = ""
    return draft


def summarize_profile(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    type_summary = (
        profile.groupby(["suggested_type", "suggested_encoding"], dropna=False)
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values(["feature_count", "suggested_type"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )

    support_summary = (
        profile.groupby("snapshot_train_participant_support_band", dropna=False)
        .size()
        .rename("feature_count")
        .reset_index()
        .sort_values("snapshot_train_participant_support_band", kind="stable")
        .reset_index(drop=True)
    )

    source_profile_summary = (
        profile.groupby(["source_name", "grain"], dropna=False)
        .agg(
            feature_count=("column_name", "size"),
            snapshot_train_all_missing_features=("snapshot_train_all_missing", "sum"),
            snapshot_train_effective_constant_features=("snapshot_train_effective_constant", "sum"),
            snapshot_train_observed_constant_features=("snapshot_train_observed_constant", "sum"),
            snapshot_train_missingness_at_least_95_features=(
                "snapshot_train_missing_fraction",
                lambda series: int((series.fillna(1.0) >= HIGH_MISSINGNESS_THRESHOLD).sum()),
            ),
            snapshot_train_missingness_at_least_99_features=(
                "snapshot_train_missing_fraction",
                lambda series: int((series.fillna(1.0) >= EXTREME_MISSINGNESS_THRESHOLD).sum()),
            ),
            multi_response_candidate_features=(
                "suggested_type",
                lambda series: int(series.eq("multi_response_candidate").sum()),
            ),
            low_support_30_or_less_features=(
                "snapshot_train_observed_participants",
                lambda series: int((series <= 30).sum()),
            ),
        )
        .reset_index()
        .sort_values("source_name", kind="stable")
        .reset_index(drop=True)
    )
    return type_summary, support_summary, source_profile_summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


def write_outputs(
    paths: Paths,
    checks: Checks,
    profile: pd.DataFrame,
    source_summary: pd.DataFrame,
    aggregate: dict[str, Any],
    fatal_error: str | None,
) -> None:
    checks_frame = pd.DataFrame(
        [
            {
                "check_name": item.name,
                "passed": item.passed,
                "expected": json.dumps(json_safe(item.expected), ensure_ascii=False),
                "observed": json.dumps(json_safe(item.observed), ensure_ascii=False),
                "details": item.details,
            }
            for item in checks.items
        ]
    )
    checks_frame.to_csv(
        paths.safe / "benchmark02b_validation_checks.csv",
        index=False,
        encoding="utf-8",
    )

    if not profile.empty:
        profile.to_csv(
            paths.safe / "benchmark02b_feature_profile.csv",
            index=False,
            encoding="utf-8",
        )
        build_schema_draft(profile).to_csv(
            paths.safe / "benchmark02b_feature_schema_draft.csv",
            index=False,
            encoding="utf-8",
        )
        type_summary, support_summary, source_profile_summary = summarize_profile(profile)
        type_summary.to_csv(
            paths.safe / "benchmark02b_type_suggestion_summary.csv",
            index=False,
            encoding="utf-8",
        )
        support_summary.to_csv(
            paths.safe / "benchmark02b_support_band_summary.csv",
            index=False,
            encoding="utf-8",
        )
        source_profile_summary.to_csv(
            paths.safe / "benchmark02b_source_feature_summary.csv",
            index=False,
            encoding="utf-8",
        )
    else:
        type_summary = pd.DataFrame()
        support_summary = pd.DataFrame()
        source_profile_summary = pd.DataFrame()

    source_summary.to_csv(
        paths.safe / "benchmark02b_source_table_summary.csv",
        index=False,
        encoding="utf-8",
    )

    passed = checks.passed and fatal_error is None
    summary = {
        "benchmark02b_script_version": SCRIPT_VERSION,
        "benchmark02_script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "benchmark02b",
        "profiling_passed": passed,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": fatal_error,
        "upstream_contract": {
            "logical_frozen_package": EXPECTED_LOGICAL_PACKAGE_NAME,
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "crosschecked_inputs": CROSSCHECKED_INPUTS_NAME,
            "a06_script_version": EXPECTED_A06_SCRIPT_VERSION,
            "crosscheck_ledger_version": EXPECTED_LEDGER_VERSION,
            "crosscheck_ledger_sha256": EXPECTED_LEDGER_SHA256,
        },
        "aggregate_summary": aggregate,
        "type_suggestion_counts": (
            profile["suggested_type"].value_counts(dropna=False).to_dict()
            if not profile.empty
            else {}
        ),
        "train_support_band_counts": (
            profile["snapshot_train_participant_support_band"].value_counts(dropna=False).to_dict()
            if not profile.empty
            else {}
        ),
        "advisory_flags": {
            "snapshot_train_all_missing_features": int(profile["snapshot_train_all_missing"].sum()) if not profile.empty else 0,
            "snapshot_train_effective_constant_features": int(profile["snapshot_train_effective_constant"].sum()) if not profile.empty else 0,
            "snapshot_train_missingness_at_least_95_features": int((profile["snapshot_train_missing_fraction"].fillna(1.0) >= HIGH_MISSINGNESS_THRESHOLD).sum()) if not profile.empty else 0,
            "snapshot_train_missingness_at_least_99_features": int((profile["snapshot_train_missing_fraction"].fillna(1.0) >= EXTREME_MISSINGNESS_THRESHOLD).sum()) if not profile.empty else 0,
            "snapshot_train_support_30_or_less_features": int((profile["snapshot_train_observed_participants"] <= 30).sum()) if not profile.empty else 0,
            "multi_response_candidate_features": int(profile["suggested_type"].eq("multi_response_candidate").sum()) if not profile.empty else 0,
        },
        "privacy": {
            "contains_individual_records": False,
            "contains_participant_identifiers": False,
            "contains_visit_identifiers": False,
            "contains_medical_values": False,
            "contains_only_column_names_and_aggregate_profile_statistics": True,
        },
        "methodological_note": (
            "No additional feature was automatically included or excluded in BENCHMARK02. "
            "A06 semantic directives are frozen upstream and take precedence over heuristic "
            "type suggestions. Remaining heuristic suggestions are diagnostic only."
        ),
    }
    (paths.safe / "benchmark02b_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    story = [
        "# BENCHMARK02B — Aggregate input-feature profile",
        "",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        f"- Sources profiled: {aggregate.get('source_count', 'NA')}",
        f"- Candidate input features profiled: {aggregate.get('feature_count', 'NA')}",
        f"- Snapshot-all anchor visits used for aggregate support: {aggregate.get('anchor_rows', 'NA')}",
        f"- Snapshot-all participants: {aggregate.get('anchor_participants', 'NA')}",
        "",
        "## Scope",
        "",
        "The profiler used the A06 crosschecked input overlay plus the unchanged supervised anchor from frozen_experiment_v1_1. It produced aggregate statistics only. It did not construct final model matrices, impute values, remove additional features, fit encoders, select a support threshold, or train models.",
        "",
        "## Interpretation",
        "",
        "- `native_train_*` columns describe canonical model-ready rows belonging only to training participants, including rows that may later serve as longitudinal history.",
        "- `snapshot_train_*` columns describe support after alignment only to supervised target visits in the training split.",
        "- Validation and test feature distributions are intentionally not profiled in this stage, so they cannot influence type decisions, encoding, or support thresholds.",
        "- Type suggestions are heuristic diagnostics. Frozen A06 semantic directives take precedence whenever present.",
        "- The support threshold remains frozen at 50 participants; BENCHMARK02 does not reselect it.",
        "- No new automatic feature decision was made.",
        "",
        "## Privacy",
        "",
        "No RID, VISCODE2, individualized row, medical value, category value, minimum, maximum, mean, median, or example value was written to these reports. Feature profiling used training participants only.",
    ]
    if fatal_error:
        story.extend(["", "## Fatal error", "", fatal_error])
    (paths.safe / "BENCHMARK02_PROFILE_STORY.md").write_text(
        "\n".join(story) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    configure_logging(paths)
    logging.info("benchmark02b_profile_input_features v%s", SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Frozen package: %s", paths.package)
    logging.info("Crosschecked inputs: %s", paths.crosschecked)

    checks = Checks()
    profile = pd.DataFrame()
    source_summary = pd.DataFrame()
    aggregate: dict[str, Any] = {}
    fatal_error: str | None = None

    try:
        _, _, artifacts, directives = validate_prerequisites(paths, checks)
        profile, source_summary, aggregate = profile_features(
            artifacts, directives, checks
        )
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("BENCHMARK02 aborted")

    write_outputs(
        paths=paths,
        checks=checks,
        profile=profile,
        source_summary=source_summary,
        aggregate=aggregate,
        fatal_error=fatal_error,
    )

    passed = checks.passed and fatal_error is None
    if passed:
        logging.info(
            "BENCHMARK02 PASSED | sources=%s | features=%s | anchor_visits=%s",
            aggregate.get("source_count"),
            aggregate.get("feature_count"),
            aggregate.get("anchor_rows"),
        )
        logging.info("No feature was excluded. No model was trained.")
        return 0

    logging.error(
        "BENCHMARK02 FAILED | failed_checks=%d | fatal_error=%s",
        checks.failed,
        fatal_error,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
