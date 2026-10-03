#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AUDIT A06 — freeze cross-checked non-MRI inputs for the corrected benchmark.
#
# Purpose
# -------
# Consume the validated frozen experiment package v1.1 and the frozen
# cross-check ledger v1.0.0, apply the agreed semantic corrections to the eight
# non-MRI input sources, and materialize a new local-only overlay:
#
#     data/crosschecked_inputs_v1/
#
# The upstream frozen package is never modified. Participant split, supervised
# MRI population, MRI targets, temporal policy, support threshold, and model
# families remain frozen.
#
# A06 performs only data-layer corrections that are already frozen in the ledger:
# - remove administration/protocol metadata and semantically excluded fields;
# - harmonize ADAS UNABLE implicit/explicit conducted state;
# - derive a canonical MMSE WORLD-backwards score;
# - refine PTDEMOG baseline selection only inside the earliest protocol wave;
# - harmonize PTRACCAT to a common cross-phase ontology;
# - prepare local-only birth-year metadata for later AGE_AT_TARGET derivation;
# - validate APOE GENOTYPE against the official six-category domain;
# - emit explicit feature directives for the downstream semantic schema stages.
#
# A06 intentionally does NOT:
# - rerun or reselect the participant split;
# - change the 9,600 target-complete MRI visits or the 323 MRI targets;
# - change the temporal policy decision;
# - select a new support threshold;
# - impute, scale, one-hot encode, fit PCA, train models, or inspect test labels;
# - derive AGE_AT_TARGET itself (that is visit-specific and is frozen as a
#   BENCHMARK05-stage derivation; A06 prepares/validates its local prerequisites).
#
# Privacy
# -------
# SAFE outputs contain only aggregate counts, column names, booleans, hashes,
# and status flags. They contain no RID values, visit codes, clinical dates,
# participant-level medical/cognitive values, genotype values, predictions, or
# errors.
#
# LOCAL_ONLY outputs may contain individualized identifiers, visit metadata, and
# before/after values required for manual inspection. They are explicitly marked
# DO_NOT_SHARE and must never be uploaded, published, versioned, or sent to an
# LLM.

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Frozen contract
# -----------------------------------------------------------------------------

SCRIPT_VERSION: Final[str] = "0.1.2"
STAGE_NAME: Final[str] = "auditA06_freeze_crosschecked_inputs"
OUTPUT_NAME: Final[str] = "crosschecked_inputs_v1"

EXPECTED_PACKAGE_NAME: Final[str] = "frozen_experiment_v1_1"
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"
EXPECTED_LEDGER_VERSION: Final[str] = "1.0.0"
EXPECTED_LEDGER_SHA256: Final[str] = (
    "0bfe87ca42e9d9c74c804878a7c09372bc4eb856cf2d60200c315dd301b04fa5"
)

EXPECTED_PARTICIPANTS: Final[int] = 2_839
EXPECTED_VISITS: Final[int] = 9_600
EXPECTED_TARGETS: Final[int] = 323
EXPECTED_SPLIT_COUNTS: Final[dict[str, int]] = {
    "train": 1_987,
    "validation": 426,
    "test": 426,
}
EXPECTED_SUPPORT_THRESHOLD: Final[int] = 50

SAFE_MINIMUM_CELL_COUNT: Final[int] = 5

RID: Final[str] = "RID"
VISCODE2: Final[str] = "VISCODE2"
SPLIT: Final[str] = "split"

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

SOURCE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    **{source: (RID, VISCODE2) for source in VISIT_SOURCES},
    "PTDEMOG": (RID,),
    "APOERES": (RID,),
}

MANIFEST_RELATIVE: Final[str] = "04_official_split/official_frozen_manifest.json"
POLICY_LOCK_RELATIVE: Final[str] = "04_official_split/strict_date_temporal_policy_lock.json"
PARTICIPANT_SPLIT_RELATIVE: Final[str] = "04_official_split/official_participant_split.csv"
ANCHOR_RELATIVE: Final[str] = "03_final_ready_tables/supervised_anchor_visits.csv.gz"
TARGETS_RELATIVE: Final[str] = "03_final_ready_tables/UCSFFSX7_targets_323.csv.gz"
TARGET_CATALOG_RELATIVE: Final[str] = "03_final_ready_tables/MRI_target_catalog_323.csv"
FEATURE_CATALOG_RELATIVE: Final[str] = "03_final_ready_tables/input_feature_catalog.csv"
INPUT_SOURCE_DIR_RELATIVE: Final[str] = "03_final_ready_tables/input_sources"
PTDEMOG_CANONICAL_RELATIVE: Final[str] = (
    "01_postprocessed_sources/PTDEMOG_canonical_resolved.csv.gz"
)
PTDEMOG_EXCLUDED_RELATIVE: Final[str] = (
    "02_cuts_and_adjustments/PTDEMOG_excluded_rows.csv.gz"
)
ROW_ACTION_SUMMARY_RELATIVE: Final[str] = "02_cuts_and_adjustments/row_action_summary.csv"

BENCHMARK01_SUMMARY_RELATIVE: Final[str] = (
    "reports/benchmark01b_validate_frozen_package/safe/"
    "benchmark01b_validation_summary.json"
)

# Individualized local audit files. Never share.
LOCAL_RECORD_AUDIT_NAME: Final[str] = "LOCAL_ONLY_record_actions.csv.gz"
LOCAL_PTDEMOG_BASELINE_NAME: Final[str] = "LOCAL_ONLY_ptdemog_baseline_resolution.csv.gz"
LOCAL_AGE_METADATA_NAME: Final[str] = "LOCAL_ONLY_ptdemog_age_derivation_metadata.csv.gz"
LOCAL_COLUMN_ACTIONS_NAME: Final[str] = "LOCAL_ONLY_column_actions.csv"
LOCAL_README_NAME: Final[str] = "LOCAL_ONLY_DO_NOT_SHARE.md"

# -----------------------------------------------------------------------------
# Frozen modality decisions
# -----------------------------------------------------------------------------

ADAS_EXCLUDE_EXACT: Final[frozenset[str]] = frozenset(
    {
        "NDREASON",
        "SOURCE",
        "WORDLIST",
        "Q9TASK",
        "Q10TASK",
        "Q11TASK",
        "Q12TASK",
    }
)
ADAS_UNABLE_FIELDS: Final[tuple[str, ...]] = (
    "Q1UNABLE",
    "Q2UNABLE",
    "Q3UNABLE",
    "Q4UNABLE",
    "Q5UNABLE",
    "Q6UNABLE",
    "Q7UNABLE",
    "Q8UNABLE",
    "Q13UNABLE",
)
ADAS_UNABLE_COMPANIONS: Final[dict[str, tuple[str, ...]]] = {
    "Q1UNABLE": ("Q1SCORE", "Q1TR1", "Q1TR2", "Q1TR3", "Q1TRIT", "Q1TR2T", "Q1TRT"),
    "Q2UNABLE": ("Q2SCORE", "Q2TASK"),
    "Q3UNABLE": ("Q3SCORE", "Q3TASK1", "Q3TASK2", "Q3TASK3", "Q3TASK4"),
    "Q4UNABLE": ("Q4SCORE", "Q4TASK"),
    "Q5UNABLE": tuple(["Q5SCORE", "Q5SCORE_CUE", "Q5FINGER", *[f"Q5NAME{i}" for i in range(1, 13)]]),
    "Q6UNABLE": ("Q6SCORE", "Q6TASK"),
    "Q7UNABLE": ("Q7SCORE", "Q7TASK"),
    "Q8UNABLE": tuple(["Q8SCORE", *[f"Q8WORD{i}" for i in range(1, 25)]]),
    "Q13UNABLE": ("Q13SCORE", "Q13TASKA", "Q13TASKB", "Q13TASKC"),
}
ADAS_ORDINAL_SCORES: Final[tuple[str, ...]] = (
    "Q9SCORE",
    "Q10SCORE",
    "Q11SCORE",
    "Q12SCORE",
)

CDR_EXCLUDE: Final[frozenset[str]] = frozenset({"CDSOURCE", "CDVERSION"})
CDR_ORDINAL_CATEGORICAL: Final[tuple[str, ...]] = (
    "CDMEMORY",
    "CDORIENT",
    "CDJUDGE",
    "CDCOMMUN",
    "CDHOME",
    "CDCARE",
    "CDGLOBAL",
)

FAQ_EXCLUDE: Final[frozenset[str]] = frozenset({"SOURCE"})
FAQ_ITEMS: Final[tuple[str, ...]] = (
    "FAQFINAN",
    "FAQFORM",
    "FAQSHOP",
    "FAQGAME",
    "FAQBEVG",
    "FAQMEAL",
    "FAQEVENT",
    "FAQTV",
    "FAQREM",
    "FAQTRAVL",
)

MMSE_EXCLUDE_EXACT: Final[frozenset[str]] = frozenset(
    {
        "NDREASON",
        "SOURCE",
        "WORDLIST",
        "MMD",
        "MML",
        "MMR",
        "MMO",
        "MMW",
        "WORLDSCORE",
    }
)
MMSE_OLD_WORLD_FIELDS: Final[tuple[str, ...]] = ("MMD", "MML", "MMR", "MMO", "MMW")
MMSE_RAW_WORLD_LETTERS: Final[tuple[str, ...]] = tuple(f"MMLTR{i}" for i in range(1, 8))
MMSE_CANONICAL_WORLD: Final[str] = "MM_WORLD_SCORE"

MOCA_EXCLUDE: Final[frozenset[str]] = frozenset({"SOURCE"})
MOCA_DELAYED_RECALL_FIELDS: Final[tuple[str, ...]] = tuple(f"DELW{i}" for i in range(1, 6))

NEUROBAT_EXCLUDE: Final[frozenset[str]] = frozenset(
    {
        "SOURCE",
        "LMSTORY",
        "LIMMEND",
        "LDELBEGIN",
        "AVENDED",
        "AVDELBEGAN",
        "BNTND",
        "ANARTND",
        "ANART",
    }
)
NEUROBAT_RETAIN_SUMMARIES: Final[tuple[str, ...]] = (
    "CLOCKSCOR",
    "COPYSCOR",
    "BNTTOTAL",
    "MINTTOTAL",
)

PTDEMOG_CORE: Final[tuple[str, ...]] = (
    "PTGENDER",
    "PTHAND",
    "PTMARRY",
    "PTEDUCAT",
    "PTNOTRT",
    "PTHOME",
    "PTPLANG",
    "PTETHCAT",
    "PTRACCAT",
)
PTDEMOG_ALWAYS_EXCLUDE: Final[frozenset[str]] = frozenset(
    {
        "PTSOURCE",
        "PTDOB",
        "PTDOBYY",
        "PTWORKHS",
        "PTWORK",
        "PTTLANG",
        "PTRTYR",
        "PTADBEG",
        "PTCOGBEG",
        "PTADDX",
        "PTIDENT",
        "PTORIENT",
        "PTORIENTOT",
    }
)

# Frozen ADNI4-only demographic expansion identified during the manual audit.
PTDEMOG_ADNI4_ONLY_EXPANSION: Final[frozenset[str]] = frozenset(
    {
        "DD_CRF_VERSION_LABEL", "HAS_QC_ERROR", "LANGUAGE_CODE", "ORIENTOTOT",
        "PTAMIAN", "PTASIAN", "PTASIANOT", "PTBIRGR", "PTBIRGROT", "PTBIRPL",
        "PTBIRPLOT", "PTBIRPR", "PTBIRPROT", "PTBLACK", "PTBORN", "PTCLANG",
        "PTCLANGOTH", "PTENGSPK", "PTENGSPKAGE", "PTETHCATH", "PTETHCATHOT",
        "PTIDENT", "PTIDENTOTH", "PTIMMAGE", "PTIMMWHY", "PTIMMWHYOT",
        "PTLANGPR1", "PTLANGPR2", "PTLANGPR3", "PTLANGPR4", "PTLANGPR5",
        "PTLANGPR6", "PTLANGPRSP", "PTLANGPRSP1", "PTLANGPRSP2", "PTLANGPRSP3",
        "PTLANGPRSP4", "PTLANGPRSP5", "PTLANGPRSP6", "PTLANGRD1", "PTLANGRD2",
        "PTLANGRD3", "PTLANGRD4", "PTLANGRD5", "PTLANGRD6", "PTLANGSP",
        "PTLANGSP1", "PTLANGSP2", "PTLANGSP3", "PTLANGSP4", "PTLANGSP5",
        "PTLANGSP6", "PTLANGSPOTH", "PTLANGTTL", "PTLANGUN1", "PTLANGUN2",
        "PTLANGUN3", "PTLANGUN4", "PTLANGUN5", "PTLANGUN6", "PTLANGWR",
        "PTLANGWR1", "PTLANGWR2", "PTLANGWR3", "PTLANGWR4", "PTLANGWR5",
        "PTLANGWR6", "PTLANGWROTH", "PTNLANG", "PTNLANGOT", "PTOPI", "PTOPIOT",
        "PTORIENT", "PTORIENTOT", "PTSPOTTIM", "PTSPTIM", "PTWHITE", "PTWORKDET",
    }
)

APOE_ALLOWED_GENOTYPES: Final[frozenset[str]] = frozenset(
    {"2/2", "2/3", "2/4", "3/3", "3/4", "4/4"}
)

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


class A06Abort(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    output: Path
    staging: Path
    safe: Path
    local_only: Path
    logs: Path
    ledger: Path
    benchmark01_summary: Path


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


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze cross-checked non-MRI inputs without changing the frozen split."
    )
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--benchmark01-summary", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path(__file__).resolve().parents[2]
    package = (
        args.package_root.expanduser().resolve()
        if args.package_root is not None
        else root / "data" / "processed" / "frozen_experiment_v1_1"
    )
    output = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else root / "data" / "processed" / OUTPUT_NAME
    )
    staging = output.parent / f".{output.name}__staging"
    reports_root = (
        args.reports_root.expanduser().resolve()
        if args.reports_root is not None
        else root / "reports"
    )
    stage = reports_root / STAGE_NAME
    safe = stage / "safe"
    local_only = stage / "local_only"
    logs = stage / "logs"

    if args.ledger is not None:
        ledger = args.ledger.expanduser().resolve()
    else:
        candidates = [
            root / "config" / "crosscheck_ledger_frozen_v1.0.0.json",
            root / "crosscheck_ledger_frozen_v1.0.0.json",
            Path(__file__).resolve().parent / "crosscheck_ledger_frozen_v1.0.0.json",
        ]
        existing = [path for path in candidates if path.is_file()]
        ledger = existing[0] if existing else candidates[0]

    benchmark01_summary = (
        args.benchmark01_summary.expanduser().resolve()
        if args.benchmark01_summary is not None
        else root / BENCHMARK01_SUMMARY_RELATIVE
    )

    safe.mkdir(parents=True, exist_ok=True)
    local_only.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    return Paths(
        root=root,
        package=package,
        output=output,
        staging=staging,
        safe=safe,
        local_only=local_only,
        logs=logs,
        ledger=ledger,
        benchmark01_summary=benchmark01_summary,
    )


def configure_logging(paths: Paths) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "auditA06.log", encoding="utf-8"),
        ],
        force=True,
    )


def short_text(value: Any, limit: int = 600) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_text(path: Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="infer",
        dtype="string",
        keep_default_na=False,
        low_memory=False,
        usecols=list(usecols) if usecols is not None else None,
    )
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    for column in frame.columns:
        series = frame[column].astype("string").str.strip()
        series = series.mask(series.eq(""), pd.NA)
        series = series.mask(series.isin(["-1", "-4"]), pd.NA)
        frame[column] = series
    return frame


def write_csv_deterministic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def write_csv_gzip_deterministic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def normalize_rid(series: pd.Series, context: str) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    invalid = text.notna() & numeric.isna()
    non_integer = numeric.notna() & numeric.ne(numeric.round())
    if invalid.any() or non_integer.any():
        raise A06Abort(
            f"Invalid RID in {context}: nonnumeric={int(invalid.sum())}, "
            f"noninteger={int(non_integer.sum())}"
        )
    return numeric.round().astype("Int64").astype("string")


def normalize_viscode(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def normalize_code_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"-1", "-4"}:
        return None
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def parse_protocol_month(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    text = str(value).strip().casefold()
    if text in {"bl", "sc", "scmri", "m00", "v01"}:
        return 0.0
    match = re.fullmatch(r"m(\d+)", text)
    if match:
        return float(int(match.group(1)))
    match = re.fullmatch(r"y(\d+)", text)
    if match:
        return float(int(match.group(1)) * 12)
    return float("nan")


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().mask(lambda x: x.eq(""), pd.NA)
    try:
        return pd.to_datetime(text, errors="coerce", format="mixed").dt.normalize()
    except (TypeError, ValueError):
        return pd.to_datetime(text, errors="coerce").dt.normalize()


def suppress_count(value: int) -> int | str:
    value = int(value)
    if value == 0:
        return 0
    if value < SAFE_MINIMUM_CELL_COUNT:
        return f"<{SAFE_MINIMUM_CELL_COUNT}"
    return value


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise A06Abort(f"{label} is missing required columns: {missing}")


def row_hash(row: pd.Series, columns: Sequence[str]) -> str:
    payload: list[str] = []
    for column in columns:
        value = row.get(column, pd.NA)
        payload.append("<NA>" if pd.isna(value) else str(value).strip())
    return sha256_text("\x1f".join(payload))


def values_equal(left: Any, right: Any) -> bool:
    left_missing = left is None or pd.isna(left) or str(left).strip() == ""
    right_missing = right is None or pd.isna(right) or str(right).strip() == ""
    if left_missing or right_missing:
        return left_missing and right_missing
    left_text = str(left).strip()
    right_text = str(right).strip()
    if left_text == right_text:
        return True
    try:
        left_num = float(left_text)
        right_num = float(right_text)
        if math.isfinite(left_num) and math.isfinite(right_num):
            return math.isclose(left_num, right_num, rel_tol=0.0, abs_tol=1e-12)
    except ValueError:
        pass
    return left_text.casefold() == right_text.casefold()


def rows_equivalent(left: pd.Series, right: pd.Series, columns: Sequence[str]) -> bool:
    return all(values_equal(left.get(column), right.get(column)) for column in columns)


def row_dominates(left: pd.Series, right: pd.Series, columns: Sequence[str]) -> bool:
    has_extra = False
    for column in columns:
        lval = left.get(column, pd.NA)
        rval = right.get(column, pd.NA)
        lmissing = pd.isna(lval) or str(lval).strip() == ""
        rmissing = pd.isna(rval) or str(rval).strip() == ""
        if not rmissing:
            if lmissing or not values_equal(lval, rval):
                return False
        elif not lmissing:
            has_extra = True
    return has_extra


def sort_source(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    keys = [column for column in SOURCE_KEYS[source_name] if column in frame.columns]
    return frame.sort_values(keys, kind="stable", na_position="last").reset_index(drop=True)


def split_multi_tokens(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[|;]", text) if token.strip()]


# -----------------------------------------------------------------------------
# Source transformations
# -----------------------------------------------------------------------------


def drop_columns(
    frame: pd.DataFrame,
    source_name: str,
    exact: Iterable[str] = (),
    regexes: Sequence[re.Pattern[str]] = (),
) -> tuple[pd.DataFrame, list[str]]:
    requested = set(exact)
    removed = set(column for column in frame.columns if column in requested)
    for column in frame.columns:
        if any(pattern.fullmatch(column) for pattern in regexes):
            removed.add(column)
    protected = set(SOURCE_KEYS[source_name]) | {SPLIT}
    illegal = sorted(removed & protected)
    if illegal:
        raise A06Abort(f"Attempted to remove protected {source_name} keys: {illegal}")
    return frame.drop(columns=sorted(removed), errors="ignore"), sorted(removed)


def harmonize_adas_unable(
    frame: pd.DataFrame,
    local_actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    output = frame.copy()
    stats: dict[str, dict[str, int]] = {}

    for unable in ADAS_UNABLE_FIELDS:
        if unable not in output.columns:
            continue
        companions = [column for column in ADAS_UNABLE_COMPANIONS[unable] if column in output.columns]
        if not companions:
            raise A06Abort(f"{unable} has no available companion fields for harmonization.")

        raw = output[unable].copy()
        normalized = raw.map(normalize_code_value).astype("string")
        companion_observed = output[companions].notna().any(axis=1)
        implicit_conducted = normalized.isna() & companion_observed
        explicit_conducted = normalized.eq("0").fillna(False)
        documented_reason = normalized.notna() & ~explicit_conducted
        unknown = normalized.isna() & ~companion_observed

        output[unable] = normalized
        output.loc[implicit_conducted, unable] = "0"

        for index in output.index[implicit_conducted]:
            evidence = [column for column in companions if pd.notna(frame.at[index, column])]
            local_actions.append(
                {
                    "source_name": "ADAS",
                    "action": "harmonize_implicit_conducted",
                    "RID": frame.at[index, RID],
                    "VISCODE2": frame.at[index, VISCODE2],
                    "column_name": unable,
                    "old_value": pd.NA,
                    "new_value": "0",
                    "reason": "UNABLE missing while corresponding task/score is observed",
                    "evidence_columns": ";".join(evidence),
                }
            )

        stats[unable] = {
            "implicit_conducted_filled": int(implicit_conducted.sum()),
            "explicit_conducted_preserved": int(explicit_conducted.sum()),
            "documented_reason_preserved": int(documented_reason.sum()),
            "unknown_missing_preserved": int(unknown.sum()),
        }

    return output, stats


def derive_mmse_world_score(
    frame: pd.DataFrame,
    local_actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    require_columns(output, [*MMSE_OLD_WORLD_FIELDS, "WORLDSCORE"], "MMSE WORLD harmonization")

    old_numeric = pd.DataFrame(index=output.index)
    invalid_old = pd.Series(False, index=output.index)
    for column in MMSE_OLD_WORLD_FIELDS:
        numeric = pd.to_numeric(output[column], errors="coerce")
        observed = output[column].notna()
        invalid = observed & (~numeric.isin([0.0, 1.0]))
        invalid_old |= invalid
        old_numeric[column] = numeric

    old_observed_count = output[list(MMSE_OLD_WORLD_FIELDS)].notna().sum(axis=1)
    old_complete = old_observed_count.eq(len(MMSE_OLD_WORLD_FIELDS))
    old_partial = old_observed_count.between(1, len(MMSE_OLD_WORLD_FIELDS) - 1)
    old_score = old_numeric.sum(axis=1, min_count=len(MMSE_OLD_WORLD_FIELDS))

    new_numeric = pd.to_numeric(output["WORLDSCORE"], errors="coerce")
    new_observed = output["WORLDSCORE"].notna()
    invalid_new = new_observed & (
        new_numeric.isna()
        | new_numeric.lt(0)
        | new_numeric.gt(5)
        | ~np.isclose(new_numeric.fillna(0), np.round(new_numeric.fillna(0)))
    )

    both = old_complete & new_observed
    conflict = both & ~np.isclose(old_score.fillna(-999), new_numeric.fillna(-998))

    derived = pd.Series(pd.NA, index=output.index, dtype="Float64")
    derived.loc[old_complete] = old_score.loc[old_complete].astype("Float64")
    derived.loc[~old_complete & new_observed] = new_numeric.loc[~old_complete & new_observed].astype("Float64")

    for index in output.index[derived.notna()]:
        old_values = ";".join(
            "" if pd.isna(output.at[index, column]) else str(output.at[index, column])
            for column in MMSE_OLD_WORLD_FIELDS
        )
        local_actions.append(
            {
                "source_name": "MMSE",
                "action": "derive_MM_WORLD_SCORE",
                "RID": output.at[index, RID],
                "VISCODE2": output.at[index, VISCODE2],
                "column_name": MMSE_CANONICAL_WORLD,
                "old_value": f"legacy=[{old_values}];WORLDSCORE={output.at[index, 'WORLDSCORE']}",
                "new_value": derived.at[index],
                "reason": "canonical cross-phase WORLD-backwards representation",
                "evidence_columns": ";".join(MMSE_OLD_WORLD_FIELDS) + ";WORLDSCORE",
            }
        )

    output[MMSE_CANONICAL_WORLD] = derived

    stats = {
        "legacy_complete_rows": int(old_complete.sum()),
        "legacy_partial_rows": int(old_partial.sum()),
        "new_score_rows": int(new_observed.sum()),
        "both_representation_rows": int(both.sum()),
        "representation_conflicts": int(conflict.sum()),
        "invalid_legacy_values": int(invalid_old.sum()),
        "invalid_new_values": int(invalid_new.sum()),
        "derived_nonmissing_rows": int(derived.notna().sum()),
    }
    if stats["representation_conflicts"] or stats["invalid_legacy_values"] or stats["invalid_new_values"]:
        raise A06Abort(
            "MMSE WORLD harmonization found invalid values or conflicting representations: "
            f"{stats}"
        )

    return output, stats


def harmonize_ptraccat_value(value: Any) -> tuple[str | None, bool, bool, list[str]]:
    tokens = [normalize_code_value(token) for token in split_multi_tokens(value)]
    tokens = [token for token in tokens if token is not None]
    if not tokens:
        return None, False, False, []

    allowed_raw = {str(i) for i in range(1, 10)}
    unexpected = sorted(set(tokens) - allowed_raw)
    if unexpected:
        return None, False, False, unexpected

    mapped = ["3" if token in {"8", "9"} else token for token in tokens]
    distinct = sorted(set(mapped), key=lambda x: int(x))
    collapsed_multi = len(distinct) > 1
    if collapsed_multi:
        result = "6"
    else:
        result = distinct[0]

    changed = normalize_code_value(value) != result or len(tokens) > 1 or any(token in {"8", "9"} for token in tokens)
    return result, changed, collapsed_multi, []


def harmonize_ptraccat(
    frame: pd.DataFrame,
    local_actions: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    output = frame.copy()
    if "PTRACCAT" not in output.columns:
        raise A06Abort("PTDEMOG core output is missing PTRACCAT.")

    transformed: list[Any] = []
    changed = 0
    collapsed_multi = 0
    unexpected_count = 0
    for index, value in output["PTRACCAT"].items():
        new_value, was_changed, was_multi, unexpected = harmonize_ptraccat_value(value)
        if unexpected:
            unexpected_count += 1
            local_actions.append(
                {
                    "source_name": "PTDEMOG",
                    "action": "ptraccat_unexpected_code",
                    "RID": output.at[index, RID],
                    "VISCODE2": pd.NA,
                    "column_name": "PTRACCAT",
                    "old_value": value,
                    "new_value": pd.NA,
                    "reason": f"unexpected raw token(s): {unexpected}",
                    "evidence_columns": "PTRACCAT",
                }
            )
            transformed.append(pd.NA)
            continue
        transformed.append(new_value if new_value is not None else pd.NA)
        if was_changed:
            changed += 1
            local_actions.append(
                {
                    "source_name": "PTDEMOG",
                    "action": "harmonize_PTRACCAT",
                    "RID": output.at[index, RID],
                    "VISCODE2": pd.NA,
                    "column_name": "PTRACCAT",
                    "old_value": value,
                    "new_value": new_value,
                    "reason": "cross-phase race ontology; ADNI4 8/9 -> common 3; multiple selections -> common 6",
                    "evidence_columns": "PTRACCAT",
                }
            )
        if was_multi:
            collapsed_multi += 1

    output["PTRACCAT"] = pd.Series(transformed, index=output.index, dtype="string")
    stats = {
        "changed_cells": changed,
        "multi_response_cells_collapsed_to_common_more_than_one_category": collapsed_multi,
        "unexpected_code_cells": unexpected_count,
    }
    if unexpected_count:
        raise A06Abort(f"PTRACCAT harmonization encountered {unexpected_count} unexpected-code cells.")
    return output, stats


# -----------------------------------------------------------------------------
# PTDEMOG baseline refinement
# -----------------------------------------------------------------------------


def rebuild_ptdemog_source_rows(package: Path, checks: Checks) -> pd.DataFrame:
    canonical_path = package / PTDEMOG_CANONICAL_RELATIVE
    excluded_path = package / PTDEMOG_EXCLUDED_RELATIVE
    summary_path = package / ROW_ACTION_SUMMARY_RELATIVE

    for label, path in [
        ("ptdemog_canonical", canonical_path),
        ("ptdemog_excluded", excluded_path),
        ("row_action_summary", summary_path),
    ]:
        checks.add(f"artifact__{label}", path.is_file(), True, path.is_file(), str(path))
    if not all(path.is_file() for path in [canonical_path, excluded_path, summary_path]):
        raise A06Abort("Required PTDEMOG reconstruction artifacts are missing.")

    canonical = read_csv_text(canonical_path)
    excluded = read_csv_text(excluded_path)
    processing_columns = [column for column in excluded.columns if column.startswith("PROCESSING_") or column.startswith("__")]
    excluded_source = excluded.drop(columns=processing_columns, errors="ignore")

    common_columns = list(canonical.columns)
    if "PROCESSING_RESOLUTION_STATUS" in common_columns:
        common_columns.remove("PROCESSING_RESOLUTION_STATUS")
    canonical_source = canonical.loc[:, common_columns].copy()

    # Keep the union of original source columns. Canonical is authoritative for order.
    union_columns = list(common_columns)
    for column in excluded_source.columns:
        if column not in union_columns:
            union_columns.append(column)
    canonical_source = canonical_source.reindex(columns=union_columns)
    excluded_source = excluded_source.reindex(columns=union_columns)

    reconstructed = pd.concat([canonical_source, excluded_source], ignore_index=True, sort=False)
    reconstructed[RID] = normalize_rid(reconstructed[RID], "PTDEMOG reconstructed source")

    row_summary = pd.read_csv(summary_path, dtype="string", keep_default_na=False)
    row = row_summary.loc[row_summary["source_name"].eq("PTDEMOG")]
    if row.empty:
        raise A06Abort("row_action_summary does not contain PTDEMOG.")
    expected_original = int(row.iloc[0]["original_rows"])
    checks.add(
        "ptdemog_reconstruction_row_count",
        len(reconstructed) == expected_original,
        expected_original,
        len(reconstructed),
    )
    return reconstructed


def ptdemog_resolution_columns(frame: pd.DataFrame) -> list[str]:
    wanted = [*PTDEMOG_CORE, "PTDOB", "PTDOBYY"]
    return [column for column in wanted if column in frame.columns]


def resolve_same_time_rows(
    group: pd.DataFrame,
    compare_columns: Sequence[str],
) -> tuple[pd.Series | None, str]:
    if len(group) == 1:
        return group.iloc[0], "unique_temporal_candidate"

    first = group.iloc[0]
    if all(rows_equivalent(first, group.iloc[pos], compare_columns) for pos in range(1, len(group))):
        hash_columns = sorted(group.columns)
        hashes = group.apply(lambda row: row_hash(row, hash_columns), axis=1)
        chosen_index = hashes.sort_values(kind="stable").index[0]
        return group.loc[chosen_index], "same_time_equivalent_duplicate"

    dominant_indices: list[Any] = []
    for candidate_index, candidate in group.iterrows():
        if all(
            candidate_index == other_index
            or rows_equivalent(candidate, other, compare_columns)
            or row_dominates(candidate, other, compare_columns)
            for other_index, other in group.iterrows()
        ):
            dominant_indices.append(candidate_index)
    if len(dominant_indices) == 1:
        return group.loc[dominant_indices[0]], "same_time_unique_dominant_record"
    return None, "same_time_unresolved_conflict"


def refine_ptdemog_baseline(
    reconstructed: pd.DataFrame,
    participant_split: pd.DataFrame,
    old_model_ready: pd.DataFrame,
    checks: Checks,
    local_baseline_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    require_columns(reconstructed, [RID, "VISCODE2"], "PTDEMOG reconstructed")
    require_columns(participant_split, [RID, SPLIT], "official participant split")

    official = set(participant_split[RID].astype("string"))
    source = reconstructed.loc[reconstructed[RID].isin(official)].copy()
    source["__protocol_month"] = source["VISCODE2"].map(parse_protocol_month)
    date_column = next((column for column in ("VISDATE", "EXAMDATE") if column in source.columns), None)
    if date_column is not None:
        source["__clinical_date"] = parse_dates(source[date_column])
    else:
        source["__clinical_date"] = pd.NaT

    compare_columns = ptdemog_resolution_columns(source)
    if not compare_columns:
        raise A06Abort("No PTDEMOG cross-check comparison columns are available.")

    selected_rows: list[pd.Series] = []
    unresolved_rids: set[str] = set()
    selected_reason_counts: Counter[str] = Counter()
    candidate_group_count = 0
    same_wave_ambiguous_count = 0
    date_refined_count = 0

    for rid_value, group in source.groupby(RID, sort=False, dropna=False):
        group = group.copy()
        candidate_group_count += 1
        months = pd.to_numeric(group["__protocol_month"], errors="coerce")
        temporal_basis = "none"

        if months.notna().any():
            earliest_month = float(months.min())
            candidate = group.loc[months.eq(earliest_month)].copy()
            temporal_basis = "earliest_protocol_wave"
        elif group["__clinical_date"].notna().any():
            earliest_date = group["__clinical_date"].min()
            candidate = group.loc[group["__clinical_date"].eq(earliest_date)].copy()
            temporal_basis = "earliest_available_clinical_date"
        else:
            candidate = group.copy()
            temporal_basis = "no_temporal_evidence"

        if len(candidate) > 1:
            same_wave_ambiguous_count += 1
            if candidate["__clinical_date"].notna().any():
                earliest_date = candidate["__clinical_date"].min()
                dated = candidate.loc[candidate["__clinical_date"].eq(earliest_date)].copy()
                if len(dated) < len(candidate):
                    date_refined_count += 1
                candidate = dated
                temporal_basis += "+earliest_date_within_wave"
            elif temporal_basis == "no_temporal_evidence":
                first = candidate.iloc[0]
                if not all(rows_equivalent(first, candidate.iloc[pos], compare_columns) for pos in range(1, len(candidate))):
                    chosen = None
                    resolution_reason = "unresolved_no_temporal_evidence"
                else:
                    chosen, resolution_reason = resolve_same_time_rows(candidate, compare_columns)
                if chosen is None:
                    unresolved_rids.add(str(rid_value))
                    for _, row in group.iterrows():
                        local_baseline_rows.append(
                            {
                                "RID": rid_value,
                                "PHASE": row.get("PHASE", pd.NA),
                                "VISCODE2": row.get("VISCODE2", pd.NA),
                                "VISDATE": row.get(date_column, pd.NA) if date_column else pd.NA,
                                "protocol_month": row.get("__protocol_month", pd.NA),
                                "action": "unresolved_baseline_group",
                                "reason": resolution_reason,
                                "selected": False,
                            }
                        )
                    continue

        chosen, resolution_reason = resolve_same_time_rows(candidate, compare_columns)
        if chosen is None:
            unresolved_rids.add(str(rid_value))
            for _, row in group.iterrows():
                local_baseline_rows.append(
                    {
                        "RID": rid_value,
                        "PHASE": row.get("PHASE", pd.NA),
                        "VISCODE2": row.get("VISCODE2", pd.NA),
                        "VISDATE": row.get(date_column, pd.NA) if date_column else pd.NA,
                        "protocol_month": row.get("__protocol_month", pd.NA),
                        "action": "unresolved_baseline_group",
                        "reason": resolution_reason,
                        "selected": False,
                    }
                )
            continue

        selected_reason_counts[resolution_reason] += 1
        selected_rows.append(chosen)
        selected_hash = row_hash(chosen, sorted([column for column in group.columns if not column.startswith("__")]))
        for _, row in group.iterrows():
            this_hash = row_hash(row, sorted([column for column in group.columns if not column.startswith("__")]))
            selected_flag = this_hash == selected_hash and rows_equivalent(row, chosen, compare_columns)
            local_baseline_rows.append(
                {
                    "RID": rid_value,
                    "PHASE": row.get("PHASE", pd.NA),
                    "VISCODE2": row.get("VISCODE2", pd.NA),
                    "VISDATE": row.get(date_column, pd.NA) if date_column else pd.NA,
                    "protocol_month": row.get("__protocol_month", pd.NA),
                    "action": "selected_baseline_row" if selected_flag else "discarded_nonbaseline_or_competing_row",
                    "reason": f"{temporal_basis};{resolution_reason}",
                    "selected": bool(selected_flag),
                }
            )

    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise A06Abort("PTDEMOG baseline refinement resolved zero participants.")

    selected = selected.drop(columns=["__protocol_month", "__clinical_date"], errors="ignore")
    selected[RID] = normalize_rid(selected[RID], "PTDEMOG selected baseline")
    checks.add("ptdemog_selected_unique_rid", not selected.duplicated([RID]).any(), True, not selected.duplicated([RID]).any())

    # Build public model-ready PTDEMOG strictly from the frozen common-core fields.
    missing_core = [column for column in PTDEMOG_CORE if column not in selected.columns]
    checks.add("ptdemog_core_fields_present", not missing_core, [], missing_core)
    if missing_core:
        raise A06Abort(f"PTDEMOG source is missing frozen core fields: {missing_core}")

    corrected = selected.loc[:, [RID, *PTDEMOG_CORE]].copy()
    corrected = corrected.merge(
        participant_split.loc[:, [RID, SPLIT]],
        on=RID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    corrected = corrected.loc[:, [RID, SPLIT, *PTDEMOG_CORE]]
    corrected = corrected.loc[corrected[list(PTDEMOG_CORE)].notna().any(axis=1)].copy()
    corrected = sort_source(corrected, "PTDEMOG")

    # Compare availability against old model-ready, without exporting identities.
    old_rids = set(old_model_ready[RID].astype("string"))
    new_rids = set(corrected[RID].astype("string"))
    newly_available = new_rids - old_rids
    no_longer_available = old_rids - new_rids

    # Prepare individualized LOCAL_ONLY birth-year metadata for later AGE_AT_TARGET.
    age_rows: list[dict[str, Any]] = []
    birth_year_conflicts = 0
    for _, row in selected.iterrows():
        years: list[int] = []
        source_fields: list[str] = []
        for column in ("PTDOBYY", "PTDOB"):
            if column not in selected.columns:
                continue
            value = row.get(column, pd.NA)
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            match = re.search(r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)", str(value))
            if match:
                years.append(int(match.group(1)))
                source_fields.append(column)
        distinct = sorted(set(years))
        if len(distinct) > 1:
            birth_year_conflicts += 1
            birth_year = pd.NA
        else:
            birth_year = distinct[0] if distinct else pd.NA
        age_rows.append(
            {
                "RID": row[RID],
                SPLIT: participant_split.set_index(RID).at[row[RID], SPLIT],
                "BIRTH_YEAR_FOR_AGE": birth_year,
                "BIRTH_YEAR_SOURCE_FIELDS": ";".join(source_fields),
                "BIRTH_YEAR_CONFLICT": len(distinct) > 1,
            }
        )
    age_metadata = pd.DataFrame(age_rows)

    stats = {
        "official_rids_with_any_ptdemog_source_row": int(source[RID].nunique()),
        "baseline_groups_evaluated": candidate_group_count,
        "groups_with_multiple_rows_in_earliest_wave_or_date": same_wave_ambiguous_count,
        "groups_refined_by_earliest_date_within_wave": date_refined_count,
        "resolved_baseline_participants": int(corrected[RID].nunique()),
        "unresolved_baseline_participants": len(unresolved_rids),
        "old_model_ready_participants": len(old_rids),
        "newly_available_participants_vs_old_model_ready": len(newly_available),
        "no_longer_available_participants_vs_old_model_ready": len(no_longer_available),
        "birth_year_metadata_available": int(age_metadata["BIRTH_YEAR_FOR_AGE"].notna().sum()),
        "birth_year_conflicts": birth_year_conflicts,
        "selected_reason_counts": dict(selected_reason_counts),
    }
    if birth_year_conflicts:
        raise A06Abort(f"PTDEMOG birth-year metadata contains {birth_year_conflicts} conflicting baseline years.")

    return corrected, age_metadata, stats


# -----------------------------------------------------------------------------
# Directives/catalog
# -----------------------------------------------------------------------------


def add_directive(
    rows: list[dict[str, Any]],
    source: str,
    column: str,
    action: str,
    final_type: str,
    transform_id: str,
    role: str,
    reason: str,
    execution_stage: str,
) -> None:
    rows.append(
        {
            "source_name": source,
            "column_name": column,
            "crosscheck_action": action,
            "final_feature_type": final_type,
            "value_transform_id": transform_id,
            "semantic_role": role,
            "reason": reason,
            "execution_stage": execution_stage,
        }
    )


def build_directives(
    original_columns: Mapping[str, Sequence[str]],
    corrected_frames: Mapping[str, pd.DataFrame],
    removed_by_source: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source_name, removed_columns in removed_by_source.items():
        for column in removed_columns:
            add_directive(
                rows, source_name, column, "exclude", "excluded", "none",
                "administration_protocol_or_semantically_excluded",
                "frozen_crosscheck_exclusion",
                "A06",
            )

    for column in ADAS_UNABLE_FIELDS:
        if column in corrected_frames["ADAS"].columns:
            add_directive(rows, "ADAS", column, "retain_harmonized", "categorical", "categorical_identity", "assessment_completion_state", "implicit_and_explicit_conducted_states_harmonized", "A06_then_BENCHMARK04C")
    for column in ADAS_ORDINAL_SCORES:
        if column in corrected_frames["ADAS"].columns:
            add_directive(rows, "ADAS", column, "retain_retype", "categorical", "categorical_identity", "ordinal_clinical_severity_score", "low_cardinality_ordinal_one_hot_policy", "BENCHMARK04C")

    for column in CDR_ORDINAL_CATEGORICAL:
        if column in corrected_frames["CDR"].columns:
            add_directive(rows, "CDR", column, "retain_retype", "categorical", "categorical_identity", "ordered_clinical_stage", "low_cardinality_ordinal_one_hot_policy", "BENCHMARK04C")
    if "CDRSB" in corrected_frames["CDR"].columns:
        add_directive(rows, "CDR", "CDRSB", "retain", "numeric", "numeric_identity", "official_sum_of_boxes_score", "retain_official_summary_score_as_numeric", "BENCHMARK04C")

    for column in FAQ_ITEMS:
        if column in corrected_frames["FAQ"].columns:
            add_directive(rows, "FAQ", column, "retain_transform_retype", "categorical", "faq_official_item_score_map", "officially_scored_functional_item", "apply_official_0_to_3_scoring_then_one_hot", "BENCHMARK04C/BENCHMARK05")
    if "FAQTOTAL" in corrected_frames["FAQ"].columns:
        add_directive(rows, "FAQ", "FAQTOTAL", "retain", "numeric", "numeric_identity", "official_summary_score", "retain_official_summary_score_policy", "BENCHMARK04C")

    if MMSE_CANONICAL_WORLD in corrected_frames["MMSE"].columns:
        add_directive(rows, "MMSE", MMSE_CANONICAL_WORLD, "derive_retain", "numeric", "numeric_identity", "harmonized_world_backwards_score", "canonical_0_to_5_cross_phase_representation", "A06")
    if "MMSCORE" in corrected_frames["MMSE"].columns:
        add_directive(rows, "MMSE", "MMSCORE", "retain", "numeric", "numeric_identity", "official_summary_score", "retain_official_summary_score_policy", "BENCHMARK04C")

    for column in MOCA_DELAYED_RECALL_FIELDS:
        if column in corrected_frames["MOCA"].columns:
            add_directive(rows, "MOCA", column, "retain", "categorical", "categorical_identity", "delayed_recall_response_or_cue_mode", "distinct_response_modes_not_linear_scale", "BENCHMARK04C")
    if "MOCA" in corrected_frames["MOCA"].columns:
        add_directive(rows, "MOCA", "MOCA", "retain", "numeric", "numeric_identity", "official_summary_score", "retain_official_summary_score_policy", "BENCHMARK04C")

    for column in NEUROBAT_RETAIN_SUMMARIES:
        if column in corrected_frames["NEUROBAT"].columns:
            add_directive(rows, "NEUROBAT", column, "retain", "numeric", "numeric_identity", "official_or_documented_summary_score", "retain_official_summary_score_policy", "BENCHMARK04C")

    ptdemog_types = {
        "PTGENDER": "categorical", "PTHAND": "categorical", "PTMARRY": "categorical",
        "PTEDUCAT": "numeric", "PTNOTRT": "categorical", "PTHOME": "categorical",
        "PTPLANG": "categorical", "PTETHCAT": "categorical", "PTRACCAT": "categorical",
    }
    for column, feature_type in ptdemog_types.items():
        if column in corrected_frames["PTDEMOG"].columns:
            transform = "ptdemog_pthome_harmonization" if column == "PTHOME" else "categorical_identity" if feature_type == "categorical" else "numeric_identity"
            add_directive(rows, "PTDEMOG", column, "retain", feature_type, transform, "cross_phase_common_core_demographic", "frozen_common_core_PTDEMOG_policy", "A06/BENCHMARK04C")
    add_directive(rows, "PTDEMOG", "AGE_AT_TARGET", "derive_later", "numeric", "derive_age_at_target_from_birth_year_and_current_target_date", "visit_specific_age_covariate", "raw_birth_fields_are_not_predictors; A06_prepares_local_birth_year_metadata", "BENCHMARK05")

    if "GENOTYPE" in corrected_frames["APOERES"].columns:
        add_directive(rows, "APOERES", "GENOTYPE", "retain", "categorical", "categorical_identity", "apoe_genotype_category", "six_category_domain_validated_fail_closed", "A06/BENCHMARK04C")

    directives = pd.DataFrame(rows)
    if directives.duplicated(["source_name", "column_name"]).any():
        duplicates = directives.loc[directives.duplicated(["source_name", "column_name"], keep=False), ["source_name", "column_name"]].drop_duplicates().to_dict("records")
        raise A06Abort(f"Cross-check directives contain duplicate source/column entries: {duplicates}")
    return directives.sort_values(["source_name", "column_name"], kind="stable").reset_index(drop=True)


def validate_directive_policy(
    directives: pd.DataFrame,
    corrected_frames: Mapping[str, pd.DataFrame],
    checks: Checks,
) -> dict[str, dict[str, Any]]:
    # Validate the frozen cross-cutting directive policy without inspecting row values.
    # This creates SAFE, structure-only evidence that the semantic types/actions emitted by
    # A06 match the frozen ledger. The returned object contains column names, expected
    # directives, and PASS/FAIL only; it never includes participant-level values.
    required_cols = {
        "source_name",
        "column_name",
        "crosscheck_action",
        "final_feature_type",
        "value_transform_id",
    }
    missing = sorted(required_cols - set(directives.columns))
    checks.add(
        "directive_policy__required_columns",
        not missing,
        [],
        missing,
        "SAFE structural validation of the crosscheck directive table.",
    )
    if missing:
        return {
            "required_columns": {
                "status": "FAIL",
                "expected": sorted(required_cols),
                "observed_missing": missing,
            }
        }

    def observed_rows(source: str, columns: Sequence[str]) -> list[dict[str, str]]:
        subset = directives.loc[
            directives["source_name"].eq(source)
            & directives["column_name"].isin(list(columns)),
            ["column_name", "crosscheck_action", "final_feature_type", "value_transform_id"],
        ].copy()
        if subset.empty:
            return []
        return subset.sort_values("column_name", kind="stable").astype("string").to_dict("records")

    results: dict[str, dict[str, Any]] = {}

    def add_group(
        key: str,
        source: str,
        columns: Sequence[str],
        expected_action: str,
        expected_type: str,
        expected_transform: str | None = None,
    ) -> None:
        cols = list(columns)
        subset = directives.loc[
            directives["source_name"].eq(source)
            & directives["column_name"].isin(cols)
        ].copy()
        present = set(subset["column_name"].astype(str)) if not subset.empty else set()
        exact_columns = present == set(cols) and len(subset) == len(cols)
        action_ok = exact_columns and subset["crosscheck_action"].astype(str).eq(expected_action).all()
        type_ok = exact_columns and subset["final_feature_type"].astype(str).eq(expected_type).all()
        transform_ok = True
        if expected_transform is not None:
            transform_ok = exact_columns and subset["value_transform_id"].astype(str).eq(expected_transform).all()
        passed = bool(exact_columns and action_ok and type_ok and transform_ok)
        expected = {
            "source": source,
            "columns": cols,
            "action": expected_action,
            "final_feature_type": expected_type,
        }
        if expected_transform is not None:
            expected["value_transform_id"] = expected_transform
        observed = observed_rows(source, cols)
        checks.add(
            f"directive_policy__{key}",
            passed,
            expected,
            observed,
            "Frozen crosscheck ledger directive policy.",
        )
        results[key] = {
            "status": "PASS" if passed else "FAIL",
            "expected": expected,
            "observed": observed,
        }

    add_group(
        "ADAS_Q9SCORE_Q12SCORE_categorical",
        "ADAS",
        ADAS_ORDINAL_SCORES,
        "retain_retype",
        "categorical",
        "categorical_identity",
    )
    add_group(
        "CDR_six_boxes_categorical",
        "CDR",
        ("CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE"),
        "retain_retype",
        "categorical",
        "categorical_identity",
    )
    add_group(
        "CDR_CDGLOBAL_categorical",
        "CDR",
        ("CDGLOBAL",),
        "retain_retype",
        "categorical",
        "categorical_identity",
    )
    add_group(
        "CDR_CDRSB_numeric",
        "CDR",
        ("CDRSB",),
        "retain",
        "numeric",
        "numeric_identity",
    )
    add_group(
        "FAQ_items_official_map_then_categorical",
        "FAQ",
        FAQ_ITEMS,
        "retain_transform_retype",
        "categorical",
        "faq_official_item_score_map",
    )
    add_group(
        "FAQTOTAL_retained_numeric",
        "FAQ",
        ("FAQTOTAL",),
        "retain",
        "numeric",
        "numeric_identity",
    )
    add_group(
        "MM_WORLD_SCORE_numeric",
        "MMSE",
        (MMSE_CANONICAL_WORLD,),
        "derive_retain",
        "numeric",
        "numeric_identity",
    )
    add_group(
        "MMSCORE_retained_numeric",
        "MMSE",
        ("MMSCORE",),
        "retain",
        "numeric",
        "numeric_identity",
    )
    add_group(
        "MOCA_DELW1_DELW5_categorical",
        "MOCA",
        MOCA_DELAYED_RECALL_FIELDS,
        "retain",
        "categorical",
        "categorical_identity",
    )
    add_group(
        "MOCA_total_retained_numeric",
        "MOCA",
        ("MOCA",),
        "retain",
        "numeric",
        "numeric_identity",
    )
    add_group(
        "NEUROBAT_summary_scores_retained_numeric",
        "NEUROBAT",
        NEUROBAT_RETAIN_SUMMARIES,
        "retain",
        "numeric",
        "numeric_identity",
    )

    # PTDEMOG common-core types are checked column by column because the core intentionally
    # mixes categorical and numeric features.
    ptdemog_expected = {
        "PTGENDER": "categorical",
        "PTHAND": "categorical",
        "PTMARRY": "categorical",
        "PTEDUCAT": "numeric",
        "PTNOTRT": "categorical",
        "PTHOME": "categorical",
        "PTPLANG": "categorical",
        "PTETHCAT": "categorical",
        "PTRACCAT": "categorical",
    }
    ptdemog_observed = observed_rows("PTDEMOG", tuple(ptdemog_expected))
    ptdemog_lookup = {row["column_name"]: row for row in ptdemog_observed}
    ptdemog_ok = True
    for column, final_type in ptdemog_expected.items():
        row = ptdemog_lookup.get(column)
        if row is None or row["crosscheck_action"] != "retain" or row["final_feature_type"] != final_type:
            ptdemog_ok = False
            break
    ptdemog_ok = ptdemog_ok and len(ptdemog_lookup) == len(ptdemog_expected)
    checks.add(
        "directive_policy__PTDEMOG_common_core_types",
        ptdemog_ok,
        ptdemog_expected,
        ptdemog_observed,
        "PTDEMOG frozen cross-phase common-core schema.",
    )
    results["PTDEMOG_common_core_types"] = {
        "status": "PASS" if ptdemog_ok else "FAIL",
        "expected": ptdemog_expected,
        "observed": ptdemog_observed,
    }

    add_group(
        "AGE_AT_TARGET_derive_later_numeric",
        "PTDEMOG",
        ("AGE_AT_TARGET",),
        "derive_later",
        "numeric",
        "derive_age_at_target_from_birth_year_and_current_target_date",
    )
    add_group(
        "APOERES_GENOTYPE_categorical",
        "APOERES",
        ("GENOTYPE",),
        "retain",
        "categorical",
        "categorical_identity",
    )

    # All physically removed feature columns must also be represented as frozen exclusions
    # in the directive table.  This is a schema-only check and is safe to share.
    removed_rows = directives.loc[directives["crosscheck_action"].eq("exclude")]
    removed_directives_ok = bool(
        not removed_rows.empty
        and removed_rows["final_feature_type"].astype(str).eq("excluded").all()
    )
    checks.add(
        "directive_policy__all_removed_columns_marked_excluded",
        removed_directives_ok,
        "all exclusion directives have final_feature_type=excluded",
        {
            "exclusion_directive_count": int(len(removed_rows)),
            "nonexcluded_final_type_count": int((~removed_rows["final_feature_type"].astype(str).eq("excluded")).sum()),
        },
    )
    results["all_removed_columns_marked_excluded"] = {
        "status": "PASS" if removed_directives_ok else "FAIL",
        "exclusion_directive_count": int(len(removed_rows)),
    }

    # Verify that no directive marked as a retained current feature points to a missing
    # corrected column. AGE_AT_TARGET is the one intentional derive-later exception.
    retained_current = directives.loc[
        directives["crosscheck_action"].isin(
            ["retain", "retain_harmonized", "retain_retype", "retain_transform_retype", "derive_retain"]
        )
    ]
    missing_current: list[str] = []
    for row in retained_current.itertuples(index=False):
        source = str(row.source_name)
        column = str(row.column_name)
        if source not in corrected_frames or column not in corrected_frames[source].columns:
            missing_current.append(f"{source}::{column}")
    checks.add(
        "directive_policy__retained_current_features_exist",
        not missing_current,
        [],
        sorted(missing_current),
    )
    results["retained_current_features_exist"] = {
        "status": "PASS" if not missing_current else "FAIL",
        "missing": sorted(missing_current),
    }

    return results


def build_feature_catalog(corrected_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name in INPUT_SOURCES:
        keys = set(SOURCE_KEYS[source_name]) | {SPLIT}
        features = [column for column in corrected_frames[source_name].columns if column not in keys]
        for position, column in enumerate(features, start=1):
            rows.append(
                {
                    "source_name": source_name,
                    "feature_position_within_source": position,
                    "column_name": column,
                    "role": "candidate_input_feature_crosschecked",
                    "final_train_only_removal_pending": "remove only if below frozen support threshold or otherwise excluded by frozen schema",
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Upstream validation
# -----------------------------------------------------------------------------


def load_and_validate_upstream(paths: Paths, checks: Checks) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    required = {
        "package": paths.package,
        "manifest": paths.package / MANIFEST_RELATIVE,
        "policy_lock": paths.package / POLICY_LOCK_RELATIVE,
        "participant_split": paths.package / PARTICIPANT_SPLIT_RELATIVE,
        "anchor": paths.package / ANCHOR_RELATIVE,
        "targets": paths.package / TARGETS_RELATIVE,
        "target_catalog": paths.package / TARGET_CATALOG_RELATIVE,
        "feature_catalog": paths.package / FEATURE_CATALOG_RELATIVE,
        "ledger": paths.ledger,
        "benchmark01_summary": paths.benchmark01_summary,
    }
    for label, path in required.items():
        exists = path.is_dir() if label == "package" else path.is_file()
        checks.add(f"artifact__{label}", exists, True, exists, str(path))
    if not all((path.is_dir() if label == "package" else path.is_file()) for label, path in required.items()):
        raise A06Abort("Required upstream artifacts are missing.")

    ledger_hash = sha256_file(paths.ledger)
    ledger = read_json(paths.ledger)
    declared_historical_hash = ledger.get("historical_ledger_sha256")
    historical_identity_ok = (
        ledger_hash == EXPECTED_LEDGER_SHA256
        or declared_historical_hash == EXPECTED_LEDGER_SHA256
    )
    checks.add(
        "ledger_historical_identity",
        historical_identity_ok,
        EXPECTED_LEDGER_SHA256,
        {
            "current_file_sha256": ledger_hash,
            "declared_historical_ledger_sha256": declared_historical_hash,
        },
        "A reconstructed ledger is allowed only when it explicitly preserves the historical ledger identity.",
    )
    checks.add("ledger_status", ledger.get("status") == "FROZEN", "FROZEN", ledger.get("status"))
    checks.add("ledger_version", ledger.get("ledger_version") == EXPECTED_LEDGER_VERSION, EXPECTED_LEDGER_VERSION, ledger.get("ledger_version"))

    manifest = read_json(paths.package / MANIFEST_RELATIVE)
    checks.add("package_identity", manifest.get("frozen_package_name") == EXPECTED_PACKAGE_NAME, EXPECTED_PACKAGE_NAME, manifest.get("frozen_package_name"))
    temporal = manifest.get("temporal_extension", {}) if isinstance(manifest.get("temporal_extension"), dict) else {}
    checks.add("manifest_temporal_policy_id", temporal.get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, temporal.get("temporal_policy_decision_id"))
    # These invariants are top-level fields in the v1.1 manifest generated by SPLIT07.
    # `temporal_extension` contains the temporal policy metadata itself, not the
    # unchanged-from-v1 flags.
    for key in ("participant_split_changed", "supervised_population_changed", "mri_targets_changed", "model_ready_sources_changed"):
        checks.add(f"manifest_{key}", manifest.get(key) is False, False, manifest.get(key))

    policy = read_json(paths.package / POLICY_LOCK_RELATIVE)
    checks.add("policy_lock_decision_id", policy.get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, policy.get("temporal_policy_decision_id"))

    b01 = read_json(paths.benchmark01_summary)
    checks.add("benchmark01_passed", b01.get("validation_passed") is True, True, b01.get("validation_passed"))
    b01_aggregate = b01.get("aggregate_summary", {})
    checks.add("benchmark01_package_identity", b01_aggregate.get("manifest", {}).get("logical_package_name") == EXPECTED_PACKAGE_NAME, EXPECTED_PACKAGE_NAME, b01_aggregate.get("manifest", {}).get("logical_package_name"))
    checks.add("benchmark01_temporal_policy_id", b01_aggregate.get("temporal_policy", {}).get("temporal_policy_decision_id") == EXPECTED_TEMPORAL_POLICY_DECISION_ID, EXPECTED_TEMPORAL_POLICY_DECISION_ID, b01_aggregate.get("temporal_policy", {}).get("temporal_policy_decision_id"))

    participant_split = read_csv_text(paths.package / PARTICIPANT_SPLIT_RELATIVE)
    participant_split[RID] = normalize_rid(participant_split[RID], "official participant split")
    participant_split[SPLIT] = participant_split[SPLIT].astype("string").str.strip().str.casefold()
    require_columns(participant_split, [RID, SPLIT], "official participant split")
    split_counts = participant_split[SPLIT].value_counts().to_dict()
    checks.add("participant_total", len(participant_split) == EXPECTED_PARTICIPANTS, EXPECTED_PARTICIPANTS, len(participant_split))
    checks.add("participant_unique_rid", not participant_split.duplicated([RID]).any(), True, not participant_split.duplicated([RID]).any())
    checks.add("participant_split_counts", split_counts == EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_COUNTS, split_counts)

    anchor = read_csv_text(paths.package / ANCHOR_RELATIVE, usecols=[RID, VISCODE2, SPLIT])
    anchor[RID] = normalize_rid(anchor[RID], "anchor")
    anchor[VISCODE2] = normalize_viscode(anchor[VISCODE2])
    checks.add("anchor_visit_total", len(anchor) == EXPECTED_VISITS, EXPECTED_VISITS, len(anchor))
    checks.add("anchor_unique_visit_key", not anchor.duplicated([RID, VISCODE2]).any(), True, not anchor.duplicated([RID, VISCODE2]).any())

    target_catalog = pd.read_csv(paths.package / TARGET_CATALOG_RELATIVE, dtype="string", keep_default_na=False)
    checks.add("target_catalog_count", len(target_catalog) == EXPECTED_TARGETS, EXPECTED_TARGETS, len(target_catalog))

    targets_header = pd.read_csv(paths.package / TARGETS_RELATIVE, compression="infer", nrows=0).columns.tolist()
    target_count = len([column for column in targets_header if column not in {RID, VISCODE2, SPLIT, "eligible_visit_position"}])
    checks.add("target_table_target_count", target_count == EXPECTED_TARGETS, EXPECTED_TARGETS, target_count)

    ledger_invariants = ledger.get("immutable_invariants", {})
    checks.add("ledger_support_threshold", int(ledger_invariants.get("support_threshold", -1)) == EXPECTED_SUPPORT_THRESHOLD, EXPECTED_SUPPORT_THRESHOLD, ledger_invariants.get("support_threshold"))

    if not checks.passed:
        raise A06Abort("Upstream frozen contract validation failed.")
    return ledger, manifest, policy, participant_split


def critical_upstream_hashes(package: Path) -> dict[str, str]:
    files = {
        "participant_split": package / PARTICIPANT_SPLIT_RELATIVE,
        "anchor": package / ANCHOR_RELATIVE,
        "targets": package / TARGETS_RELATIVE,
        "target_catalog": package / TARGET_CATALOG_RELATIVE,
        "temporal_policy_lock": package / POLICY_LOCK_RELATIVE,
    }
    for source_name in INPUT_SOURCES:
        files[f"input_{source_name}"] = package / INPUT_SOURCE_DIR_RELATIVE / f"{source_name}_model_ready.csv.gz"
    return {label: sha256_file(path) for label, path in files.items()}


# -----------------------------------------------------------------------------
# Safe/local report helpers
# -----------------------------------------------------------------------------


def safe_source_summary(
    source_name: str,
    before: pd.DataFrame,
    after: pd.DataFrame,
    removed_columns: Sequence[str],
    added_columns: Sequence[str],
    output_sha256: str,
) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "columns_before": int(len(before.columns)),
        "columns_after": int(len(after.columns)),
        "feature_columns_removed_count": int(len(removed_columns)),
        "feature_columns_removed": list(removed_columns),
        "feature_columns_added_count": int(len(added_columns)),
        "feature_columns_added": list(added_columns),
        "output_sha256": output_sha256,
    }


def sanitize_stats_for_safe(stats: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            output[key] = {str(k): suppress_count(int(v)) if isinstance(v, (int, np.integer)) else v for k, v in value.items()}
        elif isinstance(value, (int, np.integer)):
            output[key] = suppress_count(int(value))
        else:
            output[key] = value
    return output


def write_local_only_readme(paths: Paths) -> None:
    text = (
        "# A06 LOCAL_ONLY audit artifacts — DO NOT SHARE\n\n"
        "These files may contain individualized ADNI identifiers, visit codes, clinical dates, "
        "baseline-selection decisions, demographic/genetic/cognitive values, or before/after "
        "transformation traces.\n\n"
        "- LOCAL_ONLY = TRUE\n"
        "- CONTAINS_INDIVIDUALIZED_ADNI_RECORDS = TRUE\n"
        "- MUST_NOT_BE_SHARED = TRUE\n"
        "- SAFE_TO_SHARE = FALSE\n\n"
        "Do not upload, publish, version, email, or send these files to ChatGPT/LLMs. "
        "Use them only for local manual verification.\n"
    )
    (paths.local_only / LOCAL_README_NAME).write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Self-test with synthetic, non-ADNI data
# -----------------------------------------------------------------------------


def run_self_test() -> int:
    # MMSE canonical score
    mmse = pd.DataFrame({
        RID: ["1", "2", "3"], VISCODE2: ["bl", "bl", "bl"], SPLIT: ["train"] * 3,
        "MMD": ["1", pd.NA, "1"], "MML": ["1", pd.NA, "0"], "MMR": ["1", pd.NA, "1"],
        "MMO": ["0", pd.NA, "0"], "MMW": ["1", pd.NA, "1"], "WORLDSCORE": [pd.NA, "3", pd.NA],
    })
    actions: list[dict[str, Any]] = []
    transformed, stats = derive_mmse_world_score(mmse, actions)
    assert transformed[MMSE_CANONICAL_WORLD].tolist() == [4.0, 3.0, 3.0]
    assert stats["representation_conflicts"] == 0

    # PTRACCAT
    assert harmonize_ptraccat_value("8")[0] == "3"
    assert harmonize_ptraccat_value("1|5")[0] == "6"
    assert harmonize_ptraccat_value("5")[0] == "5"

    # UNABLE
    adas = pd.DataFrame({RID: ["1", "2"], VISCODE2: ["bl", "bl"], SPLIT: ["train", "train"], "Q1UNABLE": [pd.NA, "2"], "Q1SCORE": ["4", pd.NA]})
    out, st = harmonize_adas_unable(adas, [])
    assert out.loc[0, "Q1UNABLE"] == "0"
    assert out.loc[1, "Q1UNABLE"] == "2"
    assert st["Q1UNABLE"]["implicit_conducted_filled"] == 1

    # APOE domain constant itself
    assert len(APOE_ALLOWED_GENOTYPES) == 6
    print("A06 SELF-TEST PASSED")
    return 0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    paths = resolve_paths(args)
    configure_logging(paths)
    checks = Checks()
    fatal_error: str | None = None

    local_actions: list[dict[str, Any]] = []
    local_baseline_rows: list[dict[str, Any]] = []
    local_column_actions: list[dict[str, Any]] = []

    source_stats: dict[str, Any] = {}
    special_stats: dict[str, Any] = {}
    corrected_frames: dict[str, pd.DataFrame] = {}
    original_frames: dict[str, pd.DataFrame] = {}
    removed_by_source: dict[str, list[str]] = {source: [] for source in INPUT_SOURCES}
    added_by_source: dict[str, list[str]] = {source: [] for source in INPUT_SOURCES}
    output_hashes: dict[str, str] = {}
    manifest: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    ledger: dict[str, Any] = {}
    before_hashes: dict[str, str] = {}
    after_hashes: dict[str, str] = {}
    age_metadata = pd.DataFrame()
    directives = pd.DataFrame()
    directive_policy_checks: dict[str, dict[str, Any]] = {}
    feature_catalog = pd.DataFrame()

    try:
        ledger, manifest, policy, participant_split = load_and_validate_upstream(paths, checks)
        before_hashes = critical_upstream_hashes(paths.package)

        if paths.staging.exists():
            shutil.rmtree(paths.staging)
        if paths.output.exists() and not args.overwrite:
            raise A06Abort(
                f"Output already exists: {paths.output}. Use --overwrite only after reviewing the previous A06 report."
            )
        paths.staging.mkdir(parents=True, exist_ok=True)
        staging_inputs = paths.staging / "input_sources"
        staging_inputs.mkdir(parents=True, exist_ok=True)
        staging_local = paths.staging / "local_only"
        staging_local.mkdir(parents=True, exist_ok=True)

        # Load eight frozen model-ready sources and verify split/key integrity.
        split_lookup = participant_split.set_index(RID)[SPLIT].to_dict()
        for source_name in INPUT_SOURCES:
            input_path = paths.package / INPUT_SOURCE_DIR_RELATIVE / f"{source_name}_model_ready.csv.gz"
            checks.add(f"artifact__input__{source_name}", input_path.is_file(), True, input_path.is_file(), str(input_path))
            if not input_path.is_file():
                raise A06Abort(f"Missing model-ready source {source_name}: {input_path}")
            frame = read_csv_text(input_path)
            keys = list(SOURCE_KEYS[source_name])
            require_columns(frame, [*keys, SPLIT], f"{source_name} model-ready")
            frame[RID] = normalize_rid(frame[RID], source_name)
            if VISCODE2 in frame.columns:
                frame[VISCODE2] = normalize_viscode(frame[VISCODE2])
            frame[SPLIT] = frame[SPLIT].astype("string").str.strip().str.casefold()
            checks.add(f"source_unique_keys__{source_name}", not frame.duplicated(keys).any(), True, not frame.duplicated(keys).any())
            split_mismatch = frame.apply(lambda row: split_lookup.get(str(row[RID])) != str(row[SPLIT]), axis=1)
            checks.add(f"source_split_consistency__{source_name}", int(split_mismatch.sum()) == 0, 0, int(split_mismatch.sum()))
            original_frames[source_name] = frame

        if not checks.passed:
            raise A06Abort("Frozen input-source integrity checks failed.")

        # ADAS
        adas, adas_unable_stats = harmonize_adas_unable(original_frames["ADAS"], local_actions)
        adas_drop_regex = (
            re.compile(r"Q8WORD\d+"),
            re.compile(r"Q8WORD\d+R"),
        )
        adas, removed = drop_columns(adas, "ADAS", ADAS_EXCLUDE_EXACT, adas_drop_regex)
        removed_by_source["ADAS"] = removed
        corrected_frames["ADAS"] = sort_source(adas, "ADAS")
        special_stats["ADAS_UNABLE"] = adas_unable_stats

        # CDR
        cdr, removed = drop_columns(original_frames["CDR"].copy(), "CDR", CDR_EXCLUDE)
        removed_by_source["CDR"] = removed
        corrected_frames["CDR"] = sort_source(cdr, "CDR")

        # FAQ
        faq, removed = drop_columns(original_frames["FAQ"].copy(), "FAQ", FAQ_EXCLUDE)
        removed_by_source["FAQ"] = removed
        corrected_frames["FAQ"] = sort_source(faq, "FAQ")

        # MMSE
        mmse, world_stats = derive_mmse_world_score(original_frames["MMSE"], local_actions)
        mmse_drop = set(MMSE_EXCLUDE_EXACT) | set(MMSE_RAW_WORLD_LETTERS)
        mmse, removed = drop_columns(mmse, "MMSE", mmse_drop)
        removed_by_source["MMSE"] = removed
        added_by_source["MMSE"] = [MMSE_CANONICAL_WORLD]
        corrected_frames["MMSE"] = sort_source(mmse, "MMSE")
        special_stats["MMSE_WORLD"] = world_stats

        # MOCA
        moca, removed = drop_columns(original_frames["MOCA"].copy(), "MOCA", MOCA_EXCLUDE)
        removed_by_source["MOCA"] = removed
        corrected_frames["MOCA"] = sort_source(moca, "MOCA")

        # NEUROBAT
        neuro, removed = drop_columns(original_frames["NEUROBAT"].copy(), "NEUROBAT", NEUROBAT_EXCLUDE)
        removed_by_source["NEUROBAT"] = removed
        corrected_frames["NEUROBAT"] = sort_source(neuro, "NEUROBAT")

        # PTDEMOG: reconstruct original rows and re-resolve baseline conservatively.
        reconstructed_ptdemog = rebuild_ptdemog_source_rows(paths.package, checks)
        ptdemog, age_metadata, ptdemog_stats = refine_ptdemog_baseline(
            reconstructed_ptdemog,
            participant_split,
            original_frames["PTDEMOG"],
            checks,
            local_baseline_rows,
        )
        ptdemog, ptraccat_stats = harmonize_ptraccat(ptdemog, local_actions)
        corrected_frames["PTDEMOG"] = sort_source(ptdemog, "PTDEMOG")
        original_pt_features = set(original_frames["PTDEMOG"].columns) - {RID, SPLIT}
        new_pt_features = set(corrected_frames["PTDEMOG"].columns) - {RID, SPLIT}
        removed_by_source["PTDEMOG"] = sorted(original_pt_features - new_pt_features)
        special_stats["PTDEMOG_BASELINE"] = ptdemog_stats
        special_stats["PTRACCAT"] = ptraccat_stats
        special_stats["AGE_AT_TARGET_PREREQUISITES"] = {
            "participant_rows_in_age_metadata": int(len(age_metadata)),
            "participants_with_birth_year_candidate": int(age_metadata["BIRTH_YEAR_FOR_AGE"].notna().sum()),
            "birth_year_conflicts": int(age_metadata["BIRTH_YEAR_CONFLICT"].fillna(False).astype(bool).sum()),
            "age_derivation_execution_stage": "BENCHMARK05",
        }

        # APOERES: retain only GENOTYPE and validate domain fail-closed.
        apoe = original_frames["APOERES"].copy()
        require_columns(apoe, [RID, SPLIT, "GENOTYPE"], "APOERES")
        normalized_genotype = apoe["GENOTYPE"].astype("string").str.strip()
        unexpected_mask = normalized_genotype.notna() & ~normalized_genotype.isin(APOE_ALLOWED_GENOTYPES)
        unexpected_count = int(unexpected_mask.sum())
        for index in apoe.index[unexpected_mask]:
            local_actions.append(
                {
                    "source_name": "APOERES",
                    "action": "unexpected_genotype_domain",
                    "RID": apoe.at[index, RID],
                    "VISCODE2": pd.NA,
                    "column_name": "GENOTYPE",
                    "old_value": apoe.at[index, "GENOTYPE"],
                    "new_value": pd.NA,
                    "reason": "value outside frozen six-category APOE genotype domain",
                    "evidence_columns": "GENOTYPE",
                }
            )
        apoe["GENOTYPE"] = normalized_genotype
        apoe = apoe.loc[:, [RID, SPLIT, "GENOTYPE"]].copy()
        original_apoe_features = set(original_frames["APOERES"].columns) - {RID, SPLIT}
        removed_by_source["APOERES"] = sorted(original_apoe_features - {"GENOTYPE"})
        corrected_frames["APOERES"] = sort_source(apoe, "APOERES")
        special_stats["APOERES_DOMAIN"] = {
            "allowed_domain_size": len(APOE_ALLOWED_GENOTYPES),
            "unexpected_genotype_cells": unexpected_count,
        }
        checks.add("apoe_unexpected_genotype_domain", unexpected_count == 0, 0, unexpected_count)
        if unexpected_count:
            raise A06Abort("APOERES GENOTYPE contains unexpected values; see LOCAL_ONLY record audit.")

        # Global corrected-source validation before writing anything permanent.
        official_rids = set(participant_split[RID].astype("string"))
        for source_name, frame in corrected_frames.items():
            keys = list(SOURCE_KEYS[source_name])
            checks.add(f"corrected_unique_keys__{source_name}", not frame.duplicated(keys).any(), True, not frame.duplicated(keys).any())
            unofficial = ~frame[RID].astype("string").isin(official_rids)
            checks.add(f"corrected_official_participants__{source_name}", int(unofficial.sum()) == 0, 0, int(unofficial.sum()))
            expected_split = frame[RID].astype("string").map(split_lookup)
            mismatch = expected_split.astype("string").ne(frame[SPLIT].astype("string"))
            checks.add(f"corrected_split_consistency__{source_name}", int(mismatch.sum()) == 0, 0, int(mismatch.sum()))
            if source_name in VISIT_SOURCES:
                old_keys = set(map(tuple, original_frames[source_name][[RID, VISCODE2]].astype("string").itertuples(index=False, name=None)))
                new_keys = set(map(tuple, frame[[RID, VISCODE2]].astype("string").itertuples(index=False, name=None)))
                checks.add(f"visit_key_set_unchanged__{source_name}", new_keys == old_keys, len(old_keys), len(new_keys))
            if source_name == "APOERES":
                old_keys = set(original_frames[source_name][RID].astype("string"))
                new_keys = set(frame[RID].astype("string"))
                checks.add("apoeres_key_set_unchanged", new_keys == old_keys, len(old_keys), len(new_keys))

        # Explicit frozen-decision presence/absence checks.
        for source_name in INPUT_SOURCES:
            remaining = set(corrected_frames[source_name].columns)
            for column in removed_by_source[source_name]:
                checks.add(f"excluded_column_absent__{source_name}__{column}", column not in remaining, False, column in remaining)

        checks.add("mmse_world_score_present", MMSE_CANONICAL_WORLD in corrected_frames["MMSE"].columns, True, MMSE_CANONICAL_WORLD in corrected_frames["MMSE"].columns)
        checks.add("mmse_legacy_world_removed", all(column not in corrected_frames["MMSE"].columns for column in [*MMSE_OLD_WORLD_FIELDS, *MMSE_RAW_WORLD_LETTERS, "WORLDSCORE"]), True, [column for column in [*MMSE_OLD_WORLD_FIELDS, *MMSE_RAW_WORLD_LETTERS, "WORLDSCORE"] if column in corrected_frames["MMSE"].columns])
        checks.add("ptdemog_raw_birth_not_predictor", all(column not in corrected_frames["PTDEMOG"].columns for column in ["PTDOB", "PTDOBYY"]), True, [column for column in ["PTDOB", "PTDOBYY"] if column in corrected_frames["PTDEMOG"].columns])
        checks.add("ptdemog_only_frozen_core_features", set(corrected_frames["PTDEMOG"].columns) == {RID, SPLIT, *PTDEMOG_CORE}, {RID, SPLIT, *PTDEMOG_CORE}, set(corrected_frames["PTDEMOG"].columns))
        checks.add("age_metadata_birth_year_conflicts", int(age_metadata["BIRTH_YEAR_CONFLICT"].fillna(False).astype(bool).sum()) == 0, 0, int(age_metadata["BIRTH_YEAR_CONFLICT"].fillna(False).astype(bool).sum()))

        directives = build_directives(
            {source: list(frame.columns) for source, frame in original_frames.items()},
            corrected_frames,
            removed_by_source,
        )
        directive_policy_checks = validate_directive_policy(
            directives,
            corrected_frames,
            checks,
        )
        feature_catalog = build_feature_catalog(corrected_frames)

        # Record global column actions locally.
        for source_name in INPUT_SOURCES:
            for column in removed_by_source[source_name]:
                local_column_actions.append({
                    "source_name": source_name,
                    "column_name": column,
                    "action": "removed_from_crosschecked_input_table",
                    "reason": "frozen_crosscheck_ledger_v1.0.0",
                })
            for column in added_by_source[source_name]:
                local_column_actions.append({
                    "source_name": source_name,
                    "column_name": column,
                    "action": "added_to_crosschecked_input_table",
                    "reason": "frozen_crosscheck_ledger_v1.0.0",
                })

        if not checks.passed:
            raise A06Abort("Cross-checked source validation failed before commit.")

        # Write staging output deterministically.
        for source_name in INPUT_SOURCES:
            out_path = staging_inputs / f"{source_name}_model_ready.csv.gz"
            write_csv_gzip_deterministic(corrected_frames[source_name], out_path)
            output_hashes[source_name] = sha256_file(out_path)

        write_csv_deterministic(feature_catalog, paths.staging / "crosschecked_input_feature_catalog.csv")
        write_csv_deterministic(directives, paths.staging / "crosscheck_feature_directives.csv")
        write_csv_gzip_deterministic(age_metadata, staging_local / LOCAL_AGE_METADATA_NAME)

        output_manifest = {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "output_name": OUTPUT_NAME,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "upstream_frozen_package_name": EXPECTED_PACKAGE_NAME,
            "upstream_temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "crosscheck_ledger_version": EXPECTED_LEDGER_VERSION,
            "crosscheck_ledger_sha256": EXPECTED_LEDGER_SHA256,
            "reconstruction_ledger_sha256": sha256_file(paths.ledger),
            "frozen_invariants": {
                "participant_split_changed": False,
                "supervised_population_changed": False,
                "mri_targets_changed": False,
                "temporal_policy_changed": False,
                "support_threshold_changed": False,
                "support_threshold": EXPECTED_SUPPORT_THRESHOLD,
            },
            "age_at_target": {
                "status": "deferred_by_design_to_BENCHMARK05",
                "A06_output": "local_only birth-year derivation metadata only",
                "raw_birth_fields_exposed_as_predictors": False,
            },
            "input_source_sha256": output_hashes,
            "feature_catalog_sha256": sha256_file(paths.staging / "crosschecked_input_feature_catalog.csv"),
            "directives_sha256": sha256_file(paths.staging / "crosscheck_feature_directives.csv"),
            "privacy": {
                "crosschecked_input_tables_are_local_only": True,
                "local_only_age_metadata_must_not_be_shared": True,
            },
        }
        (paths.staging / "crosschecked_input_manifest.json").write_text(
            json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Verify frozen package byte-level invariants after all transformations.
        after_hashes = critical_upstream_hashes(paths.package)
        for label, before_hash in before_hashes.items():
            checks.add(f"upstream_hash_unchanged__{label}", after_hashes.get(label) == before_hash, before_hash, after_hashes.get(label))

        if not checks.passed:
            raise A06Abort("Upstream frozen package changed during A06 execution.")

        # Atomic commit only after all checks pass.
        if paths.output.exists():
            if not args.overwrite:
                raise A06Abort(f"Output already exists and --overwrite was not provided: {paths.output}")
            shutil.rmtree(paths.output)
        paths.staging.rename(paths.output)

        # Summaries after committed output.
        for source_name in INPUT_SOURCES:
            output_path = paths.output / "input_sources" / f"{source_name}_model_ready.csv.gz"
            source_stats[source_name] = safe_source_summary(
                source_name,
                original_frames[source_name],
                corrected_frames[source_name],
                removed_by_source[source_name],
                added_by_source[source_name],
                sha256_file(output_path),
            )

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("A06 aborted")
        if paths.staging.exists():
            shutil.rmtree(paths.staging, ignore_errors=True)

    # LOCAL_ONLY reports are written even on failure when partial traces exist.
    write_local_only_readme(paths)
    local_actions_frame = pd.DataFrame(local_actions)
    if local_actions_frame.empty:
        local_actions_frame = pd.DataFrame(columns=["source_name", "action", RID, VISCODE2, "column_name", "old_value", "new_value", "reason", "evidence_columns"])
    write_csv_gzip_deterministic(local_actions_frame, paths.local_only / LOCAL_RECORD_AUDIT_NAME)

    baseline_frame = pd.DataFrame(local_baseline_rows)
    if baseline_frame.empty:
        baseline_frame = pd.DataFrame(columns=[RID, "PHASE", VISCODE2, "VISDATE", "protocol_month", "action", "reason", "selected"])
    write_csv_gzip_deterministic(baseline_frame, paths.local_only / LOCAL_PTDEMOG_BASELINE_NAME)

    column_actions_frame = pd.DataFrame(local_column_actions)
    if column_actions_frame.empty:
        column_actions_frame = pd.DataFrame(columns=["source_name", "column_name", "action", "reason"])
    write_csv_deterministic(column_actions_frame, paths.local_only / LOCAL_COLUMN_ACTIONS_NAME)

    # SAFE verification outputs.
    checks_frame = pd.DataFrame(
        [
            {
                "check_name": item.name,
                "passed": item.passed,
                "expected": short_text(item.expected, 300),
                "observed": short_text(item.observed, 300),
                "details": item.details,
            }
            for item in checks.items
        ]
    )
    write_csv_deterministic(checks_frame, paths.safe / "a06_validation_checks.csv")

    passed = checks.passed and fatal_error is None and paths.output.is_dir()

    safe_report = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "fatal_error": fatal_error,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "upstream": {
            "logical_package_name": EXPECTED_PACKAGE_NAME,
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "benchmark01_validation_required": True,
            "crosscheck_ledger_version": EXPECTED_LEDGER_VERSION,
            "crosscheck_ledger_sha256": EXPECTED_LEDGER_SHA256,
            "reconstruction_ledger_sha256": sha256_file(paths.ledger),
        },
        "frozen_invariants": {
            "participant_mapping_changed": False if passed else None,
            "participant_count": EXPECTED_PARTICIPANTS,
            "participant_split_counts": EXPECTED_SPLIT_COUNTS,
            "supervised_population_changed": False if passed else None,
            "supervised_visit_count": EXPECTED_VISITS,
            "target_catalog_changed": False if passed else None,
            "target_values_changed": False if passed else None,
            "target_count": EXPECTED_TARGETS,
            "temporal_policy_changed": False if passed else None,
            "support_threshold_changed": False if passed else None,
            "support_threshold": EXPECTED_SUPPORT_THRESHOLD,
        },
        "sources": source_stats,
        "special_rule_stats": {key: sanitize_stats_for_safe(value) for key, value in special_stats.items()},
        "directives": {
            "count": int(len(directives)) if not directives.empty else 0,
            "sha256": sha256_file(paths.output / "crosscheck_feature_directives.csv") if passed else None,
        },
        "directive_policy_checks": directive_policy_checks,
        "feature_catalog": {
            "count": int(len(feature_catalog)) if not feature_catalog.empty else 0,
            "sha256": sha256_file(paths.output / "crosschecked_input_feature_catalog.csv") if passed else None,
        },
        "age_at_target": {
            "derived_in_A06": False,
            "execution_stage": "BENCHMARK05",
            "A06_prerequisite_metadata_local_only": True,
            "raw_birth_fields_exposed_as_predictors": False,
        },
        "privacy": {
            "contains_participant_identifier_values": False,
            "contains_visit_identifier_values": False,
            "contains_individual_clinical_dates": False,
            "contains_individual_clinical_or_cognitive_values": False,
            "contains_individual_genotype_values": False,
            "contains_predictions_or_errors": False,
            "contains_only_aggregate_counts_column_names_booleans_and_hashes": True,
            "SAFE_TO_SHARE": True,
            "local_only_artifacts_exist_and_must_not_be_shared": True,
        },
    }
    (paths.safe / "a06_verification_report.json").write_text(
        json.dumps(safe_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md: list[str] = [
        "# A06 cross-checked input verification",
        "",
        f"- Status: **{'PASS' if passed else 'FAIL'}**",
        f"- Script version: `{SCRIPT_VERSION}`",
        f"- Upstream package: `{EXPECTED_PACKAGE_NAME}`",
        f"- Temporal policy decision ID: `{EXPECTED_TEMPORAL_POLICY_DECISION_ID}`",
        f"- Frozen ledger: `v{EXPECTED_LEDGER_VERSION}`",
        f"- Checks: {len(checks.items)} total / {checks.failed} failed",
        "",
        "## Frozen invariants",
        "",
        f"- Participant split unchanged: **{'PASS' if passed else 'NOT CONFIRMED'}**",
        f"- Participant counts: `{EXPECTED_SPLIT_COUNTS}`",
        f"- Supervised population: `{EXPECTED_VISITS}` visits / `{EXPECTED_PARTICIPANTS}` participants",
        f"- MRI targets: `{EXPECTED_TARGETS}`",
        f"- Temporal policy unchanged: **{'PASS' if passed else 'NOT CONFIRMED'}**",
        f"- Support threshold remains `{EXPECTED_SUPPORT_THRESHOLD}`",
        "",
        "## Source-level changes",
        "",
    ]
    if source_stats:
        for source_name in INPUT_SOURCES:
            stats = source_stats[source_name]
            md.extend(
                [
                    f"### {source_name}",
                    f"- Rows: {stats['rows_before']} → {stats['rows_after']}",
                    f"- Columns: {stats['columns_before']} → {stats['columns_after']}",
                    f"- Removed feature columns: {stats['feature_columns_removed_count']}",
                    f"- Removed names: `{', '.join(stats['feature_columns_removed']) if stats['feature_columns_removed'] else 'none'}`",
                    f"- Added feature columns: `{', '.join(stats['feature_columns_added']) if stats['feature_columns_added'] else 'none'}`",
                    f"- Output SHA-256: `{stats['output_sha256']}`",
                    "",
                ]
            )
    else:
        md.append("No committed crosschecked source output was produced.")
        md.append("")

    md.extend(["## Special semantic checks", ""])
    for key, value in safe_report["special_rule_stats"].items():
        md.append(f"### {key}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                md.append(f"- {subkey}: `{subvalue}`")
        md.append("")

    md.extend(["## Frozen directive policy checks", ""])
    if directive_policy_checks:
        for key, value in directive_policy_checks.items():
            md.append(f"- {key}: **{value.get('status', 'UNKNOWN')}**")
    else:
        md.append("- No directive policy checks were produced.")
    md.append("")

    md.extend(
        [
            "## AGE_AT_TARGET",
            "",
            "A06 does **not** derive participant ages. It prepares and validates individualized birth-year metadata only in `local_only/`. The visit-specific `AGE_AT_TARGET` derivation remains frozen for BENCHMARK05, where the current target visit date is available. Raw birth fields are not predictors in the crosschecked PTDEMOG table.",
            "",
            "## Privacy",
            "",
            "This Markdown file is SAFE to share. It contains no participant identifier values, visit identifier values, individualized clinical dates, individualized clinical/cognitive/genetic values, predictions, or participant-level errors.",
            "",
            "The sibling `local_only/` directory is **NOT SAFE TO SHARE** and may contain individualized audit details.",
        ]
    )
    if fatal_error:
        md.extend(["", "## Fatal error", "", f"`{fatal_error}`"])

    (paths.safe / "a06_verification_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if passed:
        logging.info(
            "A06 PASSED | output=%s | package=%s | temporal_policy=%s | split_unchanged=true | targets_unchanged=true",
            paths.output,
            EXPECTED_PACKAGE_NAME,
            EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        )
        logging.info("SAFE report: %s", paths.safe / "a06_verification_report.json")
        logging.warning("LOCAL_ONLY audit artifacts contain individualized records and MUST NOT be shared.")
        return 0

    logging.error("A06 FAILED | failed_checks=%d | fatal_error=%s", checks.failed, fatal_error)
    logging.info("Diagnostic SAFE report: %s", paths.safe / "a06_verification_report.json")
    logging.warning("LOCAL_ONLY diagnostic artifacts MUST NOT be shared.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
