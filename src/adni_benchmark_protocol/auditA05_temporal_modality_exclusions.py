#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# AUDIT A05 — local-only temporal modality exclusion manifest.
#
# For the eligible longitudinal pairs, this stage compares each previous
# visit-level non-MRI source date against the current target MRI EXAMDATE.
#
# A previous source is marked unsafe when its clinical date is:
# - the same day as the current MRI; or
# - after the current MRI.
#
# The individualized exclusion manifest is written only to local_only and must
# never be uploaded, published, or versioned. Safe outputs contain only source
# names, boolean flags, suppressed counts, hashes, and totals.
#
# This script does not modify BENCHMARK05 scenarios.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd


STAGE_NAME: Final[str] = "auditA05_temporal_modality_exclusions"
SCRIPT_VERSION: Final[str] = "0.1.1"

RID: Final[str] = "RID"
VISCODE2: Final[str] = "VISCODE2"
SPLIT: Final[str] = "split"

PAIRING_RELATIVE: Final[str] = (
    "02_cuts_and_adjustments/development_strict_date_pairing_reference.csv.gz"
)
POSTPROCESSED_RELATIVE: Final[str] = "01_postprocessed_sources"

VISIT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
)
SOURCE_DATE_COLUMN: Final[dict[str, str]] = {
    "ADAS": "VISDATE",
    "CDR": "VISDATE",
    "FAQ": "VISDATE",
    "MMSE": "VISDATE",
    "MOCA": "VISDATE",
    "NEUROBAT": "VISDATE",
}
MRI_SOURCE: Final[str] = "UCSFFSX7"
MRI_DATE_COLUMN: Final[str] = "EXAMDATE"

EXPECTED_ELIGIBLE_PAIRS: Final[int] = 1727
SAFE_MINIMUM_CELL_COUNT: Final[int] = 5


class AuditAbort(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rid(series: pd.Series, context: str) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    invalid = text.notna() & text.ne("") & numeric.isna()
    non_integer = numeric.notna() & numeric.ne(numeric.round())
    if invalid.any() or non_integer.any():
        raise AuditAbort(
            f"Invalid RID in {context}: nonnumeric={int(invalid.sum())}, "
            f"noninteger={int(non_integer.sum())}"
        )
    return numeric.round().astype("Int64").astype("string")


def normalize_viscode(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.eq(""), pd.NA)
    try:
        return pd.to_datetime(text, errors="coerce", format="mixed").dt.normalize()
    except (TypeError, ValueError):
        return pd.to_datetime(text, errors="coerce").dt.normalize()


def read_source_dates(
    path: Path,
    source_name: str,
    date_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[RID, VISCODE2, date_column],
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )
    frame[RID] = normalize_rid(frame[RID], source_name)
    frame[VISCODE2] = normalize_viscode(frame[VISCODE2])
    frame["__date"] = parse_dates(frame[date_column])
    frame = frame.loc[frame["__date"].notna(), [RID, VISCODE2, "__date"]].copy()

    if frame.duplicated([RID, VISCODE2]).any():
        raise AuditAbort(
            f"{source_name} canonical source contains duplicate visit keys."
        )
    return frame


def suppress_count(value: int) -> int | None:
    if 0 < value < SAFE_MINIMUM_CELL_COUNT:
        return None
    return int(value)


def main() -> int:
    args = parse_args()

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
    reports = (
        args.reports_root.expanduser().resolve()
        if args.reports_root is not None
        else root / "reports"
    )

    stage = reports / STAGE_NAME
    safe = stage / "safe"
    logs = stage / "logs"
    local_only = stage / "local_only"
    safe.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    local_only.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logs / "auditA05.log", encoding="utf-8"),
        ],
        force=True,
    )

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, expected: Any, observed: Any) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "expected": str(expected),
                "observed": str(observed),
            }
        )
        logging.log(
            logging.INFO if passed else logging.ERROR,
            "%s | %s",
            "PASS" if passed else "FAIL",
            name,
        )

    fatal_error: str | None = None
    local_manifest_path = (
        local_only / "LOCAL_ONLY_temporal_modality_exclusions.csv"
    )
    local_rows: list[dict[str, Any]] = []
    safe_rows: list[dict[str, Any]] = []

    try:
        pairing_path = package / PAIRING_RELATIVE
        postprocessed = package / POSTPROCESSED_RELATIVE
        mri_path = postprocessed / f"{MRI_SOURCE}_canonical_resolved.csv.gz"

        add("artifact__pairing", pairing_path.is_file(), True, pairing_path.is_file())
        add("artifact__postprocessed", postprocessed.is_dir(), True, postprocessed.is_dir())
        add("artifact__mri", mri_path.is_file(), True, mri_path.is_file())

        for source_name in VISIT_SOURCES:
            source_path = (
                postprocessed / f"{source_name}_canonical_resolved.csv.gz"
            )
            add(
                f"artifact__{source_name}",
                source_path.is_file(),
                True,
                source_path.is_file(),
            )

        if not all(row["passed"] for row in checks):
            raise AuditAbort("Required artifacts are missing.")

        pairing = pd.read_csv(
            pairing_path,
            usecols=[
                RID,
                SPLIT,
                "current_VISCODE2",
                "previous_VISCODE2",
                "pairing_status",
            ],
            dtype="string",
            keep_default_na=False,
            low_memory=False,
        )
        pairing[RID] = normalize_rid(pairing[RID], "pairing")
        pairing[SPLIT] = pairing[SPLIT].astype("string").str.strip().str.casefold()
        pairing["current_VISCODE2"] = normalize_viscode(
            pairing["current_VISCODE2"]
        )
        pairing["previous_VISCODE2"] = normalize_viscode(
            pairing["previous_VISCODE2"]
        )

        eligible = pairing.loc[
            pairing["pairing_status"].eq("eligible")
        ].copy()

        add(
            "eligible_pair_count",
            len(eligible) == EXPECTED_ELIGIBLE_PAIRS,
            EXPECTED_ELIGIBLE_PAIRS,
            len(eligible),
        )
        add(
            "eligible_participant_unique",
            not eligible[RID].duplicated().any(),
            True,
            not eligible[RID].duplicated().any(),
        )

        mri = read_source_dates(
            mri_path,
            MRI_SOURCE,
            MRI_DATE_COLUMN,
        ).rename(
            columns={
                VISCODE2: "current_VISCODE2",
                "__date": "__current_mri_date",
            }
        )

        work = eligible.merge(
            mri,
            on=[RID, "current_VISCODE2"],
            how="left",
            validate="many_to_one",
        )
        missing_current_mri_date = int(
            work["__current_mri_date"].isna().sum()
        )
        add(
            "all_eligible_current_mri_dates_available",
            missing_current_mri_date == 0,
            0,
            missing_current_mri_date,
        )

        per_source_counts: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {
                "same_day": 0,
                "after_current_mri": 0,
                "unsafe_total": 0,
            }
        )

        for source_name in VISIT_SOURCES:
            source_path = (
                postprocessed / f"{source_name}_canonical_resolved.csv.gz"
            )
            source = read_source_dates(
                source_path,
                source_name,
                SOURCE_DATE_COLUMN[source_name],
            ).rename(
                columns={
                    VISCODE2: "previous_VISCODE2",
                    "__date": "__previous_source_date",
                }
            )

            merged = work.merge(
                source,
                on=[RID, "previous_VISCODE2"],
                how="left",
                validate="many_to_one",
            )

            dated = (
                merged["__previous_source_date"].notna()
                & merged["__current_mri_date"].notna()
            )
            same_day = dated & merged["__previous_source_date"].eq(
                merged["__current_mri_date"]
            )
            after = dated & merged["__previous_source_date"].gt(
                merged["__current_mri_date"]
            )
            unsafe = same_day | after

            per_source_counts[source_name]["same_day"] = int(same_day.sum())
            per_source_counts[source_name]["after_current_mri"] = int(after.sum())
            per_source_counts[source_name]["unsafe_total"] = int(unsafe.sum())

            unsafe_records = merged.loc[
                unsafe,
                [
                    RID,
                    SPLIT,
                    "current_VISCODE2",
                    "previous_VISCODE2",
                    "__previous_source_date",
                    "__current_mri_date",
                ],
            ].to_dict(orient="records")

            for record in unsafe_records:
                relation = (
                    "same_day_as_current_mri"
                    if record["__previous_source_date"]
                    == record["__current_mri_date"]
                    else "after_current_mri"
                )
                local_rows.append(
                    {
                        RID: str(record[RID]),
                        SPLIT: str(record[SPLIT]),
                        "current_VISCODE2": str(
                            record["current_VISCODE2"]
                        ),
                        "previous_VISCODE2": str(
                            record["previous_VISCODE2"]
                        ),
                        "source_name": source_name,
                        "temporal_relation": relation,
                        "action": "exclude_previous_source_block",
                    }
                )

        local_manifest = pd.DataFrame(
            local_rows,
            columns=[
                RID,
                SPLIT,
                "current_VISCODE2",
                "previous_VISCODE2",
                "source_name",
                "temporal_relation",
                "action",
            ],
        )
        if not local_manifest.empty:
            local_manifest = local_manifest.sort_values(
                [RID, "source_name"]
            ).reset_index(drop=True)
        local_manifest.to_csv(
            local_manifest_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        duplicate_actions = int(
            local_manifest.duplicated([RID, "source_name"]).sum()
            if not local_manifest.empty
            else 0
        )
        add(
            "local_manifest_unique_participant_source",
            duplicate_actions == 0,
            0,
            duplicate_actions,
        )

        for source_name in VISIT_SOURCES:
            counts = per_source_counts[source_name]
            safe_rows.append(
                {
                    "source_name": source_name,
                    "unsafe_previous_block_present": (
                        counts["unsafe_total"] > 0
                    ),
                    "same_day_present": counts["same_day"] > 0,
                    "after_current_mri_present": (
                        counts["after_current_mri"] > 0
                    ),
                    "unsafe_pair_count": suppress_count(
                        counts["unsafe_total"]
                    ),
                    "unsafe_pair_count_suppressed": (
                        0 < counts["unsafe_total"] < SAFE_MINIMUM_CELL_COUNT
                    ),
                }
            )

        total_unsafe_actions = len(local_manifest)
        affected_participants = (
            int(local_manifest[RID].nunique())
            if not local_manifest.empty
            else 0
        )
        add(
            "unsafe_actions_identified",
            total_unsafe_actions > 0,
            True,
            total_unsafe_actions > 0,
        )

        safe_frame = pd.DataFrame(safe_rows)
        safe_frame.to_csv(
            safe / "auditA05_source_exclusion_summary.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        local_sha = sha256_file(local_manifest_path)
        summary = {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_passed": all(bool(row["passed"]) for row in checks),
            "check_count": len(checks),
            "failed_check_count": sum(
                not bool(row["passed"]) for row in checks
            ),
            "fatal_error": None,
            "policy": {
                "comparison": (
                    "previous source clinical date versus current target MRI "
                    "EXAMDATE"
                ),
                "unsafe_relations": [
                    "same_day_as_current_mri",
                    "after_current_mri",
                ],
                "recommended_action": (
                    "exclude only the affected previous source block"
                ),
            },
            "aggregate": {
                "unsafe_source_pair_actions_present": (
                    total_unsafe_actions > 0
                ),
                "unsafe_source_pair_action_count": suppress_count(
                    total_unsafe_actions
                ),
                "unsafe_source_pair_action_count_suppressed": (
                    0 < total_unsafe_actions < SAFE_MINIMUM_CELL_COUNT
                ),
                "affected_participants_present": affected_participants > 0,
                "affected_participant_count": suppress_count(
                    affected_participants
                ),
                "affected_participant_count_suppressed": (
                    0 < affected_participants < SAFE_MINIMUM_CELL_COUNT
                ),
            },
            "local_only_manifest": {
                "relative_path": str(
                    local_manifest_path.relative_to(root)
                ).replace("\\", "/"),
                "sha256": local_sha,
                "contains_individualized_keys": True,
                "must_not_be_uploaded_or_versioned": True,
            },
            "privacy": {
                "safe_outputs_contain_participant_identifiers": False,
                "safe_outputs_contain_visit_identifiers": False,
                "safe_outputs_contain_dates": False,
                "safe_outputs_contain_medical_values": False,
                "local_only_manifest_contains_individualized_keys": True,
                "small_cell_suppression_threshold": (
                    SAFE_MINIMUM_CELL_COUNT
                ),
            },
        }
        (
            safe / "auditA05_summary.json"
        ).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("AUDIT A05 aborted")

        failure_summary = {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit_passed": False,
            "check_count": len(checks),
            "failed_check_count": sum(
                not bool(row["passed"]) for row in checks
            ),
            "fatal_error": fatal_error,
            "privacy": {
                "safe_outputs_contain_participant_identifiers": False,
                "safe_outputs_contain_visit_identifiers": False,
                "safe_outputs_contain_dates": False,
                "safe_outputs_contain_medical_values": False,
            },
        }
        (
            safe / "auditA05_summary.json"
        ).write_text(
            json.dumps(
                failure_summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    pd.DataFrame(checks).to_csv(
        safe / "auditA05_validation_checks.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    passed = (
        fatal_error is None
        and bool(checks)
        and all(bool(row["passed"]) for row in checks)
    )

    if passed:
        logging.info("AUDIT A05 PASSED")
        logging.info(
            "LOCAL ONLY manifest written: %s",
            local_manifest_path,
        )
        return 0

    logging.error(
        "AUDIT A05 FAILED | failed_checks=%d | fatal_error=%s",
        sum(not bool(row["passed"]) for row in checks),
        fatal_error,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
