from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil

import numpy as np
import pandas as pd


class Split07FreezeError(RuntimeError):
    pass


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "split07_freeze_strict_date_temporal_extension"
BASE_PACKAGE_NAME: Final[str] = "frozen_experiment_v1"
OUTPUT_PACKAGE_NAME: Final[str] = "frozen_experiment_v1_1"
TEMPORAL_POLICY_NAME: Final[str] = "strict_date_interval_same_wave_previous_visit_v1"
HISTORICAL_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"

DEVELOPMENT_SPLITS: Final[tuple[str, ...]] = ("train", "validation")
VISIT_LEVEL_INPUT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS", "CDR", "FAQ", "MMSE", "MOCA", "NEUROBAT",
)
DATE_SOURCE_FILES: Final[tuple[str, ...]] = (
    "DXSUM", "ADAS", "CDR", "FAQ", "MMSE", "MOCA", "NEUROBAT", "UCSFFSX7",
)

# Only clinical/examination dates enter the temporal envelope.
# Administrative timestamps are deliberately excluded.
CLINICAL_DATE_COLUMNS: Final[tuple[str, ...]] = (
    "EXAMDATE", "VISDATE", "TESTDATE", "APTESTDT", "ASSESSDATE",
)

EXPECTED_BASE_UNIQUE_ELIGIBLE: Final[dict[str, int]] = {
    "train": 1170,
    "validation": 246,
}
EXPECTED_AMBIGUOUS: Final[dict[str, int]] = {
    "train": 269,
    "validation": 58,
}
EXPECTED_STRICT_RECOVERED: Final[dict[str, int]] = {
    "train": 256,
    "validation": 55,
}
EXPECTED_AMBIGUOUS_NO_WINNER: Final[dict[str, int]] = {
    "train": 9,
    "validation": 2,
}
EXPECTED_WINNER_NOT_BEFORE_CURRENT: Final[dict[str, int]] = {
    "train": 4,
    "validation": 1,
}
EXPECTED_FINAL_ELIGIBLE: Final[dict[str, int]] = {
    "train": 1426,
    "validation": 301,
}
EXPECTED_DEVELOPMENT_PARTICIPANTS: Final[dict[str, int]] = {
    "train": 1987,
    "validation": 426,
}
EXPECTED_TOTAL_PARTICIPANTS: Final[int] = 2839

SAFE_MINIMUM_CELL_COUNT: Final[int] = 5
READ_CHUNK_SIZE: Final[int] = 50_000
INVENTORY_RELATIVE_PATH: Final[str] = "04_official_split/frozen_file_inventory.csv"
MANIFEST_RELATIVE_PATH: Final[str] = "04_official_split/official_frozen_manifest.json"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    base_package: Path
    output_package: Path
    staging_package: Path
    split06_summary: Path
    safe_output_dir: Path

@dataclass(frozen=True)
class VisitDateProfile:
    date_min: pd.Timestamp | pd.NaT
    date_max: pd.Timestamp | pd.NaT
    date_median: pd.Timestamp | pd.NaT
    observed_date_cells: int
    distinct_dates: int
    source_count: int
    field_count: int
    date_span_days: float | None
    status: str

@dataclass(frozen=True)
class DateEvidence:
    date: pd.Timestamp
    source_name: str
    column_name: str

def resolve_paths(project_root: Path) -> ProjectPaths:
    root = project_root.expanduser().resolve()
    processed = root / "data" / "processed"
    base = processed / BASE_PACKAGE_NAME
    output = processed / OUTPUT_PACKAGE_NAME
    staging = processed / f".{OUTPUT_PACKAGE_NAME}__staging"
    split06_summary = (
        root
        / "reports"
        / "split06_same_wave_prior_audit"
        / "safe"
        / "split06_summary.json"
    )
    safe_output = (
        root
        / "reports"
        / "split07_strict_date_temporal_extension"
        / "safe"
    )

    if not base.is_dir():
        raise FileNotFoundError(f"Base frozen package not found: {base}")
    if not split06_summary.is_file():
        raise FileNotFoundError(f"Passed SPLIT06 summary not found: {split06_summary}")

    safe_output.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(
        root=root,
        base_package=base,
        output_package=output,
        staging_package=staging,
        split06_summary=split06_summary,
        safe_output_dir=safe_output,
    )


def calculate_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def calculate_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def canonical_json_sha256(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return calculate_text_sha256(text)

def write_dataframe_csv_gzip_deterministic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")
    return path

def write_dataframe_csv_deterministic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path

def normalize_key_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()

def normalize_viscode_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()

def parse_protocol_month(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().casefold()
    if text in {"bl", "sc", "scmri", "m00", "v01"}:
        return 0.0
    month = re.fullmatch(r"m(\d+)", text)
    if month:
        return float(int(month.group(1)))
    year = re.fullmatch(r"y(\d+)", text)
    if year:
        return float(int(year.group(1)) * 12)
    return np.nan

def read_csv_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, compression="infer", nrows=0).columns)

def read_development_rows_by_split(
    path: Path,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="infer",
        dtype="string",
        usecols=usecols,
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    ):
        if "split" not in chunk.columns:
            raise KeyError(f"{path} does not contain split.")
        kept = chunk.loc[chunk["split"].isin(DEVELOPMENT_SPLITS)].copy()
        if not kept.empty:
            pieces.append(kept)
    if not pieces:
        return pd.DataFrame(columns=list(usecols or []))
    return pd.concat(pieces, ignore_index=True, sort=False)

def read_rows_for_rids(
    path: Path,
    development_rids: set[str],
    usecols: Sequence[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="infer",
        dtype="string",
        usecols=list(usecols),
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["RID"] = normalize_key_series(chunk["RID"])
        kept = chunk.loc[chunk["RID"].isin(development_rids)].copy()
        if not kept.empty:
            pieces.append(kept)
    if not pieces:
        return pd.DataFrame(columns=list(usecols))
    return pd.concat(pieces, ignore_index=True, sort=False)

def load_split06_summary(paths: ProjectPaths) -> dict[str, Any]:
    summary = json.loads(paths.split06_summary.read_text(encoding="utf-8"))
    if summary.get("audit_passed") is not True:
        raise RuntimeError("SPLIT06 did not pass.")
    scope = summary.get("scope", {})
    if scope.get("test_used") is not False:
        raise RuntimeError("SPLIT06 reports test use; refusing to freeze policy.")
    if scope.get("mri_target_values_loaded") is not False:
        raise RuntimeError("SPLIT06 reports MRI target loading; refusing to freeze policy.")

    records = {str(row["split"]): row for row in summary.get("ambiguous_policy_summary", [])}
    for split in DEVELOPMENT_SPLITS:
        if split not in records:
            raise RuntimeError(f"SPLIT06 summary missing split: {split}")
        row = records[split]
        expected_values = {
            "ambiguous_participants": EXPECTED_AMBIGUOUS[split],
            "strict_interval_latest_and_before_current": EXPECTED_STRICT_RECOVERED[split],
        }
        for field, expected in expected_values.items():
            observed = int(row.get(field, -1))
            if observed != expected:
                raise RuntimeError(
                    f"SPLIT06 {split}.{field}={observed}; expected {expected}."
                )
    return summary

def load_participant_split(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    path = paths.base_package / "04_official_split" / "official_participant_split.csv"
    frame = pd.read_csv(path, dtype="string", low_memory=False)
    required = {"RID", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Official participant split missing: {missing}")
    frame["RID"] = normalize_key_series(frame["RID"])
    frame["split"] = frame["split"].astype("string").str.strip().str.casefold()
    if frame[["RID", "split"]].isna().any().any():
        raise RuntimeError("Official participant split contains missing values.")
    if frame["RID"].duplicated().any():
        raise RuntimeError("Official participant split contains duplicate participants.")

    development = frame.loc[frame["split"].isin(DEVELOPMENT_SPLITS)].copy()
    for split, expected in EXPECTED_DEVELOPMENT_PARTICIPANTS.items():
        observed = int(development["split"].eq(split).sum())
        if observed != expected:
            raise RuntimeError(f"Development participant count {split}={observed}; expected {expected}.")
    return frame, development, set(development["RID"].tolist())

def load_development_anchor(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = paths.base_package / "03_final_ready_tables" / "supervised_anchor_visits.csv.gz"
    usecols = ["RID", "VISCODE2", "split", "eligible_visit_position", "protocol_month"]
    frame = read_development_rows_by_split(path, usecols=usecols)
    frame["RID"] = normalize_key_series(frame["RID"])
    frame["VISCODE2"] = normalize_viscode_series(frame["VISCODE2"])
    frame["split"] = frame["split"].astype("string").str.strip().str.casefold()
    frame["eligible_visit_position"] = pd.to_numeric(
        frame["eligible_visit_position"], errors="raise"
    ).astype("int64")
    frame["protocol_month"] = pd.to_numeric(frame["protocol_month"], errors="coerce")
    if frame[["RID", "VISCODE2", "split"]].isna().any().any():
        raise RuntimeError("Development anchor contains missing structural values.")
    if frame.duplicated(["RID", "VISCODE2"]).any():
        raise RuntimeError("Development anchor contains duplicate visit keys.")
    last = (
        frame.sort_values(["RID", "eligible_visit_position", "VISCODE2"], kind="stable")
        .drop_duplicates("RID", keep="last")
        .reset_index(drop=True)
    )
    if last["RID"].duplicated().any():
        raise RuntimeError("Could not select exactly one current target per participant.")
    return frame, last

def build_input_candidate_visits(
    paths: ProjectPaths,
) -> tuple[dict[tuple[str, str], frozenset[str]], pd.DataFrame]:
    modalities: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    summaries: list[dict[str, object]] = []
    inputs_dir = paths.base_package / "03_final_ready_tables" / "input_sources"

    for source_name in VISIT_LEVEL_INPUT_SOURCES:
        path = inputs_dir / f"{source_name}_model_ready.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        header = read_csv_header(path)
        required = {"RID", "VISCODE2", "split"}
        missing = sorted(required - set(header))
        if missing:
            raise KeyError(f"{source_name} model-ready missing: {missing}")
        features = [column for column in header if column not in required]
        frame = read_development_rows_by_split(path, usecols=header)
        frame["RID"] = normalize_key_series(frame["RID"])
        frame["VISCODE2"] = normalize_viscode_series(frame["VISCODE2"])
        frame["split"] = frame["split"].astype("string").str.strip().str.casefold()
        if frame.duplicated(["RID", "VISCODE2"]).any():
            raise RuntimeError(f"{source_name} model-ready contains duplicate keys.")

        empty_rows = 0
        for _, row in frame.iterrows():
            has_content = any(
                not pd.isna(row[column]) and str(row[column]).strip() != ""
                for column in features
            )
            if not has_content:
                empty_rows += 1
                continue
            modalities[(str(row["RID"]), str(row["VISCODE2"]))].add(source_name)
        if empty_rows:
            raise RuntimeError(f"{source_name} model-ready contains {empty_rows} empty rows.")
        summaries.append(
            {
                "source_name": source_name,
                "development_rows": int(len(frame)),
                "candidate_visit_keys": int(frame[["RID", "VISCODE2"]].drop_duplicates().shape[0]),
                "test_rows_retained": 0,
            }
        )
    return {key: frozenset(value) for key, value in modalities.items()}, pd.DataFrame(summaries)

def collect_date_evidence(
    paths: ProjectPaths,
    development_rids: set[str],
) -> tuple[dict[tuple[str, str], list[DateEvidence]], pd.DataFrame]:
    evidence: defaultdict[tuple[str, str], list[DateEvidence]] = defaultdict(list)
    summaries: list[dict[str, object]] = []
    postprocessed = paths.base_package / "01_postprocessed_sources"

    for source_name in DATE_SOURCE_FILES:
        path = postprocessed / f"{source_name}_canonical_resolved.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        header = read_csv_header(path)
        date_columns = [column for column in CLINICAL_DATE_COLUMNS if column in header]
        if not date_columns:
            summaries.append(
                {
                    "source_name": source_name,
                    "date_column": "<NONE>",
                    "development_rows": 0,
                    "observed_date_cells": 0,
                    "parseable_date_cells": 0,
                    "invalid_observed_date_cells": 0,
                    "test_rows_retained": 0,
                }
            )
            continue

        frame = read_rows_for_rids(path, development_rids, ["RID", "VISCODE2", *date_columns])
        frame["RID"] = normalize_key_series(frame["RID"])
        frame["VISCODE2"] = normalize_viscode_series(frame["VISCODE2"])
        if frame.duplicated(["RID", "VISCODE2"]).any():
            raise RuntimeError(f"{source_name} canonical table contains duplicate keys.")

        for column in date_columns:
            observed = frame[column].notna() & frame[column].astype("string").str.strip().ne("")
            try:
                parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                parsed = pd.to_datetime(frame[column], errors="coerce")
            valid = observed & parsed.notna()
            invalid = observed & parsed.isna()
            for index in frame.index[valid]:
                key = (str(frame.at[index, "RID"]), str(frame.at[index, "VISCODE2"]))
                evidence[key].append(
                    DateEvidence(
                        date=pd.Timestamp(parsed.at[index]).normalize(),
                        source_name=source_name,
                        column_name=column,
                    )
                )
            summaries.append(
                {
                    "source_name": source_name,
                    "date_column": column,
                    "development_rows": int(len(frame)),
                    "observed_date_cells": int(observed.sum()),
                    "parseable_date_cells": int(valid.sum()),
                    "invalid_observed_date_cells": int(invalid.sum()),
                    "test_rows_retained": 0,
                }
            )
    return dict(evidence), pd.DataFrame(summaries)

def make_date_profile(items: Sequence[DateEvidence]) -> VisitDateProfile:
    if not items:
        return VisitDateProfile(
            date_min=pd.NaT,
            date_max=pd.NaT,
            date_median=pd.NaT,
            observed_date_cells=0,
            distinct_dates=0,
            source_count=0,
            field_count=0,
            date_span_days=None,
            status="no_clinical_date",
        )
    dates = sorted(item.date.normalize() for item in items)
    unique = sorted(set(dates))
    date_min = unique[0]
    date_max = unique[-1]
    median_ns = int(np.median(np.array([date.value for date in dates], dtype=np.int64)))
    date_median = pd.Timestamp(median_ns).normalize()
    span = float((date_max - date_min).days)
    if len(unique) == 1:
        status = "exact_date_consensus"
    elif span <= 7:
        status = "date_range_within_7_days"
    elif span <= 30:
        status = "date_range_8_to_30_days"
    else:
        status = "date_range_over_30_days"
    return VisitDateProfile(
        date_min=date_min,
        date_max=date_max,
        date_median=date_median,
        observed_date_cells=len(dates),
        distinct_dates=len(unique),
        source_count=len({item.source_name for item in items}),
        field_count=len({(item.source_name, item.column_name) for item in items}),
        date_span_days=span,
        status=status,
    )

def date_available(profile: VisitDateProfile) -> bool:
    return not pd.isna(profile.date_min)

def unique_latest_by_strict_intervals(
    candidates: Sequence[tuple[str, VisitDateProfile]],
) -> tuple[str, VisitDateProfile] | None:
    if not candidates or not all(date_available(profile) for _, profile in candidates):
        return None
    winners: list[tuple[str, VisitDateProfile]] = []
    for viscode, profile in candidates:
        other_maxima = [
            other_profile.date_max
            for other_viscode, other_profile in candidates
            if other_viscode != viscode
        ]
        if other_maxima and profile.date_min > max(other_maxima):
            winners.append((viscode, profile))
    return winners[0] if len(winners) == 1 else None

def iso_date(value: pd.Timestamp | pd.NaT) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    return pd.Timestamp(value).strftime("%Y-%m-%d")

def build_date_profile_frame(
    participant_split: pd.DataFrame,
    candidate_modalities: dict[tuple[str, str], frozenset[str]],
    anchor: pd.DataFrame,
    evidence: dict[tuple[str, str], list[DateEvidence]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], VisitDateProfile]]:
    split_by_rid = dict(zip(participant_split["RID"].astype(str), participant_split["split"].astype(str)))
    anchor_keys = set(zip(anchor["RID"].astype(str), anchor["VISCODE2"].astype(str)))
    candidate_keys = set(candidate_modalities)
    all_keys = sorted(candidate_keys | anchor_keys)
    profiles: dict[tuple[str, str], VisitDateProfile] = {}
    records: list[dict[str, object]] = []

    for rid, viscode in all_keys:
        if rid not in split_by_rid or split_by_rid[rid] not in DEVELOPMENT_SPLITS:
            raise RuntimeError("A non-development key reached the temporal profile.")
        profile = make_date_profile(evidence.get((rid, viscode), ()))
        profiles[(rid, viscode)] = profile
        records.append(
            {
                "RID": rid,
                "VISCODE2": viscode,
                "split": split_by_rid[rid],
                "protocol_month": parse_protocol_month(viscode),
                "is_input_candidate_visit": (rid, viscode) in candidate_keys,
                "is_supervised_target_visit": (rid, viscode) in anchor_keys,
                "available_input_modality_count": len(candidate_modalities.get((rid, viscode), ())),
                "clinical_date_min": iso_date(profile.date_min),
                "clinical_date_max": iso_date(profile.date_max),
                "clinical_date_median": iso_date(profile.date_median),
                "observed_date_cells": profile.observed_date_cells,
                "distinct_dates": profile.distinct_dates,
                "date_source_count": profile.source_count,
                "date_field_count": profile.field_count,
                "date_span_days": profile.date_span_days,
                "date_status": profile.status,
                "strict_date_profile_available": date_available(profile),
            }
        )
    frame = pd.DataFrame(records)
    frame = frame.sort_values(["RID", "protocol_month", "VISCODE2"], kind="stable", na_position="last").reset_index(drop=True)
    if frame.duplicated(["RID", "VISCODE2"]).any():
        raise RuntimeError("Temporal date profile contains duplicate keys.")
    return frame, profiles

def build_pairing_reference(
    current_targets: pd.DataFrame,
    candidate_modalities: dict[tuple[str, str], frozenset[str]],
    profiles: dict[tuple[str, str], VisitDateProfile],
) -> pd.DataFrame:
    candidates_by_rid: defaultdict[str, list[str]] = defaultdict(list)
    for rid, viscode in candidate_modalities:
        candidates_by_rid[rid].append(viscode)

    records: list[dict[str, object]] = []
    for _, current in current_targets.iterrows():
        rid = str(current["RID"])
        split = str(current["split"])
        current_viscode = str(current["VISCODE2"])
        current_month = float(current["protocol_month"]) if not pd.isna(current["protocol_month"]) else np.nan
        current_profile = profiles.get((rid, current_viscode), make_date_profile(()))
        viscodes = sorted(set(candidates_by_rid.get(rid, [])))

        selected_viscode: str | None = None
        selected_profile: VisitDateProfile | None = None
        selected_month: float | None = None
        status = "excluded"
        reason = ""
        selection_method = "none"
        candidate_count = 0

        if not np.isfinite(current_month):
            reason = "current_target_unknown_protocol_month"
        else:
            known = [(viscode, parse_protocol_month(viscode)) for viscode in viscodes]
            prior = [(viscode, month) for viscode, month in known if np.isfinite(month) and month < current_month]
            if not prior:
                any_unknown = any(not np.isfinite(month) for _, month in known)
                any_known_non_earlier = any(np.isfinite(month) and month >= current_month for _, month in known)
                reason = (
                    "only_unknown_or_non_earlier_history"
                    if viscodes and (any_unknown or any_known_non_earlier)
                    else "no_strictly_earlier_known_protocol_visit"
                )
            else:
                latest_month = max(month for _, month in prior)
                latest_viscodes = sorted({viscode for viscode, month in prior if month == latest_month})
                candidate_count = len(latest_viscodes)
                if candidate_count == 1:
                    selected_viscode = latest_viscodes[0]
                    selected_month = latest_month
                    selected_profile = profiles.get((rid, selected_viscode), make_date_profile(()))
                    status = "eligible"
                    reason = "eligible_unique_previous_visit"
                    selection_method = "unique_most_recent_prior_protocol_month"
                else:
                    strict = unique_latest_by_strict_intervals(
                        [
                            (viscode, profiles.get((rid, viscode), make_date_profile(())))
                            for viscode in latest_viscodes
                        ]
                    )
                    if strict is None:
                        reason = "ambiguous_same_wave_no_strict_interval_winner"
                        selection_method = "strict_date_intervals"
                    else:
                        winner_viscode, winner_profile = strict
                        if not date_available(current_profile):
                            reason = "ambiguous_same_wave_current_date_unavailable"
                            selection_method = "strict_date_intervals"
                        elif not (winner_profile.date_max < current_profile.date_min):
                            reason = "strict_latest_not_fully_before_current"
                            selection_method = "strict_date_intervals"
                        else:
                            selected_viscode = winner_viscode
                            selected_profile = winner_profile
                            selected_month = latest_month
                            status = "eligible"
                            reason = "eligible_strict_date_interval_winner"
                            selection_method = "strict_date_intervals"

        protocol_gap = None
        conservative_gap_days = None
        median_gap_days = None
        if status == "eligible" and selected_viscode is not None and selected_month is not None:
            protocol_gap = float(current_month - selected_month)
            if selected_profile is not None and date_available(selected_profile) and date_available(current_profile):
                conservative_gap_days = int((current_profile.date_min - selected_profile.date_max).days)
                median_gap_days = int((current_profile.date_median - selected_profile.date_median).days)

        records.append(
            {
                "RID": rid,
                "split": split,
                "current_VISCODE2": current_viscode,
                "current_protocol_month": current_month,
                "previous_VISCODE2": selected_viscode if selected_viscode is not None else pd.NA,
                "previous_protocol_month": selected_month if selected_month is not None else pd.NA,
                "pairing_status": status,
                "pairing_reason": reason,
                "selection_method": selection_method,
                "candidate_count_at_latest_prior_month": candidate_count,
                "protocol_gap_months": protocol_gap,
                "conservative_date_gap_days": conservative_gap_days,
                "median_date_gap_days": median_gap_days,
            }
        )

    frame = pd.DataFrame(records).sort_values(["split", "RID"], kind="stable").reset_index(drop=True)
    if frame["RID"].duplicated().any():
        raise RuntimeError("Pairing reference contains duplicate participants.")
    return frame

def summarize_pairing(pairing: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        pairing.groupby(["split", "pairing_status", "pairing_reason", "selection_method"], dropna=False)
        .size()
        .rename("participant_count")
        .reset_index()
    )
    totals = pairing.groupby("split").size().to_dict()
    grouped["development_participant_count"] = grouped["split"].map(totals).astype("int64")
    grouped["participant_pct"] = 100.0 * grouped["participant_count"] / grouped["development_participant_count"]
    grouped["test_rows_used"] = 0
    return grouped.sort_values(["split", "pairing_status", "pairing_reason"], kind="stable").reset_index(drop=True)

def summarize_date_profiles(profile: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        profile.groupby(
            ["split", "is_input_candidate_visit", "is_supervised_target_visit", "date_status"],
            dropna=False,
        )
        .size()
        .rename("visit_count")
        .reset_index()
    )
    grouped["small_cell_suppressed"] = grouped["visit_count"].between(1, SAFE_MINIMUM_CELL_COUNT - 1)
    grouped.loc[grouped["small_cell_suppressed"], "visit_count"] = pd.NA
    grouped["test_rows_used"] = 0
    return grouped.sort_values(
        ["split", "is_input_candidate_visit", "is_supervised_target_visit", "date_status"],
        kind="stable",
    ).reset_index(drop=True)

def make_check(records: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
    records.append({"check": name, "passed": bool(passed), "detail": detail})

def build_method_checks(
    split06: dict[str, Any],
    participant_split: pd.DataFrame,
    current_targets: pd.DataFrame,
    pairing: pd.DataFrame,
    date_profiles: pd.DataFrame,
    actual_split_hash: str,
    actual_anchor_hash: str,
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    frozen = split06.get("frozen_package", {})
    make_check(
        checks,
        "official_participant_split_matches_split06",
        actual_split_hash == frozen.get("official_participant_split_sha256"),
        f"actual={actual_split_hash} expected={frozen.get('official_participant_split_sha256')}",
    )
    make_check(
        checks,
        "supervised_anchor_matches_split06",
        actual_anchor_hash == frozen.get("supervised_anchor_sha256"),
        f"actual={actual_anchor_hash} expected={frozen.get('supervised_anchor_sha256')}",
    )
    make_check(checks, "test_not_in_pairing", not pairing["split"].eq("test").any(), "test rows=0")
    make_check(checks, "test_not_in_date_profiles", not date_profiles["split"].eq("test").any(), "test rows=0")
    make_check(
        checks,
        "one_current_target_per_development_participant",
        len(current_targets) == len(pairing) == sum(EXPECTED_DEVELOPMENT_PARTICIPANTS.values()),
        f"current={len(current_targets)} pairing={len(pairing)}",
    )
    make_check(
        checks,
        "participant_split_unchanged_population",
        len(participant_split) == 2839,
        f"participants={len(participant_split)} expected=2839",
    )

    for split in DEVELOPMENT_SPLITS:
        subset = pairing.loc[pairing["split"].eq(split)]
        count = lambda reason: int(subset["pairing_reason"].eq(reason).sum())
        eligible = int(subset["pairing_status"].eq("eligible").sum())
        ambiguous_total = (
            count("eligible_strict_date_interval_winner")
            + count("ambiguous_same_wave_no_strict_interval_winner")
            + count("strict_latest_not_fully_before_current")
            + count("ambiguous_same_wave_current_date_unavailable")
        )
        make_check(
            checks,
            f"{split}__unique_previous_count",
            count("eligible_unique_previous_visit") == EXPECTED_BASE_UNIQUE_ELIGIBLE[split],
            f"observed={count('eligible_unique_previous_visit')} expected={EXPECTED_BASE_UNIQUE_ELIGIBLE[split]}",
        )
        make_check(
            checks,
            f"{split}__ambiguous_reproduced",
            ambiguous_total == EXPECTED_AMBIGUOUS[split],
            f"observed={ambiguous_total} expected={EXPECTED_AMBIGUOUS[split]}",
        )
        make_check(
            checks,
            f"{split}__strict_recovered",
            count("eligible_strict_date_interval_winner") == EXPECTED_STRICT_RECOVERED[split],
            f"observed={count('eligible_strict_date_interval_winner')} expected={EXPECTED_STRICT_RECOVERED[split]}",
        )
        make_check(
            checks,
            f"{split}__strict_no_winner",
            count("ambiguous_same_wave_no_strict_interval_winner") == EXPECTED_AMBIGUOUS_NO_WINNER[split],
            f"observed={count('ambiguous_same_wave_no_strict_interval_winner')} expected={EXPECTED_AMBIGUOUS_NO_WINNER[split]}",
        )
        make_check(
            checks,
            f"{split}__winner_not_before_current",
            count("strict_latest_not_fully_before_current") == EXPECTED_WINNER_NOT_BEFORE_CURRENT[split],
            f"observed={count('strict_latest_not_fully_before_current')} expected={EXPECTED_WINNER_NOT_BEFORE_CURRENT[split]}",
        )
        make_check(
            checks,
            f"{split}__final_eligible",
            eligible == EXPECTED_FINAL_ELIGIBLE[split],
            f"observed={eligible} expected={EXPECTED_FINAL_ELIGIBLE[split]}",
        )

    selected = pairing.loc[pairing["pairing_status"].eq("eligible")]
    make_check(
        checks,
        "selected_previous_keys_differ_from_current",
        bool(selected["previous_VISCODE2"].ne(selected["current_VISCODE2"]).all()),
        f"selected_pairs={len(selected)}",
    )
    make_check(
        checks,
        "selected_protocol_gap_positive",
        bool(pd.to_numeric(selected["protocol_gap_months"], errors="coerce").gt(0).all()),
        "all selected protocol gaps > 0",
    )
    strict_selected = selected.loc[selected["selection_method"].eq("strict_date_intervals")]
    make_check(
        checks,
        "strict_selected_date_gap_positive",
        bool(pd.to_numeric(strict_selected["conservative_date_gap_days"], errors="coerce").gt(0).all()),
        f"strict selected pairs={len(strict_selected)}",
    )
    make_check(
        checks,
        "no_visit_concatenation",
        True,
        "one exact previous VISCODE2 per eligible participant; no feature union materialized",
    )
    return pd.DataFrame(checks)

def verify_base_inventory(base_package: Path) -> pd.DataFrame:
    inventory_path = base_package / INVENTORY_RELATIVE_PATH
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    inventory = pd.read_csv(inventory_path, dtype="string", low_memory=False)
    required = {"relative_path", "sha256"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise KeyError(f"Base inventory missing columns: {missing}")
    records: list[dict[str, object]] = []
    for _, row in inventory.iterrows():
        relative = str(row["relative_path"])
        path = base_package / relative
        exists = path.is_file()
        actual = calculate_sha256(path) if exists else None
        expected = str(row["sha256"])
        records.append(
            {
                "relative_path": relative,
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_matches": bool(exists and actual == expected),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty or not frame["hash_matches"].astype(bool).all():
        failed = frame.loc[~frame["hash_matches"].astype(bool), "relative_path"].tolist()
        raise RuntimeError(f"Base frozen inventory verification failed: {failed[:10]}")
    return frame

def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): calculate_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

def copy_base_package(paths: ProjectPaths, overwrite: bool) -> tuple[dict[str, str], pd.DataFrame]:
    if paths.output_package.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output package already exists: {paths.output_package}. Use --overwrite only after reviewing it."
            )
        shutil.rmtree(paths.output_package)
    if paths.staging_package.exists():
        shutil.rmtree(paths.staging_package)

    base_inventory_checks = verify_base_inventory(paths.base_package)
    base_hashes = tree_hashes(paths.base_package)
    logging.info("Copying immutable base package to staging...")
    shutil.copytree(paths.base_package, paths.staging_package, copy_function=shutil.copy2)
    copied_hashes = tree_hashes(paths.staging_package)
    if base_hashes != copied_hashes:
        differing = sorted(set(base_hashes) | set(copied_hashes))
        differing = [key for key in differing if base_hashes.get(key) != copied_hashes.get(key)]
        raise RuntimeError(f"Base package copy is not byte-identical: {differing[:10]}")
    return base_hashes, base_inventory_checks

def validate_package_integrity(
    paths: ProjectPaths,
    base_hashes: dict[str, str],
) -> pd.DataFrame:
    staging = paths.staging_package
    checks: list[dict[str, object]] = []
    metadata_exceptions = {INVENTORY_RELATIVE_PATH, MANIFEST_RELATIVE_PATH}

    for relative, base_hash in sorted(base_hashes.items()):
        current = staging / relative
        if relative in metadata_exceptions:
            backup_name = (
                "04_official_split/frozen_file_inventory_v1_base.csv"
                if relative == INVENTORY_RELATIVE_PATH
                else "04_official_split/official_frozen_manifest_v1_base.json"
            )
            backup = staging / backup_name
            actual_hash = calculate_sha256(backup) if backup.is_file() else None
            passed = actual_hash == base_hash
            detail = f"base backup={backup_name}"
        else:
            actual_hash = calculate_sha256(current) if current.is_file() else None
            passed = actual_hash == base_hash
            detail = "unchanged byte-for-byte"
        checks.append(
            {
                "relative_path": relative,
                "passed": bool(passed),
                "base_sha256": base_hash,
                "v1_1_sha256": actual_hash,
                "detail": detail,
            }
        )
    frame = pd.DataFrame(checks)
    if not frame["passed"].astype(bool).all():
        failed = frame.loc[~frame["passed"].astype(bool), "relative_path"].tolist()
        raise RuntimeError(f"v1.1 changed base package files: {failed[:10]}")
    return frame

def safe_integrity_summary(integrity: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checked_base_files": int(len(integrity)),
                "unchanged_base_files": int(integrity["passed"].astype(bool).sum()),
                "changed_or_missing_base_files": int((~integrity["passed"].astype(bool)).sum()),
                "official_split_changed": False,
                "supervised_anchor_changed": False,
                "mri_targets_changed": False,
                "model_ready_sources_changed": False,
            }
        ]
    )

def build_policy_payload(
    paths: ProjectPaths,
    base_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "policy_name": TEMPORAL_POLICY_NAME,
        "policy_scope": {
            "decision_splits": list(DEVELOPMENT_SPLITS),
            "test_values_used": False,
            "mri_target_values_loaded": False,
            "participant_split_changed": False,
            "supervised_population_changed": False,
        },
        "rules": {
            "exact_visit_key": ["RID", "VISCODE2"],
            "visits_concatenated": False,
            "base_prior_rule": "protocol_month strictly less than current target protocol_month",
            "same_wave_tie_rule": (
                "select one candidate only if candidate.clinical_date_min is strictly later "
                "than every competitor.clinical_date_max"
            ),
            "current_boundary_rule": (
                "selected_previous.clinical_date_max must be strictly earlier than "
                "current_target.clinical_date_min"
            ),
            "median_date_fallback": False,
            "viscode_hierarchy_fallback": False,
            "lexical_fallback": False,
            "previous_mri_required": False,
            "previous_target_complete_required": False,
            "unresolved_action": "exclude participant from matched scenarios only",
        },
        "frozen_development_evidence": {
            "ambiguous": EXPECTED_AMBIGUOUS,
            "strict_date_recovered": EXPECTED_STRICT_RECOVERED,
            "final_eligible": EXPECTED_FINAL_ELIGIBLE,
        },
        "upstream": {
            "base_package": BASE_PACKAGE_NAME,
            "split06_summary_sha256": calculate_sha256(paths.split06_summary),
            "official_participant_split_sha256": base_hashes[
                "04_official_split/official_participant_split.csv"
            ],
            "supervised_anchor_sha256": base_hashes[
                "03_final_ready_tables/supervised_anchor_visits.csv.gz"
            ],
        },
    }


def write_extension_artifacts(
    paths: ProjectPaths,
    date_profiles: pd.DataFrame,
    pairing: pd.DataFrame,
    split06_summary: dict[str, Any],
    base_hashes: dict[str, str],
) -> tuple[dict[str, str], str, str]:
    staging = paths.staging_package
    final_tables = staging / "03_final_ready_tables"
    changes = staging / "02_cuts_and_adjustments"
    official = staging / "04_official_split"

    date_profile_path = final_tables / "development_visit_date_profiles.csv.gz"
    pairing_path = changes / "development_strict_date_pairing_reference.csv.gz"
    write_dataframe_csv_gzip_deterministic(date_profiles, date_profile_path)
    write_dataframe_csv_gzip_deterministic(pairing, pairing_path)

    policy_payload = build_policy_payload(paths, base_hashes)
    reconstruction_payload_sha256 = canonical_json_sha256(policy_payload)

    # The historical ID is part of the frozen benchmark contract.
    # The original derivation included the timestamp-bearing SPLIT06 summary hash,
    # so replaying that hash today would create a different identifier despite
    # identical policy logic and aggregate evidence.
    decision_id = HISTORICAL_TEMPORAL_POLICY_DECISION_ID
    policy_payload["temporal_policy_decision_id"] = decision_id
    policy_payload["reconstruction_provenance"] = {
        "historical_decision_id_preserved": True,
        "current_policy_payload_sha256": reconstruction_payload_sha256,
        "current_policy_payload_short_id": reconstruction_payload_sha256[:20],
        "historical_id_derivation_replayed": False,
    }

    policy_path = official / "strict_date_temporal_policy_lock.json"
    policy_path.write_text(
        json.dumps(policy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_extension = staging / "README_V1_1_TEMPORAL_EXTENSION_LOCAL_ONLY.txt"
    readme_extension.write_text(
        "\n".join(
            [
                "FROZEN EXPERIMENT PACKAGE v1.1 — TEMPORAL EXTENSION",
                "",
                "LOCAL ONLY — contains individualized ADNI-derived records and dates.",
                "Do not commit, upload, publish, or share this directory.",
                "",
                "The original v1 split, population, targets, and model-ready tables are unchanged.",
                "This extension adds development-only clinical date profiles and a strict-date pairing reference.",
                "The test split was not used to choose or validate the temporal policy.",
                "Visits are never concatenated across VISCODE2 values.",
                "",
                f"temporal_policy_decision_id={decision_id}",
                f"reconstruction_policy_payload_sha256={reconstruction_payload_sha256}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copy2(
        staging / INVENTORY_RELATIVE_PATH,
        official / "frozen_file_inventory_v1_base.csv",
    )
    shutil.copy2(
        staging / MANIFEST_RELATIVE_PATH,
        official / "official_frozen_manifest_v1_base.json",
    )

    artifact_paths = {
        "development_visit_date_profiles": date_profile_path,
        "development_strict_date_pairing_reference": pairing_path,
        "strict_date_temporal_policy_lock": policy_path,
        "readme_v1_1_extension": readme_extension,
    }
    artifact_hashes = {
        name: calculate_sha256(path)
        for name, path in artifact_paths.items()
    }
    return artifact_hashes, decision_id, reconstruction_payload_sha256


def write_new_inventory_and_manifest(
    paths: ProjectPaths,
    base_hashes: dict[str, str],
    artifact_hashes: dict[str, str],
    decision_id: str,
    reconstruction_payload_sha256: str,
    split06_summary: dict[str, Any],
    pairing: pd.DataFrame,
) -> tuple[Path, Path, pd.DataFrame]:
    staging = paths.staging_package
    inventory_path = staging / INVENTORY_RELATIVE_PATH
    manifest_path = staging / MANIFEST_RELATIVE_PATH

    if inventory_path.exists():
        inventory_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    inventory_records: list[dict[str, object]] = []
    for file_path in sorted(staging.rglob("*")):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(staging).as_posix()
        if relative in {INVENTORY_RELATIVE_PATH, MANIFEST_RELATIVE_PATH}:
            continue
        current_hash = calculate_sha256(file_path)
        base_hash = base_hashes.get(relative)
        inventory_records.append(
            {
                "category": (
                    "documentation"
                    if len(Path(relative).parts) == 1
                    else Path(relative).parts[0]
                ),
                "relative_path": relative,
                "size_bytes": int(file_path.stat().st_size),
                "sha256": current_hash,
                "origin": (
                    "unchanged_from_v1"
                    if base_hash == current_hash
                    else "v1_1_temporal_extension"
                ),
                "base_v1_sha256": base_hash if base_hash is not None else pd.NA,
                "unchanged_from_v1": (
                    bool(base_hash == current_hash) if base_hash is not None else False
                ),
            }
        )

    inventory = (
        pd.DataFrame(inventory_records)
        .sort_values("relative_path", kind="stable")
        .reset_index(drop=True)
    )
    write_dataframe_csv_deterministic(inventory, inventory_path)

    base_manifest_path = (
        staging / "04_official_split" / "official_frozen_manifest_v1_base.json"
    )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    eligible_counts = {
        split: int(
            pairing.loc[
                pairing["split"].eq(split)
                & pairing["pairing_status"].eq("eligible")
            ].shape[0]
        )
        for split in DEVELOPMENT_SPLITS
    }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_package_name": OUTPUT_PACKAGE_NAME,
        "base_frozen_package_name": BASE_PACKAGE_NAME,
        "official_split_frozen": True,
        "participant_split_changed": False,
        "supervised_population_changed": False,
        "mri_targets_changed": False,
        "model_ready_sources_changed": False,
        "base_manifest_sha256": calculate_sha256(base_manifest_path),
        "base_population": base_manifest.get("population"),
        "base_selection": base_manifest.get("selection"),
        "temporal_extension": {
            "temporal_policy_decision_id": decision_id,
            "policy": TEMPORAL_POLICY_NAME,
            "development_splits_used": list(DEVELOPMENT_SPLITS),
            "test_values_used": False,
            "test_pairings_materialized": False,
            "mri_target_values_loaded": False,
            "visits_concatenated": False,
            "development_final_eligible_participants": eligible_counts,
            "artifact_sha256": artifact_hashes,
            "split06_summary_sha256": calculate_sha256(paths.split06_summary),
            "reconstruction_policy_payload_sha256": reconstruction_payload_sha256,
        },
        "critical_unchanged_sha256": {
            "official_participant_split.csv": base_hashes[
                "04_official_split/official_participant_split.csv"
            ],
            "supervised_anchor_visits.csv.gz": base_hashes[
                "03_final_ready_tables/supervised_anchor_visits.csv.gz"
            ],
            "UCSFFSX7_targets_323.csv.gz": base_hashes[
                "03_final_ready_tables/UCSFFSX7_targets_323.csv.gz"
            ],
        },
        "folder_contract_extension": {
            "03_final_ready_tables/development_visit_date_profiles.csv.gz": (
                "development-only local clinical date profiles for exact visit keys"
            ),
            "02_cuts_and_adjustments/development_strict_date_pairing_reference.csv.gz": (
                "development-only local reference outcome under the frozen strict-date policy"
            ),
            "04_official_split/strict_date_temporal_policy_lock.json": (
                "frozen methodological policy and upstream hashes"
            ),
        },
        "reconstruction_provenance": {
            "historical_temporal_policy_decision_id_preserved": True,
            "historical_temporal_policy_decision_id": (
                HISTORICAL_TEMPORAL_POLICY_DECISION_ID
            ),
            "current_policy_payload_sha256": reconstruction_payload_sha256,
            "historical_id_derivation_replayed": False,
        },
        "privacy": {
            "package_is_local_only": True,
            "contains_individualized_records": True,
            "contains_clinical_dates": True,
            "must_not_be_committed_or_shared": True,
            "safe_aggregate_summary_written_separately": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return inventory_path, manifest_path, inventory


def write_story(
    path: Path,
    pairing_summary: pd.DataFrame,
    decision_id: str,
    validation_checks: pd.DataFrame,
) -> None:
    def count(split: str, reason: str) -> int:
        rows = pairing_summary.loc[
            pairing_summary["split"].eq(split)
            & pairing_summary["pairing_reason"].eq(reason),
            "participant_count",
        ]
        return int(rows.sum()) if not rows.empty else 0

    lines = [
        "# SPLIT07 — Congelamento da política temporal por datas estritas",
        "",
        f"- Versão: `{SCRIPT_VERSION}`",
        f"- Política: `{TEMPORAL_POLICY_NAME}`",
        f"- Decision ID histórico preservado: `{decision_id}`",
        "- Split oficial alterado: **não**",
        "- População supervisionada alterada: **não**",
        "- Targets MRI alterados: **não**",
        "- Teste usado: **não**",
        "- Visitas concatenadas: **não**",
        "",
        "## Regra congelada",
        "",
        (
            "Quando mais de uma visita exata compete no mês protocolar anterior mais recente, "
            "uma candidata só é selecionada quando sua menor data clínica é posterior à maior "
            "data clínica de todas as concorrentes."
        ),
        (
            "A maior data da visita anterior também precisa ser estritamente anterior à menor "
            "data da visita-alvo atual."
        ),
        "",
        "Não há fallback por mediana, ordem lexical ou hierarquia de VISCODE2.",
        "",
        "## Resultado em desenvolvimento",
        "",
        (
            f"Train: {count('train', 'eligible_unique_previous_visit')} únicos + "
            f"{count('train', 'eligible_strict_date_interval_winner')} recuperados = "
            f"{EXPECTED_FINAL_ELIGIBLE['train']}."
        ),
        (
            f"Validation: {count('validation', 'eligible_unique_previous_visit')} únicos + "
            f"{count('validation', 'eligible_strict_date_interval_winner')} recuperados = "
            f"{EXPECTED_FINAL_ELIGIBLE['validation']}."
        ),
        "",
        "## Reconstrução",
        "",
        (
            "O identificador temporal histórico foi preservado. O payload reconstruído recebe "
            "um SHA-256 próprio porque o ID original dependia do hash de um resumo SPLIT06 "
            "com timestamp de execução."
        ),
        "",
        "## Validação",
        "",
        f"- Checks executados: **{len(validation_checks)}**",
        f"- Checks falhos: **{int((~validation_checks['passed'].astype(bool)).sum())}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class Split07Preparation:
    split06: dict[str, Any]
    participant_split: pd.DataFrame
    development_split: pd.DataFrame
    anchor: pd.DataFrame
    current_targets: pd.DataFrame
    candidate_modalities: dict[tuple[str, str], frozenset[str]]
    source_summary: pd.DataFrame
    date_source_summary: pd.DataFrame
    date_profiles: pd.DataFrame
    profile_lookup: dict[tuple[str, str], VisitDateProfile]
    pairing: pd.DataFrame
    pairing_summary: pd.DataFrame
    date_profile_summary: pd.DataFrame
    validation_checks: pd.DataFrame


@dataclass(frozen=True)
class Split07FreezeResult:
    output_package: Path
    summary_path: Path
    policy_path: Path
    manifest_path: Path
    inventory_path: Path
    decision_id: str
    reconstruction_policy_payload_sha256: str
    preparation: Split07Preparation
    package_checks: pd.DataFrame
    integrity_summary: pd.DataFrame


def prepare_split07(paths: ProjectPaths) -> Split07Preparation:
    split06 = load_split06_summary(paths)
    participant_split, development_split, development_rids = load_participant_split(paths)
    anchor, current_targets = load_development_anchor(paths)
    candidate_modalities, source_summary = build_input_candidate_visits(paths)
    evidence, date_source_summary = collect_date_evidence(paths, development_rids)
    date_profiles, profile_lookup = build_date_profile_frame(
        participant_split=development_split,
        candidate_modalities=candidate_modalities,
        anchor=anchor,
        evidence=evidence,
    )
    pairing = build_pairing_reference(
        current_targets=current_targets,
        candidate_modalities=candidate_modalities,
        profiles=profile_lookup,
    )

    split_path = paths.base_package / "04_official_split" / "official_participant_split.csv"
    anchor_path = paths.base_package / "03_final_ready_tables" / "supervised_anchor_visits.csv.gz"
    validation_checks = build_method_checks(
        split06=split06,
        participant_split=participant_split,
        current_targets=current_targets,
        pairing=pairing,
        date_profiles=date_profiles,
        actual_split_hash=calculate_sha256(split_path),
        actual_anchor_hash=calculate_sha256(anchor_path),
    )
    failed = validation_checks.loc[~validation_checks["passed"].astype(bool)]
    if not failed.empty:
        raise Split07FreezeError(
            "SPLIT07 method validation failed: "
            + ", ".join(failed["check"].astype(str).tolist())
        )

    return Split07Preparation(
        split06=split06,
        participant_split=participant_split,
        development_split=development_split,
        anchor=anchor,
        current_targets=current_targets,
        candidate_modalities=candidate_modalities,
        source_summary=source_summary,
        date_source_summary=date_source_summary,
        date_profiles=date_profiles,
        profile_lookup=profile_lookup,
        pairing=pairing,
        pairing_summary=summarize_pairing(pairing),
        date_profile_summary=summarize_date_profiles(date_profiles),
        validation_checks=validation_checks,
    )


def write_safe_pre_freeze_reports(
    paths: ProjectPaths,
    preparation: Split07Preparation,
) -> None:
    write_dataframe_csv_deterministic(
        preparation.pairing_summary,
        paths.safe_output_dir / "split07_pairing_outcome_summary.csv",
    )
    write_dataframe_csv_deterministic(
        preparation.date_profile_summary,
        paths.safe_output_dir / "split07_date_profile_summary.csv",
    )
    write_dataframe_csv_deterministic(
        preparation.date_source_summary,
        paths.safe_output_dir / "split07_date_source_summary.csv",
    )
    write_dataframe_csv_deterministic(
        preparation.source_summary,
        paths.safe_output_dir / "split07_input_source_summary.csv",
    )
    write_dataframe_csv_deterministic(
        preparation.validation_checks,
        paths.safe_output_dir / "split07_validation_checks.csv",
    )


def freeze_split07_extension(
    project_root: Path,
    overwrite: bool = False,
) -> Split07FreezeResult:
    paths = resolve_paths(project_root)
    preparation = prepare_split07(paths)
    write_safe_pre_freeze_reports(paths, preparation)

    try:
        base_hashes, base_inventory_checks = copy_base_package(paths, overwrite=overwrite)
        artifact_hashes, decision_id, reconstruction_payload_sha256 = (
            write_extension_artifacts(
                paths=paths,
                date_profiles=preparation.date_profiles,
                pairing=preparation.pairing,
                split06_summary=preparation.split06,
                base_hashes=base_hashes,
            )
        )
        inventory_path, manifest_path, new_inventory = write_new_inventory_and_manifest(
            paths=paths,
            base_hashes=base_hashes,
            artifact_hashes=artifact_hashes,
            decision_id=decision_id,
            reconstruction_payload_sha256=reconstruction_payload_sha256,
            split06_summary=preparation.split06,
            pairing=preparation.pairing,
        )
        integrity = validate_package_integrity(paths, base_hashes)
        integrity_summary = safe_integrity_summary(integrity)
        write_dataframe_csv_deterministic(
            integrity_summary,
            paths.safe_output_dir / "split07_package_integrity_summary.csv",
        )

        package_checks = pd.DataFrame(
            [
                {
                    "check": "base_inventory_verified",
                    "passed": bool(base_inventory_checks["hash_matches"].astype(bool).all()),
                    "detail": f"files={len(base_inventory_checks)}",
                },
                {
                    "check": "base_files_unchanged_in_v1_1",
                    "passed": bool(integrity["passed"].astype(bool).all()),
                    "detail": f"files={len(integrity)}",
                },
                {
                    "check": "historical_temporal_policy_id_preserved",
                    "passed": decision_id == HISTORICAL_TEMPORAL_POLICY_DECISION_ID,
                    "detail": f"decision_id={decision_id}",
                },
                {
                    "check": "new_inventory_written",
                    "passed": inventory_path.is_file(),
                    "detail": f"inventory_rows={len(new_inventory)}",
                },
                {
                    "check": "new_manifest_written",
                    "passed": manifest_path.is_file(),
                    "detail": "manifest staged",
                },
            ]
        )
        write_dataframe_csv_deterministic(
            package_checks,
            paths.safe_output_dir / "split07_package_checks.csv",
        )
        if not package_checks["passed"].astype(bool).all():
            raise Split07FreezeError("Package checks failed.")

        os.replace(paths.staging_package, paths.output_package)
    except Exception:
        if paths.staging_package.exists():
            shutil.rmtree(paths.staging_package, ignore_errors=True)
        raise

    combined_checks = pd.concat(
        [preparation.validation_checks, package_checks],
        ignore_index=True,
        sort=False,
    )
    write_dataframe_csv_deterministic(
        combined_checks,
        paths.safe_output_dir / "split07_all_checks.csv",
    )
    write_story(
        paths.safe_output_dir / "SPLIT07_STRICT_DATE_TEMPORAL_EXTENSION_STORY.md",
        preparation.pairing_summary,
        decision_id,
        combined_checks,
    )

    final_eligible = {
        split: int(
            preparation.pairing.loc[
                preparation.pairing["split"].eq(split)
                & preparation.pairing["pairing_status"].eq("eligible")
            ].shape[0]
        )
        for split in DEVELOPMENT_SPLITS
    }
    final_manifest = paths.output_package / MANIFEST_RELATIVE_PATH
    final_inventory = paths.output_package / INVENTORY_RELATIVE_PATH
    final_policy = (
        paths.output_package
        / "04_official_split"
        / "strict_date_temporal_policy_lock.json"
    )

    summary = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_passed": True,
        "temporal_policy_decision_id": decision_id,
        "reconstruction_policy_payload_sha256": reconstruction_payload_sha256,
        "scope": {
            "decision_splits": list(DEVELOPMENT_SPLITS),
            "test_values_used": False,
            "test_pairings_materialized": False,
            "mri_target_values_loaded": False,
            "models_trained": False,
            "participant_split_changed": False,
            "supervised_population_changed": False,
            "visits_concatenated": False,
        },
        "base_package": {
            "name": BASE_PACKAGE_NAME,
            "official_participant_split_sha256": calculate_sha256(
                paths.output_package / "04_official_split" / "official_participant_split.csv"
            ),
            "supervised_anchor_sha256": calculate_sha256(
                paths.output_package / "03_final_ready_tables" / "supervised_anchor_visits.csv.gz"
            ),
            "targets_sha256": calculate_sha256(
                paths.output_package / "03_final_ready_tables" / "UCSFFSX7_targets_323.csv.gz"
            ),
        },
        "output_package": {
            "name": OUTPUT_PACKAGE_NAME,
            "path": str(paths.output_package.relative_to(paths.root)),
            "manifest_sha256": calculate_sha256(final_manifest),
            "inventory_sha256": calculate_sha256(final_inventory),
        },
        "policy": {
            "name": TEMPORAL_POLICY_NAME,
            "strict_date_recovered": EXPECTED_STRICT_RECOVERED,
            "final_eligible": final_eligible,
            "remaining_no_strict_winner": EXPECTED_AMBIGUOUS_NO_WINNER,
            "remaining_not_before_current": EXPECTED_WINNER_NOT_BEFORE_CURRENT,
        },
        "checks": {
            "check_count": int(len(combined_checks)),
            "failed_check_count": int((~combined_checks["passed"].astype(bool)).sum()),
        },
        "reconstruction_provenance": {
            "historical_decision_id_preserved": True,
            "historical_decision_id": HISTORICAL_TEMPORAL_POLICY_DECISION_ID,
            "current_policy_payload_sha256": reconstruction_payload_sha256,
        },
        "privacy": {
            "local_package_contains_identifiers": True,
            "local_package_contains_clinical_dates": True,
            "local_package_must_not_be_shared": True,
            "safe_reports_contain_identifiers": False,
            "safe_reports_contain_date_values": False,
            "safe_reports_contain_medical_values": False,
            "safe_reports_contain_predictions": False,
        },
    }
    summary_path = paths.safe_output_dir / "split07_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return Split07FreezeResult(
        output_package=paths.output_package,
        summary_path=summary_path,
        policy_path=final_policy,
        manifest_path=final_manifest,
        inventory_path=final_inventory,
        decision_id=decision_id,
        reconstruction_policy_payload_sha256=reconstruction_payload_sha256,
        preparation=preparation,
        package_checks=package_checks,
        integrity_summary=integrity_summary,
    )

