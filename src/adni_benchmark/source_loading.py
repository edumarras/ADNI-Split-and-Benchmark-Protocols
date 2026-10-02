from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Final, Mapping

import pandas as pd

from adni_benchmark.source_discovery import REQUIRED_SOURCES
from adni_benchmark.source_inspection import SourceSnapshot
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
    SourceContract,
)


MISSING_SENTINELS: Final[frozenset[str]] = frozenset(
    {
        "",
        "-1",
        "-4",
    }
)


class SourceLoadingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceLoadSummary:
    source_name: str
    n_rows: int
    n_columns: int
    rows_with_missing_key: int
    duplicate_key_rows: int
    missing_key_counts: tuple[tuple[str, int], ...]


def normalize_missing_values(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    # Normalize whitespace and the frozen ADNI missing-value sentinels
    normalized = frame.copy()

    for column in normalized.columns:
        values = (
            normalized[column]
            .astype("string")
            .str.strip()
        )

        normalized[column] = values.mask(
            values.isin(MISSING_SENTINELS),
            pd.NA,
        )

    return normalized


def normalize_integer_identifier(
    series: pd.Series,
    source_name: str,
    column_name: str,
) -> pd.Series:
    # Identifiers are numeric in the files but remain labels, not quantities
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_mask = (
        series.notna()
        & numeric.isna()
    )

    if invalid_mask.any():
        invalid_count = int(
            invalid_mask.sum()
        )

        raise SourceLoadingError(
            f"{source_name}.{column_name} contains "
            f"{invalid_count} observed non-numeric values."
        )

    non_integer_mask = (
        numeric.notna()
        & numeric.ne(numeric.round())
    )

    if non_integer_mask.any():
        non_integer_count = int(
            non_integer_mask.sum()
        )

        raise SourceLoadingError(
            f"{source_name}.{column_name} contains "
            f"{non_integer_count} non-integer values."
        )

    return (
        numeric
        .round()
        .astype("Int64")
        .astype("string")
    )


def normalize_source_keys(
    frame: pd.DataFrame,
    contract: SourceContract,
) -> pd.DataFrame:
    # Normalize only structural identifiers and protocol labels at this stage
    normalized = frame.copy()

    if "RID" in normalized.columns:
        normalized["RID"] = normalize_integer_identifier(
            series=normalized["RID"],
            source_name=contract.source_name,
            column_name="RID",
        )

    if "IMAGEUID" in normalized.columns:
        normalized["IMAGEUID"] = normalize_integer_identifier(
            series=normalized["IMAGEUID"],
            source_name=contract.source_name,
            column_name="IMAGEUID",
        )

    if "VISCODE2" in normalized.columns:
        normalized["VISCODE2"] = (
            normalized["VISCODE2"]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

    if "PHASE" in normalized.columns:
        normalized["PHASE"] = (
            normalized["PHASE"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    return normalized


def load_source_table(
    snapshot: SourceSnapshot,
    contract: SourceContract,
) -> pd.DataFrame:
    # Fail closed if the wrong source contract is supplied
    if snapshot.source_name != contract.source_name:
        raise SourceLoadingError(
            "Snapshot and contract refer to different sources. "
            f"Snapshot={snapshot.source_name}, "
            f"contract={contract.source_name}."
        )

    # Disable pandas automatic missing-value inference
    frame = pd.read_csv(
        snapshot.absolute_path,
        compression="infer",
        encoding=snapshot.encoding,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
        low_memory=False,
    )

    frame.columns = [
        column.strip().lstrip("\ufeff")
        for column in frame.columns
    ]

    loaded_columns = tuple(
        frame.columns
    )

    # Detect file replacement or modification after source inspection
    if loaded_columns != snapshot.columns:
        raise SourceLoadingError(
            f"{snapshot.source_name} header changed between "
            "inspection and full loading."
        )

    frame = normalize_missing_values(
        frame
    )

    frame = normalize_source_keys(
        frame=frame,
        contract=contract,
    )

    return frame


def summarize_loaded_source(
    frame: pd.DataFrame,
    contract: SourceContract,
) -> SourceLoadSummary:
    # Count structural problems but do not resolve them yet
    key_columns = list(
        contract.key_columns
    )

    key_frame = frame.loc[
        :,
        key_columns,
    ]

    missing_key_counts = tuple(
        (
            column,
            int(key_frame[column].isna().sum()),
        )
        for column in key_columns
    )

    missing_key_mask = (
        key_frame
        .isna()
        .any(axis=1)
    )

    complete_key_frame = frame.loc[
        ~missing_key_mask
    ]

    duplicate_key_rows = int(
        complete_key_frame.duplicated(
            subset=key_columns,
            keep=False,
        ).sum()
    )

    return SourceLoadSummary(
        source_name=contract.source_name,
        n_rows=len(frame),
        n_columns=len(frame.columns),
        rows_with_missing_key=int(
            missing_key_mask.sum()
        ),
        duplicate_key_rows=duplicate_key_rows,
        missing_key_counts=missing_key_counts,
    )


def load_all_sources(
    snapshots: Mapping[str, SourceSnapshot],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, SourceLoadSummary],
]:
    missing_snapshots = [
        source_name
        for source_name in REQUIRED_SOURCES
        if source_name not in snapshots
    ]

    if missing_snapshots:
        raise SourceLoadingError(
            f"Missing source snapshots: {missing_snapshots}"
        )

    source_tables: dict[str, pd.DataFrame] = {}
    summaries: dict[str, SourceLoadSummary] = {}

    for source_name in REQUIRED_SOURCES:
        logging.info(
            "Loading %s...",
            source_name,
        )

        snapshot = snapshots[source_name]
        contract = SOURCE_CONTRACTS[source_name]

        frame = load_source_table(
            snapshot=snapshot,
            contract=contract,
        )

        summary = summarize_loaded_source(
            frame=frame,
            contract=contract,
        )

        source_tables[source_name] = frame
        summaries[source_name] = summary

    return source_tables, summaries