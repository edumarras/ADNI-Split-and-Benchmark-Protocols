from __future__ import annotations

import hashlib
import re

from collections import Counter
from dataclasses import dataclass
from typing import Final, Sequence

import pandas as pd


MRI_MEASURE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^ST\d+(?:SV|CV|SA|TA|TS)$",
    flags=re.IGNORECASE,
)

EXPECTED_MRI_MEASURE_COUNT: Final[int] = 325
EXPECTED_MRI_TARGET_COUNT: Final[int] = 323

FIXED_EXCLUDED_MRI_TARGETS: Final[tuple[str, ...]] = (
    "ST8SV",
    "ST68SV",
)


class MRICatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRICatalogSummary:
    source_name: str
    n_source_columns: int
    n_non_measure_columns: int
    n_measure_columns: int
    n_target_columns: int
    excluded_targets: tuple[str, ...]
    measure_catalog_sha256: str
    target_catalog_sha256: str


def calculate_column_catalog_sha256(
    columns: Sequence[str],
) -> str:
    # Normalize case and whitespace while preserving column order
    canonical_columns = [
        str(column).strip().upper()
        for column in columns
    ]

    catalog_text = "\n".join(
        canonical_columns
    )

    return hashlib.sha256(
        catalog_text.encode("utf-8")
    ).hexdigest()


def build_mri_catalog(
    frame: pd.DataFrame,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    MRICatalogSummary,
]:
    # Preserve the exact source-column order
    source_columns = tuple(
        str(column).strip()
        for column in frame.columns
    )

    measure_columns = tuple(
        column
        for column in source_columns
        if MRI_MEASURE_PATTERN.fullmatch(column)
    )

    # Compare names case-insensitively without changing the original names
    canonical_measure_columns = tuple(
        column.upper()
        for column in measure_columns
    )

    duplicate_measure_names = sorted(
        column
        for column, count in Counter(
            canonical_measure_columns
        ).items()
        if count > 1
    )

    if duplicate_measure_names:
        raise MRICatalogError(
            "UCSFFSX7 contains MRI measurement column names "
            "duplicated case-insensitively: "
            f"{duplicate_measure_names}"
        )

    if len(measure_columns) != EXPECTED_MRI_MEASURE_COUNT:
        raise MRICatalogError(
            "Unexpected UCSFFSX7 measurement catalog size. "
            f"Observed {len(measure_columns)} ST columns; "
            f"expected {EXPECTED_MRI_MEASURE_COUNT}."
        )

    canonical_measure_set = set(
        canonical_measure_columns
    )

    canonical_exclusions = tuple(
        column.strip().upper()
        for column in FIXED_EXCLUDED_MRI_TARGETS
    )

    duplicated_exclusions = sorted(
        column
        for column, count in Counter(
            canonical_exclusions
        ).items()
        if count > 1
    )

    if duplicated_exclusions:
        raise MRICatalogError(
            "The fixed MRI exclusion catalog contains duplicates: "
            f"{duplicated_exclusions}"
        )

    missing_exclusions = [
        column
        for column in canonical_exclusions
        if column not in canonical_measure_set
    ]

    if missing_exclusions:
        raise MRICatalogError(
            "The following fixed MRI exclusions are absent "
            f"from UCSFFSX7: {missing_exclusions}"
        )

    exclusion_set = set(
        canonical_exclusions
    )

    target_columns = tuple(
        original_column
        for original_column, canonical_column
        in zip(
            measure_columns,
            canonical_measure_columns,
            strict=True,
        )
        if canonical_column not in exclusion_set
    )

    if len(target_columns) != EXPECTED_MRI_TARGET_COUNT:
        raise MRICatalogError(
            "Unexpected supervised MRI target catalog size. "
            f"Observed {len(target_columns)} targets; "
            f"expected {EXPECTED_MRI_TARGET_COUNT}."
        )

    n_non_measure_columns = (
        len(source_columns)
        - len(measure_columns)
    )

    if (
        n_non_measure_columns
        + len(measure_columns)
        != len(source_columns)
    ):
        raise MRICatalogError(
            "UCSFFSX7 column accounting failed."
        )

    measure_catalog_sha256 = (
        calculate_column_catalog_sha256(
            measure_columns
        )
    )

    target_catalog_sha256 = (
        calculate_column_catalog_sha256(
            target_columns
        )
    )

    summary = MRICatalogSummary(
        source_name="UCSFFSX7",
        n_source_columns=len(source_columns),
        n_non_measure_columns=n_non_measure_columns,
        n_measure_columns=len(measure_columns),
        n_target_columns=len(target_columns),
        excluded_targets=FIXED_EXCLUDED_MRI_TARGETS,
        measure_catalog_sha256=measure_catalog_sha256,
        target_catalog_sha256=target_catalog_sha256,
    )

    return (
        measure_columns,
        target_columns,
        summary,
    )