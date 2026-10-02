from __future__ import annotations

import hashlib
import logging
import re

from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
    SourceContract,
)


ADMIN_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "PHASE",
        "PTID",
        "RID",
        "VISCODE",
        "VISCODE2",
        "VISDATE",
        "EXAMDATE",
        "RUNDATE",
        "APTESTDT",
        "ID",
        "SITEID",
        "USERDATE",
        "USERDATE2",
        "UPDATE_STAMP",
        "update_stamp",
        "DD_CRF_VERSION_LABEL",
        "LANGUAGE_CODE",
        "HAS_QC_ERROR",
        "SPID",
    }
)


NON_MRI_SOURCE_TOKENS: Final[tuple[str, ...]] = (
    "DXSUM",
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "PTDEMOG",
    "APOERES",
)


RESOLUTION_STATUSES: Final[tuple[str, ...]] = (
    "unique_record",
    "exact_or_equivalent_duplicate",
    "dominant_record",
    "unresolved_conflict",
)


class NonMRIResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceResolutionSummary:
    source_name: str
    original_rows: int
    valid_key_rows: int
    rows_with_missing_key: int
    valid_key_groups: int
    duplicate_groups: int
    resolved_rows: int
    unresolved_groups: int
    status_counts: tuple[
        tuple[str, int],
        ...,
    ]


def calculate_text_sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def canonical_scalar(
    value: Any,
) -> str:
    if pd.isna(value):
        return "<NA>"

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return str(value).strip()


def canonical_row_hash(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.Series:
    # Column order must not affect deterministic row selection
    ordered_columns = sorted(
        columns
    )

    return frame.loc[
        :,
        ordered_columns,
    ].apply(
        lambda row: calculate_text_sha256(
            "\x1f".join(
                f"{column}={canonical_scalar(row[column])}"
                for column in ordered_columns
            )
        ),
        axis=1,
    )


def values_equal(
    left: Any,
    right: Any,
) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True

    if pd.isna(left) or pd.isna(right):
        return False

    try:
        left_numeric = float(left)
        right_numeric = float(right)

        if (
            np.isfinite(left_numeric)
            and np.isfinite(right_numeric)
        ):
            return bool(
                np.isclose(
                    left_numeric,
                    right_numeric,
                    rtol=0.0,
                    atol=0.0,
                )
            )

    except (
        TypeError,
        ValueError,
    ):
        pass

    return (
        str(left).strip()
        == str(right).strip()
    )


def rows_are_equivalent(
    left: pd.Series,
    right: pd.Series,
    comparison_columns: Sequence[str],
) -> bool:
    return all(
        values_equal(
            left[column],
            right[column],
        )
        for column in comparison_columns
    )


def row_dominates(
    candidate: pd.Series,
    other: pd.Series,
    comparison_columns: Sequence[str],
) -> bool:
    # A dominant row must preserve every observed value from the other row
    # and contain at least one additional observed value
    has_extra_information = False

    for column in comparison_columns:
        candidate_value = candidate[column]
        other_value = other[column]

        candidate_missing = pd.isna(
            candidate_value
        )

        other_missing = pd.isna(
            other_value
        )

        if not other_missing:
            if (
                candidate_missing
                or not values_equal(
                    candidate_value,
                    other_value,
                )
            ):
                return False

        elif not candidate_missing:
            has_extra_information = True

    return has_extra_information


def resolve_group_by_equivalence_or_dominance(
    group: pd.DataFrame,
    comparison_columns: Sequence[str],
) -> tuple[
    pd.Series | None,
    str,
]:
    if group.empty:
        raise NonMRIResolutionError(
            "Cannot resolve an empty non-MRI group."
        )

    if len(group) == 1:
        return (
            group.iloc[0],
            "unique_record",
        )

    first_row = group.iloc[0]

    all_equivalent = all(
        rows_are_equivalent(
            first_row,
            group.iloc[position],
            comparison_columns,
        )
        for position in range(
            1,
            len(group),
        )
    )

    if all_equivalent:
        chosen = (
            group
            .sort_values(
                "__row_hash",
                kind="stable",
            )
            .iloc[0]
        )

        return (
            chosen,
            "exact_or_equivalent_duplicate",
        )

    dominant_indices: list[Any] = []

    for candidate_index, candidate_row in group.iterrows():
        candidate_dominates_group = all(
            candidate_index == other_index
            or row_dominates(
                candidate_row,
                other_row,
                comparison_columns,
            )
            or rows_are_equivalent(
                candidate_row,
                other_row,
                comparison_columns,
            )
            for other_index, other_row in group.iterrows()
        )

        if candidate_dominates_group:
            dominant_indices.append(
                candidate_index
            )

    if len(dominant_indices) == 1:
        return (
            group.loc[
                dominant_indices[0]
            ],
            "dominant_record",
        )

    return (
        None,
        "unresolved_conflict",
    )


def parse_protocol_month(
    value: Any,
) -> float:
    if pd.isna(value):
        return np.nan

    text = (
        str(value)
        .strip()
        .casefold()
    )

    if text in {
        "bl",
        "sc",
        "scmri",
        "m00",
        "v01",
    }:
        return 0.0

    month_match = re.fullmatch(
        r"m(\d+)",
        text,
    )

    if month_match:
        return float(
            int(
                month_match.group(1)
            )
        )

    year_match = re.fullmatch(
        r"y(\d+)",
        text,
    )

    if year_match:
        return float(
            int(
                year_match.group(1)
            )
            * 12
        )

    return np.nan


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def comparison_columns_for_source(
    frame: pd.DataFrame,
    contract: SourceContract,
) -> list[str]:
    # APOERES is intentionally resolved only through GENOTYPE
    if contract.source_name == "APOERES":
        if "GENOTYPE" not in frame.columns:
            raise NonMRIResolutionError(
                "APOERES is missing the required GENOTYPE column."
            )

        comparison_columns = [
            "GENOTYPE",
        ]

    else:
        excluded_columns = (
            set(
                contract.key_columns
            )
            | set(
                ADMIN_COLUMNS
            )
            | {
                "__source_row",
                "__row_hash",
                "__resolution_status",
            }
        )

        comparison_columns = [
            column
            for column in frame.columns
            if column not in excluded_columns
        ]

    if not comparison_columns:
        raise NonMRIResolutionError(
            f"{contract.source_name} has no substantive columns "
            "available for record resolution."
        )

    missing_comparison_columns = [
        column
        for column in comparison_columns
        if column not in frame.columns
    ]

    if missing_comparison_columns:
        raise NonMRIResolutionError(
            f"{contract.source_name} comparison columns are missing: "
            f"{missing_comparison_columns}"
        )

    return comparison_columns


def resolve_generic_source(
    frame: pd.DataFrame,
    contract: SourceContract,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    SourceResolutionSummary,
]:
    missing_key_columns = [
        column
        for column in contract.key_columns
        if column not in frame.columns
    ]

    if missing_key_columns:
        raise NonMRIResolutionError(
            f"{contract.source_name} is missing key columns: "
            f"{missing_key_columns}"
        )

    key_columns = list(
        contract.key_columns
    )

    # Preserve the original parsed source-row position
    work = frame.copy()

    work["__source_row"] = np.arange(
        len(work),
        dtype=np.int64,
    )

    valid_key_mask = (
        work.loc[
            :,
            key_columns,
        ]
        .notna()
        .all(axis=1)
    )

    valid_work = work.loc[
        valid_key_mask
    ].copy()

    hash_columns = [
        column
        for column in valid_work.columns
        if not column.startswith("__")
    ]

    if not hash_columns:
        raise NonMRIResolutionError(
            f"{contract.source_name} has no real source columns "
            "available for row hashing."
        )

    valid_work["__row_hash"] = canonical_row_hash(
        frame=valid_work,
        columns=hash_columns,
    )

    comparison_columns = comparison_columns_for_source(
        frame=valid_work,
        contract=contract,
    )

    selected_rows: list[
        pd.Series
    ] = []

    unresolved_records: list[
        dict[str, Any]
    ] = []

    duplicate_groups = 0

    groupby_columns: str | list[str]

    if len(key_columns) == 1:
        groupby_columns = (
            key_columns[0]
        )
    else:
        groupby_columns = (
            key_columns
        )

    grouped = valid_work.groupby(
        groupby_columns,
        dropna=False,
        sort=False,
    )

    for key, group in grouped:
        if len(group) > 1:
            duplicate_groups += 1

        candidate_group = group

        # PTDEMOG-like participant-baseline sources are first restricted
        # to the earliest interpretable protocol wave
        if (
            contract.grain
            == "participant_baseline"
            and "VISCODE2" in group.columns
        ):
            protocol_months = group[
                "VISCODE2"
            ].map(
                parse_protocol_month
            )

            if protocol_months.notna().any():
                earliest_month = float(
                    protocol_months.min()
                )

                candidate_group = group.loc[
                    protocol_months.eq(
                        earliest_month
                    )
                ]

            else:
                # Historical fallback when no protocol wave is interpretable
                date_column = first_existing_column(
                    frame=group,
                    candidates=(
                        "VISDATE",
                        "EXAMDATE",
                    ),
                )

                if date_column is not None:
                    parsed_dates = pd.to_datetime(
                        group[
                            date_column
                        ],
                        errors="coerce",
                    )

                    if parsed_dates.notna().any():
                        earliest_date = (
                            parsed_dates.min()
                        )

                        candidate_group = group.loc[
                            parsed_dates.eq(
                                earliest_date
                            )
                        ]

        if candidate_group.empty:
            raise NonMRIResolutionError(
                f"{contract.source_name} produced an empty candidate "
                "group during record resolution."
            )

        (
            chosen_row,
            resolution_status,
        ) = resolve_group_by_equivalence_or_dominance(
            group=candidate_group,
            comparison_columns=comparison_columns,
        )

        if chosen_row is None:
            if isinstance(
                key,
                tuple,
            ):
                key_values = key

            else:
                key_values = (
                    key,
                )

            if (
                len(key_values)
                != len(key_columns)
            ):
                raise NonMRIResolutionError(
                    f"{contract.source_name} group key length does "
                    "not match its contract."
                )

            # LOCAL_ONLY because key values may identify a participant/visit
            unresolved_record = {
                column: key_values[
                    position
                ]
                for (
                    position,
                    column,
                ) in enumerate(
                    key_columns
                )
            }

            unresolved_record.update(
                {
                    "source_name": (
                        contract.source_name
                    ),
                    "n_source_rows": int(
                        len(group)
                    ),
                    "n_candidate_rows": int(
                        len(candidate_group)
                    ),
                    "resolution_status": (
                        resolution_status
                    ),
                }
            )

            unresolved_records.append(
                unresolved_record
            )

        else:
            selected_row = (
                chosen_row.copy()
            )

            selected_row[
                "__resolution_status"
            ] = resolution_status

            selected_rows.append(
                selected_row
            )

    if selected_rows:
        resolved_frame = (
            pd.DataFrame(
                selected_rows
            )
            .reset_index(
                drop=True
            )
        )

    else:
        resolved_frame = (
            valid_work
            .head(0)
            .copy()
        )

        resolved_frame[
            "__resolution_status"
        ] = pd.Series(
            dtype="string"
        )

    if unresolved_records:
        unresolved_frame = pd.DataFrame(
            unresolved_records
        )

    else:
        unresolved_frame = pd.DataFrame(
            columns=[
                *key_columns,
                "source_name",
                "n_source_rows",
                "n_candidate_rows",
                "resolution_status",
            ]
        )

    status_counter = Counter(
        resolved_frame[
            "__resolution_status"
        ]
        .dropna()
        .astype("string")
        .tolist()
    )

    status_counter[
        "unresolved_conflict"
    ] = int(
        len(
            unresolved_frame
        )
    )

    status_counts = tuple(
        (
            status,
            int(
                status_counter.get(
                    status,
                    0,
                )
            ),
        )
        for status in RESOLUTION_STATUSES
    )

    summary = SourceResolutionSummary(
        source_name=(
            contract.source_name
        ),
        original_rows=int(
            len(frame)
        ),
        valid_key_rows=int(
            valid_key_mask.sum()
        ),
        rows_with_missing_key=int(
            (
                ~valid_key_mask
            ).sum()
        ),
        valid_key_groups=int(
            grouped.ngroups
        ),
        duplicate_groups=int(
            duplicate_groups
        ),
        resolved_rows=int(
            len(
                resolved_frame
            )
        ),
        unresolved_groups=int(
            len(
                unresolved_frame
            )
        ),
        status_counts=status_counts,
    )

    if (
        summary.valid_key_rows
        + summary.rows_with_missing_key
        != summary.original_rows
    ):
        raise NonMRIResolutionError(
            f"{contract.source_name} key-validity accounting failed."
        )

    if (
        summary.resolved_rows
        + summary.unresolved_groups
        != summary.valid_key_groups
    ):
        raise NonMRIResolutionError(
            f"{contract.source_name} resolution accounting failed."
        )

    if (
        sum(
            count
            for _, count
            in summary.status_counts
        )
        != summary.valid_key_groups
    ):
        raise NonMRIResolutionError(
            f"{contract.source_name} resolution-status "
            "accounting failed."
        )

    duplicated_resolved_keys = (
        resolved_frame.duplicated(
            subset=key_columns,
            keep=False,
        )
    )

    if duplicated_resolved_keys.any():
        raise NonMRIResolutionError(
            f"{contract.source_name} resolution produced "
            "duplicated output keys."
        )

    return (
        resolved_frame,
        unresolved_frame,
        summary,
    )


def resolve_all_non_mri_sources(
    source_tables: dict[
        str,
        pd.DataFrame,
    ],
) -> tuple[
    dict[
        str,
        pd.DataFrame,
    ],
    dict[
        str,
        pd.DataFrame,
    ],
    dict[
        str,
        SourceResolutionSummary,
    ],
]:
    missing_sources = [
        source_name
        for source_name
        in NON_MRI_SOURCE_TOKENS
        if source_name
        not in source_tables
    ]

    if missing_sources:
        raise NonMRIResolutionError(
            "The following non-MRI sources were not loaded: "
            f"{missing_sources}"
        )

    resolved_sources: dict[
        str,
        pd.DataFrame,
    ] = {}

    unresolved_sources: dict[
        str,
        pd.DataFrame,
    ] = {}

    resolution_summaries: dict[
        str,
        SourceResolutionSummary,
    ] = {}

    for source_name in NON_MRI_SOURCE_TOKENS:
        logging.info(
            "Resolving %s...",
            source_name,
        )

        frame = (
            source_tables[
                source_name
            ]
        )

        contract = (
            SOURCE_CONTRACTS[
                source_name
            ]
        )

        (
            resolved_frame,
            unresolved_frame,
            summary,
        ) = resolve_generic_source(
            frame=frame,
            contract=contract,
        )

        if (
            len(
                resolved_frame
            )
            != summary.resolved_rows
        ):
            raise NonMRIResolutionError(
                f"{source_name} resolved-row count mismatch."
            )

        if (
            len(
                unresolved_frame
            )
            != summary.unresolved_groups
        ):
            raise NonMRIResolutionError(
                f"{source_name} unresolved-group count mismatch."
            )

        resolved_sources[
            source_name
        ] = resolved_frame

        unresolved_sources[
            source_name
        ] = unresolved_frame

        resolution_summaries[
            source_name
        ] = summary

    return (
        resolved_sources,
        unresolved_sources,
        resolution_summaries,
    )