from __future__ import annotations

import hashlib

from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from adni_benchmark.mri_secondary_tie_audit import (
    convert_mri_targets_to_numeric,
    mri_target_vectors_are_equivalent,
)


MRI_RESOLUTION_STATUSES: Final[tuple[str, ...]] = (
    "unique_candidate",
    "unique_maximum_target_completeness",
    "equivalent_maximum_target_candidates",
    "unresolved_maximum_target_conflict",
)


class MRIResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRIResolutionSummary:
    source_name: str
    original_rows: int
    valid_key_rows: int
    rows_with_missing_key: int
    rows_with_missing_imageuid: int
    valid_visit_groups: int
    duplicate_visit_groups: int
    resolved_rows: int
    unresolved_groups: int
    selected_target_complete_rows: int
    selected_target_incomplete_rows: int
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
    # Column order must not affect the row fingerprint
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


def resolve_mri_visit_group(
    group: pd.DataFrame,
    numeric_targets: pd.DataFrame,
) -> tuple[
    pd.Series | None,
    str,
    int,
    int,
]:
    if group.empty:
        raise MRIResolutionError(
            "Cannot resolve an empty MRI visit group."
        )

    # A unique source row requires no competition
    if len(group) == 1:
        return (
            group.iloc[0],
            "unique_candidate",
            1,
            int(
                group.iloc[0][
                    "__target_nonmissing_count"
                ]
            ),
        )

    # Primary rule: maximize availability across the fixed target catalog
    maximum_target_count = int(
        group[
            "__target_nonmissing_count"
        ].max()
    )

    best_candidates = group.loc[
        group[
            "__target_nonmissing_count"
        ].eq(
            maximum_target_count
        )
    ].copy()

    if best_candidates.empty:
        raise MRIResolutionError(
            "MRI resolution produced an empty maximum-"
            "completeness candidate set."
        )

    # A unique maximum-completeness candidate is selected
    if len(best_candidates) == 1:
        return (
            best_candidates.iloc[0],
            "unique_maximum_target_completeness",
            1,
            maximum_target_count,
        )

    # If maximum-completeness candidates remain tied,
    # they must be numerically equivalent across all targets
    best_numeric_targets = numeric_targets.loc[
        best_candidates.index
    ]

    if mri_target_vectors_are_equivalent(
        best_numeric_targets
    ):
        # Equivalent candidates are resolved deterministically
        # by numeric IMAGEUID and then canonical row hash
        chosen = (
            best_candidates
            .sort_values(
                by=[
                    "__imageuid_numeric",
                    "__row_hash",
                ],
                ascending=[
                    True,
                    True,
                ],
                kind="stable",
            )
            .iloc[0]
        )

        return (
            chosen,
            "equivalent_maximum_target_candidates",
            int(
                len(best_candidates)
            ),
            maximum_target_count,
        )

    # Conflicting maximum-completeness target vectors are rejected
    return (
        None,
        "unresolved_maximum_target_conflict",
        int(
            len(best_candidates)
        ),
        maximum_target_count,
    )


def resolve_mri_source(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    MRIResolutionSummary,
]:
    required_columns = {
        "RID",
        "VISCODE2",
        "IMAGEUID",
        *target_columns,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise MRIResolutionError(
            "UCSFFSX7 is missing columns required for "
            f"MRI resolution: {missing_columns}"
        )

    if not target_columns:
        raise MRIResolutionError(
            "The MRI target catalog is empty."
        )

    key_columns = [
        "RID",
        "VISCODE2",
    ]

    work = frame.copy()

    # Retain source-row position only as trace metadata
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

    rows_with_missing_key = int(
        (~valid_key_mask).sum()
    )

    valid_work = work.loc[
        valid_key_mask
    ].copy()

    # Every resolvable MRI row must have IMAGEUID
    rows_with_missing_imageuid = int(
        valid_work[
            "IMAGEUID"
        ].isna().sum()
    )

    if rows_with_missing_imageuid:
        raise MRIResolutionError(
            "UCSFFSX7 contains "
            f"{rows_with_missing_imageuid} rows with a valid "
            "visit key but missing IMAGEUID."
        )

    imageuid_numeric = pd.to_numeric(
        valid_work["IMAGEUID"],
        errors="coerce",
    )

    invalid_imageuid_mask = (
        valid_work["IMAGEUID"].notna()
        & imageuid_numeric.isna()
    )

    if invalid_imageuid_mask.any():
        raise MRIResolutionError(
            "UCSFFSX7 contains observed IMAGEUID values "
            "that cannot be converted to numbers."
        )

    non_integer_imageuid_mask = (
        imageuid_numeric.notna()
        & imageuid_numeric.ne(
            imageuid_numeric.round()
        )
    )

    if non_integer_imageuid_mask.any():
        raise MRIResolutionError(
            "UCSFFSX7 contains non-integer IMAGEUID values."
        )

    valid_work["__imageuid_numeric"] = (
        imageuid_numeric
        .round()
        .astype("Int64")
    )

    numeric_targets = convert_mri_targets_to_numeric(
        frame=valid_work,
        target_columns=target_columns,
    )

    valid_work["__target_nonmissing_count"] = (
        numeric_targets
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    valid_work["__target_complete"] = (
        valid_work[
            "__target_nonmissing_count"
        ].eq(
            len(target_columns)
        )
    )

    # Hash only real source fields, never temporary helper fields
    hash_columns = [
        column
        for column in valid_work.columns
        if not column.startswith("__")
    ]

    valid_work["__row_hash"] = canonical_row_hash(
        frame=valid_work,
        columns=hash_columns,
    )

    grouped = valid_work.groupby(
        key_columns,
        dropna=False,
        sort=False,
    )

    selected_rows: list[
        pd.Series
    ] = []

    unresolved_records: list[
        dict[str, Any]
    ] = []

    duplicate_visit_groups = 0

    for key, group in grouped:
        if len(group) > 1:
            duplicate_visit_groups += 1

        (
            chosen_row,
            resolution_status,
            n_best_candidates,
            maximum_target_count,
        ) = resolve_mri_visit_group(
            group=group,
            numeric_targets=numeric_targets,
        )

        if chosen_row is None:
            if not isinstance(
                key,
                tuple,
            ):
                raise MRIResolutionError(
                    "MRI group key is not a tuple."
                )

            # This table is local-only because it contains visit identifiers
            unresolved_records.append(
                {
                    "RID": key[0],
                    "VISCODE2": key[1],
                    "source_name": "UCSFFSX7",
                    "n_source_rows": int(
                        len(group)
                    ),
                    "n_best_candidates": (
                        n_best_candidates
                    ),
                    "maximum_target_nonmissing_count": (
                        maximum_target_count
                    ),
                    "resolution_status": (
                        resolution_status
                    ),
                }
            )

        else:
            selected_row = chosen_row.copy()

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
        resolved_frame = valid_work.head(
            0
        ).copy()

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
                "RID",
                "VISCODE2",
                "source_name",
                "n_source_rows",
                "n_best_candidates",
                "maximum_target_nonmissing_count",
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
        "unresolved_maximum_target_conflict"
    ] = int(
        len(unresolved_frame)
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
        for status in MRI_RESOLUTION_STATUSES
    )

    selected_target_complete_rows = int(
        resolved_frame[
            "__target_complete"
        ]
        .fillna(
            False
        )
        .sum()
    )

    selected_target_incomplete_rows = int(
        len(resolved_frame)
        - selected_target_complete_rows
    )

    summary = MRIResolutionSummary(
        source_name="UCSFFSX7",
        original_rows=int(
            len(frame)
        ),
        valid_key_rows=int(
            valid_key_mask.sum()
        ),
        rows_with_missing_key=(
            rows_with_missing_key
        ),
        rows_with_missing_imageuid=(
            rows_with_missing_imageuid
        ),
        valid_visit_groups=int(
            grouped.ngroups
        ),
        duplicate_visit_groups=int(
            duplicate_visit_groups
        ),
        resolved_rows=int(
            len(resolved_frame)
        ),
        unresolved_groups=int(
            len(unresolved_frame)
        ),
        selected_target_complete_rows=(
            selected_target_complete_rows
        ),
        selected_target_incomplete_rows=(
            selected_target_incomplete_rows
        ),
        status_counts=status_counts,
    )

    # Internal accounting checks
    if (
        summary.valid_key_rows
        + summary.rows_with_missing_key
        != summary.original_rows
    ):
        raise MRIResolutionError(
            "MRI key-validity accounting failed."
        )

    if (
        summary.resolved_rows
        + summary.unresolved_groups
        != summary.valid_visit_groups
    ):
        raise MRIResolutionError(
            "MRI visit-resolution accounting failed."
        )

    if (
        summary.selected_target_complete_rows
        + summary.selected_target_incomplete_rows
        != summary.resolved_rows
    ):
        raise MRIResolutionError(
            "MRI selected-row completeness accounting failed."
        )

    if (
        sum(
            count
            for _, count in summary.status_counts
        )
        != summary.valid_visit_groups
    ):
        raise MRIResolutionError(
            "MRI resolution-status accounting failed."
        )

    duplicated_resolved_keys = (
        resolved_frame.duplicated(
            subset=key_columns,
            keep=False,
        )
    )

    if duplicated_resolved_keys.any():
        raise MRIResolutionError(
            "MRI resolution produced duplicate visit keys."
        )

    return (
        resolved_frame,
        unresolved_frame,
        summary,
    )