from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Sequence

import pandas as pd


MRI_REGIONAL_QC_COLUMNS: Final[tuple[str, ...]] = (
    "TEMPQC",
    "FRONTQC",
    "PARQC",
    "INSULAQC",
    "OCCQC",
    "BGQC",
    "CWMQC",
    "VENTQC",
    "HIPPOQC",
)

MRI_SELECTION_AUDIT_COLUMNS: Final[tuple[str, ...]] = (
    "PHASE",
    "FIELD_STRENGTH",
    "FSVER",
    "STATUS",
    "OVERALLQC",
    *MRI_REGIONAL_QC_COLUMNS,
)

MRI_AUDIT_MISSING_LABEL: Final[str] = "<MISSING>"


class MRISelectionAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRISelectionAuditSummary:
    source_name: str
    n_rows: int
    rows_with_missing_visit_key: int
    rows_with_missing_imageuid: int
    valid_visit_rows: int
    valid_visit_groups: int
    duplicate_visit_groups: int
    max_candidates_per_visit: int
    candidate_count_distribution: tuple[
        tuple[int, int],
        ...,
    ]
    groups_with_multiple_phase_values: int
    groups_with_multiple_fsver_values: int
    groups_with_multiple_field_strength_values: int
    rows_complete_measure_catalog: int
    rows_complete_target_catalog: int
    measure_nonmissing_min: int
    measure_nonmissing_median: float
    measure_nonmissing_max: int
    field_value_counts: tuple[
        tuple[
            str,
            tuple[
                tuple[str, int],
                ...,
            ],
        ],
        ...,
    ]


def canonicalize_audit_series(
    series: pd.Series,
) -> pd.Series:
    # Normalize technical categorical values for aggregate comparison
    canonical = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return canonical.fillna(
        MRI_AUDIT_MISSING_LABEL
    )


def count_visit_groups_with_multiple_values(
    frame: pd.DataFrame,
    column: str,
) -> int:
    required_columns = [
        "RID",
        "VISCODE2",
        column,
    ]

    missing_columns = [
        required_column
        for required_column in required_columns
        if required_column not in frame.columns
    ]

    if missing_columns:
        raise MRISelectionAuditError(
            "Cannot evaluate MRI visit-group consistency. "
            f"Missing columns: {missing_columns}"
        )

    work = frame.loc[
        :,
        required_columns,
    ].copy()

    work["__canonical_value"] = (
        canonicalize_audit_series(
            work[column]
        )
    )

    # Missing values do not count as a conflicting observed category
    work = work.loc[
        work["__canonical_value"].ne(
            MRI_AUDIT_MISSING_LABEL
        )
    ]

    if work.empty:
        return 0

    distinct_counts = (
        work.groupby(
            [
                "RID",
                "VISCODE2",
            ],
            dropna=False,
            sort=False,
        )["__canonical_value"]
        .nunique(
            dropna=True
        )
    )

    return int(
        distinct_counts.gt(1).sum()
    )


def build_mri_selection_audit(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
    target_columns: Sequence[str],
) -> MRISelectionAuditSummary:
    required_columns = {
        "RID",
        "VISCODE2",
        "IMAGEUID",
        *MRI_SELECTION_AUDIT_COLUMNS,
        *measure_columns,
        *target_columns,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise MRISelectionAuditError(
            "UCSFFSX7 is missing columns required for the "
            f"MRI selection audit: {missing_columns}"
        )

    canonical_measure_names = [
        str(column).strip().upper()
        for column in measure_columns
    ]

    canonical_target_names = [
        str(column).strip().upper()
        for column in target_columns
    ]

    if len(
        set(canonical_measure_names)
    ) != len(canonical_measure_names):
        raise MRISelectionAuditError(
            "The MRI measure catalog contains duplicate names."
        )

    if len(
        set(canonical_target_names)
    ) != len(canonical_target_names):
        raise MRISelectionAuditError(
            "The MRI target catalog contains duplicate names."
        )

    if not set(
        canonical_target_names
    ).issubset(
        set(canonical_measure_names)
    ):
        raise MRISelectionAuditError(
            "The MRI target catalog is not a subset of the "
            "MRI measurement catalog."
        )

    visit_key_columns = [
        "RID",
        "VISCODE2",
    ]

    valid_visit_key_mask = (
        frame.loc[
            :,
            visit_key_columns,
        ]
        .notna()
        .all(axis=1)
    )

    valid_visit_frame = frame.loc[
        valid_visit_key_mask
    ].copy()

    visit_group_sizes = (
        valid_visit_frame.groupby(
            visit_key_columns,
            dropna=False,
            sort=False,
        )
        .size()
    )

    candidate_distribution_counter = Counter(
        int(value)
        for value in visit_group_sizes.tolist()
    )

    candidate_count_distribution = tuple(
        (
            candidate_count,
            int(
                candidate_distribution_counter[
                    candidate_count
                ]
            ),
        )
        for candidate_count in sorted(
            candidate_distribution_counter
        )
    )

    duplicate_visit_groups = int(
        visit_group_sizes.gt(1).sum()
    )

    max_candidates_per_visit = (
        int(
            visit_group_sizes.max()
        )
        if not visit_group_sizes.empty
        else 0
    )

    measure_nonmissing_counts = (
        frame.loc[
            :,
            list(measure_columns),
        ]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    target_nonmissing_counts = (
        frame.loc[
            :,
            list(target_columns),
        ]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    field_value_counts_records: list[
        tuple[
            str,
            tuple[
                tuple[str, int],
                ...,
            ],
        ]
    ] = []

    for column in MRI_SELECTION_AUDIT_COLUMNS:
        canonical_values = canonicalize_audit_series(
            frame[column]
        )

        counts = Counter(
            str(value)
            for value in canonical_values.tolist()
        )

        ordered_counts = tuple(
            (
                value,
                int(counts[value]),
            )
            for value in sorted(counts)
        )

        field_value_counts_records.append(
            (
                column,
                ordered_counts,
            )
        )

    summary = MRISelectionAuditSummary(
        source_name="UCSFFSX7",
        n_rows=int(
            len(frame)
        ),
        rows_with_missing_visit_key=int(
            (~valid_visit_key_mask).sum()
        ),
        rows_with_missing_imageuid=int(
            frame["IMAGEUID"].isna().sum()
        ),
        valid_visit_rows=int(
            valid_visit_key_mask.sum()
        ),
        valid_visit_groups=int(
            len(visit_group_sizes)
        ),
        duplicate_visit_groups=duplicate_visit_groups,
        max_candidates_per_visit=max_candidates_per_visit,
        candidate_count_distribution=(
            candidate_count_distribution
        ),
        groups_with_multiple_phase_values=(
            count_visit_groups_with_multiple_values(
                frame=valid_visit_frame,
                column="PHASE",
            )
        ),
        groups_with_multiple_fsver_values=(
            count_visit_groups_with_multiple_values(
                frame=valid_visit_frame,
                column="FSVER",
            )
        ),
        groups_with_multiple_field_strength_values=(
            count_visit_groups_with_multiple_values(
                frame=valid_visit_frame,
                column="FIELD_STRENGTH",
            )
        ),
        rows_complete_measure_catalog=int(
            measure_nonmissing_counts.eq(
                len(measure_columns)
            ).sum()
        ),
        rows_complete_target_catalog=int(
            target_nonmissing_counts.eq(
                len(target_columns)
            ).sum()
        ),
        measure_nonmissing_min=int(
            measure_nonmissing_counts.min()
        ),
        measure_nonmissing_median=float(
            measure_nonmissing_counts.median()
        ),
        measure_nonmissing_max=int(
            measure_nonmissing_counts.max()
        ),
        field_value_counts=tuple(
            field_value_counts_records
        ),
    )

    if (
        summary.valid_visit_rows
        + summary.rows_with_missing_visit_key
        != summary.n_rows
    ):
        raise MRISelectionAuditError(
            "MRI visit-key accounting failed."
        )

    if (
        sum(
            group_count
            for _, group_count
            in summary.candidate_count_distribution
        )
        != summary.valid_visit_groups
    ):
        raise MRISelectionAuditError(
            "MRI candidate-count distribution accounting failed."
        )

    return summary