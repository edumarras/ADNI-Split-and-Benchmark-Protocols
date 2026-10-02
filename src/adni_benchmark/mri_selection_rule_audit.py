from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from adni_benchmark.mri_catalog import (
    FIXED_EXCLUDED_MRI_TARGETS,
)
from adni_benchmark.mri_selection_audit import (
    MRI_AUDIT_MISSING_LABEL,
    MRI_REGIONAL_QC_COLUMNS,
    canonicalize_audit_series,
)


class MRISelectionRuleAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRISelectionRuleAuditSummary:
    source_name: str
    n_rows: int
    rows_target_complete_but_measure_incomplete: int

    status_target_completeness_counts: tuple[
        tuple[str, bool, int],
        ...,
    ]

    status_qc_availability_counts: tuple[
        tuple[str, bool, int, int],
        ...,
    ]

    excluded_measure_missing_counts: tuple[
        tuple[str, int, int],
        ...,
    ]

    duplicate_visit_groups: int

    duplicate_groups_with_status_disagreement: int
    duplicate_groups_with_complete_status_qc_disagreement: int
    duplicate_groups_with_complete_status_regional_fail_disagreement: int

    duplicate_groups_with_measure_nonmissing_disagreement: int
    duplicate_groups_with_target_nonmissing_disagreement: int
    duplicate_groups_with_mixed_target_completeness: int

    duplicate_groups_with_at_least_one_target_complete: int
    duplicate_groups_with_multiple_target_complete_candidates: int
    duplicate_groups_without_target_complete_candidate: int

    duplicate_groups_with_complete_status_candidate: int
    duplicate_groups_with_status_target_tension: int
    duplicate_groups_where_both_statuses_offer_target_complete: int


def build_mri_selection_rule_audit(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
    target_columns: Sequence[str],
) -> MRISelectionRuleAuditSummary:
    # Validate the fields required by this diagnostic stage
    required_columns = {
        "RID",
        "VISCODE2",
        "STATUS",
        "OVERALLQC",
        *MRI_REGIONAL_QC_COLUMNS,
        *measure_columns,
        *target_columns,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise MRISelectionRuleAuditError(
            "UCSFFSX7 is missing columns required for the "
            f"MRI rule audit: {missing_columns}"
        )

    if not measure_columns:
        raise MRISelectionRuleAuditError(
            "The MRI measurement catalog is empty."
        )

    if not target_columns:
        raise MRISelectionRuleAuditError(
            "The MRI target catalog is empty."
        )

    # Recover the original source spelling of the two excluded measurements
    canonical_measure_map = {
        str(column).strip().upper(): str(column)
        for column in measure_columns
    }

    excluded_measure_columns: list[str] = []

    for excluded_name in FIXED_EXCLUDED_MRI_TARGETS:
        canonical_name = (
            excluded_name
            .strip()
            .upper()
        )

        if canonical_name not in canonical_measure_map:
            raise MRISelectionRuleAuditError(
                "Fixed excluded MRI measurement is absent "
                f"from the measurement catalog: {excluded_name}"
            )

        excluded_measure_columns.append(
            canonical_measure_map[
                canonical_name
            ]
        )

    work = frame.copy()

    # Canonical technical fields
    work["__status"] = canonicalize_audit_series(
        work["STATUS"]
    )

    work["__overall_qc"] = canonicalize_audit_series(
        work["OVERALLQC"]
    )

    regional_qc_frame = pd.DataFrame(
        {
            column: canonicalize_audit_series(
                work[column]
            )
            for column in MRI_REGIONAL_QC_COLUMNS
        },
        index=work.index,
    )

    # QC availability and explicit failures
    work["__overall_qc_present"] = (
        work["__overall_qc"].ne(
            MRI_AUDIT_MISSING_LABEL
        )
    )

    work["__regional_qc_present_count"] = (
        regional_qc_frame
        .ne(
            MRI_AUDIT_MISSING_LABEL
        )
        .sum(axis=1)
        .astype("int64")
    )

    work["__regional_qc_fail_count"] = (
        regional_qc_frame
        .eq("FAIL")
        .sum(axis=1)
        .astype("int64")
    )

    # Measure and target completeness
    work["__measure_nonmissing_count"] = (
        work.loc[
            :,
            list(measure_columns),
        ]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    work["__target_nonmissing_count"] = (
        work.loc[
            :,
            list(target_columns),
        ]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    work["__target_complete"] = (
        work["__target_nonmissing_count"]
        .eq(
            len(target_columns)
        )
    )

    work["__measure_complete"] = (
        work["__measure_nonmissing_count"]
        .eq(
            len(measure_columns)
        )
    )

    # Global STATUS versus target-completeness table
    status_target_counter = (
        work.groupby(
            [
                "__status",
                "__target_complete",
            ],
            dropna=False,
            sort=True,
        )
        .size()
    )

    status_target_completeness_counts = tuple(
        (
            str(status),
            bool(is_target_complete),
            int(count),
        )
        for (
            status,
            is_target_complete,
        ), count in status_target_counter.items()
    )

    # Global STATUS versus QC-availability table
    status_qc_counter = (
        work.groupby(
            [
                "__status",
                "__overall_qc_present",
                "__regional_qc_present_count",
            ],
            dropna=False,
            sort=True,
        )
        .size()
    )

    status_qc_availability_counts = tuple(
        (
            str(status),
            bool(overall_qc_present),
            int(regional_qc_present_count),
            int(count),
        )
        for (
            status,
            overall_qc_present,
            regional_qc_present_count,
        ), count in status_qc_counter.items()
    )

    # Diagnose why 325-measure completeness is much rarer than 323-target completeness
    excluded_measure_missing_counts = tuple(
        (
            column,
            int(
                work[column].isna().sum()
            ),
            int(
                work[column].notna().sum()
            ),
        )
        for column in excluded_measure_columns
    )

    rows_target_complete_but_measure_incomplete = int(
        (
            work["__target_complete"]
            & ~work["__measure_complete"]
        ).sum()
    )

    # Duplicate-group diagnostics only use complete visit keys
    valid_visit_key_mask = (
        work.loc[
            :,
            [
                "RID",
                "VISCODE2",
            ],
        ]
        .notna()
        .all(axis=1)
    )

    valid_work = work.loc[
        valid_visit_key_mask
    ].copy()

    duplicated_visit_mask = (
        valid_work.duplicated(
            subset=[
                "RID",
                "VISCODE2",
            ],
            keep=False,
        )
    )

    duplicate_work = valid_work.loc[
        duplicated_visit_mask
    ].copy()

    duplicate_groups = duplicate_work.groupby(
        [
            "RID",
            "VISCODE2",
        ],
        dropna=False,
        sort=False,
    )

    duplicate_visit_groups = int(
        duplicate_groups.ngroups
    )

    counters = Counter()

    for _, group in duplicate_groups:
        statuses = set(
            group["__status"].tolist()
        )

        target_complete_count = int(
            group["__target_complete"].sum()
        )

        if len(statuses) > 1:
            counters[
                "status_disagreement"
            ] += 1

        complete_status_group = group.loc[
            group["__status"].eq(
                "COMPLETE"
            )
        ]

        partial_status_group = group.loc[
            group["__status"].eq(
                "PARTIAL"
            )
        ]

        if not complete_status_group.empty:
            counters[
                "complete_status_candidate"
            ] += 1

        if (
            len(complete_status_group) > 1
            and complete_status_group[
                "__overall_qc"
            ].nunique(
                dropna=False
            ) > 1
        ):
            counters[
                "complete_status_qc_disagreement"
            ] += 1

        if (
            len(complete_status_group) > 1
            and complete_status_group[
                "__regional_qc_fail_count"
            ].nunique(
                dropna=False
            ) > 1
        ):
            counters[
                "complete_status_regional_fail_disagreement"
            ] += 1

        if group[
            "__measure_nonmissing_count"
        ].nunique(
            dropna=False
        ) > 1:
            counters[
                "measure_nonmissing_disagreement"
            ] += 1

        if group[
            "__target_nonmissing_count"
        ].nunique(
            dropna=False
        ) > 1:
            counters[
                "target_nonmissing_disagreement"
            ] += 1

        if group[
            "__target_complete"
        ].nunique(
            dropna=False
        ) > 1:
            counters[
                "mixed_target_completeness"
            ] += 1

        if target_complete_count >= 1:
            counters[
                "at_least_one_target_complete"
            ] += 1

        if target_complete_count >= 2:
            counters[
                "multiple_target_complete"
            ] += 1

        if target_complete_count == 0:
            counters[
                "no_target_complete"
            ] += 1

        complete_status_has_target_complete = (
            not complete_status_group.empty
            and bool(
                complete_status_group[
                    "__target_complete"
                ].any()
            )
        )

        partial_status_has_target_complete = (
            not partial_status_group.empty
            and bool(
                partial_status_group[
                    "__target_complete"
                ].any()
            )
        )

        # COMPLETE exists, PARTIAL has complete targets, but COMPLETE does not
        if (
            not complete_status_group.empty
            and not partial_status_group.empty
            and partial_status_has_target_complete
            and not complete_status_has_target_complete
        ):
            counters[
                "status_target_tension"
            ] += 1

        # Both STATUS categories provide at least one target-complete candidate
        if (
            complete_status_has_target_complete
            and partial_status_has_target_complete
        ):
            counters[
                "both_statuses_offer_target_complete"
            ] += 1

    summary = MRISelectionRuleAuditSummary(
        source_name="UCSFFSX7",
        n_rows=int(
            len(work)
        ),
        rows_target_complete_but_measure_incomplete=(
            rows_target_complete_but_measure_incomplete
        ),
        status_target_completeness_counts=(
            status_target_completeness_counts
        ),
        status_qc_availability_counts=(
            status_qc_availability_counts
        ),
        excluded_measure_missing_counts=(
            excluded_measure_missing_counts
        ),
        duplicate_visit_groups=duplicate_visit_groups,
        duplicate_groups_with_status_disagreement=int(
            counters["status_disagreement"]
        ),
        duplicate_groups_with_complete_status_qc_disagreement=int(
            counters[
                "complete_status_qc_disagreement"
            ]
        ),
        duplicate_groups_with_complete_status_regional_fail_disagreement=int(
            counters[
                "complete_status_regional_fail_disagreement"
            ]
        ),
        duplicate_groups_with_measure_nonmissing_disagreement=int(
            counters[
                "measure_nonmissing_disagreement"
            ]
        ),
        duplicate_groups_with_target_nonmissing_disagreement=int(
            counters[
                "target_nonmissing_disagreement"
            ]
        ),
        duplicate_groups_with_mixed_target_completeness=int(
            counters[
                "mixed_target_completeness"
            ]
        ),
        duplicate_groups_with_at_least_one_target_complete=int(
            counters[
                "at_least_one_target_complete"
            ]
        ),
        duplicate_groups_with_multiple_target_complete_candidates=int(
            counters[
                "multiple_target_complete"
            ]
        ),
        duplicate_groups_without_target_complete_candidate=int(
            counters[
                "no_target_complete"
            ]
        ),
        duplicate_groups_with_complete_status_candidate=int(
            counters[
                "complete_status_candidate"
            ]
        ),
        duplicate_groups_with_status_target_tension=int(
            counters[
                "status_target_tension"
            ]
        ),
        duplicate_groups_where_both_statuses_offer_target_complete=int(
            counters[
                "both_statuses_offer_target_complete"
            ]
        ),
    )

    if (
        summary.duplicate_visit_groups
        != (
            summary.duplicate_groups_with_at_least_one_target_complete
            + summary.duplicate_groups_without_target_complete_candidate
        )
    ):
        raise MRISelectionRuleAuditError(
            "MRI duplicate-group target-completeness "
            "accounting failed."
        )

    return summary