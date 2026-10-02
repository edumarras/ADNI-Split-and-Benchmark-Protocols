from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import pandas as pd

from adni_benchmark.mri_selection_audit import (
    MRI_AUDIT_MISSING_LABEL,
    MRI_REGIONAL_QC_COLUMNS,
    canonicalize_audit_series,
)


MRI_NUMERIC_RTOL: Final[float] = 1e-6
MRI_NUMERIC_ATOL: Final[float] = 1e-8


class MRISecondaryTieAuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class MRISecondaryTieAuditSummary:
    source_name: str

    duplicate_visit_groups: int
    primary_tie_groups: int
    primary_tie_target_complete_groups: int
    primary_tie_target_incomplete_groups: int

    primary_ties_with_measure_completeness_disagreement: int
    primary_ties_with_qc_availability_disagreement: int
    primary_ties_with_status_disagreement: int
    primary_ties_with_overall_qc_disagreement: int
    primary_ties_with_field_strength_disagreement: int

    resolved_by_measure_completeness: int
    resolved_by_qc_availability: int
    resolved_by_unique_overall_pass: int
    resolved_by_fewer_regional_failures: int
    equivalent_after_secondary_filters: int
    remaining_non_equivalent_conflicts: int

    recoverable_target_complete_groups: int
    recoverable_target_incomplete_groups: int

    step_flow: tuple[
        tuple[
            str,
            int,
            int,
            int,
        ],
        ...,
    ]


def convert_mri_targets_to_numeric(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    if not target_columns:
        raise MRISecondaryTieAuditError(
            "The MRI target catalog is empty."
        )

    missing_columns = [
        column
        for column in target_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise MRISecondaryTieAuditError(
            "MRI target conversion is missing columns: "
            f"{missing_columns}"
        )

    original_targets = frame.loc[
        :,
        list(target_columns),
    ]

    numeric_targets = original_targets.apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_observed_mask = (
        original_targets.notna()
        & numeric_targets.isna()
    )

    invalid_observed_count = int(
        invalid_observed_mask
        .to_numpy()
        .sum()
    )

    if invalid_observed_count:
        raise MRISecondaryTieAuditError(
            "The MRI target catalog contains "
            f"{invalid_observed_count} observed non-numeric cells."
        )

    numeric_array = numeric_targets.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    infinite_count = int(
        np.isinf(
            numeric_array
        ).sum()
    )

    if infinite_count:
        raise MRISecondaryTieAuditError(
            "The MRI target catalog contains "
            f"{infinite_count} infinite numeric cells."
        )

    return numeric_targets.astype(
        "float64"
    )


def mri_target_vectors_are_equivalent(
    numeric_target_frame: pd.DataFrame,
) -> bool:
    if numeric_target_frame.empty:
        raise MRISecondaryTieAuditError(
            "Cannot compare an empty MRI candidate set."
        )

    if len(numeric_target_frame) == 1:
        return True

    values = numeric_target_frame.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    reference = values[0]

    for position in range(
        1,
        len(values),
    ):
        candidate = values[position]

        if not np.allclose(
            reference,
            candidate,
            rtol=MRI_NUMERIC_RTOL,
            atol=MRI_NUMERIC_ATOL,
            equal_nan=True,
        ):
            return False

    return True


def build_mri_secondary_tie_audit(
    frame: pd.DataFrame,
    measure_columns: Sequence[str],
    target_columns: Sequence[str],
) -> MRISecondaryTieAuditSummary:
    required_columns = {
        "RID",
        "VISCODE2",
        "IMAGEUID",
        "STATUS",
        "OVERALLQC",
        "FIELD_STRENGTH",
        *MRI_REGIONAL_QC_COLUMNS,
        *measure_columns,
        *target_columns,
    }

    missing_columns = sorted(
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise MRISecondaryTieAuditError(
            "UCSFFSX7 is missing columns required for the "
            f"secondary MRI tie audit: {missing_columns}"
        )

    if not measure_columns:
        raise MRISecondaryTieAuditError(
            "The MRI measurement catalog is empty."
        )

    if not target_columns:
        raise MRISecondaryTieAuditError(
            "The MRI target catalog is empty."
        )

    work = frame.copy()

    # Exclude rows without a complete visit key
    valid_key_mask = (
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

    work = work.loc[
        valid_key_mask
    ].copy()

    # Targets are converted to numeric only for completeness
    # and numerical-equivalence diagnostics
    numeric_targets = convert_mri_targets_to_numeric(
        frame=work,
        target_columns=target_columns,
    )

    work["__target_nonmissing_count"] = (
        numeric_targets
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    work["__measure_nonmissing_count"] = (
        work.loc[
            :,
            list(measure_columns),
        ]
        .notna()
        .sum(axis=1)
        .astype("int64")
    )

    work["__status"] = canonicalize_audit_series(
        work["STATUS"]
    )

    work["__overall_qc"] = canonicalize_audit_series(
        work["OVERALLQC"]
    )

    work["__field_strength"] = canonicalize_audit_series(
        work["FIELD_STRENGTH"]
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

    work["__overall_qc_present"] = (
        work["__overall_qc"]
        .ne(
            MRI_AUDIT_MISSING_LABEL
        )
        .astype("int64")
    )

    work["__regional_qc_present_count"] = (
        regional_qc_frame
        .ne(
            MRI_AUDIT_MISSING_LABEL
        )
        .sum(axis=1)
        .astype("int64")
    )

    work["__qc_available_count"] = (
        work["__overall_qc_present"]
        + work["__regional_qc_present_count"]
    )

    work["__overall_pass"] = (
        work["__overall_qc"]
        .eq("PASS")
        .astype("int64")
    )

    work["__regional_fail_count"] = (
        regional_qc_frame
        .eq("FAIL")
        .sum(axis=1)
        .astype("int64")
    )

    # Only visits with more than one MRI candidate are relevant
    duplicated_mask = work.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    )

    duplicate_work = work.loc[
        duplicated_mask
    ].copy()

    grouped = duplicate_work.groupby(
        [
            "RID",
            "VISCODE2",
        ],
        dropna=False,
        sort=False,
    )

    duplicate_visit_groups = int(
        grouped.ngroups
    )

    counters = Counter()

    entering_measure = 0
    entering_qc = 0
    entering_pass = 0
    entering_regional_fail = 0
    entering_equivalence = 0

    def register_recovery(
        maximum_target_count: int,
    ) -> None:
        if (
            maximum_target_count
            == len(target_columns)
        ):
            counters[
                "recoverable_target_complete"
            ] += 1
        else:
            counters[
                "recoverable_target_incomplete"
            ] += 1

    for _, group in grouped:
        # Primary criterion: maximum completeness across the target catalog
        maximum_target_count = int(
            group[
                "__target_nonmissing_count"
            ].max()
        )

        primary_candidates = group.loc[
            group[
                "__target_nonmissing_count"
            ].eq(
                maximum_target_count
            )
        ].copy()

        # No tie remains if only one candidate has maximum target completeness
        if len(primary_candidates) <= 1:
            continue

        counters[
            "primary_tie_groups"
        ] += 1

        if (
            maximum_target_count
            == len(target_columns)
        ):
            counters[
                "primary_tie_target_complete"
            ] += 1
        else:
            counters[
                "primary_tie_target_incomplete"
            ] += 1

        # These fields are audited for disagreement
        if (
            primary_candidates[
                "__measure_nonmissing_count"
            ].nunique(
                dropna=False
            )
            > 1
        ):
            counters[
                "measure_completeness_disagreement"
            ] += 1

        if (
            primary_candidates[
                "__qc_available_count"
            ].nunique(
                dropna=False
            )
            > 1
        ):
            counters[
                "qc_availability_disagreement"
            ] += 1

        if (
            primary_candidates[
                "__status"
            ].nunique(
                dropna=False
            )
            > 1
        ):
            counters[
                "status_disagreement"
            ] += 1

        if (
            primary_candidates[
                "__overall_qc"
            ].nunique(
                dropna=False
            )
            > 1
        ):
            counters[
                "overall_qc_disagreement"
            ] += 1

        if (
            primary_candidates[
                "__field_strength"
            ].nunique(
                dropna=False
            )
            > 1
        ):
            counters[
                "field_strength_disagreement"
            ] += 1

        current_candidates = primary_candidates

        # Diagnostic step 1: maximum completeness across all 325 measures
        entering_measure += 1

        maximum_measure_count = int(
            current_candidates[
                "__measure_nonmissing_count"
            ].max()
        )

        current_candidates = current_candidates.loc[
            current_candidates[
                "__measure_nonmissing_count"
            ].eq(
                maximum_measure_count
            )
        ]

        if len(current_candidates) == 1:
            counters[
                "resolved_by_measure_completeness"
            ] += 1

            register_recovery(
                maximum_target_count
            )

            continue

        # Diagnostic step 2: maximum number of populated QC fields
        entering_qc += 1

        maximum_qc_available = int(
            current_candidates[
                "__qc_available_count"
            ].max()
        )

        current_candidates = current_candidates.loc[
            current_candidates[
                "__qc_available_count"
            ].eq(
                maximum_qc_available
            )
        ]

        if len(current_candidates) == 1:
            counters[
                "resolved_by_qc_availability"
            ] += 1

            register_recovery(
                maximum_target_count
            )

            continue

        # Diagnostic step 3: unique OVERALLQC=PASS
        entering_pass += 1

        maximum_pass_indicator = int(
            current_candidates[
                "__overall_pass"
            ].max()
        )

        pass_filtered = current_candidates.loc[
            current_candidates[
                "__overall_pass"
            ].eq(
                maximum_pass_indicator
            )
        ]

        if (
            maximum_pass_indicator == 1
            and len(pass_filtered) == 1
        ):
            counters[
                "resolved_by_unique_overall_pass"
            ] += 1

            register_recovery(
                maximum_target_count
            )

            continue

        current_candidates = pass_filtered

        # Diagnostic step 4: minimum explicit regional QC failures
        entering_regional_fail += 1

        minimum_regional_failures = int(
            current_candidates[
                "__regional_fail_count"
            ].min()
        )

        current_candidates = current_candidates.loc[
            current_candidates[
                "__regional_fail_count"
            ].eq(
                minimum_regional_failures
            )
        ]

        if len(current_candidates) == 1:
            counters[
                "resolved_by_fewer_regional_failures"
            ] += 1

            register_recovery(
                maximum_target_count
            )

            continue

        # Diagnostic step 5: numerical equivalence across the targets
        entering_equivalence += 1

        current_numeric_targets = numeric_targets.loc[
            current_candidates.index
        ]

        if mri_target_vectors_are_equivalent(
            current_numeric_targets
        ):
            counters[
                "equivalent_after_secondary_filters"
            ] += 1

            register_recovery(
                maximum_target_count
            )
        else:
            counters[
                "remaining_non_equivalent_conflicts"
            ] += 1

    primary_tie_groups = int(
        counters[
            "primary_tie_groups"
        ]
    )

    resolved_by_measure = int(
        counters[
            "resolved_by_measure_completeness"
        ]
    )

    resolved_by_qc = int(
        counters[
            "resolved_by_qc_availability"
        ]
    )

    resolved_by_pass = int(
        counters[
            "resolved_by_unique_overall_pass"
        ]
    )

    resolved_by_regional_fail = int(
        counters[
            "resolved_by_fewer_regional_failures"
        ]
    )

    equivalent_after_filters = int(
        counters[
            "equivalent_after_secondary_filters"
        ]
    )

    remaining_conflicts = int(
        counters[
            "remaining_non_equivalent_conflicts"
        ]
    )

    step_flow = (
        (
            "measure_completeness",
            entering_measure,
            resolved_by_measure,
            entering_measure - resolved_by_measure,
        ),
        (
            "qc_availability",
            entering_qc,
            resolved_by_qc,
            entering_qc - resolved_by_qc,
        ),
        (
            "unique_overall_pass",
            entering_pass,
            resolved_by_pass,
            entering_pass - resolved_by_pass,
        ),
        (
            "fewer_regional_failures",
            entering_regional_fail,
            resolved_by_regional_fail,
            (
                entering_regional_fail
                - resolved_by_regional_fail
            ),
        ),
        (
            "numerical_equivalence",
            entering_equivalence,
            equivalent_after_filters,
            remaining_conflicts,
        ),
    )

    summary = MRISecondaryTieAuditSummary(
        source_name="UCSFFSX7",
        duplicate_visit_groups=duplicate_visit_groups,
        primary_tie_groups=primary_tie_groups,
        primary_tie_target_complete_groups=int(
            counters[
                "primary_tie_target_complete"
            ]
        ),
        primary_tie_target_incomplete_groups=int(
            counters[
                "primary_tie_target_incomplete"
            ]
        ),
        primary_ties_with_measure_completeness_disagreement=int(
            counters[
                "measure_completeness_disagreement"
            ]
        ),
        primary_ties_with_qc_availability_disagreement=int(
            counters[
                "qc_availability_disagreement"
            ]
        ),
        primary_ties_with_status_disagreement=int(
            counters[
                "status_disagreement"
            ]
        ),
        primary_ties_with_overall_qc_disagreement=int(
            counters[
                "overall_qc_disagreement"
            ]
        ),
        primary_ties_with_field_strength_disagreement=int(
            counters[
                "field_strength_disagreement"
            ]
        ),
        resolved_by_measure_completeness=resolved_by_measure,
        resolved_by_qc_availability=resolved_by_qc,
        resolved_by_unique_overall_pass=resolved_by_pass,
        resolved_by_fewer_regional_failures=resolved_by_regional_fail,
        equivalent_after_secondary_filters=equivalent_after_filters,
        remaining_non_equivalent_conflicts=remaining_conflicts,
        recoverable_target_complete_groups=int(
            counters[
                "recoverable_target_complete"
            ]
        ),
        recoverable_target_incomplete_groups=int(
            counters[
                "recoverable_target_incomplete"
            ]
        ),
        step_flow=step_flow,
    )

    classified_primary_ties = (
        summary.resolved_by_measure_completeness
        + summary.resolved_by_qc_availability
        + summary.resolved_by_unique_overall_pass
        + summary.resolved_by_fewer_regional_failures
        + summary.equivalent_after_secondary_filters
        + summary.remaining_non_equivalent_conflicts
    )

    if (
        classified_primary_ties
        != summary.primary_tie_groups
    ):
        raise MRISecondaryTieAuditError(
            "MRI secondary tie classification accounting failed."
        )

    if (
        summary.primary_tie_target_complete_groups
        + summary.primary_tie_target_incomplete_groups
        != summary.primary_tie_groups
    ):
        raise MRISecondaryTieAuditError(
            "MRI secondary tie target-completeness "
            "accounting failed."
        )

    potentially_recoverable = (
        summary.primary_tie_groups
        - summary.remaining_non_equivalent_conflicts
    )

    if (
        summary.recoverable_target_complete_groups
        + summary.recoverable_target_incomplete_groups
        != potentially_recoverable
    ):
        raise MRISecondaryTieAuditError(
            "MRI secondary tie recovery accounting failed."
        )

    return summary