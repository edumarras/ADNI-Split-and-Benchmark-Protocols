from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from adni_benchmark.non_mri_resolution import (
    NON_MRI_SOURCE_TOKENS,
    comparison_columns_for_source,
    first_existing_column,
    parse_protocol_month,
    values_equal,
)
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
    SourceContract,
)


class NonMRIConflictImpactError(RuntimeError):
    pass


@dataclass(frozen=True)
class NonMRISourceConflictImpactSummary:
    source_name: str
    grain: str
    unresolved_groups: int

    source_row_count_distribution: tuple[
        tuple[int, int],
        ...,
    ]

    candidate_row_count_distribution: tuple[
        tuple[int, int],
        ...,
    ]

    groups_with_observed_value_disagreement: int
    groups_with_missingness_disagreement: int
    groups_with_only_missingness_disagreement: int
    groups_with_ptgender_disagreement: int

    observed_value_disagreement_column_counts: tuple[
        tuple[str, int],
        ...,
    ]

    missingness_disagreement_column_counts: tuple[
        tuple[str, int],
        ...,
    ]

    unresolved_groups_reaching_resolved_mri: int
    unresolved_groups_reaching_target_complete_mri: int

    affected_resolved_mri_visits: int
    affected_target_complete_mri_visits: int

    affected_resolved_mri_participants: int
    affected_target_complete_mri_participants: int


@dataclass(frozen=True)
class NonMRIConflictImpactSummary:
    mri_resolved_visits: int
    mri_target_complete_visits: int
    mri_resolved_participants: int
    mri_target_complete_participants: int

    total_unresolved_groups: int
    sources_with_unresolved_groups: int

    resolved_mri_visits_affected_by_any_conflict: int
    target_complete_mri_visits_affected_by_any_conflict: int

    resolved_mri_participants_affected_by_any_conflict: int
    target_complete_mri_participants_affected_by_any_conflict: int

    source_summaries: tuple[
        NonMRISourceConflictImpactSummary,
        ...,
    ]


def count_distinct_nonmissing_equivalence_classes(
    series: pd.Series,
) -> int:
    # Use exactly the same equality semantics as non-MRI resolution
    representatives: list[Any] = []

    for value in series.tolist():
        if pd.isna(value):
            continue

        already_represented = any(
            values_equal(
                value,
                representative,
            )
            for representative in representatives
        )

        if not already_represented:
            representatives.append(
                value
            )

    return len(
        representatives
    )


def reconstruct_unresolved_candidate_group(
    frame: pd.DataFrame,
    contract: SourceContract,
    unresolved_record: pd.Series,
) -> pd.DataFrame:
    # Rebuild the exact candidate group used by the canonical resolver
    mask = pd.Series(
        True,
        index=frame.index,
        dtype="boolean",
    )

    for key_column in contract.key_columns:
        if key_column not in unresolved_record.index:
            raise NonMRIConflictImpactError(
                f"{contract.source_name} unresolved record is "
                f"missing key column {key_column}."
            )

        key_value = unresolved_record[
            key_column
        ]

        if pd.isna(key_value):
            raise NonMRIConflictImpactError(
                f"{contract.source_name} unresolved record has "
                f"a missing key value in {key_column}."
            )

        mask = (
            mask
            & frame[key_column]
            .eq(
                key_value
            )
            .fillna(
                False
            )
        )

    group = frame.loc[
        mask.fillna(
            False
        )
    ].copy()

    if group.empty:
        raise NonMRIConflictImpactError(
            f"{contract.source_name} unresolved group could "
            "not be reconstructed."
        )

    candidate_group = group

    # Preserve the exact PTDEMOG candidate restriction used by the
    # initial non-MRI resolver
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

    expected_source_rows = int(
        unresolved_record[
            "n_source_rows"
        ]
    )

    expected_candidate_rows = int(
        unresolved_record[
            "n_candidate_rows"
        ]
    )

    if (
        len(group)
        != expected_source_rows
    ):
        raise NonMRIConflictImpactError(
            f"{contract.source_name} reconstructed source-row "
            "count does not match the canonical resolver."
        )

    if (
        len(candidate_group)
        != expected_candidate_rows
    ):
        raise NonMRIConflictImpactError(
            f"{contract.source_name} reconstructed candidate-row "
            "count does not match the canonical resolver."
        )

    return candidate_group


def profile_candidate_group_disagreements(
    candidate_group: pd.DataFrame,
    comparison_columns: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    if candidate_group.empty:
        raise NonMRIConflictImpactError(
            "Cannot profile an empty candidate group."
        )

    observed_value_disagreements: list[str] = []
    missingness_disagreements: list[str] = []

    for column in comparison_columns:
        if column not in candidate_group.columns:
            raise NonMRIConflictImpactError(
                "Candidate group is missing comparison column "
                f"{column}."
            )

        series = candidate_group[
            column
        ]

        if (
            series.isna().any()
            and series.notna().any()
        ):
            missingness_disagreements.append(
                column
            )

        distinct_observed_values = (
            count_distinct_nonmissing_equivalence_classes(
                series
            )
        )

        if (
            distinct_observed_values
            > 1
        ):
            observed_value_disagreements.append(
                column
            )

    return (
        tuple(
            sorted(
                observed_value_disagreements
            )
        ),
        tuple(
            sorted(
                missingness_disagreements
            )
        ),
    )


def build_non_mri_conflict_impact_audit(
    source_tables: dict[
        str,
        pd.DataFrame,
    ],
    unresolved_non_mri_sources: dict[
        str,
        pd.DataFrame,
    ],
    resolved_mri_source: pd.DataFrame,
) -> NonMRIConflictImpactSummary:
    required_mri_columns = {
        "RID",
        "VISCODE2",
        "__target_complete",
    }

    missing_mri_columns = sorted(
        required_mri_columns
        - set(
            resolved_mri_source.columns
        )
    )

    if missing_mri_columns:
        raise NonMRIConflictImpactError(
            "Resolved MRI table is missing columns required "
            "for conflict-impact auditing: "
            f"{missing_mri_columns}"
        )

    if resolved_mri_source.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    ).any():
        raise NonMRIConflictImpactError(
            "Resolved MRI table contains duplicate visit keys."
        )

    missing_source_tables = [
        source_name
        for source_name
        in NON_MRI_SOURCE_TOKENS
        if source_name
        not in source_tables
    ]

    missing_unresolved_tables = [
        source_name
        for source_name
        in NON_MRI_SOURCE_TOKENS
        if source_name
        not in unresolved_non_mri_sources
    ]

    if (
        missing_source_tables
        or missing_unresolved_tables
    ):
        raise NonMRIConflictImpactError(
            "Non-MRI conflict audit registry mismatch. "
            f"Missing source tables: {missing_source_tables}. "
            "Missing unresolved tables: "
            f"{missing_unresolved_tables}."
        )

    mri = (
        resolved_mri_source
        .copy()
    )

    target_complete_mask = (
        mri[
            "__target_complete"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    affected_by_any_conflict = pd.Series(
        False,
        index=mri.index,
        dtype=bool,
    )

    source_summaries: list[
        NonMRISourceConflictImpactSummary
    ] = []

    total_unresolved_groups = 0
    sources_with_unresolved_groups = 0

    for source_name in NON_MRI_SOURCE_TOKENS:
        contract = (
            SOURCE_CONTRACTS[
                source_name
            ]
        )

        source_frame = (
            source_tables[
                source_name
            ]
        )

        unresolved_frame = (
            unresolved_non_mri_sources[
                source_name
            ]
        )

        unresolved_groups = int(
            len(
                unresolved_frame
            )
        )

        total_unresolved_groups += (
            unresolved_groups
        )

        if unresolved_groups:
            sources_with_unresolved_groups += 1

        source_row_counter = Counter(
            int(
                value
            )
            for value
            in unresolved_frame.get(
                "n_source_rows",
                pd.Series(
                    dtype="int64"
                ),
            ).tolist()
        )

        candidate_row_counter = Counter(
            int(
                value
            )
            for value
            in unresolved_frame.get(
                "n_candidate_rows",
                pd.Series(
                    dtype="int64"
                ),
            ).tolist()
        )

        observed_column_counter: Counter[
            str
        ] = Counter()

        missingness_column_counter: Counter[
            str
        ] = Counter()

        groups_with_observed_disagreement = 0
        groups_with_missingness_disagreement = 0
        groups_with_only_missingness_disagreement = 0
        groups_with_ptgender_disagreement = 0

        comparison_columns = (
            comparison_columns_for_source(
                frame=source_frame,
                contract=contract,
            )
        )

        for _, unresolved_record in (
            unresolved_frame.iterrows()
        ):
            candidate_group = (
                reconstruct_unresolved_candidate_group(
                    frame=source_frame,
                    contract=contract,
                    unresolved_record=(
                        unresolved_record
                    ),
                )
            )

            (
                observed_columns,
                missingness_columns,
            ) = profile_candidate_group_disagreements(
                candidate_group=(
                    candidate_group
                ),
                comparison_columns=(
                    comparison_columns
                ),
            )

            if observed_columns:
                groups_with_observed_disagreement += 1

            if missingness_columns:
                groups_with_missingness_disagreement += 1

            if (
                missingness_columns
                and not observed_columns
            ):
                groups_with_only_missingness_disagreement += 1

            disagreement_union = (
                set(
                    observed_columns
                )
                | set(
                    missingness_columns
                )
            )

            if (
                "PTGENDER"
                in disagreement_union
            ):
                groups_with_ptgender_disagreement += 1

            observed_column_counter.update(
                observed_columns
            )

            missingness_column_counter.update(
                missingness_columns
            )

        if (
            contract.key_columns
            == (
                "RID",
                "VISCODE2",
            )
        ):
            unresolved_keys = (
                unresolved_frame.loc[
                    :,
                    [
                        "RID",
                        "VISCODE2",
                    ],
                ]
                .drop_duplicates()
            )

            unresolved_key_index = (
                pd.MultiIndex.from_frame(
                    unresolved_keys
                )
            )

            mri_key_index = (
                pd.MultiIndex.from_frame(
                    mri.loc[
                        :,
                        [
                            "RID",
                            "VISCODE2",
                        ],
                    ]
                )
            )

            source_impact_mask = pd.Series(
                mri_key_index.isin(
                    unresolved_key_index
                ),
                index=mri.index,
                dtype=bool,
            )

            groups_reaching_resolved_mri = int(
                source_impact_mask.sum()
            )

            groups_reaching_target_complete_mri = int(
                (
                    source_impact_mask
                    & target_complete_mask
                ).sum()
            )

        elif (
            contract.key_columns
            == (
                "RID",
            )
        ):
            unresolved_rids = set(
                unresolved_frame[
                    "RID"
                ]
                .dropna()
                .astype(
                    "string"
                )
                .tolist()
            )

            source_impact_mask = (
                mri[
                    "RID"
                ]
                .astype(
                    "string"
                )
                .isin(
                    unresolved_rids
                )
            )

            groups_reaching_resolved_mri = int(
                mri.loc[
                    source_impact_mask,
                    "RID",
                ].nunique()
            )

            groups_reaching_target_complete_mri = int(
                mri.loc[
                    (
                        source_impact_mask
                        & target_complete_mask
                    ),
                    "RID",
                ].nunique()
            )

        else:
            raise NonMRIConflictImpactError(
                "Unsupported key contract for "
                f"{source_name}: "
                f"{contract.key_columns}"
            )

        affected_by_any_conflict = (
            affected_by_any_conflict
            | source_impact_mask
        )

        target_complete_impact_mask = (
            source_impact_mask
            & target_complete_mask
        )

        source_summary = (
            NonMRISourceConflictImpactSummary(
                source_name=(
                    source_name
                ),
                grain=(
                    contract.grain
                ),
                unresolved_groups=(
                    unresolved_groups
                ),
                source_row_count_distribution=tuple(
                    (
                        row_count,
                        int(
                            source_row_counter[
                                row_count
                            ]
                        ),
                    )
                    for row_count
                    in sorted(
                        source_row_counter
                    )
                ),
                candidate_row_count_distribution=tuple(
                    (
                        row_count,
                        int(
                            candidate_row_counter[
                                row_count
                            ]
                        ),
                    )
                    for row_count
                    in sorted(
                        candidate_row_counter
                    )
                ),
                groups_with_observed_value_disagreement=(
                    groups_with_observed_disagreement
                ),
                groups_with_missingness_disagreement=(
                    groups_with_missingness_disagreement
                ),
                groups_with_only_missingness_disagreement=(
                    groups_with_only_missingness_disagreement
                ),
                groups_with_ptgender_disagreement=(
                    groups_with_ptgender_disagreement
                ),
                observed_value_disagreement_column_counts=tuple(
                    (
                        column,
                        int(
                            observed_column_counter[
                                column
                            ]
                        ),
                    )
                    for column
                    in sorted(
                        observed_column_counter
                    )
                ),
                missingness_disagreement_column_counts=tuple(
                    (
                        column,
                        int(
                            missingness_column_counter[
                                column
                            ]
                        ),
                    )
                    for column
                    in sorted(
                        missingness_column_counter
                    )
                ),
                unresolved_groups_reaching_resolved_mri=(
                    groups_reaching_resolved_mri
                ),
                unresolved_groups_reaching_target_complete_mri=(
                    groups_reaching_target_complete_mri
                ),
                affected_resolved_mri_visits=int(
                    source_impact_mask.sum()
                ),
                affected_target_complete_mri_visits=int(
                    target_complete_impact_mask.sum()
                ),
                affected_resolved_mri_participants=int(
                    mri.loc[
                        source_impact_mask,
                        "RID",
                    ].nunique()
                ),
                affected_target_complete_mri_participants=int(
                    mri.loc[
                        target_complete_impact_mask,
                        "RID",
                    ].nunique()
                ),
            )
        )

        source_summaries.append(
            source_summary
        )

    affected_target_complete = (
        affected_by_any_conflict
        & target_complete_mask
    )

    summary = NonMRIConflictImpactSummary(
        mri_resolved_visits=int(
            len(
                mri
            )
        ),
        mri_target_complete_visits=int(
            target_complete_mask.sum()
        ),
        mri_resolved_participants=int(
            mri[
                "RID"
            ].nunique()
        ),
        mri_target_complete_participants=int(
            mri.loc[
                target_complete_mask,
                "RID",
            ].nunique()
        ),
        total_unresolved_groups=(
            total_unresolved_groups
        ),
        sources_with_unresolved_groups=(
            sources_with_unresolved_groups
        ),
        resolved_mri_visits_affected_by_any_conflict=int(
            affected_by_any_conflict.sum()
        ),
        target_complete_mri_visits_affected_by_any_conflict=int(
            affected_target_complete.sum()
        ),
        resolved_mri_participants_affected_by_any_conflict=int(
            mri.loc[
                affected_by_any_conflict,
                "RID",
            ].nunique()
        ),
        target_complete_mri_participants_affected_by_any_conflict=int(
            mri.loc[
                affected_target_complete,
                "RID",
            ].nunique()
        ),
        source_summaries=tuple(
            source_summaries
        ),
    )

    if (
        sum(
            source_summary.unresolved_groups
            for source_summary
            in summary.source_summaries
        )
        != summary.total_unresolved_groups
    ):
        raise NonMRIConflictImpactError(
            "Non-MRI unresolved-group accounting failed."
        )

    if (
        sum(
            int(
                source_summary.unresolved_groups
                > 0
            )
            for source_summary
            in summary.source_summaries
        )
        != summary.sources_with_unresolved_groups
    ):
        raise NonMRIConflictImpactError(
            "Non-MRI unresolved-source accounting failed."
        )

    if (
        summary.target_complete_mri_visits_affected_by_any_conflict
        > summary.mri_target_complete_visits
    ):
        raise NonMRIConflictImpactError(
            "Non-MRI conflict impact exceeds the target-complete "
            "MRI population."
        )

    return summary