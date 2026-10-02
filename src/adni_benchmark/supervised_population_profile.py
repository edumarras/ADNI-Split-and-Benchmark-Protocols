from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from adni_benchmark.non_mri_resolution import (
    comparison_columns_for_source,
    parse_protocol_month,
    values_equal,
)
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
    SourceContract,
)


class SupervisedPopulationProfileError(RuntimeError):
    pass


# These are the eight model-input modalities. DXSUM is used only as metadata
# for split profiling and is intentionally not counted as an input modality.
INPUT_MODALITY_TOKENS: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "PTDEMOG",
    "APOERES",
)

SAFE_MINIMUM_CELL_COUNT: Final[int] = 5
UNAVAILABLE_CATEGORY_LABEL: Final[str] = "<UNAVAILABLE>"
NO_MODALITY_COMBINATION_LABEL: Final[str] = "<NONE>"


@dataclass(frozen=True)
class ModalityAvailabilitySummary:
    source_name: str
    grain: str

    anchor_visits_available: int
    anchor_visits_unavailable_due_to_conflict: int
    anchor_visits_unavailable_without_resolved_record: int

    anchor_participants_any_available: int
    anchor_participants_first_visit_available: int
    anchor_participants_all_visits_available: int
    anchor_participants_affected_by_conflict: int

    anchor_visits_resolved_but_no_substantive_observation: int
    anchor_participants_resolved_but_no_substantive_observation: int

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "grain": self.grain,
            "anchor_visit_coverage": {
                "available": self.anchor_visits_available,
                "unavailable_due_to_unresolved_conflict": (
                    self.anchor_visits_unavailable_due_to_conflict
                ),
                "unavailable_without_resolved_record": (
                    self.anchor_visits_unavailable_without_resolved_record
                ),
            },
            "anchor_participant_coverage": {
                "available_on_any_eligible_visit": (
                    self.anchor_participants_any_available
                ),
                "available_on_first_eligible_visit": (
                    self.anchor_participants_first_visit_available
                ),
                "available_on_all_eligible_visits": (
                    self.anchor_participants_all_visits_available
                ),
                "affected_by_unresolved_conflict": (
                    self.anchor_participants_affected_by_conflict
                ),
            },
            "resolved_record_content_audit": {
                "anchor_visit_exposures_with_no_observed_substantive_field": (
                    self.anchor_visits_resolved_but_no_substantive_observation
                ),
                "anchor_participants_with_no_observed_substantive_field": (
                    self.anchor_participants_resolved_but_no_substantive_observation
                ),
            },
        }


@dataclass(frozen=True)
class PTGenderConsensusSummary:
    anchor_participants: int
    status_counts: tuple[tuple[str, int], ...]
    consistent_category_counts: tuple[tuple[str, int], ...]
    consistent_participants_with_mixed_missingness: int
    participants_with_consensus: int
    participants_without_consensus: int
    participants_with_consensus_and_resolved_ptdemog: int
    participants_with_consensus_without_resolved_ptdemog: int

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "anchor_participants": self.anchor_participants,
            "consensus_status_distribution": (
                serialize_safe_count_distribution(
                    self.status_counts,
                    suppress_small_cells=False,
                )
            ),
            "consistent_category_distribution": (
                serialize_safe_count_distribution(
                    self.consistent_category_counts,
                    pseudonymize_labels=True,
                )
            ),
            "consistent_participants_with_mixed_missingness": (
                self.consistent_participants_with_mixed_missingness
            ),
            "participants_with_consensus": (
                self.participants_with_consensus
            ),
            "participants_without_consensus": (
                self.participants_without_consensus
            ),
            "relationship_to_model_ptdemog_availability": {
                "consensus_and_resolved_ptdemog": (
                    self.participants_with_consensus_and_resolved_ptdemog
                ),
                "consensus_without_resolved_ptdemog": (
                    self.participants_with_consensus_without_resolved_ptdemog
                ),
            },
            "category_labels_exposed": False,
        }


@dataclass(frozen=True)
class SupervisedPopulationProfileSummary:
    anchor_visits: int
    anchor_participants: int
    modality_summaries: tuple[ModalityAvailabilitySummary, ...]

    eligible_visit_count_min: int
    eligible_visit_count_median: float
    eligible_visit_count_max: int
    eligible_visit_count_distribution: tuple[tuple[str, int], ...]
    longitudinal_class_counts: tuple[tuple[str, int], ...]
    first_visit_order_basis_counts: tuple[tuple[str, int], ...]

    visit_available_modality_count_distribution: tuple[
        tuple[str, int],
        ...,
    ]
    participant_any_available_modality_count_distribution: tuple[
        tuple[str, int],
        ...,
    ]
    participant_first_available_modality_count_distribution: tuple[
        tuple[str, int],
        ...,
    ]
    participant_all_available_modality_count_distribution: tuple[
        tuple[str, int],
        ...,
    ]

    visit_modality_combination_counts: tuple[tuple[str, int], ...]
    participant_any_modality_combination_counts: tuple[
        tuple[str, int],
        ...,
    ]
    participant_first_modality_combination_counts: tuple[
        tuple[str, int],
        ...,
    ]

    visit_input_completeness_counts: tuple[tuple[str, int], ...]
    participant_any_input_completeness_counts: tuple[
        tuple[str, int],
        ...,
    ]
    participant_first_input_completeness_counts: tuple[
        tuple[str, int],
        ...,
    ]
    participant_all_input_completeness_counts: tuple[
        tuple[str, int],
        ...,
    ]

    first_diagnosis_status_counts: tuple[tuple[str, int], ...]
    first_diagnosis_category_counts: tuple[tuple[str, int], ...]
    first_diagnosis_eligible_visit_position_counts: tuple[
        tuple[str, int],
        ...,
    ]

    apoe_resolved_participants: int
    apoe_genotype_observed_participants: int
    apoe_resolved_but_genotype_missing_participants: int
    apoe_unresolved_conflict_participants: int
    apoe_without_resolved_record_participants: int

    ptgender_consensus: PTGenderConsensusSummary

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "supervised_anchor": {
                "visits": self.anchor_visits,
                "participants": self.anchor_participants,
                "target_definition": (
                    "resolved UCSFFSX7 visit with all fixed 323 "
                    "supervised targets observed"
                ),
            },
            "modality_coverage": [
                summary.to_manifest_record()
                for summary in self.modality_summaries
            ],
            "longitudinal_profile": {
                "eligible_visits_per_participant": {
                    "minimum": self.eligible_visit_count_min,
                    "median": self.eligible_visit_count_median,
                    "maximum": self.eligible_visit_count_max,
                    "distribution": (
                        serialize_safe_count_distribution(
                            self.eligible_visit_count_distribution
                        )
                    ),
                },
                "longitudinal_class_distribution": (
                    serialize_safe_count_distribution(
                        self.longitudinal_class_counts,
                        suppress_small_cells=False,
                    )
                ),
                "first_visit_order_basis_distribution": (
                    serialize_safe_count_distribution(
                        self.first_visit_order_basis_counts,
                        suppress_small_cells=False,
                    )
                ),
            },
            "input_availability": {
                "visit_level": {
                    "available_modality_count_distribution": (
                        serialize_safe_count_distribution(
                            self.visit_available_modality_count_distribution
                        )
                    ),
                    "input_completeness_distribution": (
                        serialize_safe_count_distribution(
                            self.visit_input_completeness_counts,
                            suppress_small_cells=False,
                        )
                    ),
                    "modality_combination_distribution": (
                        serialize_safe_count_distribution(
                            self.visit_modality_combination_counts
                        )
                    ),
                },
                "participant_any_eligible_visit": {
                    "available_modality_count_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_any_available_modality_count_distribution
                        )
                    ),
                    "input_completeness_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_any_input_completeness_counts,
                            suppress_small_cells=False,
                        )
                    ),
                    "modality_combination_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_any_modality_combination_counts
                        )
                    ),
                },
                "participant_first_eligible_visit": {
                    "available_modality_count_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_first_available_modality_count_distribution
                        )
                    ),
                    "input_completeness_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_first_input_completeness_counts,
                            suppress_small_cells=False,
                        )
                    ),
                    "modality_combination_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_first_modality_combination_counts
                        )
                    ),
                },
                "participant_all_eligible_visits": {
                    "available_modality_count_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_all_available_modality_count_distribution
                        )
                    ),
                    "input_completeness_distribution": (
                        serialize_safe_count_distribution(
                            self.participant_all_input_completeness_counts,
                            suppress_small_cells=False,
                        )
                    ),
                },
            },
            "first_eligible_diagnosis": {
                "status_distribution": (
                    serialize_safe_count_distribution(
                        self.first_diagnosis_status_counts,
                        suppress_small_cells=False,
                    )
                ),
                "category_distribution": (
                    serialize_safe_count_distribution(
                        self.first_diagnosis_category_counts,
                        pseudonymize_labels=True,
                    )
                ),
                "eligible_visit_position_distribution": (
                    serialize_safe_count_distribution(
                        self.first_diagnosis_eligible_visit_position_counts
                    )
                ),
                "category_labels_exposed": False,
            },
            "apoe_coverage": {
                "resolved_participants": self.apoe_resolved_participants,
                "genotype_observed_participants": (
                    self.apoe_genotype_observed_participants
                ),
                "resolved_but_genotype_missing_participants": (
                    self.apoe_resolved_but_genotype_missing_participants
                ),
                "unresolved_conflict_participants": (
                    self.apoe_unresolved_conflict_participants
                ),
                "without_resolved_record_participants": (
                    self.apoe_without_resolved_record_participants
                ),
                "genotype_values_exposed": False,
            },
            "ptgender_auxiliary_consensus": (
                self.ptgender_consensus.to_manifest_record()
            ),
        }


def serialize_safe_count_distribution(
    distribution: Sequence[tuple[str, int]],
    *,
    suppress_small_cells: bool = True,
    pseudonymize_labels: bool = False,
) -> dict[str, object]:
    normalized = [
        (
            str(label),
            int(count),
        )
        for label, count in distribution
    ]

    if any(count < 0 for _, count in normalized):
        raise ValueError(
            "Aggregate distributions cannot contain negative counts."
        )

    normalized.sort(
        key=lambda item: item[0]
    )

    label_map: dict[str, str] = {}

    if pseudonymize_labels:
        label_map = {
            label: f"category_{position:03d}"
            for position, (label, _) in enumerate(
                normalized,
                start=1,
            )
        }

    reported_counts: dict[str, int] = {}
    suppressed_category_count = 0
    suppressed_observation_count = 0

    for label, count in normalized:
        output_label = label_map.get(
            label,
            label,
        )

        should_suppress = (
            suppress_small_cells
            and 0 < count < SAFE_MINIMUM_CELL_COUNT
        )

        if should_suppress:
            suppressed_category_count += 1
            suppressed_observation_count += count
            continue

        reported_counts[output_label] = count

    return {
        "total_observations": int(
            sum(count for _, count in normalized)
        ),
        "reported_counts": reported_counts,
        "small_cell_policy": {
            "enabled": suppress_small_cells,
            "minimum_reported_count": (
                SAFE_MINIMUM_CELL_COUNT
                if suppress_small_cells
                else None
            ),
            "suppressed_category_count": (
                suppressed_category_count
            ),
            "suppressed_observation_count": (
                suppressed_observation_count
            ),
        },
        "labels_pseudonymized": pseudonymize_labels,
    }


def canonical_category_label(
    value: Any,
) -> str:
    if pd.isna(value):
        return UNAVAILABLE_CATEGORY_LABEL

    try:
        numeric = float(value)

        if np.isfinite(numeric):
            if numeric.is_integer():
                return str(
                    int(numeric)
                )

            return format(
                numeric,
                ".15g",
            )

    except (TypeError, ValueError):
        pass

    return (
        str(value)
        .strip()
        .upper()
    )


def counter_to_sorted_tuple(
    counter: Counter[Any],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            str(label),
            int(counter[label]),
        )
        for label in sorted(
            counter,
            key=lambda value: str(value),
        )
    )


def classify_input_completeness(
    available_count: int,
) -> str:
    if available_count < 0:
        raise ValueError(
            "Available modality count cannot be negative."
        )

    if available_count == 0:
        return "no_input_modality_available"

    if available_count == len(INPUT_MODALITY_TOKENS):
        return "all_input_modalities_available"

    if available_count > len(INPUT_MODALITY_TOKENS):
        raise ValueError(
            "Available modality count exceeds the modality registry."
        )

    return "partially_available_inputs"


def build_modality_combination_label(
    row: pd.Series,
    columns_by_source: dict[str, str],
) -> str:
    available_sources = [
        source_name
        for source_name in INPUT_MODALITY_TOKENS
        if bool(
            row[
                columns_by_source[source_name]
            ]
        )
    ]

    if not available_sources:
        return NO_MODALITY_COMBINATION_LABEL

    return "+".join(
        available_sources
    )


def validate_resolved_unresolved_key_disjoint(
    resolved_frame: pd.DataFrame,
    unresolved_frame: pd.DataFrame,
    contract: SourceContract,
) -> None:
    key_columns = list(
        contract.key_columns
    )

    for label, frame in (
        (
            "resolved",
            resolved_frame,
        ),
        (
            "unresolved",
            unresolved_frame,
        ),
    ):
        missing_columns = [
            column
            for column in key_columns
            if column not in frame.columns
        ]

        if missing_columns:
            raise KeyError(
                f"{contract.source_name} {label} table is missing "
                f"key columns: {missing_columns}"
            )

        if frame.duplicated(
            subset=key_columns,
            keep=False,
        ).any():
            raise SupervisedPopulationProfileError(
                f"{contract.source_name} {label} table contains "
                "duplicate canonical keys."
            )

    if resolved_frame.empty or unresolved_frame.empty:
        return

    if len(key_columns) == 1:
        overlap = set(
            resolved_frame[
                key_columns[0]
            ].dropna().tolist()
        ) & set(
            unresolved_frame[
                key_columns[0]
            ].dropna().tolist()
        )

        overlap_count = len(overlap)

    else:
        resolved_index = pd.MultiIndex.from_frame(
            resolved_frame.loc[
                :,
                key_columns,
            ]
        )

        unresolved_index = pd.MultiIndex.from_frame(
            unresolved_frame.loc[
                :,
                key_columns,
            ]
        )

        overlap_count = int(
            resolved_index.isin(
                unresolved_index
            ).sum()
        )

    if overlap_count:
        raise SupervisedPopulationProfileError(
            f"{contract.source_name} has {overlap_count} keys "
            "classified as both resolved and unresolved."
        )


# PTGENDER consensus is auxiliary split-balancing metadata only. It never
# makes a conflicted PTDEMOG block available to the predictive model.
def build_ptgender_consensus_profile(
    ptdemog_source: pd.DataFrame,
    anchor_participant_ids: Sequence[str],
    resolved_ptdemog_participant_ids: set[str],
) -> tuple[
    pd.DataFrame,
    PTGenderConsensusSummary,
]:
    required_columns = {
        "RID",
        "PTGENDER",
    }

    missing_columns = sorted(
        required_columns
        - set(ptdemog_source.columns)
    )

    if missing_columns:
        raise KeyError(
            "PTDEMOG is missing columns required for the PTGENDER "
            f"consensus audit: {missing_columns}"
        )

    participant_ids = [
        str(value)
        for value in anchor_participant_ids
    ]

    if len(participant_ids) != len(
        set(participant_ids)
    ):
        raise SupervisedPopulationProfileError(
            "Anchor participant identifiers are not unique during "
            "PTGENDER consensus construction."
        )

    valid_source = ptdemog_source.loc[
        ptdemog_source["RID"].notna()
    ].copy()

    grouped = {
        str(rid): group
        for rid, group in valid_source.groupby(
            "RID",
            dropna=False,
            sort=False,
        )
    }

    records: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    mixed_missingness_count = 0

    for rid in participant_ids:
        group = grouped.get(
            rid
        )

        if group is None:
            status = "no_ptdemog_source_rows"
            category = UNAVAILABLE_CATEGORY_LABEL
            has_mixed_missingness = False

        else:
            series = group[
                "PTGENDER"
            ]

            observed_values = [
                value
                for value in series.tolist()
                if not pd.isna(value)
            ]

            representatives: list[Any] = []

            for value in observed_values:
                represented = any(
                    values_equal(
                        value,
                        representative,
                    )
                    for representative in representatives
                )

                if not represented:
                    representatives.append(
                        value
                    )

            has_mixed_missingness = bool(
                series.isna().any()
                and series.notna().any()
            )

            if not representatives:
                status = "no_observed_ptgender_value"
                category = UNAVAILABLE_CATEGORY_LABEL

            elif len(representatives) == 1:
                status = "consistent_observed_ptgender"
                category = canonical_category_label(
                    representatives[0]
                )

                category_counter[
                    category
                ] += 1

                if has_mixed_missingness:
                    mixed_missingness_count += 1

            else:
                status = "conflicting_observed_ptgender_values"
                category = UNAVAILABLE_CATEGORY_LABEL

        status_counter[
            status
        ] += 1

        records.append(
            {
                "RID": rid,
                "__ptgender_consensus_status": status,
                "__ptgender_balance_category": category,
                "__ptgender_consensus_available": (
                    status
                    == "consistent_observed_ptgender"
                ),
                "__ptgender_mixed_missingness": (
                    has_mixed_missingness
                ),
            }
        )

    profile = pd.DataFrame(
        records
    )

    if profile.duplicated(
        subset=[
            "RID",
        ],
        keep=False,
    ).any():
        raise SupervisedPopulationProfileError(
            "PTGENDER consensus profile contains duplicate participants."
        )

    consensus_mask = profile[
        "__ptgender_consensus_available"
    ].astype(bool)

    resolved_ptdemog_mask = profile[
        "RID"
    ].isin(
        resolved_ptdemog_participant_ids
    )

    summary = PTGenderConsensusSummary(
        anchor_participants=int(
            len(profile)
        ),
        status_counts=counter_to_sorted_tuple(
            status_counter
        ),
        consistent_category_counts=counter_to_sorted_tuple(
            category_counter
        ),
        consistent_participants_with_mixed_missingness=int(
            mixed_missingness_count
        ),
        participants_with_consensus=int(
            consensus_mask.sum()
        ),
        participants_without_consensus=int(
            (~consensus_mask).sum()
        ),
        participants_with_consensus_and_resolved_ptdemog=int(
            (
                consensus_mask
                & resolved_ptdemog_mask
            ).sum()
        ),
        participants_with_consensus_without_resolved_ptdemog=int(
            (
                consensus_mask
                & ~resolved_ptdemog_mask
            ).sum()
        ),
    )

    if (
        summary.participants_with_consensus
        + summary.participants_without_consensus
        != summary.anchor_participants
    ):
        raise SupervisedPopulationProfileError(
            "PTGENDER consensus availability accounting failed."
        )

    if (
        sum(count for _, count in summary.status_counts)
        != summary.anchor_participants
    ):
        raise SupervisedPopulationProfileError(
            "PTGENDER consensus status accounting failed."
        )

    return (
        profile,
        summary,
    )


# LOCAL_ONLY: the returned anchor and participant profile contain individual
# keys. Only summary.to_manifest_record() is safe for serialization/sharing.
def build_supervised_population_profile(
    source_tables: dict[str, pd.DataFrame],
    resolved_non_mri_sources: dict[str, pd.DataFrame],
    unresolved_non_mri_sources: dict[str, pd.DataFrame],
    resolved_mri_source: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    SupervisedPopulationProfileSummary,
]:
    required_mri_columns = {
        "RID",
        "VISCODE2",
        "EXAMDATE",
        "__target_complete",
    }

    missing_mri_columns = sorted(
        required_mri_columns
        - set(resolved_mri_source.columns)
    )

    if missing_mri_columns:
        raise KeyError(
            "Resolved MRI source is missing columns required for the "
            f"supervised anchor: {missing_mri_columns}"
        )

    missing_loaded_sources = [
        source_name
        for source_name in INPUT_MODALITY_TOKENS
        if source_name not in source_tables
    ]

    missing_resolved_sources = [
        source_name
        for source_name in INPUT_MODALITY_TOKENS
        if source_name not in resolved_non_mri_sources
    ]

    missing_unresolved_sources = [
        source_name
        for source_name in INPUT_MODALITY_TOKENS
        if source_name not in unresolved_non_mri_sources
    ]

    if (
        missing_loaded_sources
        or missing_resolved_sources
        or missing_unresolved_sources
    ):
        raise KeyError(
            "Supervised-profile source registry mismatch. "
            f"Missing loaded: {missing_loaded_sources}. "
            f"Missing resolved: {missing_resolved_sources}. "
            f"Missing unresolved: {missing_unresolved_sources}."
        )

    target_complete_mask = (
        resolved_mri_source[
            "__target_complete"
        ]
        .fillna(False)
        .astype(bool)
    )

    anchor = resolved_mri_source.loc[
        target_complete_mask,
        [
            "RID",
            "VISCODE2",
            "EXAMDATE",
        ],
    ].copy()

    if anchor.empty:
        raise SupervisedPopulationProfileError(
            "The supervised target-complete MRI anchor is empty."
        )

    if anchor.loc[
        :,
        [
            "RID",
            "VISCODE2",
        ],
    ].isna().any().any():
        raise SupervisedPopulationProfileError(
            "The supervised anchor contains missing visit keys."
        )

    if anchor.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    ).any():
        raise SupervisedPopulationProfileError(
            "The supervised anchor contains duplicate visit keys."
        )

    anchor["__protocol_month"] = anchor[
        "VISCODE2"
    ].map(
        parse_protocol_month
    )

    anchor["__examdate_parsed"] = pd.to_datetime(
        anchor["EXAMDATE"],
        errors="coerce",
    )

    anchor["__examdate_for_split"] = anchor[
        "__examdate_parsed"
    ]

    anchor["__protocol_month_missing"] = anchor[
        "__protocol_month"
    ].isna()

    anchor["__examdate_missing"] = anchor[
        "__examdate_parsed"
    ].isna()

    anchor = anchor.sort_values(
        by=[
            "RID",
            "__protocol_month_missing",
            "__protocol_month",
            "__examdate_missing",
            "__examdate_parsed",
            "VISCODE2",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
        na_position="last",
    ).reset_index(
        drop=True
    )

    anchor["__eligible_visit_position"] = (
        anchor.groupby(
            "RID",
            dropna=False,
            sort=False,
        )
        .cumcount()
        .add(1)
        .astype("int64")
    )

    anchor["__first_eligible_visit"] = anchor[
        "__eligible_visit_position"
    ].eq(1)

    anchor["__visit_order_basis"] = np.select(
        condlist=[
            anchor[
                "__protocol_month"
            ].notna(),
            anchor[
                "__examdate_parsed"
            ].notna(),
        ],
        choicelist=[
            "protocol_month",
            "examdate_fallback",
        ],
        default="viscode_lexical_fallback",
    )

    anchor_visit_index = pd.MultiIndex.from_frame(
        anchor.loc[
            :,
            [
                "RID",
                "VISCODE2",
            ],
        ]
    )

    availability_columns: dict[str, str] = {}
    conflict_columns: dict[str, str] = {}
    empty_content_columns: dict[str, str] = {}

    for source_name in INPUT_MODALITY_TOKENS:
        contract = SOURCE_CONTRACTS[
            source_name
        ]

        resolved_frame = resolved_non_mri_sources[
            source_name
        ]

        unresolved_frame = unresolved_non_mri_sources[
            source_name
        ]

        validate_resolved_unresolved_key_disjoint(
            resolved_frame=resolved_frame,
            unresolved_frame=unresolved_frame,
            contract=contract,
        )

        comparison_columns = comparison_columns_for_source(
            frame=resolved_frame,
            contract=contract,
        )

        resolved_has_content = (
            resolved_frame.loc[
                :,
                comparison_columns,
            ]
            .notna()
            .any(axis=1)
        )

        resolved_empty_frame = resolved_frame.loc[
            ~resolved_has_content
        ]

        availability_column = (
            f"__available_{source_name.casefold()}"
        )
        conflict_column = (
            f"__conflict_{source_name.casefold()}"
        )
        empty_content_column = (
            f"__resolved_empty_{source_name.casefold()}"
        )

        availability_columns[
            source_name
        ] = availability_column
        conflict_columns[
            source_name
        ] = conflict_column
        empty_content_columns[
            source_name
        ] = empty_content_column

        if contract.key_columns == (
            "RID",
            "VISCODE2",
        ):
            resolved_key_index = pd.MultiIndex.from_frame(
                resolved_frame.loc[
                    :,
                    [
                        "RID",
                        "VISCODE2",
                    ],
                ]
            )

            unresolved_key_index = pd.MultiIndex.from_frame(
                unresolved_frame.loc[
                    :,
                    [
                        "RID",
                        "VISCODE2",
                    ],
                ]
            )

            empty_key_index = pd.MultiIndex.from_frame(
                resolved_empty_frame.loc[
                    :,
                    [
                        "RID",
                        "VISCODE2",
                    ],
                ]
            )

            available_mask = anchor_visit_index.isin(
                resolved_key_index
            )

            conflict_mask = anchor_visit_index.isin(
                unresolved_key_index
            )

            empty_content_mask = anchor_visit_index.isin(
                empty_key_index
            )

        elif contract.key_columns == (
            "RID",
        ):
            resolved_rids = set(
                resolved_frame[
                    "RID"
                ]
                .dropna()
                .astype("string")
                .tolist()
            )

            unresolved_rids = set(
                unresolved_frame[
                    "RID"
                ]
                .dropna()
                .astype("string")
                .tolist()
            )

            empty_rids = set(
                resolved_empty_frame[
                    "RID"
                ]
                .dropna()
                .astype("string")
                .tolist()
            )

            available_mask = anchor[
                "RID"
            ].astype("string").isin(
                resolved_rids
            ).to_numpy()

            conflict_mask = anchor[
                "RID"
            ].astype("string").isin(
                unresolved_rids
            ).to_numpy()

            empty_content_mask = anchor[
                "RID"
            ].astype("string").isin(
                empty_rids
            ).to_numpy()

        else:
            raise SupervisedPopulationProfileError(
                f"Unsupported modality key contract for {source_name}: "
                f"{contract.key_columns}"
            )

        resolved_series = pd.Series(
            available_mask,
            index=anchor.index,
            dtype=bool,
        )

        conflict_series = pd.Series(
            conflict_mask,
            index=anchor.index,
            dtype=bool,
        )

        empty_content_series = pd.Series(
            empty_content_mask,
            index=anchor.index,
            dtype=bool,
        )

        # A resolved record with every substantive field missing is not
        # considered available. Partial within-record missingness is allowed.
        available_series = (
            resolved_series
            & ~empty_content_series
        )

        if (
            resolved_series
            & conflict_series
        ).any():
            raise SupervisedPopulationProfileError(
                f"{source_name} marks anchor records as both "
                "resolved and conflicted."
            )

        if (
            empty_content_series
            & ~resolved_series
        ).any():
            raise SupervisedPopulationProfileError(
                f"{source_name} empty-content audit identified "
                "a record that is not resolved."
            )

        anchor[
            availability_column
        ] = available_series

        anchor[
            conflict_column
        ] = conflict_series

        anchor[
            empty_content_column
        ] = empty_content_series

    grouped_anchor = anchor.groupby(
        "RID",
        dropna=False,
        sort=False,
    )

    participant_profile = (
        grouped_anchor
        .size()
        .rename("__eligible_visit_count")
        .reset_index()
    )

    participant_profile[
        "__eligible_visit_count"
    ] = participant_profile[
        "__eligible_visit_count"
    ].astype("int64")

    first_visit_rows = anchor.loc[
        anchor[
            "__first_eligible_visit"
        ]
    ].copy()

    if len(first_visit_rows) != len(
        participant_profile
    ):
        raise SupervisedPopulationProfileError(
            "The supervised anchor did not produce exactly one first "
            "eligible visit per participant."
        )

    first_visit_indexed = first_visit_rows.set_index(
        "RID",
    )

    if not first_visit_indexed.index.is_unique:
        raise SupervisedPopulationProfileError(
            "First eligible visit index is not unique by participant."
        )

    participant_profile = participant_profile.set_index(
        "RID",
    )

    if not participant_profile.index.is_unique:
        raise SupervisedPopulationProfileError(
            "Participant profile index is not unique."
        )

    participant_any_columns: dict[str, str] = {}
    participant_first_columns: dict[str, str] = {}
    participant_all_columns: dict[str, str] = {}
    participant_conflict_columns: dict[str, str] = {}
    participant_empty_columns: dict[str, str] = {}

    for source_name in INPUT_MODALITY_TOKENS:
        availability_column = availability_columns[
            source_name
        ]
        conflict_column = conflict_columns[
            source_name
        ]
        empty_content_column = empty_content_columns[
            source_name
        ]

        any_column = (
            f"__any_available_{source_name.casefold()}"
        )
        first_column = (
            f"__first_available_{source_name.casefold()}"
        )
        all_column = (
            f"__all_available_{source_name.casefold()}"
        )
        any_conflict_column = (
            f"__any_conflict_{source_name.casefold()}"
        )
        any_empty_column = (
            f"__any_resolved_empty_{source_name.casefold()}"
        )

        participant_any_columns[
            source_name
        ] = any_column
        participant_first_columns[
            source_name
        ] = first_column
        participant_all_columns[
            source_name
        ] = all_column
        participant_conflict_columns[
            source_name
        ] = any_conflict_column
        participant_empty_columns[
            source_name
        ] = any_empty_column

        participant_profile[
            any_column
        ] = grouped_anchor[
            availability_column
        ].any()

        participant_profile[
            first_column
        ] = first_visit_indexed[
            availability_column
        ].astype(bool)

        participant_profile[
            all_column
        ] = grouped_anchor[
            availability_column
        ].all()

        participant_profile[
            any_conflict_column
        ] = grouped_anchor[
            conflict_column
        ].any()

        participant_profile[
            any_empty_column
        ] = grouped_anchor[
            empty_content_column
        ].any()

    participant_profile[
        "__first_visit_order_basis"
    ] = first_visit_indexed[
        "__visit_order_basis"
    ].astype("string")

    visit_available_columns = [
        availability_columns[
            source_name
        ]
        for source_name in INPUT_MODALITY_TOKENS
    ]

    participant_any_available_columns = [
        participant_any_columns[
            source_name
        ]
        for source_name in INPUT_MODALITY_TOKENS
    ]

    participant_first_available_columns = [
        participant_first_columns[
            source_name
        ]
        for source_name in INPUT_MODALITY_TOKENS
    ]

    participant_all_available_columns = [
        participant_all_columns[
            source_name
        ]
        for source_name in INPUT_MODALITY_TOKENS
    ]

    anchor[
        "__available_modality_count"
    ] = (
        anchor.loc[
            :,
            visit_available_columns,
        ]
        .sum(axis=1)
        .astype("int64")
    )

    participant_profile[
        "__any_available_modality_count"
    ] = (
        participant_profile.loc[
            :,
            participant_any_available_columns,
        ]
        .sum(axis=1)
        .astype("int64")
    )

    participant_profile[
        "__first_available_modality_count"
    ] = (
        participant_profile.loc[
            :,
            participant_first_available_columns,
        ]
        .sum(axis=1)
        .astype("int64")
    )

    participant_profile[
        "__all_available_modality_count"
    ] = (
        participant_profile.loc[
            :,
            participant_all_available_columns,
        ]
        .sum(axis=1)
        .astype("int64")
    )

    anchor[
        "__modality_combination"
    ] = anchor.apply(
        build_modality_combination_label,
        axis=1,
        columns_by_source=(
            availability_columns
        ),
    )

    participant_profile[
        "__any_modality_combination"
    ] = participant_profile.apply(
        build_modality_combination_label,
        axis=1,
        columns_by_source=(
            participant_any_columns
        ),
    )

    participant_profile[
        "__first_modality_combination"
    ] = participant_profile.apply(
        build_modality_combination_label,
        axis=1,
        columns_by_source=(
            participant_first_columns
        ),
    )

    participant_profile[
        "__longitudinal_class"
    ] = np.select(
        condlist=[
            participant_profile[
                "__eligible_visit_count"
            ].eq(1),
            participant_profile[
                "__eligible_visit_count"
            ].eq(2),
        ],
        choicelist=[
            "single_eligible_visit",
            "two_eligible_visits",
        ],
        default="three_or_more_eligible_visits",
    )

    dxsum_resolved = resolved_non_mri_sources[
        "DXSUM"
    ]

    required_dx_columns = {
        "RID",
        "VISCODE2",
        "DIAGNOSIS",
    }

    missing_dx_columns = sorted(
        required_dx_columns
        - set(dxsum_resolved.columns)
    )

    if missing_dx_columns:
        raise KeyError(
            "Resolved DXSUM is missing columns required for first "
            f"diagnosis profiling: {missing_dx_columns}"
        )

    dx_lookup = dxsum_resolved.loc[
        :,
        [
            "RID",
            "VISCODE2",
            "DIAGNOSIS",
        ],
    ].copy()

    if dx_lookup.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    ).any():
        raise SupervisedPopulationProfileError(
            "Resolved DXSUM contains duplicate visit keys."
        )

    anchor = anchor.merge(
        dx_lookup,
        on=[
            "RID",
            "VISCODE2",
        ],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    dx_key_index = pd.MultiIndex.from_frame(
        dx_lookup.loc[:, ["RID", "VISCODE2"]]
    )
    current_anchor_index = pd.MultiIndex.from_frame(
        anchor.loc[:, ["RID", "VISCODE2"]]
    )
    anchor["__dxsum_resolved_metadata"] = (
        current_anchor_index.isin(dx_key_index)
    )

    diagnosis_observed_mask = anchor[
        "DIAGNOSIS"
    ].notna()

    first_observed_diagnosis = (
        anchor.loc[
            diagnosis_observed_mask,
            [
                "RID",
                "DIAGNOSIS",
                "__eligible_visit_position",
            ],
        ]
        .drop_duplicates(
            subset=[
                "RID",
            ],
            keep="first",
        )
        .copy()
    )

    if not first_observed_diagnosis.empty:
        first_observed_diagnosis[
            "__first_diagnosis_category"
        ] = first_observed_diagnosis[
            "DIAGNOSIS"
        ].map(
            canonical_category_label
        )

        first_observed_diagnosis = (
            first_observed_diagnosis
            .drop(
                columns=[
                    "DIAGNOSIS",
                ]
            )
            .set_index(
                "RID",
            )
        )

        participant_profile = participant_profile.join(
            first_observed_diagnosis,
            how="left",
            validate="one_to_one",
        )

    else:
        participant_profile[
            "__eligible_visit_position"
        ] = pd.Series(
            pd.NA,
            index=participant_profile.index,
            dtype="Int64",
        )

        participant_profile[
            "__first_diagnosis_category"
        ] = pd.Series(
            pd.NA,
            index=participant_profile.index,
            dtype="string",
        )

    participant_profile[
        "__first_diagnosis_status"
    ] = "no_resolved_dxsum_on_eligible_visits"

    any_dxsum_mask = (
        anchor.groupby(
            "RID",
            dropna=False,
            sort=False,
        )["__dxsum_resolved_metadata"]
        .any()
        .reindex(participant_profile.index)
        .fillna(False)
        .astype(bool)
    )

    diagnosis_available_mask = participant_profile[
        "__first_diagnosis_category"
    ].notna()

    participant_profile.loc[
        any_dxsum_mask
        & ~diagnosis_available_mask,
        "__first_diagnosis_status",
    ] = "resolved_dxsum_but_no_observed_diagnosis"

    participant_profile.loc[
        diagnosis_available_mask
        & participant_profile[
            "__eligible_visit_position"
        ].eq(1),
        "__first_diagnosis_status",
    ] = "diagnosis_observed_at_first_eligible_visit"

    participant_profile.loc[
        diagnosis_available_mask
        & participant_profile[
            "__eligible_visit_position"
        ].gt(1),
        "__first_diagnosis_status",
    ] = "diagnosis_first_observed_after_first_eligible_visit"

    resolved_ptdemog_ids = set(
        resolved_non_mri_sources[
            "PTDEMOG"
        ][
            "RID"
        ]
        .dropna()
        .astype("string")
        .tolist()
    )

    (
        ptgender_profile,
        ptgender_summary,
    ) = build_ptgender_consensus_profile(
        ptdemog_source=source_tables[
            "PTDEMOG"
        ],
        anchor_participant_ids=(
            participant_profile.index
            .astype("string")
            .tolist()
        ),
        resolved_ptdemog_participant_ids=(
            resolved_ptdemog_ids
        ),
    )

    participant_profile = participant_profile.join(
        ptgender_profile.set_index(
            "RID",
        ),
        how="left",
        validate="one_to_one",
    )

    apoe_resolved = resolved_non_mri_sources[
        "APOERES"
    ]

    required_apoe_columns = {
        "RID",
        "GENOTYPE",
    }

    missing_apoe_columns = sorted(
        required_apoe_columns
        - set(apoe_resolved.columns)
    )

    if missing_apoe_columns:
        raise KeyError(
            "Resolved APOERES is missing columns required for APOE "
            f"coverage auditing: {missing_apoe_columns}"
        )

    apoe_lookup = apoe_resolved.loc[
        :,
        [
            "RID",
            "GENOTYPE",
        ],
    ].copy()

    if apoe_lookup.duplicated(
        subset=[
            "RID",
        ],
        keep=False,
    ).any():
        raise SupervisedPopulationProfileError(
            "Resolved APOERES contains duplicate participant keys."
        )

    apoe_lookup[
        "__apoe_genotype_observed"
    ] = apoe_lookup[
        "GENOTYPE"
    ].notna()

    participant_profile = participant_profile.join(
        apoe_lookup
        .drop(
            columns=[
                "GENOTYPE",
            ]
        )
        .set_index(
            "RID",
        ),
        how="left",
        validate="one_to_one",
    )

    participant_profile[
        "__apoe_genotype_observed"
    ] = participant_profile[
        "__apoe_genotype_observed"
    ].astype("boolean").fillna(False).astype(bool)

    modality_summaries: list[
        ModalityAvailabilitySummary
    ] = []

    for source_name in INPUT_MODALITY_TOKENS:
        availability_column = availability_columns[
            source_name
        ]
        conflict_column = conflict_columns[
            source_name
        ]
        empty_content_column = empty_content_columns[
            source_name
        ]

        available_visits = int(
            anchor[
                availability_column
            ].sum()
        )

        conflict_visits = int(
            anchor[
                conflict_column
            ].sum()
        )

        unavailable_without_record = int(
            len(anchor)
            - available_visits
            - conflict_visits
        )

        if unavailable_without_record < 0:
            raise SupervisedPopulationProfileError(
                f"{source_name} availability accounting produced a "
                "negative absent-record count."
            )

        modality_summaries.append(
            ModalityAvailabilitySummary(
                source_name=source_name,
                grain=SOURCE_CONTRACTS[
                    source_name
                ].grain,
                anchor_visits_available=(
                    available_visits
                ),
                anchor_visits_unavailable_due_to_conflict=(
                    conflict_visits
                ),
                anchor_visits_unavailable_without_resolved_record=(
                    unavailable_without_record
                ),
                anchor_participants_any_available=int(
                    participant_profile[
                        participant_any_columns[
                            source_name
                        ]
                    ].sum()
                ),
                anchor_participants_first_visit_available=int(
                    participant_profile[
                        participant_first_columns[
                            source_name
                        ]
                    ].sum()
                ),
                anchor_participants_all_visits_available=int(
                    participant_profile[
                        participant_all_columns[
                            source_name
                        ]
                    ].sum()
                ),
                anchor_participants_affected_by_conflict=int(
                    participant_profile[
                        participant_conflict_columns[
                            source_name
                        ]
                    ].sum()
                ),
                anchor_visits_resolved_but_no_substantive_observation=int(
                    anchor[
                        empty_content_column
                    ].sum()
                ),
                anchor_participants_resolved_but_no_substantive_observation=int(
                    participant_profile[
                        participant_empty_columns[
                            source_name
                        ]
                    ].sum()
                ),
            )
        )

    visit_count_counter = Counter(
        int(value)
        for value in participant_profile[
            "__eligible_visit_count"
        ].tolist()
    )

    longitudinal_counter = Counter(
        participant_profile[
            "__longitudinal_class"
        ].astype("string").tolist()
    )

    first_order_basis_counter = Counter(
        participant_profile[
            "__first_visit_order_basis"
        ].astype("string").tolist()
    )

    visit_available_count_counter = Counter(
        int(value)
        for value in anchor[
            "__available_modality_count"
        ].tolist()
    )

    participant_any_count_counter = Counter(
        int(value)
        for value in participant_profile[
            "__any_available_modality_count"
        ].tolist()
    )

    participant_first_count_counter = Counter(
        int(value)
        for value in participant_profile[
            "__first_available_modality_count"
        ].tolist()
    )

    participant_all_count_counter = Counter(
        int(value)
        for value in participant_profile[
            "__all_available_modality_count"
        ].tolist()
    )

    visit_combination_counter = Counter(
        anchor[
            "__modality_combination"
        ].astype("string").tolist()
    )

    participant_any_combination_counter = Counter(
        participant_profile[
            "__any_modality_combination"
        ].astype("string").tolist()
    )

    participant_first_combination_counter = Counter(
        participant_profile[
            "__first_modality_combination"
        ].astype("string").tolist()
    )

    visit_completeness_counter = Counter(
        classify_input_completeness(
            int(value)
        )
        for value in anchor[
            "__available_modality_count"
        ].tolist()
    )

    participant_any_completeness_counter = Counter(
        classify_input_completeness(
            int(value)
        )
        for value in participant_profile[
            "__any_available_modality_count"
        ].tolist()
    )

    participant_first_completeness_counter = Counter(
        classify_input_completeness(
            int(value)
        )
        for value in participant_profile[
            "__first_available_modality_count"
        ].tolist()
    )

    participant_all_completeness_counter = Counter(
        classify_input_completeness(
            int(value)
        )
        for value in participant_profile[
            "__all_available_modality_count"
        ].tolist()
    )

    first_diagnosis_status_counter = Counter(
        participant_profile[
            "__first_diagnosis_status"
        ].astype("string").tolist()
    )

    first_diagnosis_category_counter = Counter(
        participant_profile.loc[
            participant_profile[
                "__first_diagnosis_category"
            ].notna(),
            "__first_diagnosis_category",
        ].astype("string").tolist()
    )

    first_diagnosis_position_counter = Counter(
        str(
            int(value)
        )
        for value in participant_profile[
            "__eligible_visit_position"
        ].dropna().tolist()
    )

    apoe_available_mask = participant_profile[
        participant_any_columns[
            "APOERES"
        ]
    ].astype(bool)

    apoe_conflict_mask = participant_profile[
        participant_conflict_columns[
            "APOERES"
        ]
    ].astype(bool)

    apoe_genotype_observed_mask = participant_profile[
        "__apoe_genotype_observed"
    ].astype(bool)

    if (
        apoe_genotype_observed_mask
        & ~apoe_available_mask
    ).any():
        raise SupervisedPopulationProfileError(
            "Observed APOE genotype was found without a resolved APOERES "
            "participant record."
        )

    summary = SupervisedPopulationProfileSummary(
        anchor_visits=int(
            len(anchor)
        ),
        anchor_participants=int(
            len(participant_profile)
        ),
        modality_summaries=tuple(
            modality_summaries
        ),
        eligible_visit_count_min=int(
            participant_profile[
                "__eligible_visit_count"
            ].min()
        ),
        eligible_visit_count_median=float(
            participant_profile[
                "__eligible_visit_count"
            ].median()
        ),
        eligible_visit_count_max=int(
            participant_profile[
                "__eligible_visit_count"
            ].max()
        ),
        eligible_visit_count_distribution=counter_to_sorted_tuple(
            visit_count_counter
        ),
        longitudinal_class_counts=counter_to_sorted_tuple(
            longitudinal_counter
        ),
        first_visit_order_basis_counts=counter_to_sorted_tuple(
            first_order_basis_counter
        ),
        visit_available_modality_count_distribution=counter_to_sorted_tuple(
            visit_available_count_counter
        ),
        participant_any_available_modality_count_distribution=counter_to_sorted_tuple(
            participant_any_count_counter
        ),
        participant_first_available_modality_count_distribution=counter_to_sorted_tuple(
            participant_first_count_counter
        ),
        participant_all_available_modality_count_distribution=counter_to_sorted_tuple(
            participant_all_count_counter
        ),
        visit_modality_combination_counts=counter_to_sorted_tuple(
            visit_combination_counter
        ),
        participant_any_modality_combination_counts=counter_to_sorted_tuple(
            participant_any_combination_counter
        ),
        participant_first_modality_combination_counts=counter_to_sorted_tuple(
            participant_first_combination_counter
        ),
        visit_input_completeness_counts=counter_to_sorted_tuple(
            visit_completeness_counter
        ),
        participant_any_input_completeness_counts=counter_to_sorted_tuple(
            participant_any_completeness_counter
        ),
        participant_first_input_completeness_counts=counter_to_sorted_tuple(
            participant_first_completeness_counter
        ),
        participant_all_input_completeness_counts=counter_to_sorted_tuple(
            participant_all_completeness_counter
        ),
        first_diagnosis_status_counts=counter_to_sorted_tuple(
            first_diagnosis_status_counter
        ),
        first_diagnosis_category_counts=counter_to_sorted_tuple(
            first_diagnosis_category_counter
        ),
        first_diagnosis_eligible_visit_position_counts=counter_to_sorted_tuple(
            first_diagnosis_position_counter
        ),
        apoe_resolved_participants=int(
            apoe_available_mask.sum()
        ),
        apoe_genotype_observed_participants=int(
            apoe_genotype_observed_mask.sum()
        ),
        apoe_resolved_but_genotype_missing_participants=int(
            (
                apoe_available_mask
                & ~apoe_genotype_observed_mask
            ).sum()
        ),
        apoe_unresolved_conflict_participants=int(
            apoe_conflict_mask.sum()
        ),
        apoe_without_resolved_record_participants=int(
            (
                ~apoe_available_mask
                & ~apoe_conflict_mask
            ).sum()
        ),
        ptgender_consensus=ptgender_summary,
    )

    if (
        participant_profile[
            "__eligible_visit_count"
        ].sum()
        != len(anchor)
    ):
        raise SupervisedPopulationProfileError(
            "Supervised anchor visit accounting failed."
        )

    if len(summary.modality_summaries) != len(
        INPUT_MODALITY_TOKENS
    ):
        raise SupervisedPopulationProfileError(
            "Supervised modality-summary registry accounting failed."
        )

    for modality_summary in summary.modality_summaries:
        if (
            modality_summary.anchor_visits_available
            + modality_summary.anchor_visits_unavailable_due_to_conflict
            + modality_summary.anchor_visits_unavailable_without_resolved_record
            != summary.anchor_visits
        ):
            raise SupervisedPopulationProfileError(
                f"{modality_summary.source_name} anchor-visit "
                "availability accounting failed."
            )

    for column in (
        "__available_modality_count",
    ):
        if not anchor[column].between(
            0,
            len(INPUT_MODALITY_TOKENS),
            inclusive="both",
        ).all():
            raise SupervisedPopulationProfileError(
                f"Anchor modality count is outside valid bounds: {column}"
            )

    for column in (
        "__any_available_modality_count",
        "__first_available_modality_count",
        "__all_available_modality_count",
    ):
        if not participant_profile[column].between(
            0,
            len(INPUT_MODALITY_TOKENS),
            inclusive="both",
        ).all():
            raise SupervisedPopulationProfileError(
                "Participant modality count is outside valid bounds: "
                f"{column}"
            )

    observed_diagnosis_positions = participant_profile.loc[
        participant_profile[
            "__eligible_visit_position"
        ].notna(),
        [
            "__eligible_visit_position",
            "__eligible_visit_count",
        ],
    ]

    if (
        observed_diagnosis_positions[
            "__eligible_visit_position"
        ]
        > observed_diagnosis_positions[
            "__eligible_visit_count"
        ]
    ).any():
        raise SupervisedPopulationProfileError(
            "First eligible diagnosis position exceeds participant "
            "eligible-visit count."
        )

    participant_profile = participant_profile.reset_index()

    # Dates are required only for deterministic ordering and are deliberately
    # removed from the in-memory shareable-stage objects after use.
    anchor = anchor.drop(
        columns=[
            "EXAMDATE",
            "__examdate_parsed",
            "__protocol_month_missing",
            "__examdate_missing",
            "__dxsum_resolved_metadata",
            "DIAGNOSIS",
        ],
    )

    return (
        anchor,
        participant_profile,
        summary,
    )
