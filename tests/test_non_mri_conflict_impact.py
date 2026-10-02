import pandas as pd
import pytest

from adni_benchmark.non_mri_conflict_impact import (
    NonMRIConflictImpactError,
    build_non_mri_conflict_impact_audit,
    count_distinct_nonmissing_equivalence_classes,
    profile_candidate_group_disagreements,
    reconstruct_unresolved_candidate_group,
)
from adni_benchmark.non_mri_resolution import (
    NON_MRI_SOURCE_TOKENS,
)
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
)


def make_source_tables() -> dict[str, pd.DataFrame]:
    tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    for source_name in NON_MRI_SOURCE_TOKENS:
        contract = (
            SOURCE_CONTRACTS[
                source_name
            ]
        )

        if source_name == "PTDEMOG":
            frame = pd.DataFrame(
                {
                    "RID": [
                        "XX11",
                    ],
                    "VISCODE2": [
                        "bl",
                    ],
                    "PHASE": [
                        "ADNI1",
                    ],
                    "PTGENDER": [
                        "A",
                    ],
                },
                dtype="string",
            )

        elif source_name == "APOERES":
            frame = pd.DataFrame(
                {
                    "RID": [
                        "XX11",
                    ],
                    "GENOTYPE": [
                        "XX11AABBCC",
                    ],
                },
                dtype="string",
            )

        else:
            frame = pd.DataFrame(
                {
                    "RID": [
                        "XX11",
                    ],
                    "VISCODE2": [
                        "bl",
                    ],
                    "VALUE": [
                        "1",
                    ],
                },
                dtype="string",
            )

        tables[
            source_name
        ] = frame

    return tables


def make_empty_unresolved_tables() -> dict[
    str,
    pd.DataFrame,
]:
    unresolved: dict[
        str,
        pd.DataFrame,
    ] = {}

    for source_name in NON_MRI_SOURCE_TOKENS:
        contract = (
            SOURCE_CONTRACTS[
                source_name
            ]
        )

        unresolved[
            source_name
        ] = pd.DataFrame(
            columns=[
                *contract.key_columns,
                "source_name",
                "n_source_rows",
                "n_candidate_rows",
                "resolution_status",
            ]
        )

    return unresolved


def test_equivalence_class_count_uses_resolution_semantics() -> None:
    series = pd.Series(
        [
            "1",
            "1.0",
            "2",
            pd.NA,
        ],
        dtype="string",
    )

    assert (
        count_distinct_nonmissing_equivalence_classes(
            series
        )
        == 2
    )


def test_profile_detects_observed_value_disagreement() -> None:
    frame = pd.DataFrame(
        {
            "A": [
                "1",
                "2",
            ],
        },
        dtype="string",
    )

    (
        observed,
        missingness,
    ) = profile_candidate_group_disagreements(
        candidate_group=frame,
        comparison_columns=(
            "A",
        ),
    )

    assert observed == (
        "A",
    )

    assert missingness == ()


def test_profile_detects_missingness_disagreement() -> None:
    frame = pd.DataFrame(
        {
            "A": [
                "1",
                pd.NA,
            ],
        },
        dtype="string",
    )

    (
        observed,
        missingness,
    ) = profile_candidate_group_disagreements(
        candidate_group=frame,
        comparison_columns=(
            "A",
        ),
    )

    assert observed == ()

    assert missingness == (
        "A",
    )


def test_reconstruct_visit_candidate_group() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
                "XX22",
            ],
            "VISCODE2": [
                "bl",
                "bl",
                "bl",
            ],
            "VALUE": [
                "1",
                "2",
                "3",
            ],
        },
        dtype="string",
    )

    unresolved_record = pd.Series(
        {
            "RID": "XX11",
            "VISCODE2": "bl",
            "n_source_rows": 2,
            "n_candidate_rows": 2,
        }
    )

    result = reconstruct_unresolved_candidate_group(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
        unresolved_record=unresolved_record,
    )

    assert len(
        result
    ) == 2


def test_reconstruct_ptdemog_uses_earliest_protocol_wave() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
                "XX11",
            ],
            "VISCODE2": [
                "sc",
                "bl",
                "m12",
            ],
            "PHASE": [
                "ADNI1",
                "ADNI1",
                "ADNI1",
            ],
            "PTGENDER": [
                "A",
                "B",
                "C",
            ],
        },
        dtype="string",
    )

    unresolved_record = pd.Series(
        {
            "RID": "XX11",
            "n_source_rows": 3,
            "n_candidate_rows": 2,
        }
    )

    result = reconstruct_unresolved_candidate_group(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "PTDEMOG"
        ],
        unresolved_record=unresolved_record,
    )

    assert len(
        result
    ) == 2

    assert set(
        result[
            "VISCODE2"
        ]
    ) == {
        "sc",
        "bl",
    }


def test_visit_conflict_reaches_exact_mri_visit() -> None:
    source_tables = (
        make_source_tables()
    )

    source_tables[
        "ADAS"
    ] = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "VALUE": [
                "1",
                "2",
            ],
        },
        dtype="string",
    )

    unresolved = (
        make_empty_unresolved_tables()
    )

    unresolved[
        "ADAS"
    ] = pd.DataFrame(
        {
            "RID": [
                "XX11",
            ],
            "VISCODE2": [
                "bl",
            ],
            "source_name": [
                "ADAS",
            ],
            "n_source_rows": [
                2,
            ],
            "n_candidate_rows": [
                2,
            ],
            "resolution_status": [
                "unresolved_conflict",
            ],
        }
    )

    mri = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
                "XX22",
            ],
            "VISCODE2": [
                "bl",
                "m12",
                "bl",
            ],
            "__target_complete": [
                True,
                True,
                False,
            ],
        }
    ).astype(
        {
            "RID": "string",
            "VISCODE2": "string",
            "__target_complete": "boolean",
        }
    )

    summary = build_non_mri_conflict_impact_audit(
        source_tables=source_tables,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    assert (
        summary.total_unresolved_groups
        == 1
    )

    assert (
        summary.sources_with_unresolved_groups
        == 1
    )

    assert (
        summary.resolved_mri_visits_affected_by_any_conflict
        == 1
    )

    assert (
        summary.target_complete_mri_visits_affected_by_any_conflict
        == 1
    )


def test_participant_conflict_affects_all_mri_visits_for_rid() -> None:
    source_tables = (
        make_source_tables()
    )

    source_tables[
        "PTDEMOG"
    ] = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
            ],
            "VISCODE2": [
                "bl",
                "sc",
            ],
            "PHASE": [
                "ADNI1",
                "ADNI1",
            ],
            "PTGENDER": [
                "A",
                "B",
            ],
        },
        dtype="string",
    )

    unresolved = (
        make_empty_unresolved_tables()
    )

    unresolved[
        "PTDEMOG"
    ] = pd.DataFrame(
        {
            "RID": [
                "XX11",
            ],
            "source_name": [
                "PTDEMOG",
            ],
            "n_source_rows": [
                2,
            ],
            "n_candidate_rows": [
                2,
            ],
            "resolution_status": [
                "unresolved_conflict",
            ],
        }
    )

    mri = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
                "XX22",
            ],
            "VISCODE2": [
                "bl",
                "m12",
                "bl",
            ],
            "__target_complete": [
                True,
                False,
                True,
            ],
        }
    ).astype(
        {
            "RID": "string",
            "VISCODE2": "string",
            "__target_complete": "boolean",
        }
    )

    summary = build_non_mri_conflict_impact_audit(
        source_tables=source_tables,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    pt_summary = next(
        item
        for item
        in summary.source_summaries
        if item.source_name
        == "PTDEMOG"
    )

    assert (
        pt_summary.unresolved_groups_reaching_resolved_mri
        == 1
    )

    assert (
        pt_summary.affected_resolved_mri_visits
        == 2
    )

    assert (
        pt_summary.affected_resolved_mri_participants
        == 1
    )


def test_duplicate_resolved_mri_keys_abort() -> None:
    source_tables = (
        make_source_tables()
    )

    unresolved = (
        make_empty_unresolved_tables()
    )

    mri = pd.DataFrame(
        {
            "RID": [
                "XX11",
                "XX11",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "__target_complete": [
                True,
                True,
            ],
        }
    )

    with pytest.raises(
        NonMRIConflictImpactError,
        match="duplicate visit keys",
    ):
        build_non_mri_conflict_impact_audit(
            source_tables=source_tables,
            unresolved_non_mri_sources=unresolved,
            resolved_mri_source=mri,
        )


def test_missing_registry_entry_aborts() -> None:
    source_tables = (
        make_source_tables()
    )

    unresolved = (
        make_empty_unresolved_tables()
    )

    del unresolved[
        "ADAS"
    ]

    mri = pd.DataFrame(
        {
            "RID": [
                "XX11",
            ],
            "VISCODE2": [
                "bl",
            ],
            "__target_complete": [
                True,
            ],
        }
    )

    with pytest.raises(
        NonMRIConflictImpactError,
        match="registry mismatch",
    ):
        build_non_mri_conflict_impact_audit(
            source_tables=source_tables,
            unresolved_non_mri_sources=unresolved,
            resolved_mri_source=mri,
        )