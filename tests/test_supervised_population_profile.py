import pandas as pd
import pytest

from adni_benchmark.source_schema import SOURCE_CONTRACTS
from adni_benchmark.supervised_population_profile import (
    INPUT_MODALITY_TOKENS,
    SupervisedPopulationProfileError,
    build_modality_combination_label,
    build_ptgender_consensus_profile,
    build_supervised_population_profile,
    classify_input_completeness,
    serialize_safe_count_distribution,
    validate_resolved_unresolved_key_disjoint,
)


def make_visit_frame(rows, value_column="VALUE") -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["RID", "VISCODE2", value_column],
    ).astype(
        {
            "RID": "string",
            "VISCODE2": "string",
            value_column: "string",
        }
    )


def make_empty_unresolved(source_name: str) -> pd.DataFrame:
    contract = SOURCE_CONTRACTS[source_name]
    return pd.DataFrame(
        columns=[
            *contract.key_columns,
            "source_name",
            "n_source_rows",
            "n_candidate_rows",
            "resolution_status",
        ]
    )


def make_profile_inputs():
    mri = pd.DataFrame(
        {
            "RID": ["XX11", "XX11", "XX22", "XX33"],
            "VISCODE2": ["m12", "bl", "bl", "bl"],
            "EXAMDATE": [
                "2020-02-01",
                "2020-01-01",
                "2020-03-01",
                "2020-04-01",
            ],
            "__target_complete": [True, True, True, False],
        }
    ).astype(
        {
            "RID": "string",
            "VISCODE2": "string",
            "EXAMDATE": "string",
            "__target_complete": "boolean",
        }
    )

    resolved = {
        "DXSUM": pd.DataFrame(
            {
                "RID": ["XX11", "XX11", "XX22"],
                "VISCODE2": ["bl", "m12", "bl"],
                "DIAGNOSIS": [pd.NA, "2", "1"],
            }
        ).astype(
            {
                "RID": "string",
                "VISCODE2": "string",
                "DIAGNOSIS": "string",
            }
        ),
        "ADAS": make_visit_frame(
            [
                ("XX11", "bl", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "CDR": make_visit_frame(
            [
                ("XX11", "bl", "1"),
                ("XX11", "m12", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "FAQ": make_visit_frame(
            [
                ("XX11", "bl", pd.NA),
                ("XX11", "m12", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "MMSE": make_visit_frame(
            [
                ("XX11", "bl", "1"),
                ("XX11", "m12", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "MOCA": make_visit_frame(
            [
                ("XX11", "bl", "1"),
                ("XX11", "m12", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "NEUROBAT": make_visit_frame(
            [
                ("XX11", "bl", "1"),
                ("XX11", "m12", "1"),
                ("XX22", "bl", "1"),
            ]
        ),
        "PTDEMOG": pd.DataFrame(
            {
                "RID": ["XX11"],
                "VISCODE2": ["bl"],
                "PHASE": ["ADNI1"],
                "PTGENDER": ["A"],
                "OTHER": ["1"],
            }
        ).astype("string"),
        "APOERES": pd.DataFrame(
            {
                "RID": ["XX11"],
                "GENOTYPE": ["XX11AABBCC"],
            }
        ).astype("string"),
    }


    source_tables = {
        source_name: resolved[source_name].copy()
        for source_name in INPUT_MODALITY_TOKENS
    }

    source_tables["PTDEMOG"] = pd.DataFrame(
        {
            "RID": ["XX11", "XX11", "XX22", "XX22"],
            "VISCODE2": ["bl", "m12", "bl", "m12"],
            "PHASE": ["ADNI1", "ADNI1", "ADNI1", "ADNI1"],
            "PTGENDER": ["A", "A", "B", "B"],
            "OTHER": ["1", "1", "1", "2"],
        }
    ).astype("string")

    unresolved = {
        source_name: make_empty_unresolved(source_name)
        for source_name in SOURCE_CONTRACTS
        if source_name != "UCSFFSX7"
    }

    unresolved["ADAS"] = pd.DataFrame(
        {
            "RID": ["XX11"],
            "VISCODE2": ["m12"],
            "source_name": ["ADAS"],
            "n_source_rows": [2],
            "n_candidate_rows": [2],
            "resolution_status": ["unresolved_conflict"],
        }
    )

    unresolved["PTDEMOG"] = pd.DataFrame(
        {
            "RID": ["XX22"],
            "source_name": ["PTDEMOG"],
            "n_source_rows": [2],
            "n_candidate_rows": [2],
            "resolution_status": ["unresolved_conflict"],
        }
    )

    return source_tables, resolved, unresolved, mri


def test_classify_input_completeness() -> None:
    assert classify_input_completeness(0) == "no_input_modality_available"
    assert classify_input_completeness(1) == "partially_available_inputs"
    assert (
        classify_input_completeness(len(INPUT_MODALITY_TOKENS))
        == "all_input_modalities_available"
    )


def test_classify_input_completeness_rejects_invalid_count() -> None:
    with pytest.raises(ValueError):
        classify_input_completeness(-1)

    with pytest.raises(ValueError):
        classify_input_completeness(len(INPUT_MODALITY_TOKENS) + 1)


def test_modality_combination_uses_registry_order() -> None:
    columns = {
        source_name: f"available_{source_name.casefold()}"
        for source_name in INPUT_MODALITY_TOKENS
    }

    row = pd.Series(
        {
            column: source_name in {"ADAS", "MMSE", "APOERES"}
            for source_name, column in columns.items()
        }
    )

    assert (
        build_modality_combination_label(row, columns)
        == "ADAS+MMSE+APOERES"
    )


def test_safe_distribution_suppresses_small_cells_and_pseudonymizes() -> None:
    result = serialize_safe_count_distribution(
        (("SECRET_A", 3), ("SECRET_B", 7)),
        pseudonymize_labels=True,
    )

    assert result["total_observations"] == 10
    assert result["reported_counts"] == {"category_002": 7}
    assert result["small_cell_policy"]["suppressed_category_count"] == 1
    assert result["small_cell_policy"]["suppressed_observation_count"] == 3
    assert result["labels_pseudonymized"] is True


def test_resolved_and_unresolved_keys_must_be_disjoint() -> None:
    resolved = make_visit_frame([("XX11", "bl", "1")])
    unresolved = pd.DataFrame(
        {
            "RID": ["XX11"],
            "VISCODE2": ["bl"],
        }
    )

    with pytest.raises(
        SupervisedPopulationProfileError,
        match="classified as both resolved and unresolved",
    ):
        validate_resolved_unresolved_key_disjoint(
            resolved_frame=resolved,
            unresolved_frame=unresolved,
            contract=SOURCE_CONTRACTS["ADAS"],
        )


def test_ptgender_consensus_can_exist_without_resolved_ptdemog() -> None:
    source = pd.DataFrame(
        {
            "RID": ["XX11", "XX11", "XX22", "XX22"],
            "PTGENDER": ["A", pd.NA, "B", "B"],
        }
    ).astype({"RID": "string", "PTGENDER": "string"})

    profile, summary = build_ptgender_consensus_profile(
        ptdemog_source=source,
        anchor_participant_ids=["XX11", "XX22"],
        resolved_ptdemog_participant_ids={"XX11"},
    )

    assert len(profile) == 2
    assert summary.participants_with_consensus == 2
    assert summary.consistent_participants_with_mixed_missingness == 1
    assert summary.participants_with_consensus_and_resolved_ptdemog == 1
    assert summary.participants_with_consensus_without_resolved_ptdemog == 1


def test_supervised_anchor_retains_target_complete_visit_with_input_conflict() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    anchor, participant_profile, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    assert len(anchor) == 3
    assert len(participant_profile) == 2
    assert summary.anchor_visits == 3
    assert summary.anchor_participants == 2

    adas = next(
        item
        for item in summary.modality_summaries
        if item.source_name == "ADAS"
    )

    assert adas.anchor_visits_available == 2
    assert adas.anchor_visits_unavailable_due_to_conflict == 1
    assert adas.anchor_visits_unavailable_without_resolved_record == 0


def test_participant_level_conflict_marks_modality_unavailable_without_dropping_visit() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    anchor, _, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    ptdemog = next(
        item
        for item in summary.modality_summaries
        if item.source_name == "PTDEMOG"
    )

    assert len(anchor) == 3
    assert ptdemog.anchor_visits_available == 2
    assert ptdemog.anchor_visits_unavailable_due_to_conflict == 1
    assert ptdemog.anchor_participants_affected_by_conflict == 1


def test_resolved_record_with_no_substantive_content_is_not_available() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    _, _, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    faq = next(
        item
        for item in summary.modality_summaries
        if item.source_name == "FAQ"
    )

    assert faq.anchor_visits_available == 2
    assert faq.anchor_visits_resolved_but_no_substantive_observation == 1


def test_first_eligible_visit_uses_protocol_month_not_source_order() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    anchor, participant_profile, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    first_xx11 = anchor.loc[
        (anchor["RID"] == "XX11")
        & anchor["__first_eligible_visit"]
    ]

    assert first_xx11["VISCODE2"].tolist() == ["bl"]
    assert summary.first_visit_order_basis_counts == (("protocol_month", 2),)
    assert "EXAMDATE" not in anchor.columns
    assert len(participant_profile) == 2


def test_first_observed_diagnosis_can_occur_after_first_eligible_visit() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    _, participant_profile, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    xx11 = participant_profile.loc[
        participant_profile["RID"] == "XX11"
    ].iloc[0]

    assert xx11["__eligible_visit_position"] == 2
    assert (
        xx11["__first_diagnosis_status"]
        == "diagnosis_first_observed_after_first_eligible_visit"
    )
    assert dict(summary.first_diagnosis_status_counts) == {
        "diagnosis_first_observed_after_first_eligible_visit": 1,
        "diagnosis_observed_at_first_eligible_visit": 1,
    }


def test_apoe_coverage_is_participant_level() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    _, _, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    assert summary.apoe_resolved_participants == 1
    assert summary.apoe_genotype_observed_participants == 1
    assert summary.apoe_resolved_but_genotype_missing_participants == 0
    assert summary.apoe_unresolved_conflict_participants == 0
    assert summary.apoe_without_resolved_record_participants == 1


def test_manifest_pseudonymizes_sensitive_category_labels() -> None:
    source_tables, resolved, unresolved, mri = make_profile_inputs()

    _, _, summary = build_supervised_population_profile(
        source_tables=source_tables,
        resolved_non_mri_sources=resolved,
        unresolved_non_mri_sources=unresolved,
        resolved_mri_source=mri,
    )

    manifest = summary.to_manifest_record()

    diagnosis_distribution = manifest[
        "first_eligible_diagnosis"
    ]["category_distribution"]

    ptgender_distribution = manifest[
        "ptgender_auxiliary_consensus"
    ]["consistent_category_distribution"]

    assert diagnosis_distribution["labels_pseudonymized"] is True
    assert ptgender_distribution["labels_pseudonymized"] is True

    assert all(
        label.startswith("category_")
        for label in diagnosis_distribution["reported_counts"]
    )

    assert all(
        label.startswith("category_")
        for label in ptgender_distribution["reported_counts"]
    )
