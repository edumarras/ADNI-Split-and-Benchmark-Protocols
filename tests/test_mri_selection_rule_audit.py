import pandas as pd
import pytest

from adni_benchmark.mri_selection_rule_audit import (
    MRISelectionRuleAuditError,
    build_mri_selection_rule_audit,
)


REGIONAL_QC_COLUMNS = (
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


MEASURE_COLUMNS = (
    "ST1SV",
    "ST2SV",
    "ST8SV",
    "ST68SV",
)


TARGET_COLUMNS = (
    "ST1SV",
    "ST2SV",
)


def make_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
                "2",
                "2",
                "3",
            ],
            "VISCODE2": [
                "bl",
                "bl",
                "m06",
                "m06",
                "m12",
            ],
            "STATUS": [
                "COMPLETE",
                "PARTIAL",
                "COMPLETE",
                "PARTIAL",
                "COMPLETE",
            ],
            "OVERALLQC": [
                "PASS",
                pd.NA,
                "PASS",
                pd.NA,
                "PASS",
            ],
            "ST1SV": [
                "1",
                "1",
                "1",
                "1",
                "1",
            ],
            "ST2SV": [
                pd.NA,
                "2",
                "2",
                "2",
                "2",
            ],
            "ST8SV": [
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "ST68SV": [
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
        },
        dtype="string",
    )

    for column in REGIONAL_QC_COLUMNS:
        frame[column] = pd.Series(
            [
                "PASS",
                pd.NA,
                "PASS",
                pd.NA,
                "PASS",
            ],
            dtype="string",
        )

    return frame


def test_rule_audit_counts_duplicate_groups() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert summary.duplicate_visit_groups == 2


def test_rule_audit_detects_status_disagreement() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.duplicate_groups_with_status_disagreement
        == 2
    )


def test_rule_audit_detects_status_target_tension() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.duplicate_groups_with_status_target_tension
        == 1
    )


def test_rule_audit_detects_both_statuses_target_complete() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.duplicate_groups_where_both_statuses_offer_target_complete
        == 1
    )


def test_rule_audit_detects_mixed_target_completeness() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.duplicate_groups_with_mixed_target_completeness
        == 1
    )

    assert (
        summary.duplicate_groups_with_target_nonmissing_disagreement
        == 1
    )


def test_rule_audit_counts_complete_target_candidates() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.duplicate_groups_with_at_least_one_target_complete
        == 2
    )

    assert (
        summary.duplicate_groups_with_multiple_target_complete_candidates
        == 1
    )

    assert (
        summary.duplicate_groups_without_target_complete_candidate
        == 0
    )


def test_rule_audit_counts_target_complete_but_measure_incomplete() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.rows_target_complete_but_measure_incomplete
        == 4
    )


def test_rule_audit_profiles_excluded_measure_missingness() -> None:
    summary = build_mri_selection_rule_audit(
        frame=make_frame(),
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    counts = {
        column: (
            missing_rows,
            observed_rows,
        )
        for (
            column,
            missing_rows,
            observed_rows,
        ) in summary.excluded_measure_missing_counts
    }

    assert counts["ST8SV"] == (
        5,
        0,
    )

    assert counts["ST68SV"] == (
        5,
        0,
    )


def test_missing_required_column_aborts() -> None:
    frame = make_frame().drop(
        columns=["OVERALLQC"]
    )

    with pytest.raises(
        MRISelectionRuleAuditError,
        match="missing columns",
    ):
        build_mri_selection_rule_audit(
            frame=frame,
            measure_columns=MEASURE_COLUMNS,
            target_columns=TARGET_COLUMNS,
        )


def test_empty_measure_catalog_aborts() -> None:
    frame = make_frame()

    with pytest.raises(
        MRISelectionRuleAuditError,
        match="measurement catalog is empty",
    ):
        build_mri_selection_rule_audit(
            frame=frame,
            measure_columns=(),
            target_columns=TARGET_COLUMNS,
        )


def test_fixed_exclusions_must_exist_in_measure_catalog() -> None:
    frame = make_frame()

    with pytest.raises(
        MRISelectionRuleAuditError,
        match="ST68SV",
    ):
        build_mri_selection_rule_audit(
            frame=frame,
            measure_columns=(
                "ST1SV",
                "ST2SV",
                "ST8SV",
            ),
            target_columns=TARGET_COLUMNS,
        )