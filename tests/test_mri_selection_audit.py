import pandas as pd
import pytest

from adni_benchmark.mri_selection_audit import (
    MRISelectionAuditError,
    build_mri_selection_audit,
    canonicalize_audit_series,
    count_visit_groups_with_multiple_values,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
                "2",
                "3",
            ],
            "VISCODE2": [
                "bl",
                "bl",
                "m06",
                pd.NA,
            ],
            "IMAGEUID": [
                "100",
                "101",
                "200",
                "300",
            ],
            "PHASE": [
                "ADNI1",
                "adni1",
                "ADNI2",
                "ADNI3",
            ],
            "FIELD_STRENGTH": [
                "1.5T",
                "3T",
                "3T",
                "3T",
            ],
            "FSVER": [
                "7.2.0",
                "7.2.0",
                "7.2.0",
                "7.4.1",
            ],
            "STATUS": [
                "PARTIAL",
                "COMPLETE",
                "COMPLETE",
                "PARTIAL",
            ],
            "OVERALLQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "TEMPQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "FRONTQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "PARQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "INSULAQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "OCCQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "BGQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "CWMQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "VENTQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "HIPPOQC": [
                pd.NA,
                "PASS",
                "PASS",
                pd.NA,
            ],
            "ST1SV": [
                "1",
                "1",
                "1",
                pd.NA,
            ],
            "ST2SV": [
                "2",
                "2",
                pd.NA,
                pd.NA,
            ],
            "ST8SV": [
                "3",
                pd.NA,
                "3",
                pd.NA,
            ],
        },
        dtype="string",
    )


def test_canonicalize_audit_series() -> None:
    series = pd.Series(
        [
            " pass ",
            "Partial",
            pd.NA,
        ],
        dtype="string",
    )

    result = canonicalize_audit_series(
        series
    )

    assert result.tolist() == [
        "PASS",
        "PARTIAL",
        "<MISSING>",
    ]


def test_multiple_values_within_visit_are_counted() -> None:
    frame = make_frame()

    result = count_visit_groups_with_multiple_values(
        frame=frame,
        column="FIELD_STRENGTH",
    )

    assert result == 1


def test_case_only_difference_is_not_disagreement() -> None:
    frame = make_frame()

    result = count_visit_groups_with_multiple_values(
        frame=frame,
        column="PHASE",
    )

    assert result == 0


def test_selection_audit_counts_visit_candidates() -> None:
    frame = make_frame()

    summary = build_mri_selection_audit(
        frame=frame,
        measure_columns=(
            "ST1SV",
            "ST2SV",
            "ST8SV",
        ),
        target_columns=(
            "ST1SV",
            "ST2SV",
        ),
    )

    assert summary.n_rows == 4
    assert summary.rows_with_missing_visit_key == 1
    assert summary.valid_visit_rows == 3
    assert summary.valid_visit_groups == 2
    assert summary.duplicate_visit_groups == 1
    assert summary.max_candidates_per_visit == 2

    assert summary.candidate_count_distribution == (
        (1, 1),
        (2, 1),
    )


def test_selection_audit_counts_completeness() -> None:
    frame = make_frame()

    summary = build_mri_selection_audit(
        frame=frame,
        measure_columns=(
            "ST1SV",
            "ST2SV",
            "ST8SV",
        ),
        target_columns=(
            "ST1SV",
            "ST2SV",
        ),
    )

    assert summary.rows_complete_measure_catalog == 1
    assert summary.rows_complete_target_catalog == 2

    assert summary.measure_nonmissing_min == 0
    assert summary.measure_nonmissing_max == 3


def test_selection_audit_detects_field_strength_disagreement() -> None:
    frame = make_frame()

    summary = build_mri_selection_audit(
        frame=frame,
        measure_columns=(
            "ST1SV",
            "ST2SV",
            "ST8SV",
        ),
        target_columns=(
            "ST1SV",
            "ST2SV",
        ),
    )

    assert (
        summary.groups_with_multiple_field_strength_values
        == 1
    )

    assert (
        summary.groups_with_multiple_phase_values
        == 0
    )

    assert (
        summary.groups_with_multiple_fsver_values
        == 0
    )


def test_target_catalog_must_be_subset_of_measure_catalog() -> None:
    frame = make_frame()

    with pytest.raises(
        MRISelectionAuditError,
        match="not a subset",
    ):
        build_mri_selection_audit(
            frame=frame,
            measure_columns=(
                "ST1SV",
                "ST2SV",
            ),
            target_columns=(
                "ST1SV",
                "ST999SV",
            ),
        )


def test_target_catalog_must_be_subset_of_measure_catalog() -> None:
    frame = make_frame()

    frame["ST999SV"] = "1"

    with pytest.raises(
        MRISelectionAuditError,
        match="not a subset",
    ):
        build_mri_selection_audit(
            frame=frame,
            measure_columns=(
                "ST1SV",
                "ST2SV",
            ),
            target_columns=(
                "ST1SV",
                "ST999SV",
            ),
        )