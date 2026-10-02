import numpy as np
import pandas as pd
import pytest

from adni_benchmark.mri_secondary_tie_audit import (
    MRISecondaryTieAuditError,
    build_mri_secondary_tie_audit,
    convert_mri_targets_to_numeric,
    mri_target_vectors_are_equivalent,
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


def make_candidate(
    rid: str,
    viscode2: str,
    imageuid: str,
    target1: object = "1.0",
    target2: object = "2.0",
    excluded1: object = pd.NA,
    excluded2: object = pd.NA,
    status: object = "PARTIAL",
    overall_qc: object = pd.NA,
    field_strength: object = "3T",
    regional_values: object = "PASS",
) -> dict[str, object]:
    record = {
        "RID": rid,
        "VISCODE2": viscode2,
        "IMAGEUID": imageuid,
        "STATUS": status,
        "OVERALLQC": overall_qc,
        "FIELD_STRENGTH": field_strength,
        "ST1SV": target1,
        "ST2SV": target2,
        "ST8SV": excluded1,
        "ST68SV": excluded2,
    }

    for column in REGIONAL_QC_COLUMNS:
        record[column] = regional_values

    return record


def make_frame(
    records: list[dict[str, object]],
) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        dtype="string",
    )


def test_numeric_conversion_preserves_missing_values() -> None:
    frame = pd.DataFrame(
        {
            "ST1SV": [
                "1.5",
                pd.NA,
            ],
            "ST2SV": [
                "2",
                "3",
            ],
        },
        dtype="string",
    )

    result = convert_mri_targets_to_numeric(
        frame=frame,
        target_columns=(
            "ST1SV",
            "ST2SV",
        ),
    )

    assert result.loc[0, "ST1SV"] == 1.5
    assert np.isnan(
        result.loc[1, "ST1SV"]
    )


def test_non_numeric_target_aborts() -> None:
    frame = pd.DataFrame(
        {
            "ST1SV": [
                "NOT_NUMERIC",
            ],
        },
        dtype="string",
    )

    with pytest.raises(
        MRISecondaryTieAuditError,
        match="non-numeric",
    ):
        convert_mri_targets_to_numeric(
            frame=frame,
            target_columns=(
                "ST1SV",
            ),
        )


def test_target_vectors_use_numeric_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "ST1SV": [
                1.0,
                1.0 + 1e-7,
            ],
            "ST2SV": [
                2.0,
                2.0,
            ],
        }
    )

    assert mri_target_vectors_are_equivalent(
        frame
    )


def test_target_vectors_detect_real_difference() -> None:
    frame = pd.DataFrame(
        {
            "ST1SV": [
                1.0,
                1.1,
            ],
            "ST2SV": [
                2.0,
                2.0,
            ],
        }
    )

    assert not mri_target_vectors_are_equivalent(
        frame
    )


def test_measure_completeness_can_resolve_tie() -> None:
    frame = make_frame(
        [
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="100",
                excluded1="3",
            ),
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="101",
            ),
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert summary.primary_tie_groups == 1
    assert summary.resolved_by_measure_completeness == 1
    assert summary.remaining_non_equivalent_conflicts == 0


def test_qc_availability_can_resolve_tie() -> None:
    first = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="100",
        overall_qc="PARTIAL",
        regional_values="PASS",
    )

    second = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="101",
        overall_qc=pd.NA,
        regional_values=pd.NA,
    )

    frame = make_frame(
        [
            first,
            second,
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert summary.resolved_by_measure_completeness == 0
    assert summary.resolved_by_qc_availability == 1


def test_unique_overall_pass_can_resolve_tie() -> None:
    first = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="100",
        overall_qc="PASS",
        regional_values="PASS",
    )

    second = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="101",
        overall_qc="PARTIAL",
        regional_values="PASS",
    )

    frame = make_frame(
        [
            first,
            second,
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert summary.resolved_by_qc_availability == 0
    assert summary.resolved_by_unique_overall_pass == 1


def test_fewer_regional_failures_can_resolve_tie() -> None:
    first = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="100",
        overall_qc="PARTIAL",
        regional_values="PASS",
    )

    second = make_candidate(
        rid="1",
        viscode2="bl",
        imageuid="101",
        overall_qc="PARTIAL",
        regional_values="PASS",
    )

    second["TEMPQC"] = "FAIL"

    frame = make_frame(
        [
            first,
            second,
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert summary.resolved_by_fewer_regional_failures == 1


def test_numerically_equivalent_candidates_are_classified() -> None:
    frame = make_frame(
        [
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="100",
                overall_qc="PARTIAL",
            ),
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="101",
                overall_qc="PARTIAL",
            ),
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.equivalent_after_secondary_filters
        == 1
    )

    assert (
        summary.remaining_non_equivalent_conflicts
        == 0
    )


def test_non_equivalent_candidates_remain_conflicted() -> None:
    frame = make_frame(
        [
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="100",
                target1="1.0",
                overall_qc="PARTIAL",
            ),
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="101",
                target1="9.0",
                overall_qc="PARTIAL",
            ),
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.remaining_non_equivalent_conflicts
        == 1
    )


def test_field_strength_is_audited_but_not_used_to_resolve() -> None:
    frame = make_frame(
        [
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="100",
                target1="1.0",
                field_strength="1.5T",
                overall_qc="PARTIAL",
            ),
            make_candidate(
                rid="1",
                viscode2="bl",
                imageuid="101",
                target1="9.0",
                field_strength="3T",
                overall_qc="PARTIAL",
            ),
        ]
    )

    summary = build_mri_secondary_tie_audit(
        frame=frame,
        measure_columns=MEASURE_COLUMNS,
        target_columns=TARGET_COLUMNS,
    )

    assert (
        summary.primary_ties_with_field_strength_disagreement
        == 1
    )

    assert (
        summary.remaining_non_equivalent_conflicts
        == 1
    )