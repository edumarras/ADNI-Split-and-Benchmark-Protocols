import pandas as pd
import pytest

from adni_benchmark.mri_catalog import (
    EXPECTED_MRI_MEASURE_COUNT,
    EXPECTED_MRI_TARGET_COUNT,
    MRICatalogError,
    build_mri_catalog,
    calculate_column_catalog_sha256,
)


def make_valid_measure_columns() -> list[str]:
    columns = [
        "ST8SV",
        "ST68SV",
    ]

    index = 1000

    while len(columns) < EXPECTED_MRI_MEASURE_COUNT:
        columns.append(
            f"ST{index}CV"
        )
        index += 1

    return columns


def test_catalog_hash_is_case_normalized() -> None:
    left = calculate_column_catalog_sha256(
        [
            "ST1SV",
            "ST2CV",
        ]
    )

    right = calculate_column_catalog_sha256(
        [
            "st1sv",
            "St2Cv",
        ]
    )

    assert left == right


def test_catalog_hash_is_order_sensitive() -> None:
    first = calculate_column_catalog_sha256(
        [
            "ST1SV",
            "ST2CV",
        ]
    )

    second = calculate_column_catalog_sha256(
        [
            "ST2CV",
            "ST1SV",
        ]
    )

    assert first != second


def test_valid_catalog_produces_323_targets() -> None:
    measures = make_valid_measure_columns()

    frame = pd.DataFrame(
        columns=[
            "RID",
            "VISCODE2",
            *measures,
            "update_stamp",
        ]
    )

    measure_columns, target_columns, summary = (
        build_mri_catalog(frame)
    )

    assert len(measure_columns) == EXPECTED_MRI_MEASURE_COUNT
    assert len(target_columns) == EXPECTED_MRI_TARGET_COUNT

    assert "ST8SV" in measure_columns
    assert "ST68SV" in measure_columns

    assert "ST8SV" not in target_columns
    assert "ST68SV" not in target_columns

    assert summary.n_source_columns == len(frame.columns)
    assert summary.n_measure_columns == 325
    assert summary.n_target_columns == 323
    assert summary.n_non_measure_columns == 3


def test_measure_detection_is_case_insensitive() -> None:
    measures = make_valid_measure_columns()

    measures[2] = measures[2].lower()

    frame = pd.DataFrame(
        columns=measures
    )

    measure_columns, target_columns, _ = (
        build_mri_catalog(frame)
    )

    assert measures[2] in measure_columns
    assert measures[2] in target_columns


def test_source_order_is_preserved() -> None:
    measures = make_valid_measure_columns()

    frame = pd.DataFrame(
        columns=[
            "META",
            *measures,
        ]
    )

    measure_columns, target_columns, _ = (
        build_mri_catalog(frame)
    )

    assert measure_columns == tuple(measures)

    expected_targets = tuple(
        column
        for column in measures
        if column.upper()
        not in {
            "ST8SV",
            "ST68SV",
        }
    )

    assert target_columns == expected_targets


def test_unexpected_measure_count_aborts() -> None:
    measures = make_valid_measure_columns()[:-1]

    frame = pd.DataFrame(
        columns=measures
    )

    with pytest.raises(
        MRICatalogError,
        match="measurement catalog size",
    ):
        build_mri_catalog(frame)


def test_case_insensitive_duplicate_aborts() -> None:
    measures = make_valid_measure_columns()

    measures[-1] = measures[-2].lower()

    frame = pd.DataFrame(
        columns=measures
    )

    with pytest.raises(
        MRICatalogError,
        match="duplicated case-insensitively",
    ):
        build_mri_catalog(frame)


def test_missing_fixed_exclusion_aborts() -> None:
    measures = make_valid_measure_columns()

    measures[0] = "ST99999SV"

    frame = pd.DataFrame(
        columns=measures
    )

    with pytest.raises(
        MRICatalogError,
        match="ST8SV",
    ):
        build_mri_catalog(frame)