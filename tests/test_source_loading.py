from pathlib import Path

import pandas as pd
import pytest

from adni_benchmark.source_inspection import SourceSnapshot
from adni_benchmark.source_loading import (
    SourceLoadingError,
    load_source_table,
    normalize_integer_identifier,
    normalize_missing_values,
    normalize_source_keys,
    summarize_loaded_source,
)
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
)


def make_snapshot(
    source_name: str,
    path: Path,
    columns: tuple[str, ...],
) -> SourceSnapshot:
    return SourceSnapshot(
        source_name=source_name,
        absolute_path=path,
        relative_path=Path(path.name),
        sha256="0" * 64,
        size_bytes=path.stat().st_size,
        encoding="utf-8-sig",
        columns=columns,
    )


def test_missing_sentinels_are_normalized() -> None:
    frame = pd.DataFrame(
        {
            "VALUE": [
                " observed ",
                "",
                "-1",
                " -4 ",
            ]
        },
        dtype="string",
    )

    normalized = normalize_missing_values(
        frame
    )

    assert normalized.loc[0, "VALUE"] == "observed"

    assert pd.isna(
        normalized.loc[1, "VALUE"]
    )

    assert pd.isna(
        normalized.loc[2, "VALUE"]
    )

    assert pd.isna(
        normalized.loc[3, "VALUE"]
    )


def test_integer_identifier_is_canonicalized() -> None:
    series = pd.Series(
        [
            "001",
            "2.0",
            pd.NA,
        ],
        dtype="string",
    )

    result = normalize_integer_identifier(
        series=series,
        source_name="TEST",
        column_name="RID",
    )

    assert result.iloc[0] == "1"
    assert result.iloc[1] == "2"
    assert pd.isna(result.iloc[2])


def test_invalid_identifier_aborts() -> None:
    series = pd.Series(
        [
            "1",
            "XX11AABBCC",
        ],
        dtype="string",
    )

    with pytest.raises(
        SourceLoadingError,
        match="non-numeric",
    ):
        normalize_integer_identifier(
            series=series,
            source_name="TEST",
            column_name="RID",
        )


def test_non_integer_identifier_aborts() -> None:
    series = pd.Series(
        [
            "1",
            "2.5",
        ],
        dtype="string",
    )

    with pytest.raises(
        SourceLoadingError,
        match="non-integer",
    ):
        normalize_integer_identifier(
            series=series,
            source_name="TEST",
            column_name="RID",
        )


def test_source_keys_are_normalized() -> None:
    frame = pd.DataFrame(
        {
            "RID": ["001"],
            "VISCODE2": [" BL "],
            "PHASE": [" adni1 "],
            "IMAGEUID": ["0100"],
        },
        dtype="string",
    )

    normalized = normalize_source_keys(
        frame=frame,
        contract=SOURCE_CONTRACTS["UCSFFSX7"],
    )

    assert normalized.loc[0, "RID"] == "1"
    assert normalized.loc[0, "VISCODE2"] == "bl"
    assert normalized.loc[0, "PHASE"] == "ADNI1"
    assert normalized.loc[0, "IMAGEUID"] == "100"


def test_load_source_table_normalizes_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ADAS_TEST.csv"

    path.write_text(
        "RID,VISCODE2,VALUE\n"
        "001,BL,-1\n"
        "002,m06, observed \n",
        encoding="utf-8",
    )

    snapshot = make_snapshot(
        source_name="ADAS",
        path=path,
        columns=(
            "RID",
            "VISCODE2",
            "VALUE",
        ),
    )

    frame = load_source_table(
        snapshot=snapshot,
        contract=SOURCE_CONTRACTS["ADAS"],
    )

    assert frame.loc[0, "RID"] == "1"
    assert frame.loc[0, "VISCODE2"] == "bl"
    assert pd.isna(frame.loc[0, "VALUE"])

    assert frame.loc[1, "RID"] == "2"
    assert frame.loc[1, "VISCODE2"] == "m06"
    assert frame.loc[1, "VALUE"] == "observed"


def test_header_change_aborts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ADAS_TEST.csv"

    path.write_text(
        "RID,VISCODE2,VALUE\n"
        "1,bl,10\n",
        encoding="utf-8",
    )

    snapshot = make_snapshot(
        source_name="ADAS",
        path=path,
        columns=(
            "RID",
            "VISCODE2",
            "DIFFERENT_COLUMN",
        ),
    )

    with pytest.raises(
        SourceLoadingError,
        match="header changed",
    ):
        load_source_table(
            snapshot=snapshot,
            contract=SOURCE_CONTRACTS["ADAS"],
        )


def test_load_summary_counts_missing_keys_and_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
                "2",
                pd.NA,
            ],
            "VISCODE2": [
                "bl",
                "bl",
                "m06",
                "m12",
            ],
            "VALUE": [
                "10",
                "11",
                "20",
                "30",
            ],
        },
        dtype="string",
    )

    summary = summarize_loaded_source(
        frame=frame,
        contract=SOURCE_CONTRACTS["ADAS"],
    )

    assert summary.n_rows == 4
    assert summary.n_columns == 3

    assert summary.rows_with_missing_key == 1

    assert summary.duplicate_key_rows == 2

    assert dict(
        summary.missing_key_counts
    ) == {
        "RID": 1,
        "VISCODE2": 0,
    }