from __future__ import annotations

import gzip
import hashlib

from pathlib import Path

import pytest

from adni_benchmark.source_inspection import (
    SourceInspectionError,
    inspect_source,
    read_csv_header,
)


def test_reads_plain_csv_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv"
    path.write_text(
        "RID,VISCODE2,SCORE\n1,bl,10\n",
        encoding="utf-8",
    )

    columns, encoding = read_csv_header(path)

    assert columns == (
        "RID",
        "VISCODE2",
        "SCORE",
    )
    assert encoding == "utf-8-sig"


def test_reads_gzip_csv_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv.gz"

    with gzip.open(
        path,
        mode="wt",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "RID,VISCODE2,SCORE\n"
        )

    columns, _ = read_csv_header(path)

    assert columns == (
        "RID",
        "VISCODE2",
        "SCORE",
    )


def test_normalizes_header_whitespace_and_bom(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv"

    path.write_text(
        "\ufeffRID, VISCODE2 , SCORE \n",
        encoding="utf-8",
    )

    columns, _ = read_csv_header(path)

    assert columns == (
        "RID",
        "VISCODE2",
        "SCORE",
    )


def test_empty_file_aborts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv"
    path.touch()

    with pytest.raises(
        SourceInspectionError,
        match="empty",
    ):
        read_csv_header(path)


def test_empty_column_name_aborts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv"
    path.write_text(
        "RID,,SCORE\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SourceInspectionError,
        match="empty column",
    ):
        read_csv_header(path)


def test_duplicate_column_name_aborts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "TEST.csv"
    path.write_text(
        "RID,SCORE,SCORE\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SourceInspectionError,
        match="duplicated",
    ):
        read_csv_header(path)


def test_source_snapshot_contains_fingerprint(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()

    path = raw / "ADAS_TEST.csv"
    content = b"RID,VISCODE2,SCORE\n1,bl,10\n"
    path.write_bytes(content)

    snapshot = inspect_source(
        source_name="ADAS",
        source_path=path,
        raw_data=raw,
    )

    assert snapshot.source_name == "ADAS"
    assert snapshot.relative_path == Path(
        "ADAS_TEST.csv"
    )
    assert snapshot.size_bytes == len(content)
    assert snapshot.sha256 == hashlib.sha256(
        content
    ).hexdigest()
    assert snapshot.n_columns == 3