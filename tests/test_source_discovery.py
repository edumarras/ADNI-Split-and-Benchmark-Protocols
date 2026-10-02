from pathlib import Path

import pytest

from adni_benchmark.source_discovery import (
    REQUIRED_SOURCES,
    SourceDiscoveryError,
    discover_required_sources,
)


def create_source_files(
    raw_data: Path,
    compressed: bool = False,
) -> None:
    raw_data.mkdir(parents=True, exist_ok=True)

    extension = ".csv.gz" if compressed else ".csv"

    for source in REQUIRED_SOURCES:
        (raw_data / f"{source}_TEST{extension}").touch()


def test_discovers_all_required_sources(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    create_source_files(raw_data)

    discovered = discover_required_sources(raw_data)

    assert tuple(discovered.keys()) == REQUIRED_SOURCES
    assert len(discovered) == len(REQUIRED_SOURCES)

    for source in REQUIRED_SOURCES:
        assert source in discovered
        assert discovered[source].is_file()


def test_accepts_compressed_csv_files(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    create_source_files(
        raw_data,
        compressed=True,
    )

    discovered = discover_required_sources(raw_data)

    assert len(discovered) == len(REQUIRED_SOURCES)

    for path in discovered.values():
        assert path.name.casefold().endswith(".csv.gz")


def test_search_is_recursive(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    nested = raw_data / "snapshot" / "tables"

    create_source_files(nested)

    discovered = discover_required_sources(raw_data)

    assert len(discovered) == len(REQUIRED_SOURCES)


def test_source_matching_is_case_insensitive(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    raw_data.mkdir()

    for source in REQUIRED_SOURCES:
        (raw_data / f"{source.lower()}_test.CSV").touch()

    discovered = discover_required_sources(raw_data)

    assert len(discovered) == len(REQUIRED_SOURCES)


def test_missing_required_source_aborts(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    create_source_files(raw_data)

    (raw_data / "ADAS_TEST.csv").unlink()

    with pytest.raises(
        SourceDiscoveryError,
        match="ADAS",
    ):
        discover_required_sources(raw_data)


def test_ambiguous_source_aborts(
    tmp_path: Path,
) -> None:
    raw_data = tmp_path / "raw"
    create_source_files(raw_data)

    (raw_data / "ADAS_ANOTHER_SNAPSHOT.csv").touch()

    with pytest.raises(
        SourceDiscoveryError,
        match="ambiguous",
    ):
        discover_required_sources(raw_data)