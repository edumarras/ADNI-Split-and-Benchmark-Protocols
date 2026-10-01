from pathlib import Path

import pytest

from adni_benchmark.runtime import (
    RuntimeConfigurationError,
    configure_logging,
    resolve_runtime_paths,
)


def test_resolves_default_paths(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)

    paths = resolve_runtime_paths(
        project_root=tmp_path,
    )

    assert paths.project_root == tmp_path.resolve()
    assert paths.raw_data == raw.resolve()
    assert paths.processed_data == (
        tmp_path / "data" / "processed"
    ).resolve()
    assert paths.reports == (
        tmp_path / "reports"
    ).resolve()

    assert paths.processed_data.is_dir()
    assert paths.reports.is_dir()


def test_allows_explicit_raw_data_path(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()

    external_raw = tmp_path / "adni_snapshot"
    external_raw.mkdir()

    paths = resolve_runtime_paths(
        project_root=project,
        raw_data=external_raw,
    )

    assert paths.raw_data == external_raw.resolve()


def test_missing_raw_directory_aborts(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RuntimeConfigurationError,
        match="Raw-data directory",
    ):
        resolve_runtime_paths(
            project_root=tmp_path,
        )


def test_missing_project_root_aborts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist"

    with pytest.raises(
        RuntimeConfigurationError,
        match="Project root",
    ):
        resolve_runtime_paths(
            project_root=missing,
        )


def test_configure_logging_is_callable() -> None:
    configure_logging()