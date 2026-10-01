from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


class RuntimeConfigurationError(RuntimeError):
    """Raised when the execution environment cannot be resolved safely."""


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    raw_data: Path
    processed_data: Path
    reports: Path


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def resolve_runtime_paths(
    *,
    project_root: Path | None = None,
    raw_data: Path | None = None,
    processed_data: Path | None = None,
    reports: Path | None = None,
) -> RuntimePaths:
    """
    Resolve the minimal filesystem layout required by the pipeline.

    No ADNI files are opened here and no scientific decision is made.
    """

    root = _resolve_path(
        project_root if project_root is not None else Path.cwd()
    )

    if not root.is_dir():
        raise RuntimeConfigurationError(
            f"Project root does not exist or is not a directory: {root}"
        )

    resolved_raw = _resolve_path(
        raw_data if raw_data is not None else root / "data" / "raw"
    )

    resolved_processed = _resolve_path(
        processed_data
        if processed_data is not None
        else root / "data" / "processed"
    )

    resolved_reports = _resolve_path(
        reports if reports is not None else root / "reports"
    )

    if not resolved_raw.is_dir():
        raise RuntimeConfigurationError(
            f"Raw-data directory does not exist or is not a directory: "
            f"{resolved_raw}"
        )

    try:
        resolved_processed.mkdir(parents=True, exist_ok=True)
        resolved_reports.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeConfigurationError(
            "Could not create the required output directories."
        ) from error

    return RuntimePaths(
        project_root=root,
        raw_data=resolved_raw,
        processed_data=resolved_processed,
        reports=resolved_reports,
    )


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """
    Configure pipeline logging without logging participant-level data.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )