from __future__ import annotations

from pathlib import Path
from typing import Final


REQUIRED_SOURCES: Final[tuple[str, ...]] = (
    "DXSUM",
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "PTDEMOG",
    "APOERES",
    "UCSFFSX7",
)


class SourceDiscoveryError(RuntimeError):
    pass


def list_supported_raw_files(raw_data: Path) -> list[Path]:
    # search raw files recursively
    files = [
        path
        for path in raw_data.rglob("*")
        if path.is_file()
        and path.name.casefold().endswith((".csv", ".csv.gz"))
    ]

    return sorted(files)


def discover_required_sources(
    raw_data: Path,
    required_sources: tuple[str, ...] = REQUIRED_SOURCES,
) -> dict[str, Path]:
    # runtime stage should already have validated raw_data,
    # but this function remains safe when called independently
    if not raw_data.is_dir():
        raise SourceDiscoveryError(
            f"Raw-data directory does not exist: {raw_data}"
        )

    available_files = list_supported_raw_files(raw_data)

    if not available_files:
        raise SourceDiscoveryError(
            f"No CSV or CSV.GZ files were found under: {raw_data}"
        )

    discovered: dict[str, Path] = {}

    for source in required_sources:
        matches = [
            path
            for path in available_files
            if source.casefold() in path.name.casefold()
        ]

        if not matches:
            raise SourceDiscoveryError(
                f"Required source was not found: {source}"
            )

        if len(matches) > 1:
            candidates = "\n".join(
                f"  - {path.relative_to(raw_data)}"
                for path in matches
            )

            raise SourceDiscoveryError(
                f"Source {source} is ambiguous.\n"
                f"Candidates:\n{candidates}"
            )

        discovered[source] = matches[0]

    return discovered