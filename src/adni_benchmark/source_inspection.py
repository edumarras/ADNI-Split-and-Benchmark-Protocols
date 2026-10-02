from __future__ import annotations

import csv
import gzip
import hashlib

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class SourceInspectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceSnapshot:
    source_name: str
    absolute_path: Path
    relative_path: Path
    sha256: str
    size_bytes: int
    encoding: str
    columns: tuple[str, ...]

    @property
    def n_columns(self) -> int:
        return len(self.columns)


def open_csv_text(
    path: Path,
    encoding: str,
) -> TextIO:
    # Open compressed and uncompressed CSV files through the same interface
    if path.name.casefold().endswith(".csv.gz"):
        return gzip.open(
            path,
            mode="rt",
            encoding=encoding,
            newline="",
        )

    return path.open(
        mode="r",
        encoding=encoding,
        newline="",
    )


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    # Hash the raw file bytes without loading the whole file into memory
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_csv_header(
    path: Path,
) -> tuple[tuple[str, ...], str]:
    # Preserve the encoding fallback policy used by the frozen benchmark
    encodings = (
        "utf-8-sig",
        "utf-8",
        "latin1",
    )

    encoding_errors: list[str] = []

    for encoding in encodings:
        try:
            with open_csv_text(path, encoding) as handle:
                reader = csv.reader(handle)
                raw_header = next(reader, None)

        except UnicodeDecodeError as error:
            encoding_errors.append(
                f"{encoding}: {error}"
            )
            continue

        if raw_header is None:
            raise SourceInspectionError(
                f"Source file is empty: {path}"
            )

        # Normalize only the column names, not any data values
        columns = tuple(
            column.strip().lstrip("\ufeff")
            for column in raw_header
        )

        if not columns:
            raise SourceInspectionError(
                f"Source file has no header: {path}"
            )

        empty_positions = [
            index
            for index, column in enumerate(columns)
            if not column
        ]

        if empty_positions:
            raise SourceInspectionError(
                f"Header contains empty column names at positions "
                f"{empty_positions}: {path}"
            )

        counts = Counter(columns)

        duplicate_columns = sorted(
            column
            for column, count in counts.items()
            if count > 1
        )

        if duplicate_columns:
            raise SourceInspectionError(
                f"Header contains duplicated column names "
                f"{duplicate_columns}: {path}"
            )

        return columns, encoding

    raise SourceInspectionError(
        f"Could not decode source header: {path}. "
        f"Attempts: {encoding_errors}"
    )


def inspect_source(
    source_name: str,
    source_path: Path,
    raw_data: Path,
) -> SourceSnapshot:
    # Read only metadata and the header at this stage
    columns, encoding = read_csv_header(source_path)

    return SourceSnapshot(
        source_name=source_name,
        absolute_path=source_path.resolve(),
        relative_path=source_path.resolve().relative_to(
            raw_data.resolve()
        ),
        sha256=calculate_sha256(source_path),
        size_bytes=source_path.stat().st_size,
        encoding=encoding,
        columns=columns,
    )


def inspect_all_sources(
    discovered_sources: dict[str, Path],
    raw_data: Path,
) -> dict[str, SourceSnapshot]:
    snapshots: dict[str, SourceSnapshot] = {}

    for source_name, source_path in discovered_sources.items():
        snapshots[source_name] = inspect_source(
            source_name=source_name,
            source_path=source_path,
            raw_data=raw_data,
        )

    return snapshots