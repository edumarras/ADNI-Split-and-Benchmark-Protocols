"""
Differential equivalence between the clean-room package and the frozen
pipeline scripts.

The refactoring strategy keeps the frozen scripts as the executable
reference: a reimplemented function is accepted only when it produces
the same output as the original on the same input. These tests encode
that check so it runs in CI rather than being verified by reading.

Enable them by pointing ``ADNI_FROZEN_SCRIPT_DIR`` at the directory
holding the frozen scripts. They are skipped when it is unset.

Comparison notes:

- DataFrames are compared as canonical CSV with sorted columns, so
  column order inside the frame does not mask a value difference,
  while row order — which the resolution contract fixes — does count.
- Exceptions are compared on ``args[0]``, not ``str(e)``. The frozen
  scripts raise ``KeyError`` in places where the package raises its
  own module error, and ``str(KeyError("x"))`` adds quotes that
  ``args[0]`` does not. The package errors all derive from
  ``RuntimeError``; the frozen ``ValueError``/``KeyError`` raises do
  not, and that difference is intentional.

All fixtures are synthetic. No ADNI data is read.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Any, Callable, Final

import pandas as pd
import pytest

from adni_benchmark import mri_catalog, mri_resolution

from frozen_reference import (
    FROZEN_DIR_ENV_VAR,
    build_reference_module,
    frozen_script_dir,
)


FROZEN_SCRIPT_NAME: Final[str] = "auto_split_maker.py"

REFERENCE_NAMES: Final[tuple[str, ...]] = (
    "build_mri_catalog",
    "resolve_mri_source",
)

TARGET_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"ST{index}SV" for index in range(1, 9)
)

MEASURE_SUFFIXES: Final[tuple[str, ...]] = (
    "SV",
    "CV",
    "SA",
    "TA",
    "TS",
)

FUZZ_TRIALS: Final[int] = 200

FUZZ_SEED: Final[int] = 20260102


# --------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------


@pytest.fixture(scope="module")
def frozen():
    """The frozen reference subset, or a skip when unavailable."""

    directory = frozen_script_dir()

    if directory is None:
        pytest.skip(
            f"{FROZEN_DIR_ENV_VAR} is not set to an existing directory"
        )

    script_path = directory / FROZEN_SCRIPT_NAME

    if not script_path.is_file():
        pytest.skip(f"{FROZEN_SCRIPT_NAME} not found in {directory}")

    return build_reference_module(script_path, REFERENCE_NAMES)


# --------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------


def canonical_form(value: Any) -> str:
    """A comparable textual form for a resolution output element."""

    if isinstance(value, pd.DataFrame):
        ordered = value.reindex(sorted(value.columns), axis=1)

        return ordered.to_csv(index=False, lineterminator="\n")

    if dataclasses.is_dataclass(value):
        return repr(dataclasses.asdict(value))

    return repr(value)


def outcome(
    function: Callable[..., Any],
    *args: Any,
) -> tuple[str, Any]:
    """Run a callable, reporting either its result or its error message."""

    try:
        return ("returned", function(*args))
    except Exception as error:  # noqa: BLE001 - differential probe
        message = error.args[0] if error.args else ""

        return ("raised", message)


def assert_same_outcome(
    frozen_result: tuple[str, Any],
    package_result: tuple[str, Any],
    label: str,
) -> None:
    frozen_kind, frozen_value = frozen_result
    package_kind, package_value = package_result

    assert frozen_kind == package_kind, (
        f"{label}: frozen {frozen_kind}, package {package_kind} "
        f"({frozen_value!r} / {package_value!r})"
    )

    if frozen_kind == "raised":
        assert frozen_value == package_value, label

        return

    assert isinstance(frozen_value, tuple)
    assert len(frozen_value) == len(package_value), label

    for position, (left, right) in enumerate(
        zip(frozen_value, package_value)
    ):
        assert canonical_form(left) == canonical_form(right), (
            f"{label}: element {position} differs"
        )


# --------------------------------------------------------------
# Synthetic UCSFFSX7-shaped frames
# --------------------------------------------------------------


def make_candidate_frame(
    rng: random.Random,
    *,
    participants: int,
    max_duplicates: int,
    missing_rate: float,
) -> pd.DataFrame:
    """A UCSFFSX7-shaped frame with duplicate visit groups."""

    value_pool = ("1.0", "2.0", "3.5", "9.0", "0.0")
    records: list[dict[str, object]] = []

    for participant in range(1, participants + 1):
        for viscode in ("bl", "m06", "m12"):
            for _ in range(rng.randint(1, max_duplicates)):
                record: dict[str, object] = {
                    "RID": str(participant),
                    "VISCODE2": viscode,
                    "IMAGEUID": str(rng.randint(100, 140)),
                    "STATUS": rng.choice(
                        ["COMPLETE", "PARTIAL", pd.NA]
                    ),
                    "OVERALLQC": rng.choice(["Pass", "Fail", pd.NA]),
                    "FLDSTRENG": rng.choice(["1.5", "3.0", pd.NA]),
                }

                for column in TARGET_COLUMNS:
                    record[column] = (
                        pd.NA
                        if rng.random() < missing_rate
                        else rng.choice(value_pool)
                    )

                records.append(record)

    rng.shuffle(records)

    return pd.DataFrame(records, dtype="string")


def make_single_record() -> list[dict[str, object]]:
    return [
        {
            "RID": "1",
            "VISCODE2": "bl",
            "IMAGEUID": "100",
            "STATUS": "COMPLETE",
            "OVERALLQC": "Pass",
            "FLDSTRENG": "3.0",
            **{column: "1.0" for column in TARGET_COLUMNS},
        }
    ]


def make_measure_columns(total: int = 325) -> list[str]:
    """Column names matching the fixed UCSFFSX7 ST naming contract."""

    columns: list[str] = []
    index = 1

    while len(columns) < total:
        for suffix in MEASURE_SUFFIXES:
            if len(columns) >= total:
                break

            columns.append(f"ST{index}{suffix}")

        index += 1

    for required in ("ST8SV", "ST68SV"):
        if required not in columns:
            columns[-1] = required

    columns = list(dict.fromkeys(columns))

    while len(columns) < total:
        index += 1
        candidate = f"ST{index}SV"

        if candidate not in columns:
            columns.append(candidate)

    return columns


# --------------------------------------------------------------
# resolve_mri_source
# --------------------------------------------------------------


def test_resolution_matches_frozen_on_randomized_frames(
    frozen,
) -> None:
    rng = random.Random(FUZZ_SEED)

    for trial in range(FUZZ_TRIALS):
        frame = make_candidate_frame(
            rng,
            participants=rng.randint(2, 8),
            max_duplicates=rng.randint(1, 5),
            missing_rate=rng.choice([0.0, 0.15, 0.35, 0.6]),
        )

        assert_same_outcome(
            outcome(
                frozen.resolve_mri_source,
                frame.copy(),
                TARGET_COLUMNS,
            ),
            outcome(
                mri_resolution.resolve_mri_source,
                frame.copy(),
                TARGET_COLUMNS,
            ),
            f"trial {trial}",
        )


ADVERSARIAL_CASES: Final[tuple[tuple[str, str, object], ...]] = (
    ("missing_imageuid", "IMAGEUID", pd.NA),
    ("non_integer_imageuid", "IMAGEUID", "100.5"),
    ("non_numeric_imageuid", "IMAGEUID", "abc"),
    ("non_numeric_target", "ST1SV", "texto"),
    ("infinite_target", "ST1SV", "inf"),
    ("missing_rid", "RID", pd.NA),
    ("empty_viscode", "VISCODE2", ""),
)


@pytest.mark.parametrize(
    ("label", "column", "value"),
    ADVERSARIAL_CASES,
    ids=[case[0] for case in ADVERSARIAL_CASES],
)
def test_resolution_matches_frozen_on_fail_fast_inputs(
    frozen,
    label: str,
    column: str,
    value: object,
) -> None:
    records = make_single_record()
    records[0][column] = value

    frame = pd.DataFrame(records, dtype="string")

    assert_same_outcome(
        outcome(
            frozen.resolve_mri_source,
            frame.copy(),
            TARGET_COLUMNS,
        ),
        outcome(
            mri_resolution.resolve_mri_source,
            frame.copy(),
            TARGET_COLUMNS,
        ),
        label,
    )


@pytest.mark.parametrize(
    "dropped",
    ["IMAGEUID", "RID", "VISCODE2"],
)
def test_resolution_matches_frozen_on_missing_columns(
    frozen,
    dropped: str,
) -> None:
    frame = pd.DataFrame(
        make_single_record(),
        dtype="string",
    ).drop(columns=[dropped])

    assert_same_outcome(
        outcome(
            frozen.resolve_mri_source,
            frame.copy(),
            TARGET_COLUMNS,
        ),
        outcome(
            mri_resolution.resolve_mri_source,
            frame.copy(),
            TARGET_COLUMNS,
        ),
        f"dropped {dropped}",
    )


def test_resolution_matches_frozen_on_empty_target_catalog(
    frozen,
) -> None:
    frame = pd.DataFrame(make_single_record(), dtype="string")

    assert_same_outcome(
        outcome(frozen.resolve_mri_source, frame.copy(), ()),
        outcome(
            mri_resolution.resolve_mri_source,
            frame.copy(),
            (),
        ),
        "empty target catalog",
    )


# --------------------------------------------------------------
# build_mri_catalog
# --------------------------------------------------------------


def test_catalog_matches_frozen_including_fingerprints(
    frozen,
) -> None:
    columns = make_measure_columns()

    frame = pd.DataFrame(
        [
            {
                "RID": "1",
                "VISCODE2": "bl",
                **{column: "1.0" for column in columns},
            }
        ],
        dtype="string",
    )

    assert_same_outcome(
        outcome(frozen.build_mri_catalog, frame.copy()),
        outcome(mri_catalog.build_mri_catalog, frame.copy()),
        "catalog",
    )


@pytest.mark.parametrize(
    ("label", "total"),
    [
        ("one_measure_short", 324),
        ("one_measure_extra", 326),
    ],
)
def test_catalog_matches_frozen_on_unexpected_sizes(
    frozen,
    label: str,
    total: int,
) -> None:
    columns = make_measure_columns(total)

    frame = pd.DataFrame(
        [
            {
                "RID": "1",
                "VISCODE2": "bl",
                **{column: "1.0" for column in columns},
            }
        ],
        dtype="string",
    )

    assert_same_outcome(
        outcome(frozen.build_mri_catalog, frame.copy()),
        outcome(mri_catalog.build_mri_catalog, frame.copy()),
        label,
    )
