import pandas as pd

from adni_benchmark_protocol.auditA05_temporal_modality_exclusions import (
    AuditAbort,
    normalize_rid,
    normalize_viscode,
    parse_dates,
    suppress_count,
)


def test_normalize_rid() -> None:
    series = pd.Series(
        ["001", "2", "0003"],
        dtype="string",
    )

    observed = normalize_rid(
        series,
        "synthetic",
    )

    assert observed.tolist() == [
        "1",
        "2",
        "3",
    ]


def test_normalize_rid_rejects_nonnumeric() -> None:
    series = pd.Series(
        ["XX11"],
        dtype="string",
    )

    try:
        normalize_rid(
            series,
            "synthetic",
        )
    except AuditAbort:
        return

    raise AssertionError(
        "Expected AuditAbort for nonnumeric RID."
    )


def test_normalize_viscode() -> None:
    series = pd.Series(
        [" BL ", "M06"],
        dtype="string",
    )

    observed = normalize_viscode(
        series,
    )

    assert observed.tolist() == [
        "bl",
        "m06",
    ]


def test_parse_dates_normalizes_day() -> None:
    series = pd.Series(
        [
            "2026-01-02 14:15:00",
            "2026-01-03",
        ],
        dtype="string",
    )

    observed = parse_dates(
        series,
    )

    assert str(
        observed.iloc[0].date()
    ) == "2026-01-02"

    assert str(
        observed.iloc[1].date()
    ) == "2026-01-03"


def test_safe_count_suppression() -> None:
    assert suppress_count(
        0
    ) == 0

    assert suppress_count(
        1
    ) is None

    assert suppress_count(
        4
    ) is None

    assert suppress_count(
        5
    ) == 5
