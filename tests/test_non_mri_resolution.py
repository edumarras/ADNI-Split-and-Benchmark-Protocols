import math

import pandas as pd
import pytest

from adni_benchmark.non_mri_resolution import (
    NonMRIResolutionError,
    comparison_columns_for_source,
    parse_protocol_month,
    resolve_all_non_mri_sources,
    resolve_generic_source,
    row_dominates,
    rows_are_equivalent,
    values_equal,
)
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
)


def test_values_equal_accepts_numeric_equivalence() -> None:
    assert values_equal(
        "1",
        "1.0",
    )


def test_values_equal_rejects_different_values() -> None:
    assert not values_equal(
        "1",
        "2",
    )


def test_values_equal_handles_missingness() -> None:
    assert values_equal(
        pd.NA,
        pd.NA,
    )

    assert not values_equal(
        pd.NA,
        "1",
    )


def test_rows_are_equivalent_uses_substantive_columns() -> None:
    left = pd.Series(
        {
            "A": "1",
            "B": "x",
        }
    )

    right = pd.Series(
        {
            "A": "1.0",
            "B": "x",
        }
    )

    assert rows_are_equivalent(
        left=left,
        right=right,
        comparison_columns=(
            "A",
            "B",
        ),
    )


def test_row_dominance_requires_extra_nonconflicting_information() -> None:
    candidate = pd.Series(
        {
            "A": "1",
            "B": "2",
        }
    )

    other = pd.Series(
        {
            "A": "1",
            "B": pd.NA,
        }
    )

    assert row_dominates(
        candidate=candidate,
        other=other,
        comparison_columns=(
            "A",
            "B",
        ),
    )


def test_row_dominance_rejects_conflict() -> None:
    candidate = pd.Series(
        {
            "A": "9",
            "B": "2",
        }
    )

    other = pd.Series(
        {
            "A": "1",
            "B": pd.NA,
        }
    )

    assert not row_dominates(
        candidate=candidate,
        other=other,
        comparison_columns=(
            "A",
            "B",
        ),
    )


def test_parse_protocol_month() -> None:
    assert parse_protocol_month(
        "bl"
    ) == 0.0

    assert parse_protocol_month(
        "SC"
    ) == 0.0

    assert parse_protocol_month(
        "m12"
    ) == 12.0

    assert parse_protocol_month(
        "y2"
    ) == 24.0

    assert math.isnan(
        parse_protocol_month(
            "unknown"
        )
    )


def test_equivalent_visit_duplicates_are_resolved() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "SCORE": [
                "10",
                "10.0",
            ],
            "USERDATE": [
                "XX11AABBCC",
                "XX22AABBCC",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        summary,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert len(
        resolved
    ) == 1

    assert unresolved.empty

    assert dict(
        summary.status_counts
    )[
        "exact_or_equivalent_duplicate"
    ] == 1


def test_dominant_visit_record_is_selected() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "A": [
                "1",
                "1",
            ],
            "B": [
                pd.NA,
                "2",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        summary,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert len(
        resolved
    ) == 1

    assert unresolved.empty

    assert (
        resolved.loc[
            0,
            "B",
        ]
        == "2"
    )

    assert dict(
        summary.status_counts
    )[
        "dominant_record"
    ] == 1


def test_conflicting_visit_records_remain_unresolved() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "A": [
                "1",
                "2",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        summary,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert resolved.empty

    assert len(
        unresolved
    ) == 1

    assert dict(
        summary.status_counts
    )[
        "unresolved_conflict"
    ] == 1


def test_no_cell_wise_coalescence_occurs() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "A": [
                "1",
                pd.NA,
            ],
            "B": [
                pd.NA,
                "2",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        _,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert resolved.empty

    assert len(
        unresolved
    ) == 1


def test_ptdemog_restricts_to_earliest_protocol_wave() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "PHASE": [
                "ADNI1",
                "ADNI1",
            ],
            "VISCODE2": [
                "m12",
                "sc",
            ],
            "PTGENDER": [
                "A",
                "B",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        summary,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "PTDEMOG"
        ],
    )

    assert unresolved.empty

    assert len(
        resolved
    ) == 1

    assert (
        resolved.loc[
            0,
            "VISCODE2",
        ]
        == "sc"
    )

    assert summary.duplicate_groups == 1


def test_ptdemog_falls_back_to_earliest_date() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "PHASE": [
                "ADNI1",
                "ADNI1",
            ],
            "VISCODE2": [
                "unknown_a",
                "unknown_b",
            ],
            "VISDATE": [
                "2020-05-01",
                "2020-01-01",
            ],
            "PTGENDER": [
                "A",
                "B",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        _,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "PTDEMOG"
        ],
    )

    assert unresolved.empty

    assert (
        resolved.loc[
            0,
            "VISDATE",
        ]
        == "2020-01-01"
    )


def test_apoeres_compares_only_genotype() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "1",
            ],
            "GENOTYPE": [
                "XX11AABBCC",
                "XX11AABBCC",
            ],
            "OTHER_COLUMN": [
                "A",
                "B",
            ],
        },
        dtype="string",
    )

    columns = comparison_columns_for_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "APOERES"
        ],
    )

    assert columns == [
        "GENOTYPE",
    ]


def test_rows_with_missing_key_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "2",
            ],
            "VISCODE2": [
                "bl",
                pd.NA,
            ],
            "A": [
                "1",
                "2",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        unresolved,
        summary,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert len(
        resolved
    ) == 1

    assert unresolved.empty

    assert (
        summary.rows_with_missing_key
        == 1
    )


def test_resolved_keys_are_unique() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "1",
                "2",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "A": [
                "1",
                "2",
            ],
        },
        dtype="string",
    )

    (
        resolved,
        _,
        _,
    ) = resolve_generic_source(
        frame=frame,
        contract=SOURCE_CONTRACTS[
            "ADAS"
        ],
    )

    assert not resolved.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    ).any()


def test_resolve_all_requires_every_non_mri_source() -> None:
    with pytest.raises(
        NonMRIResolutionError,
        match="were not loaded",
    ):
        resolve_all_non_mri_sources(
            source_tables={}
        )