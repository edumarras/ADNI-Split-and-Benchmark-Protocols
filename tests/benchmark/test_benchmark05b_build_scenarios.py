import math

import pandas as pd

from adni_benchmark_protocol.benchmark05b_build_scenarios import (
    apply_scalar_transform,
    apply_temporal_exclusion_to_aligned,
    parse_multi_tokens,
    parse_protocol_month,
    temporal_exclusion_flag,
)


def test_parse_protocol_month() -> None:
    assert parse_protocol_month("bl") == 0.0
    assert parse_protocol_month("sc") == 0.0
    assert parse_protocol_month("m06") == 6.0
    assert parse_protocol_month("y2") == 24.0
    assert math.isnan(parse_protocol_month("unknown"))


def test_parse_multi_tokens() -> None:
    assert parse_multi_tokens("1|2;3") == ["1", "2", "3"]
    assert parse_multi_tokens("") == []
    assert parse_multi_tokens(pd.NA) == []


def test_temporal_exclusion_flag() -> None:
    assert temporal_exclusion_flag(
        "CDR"
    ) == "exclude_previous__CDR"


def test_temporal_exclusion_only_masks_previous_longitudinal_block() -> None:
    manifest = pd.DataFrame(
        {
            "exclude_previous__CDR": [
                True,
                False,
            ]
        }
    )

    aligned = pd.DataFrame(
        {
            "FEATURE_A": [
                "1",
                "2",
            ],
            "FEATURE_B": [
                "3",
                "4",
            ],
        }
    )

    observed = apply_temporal_exclusion_to_aligned(
        scenario_name="longitudinal_previous_last",
        manifest=manifest,
        aligned=aligned,
        source_name="CDR",
        block="previous_visit",
        features=[
            "FEATURE_A",
            "FEATURE_B",
        ],
    )

    assert observed.loc[
        0,
        [
            "FEATURE_A",
            "FEATURE_B",
        ],
    ].isna().all()

    assert observed.loc[
        1,
        "FEATURE_A",
    ] == "2"


def test_temporal_exclusion_does_not_mask_current_block() -> None:
    manifest = pd.DataFrame(
        {
            "exclude_previous__CDR": [
                True,
            ]
        }
    )

    aligned = pd.DataFrame(
        {
            "FEATURE_A": [
                "1",
            ]
        }
    )

    observed = apply_temporal_exclusion_to_aligned(
        scenario_name="longitudinal_previous_last",
        manifest=manifest,
        aligned=aligned,
        source_name="CDR",
        block="current_visit",
        features=[
            "FEATURE_A",
        ],
    )

    assert observed.loc[
        0,
        "FEATURE_A",
    ] == "1"


def test_identity_scalar_transform() -> None:
    series = pd.Series(
        ["1", "2", pd.NA],
        dtype="string",
    )

    observed, failures = apply_scalar_transform(
        series,
        "identity",
        {},
        "SYNTHETIC__FEATURE",
    )

    assert failures == 0

    assert observed.astype("string").tolist() == [
        "1",
        "2",
        pd.NA,
    ]
