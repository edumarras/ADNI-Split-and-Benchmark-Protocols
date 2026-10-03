import numpy as np
import pandas as pd

from adni_benchmark_protocol.benchmark06b_fit_input_preprocessing import (
    PreprocessingAbort,
    canonical_value_series,
    fit_medians,
    make_one_hot_encoder,
    normalize_rid_key,
    normalize_split_key,
    normalize_viscode_key,
    temporal_exclusion_flag,
)


def test_normalize_rid_key() -> None:
    series = pd.Series(
        ["001", 2, "0003"],
        dtype="object",
    )

    observed = normalize_rid_key(
        series,
        "synthetic",
    )

    assert observed.tolist() == [
        "1",
        "2",
        "3",
    ]


def test_normalize_rid_key_rejects_nonnumeric() -> None:
    series = pd.Series(
        ["XX11"],
        dtype="string",
    )

    try:
        normalize_rid_key(
            series,
            "synthetic",
        )
    except PreprocessingAbort:
        return

    raise AssertionError(
        "Expected PreprocessingAbort."
    )


def test_normalize_visit_and_split_keys() -> None:
    visits = pd.Series(
        [" BL ", "M06"],
        dtype="string",
    )

    splits = pd.Series(
        [" TRAIN ", "Validation"],
        dtype="string",
    )

    assert normalize_viscode_key(
        visits
    ).tolist() == [
        "bl",
        "m06",
    ]

    assert normalize_split_key(
        splits
    ).tolist() == [
        "train",
        "validation",
    ]


def test_canonical_value_series_masks_empty_strings() -> None:
    series = pd.Series(
        ["1", " ", pd.NA],
        dtype="string",
    )

    observed = canonical_value_series(
        series
    )

    assert observed.iloc[0] == "1"
    assert pd.isna(
        observed.iloc[1]
    )
    assert pd.isna(
        observed.iloc[2]
    )


def test_fit_medians() -> None:
    values = np.asarray(
        [
            [1.0, np.nan],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )

    observed = fit_medians(
        values,
        "synthetic",
    )

    assert np.allclose(
        observed,
        np.asarray(
            [
                3.0,
                5.0,
            ]
        ),
    )


def test_one_hot_encoder_is_unknown_safe() -> None:
    encoder = make_one_hot_encoder()

    train = np.asarray(
        [
            ["A"],
            ["B"],
        ],
        dtype=object,
    )

    encoder.fit(
        train
    )

    transformed = encoder.transform(
        np.asarray(
            [
                ["C"],
            ],
            dtype=object,
        )
    )

    assert transformed.shape[0] == 1


def test_temporal_exclusion_flag() -> None:
    assert temporal_exclusion_flag(
        "FAQ"
    ) == "exclude_previous__FAQ"
