import numpy as np
import pandas as pd

from adni_benchmark_protocol.benchmark07b_fit_target_representations import (
    Checks,
    PCA_VARIANCE_THRESHOLD,
    TARGET_COUNT,
    TargetStageAbort,
    fit_group,
    normalize_manifest,
    normalize_rid_scalar,
    normalize_split_scalar,
    normalize_viscode_scalar,
    reconstruction_metrics,
)


def test_normalize_scalar_keys() -> None:
    assert normalize_rid_scalar(
        "001",
        "synthetic",
    ) == "1"

    assert normalize_viscode_scalar(
        " M06 ",
        "synthetic",
    ) == "m06"

    assert normalize_split_scalar(
        " TRAIN ",
        "synthetic",
    ) == "train"


def test_invalid_split_rejected() -> None:
    try:
        normalize_split_scalar(
            "holdout",
            "synthetic",
        )
    except TargetStageAbort:
        return

    raise AssertionError(
        "Expected TargetStageAbort."
    )


def test_normalize_manifest() -> None:
    frame = pd.DataFrame(
        {
            "RID": [
                "001",
                "002",
            ],
            "split": [
                "train",
                "train",
            ],
            "current_VISCODE2": [
                "BL",
                "M06",
            ],
        }
    )

    observed = normalize_manifest(
        frame,
        "synthetic",
    )

    assert observed["RID"].tolist() == [
        "1",
        "2",
    ]

    assert observed[
        "current_VISCODE2"
    ].tolist() == [
        "bl",
        "m06",
    ]


def test_fit_group_train_only_full_and_pca90() -> None:
    rng = np.random.default_rng(
        7
    )

    n_train = 140
    n_validation = 40
    latent_dimensions = 35

    latent_train = rng.normal(
        size=(
            n_train,
            latent_dimensions,
        )
    )

    latent_validation = rng.normal(
        size=(
            n_validation,
            latent_dimensions,
        )
    )

    loadings = rng.normal(
        size=(
            latent_dimensions,
            TARGET_COUNT,
        )
    )

    train = (
        latent_train
        @ loadings
        + 0.2
        * rng.normal(
            size=(
                n_train,
                TARGET_COUNT,
            )
        )
    )

    validation = (
        latent_validation
        @ loadings
        + 0.2
        * rng.normal(
            size=(
                n_validation,
                TARGET_COUNT,
            )
        )
    )

    train_manifest = pd.DataFrame(
        {
            "RID": [
                str(i)
                for i
                in range(
                    n_train
                )
            ]
        }
    )

    validation_manifest = pd.DataFrame(
        {
            "RID": [
                str(i)
                for i
                in range(
                    n_validation
                )
            ]
        }
    )

    checks = Checks()

    result = fit_group(
        "synthetic",
        train,
        validation,
        train_manifest,
        validation_manifest,
        checks,
    )

    assert checks.passed

    assert result[
        "y_full_train"
    ].shape == (
        n_train,
        TARGET_COUNT,
    )

    assert result[
        "y_full_validation"
    ].shape == (
        n_validation,
        TARGET_COUNT,
    )

    assert (
        1
        <= result[
            "component_count"
        ]
        <= TARGET_COUNT
    )

    assert result[
        "cumulative_explained_variance"
    ] >= PCA_VARIANCE_THRESHOLD

    assert result[
        "previous_cumulative_explained_variance"
    ] < PCA_VARIANCE_THRESHOLD


def test_reconstruction_metrics_zero_error() -> None:
    values = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=float,
    )

    manifest = pd.DataFrame(
        {
            "RID": [
                "1",
                "2",
            ]
        }
    )

    observed = reconstruction_metrics(
        values,
        values.copy(),
        manifest,
    )

    assert observed[
        "cell_rmse"
    ] == 0.0

    assert observed[
        "participant_macro_rmse"
    ] == 0.0
