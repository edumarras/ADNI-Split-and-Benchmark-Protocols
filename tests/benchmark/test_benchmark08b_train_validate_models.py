import numpy as np
from scipy import sparse

from adni_benchmark_protocol.benchmark08b_train_validate_models import (
    MODELS,
    SCENARIOS,
    TARGET_COUNT,
    TARGET_REPRESENTATIONS,
    aggregate_metrics,
    centered_alpha_base,
    expected_configuration_count,
    model_grid_payload,
    reconstruct_prediction,
)


DECISION_IDS = {
    "preprocessing_decision_id": "prep-test",
    "target_decision_id": "target-test",
    "scenario_decision_id": "scenario-test",
}


def test_model_grid_contract() -> None:
    grid = model_grid_payload(
        DECISION_IDS
    )

    assert grid[
        "upstream_decision_ids"
    ][
        "preprocessing_decision_id"
    ] == "prep-test"

    assert grid[
        "upstream_decision_ids"
    ][
        "target_decision_id"
    ] == "target-test"

    assert grid[
        "upstream_decision_ids"
    ][
        "scenario_decision_id"
    ] == "scenario-test"

    assert set(
        grid[
            "models"
        ]
    ) == set(
        MODELS
    )

    assert set(
        grid[
            "scenarios"
        ]
    ) == set(
        SCENARIOS
    )

    assert set(
        grid[
            "target_representations"
        ]
    ) == set(
        TARGET_REPRESENTATIONS
    )


def test_expected_configuration_count() -> None:
    grid = model_grid_payload(
        DECISION_IDS
    )

    assert expected_configuration_count(
        grid
    ) == 318


def test_participant_macro_metrics_zero_error() -> None:
    truth = np.zeros(
        (
            4,
            TARGET_COUNT,
        ),
        dtype=float,
    )

    prediction = truth.copy()

    participants = np.asarray(
        [
            "A",
            "A",
            "B",
            "C",
        ]
    )

    metrics = aggregate_metrics(
        truth,
        prediction,
        participants,
    )

    assert metrics[
        "participant_macro_rmse"
    ] == 0.0

    assert metrics[
        "participant_macro_mae"
    ] == 0.0

    assert metrics[
        "visit_weighted_rmse"
    ] == 0.0


def test_centered_alpha_base_positive() -> None:
    rng = np.random.default_rng(
        7
    )

    x = rng.normal(
        size=(
            30,
            8,
        )
    )

    y = rng.normal(
        size=(
            30,
            5,
        )
    )

    assert centered_alpha_base(
        x,
        y,
    ) > 0.0


def test_full_reconstruction_identity() -> None:
    rng = np.random.default_rng(
        9
    )

    prediction = rng.normal(
        size=(
            5,
            TARGET_COUNT,
        )
    )

    observed = reconstruct_prediction(
        prediction,
        "full",
        pca=None,
    )

    assert np.array_equal(
        observed,
        prediction,
    )


def test_sparse_input_shape_smoke() -> None:
    matrix = sparse.csr_matrix(
        np.eye(
            5,
            dtype=float,
        )
    )

    assert matrix.shape == (
        5,
        5,
    )
