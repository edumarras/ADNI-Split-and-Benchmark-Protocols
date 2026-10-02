import numpy as np
import pandas as pd
import pytest

from adni_benchmark.participant_split_search import (
    ParticipantSplitSearchError,
    SPLIT_MASTER_SEED,
    SPLIT_TARGET_PROPORTIONS,
    aggregate_split_distances,
    build_split_search_context,
    compute_split_sizes,
    evaluate_split_candidate,
    generate_candidate_indices,
    make_categorical_balance_criterion,
    make_numeric_balance_criterion,
)


def make_balance_profile(
    n_participants: int = 20,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for index in range(
        n_participants
    ):
        records.append(
            {
                "RID": f"XX{index:04d}",
                "n_visits": 1 + (index % 4),
                "diagnosis_first": (
                    "DX_A"
                    if index < (
                        n_participants // 2
                    )
                    else "DX_B"
                ),
                "sex": (
                    "A"
                    if index % 2 == 0
                    else "B"
                ),
                "apoe_availability": (
                    "available"
                    if index % 3
                    else "missing"
                ),
                "modality_pattern_first": (
                    "pattern_a"
                    if index % 2 == 0
                    else "pattern_b"
                ),
                "longitudinal_class": (
                    "single_eligible_visit"
                    if index % 4 == 0
                    else (
                        "two_eligible_visits"
                        if index % 4 == 1
                        else "three_or_more_eligible_visits"
                    )
                ),
                "followup_months": float(index),
                "modality_count_mean": (
                    4.0 + (index % 4) / 2.0
                ),
            }
        )

    return pd.DataFrame(records)


def test_compute_split_sizes_for_2839_participants() -> None:
    sizes = compute_split_sizes(
        2839
    )

    assert sizes.n_train == 1987
    assert sizes.n_validation == 426
    assert sizes.n_test == 426


def test_compute_split_sizes_requires_three_participants() -> None:
    with pytest.raises(
        ParticipantSplitSearchError,
        match="At least three participants",
    ):
        compute_split_sizes(
            2
        )


def test_target_proportions_sum_to_one() -> None:
    assert np.isclose(
        sum(
            SPLIT_TARGET_PROPORTIONS.values()
        ),
        1.0,
    )


def test_categorical_criterion_probabilities_sum_to_one() -> None:
    profile = make_balance_profile()

    criterion = (
        make_categorical_balance_criterion(
            profile=profile,
            column="sex",
        )
    )

    assert np.isclose(
        criterion.global_probabilities.sum(),
        1.0,
    )

    assert len(
        criterion.codes
    ) == len(
        profile
    )


def test_numeric_criterion_uses_population_standard_deviation() -> None:
    profile = make_balance_profile()

    criterion = (
        make_numeric_balance_criterion(
            profile=profile,
            column="followup_months",
        )
    )

    assert criterion is not None

    expected_std = float(
        np.std(
            profile[
                "followup_months"
            ].to_numpy(
                dtype=float
            ),
            ddof=0,
        )
    )

    assert np.isclose(
        criterion.original_std,
        expected_std,
    )

    assert np.isclose(
        np.mean(
            criterion.global_finite_values
        ),
        0.0,
        atol=1e-12,
    )


def test_constant_numeric_criterion_is_inactive() -> None:
    profile = make_balance_profile()

    profile[
        "followup_months"
    ] = 1.0

    assert (
        make_numeric_balance_criterion(
            profile=profile,
            column="followup_months",
        )
        is None
    )


def test_context_contains_expected_split_sizes() -> None:
    profile = make_balance_profile(
        20
    )

    context = build_split_search_context(
        profile
    )

    assert (
        context.split_sizes.n_train
        + context.split_sizes.n_validation
        + context.split_sizes.n_test
        == 20
    )

    assert (
        context.total_visits
        == int(
            profile[
                "n_visits"
            ].sum()
        )
    )


def test_missing_required_profile_column_aborts() -> None:
    profile = (
        make_balance_profile()
        .drop(
            columns=[
                "sex",
            ]
        )
    )

    with pytest.raises(
        ParticipantSplitSearchError,
        match="missing required columns",
    ):
        build_split_search_context(
            profile
        )


def test_nonpositive_visit_count_aborts() -> None:
    profile = make_balance_profile()

    profile.loc[
        0,
        "n_visits",
    ] = 0

    with pytest.raises(
        ParticipantSplitSearchError,
        match="participant with no visits",
    ):
        build_split_search_context(
            profile
        )


def test_candidate_generation_is_deterministic() -> None:
    sizes = compute_split_sizes(
        20
    )

    first = generate_candidate_indices(
        n_participants=20,
        split_sizes=sizes,
        seed=12345,
    )

    second = generate_candidate_indices(
        n_participants=20,
        split_sizes=sizes,
        seed=12345,
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        assert np.array_equal(
            first[
                split_name
            ],
            second[
                split_name
            ],
        )


def test_candidate_indices_cover_each_participant_once() -> None:
    sizes = compute_split_sizes(
        20
    )

    indices = generate_candidate_indices(
        n_participants=20,
        split_sizes=sizes,
        seed=12345,
    )

    combined = np.concatenate(
        [
            indices["train"],
            indices["validation"],
            indices["test"],
        ]
    )

    assert sorted(
        combined.tolist()
    ) == list(
        range(20)
    )


def test_aggregate_split_distances_uses_half_mean_half_max() -> None:
    mean_value, maximum, score = (
        aggregate_split_distances(
            {
                "train": 0.1,
                "validation": 0.2,
                "test": 0.3,
            }
        )
    )

    assert np.isclose(
        mean_value,
        0.2,
    )

    assert np.isclose(
        maximum,
        0.3,
    )

    assert np.isclose(
        score,
        0.25,
    )


def test_evaluate_candidate_uses_seed_contract() -> None:
    profile = make_balance_profile(
        20
    )

    context = build_split_search_context(
        profile
    )

    candidate_id = 7

    seed = (
        SPLIT_MASTER_SEED
        + candidate_id
        - 1
    )

    result = evaluate_split_candidate(
        context=context,
        candidate_id=candidate_id,
        seed=seed,
    )

    assert (
        result[
            "candidate_id"
        ]
        == candidate_id
    )

    assert (
        result[
            "seed"
        ]
        == seed
    )

    assert np.isfinite(
        float(
            result[
                "score_final"
            ]
        )
    )


def test_evaluate_candidate_records_all_nine_scores() -> None:
    profile = make_balance_profile(
        40
    )

    context = build_split_search_context(
        profile
    )

    result = evaluate_split_candidate(
        context=context,
        candidate_id=1,
        seed=SPLIT_MASTER_SEED,
    )

    expected = {
        "score_visit_proportion",
        "score_diagnosis_first",
        "score_n_visits",
        "score_longitudinal_class",
        "score_modality_pattern_first",
        "score_followup_months",
        "score_modality_count_mean",
        "score_sex",
        "score_apoe_availability",
    }

    assert expected.issubset(
        result
    )


def test_candidate_participant_counts_match_fixed_sizes() -> None:
    profile = make_balance_profile(
        40
    )

    context = build_split_search_context(
        profile
    )

    result = evaluate_split_candidate(
        context=context,
        candidate_id=1,
        seed=SPLIT_MASTER_SEED,
    )

    assert (
        result[
            "n_participants_train"
        ]
        == context.split_sizes.n_train
    )

    assert (
        result[
            "n_participants_validation"
        ]
        == context.split_sizes.n_validation
    )

    assert (
        result[
            "n_participants_test"
        ]
        == context.split_sizes.n_test
    )
