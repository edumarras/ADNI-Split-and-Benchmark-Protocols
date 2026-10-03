import numpy as np
import pandas as pd

from adni_benchmark.split06_same_wave_prior_visits import (
    CandidateVisit,
    classify_date_coverage,
    make_date_profile,
    pairwise_median_date_gaps,
    parse_protocol_month,
    select_by_explicit_zero_wave_order,
    summarize_ambiguous_cases,
    unique_latest_by_exact_consensus,
    unique_latest_by_median_date,
    unique_latest_by_strict_intervals,
)


def make_candidate(
    viscode2: str,
    dates: list[str],
) -> CandidateVisit:
    profile = make_date_profile(
        [
            pd.Timestamp(
                value
            )
            for value
            in dates
        ]
    )

    return CandidateVisit(
        rid="XX0001",
        viscode2=viscode2,
        split="train",
        protocol_month=0.0,
        modality_names=frozenset(
            {
                "ADAS",
            }
        ),
        observed_feature_tokens=frozenset(
            {
                "ADAS__A",
            }
        ),
        date_profile=profile,
    )


def test_parse_protocol_month_zero_wave_aliases() -> None:
    for value in (
        "bl",
        "sc",
        "scmri",
        "m00",
        "v01",
    ):
        assert (
            parse_protocol_month(
                value
            )
            == 0.0
        )


def test_parse_protocol_month_month_and_year() -> None:
    assert (
        parse_protocol_month(
            "m36"
        )
        == 36.0
    )

    assert (
        parse_protocol_month(
            "y3"
        )
        == 36.0
    )


def test_parse_protocol_month_unknown_is_nan() -> None:
    assert np.isnan(
        parse_protocol_month(
            "screening2"
        )
    )


def test_make_date_profile_empty() -> None:
    profile = make_date_profile(
        []
    )

    assert (
        profile.status
        == "no_clinical_date"
    )

    assert (
        profile.observed_date_cells
        == 0
    )

    assert pd.isna(
        profile.date_min
    )


def test_make_date_profile_exact_consensus() -> None:
    profile = make_date_profile(
        [
            pd.Timestamp(
                "2020-01-10"
            ),
            pd.Timestamp(
                "2020-01-10"
            ),
        ]
    )

    assert (
        profile.status
        == "exact_date_consensus"
    )

    assert (
        profile.distinct_dates
        == 1
    )

    assert (
        profile.date_span_days
        == 0.0
    )


def test_make_date_profile_classifies_short_range() -> None:
    profile = make_date_profile(
        [
            pd.Timestamp(
                "2020-01-10"
            ),
            pd.Timestamp(
                "2020-01-15"
            ),
        ]
    )

    assert (
        profile.status
        == "date_range_within_7_days"
    )

    assert (
        profile.date_span_days
        == 5.0
    )


def test_strict_interval_selects_unique_latest_candidate() -> None:
    earlier = make_candidate(
        "sc",
        [
            "2020-01-01",
            "2020-01-03",
        ],
    )

    later = make_candidate(
        "bl",
        [
            "2020-01-10",
            "2020-01-11",
        ],
    )

    winner = (
        unique_latest_by_strict_intervals(
            [
                earlier,
                later,
            ]
        )
    )

    assert winner is not None

    assert (
        winner.viscode2
        == "bl"
    )


def test_strict_interval_rejects_overlapping_ranges() -> None:
    first = make_candidate(
        "sc",
        [
            "2020-01-01",
            "2020-01-10",
        ],
    )

    second = make_candidate(
        "bl",
        [
            "2020-01-05",
            "2020-01-15",
        ],
    )

    assert (
        unique_latest_by_strict_intervals(
            [
                first,
                second,
            ]
        )
        is None
    )


def test_strict_interval_rejects_missing_date_candidate() -> None:
    dated = make_candidate(
        "sc",
        [
            "2020-01-01",
        ],
    )

    missing = make_candidate(
        "bl",
        [],
    )

    assert (
        unique_latest_by_strict_intervals(
            [
                dated,
                missing,
            ]
        )
        is None
    )


def test_exact_consensus_requires_exact_dates() -> None:
    first = make_candidate(
        "sc",
        [
            "2020-01-01",
        ],
    )

    second = make_candidate(
        "bl",
        [
            "2020-01-02",
        ],
    )

    winner = (
        unique_latest_by_exact_consensus(
            [
                first,
                second,
            ]
        )
    )

    assert winner is not None

    assert (
        winner.viscode2
        == "bl"
    )


def test_median_policy_can_select_when_intervals_overlap() -> None:
    first = make_candidate(
        "sc",
        [
            "2020-01-01",
            "2020-01-10",
        ],
    )

    second = make_candidate(
        "bl",
        [
            "2020-01-05",
            "2020-01-20",
        ],
    )

    winner = (
        unique_latest_by_median_date(
            [
                first,
                second,
            ]
        )
    )

    assert winner is not None

    assert (
        winner.viscode2
        == "bl"
    )


def test_zero_wave_explicit_order_selects_highest_rank() -> None:
    candidates = [
        make_candidate(
            "sc",
            [
                "2020-01-01",
            ],
        ),
        make_candidate(
            "scmri",
            [
                "2020-01-02",
            ],
        ),
        make_candidate(
            "bl",
            [
                "2020-01-03",
            ],
        ),
    ]

    winner = (
        select_by_explicit_zero_wave_order(
            candidates
        )
    )

    assert winner is not None

    assert (
        winner.viscode2
        == "bl"
    )


def test_zero_wave_explicit_order_refuses_v01() -> None:
    candidates = [
        make_candidate(
            "sc",
            [
                "2020-01-01",
            ],
        ),
        make_candidate(
            "v01",
            [
                "2020-01-02",
            ],
        ),
    ]

    assert (
        select_by_explicit_zero_wave_order(
            candidates
        )
        is None
    )


def test_date_coverage_classification() -> None:
    dated = make_candidate(
        "sc",
        [
            "2020-01-01",
        ],
    )

    missing = make_candidate(
        "bl",
        [],
    )

    assert (
        classify_date_coverage(
            [
                dated,
            ]
        )
        == "all_dated"
    )

    assert (
        classify_date_coverage(
            [
                dated,
                missing,
            ]
        )
        == "partially_dated"
    )

    assert (
        classify_date_coverage(
            [
                missing,
            ]
        )
        == "none_dated"
    )


def test_pairwise_median_date_gaps() -> None:
    candidates = [
        make_candidate(
            "sc",
            [
                "2020-01-01",
            ],
        ),
        make_candidate(
            "bl",
            [
                "2020-01-11",
            ],
        ),
    ]

    assert (
        pairwise_median_date_gaps(
            candidates
        )
        == (
            10.0,
        )
    )


def test_ambiguous_summary_counts_strict_policy() -> None:
    first = make_candidate(
        "sc",
        [
            "2020-01-01",
        ],
    )

    second = make_candidate(
        "bl",
        [
            "2020-01-10",
        ],
    )

    audit = {
        "split": "train",
        "candidate_viscodes": (
            "sc",
            "bl",
        ),
        "candidate_count": 2,
        "current_protocol_month": 12.0,
        "prior_protocol_month": 0.0,
        "date_coverage_status": "all_dated",
        "all_candidates_exact_date_consensus": True,
        "strict_interval_latest_recoverable": True,
        "strict_interval_latest_before_current": True,
        "exact_consensus_latest_recoverable": True,
        "median_date_latest_recoverable": True,
        "explicit_zero_wave_order_recoverable": True,
        "explicit_zero_wave_order_selected_viscode": "bl",
        "current_date_available": True,
        "has_previous_target_complete_visit": False,
        "union_modality_count": 1,
        "best_single_modality_count": 1,
        "union_modality_gain": 0,
        "union_feature_count": 1,
        "best_single_feature_count": 1,
        "union_feature_gain": 0,
        "pairwise_median_gap_days": (
            9.0,
        ),
    }

    from adni_benchmark.split06_same_wave_prior_visits import (
        AmbiguousCaseAudit,
    )

    frame = summarize_ambiguous_cases(
        [
            AmbiguousCaseAudit(
                **audit
            )
        ]
    )

    train = (
        frame.loc[
            frame[
                "split"
            ].eq(
                "train"
            )
        ]
        .iloc[
            0
        ]
    )

    assert (
        train[
            "ambiguous_participants"
        ]
        == 1
    )

    assert (
        train[
            "strict_interval_latest_recoverable"
        ]
        == 1
    )

    assert (
        train[
            "strict_interval_latest_and_before_current"
        ]
        == 1
    )
