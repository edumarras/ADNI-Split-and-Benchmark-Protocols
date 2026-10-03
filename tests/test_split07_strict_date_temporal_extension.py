from pathlib import Path

import numpy as np
import pandas as pd

from adni_benchmark.split07_strict_date_temporal_extension import (
    DateEvidence,
    HISTORICAL_TEMPORAL_POLICY_DECISION_ID,
    TEMPORAL_POLICY_NAME,
    build_pairing_reference,
    canonical_json_sha256,
    date_available,
    make_date_profile,
    parse_protocol_month,
    summarize_pairing,
    unique_latest_by_strict_intervals,
)


def evidence(
    date: str,
    source: str = "ADAS",
    column: str = "VISDATE",
) -> DateEvidence:
    return DateEvidence(
        date=pd.Timestamp(date),
        source_name=source,
        column_name=column,
    )


def test_historical_temporal_policy_id_is_frozen() -> None:
    assert (
        HISTORICAL_TEMPORAL_POLICY_DECISION_ID
        == "90fefe5ec34d49743612"
    )


def test_temporal_policy_name_is_frozen() -> None:
    assert (
        TEMPORAL_POLICY_NAME
        == "strict_date_interval_same_wave_previous_visit_v1"
    )


def test_parse_protocol_month_zero_wave_aliases() -> None:
    for value in (
        "sc",
        "scmri",
        "bl",
        "m00",
        "v01",
    ):
        assert parse_protocol_month(value) == 0.0


def test_parse_protocol_month_month_and_year() -> None:
    assert parse_protocol_month("m24") == 24.0
    assert parse_protocol_month("y2") == 24.0


def test_parse_protocol_month_unknown_is_nan() -> None:
    assert np.isnan(
        parse_protocol_month(
            "unknown_visit"
        )
    )


def test_make_date_profile_empty() -> None:
    profile = make_date_profile([])

    assert profile.status == "no_clinical_date"
    assert profile.observed_date_cells == 0
    assert profile.source_count == 0
    assert profile.field_count == 0
    assert not date_available(profile)


def test_make_date_profile_tracks_sources_and_fields() -> None:
    profile = make_date_profile(
        [
            evidence(
                "2020-01-10",
                "ADAS",
                "VISDATE",
            ),
            evidence(
                "2020-01-10",
                "CDR",
                "VISDATE",
            ),
            evidence(
                "2020-01-10",
                "UCSFFSX7",
                "EXAMDATE",
            ),
        ]
    )

    assert profile.status == "exact_date_consensus"
    assert profile.distinct_dates == 1
    assert profile.source_count == 3
    assert profile.field_count == 3


def test_strict_interval_selects_unique_latest() -> None:
    earlier = make_date_profile(
        [
            evidence("2020-01-01"),
            evidence("2020-01-03", "CDR"),
        ]
    )

    later = make_date_profile(
        [
            evidence("2020-01-10"),
            evidence("2020-01-11", "CDR"),
        ]
    )

    winner = unique_latest_by_strict_intervals(
        [
            ("sc", earlier),
            ("bl", later),
        ]
    )

    assert winner is not None
    assert winner[0] == "bl"


def test_strict_interval_rejects_overlap() -> None:
    first = make_date_profile(
        [
            evidence("2020-01-01"),
            evidence("2020-01-10", "CDR"),
        ]
    )

    second = make_date_profile(
        [
            evidence("2020-01-05"),
            evidence("2020-01-15", "CDR"),
        ]
    )

    assert (
        unique_latest_by_strict_intervals(
            [
                ("sc", first),
                ("bl", second),
            ]
        )
        is None
    )


def test_strict_interval_rejects_missing_candidate_date() -> None:
    dated = make_date_profile(
        [
            evidence("2020-01-01"),
        ]
    )

    missing = make_date_profile([])

    assert (
        unique_latest_by_strict_intervals(
            [
                ("sc", dated),
                ("bl", missing),
            ]
        )
        is None
    )


def test_build_pairing_reference_keeps_unique_prior_protocol_visit() -> None:
    current_targets = pd.DataFrame(
        {
            "RID": ["XX0001"],
            "VISCODE2": ["m12"],
            "split": ["train"],
            "protocol_month": [12.0],
        }
    )

    candidate_modalities = {
        ("XX0001", "bl"): frozenset({"ADAS"}),
        ("XX0001", "m12"): frozenset({"ADAS"}),
    }

    profiles = {
        ("XX0001", "bl"): make_date_profile(
            [
                evidence("2020-01-01"),
            ]
        ),
        ("XX0001", "m12"): make_date_profile(
            [
                evidence("2021-01-01", "UCSFFSX7", "EXAMDATE"),
            ]
        ),
    }

    pairing = build_pairing_reference(
        current_targets=current_targets,
        candidate_modalities=candidate_modalities,
        profiles=profiles,
    )

    row = pairing.iloc[0]

    assert row["pairing_status"] == "eligible"
    assert row["pairing_reason"] == "eligible_unique_previous_visit"
    assert row["previous_VISCODE2"] == "bl"
    assert row["protocol_gap_months"] == 12.0


def test_build_pairing_reference_recovers_same_wave_by_strict_date() -> None:
    current_targets = pd.DataFrame(
        {
            "RID": ["XX0001"],
            "VISCODE2": ["m12"],
            "split": ["train"],
            "protocol_month": [12.0],
        }
    )

    candidate_modalities = {
        ("XX0001", "sc"): frozenset({"ADAS"}),
        ("XX0001", "bl"): frozenset({"CDR"}),
        ("XX0001", "m12"): frozenset({"ADAS"}),
    }

    profiles = {
        ("XX0001", "sc"): make_date_profile(
            [
                evidence("2020-01-01"),
                evidence("2020-01-02", "CDR"),
            ]
        ),
        ("XX0001", "bl"): make_date_profile(
            [
                evidence("2020-01-10"),
                evidence("2020-01-11", "CDR"),
            ]
        ),
        ("XX0001", "m12"): make_date_profile(
            [
                evidence("2021-01-01", "UCSFFSX7", "EXAMDATE"),
            ]
        ),
    }

    pairing = build_pairing_reference(
        current_targets=current_targets,
        candidate_modalities=candidate_modalities,
        profiles=profiles,
    )

    row = pairing.iloc[0]

    assert row["pairing_status"] == "eligible"
    assert row["pairing_reason"] == "eligible_strict_date_interval_winner"
    assert row["selection_method"] == "strict_date_intervals"
    assert row["previous_VISCODE2"] == "bl"
    assert row["conservative_date_gap_days"] > 0


def test_build_pairing_reference_rejects_winner_not_before_current() -> None:
    current_targets = pd.DataFrame(
        {
            "RID": ["XX0001"],
            "VISCODE2": ["m12"],
            "split": ["train"],
            "protocol_month": [12.0],
        }
    )

    candidate_modalities = {
        ("XX0001", "sc"): frozenset({"ADAS"}),
        ("XX0001", "bl"): frozenset({"CDR"}),
    }

    profiles = {
        ("XX0001", "sc"): make_date_profile(
            [
                evidence("2020-01-01"),
            ]
        ),
        ("XX0001", "bl"): make_date_profile(
            [
                evidence("2020-02-10"),
            ]
        ),
        ("XX0001", "m12"): make_date_profile(
            [
                evidence("2020-02-05", "UCSFFSX7", "EXAMDATE"),
            ]
        ),
    }

    pairing = build_pairing_reference(
        current_targets=current_targets,
        candidate_modalities=candidate_modalities,
        profiles=profiles,
    )

    row = pairing.iloc[0]

    assert row["pairing_status"] == "excluded"
    assert row["pairing_reason"] == "strict_latest_not_fully_before_current"


def test_pairing_summary_is_aggregate_only() -> None:
    pairing = pd.DataFrame(
        {
            "RID": ["XX0001", "XX0002"],
            "split": ["train", "train"],
            "pairing_status": ["eligible", "excluded"],
            "pairing_reason": [
                "eligible_unique_previous_visit",
                "no_strictly_earlier_known_protocol_visit",
            ],
            "selection_method": [
                "unique_most_recent_prior_protocol_month",
                "none",
            ],
        }
    )

    summary = summarize_pairing(pairing)

    assert "RID" not in summary.columns
    assert "VISCODE2" not in summary.columns
    assert summary["participant_count"].sum() == 2
    assert summary["test_rows_used"].eq(0).all()


def test_canonical_json_hash_is_order_independent() -> None:
    first = {
        "a": 1,
        "b": {
            "x": 2,
            "y": 3,
        },
    }

    second = {
        "b": {
            "y": 3,
            "x": 2,
        },
        "a": 1,
    }

    assert canonical_json_sha256(first) == canonical_json_sha256(second)
