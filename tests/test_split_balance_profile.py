import numpy as np
import pandas as pd
import pytest

from adni_benchmark.split_balance_profile import (
    SPLIT_BALANCE_MODALITY_TOKENS,
    SPLIT_MISSING_CATEGORY,
    SPLIT_RARE_CATEGORY,
    SplitBalanceProfileError,
    build_split_balance_profile,
    calculate_participant_profile_sha256,
    collapse_split_rare_categories,
)


def make_profiles(
    *,
    include_examdates: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    participant_ids = [
        f"XX{index:02d}"
        for index in range(
            1,
            6,
        )
    ]

    anchor_records: list[
        dict[str, object]
    ] = []

    participant_records: list[
        dict[str, object]
    ] = []

    for participant_id in participant_ids:
        for position, (
            viscode,
            protocol_month,
        ) in enumerate(
            (
                (
                    "bl",
                    0.0,
                ),
                (
                    "m12",
                    12.0,
                ),
            ),
            start=1,
        ):
            record: dict[
                str,
                object,
            ] = {
                "RID": participant_id,
                "VISCODE2": viscode,
                "__protocol_month": protocol_month,
                "__eligible_visit_position": position,
            }

            if include_examdates:
                record[
                    "__examdate_for_split"
                ] = pd.Timestamp(
                    "2020-01-01"
                    if position == 1
                    else "2021-01-01"
                )

            for source_name in (
                SPLIT_BALANCE_MODALITY_TOKENS
            ):
                token = (
                    source_name.casefold()
                )

                record[
                    f"__available_{token}"
                ] = True

                record[
                    f"__resolved_empty_{token}"
                ] = (
                    source_name
                    == "CDR"
                    and position == 1
                )

            anchor_records.append(
                record
            )

        participant_records.append(
            {
                "RID": participant_id,
                "__eligible_visit_count": 2,
                "__longitudinal_class": "two_eligible_visits",
                "__first_diagnosis_category": "DX_A",
                "__ptgender_balance_category": "A",
                "__ptgender_consensus_available": True,
                "__apoe_genotype_observed": True,
            }
        )

    return (
        pd.DataFrame(
            anchor_records
        ),
        pd.DataFrame(
            participant_records
        ),
    )


def test_collapse_split_rare_categories() -> None:
    series = pd.Series(
        [
            *(
                [
                    "common"
                ]
                * 5
            ),
            *(
                [
                    "rare"
                ]
                * 4
            ),
            pd.NA,
        ],
        dtype="string",
    )

    result = (
        collapse_split_rare_categories(
            series
        )
    )

    assert (
        result.iloc[
            :5
        ]
        == "common"
    ).all()

    assert (
        result.iloc[
            5:
        ]
        == SPLIT_RARE_CATEGORY
    ).all()


def test_missing_category_is_preserved_when_not_rare() -> None:
    series = pd.Series(
        [
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
            pd.NA,
        ],
        dtype="string",
    )

    result = (
        collapse_split_rare_categories(
            series
        )
    )

    assert set(
        result.tolist()
    ) == {
        SPLIT_MISSING_CATEGORY
    }


def test_balance_profile_has_one_row_per_participant() -> None:
    anchor, participants = (
        make_profiles()
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert len(
        balance
    ) == 5

    assert (
        balance[
            "RID"
        ].is_unique
    )

    assert int(
        balance[
            "n_visits"
        ].sum()
    ) == 10


def test_resolved_empty_record_is_unavailable_for_balance() -> None:
    anchor, participants = (
        make_profiles()
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert np.allclose(
        balance[
            "availability_rate__cdr"
        ],
        0.5,
    )

    assert np.allclose(
        balance[
            "modality_count_mean"
        ],
        7.5,
    )


def test_first_modality_pattern_uses_effective_availability() -> None:
    anchor, participants = (
        make_profiles()
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    expected = (
        "adas:1|cdr:0|faq:1|mmse:1|moca:1|"
        "neurobat:1|ptdemog:1|apoeres:1"
    )

    assert set(
        balance[
            "modality_pattern_first"
        ].tolist()
    ) == {
        expected
    }


def test_date_followup_precedes_protocol_followup() -> None:
    anchor, participants = (
        make_profiles()
    )

    anchor.loc[
        anchor[
            "__eligible_visit_position"
        ].eq(
            2
        ),
        "__protocol_month",
    ] = 24.0

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    expected_months = (
        366.0
        / 30.4375
    )

    assert np.allclose(
        balance[
            "followup_months"
        ],
        expected_months,
    )

    assert not np.allclose(
        balance[
            "followup_months"
        ],
        24.0,
    )


def test_protocol_followup_is_used_without_examdates() -> None:
    anchor, participants = (
        make_profiles(
            include_examdates=False,
        )
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert np.allclose(
        balance[
            "followup_months"
        ],
        12.0,
    )


def test_single_usable_visit_has_zero_followup() -> None:
    anchor, participants = (
        make_profiles(
            include_examdates=False,
        )
    )

    anchor = (
        anchor.loc[
            anchor[
                "__eligible_visit_position"
            ].eq(
                1
            )
        ]
        .copy()
    )

    participants[
        "__eligible_visit_count"
    ] = 1

    participants[
        "__longitudinal_class"
    ] = "single_eligible_visit"

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert np.allclose(
        balance[
            "followup_months"
        ],
        0.0,
    )


def test_unavailable_ptgender_becomes_missing_split_category() -> None:
    anchor, participants = (
        make_profiles()
    )

    participants[
        "__ptgender_consensus_available"
    ] = False

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert set(
        balance[
            "sex"
        ].tolist()
    ) == {
        SPLIT_MISSING_CATEGORY
    }


def test_apoe_balance_uses_observation_availability_only() -> None:
    anchor, participants = (
        make_profiles()
    )

    participants[
        "__apoe_genotype_observed"
    ] = False

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert set(
        balance[
            "apoe_availability"
        ].tolist()
    ) == {
        "missing"
    }


def test_output_column_order_matches_split_contract() -> None:
    anchor, participants = (
        make_profiles()
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    expected = [
        "RID",
        "n_visits",
        "diagnosis_first",
        "sex",
        "apoe_availability",
        "modality_pattern_first",
        "longitudinal_class",
        "followup_months",
        "modality_count_mean",
        *[
            f"availability_rate__{source_name.casefold()}"
            for source_name
            in SPLIT_BALANCE_MODALITY_TOKENS
        ],
    ]

    assert list(
        balance.columns
    ) == expected


def test_missing_anchor_column_aborts() -> None:
    anchor, participants = (
        make_profiles()
    )

    anchor = anchor.drop(
        columns=[
            "__available_adas",
        ]
    )

    with pytest.raises(
        SplitBalanceProfileError,
        match="Supervised anchor is missing split-profile columns",
    ):
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )


def test_missing_participant_column_aborts() -> None:
    anchor, participants = (
        make_profiles()
    )

    participants = participants.drop(
        columns=[
            "__apoe_genotype_observed",
        ]
    )

    with pytest.raises(
        SplitBalanceProfileError,
        match="Participant profile is missing split-profile columns",
    ):
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )


def test_duplicate_participant_profile_aborts() -> None:
    anchor, participants = (
        make_profiles()
    )

    participants = pd.concat(
        [
            participants,
            participants.iloc[
                [
                    0
                ]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        SplitBalanceProfileError,
        match="index is not unique",
    ):
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )


def test_profile_hash_is_deterministic_and_value_sensitive() -> None:
    anchor, participants = (
        make_profiles()
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    first = (
        calculate_participant_profile_sha256(
            balance
        )
    )

    second = (
        calculate_participant_profile_sha256(
            balance.copy()
        )
    )

    changed = balance.copy()

    changed.loc[
        0,
        "n_visits",
    ] = 3

    third = (
        calculate_participant_profile_sha256(
            changed
        )
    )

    assert first == second
    assert first != third


def test_profile_is_sorted_by_rid() -> None:
    anchor, participants = (
        make_profiles()
    )

    anchor = anchor.sample(
        frac=1.0,
        random_state=11,
    ).reset_index(
        drop=True
    )

    participants = participants.sample(
        frac=1.0,
        random_state=12,
    ).reset_index(
        drop=True
    )

    balance = (
        build_split_balance_profile(
            supervised_anchor=anchor,
            participant_profile=participants,
        )
    )

    assert balance[
        "RID"
    ].tolist() == sorted(
        balance[
            "RID"
        ].tolist()
    )
