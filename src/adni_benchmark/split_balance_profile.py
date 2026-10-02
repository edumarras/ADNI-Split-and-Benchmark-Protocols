from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from adni_benchmark.non_mri_resolution import (
    calculate_text_sha256,
    canonical_scalar,
)


class SplitBalanceProfileError(RuntimeError):
    pass


SPLIT_BALANCE_MODALITY_TOKENS: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "PTDEMOG",
    "APOERES",
)

SPLIT_RARE_CATEGORY_MIN_COUNT: Final[int] = 5
SPLIT_MISSING_CATEGORY: Final[str] = "__MISSING__"
SPLIT_RARE_CATEGORY: Final[str] = "__RARE__"


def collapse_split_rare_categories(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .astype("string")
        .fillna(
            SPLIT_MISSING_CATEGORY
        )
    )

    counts = normalized.value_counts(
        dropna=False
    )

    rare_values = counts.loc[
        counts.lt(
            SPLIT_RARE_CATEGORY_MIN_COUNT
        )
    ].index

    if len(rare_values):
        normalized = normalized.mask(
            normalized.isin(
                rare_values
            ),
            SPLIT_RARE_CATEGORY,
        )

    return normalized.astype("string")


def build_split_balance_profile(
    supervised_anchor: pd.DataFrame,
    participant_profile: pd.DataFrame,
) -> pd.DataFrame:
    # Freeze exactly the participant descriptors used by the historical
    # split search. MRI target values and model outcomes never enter here.
    required_anchor_columns = {
        "RID",
        "VISCODE2",
        "__protocol_month",
        "__eligible_visit_position",
    }

    for source_name in SPLIT_BALANCE_MODALITY_TOKENS:
        source_token = source_name.casefold()

        required_anchor_columns.update(
            {
                f"__available_{source_token}",
                f"__resolved_empty_{source_token}",
            }
        )

    missing_anchor_columns = sorted(
        required_anchor_columns
        - set(
            supervised_anchor.columns
        )
    )

    if missing_anchor_columns:
        raise SplitBalanceProfileError(
            "Supervised anchor is missing split-profile columns: "
            f"{missing_anchor_columns}"
        )

    required_participant_columns = {
        "RID",
        "__eligible_visit_count",
        "__longitudinal_class",
        "__first_diagnosis_category",
        "__ptgender_balance_category",
        "__ptgender_consensus_available",
        "__apoe_genotype_observed",
    }

    missing_participant_columns = sorted(
        required_participant_columns
        - set(
            participant_profile.columns
        )
    )

    if missing_participant_columns:
        raise SplitBalanceProfileError(
            "Participant profile is missing split-profile columns: "
            f"{missing_participant_columns}"
        )

    anchor = supervised_anchor.copy()

    effective_availability_columns: dict[
        str,
        str,
    ] = {}

    # A resolved record with no substantive observation is unavailable for
    # split balancing, even though a canonical source row exists.
    for source_name in SPLIT_BALANCE_MODALITY_TOKENS:
        source_token = source_name.casefold()

        raw_available_column = (
            f"__available_{source_token}"
        )

        empty_column = (
            f"__resolved_empty_{source_token}"
        )

        effective_column = (
            f"__split_available_{source_token}"
        )

        anchor[
            effective_column
        ] = (
            anchor[
                raw_available_column
            ].astype(bool)
            & ~anchor[
                empty_column
            ].astype(bool)
        )

        effective_availability_columns[
            source_name
        ] = effective_column

    grouped = anchor.groupby(
        "RID",
        dropna=False,
        sort=False,
    )

    balance = (
        participant_profile
        .copy()
        .set_index(
            "RID"
        )
    )

    if not balance.index.is_unique:
        raise SplitBalanceProfileError(
            "Participant balance-profile index is not unique."
        )

    balance[
        "n_visits"
    ] = (
        balance[
            "__eligible_visit_count"
        ]
        .astype(
            "int64"
        )
    )

    balance[
        "longitudinal_class"
    ] = (
        balance[
            "__longitudinal_class"
        ]
        .astype(
            "string"
        )
    )

    balance[
        "diagnosis_first"
    ] = (
        balance[
            "__first_diagnosis_category"
        ]
        .astype(
            "string"
        )
    )

    sex_available = (
        balance[
            "__ptgender_consensus_available"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        )
    )

    balance[
        "sex"
    ] = (
        balance[
            "__ptgender_balance_category"
        ]
        .astype(
            "string"
        )
        .where(
            sex_available,
            pd.NA,
        )
    )

    balance[
        "apoe_availability"
    ] = np.where(
        balance[
            "__apoe_genotype_observed"
        ]
        .fillna(
            False
        )
        .astype(
            bool
        ),
        "available",
        "missing",
    )

    first_rows = anchor.loc[
        anchor[
            "__eligible_visit_position"
        ].eq(
            1
        )
    ].copy()

    first_rows = first_rows.set_index(
        "RID"
    )

    if not first_rows.index.is_unique:
        raise SplitBalanceProfileError(
            "Split balance profile found multiple first eligible visits."
        )

    first_pattern_parts: list[
        pd.Series
    ] = []

    participant_rate_columns: list[
        str
    ] = []

    for source_name in SPLIT_BALANCE_MODALITY_TOKENS:
        source_token = source_name.casefold()

        effective_column = (
            effective_availability_columns[
                source_name
            ]
        )

        rate_column = (
            f"availability_rate__{source_token}"
        )

        rates = grouped[
            effective_column
        ].mean()

        balance[
            rate_column
        ] = rates.astype(
            "float64"
        )

        participant_rate_columns.append(
            rate_column
        )

        first_part = (
            source_token
            + ":"
            + first_rows[
                effective_column
            ]
            .astype(
                "int64"
            )
            .astype(
                "string"
            )
        )

        first_pattern_parts.append(
            first_part
        )

    first_pattern = (
        first_pattern_parts[
            0
        ]
    )

    for pattern_part in (
        first_pattern_parts[
            1:
        ]
    ):
        first_pattern = (
            first_pattern
            + "|"
            + pattern_part
        )

    balance[
        "modality_pattern_first"
    ] = first_pattern

    # Sum of participant-level modality availability rates equals the mean
    # number of effectively available modalities across that participant's
    # eligible visits.
    balance[
        "modality_count_mean"
    ] = (
        balance.loc[
            :,
            participant_rate_columns,
        ]
        .sum(
            axis=1
        )
        .astype(
            "float64"
        )
    )

    examdate_count = (
        grouped[
            "__examdate_for_split"
        ].count()
        if "__examdate_for_split"
        in anchor.columns
        else None
    )

    if examdate_count is not None:
        examdate_min = grouped[
            "__examdate_for_split"
        ].min()

        examdate_max = grouped[
            "__examdate_for_split"
        ].max()

        date_followup = (
            (
                examdate_max
                - examdate_min
            )
            .dt.days
            .div(
                30.4375
            )
        )

    else:
        examdate_count = pd.Series(
            0,
            index=balance.index,
            dtype="int64",
        )

        date_followup = pd.Series(
            np.nan,
            index=balance.index,
            dtype="float64",
        )

    protocol_count = grouped[
        "__protocol_month"
    ].count()

    protocol_min = grouped[
        "__protocol_month"
    ].min()

    protocol_max = grouped[
        "__protocol_month"
    ].max()

    protocol_followup = (
        protocol_max
        - protocol_min
    ).astype(
        "float64"
    )

    # Examination-date duration wins when at least two valid dates exist.
    # Otherwise protocol month is the fallback; a single usable visit gives 0.
    balance[
        "followup_months"
    ] = (
        date_followup
        .where(
            examdate_count.ge(
                2
            ),
            protocol_followup.where(
                protocol_count.ge(
                    2
                ),
                0.0,
            ),
        )
        .fillna(
            0.0
        )
        .astype(
            "float64"
        )
    )

    categorical_columns = (
        "diagnosis_first",
        "sex",
        "apoe_availability",
        "modality_pattern_first",
        "longitudinal_class",
    )

    for column in categorical_columns:
        balance[
            column
        ] = (
            collapse_split_rare_categories(
                balance[
                    column
                ]
            )
        )

    output_columns = [
        "n_visits",
        "diagnosis_first",
        "sex",
        "apoe_availability",
        "modality_pattern_first",
        "longitudinal_class",
        "followup_months",
        "modality_count_mean",
        *participant_rate_columns,
    ]

    balance = (
        balance.loc[
            :,
            output_columns,
        ]
        .reset_index()
        .sort_values(
            "RID",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            balance
        )
        != participant_profile[
            "RID"
        ].nunique()
    ):
        raise SplitBalanceProfileError(
            "Split balance profile participant accounting failed."
        )

    if (
        int(
            balance[
                "n_visits"
            ].sum()
        )
        != len(
            supervised_anchor
        )
    ):
        raise SplitBalanceProfileError(
            "Split balance profile visit accounting failed."
        )

    return balance


def calculate_participant_profile_sha256(
    profile: pd.DataFrame,
) -> str:
    # The hash intentionally depends on both row and column order.
    columns = list(
        profile.columns
    )

    canonical_lines = [
        "\x1f".join(
            f"{column}={canonical_scalar(row[column])}"
            for column in columns
        )
        for _, row
        in profile.iterrows()
    ]

    return calculate_text_sha256(
        "\n".join(
            canonical_lines
        )
        + "\n"
    )
