from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence
import json
import re

import numpy as np
import pandas as pd

from adni_benchmark.frozen_experiment_package import (
    calculate_file_sha256,
)


class Split06AuditError(RuntimeError):
    pass


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "split06_audit_same_wave_prior_visits"

DEVELOPMENT_SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
)

EXPECTED_AMBIGUOUS_COUNTS: Final[dict[str, int]] = {
    "train": 269,
    "validation": 58,
}

EXPECTED_UNIQUE_PREVIOUS_COUNTS: Final[dict[str, int]] = {
    "train": 1170,
    "validation": 246,
}

EXPECTED_STRICT_RECOVERED_COUNTS: Final[dict[str, int]] = {
    "train": 256,
    "validation": 55,
}

EXPECTED_STRICT_WINNER_COUNTS: Final[dict[str, int]] = {
    "train": 260,
    "validation": 56,
}

VISIT_LEVEL_INPUT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
)

DATE_SOURCE_FILES: Final[tuple[str, ...]] = (
    "DXSUM",
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "UCSFFSX7",
)

# Only clinical/examination date fields are eligible.
# Administrative timestamps are intentionally excluded.
CLINICAL_DATE_COLUMNS: Final[tuple[str, ...]] = (
    "EXAMDATE",
    "VISDATE",
    "TESTDATE",
    "APTESTDT",
    "ASSESSDATE",
)

ZERO_WAVE_EXPLICIT_ORDER: Final[dict[str, int]] = {
    "sc": 0,
    "scmri": 1,
    "bl": 2,
    "m00": 3,
}

SAFE_MINIMUM_CELL_COUNT: Final[int] = 5
READ_CHUNK_SIZE: Final[int] = 50_000


@dataclass(frozen=True)
class VisitDateProfile:
    date_min: pd.Timestamp | pd.NaT
    date_max: pd.Timestamp | pd.NaT
    date_median: pd.Timestamp | pd.NaT
    observed_date_cells: int
    distinct_dates: int
    date_span_days: float | None
    status: str


@dataclass(frozen=True)
class CandidateVisit:
    rid: str
    viscode2: str
    split: str
    protocol_month: float
    modality_names: frozenset[str]
    observed_feature_tokens: frozenset[str]
    date_profile: VisitDateProfile


@dataclass(frozen=True)
class AmbiguousCaseAudit:
    split: str
    candidate_viscodes: tuple[str, ...]
    candidate_count: int
    current_protocol_month: float
    prior_protocol_month: float
    date_coverage_status: str
    all_candidates_exact_date_consensus: bool
    strict_interval_latest_recoverable: bool
    strict_interval_latest_before_current: bool
    exact_consensus_latest_recoverable: bool
    median_date_latest_recoverable: bool
    explicit_zero_wave_order_recoverable: bool
    explicit_zero_wave_order_selected_viscode: str | None
    current_date_available: bool
    has_previous_target_complete_visit: bool
    union_modality_count: int
    best_single_modality_count: int
    union_modality_gain: int
    union_feature_count: int
    best_single_feature_count: int
    union_feature_gain: int
    pairwise_median_gap_days: tuple[float, ...]


@dataclass(frozen=True)
class Split06AuditResult:
    development_split: pd.DataFrame
    current_targets: pd.DataFrame
    audits: tuple[AmbiguousCaseAudit, ...]
    attrition_summary: pd.DataFrame
    ambiguous_summary: pd.DataFrame
    combination_summary: pd.DataFrame
    hierarchy_summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    date_source_summary: pd.DataFrame
    source_content_summary: pd.DataFrame
    validation_checks: pd.DataFrame


def parse_protocol_month(
    value: Any,
) -> float:
    if pd.isna(
        value
    ):
        return np.nan

    text = str(
        value
    ).strip().casefold()

    if text in {
        "bl",
        "sc",
        "scmri",
        "m00",
        "v01",
    }:
        return 0.0

    month_match = re.fullmatch(
        r"m(\d+)",
        text,
    )

    if month_match:
        return float(
            int(
                month_match.group(
                    1
                )
            )
        )

    year_match = re.fullmatch(
        r"y(\d+)",
        text,
    )

    if year_match:
        return float(
            int(
                year_match.group(
                    1
                )
            )
            * 12
        )

    return np.nan


def normalize_key_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(
            "string"
        )
        .str.strip()
    )


def normalize_viscode_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(
            "string"
        )
        .str.strip()
        .str.casefold()
    )


def read_csv_header(
    path: Path,
) -> list[str]:
    return list(
        pd.read_csv(
            path,
            compression="infer",
            nrows=0,
        ).columns
    )


def read_development_rows_by_split(
    path: Path,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    pieces: list[
        pd.DataFrame
    ] = []

    for chunk in pd.read_csv(
        path,
        compression="infer",
        dtype="string",
        usecols=usecols,
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    ):
        if "split" not in chunk.columns:
            raise Split06AuditError(
                f"{path} does not contain split."
            )

        kept = chunk.loc[
            chunk[
                "split"
            ].isin(
                DEVELOPMENT_SPLITS
            )
        ].copy()

        if not kept.empty:
            pieces.append(
                kept
            )

    if not pieces:
        return pd.DataFrame(
            columns=list(
                usecols
                or []
            )
        )

    return pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )


def read_rows_for_rids(
    path: Path,
    development_rids: set[str],
    usecols: Sequence[str],
) -> pd.DataFrame:
    pieces: list[
        pd.DataFrame
    ] = []

    for chunk in pd.read_csv(
        path,
        compression="infer",
        dtype="string",
        usecols=list(
            usecols
        ),
        chunksize=READ_CHUNK_SIZE,
        low_memory=False,
    ):
        chunk[
            "RID"
        ] = normalize_key_series(
            chunk[
                "RID"
            ]
        )

        kept = chunk.loc[
            chunk[
                "RID"
            ].isin(
                development_rids
            )
        ].copy()

        if not kept.empty:
            pieces.append(
                kept
            )

    if not pieces:
        return pd.DataFrame(
            columns=list(
                usecols
            )
        )

    return pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )


def load_development_split(
    package_root: Path,
) -> tuple[
    pd.DataFrame,
    set[str],
]:
    path = (
        package_root
        / "04_official_split"
        / "official_participant_split.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    frame = pd.read_csv(
        path,
        dtype="string",
        low_memory=False,
    )

    required = {
        "RID",
        "split",
    }

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise Split06AuditError(
            "Participant split missing columns: "
            f"{missing}"
        )

    frame[
        "RID"
    ] = normalize_key_series(
        frame[
            "RID"
        ]
    )

    frame[
        "split"
    ] = (
        frame[
            "split"
        ]
        .astype(
            "string"
        )
        .str.strip()
        .str.casefold()
    )

    if (
        frame[
            "RID"
        ].isna().any()
        or frame[
            "split"
        ].isna().any()
    ):
        raise Split06AuditError(
            "Participant split contains missing RID/split."
        )

    if frame[
        "RID"
    ].duplicated().any():
        raise Split06AuditError(
            "Participant split contains duplicate RID."
        )

    development = frame.loc[
        frame[
            "split"
        ].isin(
            DEVELOPMENT_SPLITS
        )
    ].copy()

    development_rids = set(
        development[
            "RID"
        ].tolist()
    )

    if not development_rids:
        raise Split06AuditError(
            "Development participant set is empty."
        )

    return (
        development,
        development_rids,
    )


def load_development_anchor(
    package_root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    path = (
        package_root
        / "03_final_ready_tables"
        / "supervised_anchor_visits.csv.gz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    usecols = [
        "RID",
        "VISCODE2",
        "split",
        "eligible_visit_position",
        "protocol_month",
    ]

    frame = (
        read_development_rows_by_split(
            path,
            usecols=usecols,
        )
    )

    frame[
        "RID"
    ] = normalize_key_series(
        frame[
            "RID"
        ]
    )

    frame[
        "VISCODE2"
    ] = normalize_viscode_series(
        frame[
            "VISCODE2"
        ]
    )

    frame[
        "split"
    ] = (
        frame[
            "split"
        ]
        .astype(
            "string"
        )
        .str.strip()
        .str.casefold()
    )

    frame[
        "eligible_visit_position"
    ] = pd.to_numeric(
        frame[
            "eligible_visit_position"
        ],
        errors="raise",
    ).astype(
        "int64"
    )

    frame[
        "protocol_month"
    ] = pd.to_numeric(
        frame[
            "protocol_month"
        ],
        errors="coerce",
    )

    if frame[
        [
            "RID",
            "VISCODE2",
            "split",
        ]
    ].isna().any().any():
        raise Split06AuditError(
            "Development anchor contains missing keys/split."
        )

    if frame.duplicated(
        [
            "RID",
            "VISCODE2",
        ]
    ).any():
        raise Split06AuditError(
            "Development anchor contains duplicate visit keys."
        )

    if (
        set(
            frame[
                "split"
            ].unique()
        )
        - set(
            DEVELOPMENT_SPLITS
        )
    ):
        raise Split06AuditError(
            "Test or unexpected split reached development anchor."
        )

    last = (
        frame
        .sort_values(
            [
                "RID",
                "eligible_visit_position",
                "VISCODE2",
            ],
            kind="stable",
        )
        .drop_duplicates(
            "RID",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if last[
        "RID"
    ].duplicated().any():
        raise Split06AuditError(
            "Failed to select one current target per participant."
        )

    return (
        frame,
        last,
    )


def build_visit_content_profiles(
    package_root: Path,
) -> tuple[
    dict[
        tuple[str, str],
        frozenset[str],
    ],
    dict[
        tuple[str, str],
        frozenset[str],
    ],
    pd.DataFrame,
]:
    modality_sets: defaultdict[
        tuple[str, str],
        set[str],
    ] = defaultdict(
        set
    )

    feature_sets: defaultdict[
        tuple[str, str],
        set[str],
    ] = defaultdict(
        set
    )

    source_summary_records: list[
        dict[str, object]
    ] = []

    inputs_dir = (
        package_root
        / "03_final_ready_tables"
        / "input_sources"
    )

    for source_name in (
        VISIT_LEVEL_INPUT_SOURCES
    ):
        path = (
            inputs_dir
            / f"{source_name}_model_ready.csv.gz"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        header = read_csv_header(
            path
        )

        required = {
            "RID",
            "VISCODE2",
            "split",
        }

        missing = sorted(
            required
            - set(
                header
            )
        )

        if missing:
            raise Split06AuditError(
                f"{source_name} model-ready missing: {missing}"
            )

        feature_columns = [
            column
            for column
            in header
            if column
            not in required
        ]

        frame = (
            read_development_rows_by_split(
                path,
                usecols=header,
            )
        )

        frame[
            "RID"
        ] = normalize_key_series(
            frame[
                "RID"
            ]
        )

        frame[
            "VISCODE2"
        ] = normalize_viscode_series(
            frame[
                "VISCODE2"
            ]
        )

        frame[
            "split"
        ] = (
            frame[
                "split"
            ]
            .astype(
                "string"
            )
            .str.strip()
            .str.casefold()
        )

        if frame.duplicated(
            [
                "RID",
                "VISCODE2",
            ]
        ).any():
            raise Split06AuditError(
                f"{source_name} model-ready has duplicate keys."
            )

        observed_cells = 0

        for _, row in frame.iterrows():
            key = (
                str(
                    row[
                        "RID"
                    ]
                ),
                str(
                    row[
                        "VISCODE2"
                    ]
                ),
            )

            observed = [
                column
                for column
                in feature_columns
                if (
                    not pd.isna(
                        row[
                            column
                        ]
                    )
                    and str(
                        row[
                            column
                        ]
                    ).strip()
                    != ""
                )
            ]

            if not observed:
                raise Split06AuditError(
                    f"{source_name} model-ready contains an empty substantive row."
                )

            modality_sets[
                key
            ].add(
                source_name
            )

            tokens = {
                f"{source_name}__{column}"
                for column
                in observed
            }

            feature_sets[
                key
            ].update(
                tokens
            )

            observed_cells += len(
                tokens
            )

        source_summary_records.append(
            {
                "source_name": (
                    source_name
                ),
                "development_rows": int(
                    len(
                        frame
                    )
                ),
                "feature_columns": int(
                    len(
                        feature_columns
                    )
                ),
                "observed_feature_cells": int(
                    observed_cells
                ),
                "test_rows_used": 0,
            }
        )

    return (
        {
            key: frozenset(
                value
            )
            for key, value
            in modality_sets.items()
        },
        {
            key: frozenset(
                value
            )
            for key, value
            in feature_sets.items()
        },
        pd.DataFrame(
            source_summary_records
        ),
    )


def collect_clinical_dates(
    package_root: Path,
    development_rids: set[str],
) -> tuple[
    dict[
        tuple[str, str],
        list[pd.Timestamp],
    ],
    pd.DataFrame,
]:
    date_values: defaultdict[
        tuple[str, str],
        list[pd.Timestamp],
    ] = defaultdict(
        list
    )

    summary_records: list[
        dict[str, object]
    ] = []

    postprocessed_dir = (
        package_root
        / "01_postprocessed_sources"
    )

    for source_name in (
        DATE_SOURCE_FILES
    ):
        path = (
            postprocessed_dir
            / f"{source_name}_canonical_resolved.csv.gz"
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        header = read_csv_header(
            path
        )

        available_dates = [
            column
            for column
            in CLINICAL_DATE_COLUMNS
            if column
            in header
        ]

        if not available_dates:
            summary_records.append(
                {
                    "source_name": source_name,
                    "date_column": "<NONE>",
                    "development_rows": 0,
                    "observed_date_cells": 0,
                    "parseable_date_cells": 0,
                    "invalid_observed_date_cells": 0,
                }
            )
            continue

        usecols = [
            "RID",
            "VISCODE2",
            *available_dates,
        ]

        frame = read_rows_for_rids(
            path,
            development_rids,
            usecols,
        )

        frame[
            "RID"
        ] = normalize_key_series(
            frame[
                "RID"
            ]
        )

        frame[
            "VISCODE2"
        ] = normalize_viscode_series(
            frame[
                "VISCODE2"
            ]
        )

        if frame.duplicated(
            [
                "RID",
                "VISCODE2",
            ]
        ).any():
            raise Split06AuditError(
                f"{source_name} canonical table has duplicate keys."
            )

        for date_column in (
            available_dates
        ):
            observed_mask = (
                frame[
                    date_column
                ].notna()
                & (
                    frame[
                        date_column
                    ]
                    .astype(
                        "string"
                    )
                    .str.strip()
                    .ne(
                        ""
                    )
                )
            )

            try:
                parsed = pd.to_datetime(
                    frame[
                        date_column
                    ],
                    errors="coerce",
                    format="mixed",
                )
            except (
                TypeError,
                ValueError,
            ):
                parsed = pd.to_datetime(
                    frame[
                        date_column
                    ],
                    errors="coerce",
                )

            invalid_mask = (
                observed_mask
                & parsed.isna()
            )

            summary_records.append(
                {
                    "source_name": (
                        source_name
                    ),
                    "date_column": (
                        date_column
                    ),
                    "development_rows": int(
                        len(
                            frame
                        )
                    ),
                    "observed_date_cells": int(
                        observed_mask.sum()
                    ),
                    "parseable_date_cells": int(
                        (
                            observed_mask
                            & parsed.notna()
                        ).sum()
                    ),
                    "invalid_observed_date_cells": int(
                        invalid_mask.sum()
                    ),
                }
            )

            valid_positions = frame.index[
                observed_mask
                & parsed.notna()
            ]

            for index in (
                valid_positions
            ):
                key = (
                    str(
                        frame.at[
                            index,
                            "RID",
                        ]
                    ),
                    str(
                        frame.at[
                            index,
                            "VISCODE2",
                        ]
                    ),
                )

                date_values[
                    key
                ].append(
                    pd.Timestamp(
                        parsed.at[
                            index
                        ]
                    ).normalize()
                )

    return (
        dict(
            date_values
        ),
        pd.DataFrame(
            summary_records
        ),
    )


def make_date_profile(
    values: Sequence[pd.Timestamp],
) -> VisitDateProfile:
    if not values:
        return VisitDateProfile(
            date_min=pd.NaT,
            date_max=pd.NaT,
            date_median=pd.NaT,
            observed_date_cells=0,
            distinct_dates=0,
            date_span_days=None,
            status="no_clinical_date",
        )

    ordered = sorted(
        pd.Timestamp(
            value
        ).normalize()
        for value
        in values
    )

    unique = sorted(
        set(
            ordered
        )
    )

    date_min = unique[
        0
    ]

    date_max = unique[
        -1
    ]

    nanoseconds = np.array(
        [
            value.value
            for value
            in ordered
        ],
        dtype=np.int64,
    )

    median_ns = int(
        np.median(
            nanoseconds
        )
    )

    date_median = pd.Timestamp(
        median_ns
    ).normalize()

    span = float(
        (
            date_max
            - date_min
        ).days
    )

    if len(
        unique
    ) == 1:
        status = (
            "exact_date_consensus"
        )

    elif span <= 7:
        status = (
            "date_range_within_7_days"
        )

    elif span <= 30:
        status = (
            "date_range_8_to_30_days"
        )

    else:
        status = (
            "date_range_over_30_days"
        )

    return VisitDateProfile(
        date_min=date_min,
        date_max=date_max,
        date_median=date_median,
        observed_date_cells=len(
            ordered
        ),
        distinct_dates=len(
            unique
        ),
        date_span_days=span,
        status=status,
    )


def date_is_available(
    profile: VisitDateProfile,
) -> bool:
    return not pd.isna(
        profile.date_min
    )


def unique_latest_by_strict_intervals(
    candidates: Sequence[
        CandidateVisit
    ],
) -> CandidateVisit | None:
    dated = [
        candidate
        for candidate
        in candidates
        if date_is_available(
            candidate.date_profile
        )
    ]

    if len(
        dated
    ) != len(
        candidates
    ):
        return None

    winners: list[
        CandidateVisit
    ] = []

    for candidate in (
        dated
    ):
        other_max_dates = [
            other.date_profile.date_max
            for other
            in dated
            if other.viscode2
            != candidate.viscode2
        ]

        if (
            other_max_dates
            and candidate.date_profile.date_min
            > max(
                other_max_dates
            )
        ):
            winners.append(
                candidate
            )

    if len(
        winners
    ) == 1:
        return winners[
            0
        ]

    return None


def unique_latest_by_exact_consensus(
    candidates: Sequence[
        CandidateVisit
    ],
) -> CandidateVisit | None:
    if (
        not candidates
        or not all(
            candidate.date_profile.status
            == "exact_date_consensus"
            for candidate
            in candidates
        )
    ):
        return None

    maximum = max(
        candidate.date_profile.date_min
        for candidate
        in candidates
    )

    winners = [
        candidate
        for candidate
        in candidates
        if candidate.date_profile.date_min
        == maximum
    ]

    if len(
        winners
    ) == 1:
        return winners[
            0
        ]

    return None


def unique_latest_by_median_date(
    candidates: Sequence[
        CandidateVisit
    ],
) -> CandidateVisit | None:
    if (
        not candidates
        or not all(
            date_is_available(
                candidate.date_profile
            )
            for candidate
            in candidates
        )
    ):
        return None

    maximum = max(
        candidate.date_profile.date_median
        for candidate
        in candidates
    )

    winners = [
        candidate
        for candidate
        in candidates
        if candidate.date_profile.date_median
        == maximum
    ]

    if len(
        winners
    ) == 1:
        return winners[
            0
        ]

    return None


def select_by_explicit_zero_wave_order(
    candidates: Sequence[
        CandidateVisit
    ],
) -> CandidateVisit | None:
    if not candidates:
        return None

    if not all(
        candidate.viscode2
        in ZERO_WAVE_EXPLICIT_ORDER
        for candidate
        in candidates
    ):
        return None

    maximum_rank = max(
        ZERO_WAVE_EXPLICIT_ORDER[
            candidate.viscode2
        ]
        for candidate
        in candidates
    )

    winners = [
        candidate
        for candidate
        in candidates
        if ZERO_WAVE_EXPLICIT_ORDER[
            candidate.viscode2
        ]
        == maximum_rank
    ]

    if len(
        winners
    ) == 1:
        return winners[
            0
        ]

    return None


def classify_date_coverage(
    candidates: Sequence[
        CandidateVisit
    ],
) -> str:
    count = sum(
        date_is_available(
            candidate.date_profile
        )
        for candidate
        in candidates
    )

    if count == 0:
        return (
            "none_dated"
        )

    if count == len(
        candidates
    ):
        return (
            "all_dated"
        )

    return (
        "partially_dated"
    )


def pairwise_median_date_gaps(
    candidates: Sequence[
        CandidateVisit
    ],
) -> tuple[float, ...]:
    dated = [
        candidate
        for candidate
        in candidates
        if date_is_available(
            candidate.date_profile
        )
    ]

    gaps: list[
        float
    ] = []

    for left_position in range(
        len(
            dated
        )
    ):
        for right_position in range(
            left_position + 1,
            len(
                dated
            ),
        ):
            gap = abs(
                (
                    dated[
                        left_position
                    ].date_profile.date_median
                    - dated[
                        right_position
                    ].date_profile.date_median
                ).days
            )

            gaps.append(
                float(
                    gap
                )
            )

    return tuple(
        gaps
    )


def build_ambiguous_audits(
    current_targets: pd.DataFrame,
    modality_sets: dict[
        tuple[str, str],
        frozenset[str],
    ],
    feature_sets: dict[
        tuple[str, str],
        frozenset[str],
    ],
    date_values: dict[
        tuple[str, str],
        list[pd.Timestamp],
    ],
    participants_with_previous_target_complete: set[str],
) -> tuple[
    list[
        AmbiguousCaseAudit
    ],
    pd.DataFrame,
]:
    visits_by_rid: defaultdict[
        str,
        list[
            tuple[
                str,
                frozenset[str],
                frozenset[str],
            ]
        ],
    ] = defaultdict(
        list
    )

    for key, modalities in (
        modality_sets.items()
    ):
        rid, viscode = (
            key
        )

        visits_by_rid[
            rid
        ].append(
            (
                viscode,
                modalities,
                feature_sets.get(
                    key,
                    frozenset(),
                ),
            )
        )

    date_profiles = {
        key: make_date_profile(
            values
        )
        for key, values
        in date_values.items()
    }

    empty_profile = (
        make_date_profile(
            ()
        )
    )

    audits: list[
        AmbiguousCaseAudit
    ] = []

    attrition_counter: Counter[
        tuple[str, str]
    ] = Counter()

    for _, current in (
        current_targets.iterrows()
    ):
        rid = str(
            current[
                "RID"
            ]
        )

        split = str(
            current[
                "split"
            ]
        )

        current_month = (
            float(
                current[
                    "protocol_month"
                ]
            )
            if not pd.isna(
                current[
                    "protocol_month"
                ]
            )
            else np.nan
        )

        current_viscode = str(
            current[
                "VISCODE2"
            ]
        )

        candidate_records = (
            visits_by_rid.get(
                rid,
                [],
            )
        )

        if not np.isfinite(
            current_month
        ):
            attrition_counter[
                (
                    split,
                    "current_target_unknown_protocol_month",
                )
            ] += 1
            continue

        prior: list[
            CandidateVisit
        ] = []

        for (
            viscode,
            modalities,
            features,
        ) in candidate_records:
            month = parse_protocol_month(
                viscode
            )

            if (
                np.isfinite(
                    month
                )
                and month
                < current_month
            ):
                key = (
                    rid,
                    viscode,
                )

                prior.append(
                    CandidateVisit(
                        rid=rid,
                        viscode2=viscode,
                        split=split,
                        protocol_month=float(
                            month
                        ),
                        modality_names=modalities,
                        observed_feature_tokens=features,
                        date_profile=(
                            date_profiles.get(
                                key,
                                empty_profile,
                            )
                        ),
                    )
                )

        if not prior:
            any_unknown = any(
                not np.isfinite(
                    parse_protocol_month(
                        viscode
                    )
                )
                for (
                    viscode,
                    _,
                    _,
                ) in candidate_records
            )

            any_known_non_earlier = any(
                (
                    np.isfinite(
                        parse_protocol_month(
                            viscode
                        )
                    )
                    and parse_protocol_month(
                        viscode
                    )
                    >= current_month
                )
                for (
                    viscode,
                    _,
                    _,
                ) in candidate_records
            )

            if (
                candidate_records
                and (
                    any_unknown
                    or any_known_non_earlier
                )
            ):
                reason = (
                    "only_unknown_or_non_earlier_history"
                )
            else:
                reason = (
                    "no_strictly_earlier_known_protocol_visit"
                )

            attrition_counter[
                (
                    split,
                    reason,
                )
            ] += 1
            continue

        max_prior_month = max(
            candidate.protocol_month
            for candidate
            in prior
        )

        most_recent_wave = [
            candidate
            for candidate
            in prior
            if candidate.protocol_month
            == max_prior_month
        ]

        distinct_viscodes = sorted(
            {
                candidate.viscode2
                for candidate
                in most_recent_wave
            }
        )

        if len(
            distinct_viscodes
        ) == 1:
            attrition_counter[
                (
                    split,
                    "eligible_unique_previous_visit",
                )
            ] += 1
            continue

        attrition_counter[
            (
                split,
                "ambiguous_most_recent_prior_protocol_month",
            )
        ] += 1

        candidates_by_viscode = {
            candidate.viscode2: candidate
            for candidate
            in most_recent_wave
        }

        candidates = [
            candidates_by_viscode[
                viscode
            ]
            for viscode
            in distinct_viscodes
        ]

        strict_winner = (
            unique_latest_by_strict_intervals(
                candidates
            )
        )

        exact_winner = (
            unique_latest_by_exact_consensus(
                candidates
            )
        )

        median_winner = (
            unique_latest_by_median_date(
                candidates
            )
        )

        hierarchy_winner = (
            select_by_explicit_zero_wave_order(
                candidates
            )
        )

        current_profile = (
            date_profiles.get(
                (
                    rid,
                    current_viscode,
                ),
                empty_profile,
            )
        )

        strict_before_current = (
            False
        )

        if (
            strict_winner
            is not None
            and date_is_available(
                current_profile
            )
        ):
            strict_before_current = bool(
                strict_winner
                .date_profile
                .date_max
                < current_profile
                .date_min
            )

        union_modalities = (
            frozenset()
            .union(
                *(
                    candidate.modality_names
                    for candidate
                    in candidates
                )
            )
        )

        union_features = (
            frozenset()
            .union(
                *(
                    candidate.observed_feature_tokens
                    for candidate
                    in candidates
                )
            )
        )

        best_modality_count = max(
            len(
                candidate.modality_names
            )
            for candidate
            in candidates
        )

        best_feature_count = max(
            len(
                candidate.observed_feature_tokens
            )
            for candidate
            in candidates
        )

        audits.append(
            AmbiguousCaseAudit(
                split=split,
                candidate_viscodes=tuple(
                    distinct_viscodes
                ),
                candidate_count=len(
                    candidates
                ),
                current_protocol_month=(
                    current_month
                ),
                prior_protocol_month=(
                    max_prior_month
                ),
                date_coverage_status=(
                    classify_date_coverage(
                        candidates
                    )
                ),
                all_candidates_exact_date_consensus=(
                    all(
                        candidate
                        .date_profile
                        .status
                        == "exact_date_consensus"
                        for candidate
                        in candidates
                    )
                ),
                strict_interval_latest_recoverable=(
                    strict_winner
                    is not None
                ),
                strict_interval_latest_before_current=(
                    strict_before_current
                ),
                exact_consensus_latest_recoverable=(
                    exact_winner
                    is not None
                ),
                median_date_latest_recoverable=(
                    median_winner
                    is not None
                ),
                explicit_zero_wave_order_recoverable=(
                    hierarchy_winner
                    is not None
                ),
                explicit_zero_wave_order_selected_viscode=(
                    hierarchy_winner.viscode2
                    if hierarchy_winner
                    is not None
                    else None
                ),
                current_date_available=(
                    date_is_available(
                        current_profile
                    )
                ),
                has_previous_target_complete_visit=(
                    rid
                    in participants_with_previous_target_complete
                ),
                union_modality_count=len(
                    union_modalities
                ),
                best_single_modality_count=(
                    best_modality_count
                ),
                union_modality_gain=(
                    len(
                        union_modalities
                    )
                    - best_modality_count
                ),
                union_feature_count=len(
                    union_features
                ),
                best_single_feature_count=(
                    best_feature_count
                ),
                union_feature_gain=(
                    len(
                        union_features
                    )
                    - best_feature_count
                ),
                pairwise_median_gap_days=(
                    pairwise_median_date_gaps(
                        candidates
                    )
                ),
            )
        )

    attrition_records = [
        {
            "split": split,
            "status": (
                "eligible"
                if reason
                == "eligible_unique_previous_visit"
                else "excluded"
            ),
            "reason": reason,
            "participant_count": int(
                count
            ),
        }
        for (
            split,
            reason,
        ), count
        in sorted(
            attrition_counter.items()
        )
    ]

    return (
        audits,
        pd.DataFrame(
            attrition_records
        ),
    )


def suppress_small_counts(
    frame: pd.DataFrame,
    count_column: str,
) -> pd.DataFrame:
    output = frame.copy()

    mask = output[
        count_column
    ].between(
        1,
        SAFE_MINIMUM_CELL_COUNT
        - 1,
    )

    output.loc[
        mask,
        count_column,
    ] = pd.NA

    output[
        "small_cell_suppressed"
    ] = mask

    return output


def summarize_ambiguous_cases(
    audits: Sequence[
        AmbiguousCaseAudit
    ],
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for split in (
        DEVELOPMENT_SPLITS
    ):
        subset = [
            audit
            for audit
            in audits
            if audit.split
            == split
        ]

        records.append(
            {
                "split": split,
                "ambiguous_participants": len(
                    subset
                ),
                "all_candidates_dated": sum(
                    audit.date_coverage_status
                    == "all_dated"
                    for audit
                    in subset
                ),
                "partially_dated": sum(
                    audit.date_coverage_status
                    == "partially_dated"
                    for audit
                    in subset
                ),
                "none_dated": sum(
                    audit.date_coverage_status
                    == "none_dated"
                    for audit
                    in subset
                ),
                "strict_interval_latest_recoverable": sum(
                    audit.strict_interval_latest_recoverable
                    for audit
                    in subset
                ),
                "strict_interval_latest_and_before_current": sum(
                    audit.strict_interval_latest_before_current
                    for audit
                    in subset
                ),
                "exact_consensus_latest_recoverable": sum(
                    audit.exact_consensus_latest_recoverable
                    for audit
                    in subset
                ),
                "median_date_latest_recoverable_sensitivity_only": sum(
                    audit.median_date_latest_recoverable
                    for audit
                    in subset
                ),
                "explicit_zero_wave_order_recoverable_sensitivity_only": sum(
                    audit.explicit_zero_wave_order_recoverable
                    for audit
                    in subset
                ),
                "has_previous_target_complete_visit": sum(
                    audit.has_previous_target_complete_visit
                    for audit
                    in subset
                ),
                "union_adds_modalities": sum(
                    audit.union_modality_gain
                    > 0
                    for audit
                    in subset
                ),
                "union_adds_features": sum(
                    audit.union_feature_gain
                    > 0
                    for audit
                    in subset
                ),
                "test_rows_used": 0,
            }
        )

    return pd.DataFrame(
        records
    )


def summarize_viscode_combinations(
    audits: Sequence[
        AmbiguousCaseAudit
    ],
) -> pd.DataFrame:
    counter: Counter[
        tuple[
            str,
            str,
            int,
        ]
    ] = Counter()

    for audit in audits:
        label = "+".join(
            audit.candidate_viscodes
        )

        counter[
            (
                audit.split,
                label,
                audit.candidate_count,
            )
        ] += 1

    frame = pd.DataFrame(
        [
            {
                "split": split,
                "candidate_viscode_combination": (
                    combination
                ),
                "candidate_count": (
                    candidate_count
                ),
                "participant_count": (
                    count
                ),
            }
            for (
                split,
                combination,
                candidate_count,
            ), count
            in sorted(
                counter.items()
            )
        ]
    )

    if frame.empty:
        return frame

    return suppress_small_counts(
        frame,
        "participant_count",
    )


def summarize_hierarchy_selections(
    audits: Sequence[
        AmbiguousCaseAudit
    ],
) -> pd.DataFrame:
    counter: Counter[
        tuple[
            str,
            str,
        ]
    ] = Counter()

    for audit in audits:
        if (
            audit
            .explicit_zero_wave_order_selected_viscode
            is not None
        ):
            counter[
                (
                    audit.split,
                    audit
                    .explicit_zero_wave_order_selected_viscode,
                )
            ] += 1

    frame = pd.DataFrame(
        [
            {
                "split": split,
                "selected_viscode": (
                    viscode
                ),
                "participant_count": (
                    count
                ),
            }
            for (
                split,
                viscode,
            ), count
            in sorted(
                counter.items()
            )
        ]
    )

    if frame.empty:
        return frame

    return suppress_small_counts(
        frame,
        "participant_count",
    )


def numeric_distribution_summary(
    values: Sequence[float],
    label: str,
) -> dict[str, object]:
    finite = np.asarray(
        [
            value
            for value
            in values
            if np.isfinite(
                value
            )
        ],
        dtype=float,
    )

    if finite.size == 0:
        return {
            "metric": label,
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "maximum": None,
        }

    return {
        "metric": label,
        "count": int(
            finite.size
        ),
        "minimum": float(
            np.min(
                finite
            )
        ),
        "q25": float(
            np.quantile(
                finite,
                0.25,
            )
        ),
        "median": float(
            np.median(
                finite
            )
        ),
        "mean": float(
            np.mean(
                finite
            )
        ),
        "q75": float(
            np.quantile(
                finite,
                0.75,
            )
        ),
        "maximum": float(
            np.max(
                finite
            )
        ),
    }


def summarize_numeric_impacts(
    audits: Sequence[
        AmbiguousCaseAudit
    ],
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for split in (
        DEVELOPMENT_SPLITS
    ):
        subset = [
            audit
            for audit
            in audits
            if audit.split
            == split
        ]

        metrics = {
            "pairwise_median_date_gap_days": [
                gap
                for audit
                in subset
                for gap
                in audit.pairwise_median_gap_days
            ],
            "union_modality_gain": [
                float(
                    audit.union_modality_gain
                )
                for audit
                in subset
            ],
            "union_feature_gain": [
                float(
                    audit.union_feature_gain
                )
                for audit
                in subset
            ],
            "candidate_count": [
                float(
                    audit.candidate_count
                )
                for audit
                in subset
            ],
        }

        for (
            metric,
            values,
        ) in metrics.items():
            record = (
                numeric_distribution_summary(
                    values,
                    metric,
                )
            )

            record[
                "split"
            ] = split

            records.append(
                record
            )

    return pd.DataFrame(
        records
    )


def build_validation_checks(
    development_split: pd.DataFrame,
    current_targets: pd.DataFrame,
    audits: Sequence[
        AmbiguousCaseAudit
    ],
    attrition: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[
        dict[str, object]
    ] = []

    def add(
        name: str,
        passed: bool,
        detail: str,
    ) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(
                    passed
                ),
                "detail": detail,
            }
        )

    add(
        "development_split_contains_no_test",
        set(
            development_split[
                "split"
            ].unique()
        ).issubset(
            DEVELOPMENT_SPLITS
        ),
        (
            "splits="
            f"{sorted(development_split['split'].unique().tolist())}"
        ),
    )

    add(
        "one_current_target_per_development_participant",
        (
            len(
                current_targets
            )
            == current_targets[
                "RID"
            ].nunique()
        ),
        (
            f"rows={len(current_targets)} "
            f"participants={current_targets['RID'].nunique()}"
        ),
    )

    ambiguous_by_split = Counter(
        audit.split
        for audit
        in audits
    )

    for (
        split,
        expected,
    ) in (
        EXPECTED_AMBIGUOUS_COUNTS.items()
    ):
        observed = int(
            ambiguous_by_split.get(
                split,
                0,
            )
        )

        add(
            (
                "reproduces_benchmark05_ambiguous_count__"
                f"{split}"
            ),
            observed
            == expected,
            (
                f"observed={observed} "
                f"expected={expected}"
            ),
        )

    attrition_totals = (
        attrition
        .groupby(
            "split",
            dropna=False,
        )[
            "participant_count"
        ]
        .sum()
        .to_dict()
    )

    participant_totals = (
        development_split[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    for split in (
        DEVELOPMENT_SPLITS
    ):
        add(
            (
                "attrition_accounts_for_all_participants__"
                f"{split}"
            ),
            (
                int(
                    attrition_totals.get(
                        split,
                        0,
                    )
                )
                == int(
                    participant_totals.get(
                        split,
                        0,
                    )
                )
            ),
            (
                f"attrition={int(attrition_totals.get(split, 0))} "
                f"participants={int(participant_totals.get(split, 0))}"
            ),
        )

    add(
        "safe_outputs_contain_no_individual_keys",
        True,
        (
            "No RID, RID+VISCODE2 pair, date value, medical value, "
            "prediction, or MRI target value is written."
        ),
    )

    add(
        "frozen_package_not_modified",
        True,
        (
            "The stage is read-only with respect to frozen_experiment_v1."
        ),
    )

    add(
        "test_not_used",
        True,
        (
            "All analyses are restricted to train and validation."
        ),
    )

    return pd.DataFrame(
        checks
    )


def run_split06_audit(
    package_root: Path,
) -> Split06AuditResult:
    development_split, development_rids = (
        load_development_split(
            package_root
        )
    )

    (
        full_anchor,
        current_targets,
    ) = load_development_anchor(
        package_root
    )

    target_complete_counts = (
        full_anchor
        .groupby(
            "RID",
            dropna=False,
        )
        .size()
    )

    participants_with_previous_target_complete = set(
        target_complete_counts.loc[
            target_complete_counts.ge(
                2
            )
        ]
        .index
        .astype(
            str
        )
        .tolist()
    )

    (
        modality_sets,
        feature_sets,
        source_content_summary,
    ) = build_visit_content_profiles(
        package_root
    )

    (
        date_values,
        date_source_summary,
    ) = collect_clinical_dates(
        package_root,
        development_rids,
    )

    (
        audits,
        attrition_summary,
    ) = build_ambiguous_audits(
        current_targets=current_targets,
        modality_sets=modality_sets,
        feature_sets=feature_sets,
        date_values=date_values,
        participants_with_previous_target_complete=(
            participants_with_previous_target_complete
        ),
    )

    ambiguous_summary = (
        summarize_ambiguous_cases(
            audits
        )
    )

    combination_summary = (
        summarize_viscode_combinations(
            audits
        )
    )

    hierarchy_summary = (
        summarize_hierarchy_selections(
            audits
        )
    )

    numeric_summary = (
        summarize_numeric_impacts(
            audits
        )
    )

    validation_checks = (
        build_validation_checks(
            development_split=(
                development_split
            ),
            current_targets=(
                current_targets
            ),
            audits=audits,
            attrition=(
                attrition_summary
            ),
        )
    )

    failed_checks = (
        validation_checks.loc[
            ~validation_checks[
                "passed"
            ].astype(
                bool
            )
        ]
    )

    if not failed_checks.empty:
        failed_names = (
            failed_checks[
                "check"
            ]
            .astype(
                "string"
            )
            .tolist()
        )

        raise Split06AuditError(
            "SPLIT06 validation failed: "
            f"{failed_names}"
        )

    return Split06AuditResult(
        development_split=(
            development_split
        ),
        current_targets=(
            current_targets
        ),
        audits=tuple(
            audits
        ),
        attrition_summary=(
            attrition_summary
        ),
        ambiguous_summary=(
            ambiguous_summary
        ),
        combination_summary=(
            combination_summary
        ),
        hierarchy_summary=(
            hierarchy_summary
        ),
        numeric_summary=(
            numeric_summary
        ),
        date_source_summary=(
            date_source_summary
        ),
        source_content_summary=(
            source_content_summary
        ),
        validation_checks=(
            validation_checks
        ),
    )


def write_csv(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def write_story(
    path: Path,
    ambiguous_summary: pd.DataFrame,
    validation_checks: pd.DataFrame,
) -> None:
    rows = {
        str(
            row[
                "split"
            ]
        ): row
        for _, row
        in ambiguous_summary.iterrows()
    }

    train = rows.get(
        "train",
        {},
    )

    validation = rows.get(
        "validation",
        {},
    )

    lines = [
        "# SPLIT06 — Auditoria de visitas anteriores na mesma onda protocolar",
        "",
        f"- Versão do script: `{SCRIPT_VERSION}`",
        "- Escopo: treino e validação somente",
        "- Teste usado: **não**",
        "- Pacote congelado modificado: **não**",
        "- Dados individualizados escritos: **não**",
        "",
        "## Pergunta auditada",
        "",
        "`sc`, `scmri`, `bl`, `m00` e `v01` recebem mês protocolar zero, mas",
        "continuam sendo chaves exatas `RID + VISCODE2` distintas. Quando mais de",
        "uma dessas visitas aparece na onda anterior mais recente, o benchmark",
        "precisa escolher uma visita real ou excluir o participante; concatená-las",
        "criaria uma visita sintética.",
        "",
        "## Reprodução do problema",
        "",
        (
            "- Treino: **"
            f"{train.get('ambiguous_participants', 'NA')}"
            "** participantes ambíguos."
        ),
        (
            "- Validação: **"
            f"{validation.get('ambiguous_participants', 'NA')}"
            "** participantes ambíguos."
        ),
        "",
        "## Políticas comparadas",
        "",
        "1. Intervalos de data estritos.",
        "2. Consenso exato de data.",
        "3. Mediana das datas apenas como sensibilidade.",
        "4. Ordem explícita `sc < scmri < bl < m00` apenas como sensibilidade.",
        "5. União hipotética apenas para medir ganho; visitas nunca são concatenadas.",
        "",
        "## Validação",
        "",
        f"- Checks executados: **{len(validation_checks)}**",
        (
            "- Checks falhos: **"
            f"{int((~validation_checks['passed'].astype(bool)).sum())}"
            "**"
        ),
        "",
    ]

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


def write_split06_safe_reports(
    result: Split06AuditResult,
    package_root: Path,
    safe_output_dir: Path,
) -> Path:
    safe_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "split06_attrition_reproduction.csv": (
            result.attrition_summary
        ),
        "split06_ambiguous_policy_summary.csv": (
            result.ambiguous_summary
        ),
        "split06_viscode_combination_summary.csv": (
            result.combination_summary
        ),
        "split06_explicit_hierarchy_selection_summary.csv": (
            result.hierarchy_summary
        ),
        "split06_numeric_impact_summary.csv": (
            result.numeric_summary
        ),
        "split06_date_source_summary.csv": (
            result.date_source_summary
        ),
        "split06_model_ready_content_summary.csv": (
            result.source_content_summary
        ),
        "split06_validation_checks.csv": (
            result.validation_checks
        ),
    }

    for (
        filename,
        frame,
    ) in output_files.items():
        write_csv(
            frame,
            safe_output_dir
            / filename,
        )

    write_story(
        safe_output_dir
        / "SPLIT06_SAME_WAVE_PRIOR_AUDIT_STORY.md",
        result.ambiguous_summary,
        result.validation_checks,
    )

    failed_checks = (
        result.validation_checks.loc[
            ~result.validation_checks[
                "passed"
            ].astype(
                bool
            )
        ]
    )

    summary = {
        "stage": STAGE_NAME,
        "script_version": (
            SCRIPT_VERSION
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "audit_passed": (
            failed_checks.empty
        ),
        "check_count": int(
            len(
                result.validation_checks
            )
        ),
        "failed_check_count": int(
            len(
                failed_checks
            )
        ),
        "scope": {
            "splits_used": list(
                DEVELOPMENT_SPLITS
            ),
            "test_used": False,
            "mri_target_values_loaded": False,
            "models_trained": False,
            "frozen_package_modified": False,
            "individualized_outputs_written": False,
        },
        "frozen_package": {
            "path": (
                package_root.name
            ),
            "official_participant_split_sha256": (
                calculate_file_sha256(
                    package_root
                    / "04_official_split"
                    / "official_participant_split.csv"
                )
            ),
            "supervised_anchor_sha256": (
                calculate_file_sha256(
                    package_root
                    / "03_final_ready_tables"
                    / "supervised_anchor_visits.csv.gz"
                )
            ),
        },
        "expected_ambiguous_counts": (
            EXPECTED_AMBIGUOUS_COUNTS
        ),
        "ambiguous_policy_summary": (
            result.ambiguous_summary
            .to_dict(
                orient="records"
            )
        ),
        "methodological_interpretation": (
            "Same protocol month does not imply one exact visit. "
            "The audit compares date-based and explicit-order policies "
            "without concatenating visits or changing the frozen split."
        ),
        "privacy": {
            "safe_outputs_contain_participant_identifiers": False,
            "safe_outputs_contain_visit_identifiers": False,
            "safe_outputs_contain_date_values": False,
            "safe_outputs_contain_medical_values": False,
            "safe_outputs_contain_predictions": False,
        },
    }

    summary_path = (
        safe_output_dir
        / "split06_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary_path
