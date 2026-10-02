from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from adni_benchmark.non_mri_resolution import calculate_text_sha256
from adni_benchmark.split_balance_profile import (
    SPLIT_MISSING_CATEGORY,
    SPLIT_RARE_CATEGORY,
    calculate_participant_profile_sha256,
)


class ParticipantSplitSearchError(RuntimeError):
    pass


SCRIPT_VERSION: Final[str] = "1.0.1"

SPLIT_NAMES: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test",
)

SPLIT_TARGET_PROPORTIONS: Final[dict[str, float]] = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

SPLIT_BALANCE_WEIGHTS: Final[dict[str, float]] = {
    "visit_proportion": 1.0,
    "diagnosis_first": 1.0,
    "n_visits": 1.0,
    "longitudinal_class": 1.0,
    "modality_pattern_first": 1.0,
    "followup_months": 1.0,
    "modality_count_mean": 1.0,
    "sex": 1.0,
    "apoe_availability": 1.0,
}

SPLIT_MASTER_SEED: Final[int] = 42_000_000
SPLIT_N_CANDIDATES: Final[int] = 100_000
SPLIT_CHECKPOINT_EVERY: Final[int] = 1_000
SPLIT_REQUIRED_DIAGNOSIS_MIN_COUNT: Final[int] = 10
SPLIT_VISIT_PROPORTION_TOLERANCE: Final[float] = 0.03
SPLIT_MIN_2PLUS_PER_SPLIT: Final[int] = 1
SPLIT_MIN_3PLUS_PER_SPLIT: Final[int] = 1


@dataclass(frozen=True)
class SplitSizes:
    n_train: int
    n_validation: int
    n_test: int

    @property
    def boundaries(self) -> tuple[int, int]:
        return (
            self.n_train,
            self.n_train + self.n_validation,
        )

    def to_manifest_record(self) -> dict[str, int]:
        return {
            "train": self.n_train,
            "validation": self.n_validation,
            "test": self.n_test,
        }


@dataclass(frozen=True)
class CategoricalBalanceCriterion:
    name: str
    codes: np.ndarray
    categories: tuple[str, ...]
    global_probabilities: np.ndarray


@dataclass(frozen=True)
class NumericBalanceCriterion:
    name: str
    standardized_values: np.ndarray
    global_finite_values: np.ndarray
    original_mean: float
    original_std: float


@dataclass(frozen=True)
class SplitSearchContext:
    participant_ids: np.ndarray
    visit_counts: np.ndarray
    total_visits: int
    split_sizes: SplitSizes
    categorical: dict[str, CategoricalBalanceCriterion]
    numeric: dict[str, NumericBalanceCriterion]
    required_diagnosis_codes: frozenset[int]
    longitudinal_2plus: np.ndarray
    longitudinal_3plus: np.ndarray


@dataclass(frozen=True)
class ParticipantSplitSearchSummary:
    n_candidates: int
    valid_candidates: int
    participant_profile_sha256: str
    split_sizes: SplitSizes
    active_criteria: tuple[str, ...]
    active_weight_sum: float
    selected_candidate_id: int
    selected_seed: int
    selected_score_final: float
    selected_score_max_component: float
    selected_score_max_component_name: str
    participant_counts: tuple[tuple[str, int], ...]
    visit_counts: tuple[tuple[str, int], ...]
    participant_overlaps: tuple[tuple[str, int], ...]
    split_mapping_sha256: str


def compute_split_sizes(
    n_participants: int,
) -> SplitSizes:
    if n_participants < 3:
        raise ParticipantSplitSearchError(
            "At least three participants are required for splitting."
        )

    if not np.isclose(
        sum(SPLIT_TARGET_PROPORTIONS.values()),
        1.0,
    ):
        raise ParticipantSplitSearchError(
            "Split target proportions do not sum to one."
        )

    n_train = int(
        round(
            n_participants
            * SPLIT_TARGET_PROPORTIONS["train"]
        )
    )

    n_validation = int(
        round(
            n_participants
            * SPLIT_TARGET_PROPORTIONS["validation"]
        )
    )

    n_test = (
        n_participants
        - n_train
        - n_validation
    )

    if min(
        n_train,
        n_validation,
        n_test,
    ) <= 0:
        raise ParticipantSplitSearchError(
            "At least one split received no participants."
        )

    return SplitSizes(
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
    )


def make_categorical_balance_criterion(
    profile: pd.DataFrame,
    column: str,
) -> CategoricalBalanceCriterion:
    series = (
        profile[column]
        .astype("string")
        .fillna(SPLIT_MISSING_CATEGORY)
    )

    categories = tuple(
        sorted(
            str(value)
            for value in series.unique().tolist()
        )
    )

    category_to_code = {
        category: position
        for position, category
        in enumerate(categories)
    }

    codes = (
        series
        .map(category_to_code)
        .to_numpy(dtype=np.int32)
    )

    counts = np.bincount(
        codes,
        minlength=len(categories),
    ).astype("float64")

    probabilities = counts / counts.sum()

    return CategoricalBalanceCriterion(
        name=column,
        codes=codes,
        categories=categories,
        global_probabilities=probabilities,
    )


def make_numeric_balance_criterion(
    profile: pd.DataFrame,
    column: str,
) -> NumericBalanceCriterion | None:
    values = pd.to_numeric(
        profile[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_mask = np.isfinite(values)

    if int(finite_mask.sum()) < 2:
        return None

    mean = float(
        np.nanmean(values)
    )

    standard_deviation = float(
        np.nanstd(
            values,
            ddof=0,
        )
    )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation <= 0
    ):
        return None

    standardized_values = (
        values - mean
    ) / standard_deviation

    return NumericBalanceCriterion(
        name=column,
        standardized_values=standardized_values,
        global_finite_values=standardized_values[
            np.isfinite(standardized_values)
        ],
        original_mean=mean,
        original_std=standard_deviation,
    )


def build_split_search_context(
    profile: pd.DataFrame,
) -> SplitSearchContext:
    required_columns = {
        "RID",
        "n_visits",
        "diagnosis_first",
        "sex",
        "apoe_availability",
        "modality_pattern_first",
        "longitudinal_class",
        "followup_months",
        "modality_count_mean",
    }

    missing_columns = sorted(
        required_columns
        - set(profile.columns)
    )

    if missing_columns:
        raise ParticipantSplitSearchError(
            "Split balance profile is missing required columns: "
            f"{missing_columns}"
        )

    categorical_names = (
        "diagnosis_first",
        "sex",
        "apoe_availability",
        "modality_pattern_first",
        "longitudinal_class",
    )

    categorical = {
        name: make_categorical_balance_criterion(
            profile=profile,
            column=name,
        )
        for name in categorical_names
    }

    numeric: dict[
        str,
        NumericBalanceCriterion,
    ] = {}

    for name in (
        "n_visits",
        "followup_months",
        "modality_count_mean",
    ):
        criterion = make_numeric_balance_criterion(
            profile=profile,
            column=name,
        )

        if criterion is not None:
            numeric[name] = criterion

    diagnosis = categorical[
        "diagnosis_first"
    ]

    diagnosis_counts = np.bincount(
        diagnosis.codes,
        minlength=len(diagnosis.categories),
    )

    required_diagnosis_codes = frozenset(
        code
        for code, count
        in enumerate(diagnosis_counts)
        if (
            int(count)
            >= SPLIT_REQUIRED_DIAGNOSIS_MIN_COUNT
            and diagnosis.categories[code]
            not in {
                SPLIT_MISSING_CATEGORY,
                SPLIT_RARE_CATEGORY,
            }
        )
    )

    visit_counts = pd.to_numeric(
        profile["n_visits"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if (visit_counts <= 0).any():
        raise ParticipantSplitSearchError(
            "Split balance profile contains a participant with no visits."
        )

    return SplitSearchContext(
        participant_ids=(
            profile["RID"]
            .astype("string")
            .to_numpy()
        ),
        visit_counts=visit_counts,
        total_visits=int(
            visit_counts.sum()
        ),
        split_sizes=compute_split_sizes(
            len(profile)
        ),
        categorical=categorical,
        numeric=numeric,
        required_diagnosis_codes=required_diagnosis_codes,
        longitudinal_2plus=(
            visit_counts >= 2
        ),
        longitudinal_3plus=(
            visit_counts >= 3
        ),
    )


def generate_candidate_indices(
    n_participants: int,
    split_sizes: SplitSizes,
    seed: int,
) -> dict[str, np.ndarray]:
    random_generator = np.random.default_rng(
        seed
    )

    order = random_generator.permutation(
        n_participants
    )

    (
        train_boundary,
        validation_boundary,
    ) = split_sizes.boundaries

    return {
        "train": order[
            :train_boundary
        ],
        "validation": order[
            train_boundary:validation_boundary
        ],
        "test": order[
            validation_boundary:
        ],
    }


def aggregate_split_distances(
    distances: dict[str, float],
) -> tuple[float, float, float]:
    values = np.asarray(
        [
            distances[split_name]
            for split_name in SPLIT_NAMES
        ],
        dtype=float,
    )

    if not np.isfinite(values).all():
        return (
            float("inf"),
            float("inf"),
            float("inf"),
        )

    mean_value = float(
        values.mean()
    )

    maximum_value = float(
        values.max()
    )

    score = (
        0.5 * mean_value
        + 0.5 * maximum_value
    )

    return (
        mean_value,
        maximum_value,
        score,
    )


def evaluate_split_candidate(
    context: SplitSearchContext,
    candidate_id: int,
    seed: int,
) -> dict[str, object]:
    indices = generate_candidate_indices(
        n_participants=len(
            context.participant_ids
        ),
        split_sizes=context.split_sizes,
        seed=seed,
    )

    record: dict[str, object] = {
        "candidate_id": candidate_id,
        "seed": seed,
    }

    rejection_reasons: list[str] = []
    criterion_scores: dict[str, float] = {}
    visit_proportion_distances: dict[str, float] = {}

    for split_name in SPLIT_NAMES:
        split_index = indices[
            split_name
        ]

        n_visits = int(
            context.visit_counts[
                split_index
            ].sum()
        )

        visit_proportion = (
            n_visits
            / context.total_visits
        )

        visit_error = abs(
            visit_proportion
            - SPLIT_TARGET_PROPORTIONS[
                split_name
            ]
        )

        n_2plus = int(
            context.longitudinal_2plus[
                split_index
            ].sum()
        )

        n_3plus = int(
            context.longitudinal_3plus[
                split_index
            ].sum()
        )

        record[
            f"n_participants_{split_name}"
        ] = int(
            len(split_index)
        )

        record[
            f"n_visits_{split_name}"
        ] = n_visits

        record[
            f"visit_proportion_{split_name}"
        ] = visit_proportion

        record[
            f"visit_proportion_error_{split_name}"
        ] = visit_error

        record[
            f"n_participants_2plus_{split_name}"
        ] = n_2plus

        record[
            f"n_participants_3plus_{split_name}"
        ] = n_3plus

        visit_proportion_distances[
            split_name
        ] = visit_error

        if (
            visit_error
            > SPLIT_VISIT_PROPORTION_TOLERANCE
        ):
            rejection_reasons.append(
                "visit_proportion_outside_tolerance:"
                f"{split_name}"
            )

        if (
            n_2plus
            < SPLIT_MIN_2PLUS_PER_SPLIT
        ):
            rejection_reasons.append(
                f"insufficient_2plus:{split_name}"
            )

        if (
            n_3plus
            < SPLIT_MIN_3PLUS_PER_SPLIT
        ):
            rejection_reasons.append(
                f"insufficient_3plus:{split_name}"
            )

    (
        visit_mean,
        visit_maximum,
        visit_score,
    ) = aggregate_split_distances(
        visit_proportion_distances
    )

    record[
        "mean_distance_visit_proportion"
    ] = visit_mean
    record[
        "max_distance_visit_proportion"
    ] = visit_maximum
    record[
        "score_visit_proportion"
    ] = visit_score

    criterion_scores[
        "visit_proportion"
    ] = visit_score

    for criterion_name, criterion in (
        context.categorical.items()
    ):
        split_distances: dict[str, float] = {}

        for split_name in SPLIT_NAMES:
            split_codes = criterion.codes[
                indices[split_name]
            ]

            counts = np.bincount(
                split_codes,
                minlength=len(
                    criterion.categories
                ),
            ).astype("float64")

            probabilities = (
                counts / counts.sum()
            )

            distance = float(
                jensenshannon(
                    criterion.global_probabilities,
                    probabilities,
                    base=2.0,
                )
            )

            split_distances[
                split_name
            ] = distance

            record[
                f"distance_{criterion_name}_{split_name}"
            ] = distance

        (
            mean_value,
            maximum_value,
            criterion_score,
        ) = aggregate_split_distances(
            split_distances
        )

        record[
            f"mean_distance_{criterion_name}"
        ] = mean_value
        record[
            f"max_distance_{criterion_name}"
        ] = maximum_value
        record[
            f"score_{criterion_name}"
        ] = criterion_score

        criterion_scores[
            criterion_name
        ] = criterion_score

    diagnosis = context.categorical[
        "diagnosis_first"
    ]

    for split_name in SPLIT_NAMES:
        present_codes = set(
            diagnosis.codes[
                indices[split_name]
            ].tolist()
        )

        missing_required_codes = (
            context.required_diagnosis_codes
            - present_codes
        )

        if missing_required_codes:
            rejection_reasons.append(
                "missing_required_diagnosis_category:"
                f"{split_name}"
            )

    for criterion_name, criterion in (
        context.numeric.items()
    ):
        split_distances: dict[str, float] = {}

        for split_name in SPLIT_NAMES:
            split_values = criterion.standardized_values[
                indices[split_name]
            ]

            split_values = split_values[
                np.isfinite(split_values)
            ]

            if len(split_values) == 0:
                distance = float("inf")
            else:
                distance = float(
                    wasserstein_distance(
                        criterion.global_finite_values,
                        split_values,
                    )
                )

            split_distances[
                split_name
            ] = distance

            record[
                f"distance_{criterion_name}_{split_name}"
            ] = distance

        (
            mean_value,
            maximum_value,
            criterion_score,
        ) = aggregate_split_distances(
            split_distances
        )

        record[
            f"mean_distance_{criterion_name}"
        ] = mean_value
        record[
            f"max_distance_{criterion_name}"
        ] = maximum_value
        record[
            f"score_{criterion_name}"
        ] = criterion_score

        criterion_scores[
            criterion_name
        ] = criterion_score

    weighted_sum = 0.0
    total_weight = 0.0
    active_scores: list[float] = []
    active_names: list[str] = []

    for criterion_name, criterion_score in (
        criterion_scores.items()
    ):
        weight = float(
            SPLIT_BALANCE_WEIGHTS.get(
                criterion_name,
                0.0,
            )
        )

        record[
            f"weight_{criterion_name}"
        ] = weight

        if (
            weight <= 0
            or not np.isfinite(
                criterion_score
            )
        ):
            continue

        weighted_sum += (
            weight * criterion_score
        )

        total_weight += weight

        active_scores.append(
            criterion_score
        )

        active_names.append(
            criterion_name
        )

    score_final = (
        weighted_sum / total_weight
        if total_weight > 0
        else float("inf")
    )

    record[
        "score_mean_components"
    ] = (
        float(
            np.mean(active_scores)
        )
        if active_scores
        else float("inf")
    )

    record[
        "score_max_component"
    ] = (
        float(
            np.max(active_scores)
        )
        if active_scores
        else float("inf")
    )

    record[
        "score_max_component_name"
    ] = (
        active_names[
            int(
                np.argmax(
                    active_scores
                )
            )
        ]
        if active_scores
        else ""
    )

    record[
        "score_final"
    ] = float(
        score_final
    )

    record[
        "is_valid"
    ] = bool(
        not rejection_reasons
        and np.isfinite(
            score_final
        )
    )

    record[
        "rejection_reasons"
    ] = "|".join(
        sorted(
            set(
                rejection_reasons
            )
        )
    )

    return record


def append_candidate_records(
    output_path: Path,
    records: list[dict[str, object]],
) -> None:
    if not records:
        return

    frame = pd.DataFrame(
        records
    )

    write_header = (
        not output_path.exists()
    )

    mode = (
        "wt"
        if write_header
        else "at"
    )

    with gzip.open(
        output_path,
        mode=mode,
        encoding="utf-8",
        newline="",
    ) as handle:
        frame.to_csv(
            handle,
            index=False,
            header=write_header,
        )


def split_checkpoint_metadata(
    profile_sha256: str,
) -> dict[str, object]:
    return {
        "script_version": SCRIPT_VERSION,
        "participant_profile_sha256": profile_sha256,
        "n_candidates": SPLIT_N_CANDIDATES,
        "master_seed": SPLIT_MASTER_SEED,
        "checkpoint_every": SPLIT_CHECKPOINT_EVERY,
        "weights": SPLIT_BALANCE_WEIGHTS,
        "target_proportions": SPLIT_TARGET_PROPORTIONS,
    }


def run_participant_split_search(
    local_output_dir: Path,
    balance_profile: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    ParticipantSplitSearchSummary,
    tuple[Path, ...],
]:
    # Every individualized search artifact is LOCAL_ONLY.
    local_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    context = build_split_search_context(
        balance_profile
    )

    profile_sha256 = (
        calculate_participant_profile_sha256(
            balance_profile
        )
    )

    profile_path = (
        local_output_dir
        / "LOCAL_ONLY_participant_balance_profile.csv.gz"
    )

    balance_profile.to_csv(
        profile_path,
        index=False,
        compression="gzip",
    )

    checkpoint_path = (
        local_output_dir
        / ".LOCAL_ONLY_split_candidates_checkpoint.csv.gz"
    )

    checkpoint_metadata_path = (
        local_output_dir
        / ".LOCAL_ONLY_split_checkpoint_metadata.json"
    )

    expected_checkpoint_metadata = (
        split_checkpoint_metadata(
            profile_sha256
        )
    )

    start_candidate_id = 1
    best_score = float("inf")

    if checkpoint_path.exists():
        if not checkpoint_metadata_path.exists():
            raise ParticipantSplitSearchError(
                "Split checkpoint exists without its metadata file. "
                "Delete the local checkpoint before restarting."
            )

        observed_checkpoint_metadata = json.loads(
            checkpoint_metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            observed_checkpoint_metadata
            != expected_checkpoint_metadata
        ):
            raise ParticipantSplitSearchError(
                "Existing split checkpoint was created from a different "
                "profile or protocol. Delete the LOCAL_ONLY checkpoint "
                "files before restarting."
            )

        existing = pd.read_csv(
            checkpoint_path,
            low_memory=False,
        )

        if not existing.empty:
            completed_candidate_id = int(
                existing[
                    "candidate_id"
                ].max()
            )

            start_candidate_id = (
                completed_candidate_id + 1
            )

            valid_existing = existing.loc[
                existing[
                    "is_valid"
                ].astype(bool)
            ]

            if not valid_existing.empty:
                best_score = float(
                    valid_existing[
                        "score_final"
                    ].min()
                )

            logging.info(
                "Resuming split search at candidate %d",
                start_candidate_id,
            )

    else:
        checkpoint_metadata_path.write_text(
            json.dumps(
                expected_checkpoint_metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    candidate_batch: list[
        dict[str, object]
    ] = []

    for candidate_id in range(
        start_candidate_id,
        SPLIT_N_CANDIDATES + 1,
    ):
        seed = (
            SPLIT_MASTER_SEED
            + candidate_id
            - 1
        )

        record = evaluate_split_candidate(
            context=context,
            candidate_id=candidate_id,
            seed=seed,
        )

        candidate_batch.append(
            record
        )

        if bool(
            record[
                "is_valid"
            ]
        ):
            best_score = min(
                best_score,
                float(
                    record[
                        "score_final"
                    ]
                ),
            )

        checkpoint_due = (
            candidate_id
            % SPLIT_CHECKPOINT_EVERY
            == 0
            or candidate_id
            == SPLIT_N_CANDIDATES
        )

        if checkpoint_due:
            append_candidate_records(
                output_path=checkpoint_path,
                records=candidate_batch,
            )

            candidate_batch.clear()

            logging.info(
                "Split search | %d/%d candidates | best=%s",
                candidate_id,
                SPLIT_N_CANDIDATES,
                (
                    f"{best_score:.8f}"
                    if np.isfinite(
                        best_score
                    )
                    else "NA"
                ),
            )

    ranking = pd.read_csv(
        checkpoint_path,
        low_memory=False,
    )

    if (
        len(ranking)
        != SPLIT_N_CANDIDATES
    ):
        raise ParticipantSplitSearchError(
            "Split candidate checkpoint does not contain the expected "
            f"{SPLIT_N_CANDIDATES} candidates."
        )

    if ranking[
        "candidate_id"
    ].duplicated().any():
        raise ParticipantSplitSearchError(
            "Split candidate ranking contains duplicated candidate IDs."
        )

    ranking[
        "rank"
    ] = np.nan

    valid_ranking = ranking.loc[
        ranking[
            "is_valid"
        ].astype(bool)
    ].copy()

    if valid_ranking.empty:
        raise ParticipantSplitSearchError(
            "No valid participant split candidate was found."
        )

    valid_ranking = valid_ranking.sort_values(
        by=[
            "score_final",
            "score_max_component",
            "seed",
        ],
        kind="stable",
    )

    valid_ranking[
        "rank"
    ] = np.arange(
        1,
        len(valid_ranking) + 1,
        dtype=np.int64,
    )

    ranking.loc[
        valid_ranking.index,
        "rank",
    ] = valid_ranking[
        "rank"
    ]

    selected = (
        valid_ranking.iloc[
            0
        ].copy()
    )

    ranking_path = (
        local_output_dir
        / "LOCAL_ONLY_split_candidates_ranking.csv.gz"
    )

    ranking.to_csv(
        ranking_path,
        index=False,
        compression="gzip",
    )

    selected_indices = generate_candidate_indices(
        n_participants=len(
            balance_profile
        ),
        split_sizes=context.split_sizes,
        seed=int(
            selected[
                "seed"
            ]
        ),
    )

    mapping_parts: list[
        pd.DataFrame
    ] = []

    for split_name in SPLIT_NAMES:
        split_index = selected_indices[
            split_name
        ]

        mapping_parts.append(
            pd.DataFrame(
                {
                    "RID": (
                        balance_profile.iloc[
                            split_index
                        ]["RID"]
                        .astype("string")
                        .to_numpy()
                    ),
                    "split": split_name,
                }
            )
        )

    mapping = (
        pd.concat(
            mapping_parts,
            ignore_index=True,
        )
        .sort_values(
            "RID",
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    if mapping[
        "RID"
    ].duplicated().any():
        raise ParticipantSplitSearchError(
            "Selected split mapping contains duplicated participants."
        )

    if (
        len(mapping)
        != len(balance_profile)
    ):
        raise ParticipantSplitSearchError(
            "Selected split mapping participant accounting failed."
        )

    mapping_path = (
        local_output_dir
        / "LOCAL_ONLY_selected_participant_split.csv"
    )

    mapping.to_csv(
        mapping_path,
        index=False,
    )

    selected_summary_path = (
        local_output_dir
        / "LOCAL_ONLY_selected_split_summary.csv"
    )

    pd.DataFrame(
        [
            selected.to_dict()
        ]
    ).to_csv(
        selected_summary_path,
        index=False,
    )

    profile_with_split = balance_profile.merge(
        mapping,
        on="RID",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if profile_with_split[
        "split"
    ].isna().any():
        raise ParticipantSplitSearchError(
            "Selected split failed to map all participants."
        )

    participant_counts = tuple(
        (
            split_name,
            int(
                profile_with_split[
                    "split"
                ].eq(
                    split_name
                ).sum()
            ),
        )
        for split_name in SPLIT_NAMES
    )

    visit_counts = tuple(
        (
            split_name,
            int(
                profile_with_split.loc[
                    profile_with_split[
                        "split"
                    ].eq(
                        split_name
                    ),
                    "n_visits",
                ].sum()
            ),
        )
        for split_name in SPLIT_NAMES
    )

    split_participant_sets = {
        split_name: set(
            mapping.loc[
                mapping[
                    "split"
                ].eq(
                    split_name
                ),
                "RID",
            ].tolist()
        )
        for split_name in SPLIT_NAMES
    }

    participant_overlaps = (
        (
            "train_validation",
            len(
                split_participant_sets["train"]
                & split_participant_sets[
                    "validation"
                ]
            ),
        ),
        (
            "train_test",
            len(
                split_participant_sets["train"]
                & split_participant_sets[
                    "test"
                ]
            ),
        ),
        (
            "validation_test",
            len(
                split_participant_sets[
                    "validation"
                ]
                & split_participant_sets[
                    "test"
                ]
            ),
        ),
    )

    if any(
        overlap_count != 0
        for _, overlap_count
        in participant_overlaps
    ):
        raise ParticipantSplitSearchError(
            "Participant overlap was detected between selected splits."
        )

    if (
        dict(
            participant_counts
        )
        != context.split_sizes
        .to_manifest_record()
    ):
        raise ParticipantSplitSearchError(
            "Selected split participant counts do not match the frozen "
            "70/15/15 allocation."
        )

    if (
        sum(
            count
            for _, count
            in visit_counts
        )
        != context.total_visits
    ):
        raise ParticipantSplitSearchError(
            "Selected split visit accounting failed."
        )

    split_mapping_sha256 = (
        calculate_text_sha256(
            "\n".join(
                f"{row.RID},{row.split}"
                for row
                in mapping.itertuples(
                    index=False
                )
            )
            + "\n"
        )
    )

    active_criteria = tuple(
        criterion_name
        for criterion_name, weight
        in SPLIT_BALANCE_WEIGHTS.items()
        if (
            weight > 0
            and (
                criterion_name
                == "visit_proportion"
                or criterion_name
                in context.categorical
                or criterion_name
                in context.numeric
            )
        )
    )

    active_weight_sum = float(
        sum(
            SPLIT_BALANCE_WEIGHTS[
                criterion_name
            ]
            for criterion_name
            in active_criteria
        )
    )

    summary = ParticipantSplitSearchSummary(
        n_candidates=SPLIT_N_CANDIDATES,
        valid_candidates=int(
            len(valid_ranking)
        ),
        participant_profile_sha256=(
            profile_sha256
        ),
        split_sizes=context.split_sizes,
        active_criteria=active_criteria,
        active_weight_sum=active_weight_sum,
        selected_candidate_id=int(
            selected[
                "candidate_id"
            ]
        ),
        selected_seed=int(
            selected[
                "seed"
            ]
        ),
        selected_score_final=float(
            selected[
                "score_final"
            ]
        ),
        selected_score_max_component=float(
            selected[
                "score_max_component"
            ]
        ),
        selected_score_max_component_name=str(
            selected[
                "score_max_component_name"
            ]
        ),
        participant_counts=participant_counts,
        visit_counts=visit_counts,
        participant_overlaps=participant_overlaps,
        split_mapping_sha256=split_mapping_sha256,
    )

    validation_path = (
        local_output_dir
        / "LOCAL_ONLY_selected_split_validation.txt"
    )

    validation_lines = [
        f"script_version={SCRIPT_VERSION}",
        f"candidate_id={summary.selected_candidate_id}",
        f"seed={summary.selected_seed}",
        f"score_final={summary.selected_score_final:.12f}",
        f"participant_counts={dict(summary.participant_counts)}",
        f"visit_counts={dict(summary.visit_counts)}",
        f"participant_overlaps={dict(summary.participant_overlaps)}",
        f"split_mapping_sha256={summary.split_mapping_sha256}",
        "allocation_unit=participant",
        "mri_targets_used_for_selection=false",
        "model_performance_used_for_selection=false",
    ]

    validation_path.write_text(
        "\n".join(
            validation_lines
        )
        + "\n",
        encoding="utf-8",
    )

    checkpoint_path.unlink(
        missing_ok=True
    )

    checkpoint_metadata_path.unlink(
        missing_ok=True
    )

    local_paths = (
        profile_path,
        ranking_path,
        mapping_path,
        selected_summary_path,
        validation_path,
    )

    return (
        ranking,
        selected,
        mapping,
        summary,
        local_paths,
    )
