from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd

from adni_benchmark.non_mri_resolution import comparison_columns_for_source
from adni_benchmark.participant_split_search import (
    SPLIT_BALANCE_WEIGHTS,
    SPLIT_MASTER_SEED,
    SPLIT_N_CANDIDATES,
    ParticipantSplitSearchSummary,
    build_split_search_context,
)
from adni_benchmark.source_schema import SOURCE_CONTRACTS
from adni_benchmark.split_balance_profile import (
    SPLIT_BALANCE_MODALITY_TOKENS,
    calculate_participant_profile_sha256,
)


class FrozenExperimentPackageError(RuntimeError):
    pass


SCRIPT_VERSION: Final[str] = "1.0.1"

REQUIRED_SOURCE_TOKENS: Final[tuple[str, ...]] = (
    "DXSUM",
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
    "PTDEMOG",
    "APOERES",
    "UCSFFSX7",
)

FIXED_EXCLUDED_MRI_TARGETS: Final[tuple[str, ...]] = (
    "ST8SV",
    "ST68SV",
)


def calculate_file_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def calculate_text_sha256(
    text: str,
) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def calculate_column_catalog_sha256(
    columns: Sequence[str],
) -> str:
    canonical_columns = [
        str(column).strip().upper()
        for column in columns
    ]

    return calculate_text_sha256(
        "\n".join(canonical_columns)
    )


def write_dataframe_csv_gzip_deterministic(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(
                gzip_handle,
                encoding="utf-8",
                newline="",
            ) as text_handle:
                frame.to_csv(
                    text_handle,
                    index=False,
                    lineterminator="\n",
                )

    return output_path


def write_dataframe_csv_deterministic(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return output_path


def sorted_by_contract_keys(
    frame: pd.DataFrame,
    contract,
) -> pd.DataFrame:
    sort_columns = [
        column
        for column in contract.key_columns
        if column in frame.columns
    ]

    if "__source_row" in frame.columns:
        sort_columns.append("__source_row")

    if not sort_columns:
        return frame.reset_index(drop=True)

    return (
        frame
        .sort_values(
            by=sort_columns,
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )


def source_key_index(
    frame: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.Index:
    if len(key_columns) == 1:
        return pd.Index(
            frame[key_columns[0]].astype("string"),
            name=key_columns[0],
        )

    return pd.MultiIndex.from_frame(
        frame.loc[:, list(key_columns)].astype("string")
    )


def canonical_postprocessed_table(
    resolved_frame: pd.DataFrame,
    source_columns: Sequence[str],
    contract,
) -> pd.DataFrame:
    missing_columns = [
        column
        for column in source_columns
        if column not in resolved_frame.columns
    ]

    if missing_columns:
        raise FrozenExperimentPackageError(
            f"Resolved {contract.source_name} table is missing "
            f"original source columns: {missing_columns}"
        )

    if "__resolution_status" not in resolved_frame.columns:
        raise FrozenExperimentPackageError(
            f"Resolved {contract.source_name} table is missing "
            "__resolution_status."
        )

    output = resolved_frame.loc[
        :,
        list(source_columns),
    ].copy()

    output["PROCESSING_RESOLUTION_STATUS"] = (
        resolved_frame["__resolution_status"]
        .astype("string")
        .to_numpy()
    )

    if contract.source_name == "UCSFFSX7":
        required = {
            "__target_nonmissing_count",
            "__target_complete",
        }

        missing = sorted(
            required - set(resolved_frame.columns)
        )

        if missing:
            raise FrozenExperimentPackageError(
                "Resolved UCSFFSX7 is missing frozen target-audit "
                f"columns: {missing}"
            )

        output["PROCESSING_TARGET_NONMISSING_COUNT"] = (
            resolved_frame["__target_nonmissing_count"]
            .astype("Int64")
            .to_numpy()
        )

        output["PROCESSING_TARGET_COMPLETE"] = (
            resolved_frame["__target_complete"]
            .fillna(False)
            .astype(bool)
            .to_numpy()
        )

    return sorted_by_contract_keys(
        output,
        contract,
    )


def build_source_row_action_tables(
    source_name: str,
    normalized_source: pd.DataFrame,
    resolved_frame: pd.DataFrame,
    unresolved_frame: pd.DataFrame,
    contract,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    key_columns = list(contract.key_columns)

    work = normalized_source.copy()
    work["__source_row"] = np.arange(
        len(work),
        dtype=np.int64,
    )

    valid_key_mask = (
        work.loc[:, key_columns]
        .notna()
        .all(axis=1)
    )

    valid_work = work.loc[
        valid_key_mask
    ].copy()

    selected_source_rows = set(
        resolved_frame["__source_row"]
        .dropna()
        .astype(int)
        .tolist()
    )

    excluded = work.loc[
        ~work["__source_row"].isin(
            selected_source_rows
        )
    ].copy()

    excluded["PROCESSING_SOURCE"] = source_name
    excluded["PROCESSING_ACTION"] = "excluded"
    excluded["PROCESSING_REASON"] = (
        "nonselected_record_from_resolved_group"
    )

    excluded_missing_key = (
        ~excluded.loc[:, key_columns]
        .notna()
        .all(axis=1)
    )

    excluded.loc[
        excluded_missing_key,
        "PROCESSING_REASON",
    ] = "missing_required_key"

    valid_excluded = excluded.loc[
        ~excluded_missing_key
    ]

    if (
        not unresolved_frame.empty
        and not valid_excluded.empty
    ):
        unresolved_index = source_key_index(
            unresolved_frame,
            key_columns,
        )

        excluded_valid_index = source_key_index(
            valid_excluded,
            key_columns,
        )

        unresolved_membership = (
            excluded_valid_index.isin(
                unresolved_index
            )
        )

        excluded.loc[
            valid_excluded.index[
                unresolved_membership
            ],
            "PROCESSING_REASON",
        ] = "unresolved_conflict_group"

    if valid_work.empty:
        group_size_frame = pd.DataFrame(
            columns=[
                *key_columns,
                "PROCESSING_SOURCE_GROUP_ROWS",
            ]
        )
    else:
        group_size_frame = (
            valid_work
            .groupby(
                key_columns[0]
                if len(key_columns) == 1
                else key_columns,
                dropna=False,
                sort=False,
            )
            .size()
            .rename(
                "PROCESSING_SOURCE_GROUP_ROWS"
            )
            .reset_index()
        )

    adjusted = resolved_frame.merge(
        group_size_frame,
        on=key_columns,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    adjusted["PROCESSING_SOURCE_GROUP_ROWS"] = (
        adjusted["PROCESSING_SOURCE_GROUP_ROWS"]
        .fillna(1)
        .astype("int64")
    )

    trivial_status = (
        "unique_candidate"
        if source_name == "UCSFFSX7"
        else "unique_record"
    )

    adjusted = adjusted.loc[
        adjusted["PROCESSING_SOURCE_GROUP_ROWS"].gt(1)
        | adjusted["__resolution_status"]
        .astype("string")
        .ne(trivial_status)
    ].copy()

    adjusted["PROCESSING_SOURCE"] = source_name
    adjusted["PROCESSING_ACTION"] = (
        "selected_canonical_record"
    )
    adjusted["PROCESSING_REASON"] = (
        adjusted["__resolution_status"]
        .astype("string")
    )

    if (
        not excluded.empty
        and not group_size_frame.empty
    ):
        excluded = excluded.merge(
            group_size_frame,
            on=key_columns,
            how="left",
            validate="many_to_one",
            sort=False,
        )
    else:
        excluded["PROCESSING_SOURCE_GROUP_ROWS"] = (
            pd.Series(
                pd.NA,
                index=excluded.index,
                dtype="Int64",
            )
        )

    excluded["PROCESSING_SOURCE_GROUP_ROWS"] = (
        excluded["PROCESSING_SOURCE_GROUP_ROWS"]
        .astype("Int64")
    )

    front_columns = [
        "PROCESSING_SOURCE",
        "PROCESSING_ACTION",
        "PROCESSING_REASON",
        "PROCESSING_SOURCE_GROUP_ROWS",
        "__source_row",
    ]

    excluded_columns = [
        column
        for column in front_columns
        if column in excluded.columns
    ] + [
        column
        for column in normalized_source.columns
        if column in excluded.columns
    ]

    adjusted_columns = [
        column
        for column in front_columns
        if column in adjusted.columns
    ] + [
        column
        for column in normalized_source.columns
        if column in adjusted.columns
    ] + [
        "__resolution_status",
    ]

    excluded = excluded.loc[
        :,
        list(dict.fromkeys(excluded_columns)),
    ]

    adjusted = adjusted.loc[
        :,
        list(dict.fromkeys(adjusted_columns)),
    ]

    excluded = sorted_by_contract_keys(
        excluded,
        contract,
    )

    adjusted = sorted_by_contract_keys(
        adjusted,
        contract,
    )

    reason_counts = (
        excluded["PROCESSING_REASON"]
        .astype("string")
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    summary = {
        "source_name": source_name,
        "original_rows": int(
            len(normalized_source)
        ),
        "canonical_resolved_rows": int(
            len(resolved_frame)
        ),
        "excluded_rows": int(
            len(excluded)
        ),
        "adjusted_selected_rows": int(
            len(adjusted)
        ),
        "unresolved_groups": int(
            len(unresolved_frame)
        ),
        "excluded_reason_counts": {
            str(reason): int(count)
            for reason, count
            in reason_counts.items()
        },
    }

    if (
        summary["canonical_resolved_rows"]
        + summary["excluded_rows"]
        != summary["original_rows"]
    ):
        raise FrozenExperimentPackageError(
            f"{source_name} frozen row-action accounting failed."
        )

    return (
        excluded,
        adjusted,
        summary,
    )


def clean_anchor_for_output(
    supervised_anchor: pd.DataFrame,
    participant_split: pd.DataFrame,
) -> pd.DataFrame:
    output = supervised_anchor.merge(
        participant_split,
        on="RID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if output["split"].isna().any():
        raise FrozenExperimentPackageError(
            "Official split failed to map every supervised anchor visit."
        )

    rename_map = {
        column: column.removeprefix("__")
        for column in output.columns
        if column.startswith("__")
    }

    output = output.rename(
        columns=rename_map
    )

    preferred = [
        "RID",
        "VISCODE2",
        "split",
        "eligible_visit_position",
        "first_eligible_visit",
        "protocol_month",
        "visit_order_basis",
    ]

    remaining = [
        column
        for column in output.columns
        if column not in preferred
    ]

    return (
        output.loc[
            :,
            [
                *preferred,
                *remaining,
            ],
        ]
        .sort_values(
            by=[
                "RID",
                "eligible_visit_position",
                "VISCODE2",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def convert_mri_targets_to_numeric(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    if not target_columns:
        raise FrozenExperimentPackageError(
            "The MRI target catalog is empty."
        )

    missing_columns = [
        column
        for column in target_columns
        if column not in frame.columns
    ]

    if missing_columns:
        raise FrozenExperimentPackageError(
            "MRI target conversion is missing columns: "
            f"{missing_columns}"
        )

    original_targets = frame.loc[
        :,
        list(target_columns),
    ]

    numeric_targets = original_targets.apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_observed_mask = (
        original_targets.notna()
        & numeric_targets.isna()
    )

    invalid_observed_count = int(
        invalid_observed_mask
        .to_numpy()
        .sum()
    )

    if invalid_observed_count:
        raise FrozenExperimentPackageError(
            "The MRI target catalog contains "
            f"{invalid_observed_count} observed non-numeric cells."
        )

    numeric_array = numeric_targets.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    infinite_count = int(
        np.isinf(numeric_array).sum()
    )

    if infinite_count:
        raise FrozenExperimentPackageError(
            "The MRI target catalog contains "
            f"{infinite_count} infinite numeric cells."
        )

    return numeric_targets.astype("float64")


def build_model_ready_source_table(
    source_name: str,
    resolved_frame: pd.DataFrame,
    participant_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    contract = SOURCE_CONTRACTS[
        source_name
    ]

    substantive_columns = (
        comparison_columns_for_source(
            frame=resolved_frame,
            contract=contract,
        )
    )

    content_mask = (
        resolved_frame.loc[
            :,
            substantive_columns,
        ]
        .notna()
        .any(axis=1)
    )

    included = resolved_frame.loc[
        content_mask
    ].copy()

    empty = resolved_frame.loc[
        ~content_mask
    ].copy()

    official_rids = set(
        participant_split["RID"]
        .astype("string")
        .tolist()
    )

    included = included.loc[
        included["RID"]
        .astype("string")
        .isin(official_rids)
    ].copy()

    empty = empty.loc[
        empty["RID"]
        .astype("string")
        .isin(official_rids)
    ].copy()

    included = included.merge(
        participant_split,
        on="RID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    empty = empty.merge(
        participant_split,
        on="RID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    output_columns = [
        *contract.key_columns,
        "split",
        *substantive_columns,
    ]

    included = included.loc[
        :,
        output_columns,
    ]

    empty = empty.loc[
        :,
        output_columns,
    ]

    empty["PROCESSING_REASON"] = (
        "resolved_record_without_observed_substantive_field"
    )

    included = sorted_by_contract_keys(
        included,
        contract,
    )

    empty = sorted_by_contract_keys(
        empty,
        contract,
    )

    return (
        included,
        empty,
        tuple(substantive_columns),
    )


def reconstruct_split_summary_from_local_outputs(
    balance_profile: pd.DataFrame,
    local_output_dir: Path,
) -> tuple[
    pd.DataFrame,
    ParticipantSplitSearchSummary,
]:
    ranking_path = (
        local_output_dir
        / "LOCAL_ONLY_split_candidates_ranking.csv.gz"
    )

    mapping_path = (
        local_output_dir
        / "LOCAL_ONLY_selected_participant_split.csv"
    )

    for path in (
        ranking_path,
        mapping_path,
    ):
        if not path.exists():
            raise FrozenExperimentPackageError(
                "Required LOCAL_ONLY split artifact is missing: "
                f"{path}"
            )

    ranking = pd.read_csv(
        ranking_path,
        low_memory=False,
    )

    mapping = pd.read_csv(
        mapping_path,
        dtype={
            "RID": "string",
            "split": "string",
        },
    )

    if len(mapping) != len(balance_profile):
        raise FrozenExperimentPackageError(
            "Local split mapping participant count does not match "
            "the balance profile."
        )

    if mapping["RID"].duplicated().any():
        raise FrozenExperimentPackageError(
            "Local split mapping contains duplicated participant IDs."
        )

    valid_ranking = ranking.loc[
        ranking["is_valid"].astype(bool)
    ].copy()

    if valid_ranking.empty:
        raise FrozenExperimentPackageError(
            "Local candidate ranking contains no valid candidates."
        )

    valid_ranking = valid_ranking.sort_values(
        by=[
            "score_final",
            "score_max_component",
            "seed",
        ],
        kind="stable",
    )

    selected = valid_ranking.iloc[0]

    context = build_split_search_context(
        balance_profile
    )

    profile_sha256 = (
        calculate_participant_profile_sha256(
            balance_profile
        )
    )

    profile_with_split = balance_profile.merge(
        mapping,
        on="RID",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if profile_with_split["split"].isna().any():
        raise FrozenExperimentPackageError(
            "Local split mapping failed to cover the complete balance profile."
        )

    participant_counts = tuple(
        (
            split_name,
            int(
                profile_with_split["split"]
                .eq(split_name)
                .sum()
            ),
        )
        for split_name in (
            "train",
            "validation",
            "test",
        )
    )

    visit_counts = tuple(
        (
            split_name,
            int(
                profile_with_split.loc[
                    profile_with_split["split"]
                    .eq(split_name),
                    "n_visits",
                ].sum()
            ),
        )
        for split_name in (
            "train",
            "validation",
            "test",
        )
    )

    split_sets = {
        split_name: set(
            mapping.loc[
                mapping["split"].eq(split_name),
                "RID",
            ].tolist()
        )
        for split_name in (
            "train",
            "validation",
            "test",
        )
    }

    participant_overlaps = (
        (
            "train_validation",
            len(
                split_sets["train"]
                & split_sets["validation"]
            ),
        ),
        (
            "train_test",
            len(
                split_sets["train"]
                & split_sets["test"]
            ),
        ),
        (
            "validation_test",
            len(
                split_sets["validation"]
                & split_sets["test"]
            ),
        ),
    )

    mapping_sorted = (
        mapping
        .sort_values(
            "RID",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    split_mapping_sha256 = (
        calculate_text_sha256(
            "\n".join(
                f"{row.RID},{row.split}"
                for row
                in mapping_sorted.itertuples(
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
        n_candidates=int(
            len(ranking)
        ),
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
            selected["candidate_id"]
        ),
        selected_seed=int(
            selected["seed"]
        ),
        selected_score_final=float(
            selected["score_final"]
        ),
        selected_score_max_component=float(
            selected["score_max_component"]
        ),
        selected_score_max_component_name=str(
            selected[
                "score_max_component_name"
            ]
        ),
        participant_counts=participant_counts,
        visit_counts=visit_counts,
        participant_overlaps=participant_overlaps,
        split_mapping_sha256=(
            split_mapping_sha256
        ),
    )

    return (
        mapping_sorted,
        summary,
    )


def split_summary_manifest_record(
    summary: ParticipantSplitSearchSummary,
) -> dict[str, object]:
    return {
        "search": {
            "n_candidates": summary.n_candidates,
            "valid_candidates": summary.valid_candidates,
            "master_seed": SPLIT_MASTER_SEED,
            "candidate_seed_rule": (
                "master_seed + candidate_id - 1"
            ),
            "target_proportions": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "fixed_participant_counts": (
                summary.split_sizes
                .to_manifest_record()
            ),
            "participant_profile_sha256": (
                summary.participant_profile_sha256
            ),
        },
        "balance_contract": {
            "categorical_distance": (
                "Jensen-Shannon distance, base 2"
            ),
            "numeric_distance": (
                "exact one-dimensional Wasserstein-1 distance after global "
                "participant-level z-score standardization"
            ),
            "criterion_aggregation": (
                "0.5 * mean(train, validation, test) + 0.5 * "
                "max(train, validation, test)"
            ),
            "final_score": (
                "unweighted arithmetic mean of the nine active criterion "
                "scores; every criterion has weight 1"
            ),
            "weights": SPLIT_BALANCE_WEIGHTS,
            "active_criteria": list(
                summary.active_criteria
            ),
            "active_weight_sum": (
                summary.active_weight_sum
            ),
            "target_values_used": False,
            "model_performance_used": False,
        },
        "selection": {
            "candidate_id": (
                summary.selected_candidate_id
            ),
            "seed": summary.selected_seed,
            "score_final": (
                summary.selected_score_final
            ),
            "score_max_component": (
                summary.selected_score_max_component
            ),
            "score_max_component_name": (
                summary.selected_score_max_component_name
            ),
            "split_mapping_sha256": (
                summary.split_mapping_sha256
            ),
        },
        "validation": {
            "participant_counts": dict(
                summary.participant_counts
            ),
            "visit_counts": dict(
                summary.visit_counts
            ),
            "participant_overlaps": dict(
                summary.participant_overlaps
            ),
        },
    }


def materialize_frozen_experiment_package(
    processed_data_dir: Path,
    reports_dir: Path,
    snapshots: dict[str, object],
    source_tables: dict[str, pd.DataFrame],
    resolved_non_mri_sources: dict[str, pd.DataFrame],
    unresolved_non_mri_sources: dict[str, pd.DataFrame],
    resolved_mri_source: pd.DataFrame,
    unresolved_mri_source: pd.DataFrame,
    mri_target_columns: Sequence[str],
    supervised_anchor: pd.DataFrame,
    split_balance_profile: pd.DataFrame,
    participant_split: pd.DataFrame,
    split_summary: ParticipantSplitSearchSummary,
) -> tuple[
    Path,
    Path,
    tuple[Path, ...],
]:
    frozen_root = (
        processed_data_dir
        / "frozen_experiment_v1"
    )

    if frozen_root.exists():
        shutil.rmtree(frozen_root)

    postprocessed_dir = (
        frozen_root
        / "01_postprocessed_sources"
    )

    changes_dir = (
        frozen_root
        / "02_cuts_and_adjustments"
    )

    final_tables_dir = (
        frozen_root
        / "03_final_ready_tables"
    )

    official_split_dir = (
        frozen_root
        / "04_official_split"
    )

    for directory in (
        postprocessed_dir,
        changes_dir,
        final_tables_dir,
        official_split_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    readme_path = (
        frozen_root
        / "README_LOCAL_ONLY.txt"
    )

    readme_path.write_text(
        "\n".join(
            [
                "FROZEN EXPERIMENT PACKAGE v1",
                "",
                "LOCAL ONLY — contains individualized ADNI-derived records.",
                "Do not commit, upload, publish, or share this directory.",
                "",
                "01_postprocessed_sources: one canonical resolved row per source grain.",
                "02_cuts_and_adjustments: rows excluded, non-trivial selections, and rule manifests.",
                "03_final_ready_tables: official target-complete anchor, 323 MRI targets, and canonical input histories.",
                "04_official_split: frozen participant and visit assignments, profile, validation, and hashes.",
                "",
                "Longitudinal benchmark code must use only input records at or before each target visit.",
                "The test split must not influence preprocessing, hyperparameters, feature removal, imputation, scaling, or PCA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    written_paths: list[Path] = [
        readme_path
    ]

    row_action_summaries: list[
        dict[str, object]
    ] = []

    feature_catalog_records: list[
        dict[str, object]
    ] = []

    all_resolved_sources = {
        **resolved_non_mri_sources,
        "UCSFFSX7": resolved_mri_source,
    }

    all_unresolved_sources = {
        **unresolved_non_mri_sources,
        "UCSFFSX7": unresolved_mri_source,
    }

    for source_name in REQUIRED_SOURCE_TOKENS:
        contract = SOURCE_CONTRACTS[
            source_name
        ]

        resolved_frame = all_resolved_sources[
            source_name
        ]

        unresolved_frame = all_unresolved_sources[
            source_name
        ]

        source_columns = tuple(
            snapshots[
                source_name
            ].columns
        )

        canonical = canonical_postprocessed_table(
            resolved_frame=resolved_frame,
            source_columns=source_columns,
            contract=contract,
        )

        canonical_path = (
            postprocessed_dir
            / f"{source_name}_canonical_resolved.csv.gz"
        )

        write_dataframe_csv_gzip_deterministic(
            canonical,
            canonical_path,
        )

        written_paths.append(
            canonical_path
        )

        (
            excluded,
            adjusted,
            action_summary,
        ) = build_source_row_action_tables(
            source_name=source_name,
            normalized_source=source_tables[
                source_name
            ],
            resolved_frame=resolved_frame,
            unresolved_frame=unresolved_frame,
            contract=contract,
        )

        row_action_summaries.append(
            action_summary
        )

        excluded_path = (
            changes_dir
            / f"{source_name}_excluded_rows.csv.gz"
        )

        adjusted_path = (
            changes_dir
            / f"{source_name}_nontrivial_selected_rows.csv.gz"
        )

        unresolved_path = (
            changes_dir
            / f"{source_name}_unresolved_groups.csv.gz"
        )

        write_dataframe_csv_gzip_deterministic(
            excluded,
            excluded_path,
        )

        write_dataframe_csv_gzip_deterministic(
            adjusted,
            adjusted_path,
        )

        write_dataframe_csv_gzip_deterministic(
            sorted_by_contract_keys(
                unresolved_frame,
                contract,
            ),
            unresolved_path,
        )

        written_paths.extend(
            [
                excluded_path,
                adjusted_path,
                unresolved_path,
            ]
        )

    target_complete_mask = (
        resolved_mri_source[
            "__target_complete"
        ]
        .fillna(False)
        .astype(bool)
    )

    target_incomplete = (
        resolved_mri_source.loc[
            ~target_complete_mask,
            [
                "RID",
                "VISCODE2",
                "IMAGEUID",
                "__target_nonmissing_count",
                "__resolution_status",
            ],
        ]
        .copy()
        .rename(
            columns={
                "__target_nonmissing_count": (
                    "TARGET_NONMISSING_COUNT"
                ),
                "__resolution_status": (
                    "PROCESSING_RESOLUTION_STATUS"
                ),
            }
        )
    )

    target_incomplete[
        "PROCESSING_REASON"
    ] = (
        "resolved_mri_visit_with_incomplete_323_target_vector"
    )

    target_incomplete_path = (
        changes_dir
        / "UCSFFSX7_resolved_but_target_incomplete.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        target_incomplete
        .sort_values(
            [
                "RID",
                "VISCODE2",
            ],
            kind="stable",
        )
        .reset_index(drop=True),
        target_incomplete_path,
    )

    written_paths.append(
        target_incomplete_path
    )

    row_summary_path = (
        changes_dir
        / "row_action_summary.csv"
    )

    write_dataframe_csv_deterministic(
        pd.DataFrame(
            row_action_summaries
        ),
        row_summary_path,
    )

    written_paths.append(
        row_summary_path
    )

    column_actions = pd.DataFrame(
        [
            {
                "source_name": "UCSFFSX7",
                "column_name": column_name,
                "action": (
                    "excluded_from_supervised_target_catalog"
                ),
                "reason": (
                    "fixed methodological exclusion due to structural "
                    "missingness; retained only in the 325-measure source catalog"
                ),
            }
            for column_name
            in FIXED_EXCLUDED_MRI_TARGETS
        ]
    )

    column_actions_path = (
        changes_dir
        / "column_actions.csv"
    )

    write_dataframe_csv_deterministic(
        column_actions,
        column_actions_path,
    )

    written_paths.append(
        column_actions_path
    )

    processing_rules = pd.DataFrame(
        [
            {
                "rule_order": 1,
                "rule": "load_as_text",
                "summary": (
                    "Load source fields as strings without automatic NA inference."
                ),
            },
            {
                "rule_order": 2,
                "rule": "normalize_missing",
                "summary": (
                    "Trim whitespace and map empty, -1, and -4 to missing."
                ),
            },
            {
                "rule_order": 3,
                "rule": "require_structural_keys",
                "summary": (
                    "Exclude rows without the complete source key."
                ),
            },
            {
                "rule_order": 4,
                "rule": "preserve_real_rows",
                "summary": (
                    "Resolve duplicates by equivalence or unique dominance "
                    "without cell-wise coalescence."
                ),
            },
            {
                "rule_order": 5,
                "rule": "resolve_mri_by_target_completeness",
                "summary": (
                    "Select a unique maximum-completeness 323-target "
                    "acquisition; equivalent ties use deterministic IMAGEUID "
                    "ordering; material conflicts remain unresolved."
                ),
            },
            {
                "rule_order": 6,
                "rule": "require_complete_mri_target",
                "summary": (
                    "Retain only resolved visits with all 323 supervised MRI "
                    "targets observed."
                ),
            },
            {
                "rule_order": 7,
                "rule": "participant_isolated_split",
                "summary": (
                    "Assign all records of each participant to exactly one of "
                    "train, validation, or test."
                ),
            },
            {
                "rule_order": 8,
                "rule": "uniform_split_score",
                "summary": (
                    "Use nine criteria with weight 1 each, raw "
                    "Jensen-Shannon/Wasserstein distances, and 0.5 mean + "
                    "0.5 maximum split aggregation."
                ),
            },
        ]
    )

    processing_rules_path = (
        changes_dir
        / "processing_rules.csv"
    )

    write_dataframe_csv_deterministic(
        processing_rules,
        processing_rules_path,
    )

    written_paths.append(
        processing_rules_path
    )

    clean_anchor = clean_anchor_for_output(
        supervised_anchor=supervised_anchor,
        participant_split=participant_split,
    )

    anchor_path = (
        final_tables_dir
        / "supervised_anchor_visits.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        clean_anchor,
        anchor_path,
    )

    written_paths.append(
        anchor_path
    )

    target_rows = resolved_mri_source.loc[
        target_complete_mask,
        [
            "RID",
            "VISCODE2",
            *mri_target_columns,
        ],
    ].copy()

    numeric_targets = (
        convert_mri_targets_to_numeric(
            frame=target_rows,
            target_columns=mri_target_columns,
        )
        .reset_index(drop=True)
    )

    target_keys = (
        target_rows.loc[
            :,
            [
                "RID",
                "VISCODE2",
            ],
        ]
        .reset_index(drop=True)
    )

    target_rows = pd.concat(
        [
            target_keys,
            numeric_targets,
        ],
        axis=1,
        copy=False,
    )

    if (
        target_rows.loc[
            :,
            list(mri_target_columns),
        ]
        .isna()
        .any()
        .any()
    ):
        raise FrozenExperimentPackageError(
            "Frozen target table contains missing values among the 323 targets."
        )

    target_rows = target_rows.merge(
        participant_split,
        on="RID",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    target_rows = target_rows.merge(
        clean_anchor.loc[
            :,
            [
                "RID",
                "VISCODE2",
                "eligible_visit_position",
            ],
        ],
        on=[
            "RID",
            "VISCODE2",
        ],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    target_rows = (
        target_rows.loc[
            :,
            [
                "RID",
                "VISCODE2",
                "split",
                "eligible_visit_position",
                *mri_target_columns,
            ],
        ]
        .sort_values(
            [
                "RID",
                "eligible_visit_position",
                "VISCODE2",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(target_rows) != len(clean_anchor):
        raise FrozenExperimentPackageError(
            "Frozen target table and supervised anchor visit counts differ."
        )

    target_path = (
        final_tables_dir
        / "UCSFFSX7_targets_323.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        target_rows,
        target_path,
    )

    written_paths.append(
        target_path
    )

    target_catalog = pd.DataFrame(
        {
            "target_position_1_based": np.arange(
                1,
                len(mri_target_columns) + 1,
                dtype=np.int64,
            ),
            "target_column": list(
                mri_target_columns
            ),
        }
    )

    target_catalog_path = (
        final_tables_dir
        / "MRI_target_catalog_323.csv"
    )

    write_dataframe_csv_deterministic(
        target_catalog,
        target_catalog_path,
    )

    written_paths.append(
        target_catalog_path
    )

    inputs_dir = (
        final_tables_dir
        / "input_sources"
    )

    inputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    empty_model_ready_records: list[
        pd.DataFrame
    ] = []

    for source_name in (
        SPLIT_BALANCE_MODALITY_TOKENS
    ):
        (
            ready,
            empty,
            feature_columns,
        ) = build_model_ready_source_table(
            source_name=source_name,
            resolved_frame=(
                resolved_non_mri_sources[
                    source_name
                ]
            ),
            participant_split=participant_split,
        )

        ready_path = (
            inputs_dir
            / f"{source_name}_model_ready.csv.gz"
        )

        write_dataframe_csv_gzip_deterministic(
            ready,
            ready_path,
        )

        written_paths.append(
            ready_path
        )

        if not empty.empty:
            tagged_empty = empty.copy()
            tagged_empty.insert(
                0,
                "source_name",
                source_name,
            )
            empty_model_ready_records.append(
                tagged_empty
            )

        for position, column_name in enumerate(
            feature_columns,
            start=1,
        ):
            feature_catalog_records.append(
                {
                    "source_name": source_name,
                    "feature_position_within_source": position,
                    "column_name": column_name,
                    "role": "candidate_input_feature",
                    "final_train_only_removal_pending": (
                        "remove only if 100% missing or constant in training"
                    ),
                }
            )

    dxsum_contract = SOURCE_CONTRACTS[
        "DXSUM"
    ]

    dxsum_metadata = (
        resolved_non_mri_sources["DXSUM"]
        .loc[
            :,
            [
                *dxsum_contract.key_columns,
                "DIAGNOSIS",
            ],
        ]
        .copy()
    )

    official_rids = set(
        participant_split["RID"]
        .astype("string")
        .tolist()
    )

    dxsum_metadata = (
        dxsum_metadata.loc[
            dxsum_metadata["RID"]
            .astype("string")
            .isin(official_rids)
        ]
        .merge(
            participant_split,
            on="RID",
            how="left",
            validate="many_to_one",
            sort=False,
        )
    )

    dxsum_metadata = dxsum_metadata.loc[
        :,
        [
            "RID",
            "VISCODE2",
            "split",
            "DIAGNOSIS",
        ],
    ]

    dxsum_metadata = sorted_by_contract_keys(
        dxsum_metadata,
        dxsum_contract,
    )

    dxsum_path = (
        final_tables_dir
        / "DXSUM_split_balance_metadata.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        dxsum_metadata,
        dxsum_path,
    )

    written_paths.append(
        dxsum_path
    )

    feature_catalog_path = (
        final_tables_dir
        / "input_feature_catalog.csv"
    )

    write_dataframe_csv_deterministic(
        pd.DataFrame(
            feature_catalog_records
        ),
        feature_catalog_path,
    )

    written_paths.append(
        feature_catalog_path
    )

    if empty_model_ready_records:
        model_ready_empty = pd.concat(
            empty_model_ready_records,
            ignore_index=True,
            sort=False,
        )
    else:
        model_ready_empty = pd.DataFrame(
            columns=[
                "source_name",
                "PROCESSING_REASON",
            ]
        )

    model_ready_empty_path = (
        changes_dir
        / "resolved_empty_records_excluded_from_model_ready.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        model_ready_empty,
        model_ready_empty_path,
    )

    written_paths.append(
        model_ready_empty_path
    )

    participant_split_path = (
        official_split_dir
        / "official_participant_split.csv"
    )

    write_dataframe_csv_deterministic(
        participant_split
        .sort_values(
            "RID",
            kind="stable",
        )
        .reset_index(drop=True),
        participant_split_path,
    )

    written_paths.append(
        participant_split_path
    )

    visit_split = clean_anchor.loc[
        :,
        [
            "RID",
            "VISCODE2",
            "split",
            "eligible_visit_position",
        ],
    ].copy()

    visit_split_path = (
        official_split_dir
        / "official_visit_split.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        visit_split,
        visit_split_path,
    )

    written_paths.append(
        visit_split_path
    )

    balance_with_split = (
        split_balance_profile
        .merge(
            participant_split,
            on="RID",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .sort_values(
            "RID",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if balance_with_split["split"].isna().any():
        raise FrozenExperimentPackageError(
            "Official split failed to map the complete balance profile."
        )

    balance_profile_path = (
        official_split_dir
        / "official_participant_balance_profile.csv.gz"
    )

    write_dataframe_csv_gzip_deterministic(
        balance_with_split,
        balance_profile_path,
    )

    written_paths.append(
        balance_profile_path
    )

    validation_path = (
        official_split_dir
        / "official_split_validation.txt"
    )

    validation_lines = [
        f"script_version={SCRIPT_VERSION}",
        "official_split_frozen=true",
        "official_configuration=uniform__raw__mean_max_equal",
        f"candidate_id={split_summary.selected_candidate_id}",
        f"seed={split_summary.selected_seed}",
        f"score_final={split_summary.selected_score_final:.15f}",
        (
            "score_max_component="
            f"{split_summary.selected_score_max_component:.15f}"
        ),
        (
            "score_max_component_name="
            f"{split_summary.selected_score_max_component_name}"
        ),
        f"participant_counts={dict(split_summary.participant_counts)}",
        f"visit_counts={dict(split_summary.visit_counts)}",
        f"participant_overlaps={dict(split_summary.participant_overlaps)}",
        (
            "participant_profile_sha256="
            f"{split_summary.participant_profile_sha256}"
        ),
        (
            "split_mapping_sha256="
            f"{split_summary.split_mapping_sha256}"
        ),
        "target_count=323",
        "target_values_used_for_split_selection=false",
        "model_performance_used_for_split_selection=false",
        "allocation_unit=participant",
    ]

    validation_path.write_text(
        "\n".join(validation_lines)
        + "\n",
        encoding="utf-8",
    )

    written_paths.append(
        validation_path
    )

    file_inventory_records: list[
        dict[str, object]
    ] = []

    for file_path in sorted(written_paths):
        if file_path == readme_path:
            category = "documentation"
        else:
            category = (
                file_path
                .relative_to(frozen_root)
                .parts[0]
            )

        file_inventory_records.append(
            {
                "category": category,
                "relative_path": (
                    file_path
                    .relative_to(frozen_root)
                    .as_posix()
                ),
                "size_bytes": int(
                    file_path.stat().st_size
                ),
                "sha256": calculate_file_sha256(
                    file_path
                ),
            }
        )

    inventory_path = (
        official_split_dir
        / "frozen_file_inventory.csv"
    )

    write_dataframe_csv_deterministic(
        pd.DataFrame(
            file_inventory_records
        ),
        inventory_path,
    )

    written_paths.append(
        inventory_path
    )

    source_hashes = {
        source_name: snapshots[
            source_name
        ].sha256
        for source_name in REQUIRED_SOURCE_TOKENS
    }

    target_catalog_sha256 = (
        calculate_column_catalog_sha256(
            mri_target_columns
        )
    )

    final_manifest = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "frozen_package_name": (
            "frozen_experiment_v1"
        ),
        "official_split_frozen": True,
        "official_configuration": {
            "weight_scheme": "uniform",
            "criterion_weights": (
                SPLIT_BALANCE_WEIGHTS
            ),
            "distance_scale": "raw",
            "split_aggregation": (
                "0.5 * mean(train, validation, test) + "
                "0.5 * max(train, validation, test)"
            ),
            "final_score": (
                "arithmetic mean of nine criterion scores"
            ),
            "candidate_pool": (
                SPLIT_N_CANDIDATES
            ),
        },
        "population": {
            "supervised_target_count": int(
                len(mri_target_columns)
            ),
            "supervised_visits": int(
                len(clean_anchor)
            ),
            "participants": int(
                len(participant_split)
            ),
        },
        "selection": split_summary_manifest_record(
            split_summary
        ),
        "source_sha256": source_hashes,
        "target_catalog_sha256": (
            target_catalog_sha256
        ),
        "sensitivity": {
            "rerun_in_reconstruction": False,
            "required_for_materialization": False,
            "role": (
                "historical robustness audit; intentionally not rerun"
            ),
        },
        "folder_contract": {
            "01_postprocessed_sources": (
                "canonical resolved source tables after key validation and "
                "duplicate resolution"
            ),
            "02_cuts_and_adjustments": (
                "local row-level exclusions, non-trivial selections, empty "
                "records, and rule manifests"
            ),
            "03_final_ready_tables": (
                "target-complete supervised anchor, 323-target table, and "
                "canonical input histories for official participants"
            ),
            "04_official_split": (
                "frozen participant/visit assignments and integrity files"
            ),
        },
        "privacy": {
            "package_is_local_only": True,
            "contains_individualized_records": True,
            "must_not_be_committed_or_shared": True,
            "safe_aggregate_summary_written_separately": True,
        },
    }

    manifest_path = (
        official_split_dir
        / "official_frozen_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            final_manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    written_paths.append(
        manifest_path
    )

    safe_output_dir = (
        reports_dir
        / "auto_split_maker"
        / "safe"
    )

    safe_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_summary_path = (
        safe_output_dir
        / "official_frozen_split_summary.json"
    )

    safe_summary = {
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "official_split_frozen": True,
        "official_configuration": {
            "weight_scheme": "uniform",
            "all_criterion_weights": 1.0,
            "distance_scale": "raw",
            "split_aggregation": (
                "0.5 mean + 0.5 worst split"
            ),
            "categorical_distance": (
                "Jensen-Shannon distance, base 2"
            ),
            "numeric_distance": (
                "one-dimensional Wasserstein-1 after global participant-level "
                "z-score standardization"
            ),
        },
        "population": {
            "targets": int(
                len(mri_target_columns)
            ),
            "visits": int(
                len(clean_anchor)
            ),
            "participants": int(
                len(participant_split)
            ),
        },
        "selection": {
            "candidate_id": (
                split_summary.selected_candidate_id
            ),
            "seed": split_summary.selected_seed,
            "score_final": (
                split_summary.selected_score_final
            ),
            "score_max_component": (
                split_summary.selected_score_max_component
            ),
            "score_max_component_name": (
                split_summary.selected_score_max_component_name
            ),
            "participant_counts": dict(
                split_summary.participant_counts
            ),
            "visit_counts": dict(
                split_summary.visit_counts
            ),
            "participant_overlaps": dict(
                split_summary.participant_overlaps
            ),
            "participant_profile_sha256": (
                split_summary.participant_profile_sha256
            ),
            "split_mapping_sha256": (
                split_summary.split_mapping_sha256
            ),
            "target_catalog_sha256": (
                target_catalog_sha256
            ),
        },
        "reproducibility": {
            "n_candidates": SPLIT_N_CANDIDATES,
            "master_seed": SPLIT_MASTER_SEED,
            "candidate_seed_rule": (
                "master_seed + candidate_id - 1"
            ),
            "source_sha256": source_hashes,
            "reconstruction_script_sha256": (
                calculate_file_sha256(
                    Path(__file__).resolve()
                )
            ),
        },
        "privacy": {
            "contains_individual_records": False,
            "contains_participant_identifiers": False,
            "contains_visit_identifiers": False,
            "contains_medical_values": False,
            "contains_only_aggregate_counts_and_global_hashes": True,
        },
    }

    safe_summary_path.write_text(
        json.dumps(
            safe_summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        frozen_root,
        safe_summary_path,
        tuple(written_paths),
    )
