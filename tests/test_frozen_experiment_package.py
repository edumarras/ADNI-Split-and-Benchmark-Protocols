from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adni_benchmark.frozen_experiment_package import (
    FrozenExperimentPackageError,
    build_model_ready_source_table,
    build_source_row_action_tables,
    calculate_column_catalog_sha256,
    calculate_file_sha256,
    canonical_postprocessed_table,
    clean_anchor_for_output,
    reconstruct_split_summary_from_local_outputs,
    write_dataframe_csv_gzip_deterministic,
)
from adni_benchmark.source_schema import SOURCE_CONTRACTS


def test_deterministic_gzip_writer_produces_identical_bytes(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "A": [1, 2],
            "B": ["x", "y"],
        }
    )

    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    write_dataframe_csv_gzip_deterministic(
        frame,
        first,
    )

    write_dataframe_csv_gzip_deterministic(
        frame,
        second,
    )

    assert first.read_bytes() == second.read_bytes()
    assert calculate_file_sha256(first) == calculate_file_sha256(second)


def test_column_catalog_hash_is_case_normalized_and_order_sensitive() -> None:
    first = calculate_column_catalog_sha256(
        [
            "st1sv",
            " ST2CV ",
        ]
    )

    equivalent = calculate_column_catalog_sha256(
        [
            "ST1SV",
            "ST2CV",
        ]
    )

    reordered = calculate_column_catalog_sha256(
        [
            "ST2CV",
            "ST1SV",
        ]
    )

    assert first == equivalent
    assert first != reordered


def test_canonical_postprocessed_table_adds_resolution_status() -> None:
    contract = SOURCE_CONTRACTS[
        "DXSUM"
    ]

    resolved = pd.DataFrame(
        {
            "RID": ["XX01"],
            "VISCODE2": ["bl"],
            "DIAGNOSIS": ["A"],
            "__resolution_status": ["unique_record"],
        }
    )

    output = canonical_postprocessed_table(
        resolved_frame=resolved,
        source_columns=(
            "RID",
            "VISCODE2",
            "DIAGNOSIS",
        ),
        contract=contract,
    )

    assert output.loc[
        0,
        "PROCESSING_RESOLUTION_STATUS",
    ] == "unique_record"


def test_canonical_mri_table_adds_target_audit_columns() -> None:
    contract = SOURCE_CONTRACTS[
        "UCSFFSX7"
    ]

    resolved = pd.DataFrame(
        {
            "RID": ["XX01"],
            "VISCODE2": ["m12"],
            "IMAGEUID": ["101"],
            "EXAMDATE": ["2020-01-01"],
            "STATUS": ["COMPLETE"],
            "OVERALLQC": ["PASS"],
            "FSVER": ["7"],
            "__resolution_status": ["unique_candidate"],
            "__target_nonmissing_count": [323],
            "__target_complete": [True],
        }
    )

    output = canonical_postprocessed_table(
        resolved_frame=resolved,
        source_columns=(
            "RID",
            "VISCODE2",
            "IMAGEUID",
            "EXAMDATE",
            "STATUS",
            "OVERALLQC",
            "FSVER",
        ),
        contract=contract,
    )

    assert output.loc[
        0,
        "PROCESSING_TARGET_NONMISSING_COUNT",
    ] == 323

    assert bool(
        output.loc[
            0,
            "PROCESSING_TARGET_COMPLETE",
        ]
    )


def test_row_action_tables_account_for_original_rows() -> None:
    contract = SOURCE_CONTRACTS[
        "DXSUM"
    ]

    source = pd.DataFrame(
        {
            "RID": [
                "XX01",
                "XX01",
                "XX02",
                pd.NA,
            ],
            "VISCODE2": [
                "bl",
                "bl",
                "m12",
                "m24",
            ],
            "DIAGNOSIS": [
                "A",
                "A",
                "B",
                "C",
            ],
        }
    )

    resolved = pd.DataFrame(
        {
            "RID": [
                "XX01",
                "XX02",
            ],
            "VISCODE2": [
                "bl",
                "m12",
            ],
            "DIAGNOSIS": [
                "A",
                "B",
            ],
            "__source_row": [
                0,
                2,
            ],
            "__resolution_status": [
                "exact_or_equivalent_duplicate",
                "unique_record",
            ],
        }
    )

    unresolved = pd.DataFrame(
        columns=resolved.columns
    )

    excluded, adjusted, summary = (
        build_source_row_action_tables(
            source_name="DXSUM",
            normalized_source=source,
            resolved_frame=resolved,
            unresolved_frame=unresolved,
            contract=contract,
        )
    )

    assert len(excluded) == 2
    assert len(adjusted) == 1
    assert summary["original_rows"] == 4
    assert summary["canonical_resolved_rows"] == 2
    assert summary["excluded_rows"] == 2


def test_clean_anchor_maps_split_and_sorts_visits() -> None:
    anchor = pd.DataFrame(
        {
            "RID": [
                "XX02",
                "XX01",
                "XX01",
            ],
            "VISCODE2": [
                "bl",
                "m12",
                "bl",
            ],
            "__eligible_visit_position": [
                1,
                2,
                1,
            ],
            "__first_eligible_visit": [
                True,
                False,
                True,
            ],
            "__protocol_month": [
                0.0,
                12.0,
                0.0,
            ],
            "__visit_order_basis": [
                "protocol_month",
                "protocol_month",
                "protocol_month",
            ],
        }
    )

    split = pd.DataFrame(
        {
            "RID": [
                "XX01",
                "XX02",
            ],
            "split": [
                "train",
                "test",
            ],
        }
    )

    output = clean_anchor_for_output(
        anchor,
        split,
    )

    assert output[
        "RID"
    ].tolist() == [
        "XX01",
        "XX01",
        "XX02",
    ]

    assert output[
        "eligible_visit_position"
    ].tolist() == [
        1,
        2,
        1,
    ]

    assert "__eligible_visit_position" not in output.columns


def test_model_ready_table_separates_resolved_empty_records() -> None:
    resolved = pd.DataFrame(
        {
            "RID": [
                "XX01",
                "XX02",
            ],
            "VISCODE2": [
                "bl",
                "bl",
            ],
            "FAQTOTAL": [
                "5",
                pd.NA,
            ],
            "__source_row": [
                0,
                1,
            ],
            "__resolution_status": [
                "unique_record",
                "unique_record",
            ],
        }
    )

    split = pd.DataFrame(
        {
            "RID": [
                "XX01",
                "XX02",
            ],
            "split": [
                "train",
                "validation",
            ],
        }
    )

    ready, empty, features = (
        build_model_ready_source_table(
            source_name="FAQ",
            resolved_frame=resolved,
            participant_split=split,
        )
    )

    assert len(ready) == 1
    assert len(empty) == 1
    assert features == (
        "FAQTOTAL",
    )

    assert empty.loc[
        0,
        "PROCESSING_REASON",
    ] == (
        "resolved_record_without_observed_substantive_field"
    )


def make_balance_profile(
    n_participants: int = 20,
) -> pd.DataFrame:
    records = []

    for index in range(n_participants):
        records.append(
            {
                "RID": f"XX{index:04d}",
                "n_visits": 1 + (index % 4),
                "diagnosis_first": (
                    "DX_A"
                    if index < 10
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
                "modality_count_mean": 5.0 + (index % 3),
            }
        )

    return pd.DataFrame(records)


def test_reconstruct_split_summary_from_local_outputs(
    tmp_path: Path,
) -> None:
    profile = make_balance_profile()

    mapping = pd.DataFrame(
        {
            "RID": profile["RID"],
            "split": (
                ["train"] * 14
                + ["validation"] * 3
                + ["test"] * 3
            ),
        }
    )

    ranking = pd.DataFrame(
        {
            "candidate_id": [
                1,
                2,
            ],
            "seed": [
                42000000,
                42000001,
            ],
            "score_final": [
                0.1,
                0.2,
            ],
            "score_max_component": [
                0.3,
                0.4,
            ],
            "score_max_component_name": [
                "sex",
                "n_visits",
            ],
            "is_valid": [
                True,
                True,
            ],
        }
    )

    ranking.to_csv(
        tmp_path
        / "LOCAL_ONLY_split_candidates_ranking.csv.gz",
        index=False,
        compression="gzip",
    )

    mapping.to_csv(
        tmp_path
        / "LOCAL_ONLY_selected_participant_split.csv",
        index=False,
    )

    rebuilt_mapping, summary = (
        reconstruct_split_summary_from_local_outputs(
            balance_profile=profile,
            local_output_dir=tmp_path,
        )
    )

    assert len(rebuilt_mapping) == 20
    assert summary.selected_candidate_id == 1
    assert summary.selected_seed == 42000000
    assert summary.n_candidates == 2
    assert summary.valid_candidates == 2
    assert dict(
        summary.participant_counts
    ) == {
        "train": 14,
        "validation": 3,
        "test": 3,
    }


def test_clean_anchor_requires_complete_split_mapping() -> None:
    anchor = pd.DataFrame(
        {
            "RID": ["XX01"],
            "VISCODE2": ["bl"],
            "__eligible_visit_position": [1],
            "__first_eligible_visit": [True],
            "__protocol_month": [0.0],
            "__visit_order_basis": ["protocol_month"],
        }
    )

    split = pd.DataFrame(
        {
            "RID": ["XX02"],
            "split": ["train"],
        }
    )

    with pytest.raises(
        FrozenExperimentPackageError,
        match="failed to map every supervised anchor visit",
    ):
        clean_anchor_for_output(
            anchor,
            split,
        )
