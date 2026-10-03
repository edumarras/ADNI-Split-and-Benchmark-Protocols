import pandas as pd

from adni_benchmark_protocol.benchmark02b_profile_input_features import (
    align_source_to_anchor,
    build_schema_draft,
    infer_type,
    profile_scope,
    support_band,
)


def test_support_band_boundaries() -> None:
    assert support_band(0) == "000_all_missing"
    assert support_band(5) == "001_1_to_5"
    assert support_band(10) == "002_6_to_10"
    assert support_band(20) == "003_11_to_20"
    assert support_band(30) == "004_21_to_30"
    assert support_band(50) == "005_31_to_50"
    assert support_band(100) == "006_51_to_100"
    assert support_band(250) == "007_101_to_250"
    assert support_band(500) == "008_251_to_500"
    assert support_band(501) == "009_over_500"


def test_infer_numeric_candidate() -> None:
    native = pd.Series(
        ["1.2", "2.5", "3.1", "4.4"],
        dtype="string",
    )

    snapshot = pd.Series(
        ["1.2", "2.5"],
        dtype="string",
    )

    result = infer_type(
        native,
        snapshot,
        "TEST_SCORE",
    )

    assert result["suggested_type"] == "numeric_candidate"
    assert result["suggested_encoding"] == "numeric_median_imputation"


def test_infer_low_cardinality_numeric_code() -> None:
    native = pd.Series(
        ["0", "1", "2", "0", "1", "2"],
        dtype="string",
    )

    snapshot = native.copy()

    result = infer_type(
        native,
        snapshot,
        "TEST_CODE",
    )

    assert result["suggested_type"] == "categorical_numeric_code_candidate"
    assert result["suggested_encoding"] == "one_hot_or_numeric_review"


def test_infer_multi_response_candidate() -> None:
    native = pd.Series(
        ["1|2", "2|3", "1"],
        dtype="string",
    )

    snapshot = native.copy()

    result = infer_type(
        native,
        snapshot,
        "TEST_MULTI",
    )

    assert result["suggested_type"] == "multi_response_candidate"
    assert result["suggested_encoding"] == "multi_hot"


def test_profile_scope_counts_participants_not_only_rows() -> None:
    values = pd.Series(
        ["1", "", "2", pd.NA],
        dtype="string",
    )

    rids = pd.Series(
        ["XX01", "XX01", "XX02", "XX03"],
        dtype="string",
    )

    result = profile_scope(
        values,
        rids,
        "train",
    )

    assert result["train_rows"] == 4
    assert result["train_participants"] == 3
    assert result["train_observed_rows"] == 2
    assert result["train_observed_participants"] == 2
    assert result["train_missing_rows"] == 2


def test_align_visit_source_to_anchor_preserves_anchor_rows() -> None:
    anchor = pd.DataFrame(
        {
            "RID": ["XX01", "XX01", "XX02"],
            "VISCODE2": ["bl", "m06", "bl"],
            "split": ["train", "train", "train"],
        }
    )

    source = pd.DataFrame(
        {
            "RID": ["XX01", "XX02"],
            "VISCODE2": ["bl", "bl"],
            "split": ["train", "train"],
            "FEATURE_A": ["1", "2"],
        }
    )

    aligned = align_source_to_anchor(
        "ADAS",
        source,
        anchor,
        ["FEATURE_A"],
    )

    assert len(aligned) == 3
    assert aligned["FEATURE_A"].notna().sum() == 2


def test_align_participant_source_to_anchor_broadcasts_by_rid() -> None:
    anchor = pd.DataFrame(
        {
            "RID": ["XX01", "XX01", "XX02"],
            "VISCODE2": ["bl", "m06", "bl"],
            "split": ["train", "train", "train"],
        }
    )

    source = pd.DataFrame(
        {
            "RID": ["XX01", "XX02"],
            "split": ["train", "train"],
            "FEATURE_A": ["A", "B"],
        }
    )

    aligned = align_source_to_anchor(
        "PTDEMOG",
        source,
        anchor,
        ["FEATURE_A"],
    )

    assert len(aligned) == 3
    assert aligned["FEATURE_A"].tolist() == ["A", "A", "B"]


def test_schema_draft_keeps_a06_contract_columns() -> None:
    profile = pd.DataFrame(
        [
            {
                "source_name": "ADAS",
                "grain": "visit",
                "feature_position_within_source": 1,
                "column_name": "FEATURE_A",
                "a06_directive_present": True,
                "a06_crosscheck_action": "retain",
                "a06_final_feature_type": "numeric",
                "a06_value_transform_id": "",
                "a06_semantic_role": "score",
                "a06_reason": "frozen_rule",
                "a06_execution_stage": "A06",
                "suggested_type": "numeric_candidate",
                "suggested_encoding": "numeric_median_imputation",
                "native_train_observed_rows": 10,
                "native_train_observed_participants": 8,
                "native_train_observed_fraction": 1.0,
                "snapshot_train_observed_rows": 10,
                "snapshot_train_observed_participants": 8,
                "snapshot_train_observed_fraction": 1.0,
                "snapshot_train_observed_participant_fraction": 1.0,
                "snapshot_train_distinct_observed_values": 7,
                "snapshot_train_participant_support_band": "002_6_to_10",
                "review_flags": "a06_semantic_directive_locked",
            }
        ]
    )

    draft = build_schema_draft(
        profile
    )

    assert draft.loc[
        0,
        "a06_directive_present",
    ]

    assert draft.loc[
        0,
        "final_include",
    ] == ""

    assert draft.loc[
        0,
        "final_feature_type",
    ] == ""
