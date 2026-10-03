import pandas as pd

from adni_benchmark_protocol.benchmark04b_finalize_feature_schema import (
    Checks,
    EXPECTED_SUPPORT_THRESHOLD,
    TYPE_TO_ENCODING,
    build_transform_registry,
    canonical_schema_payload,
    parse_bool,
)


def test_parse_bool_contract() -> None:
    assert parse_bool(True) is True
    assert parse_bool("true") is True
    assert parse_bool("1") is True
    assert parse_bool(False) is False
    assert parse_bool("false") is False
    assert parse_bool("") is False


def test_support_threshold_is_frozen_at_50() -> None:
    assert EXPECTED_SUPPORT_THRESHOLD == 50


def test_type_to_encoding_contract() -> None:
    assert TYPE_TO_ENCODING["numeric"] == "numeric_standardized"
    assert TYPE_TO_ENCODING["binary"] == "binary_identity"
    assert TYPE_TO_ENCODING["categorical"] == "categorical_one_hot"
    assert TYPE_TO_ENCODING["multi_response"] == "multi_hot"
    assert TYPE_TO_ENCODING["excluded"] == "excluded"


def test_canonical_schema_payload_is_deterministic() -> None:
    schema = pd.DataFrame(
        [
            {
                "source_name": "PTDEMOG",
                "column_name": "AGE_AT_TARGET",
                "full_column_name": "PTDEMOG__AGE_AT_TARGET",
                "grain": "visit_derived",
                "current_in_crosschecked_inputs": False,
                "derived_later": True,
                "official_semantic_include": True,
                "official_feature_type": "numeric",
                "official_encoding": "numeric_standardized",
                "official_semantic_role": "derived_covariate",
                "official_value_transform_id": "derive_age_at_target_from_birth_year_and_current_target_date",
                "official_support_threshold_participants": 50,
                "official_support_unit": "distinct_fit_participants_with_valid_derived_feature",
                "official_support_application": "recomputed_per_scenario_and_per_temporal_block_on_fit_data",
                "official_statistical_removal_rule": "rule",
                "official_decision_origin": "A06_frozen_derive_later_directive",
                "official_decision_reason": "frozen",
                "official_execution_stage": "BENCHMARK05b",
            }
        ]
    )

    first = canonical_schema_payload(
        schema,
        {
            "b": "2",
            "a": "1",
        },
    )

    second = canonical_schema_payload(
        schema,
        {
            "a": "1",
            "b": "2",
        },
    )

    assert first == second


def test_age_at_target_transform_registry() -> None:
    schema = pd.DataFrame(
        [
            {
                "source_name": "PTDEMOG",
                "column_name": "AGE_AT_TARGET",
                "full_column_name": "PTDEMOG__AGE_AT_TARGET",
                "official_semantic_include": True,
                "official_value_transform_id": "derive_age_at_target_from_birth_year_and_current_target_date",
                "official_execution_stage": "BENCHMARK05b",
            }
        ]
    )

    legacy_transforms = pd.DataFrame(
        columns=[
            "source_name",
            "column_name",
            "full_column_name",
            "transform_id",
            "phase_required_for_transform",
            "phase_exposed_to_model",
            "parameters_json",
        ]
    )

    checks = Checks()

    registry = build_transform_registry(
        schema,
        legacy_transforms,
        "TESTDECISION",
        checks,
    )

    assert checks.passed
    assert len(registry) == 1
    assert registry.loc[
        0,
        "transform_id",
    ] == "derive_age_at_target_from_birth_year_and_current_target_date"
    assert registry.loc[
        0,
        "phase_exposed_to_model",
    ] is False or not bool(
        registry.loc[
            0,
            "phase_exposed_to_model",
        ]
    )
