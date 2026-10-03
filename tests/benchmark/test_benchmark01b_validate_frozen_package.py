from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from adni_benchmark_protocol.benchmark01b_validate_frozen_package import (
    Checks,
    EXPECTED_TEMPORAL_POLICY_DECISION_ID,
    EXPECTED_TEMPORAL_POLICY_NAME,
    canonical_json_sha256,
    catalog_sha256,
    mapping_sha256,
    resolve_paths,
    sha256_file,
    validate_temporal_policy_lock,
)


def test_catalog_sha256_normalizes_case_and_whitespace() -> None:
    assert catalog_sha256([" aa ", "Bb"]) == catalog_sha256(["AA", "BB"])


def test_mapping_sha256_is_order_sensitive() -> None:
    first = pd.DataFrame({"RID": ["XX01", "XX02"], "split": ["train", "validation"]})
    second = first.iloc[::-1].reset_index(drop=True)
    assert mapping_sha256(first) != mapping_sha256(second)


def test_resolve_paths_uses_new_protocol_package_location(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(package_root=None, reports_root=None)
    paths = resolve_paths(args)
    assert paths.package == tmp_path / "data" / "processed" / "frozen_experiment_v1_1"
    assert paths.safe == tmp_path / "reports" / "benchmark01b_validate_frozen_package" / "safe"


def build_policy_fixture(tmp_path: Path):
    participant_split = tmp_path / "official_participant_split.csv"
    anchor = tmp_path / "supervised_anchor_visits.csv.gz"
    participant_split.write_text("RID,split\nXX01,train\n", encoding="utf-8")
    anchor.write_bytes(b"synthetic-anchor")

    base_payload = {
        "stage": "split07_freeze_strict_date_temporal_extension",
        "script_version": "0.1.0",
        "policy_name": EXPECTED_TEMPORAL_POLICY_NAME,
        "policy_scope": {
            "decision_splits": ["train", "validation"],
            "test_values_used": False,
            "mri_target_values_loaded": False,
            "participant_split_changed": False,
            "supervised_population_changed": False,
        },
        "rules": {
            "exact_visit_key": ["RID", "VISCODE2"],
            "visits_concatenated": False,
            "base_prior_rule": "protocol_month strictly less than current target protocol_month",
            "same_wave_tie_rule": "synthetic",
            "current_boundary_rule": "synthetic",
            "median_date_fallback": False,
            "viscode_hierarchy_fallback": False,
            "lexical_fallback": False,
            "previous_mri_required": False,
            "previous_target_complete_required": False,
            "unresolved_action": "exclude participant from matched scenarios only",
        },
        "frozen_development_evidence": {
            "ambiguous": {"train": 269, "validation": 58},
            "strict_date_recovered": {"train": 256, "validation": 55},
            "final_eligible": {"train": 1426, "validation": 301},
        },
        "upstream": {
            "base_package": "frozen_experiment_v1",
            "split06_summary_sha256": "synthetic",
            "official_participant_split_sha256": sha256_file(participant_split),
            "supervised_anchor_sha256": sha256_file(anchor),
        },
    }
    payload_sha = canonical_json_sha256(base_payload)
    policy = dict(base_payload)
    policy["temporal_policy_decision_id"] = EXPECTED_TEMPORAL_POLICY_DECISION_ID
    policy["reconstruction_provenance"] = {
        "historical_decision_id_preserved": True,
        "current_policy_payload_sha256": payload_sha,
        "current_policy_payload_short_id": payload_sha[:20],
        "reason": "synthetic",
    }
    manifest = {
        "temporal_extension": {
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "policy": EXPECTED_TEMPORAL_POLICY_NAME,
            "reconstruction_policy_payload_sha256": payload_sha,
        }
    }
    artifacts = {
        "participant_split": participant_split,
        "anchor": anchor,
    }
    return policy, manifest, artifacts


def test_policy_lock_accepts_preserved_historical_id_and_reconstruction_hash(tmp_path) -> None:
    policy, manifest, artifacts = build_policy_fixture(tmp_path)
    checks = Checks()
    result = validate_temporal_policy_lock(policy, manifest, artifacts, checks)
    assert checks.passed
    assert result["temporal_policy_decision_id"] == EXPECTED_TEMPORAL_POLICY_DECISION_ID
    assert result["recomputed_policy_payload_sha256"] == policy["reconstruction_provenance"]["current_policy_payload_sha256"]


def test_policy_lock_detects_payload_tampering(tmp_path) -> None:
    policy, manifest, artifacts = build_policy_fixture(tmp_path)
    policy["rules"]["median_date_fallback"] = True
    checks = Checks()
    validate_temporal_policy_lock(policy, manifest, artifacts, checks)
    failed = {item.name for item in checks.items if not item.passed}
    assert "policy_reconstruction_payload_sha256" in failed
    assert "policy_rules_no_median_fallback" in failed
