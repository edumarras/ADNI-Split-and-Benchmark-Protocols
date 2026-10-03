from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from adni_benchmark_protocol.benchmark08b_train_validate_models import (
    MODELS,
    SCENARIOS,
    TARGET_REPRESENTATIONS,
)


PROJECT_ROOT = Path.cwd()

REPORTS_ROOT = (
    PROJECT_ROOT
    / "reports"
)

B06_SUMMARY = (
    REPORTS_ROOT
    / "benchmark06b"
    / "safe"
    / "benchmark06b_summary.json"
)

B07_SUMMARY = (
    REPORTS_ROOT
    / "benchmark07b"
    / "safe"
    / "benchmark07b_summary.json"
)

B08_SUMMARY = (
    REPORTS_ROOT
    / "benchmark08b"
    / "safe"
    / "benchmark08b_summary.json"
)

CHECKPOINTS = (
    REPORTS_ROOT
    / "benchmark08b"
    / "safe"
    / "checkpoints"
)

MODEL_GRID = (
    PROJECT_ROOT
    / "config"
    / "benchmark_model_grid_b.json"
)

SELECTION_POLICY = (
    PROJECT_ROOT
    / "config"
    / "benchmark_model_selection_policy_b.json"
)


def header(
    title: str,
) -> None:
    print()
    print(
        "=" * 80
    )
    print(
        title
    )
    print(
        "=" * 80
    )


header(
    "BENCHMARK08B — TRAIN / VALIDATE MODELS"
)

required = {
    "benchmark06b summary": B06_SUMMARY,
    "benchmark07b summary": B07_SUMMARY,
}

for (
    label,
    path,
) in required.items():
    exists = path.is_file()

    print(
        f"{label:36s}",
        (
            "PASS"
            if exists
            else "MISSING"
        ),
    )

    if not exists:
        raise RuntimeError(
            f"Required artifact not found: {path}"
        )


b06 = json.loads(
    B06_SUMMARY.read_text(
        encoding="utf-8"
    )
)

b07 = json.loads(
    B07_SUMMARY.read_text(
        encoding="utf-8"
    )
)

if b06.get(
    "preprocessing_passed"
) is not True:
    raise RuntimeError(
        "BENCHMARK06B is not PASS."
    )

if b07.get(
    "target_preprocessing_passed"
) is not True:
    raise RuntimeError(
        "BENCHMARK07B is not PASS."
    )


header(
    "RUN BENCHMARK08B"
)

command = [
    sys.executable,
    "-m",
    (
        "adni_benchmark_protocol."
        "benchmark08b_train_validate_models"
    ),
    "--project-root",
    str(
        PROJECT_ROOT
    ),
    "--reports-root",
    str(
        REPORTS_ROOT
    ),
]

existing_checkpoints = (
    CHECKPOINTS.is_dir()
    and any(
        CHECKPOINTS.rglob(
            "*.json"
        )
    )
)

if existing_checkpoints:
    print(
        "Existing compatible checkpoints detected: using --resume."
    )
    command.append(
        "--resume"
    )

completed = subprocess.run(
    command,
    check=False,
)

if completed.returncode != 0:
    raise RuntimeError(
        "BENCHMARK08B failed. Review terminal output and SAFE diagnostics."
    )


header(
    "BENCHMARK08B SAFE CONTRACT"
)

if not B08_SUMMARY.is_file():
    raise RuntimeError(
        "BENCHMARK08B SAFE summary was not created."
    )

summary = json.loads(
    B08_SUMMARY.read_text(
        encoding="utf-8"
    )
)

scope = summary.get(
    "scope",
    {},
)

progress = summary.get(
    "progress",
    {},
)

checks = {
    "validation_passed": (
        summary.get(
            "validation_only_passed"
        )
        is True
    ),
    "model_selection_complete": (
        summary.get(
            "model_selection_complete"
        )
        is True
    ),
    "validate_only_false": (
        summary.get(
            "validate_only"
        )
        is False
    ),
    "failed_check_count": (
        int(
            summary.get(
                "failed_check_count",
                -1,
            )
        )
        == 0
    ),
    "fatal_error_absent": (
        summary.get(
            "fatal_error"
        )
        is None
    ),
    "preprocessing_decision_chain": (
        str(
            summary.get(
                "preprocessing_decision_id",
                "",
            )
        )
        == str(
            b06.get(
                "preprocessing_decision_id",
                "",
            )
        )
    ),
    "target_decision_chain": (
        str(
            summary.get(
                "target_decision_id",
                "",
            )
        )
        == str(
            b07.get(
                "target_decision_id",
                "",
            )
        )
    ),
    "scenario_decision_chain": (
        str(
            summary.get(
                "scenario_decision_id",
                "",
            )
        )
        == str(
            b07.get(
                "scenario_decision_id",
                "",
            )
        )
    ),
    "test_values_unused": (
        scope.get(
            "test_values_used"
        )
        is False
    ),
    "test_not_materialized": (
        scope.get(
            "test_materialized"
        )
        is False
    ),
    "predictions_not_saved": (
        scope.get(
            "predictions_saved"
        )
        is False
    ),
    "individual_errors_not_saved": (
        scope.get(
            "individual_errors_saved"
        )
        is False
    ),
    "bootstrap_deferred": (
        scope.get(
            "bootstrap_performed"
        )
        is False
    ),
    "final_lock_not_created": (
        scope.get(
            "final_configuration_lock_created"
        )
        is False
    ),
    "completed_all_configurations": (
        int(
            progress.get(
                "completed_configurations",
                -1,
            )
        )
        == int(
            progress.get(
                "expected_configurations",
                -2,
            )
        )
        == 318
    ),
    "selected_blocks_complete": (
        int(
            progress.get(
                "selected_blocks_available",
                -1,
            )
        )
        == int(
            progress.get(
                "expected_selected_blocks",
                -2,
            )
        )
        == (
            len(
                SCENARIOS
            )
            * len(
                TARGET_REPRESENTATIONS
            )
            * len(
                MODELS
            )
        )
    ),
    "model_grid_written": (
        MODEL_GRID.is_file()
    ),
    "selection_policy_written": (
        SELECTION_POLICY.is_file()
    ),
}

for (
    name,
    passed,
) in checks.items():
    print(
        f"{name:44s}",
        (
            "PASS"
            if passed
            else "FAIL"
        ),
    )

    if not passed:
        raise RuntimeError(
            f"BENCHMARK08B contract mismatch: {name}"
        )


header(
    "METHOD BOUNDARY"
)

print(
    "Fit split:                              train"
)

print(
    "Model selection split:                  validation"
)

print(
    "Primary selection metric:               participant-macro RMSE"
)

print(
    "Metric space:                           reconstructed standardized Full 323 targets"
)

print(
    "Target representations:                 Full + PCA90"
)

print(
    "Models:                                 mean, median, Ridge, MultiTask Elastic Net, PLS, Extra Trees, KNN"
)

print(
    "Test read/materialized/evaluated:       NO"
)

print(
    "Individual predictions saved:           NO"
)

print(
    "Final test configuration lock:          NOT YET"
)


header(
    "PIPELINE STATUS"
)

print(
    "frozen_experiment_v1_1                  PASS"
)

print(
    "benchmark01b_validate_frozen_package     PASS"
)

print(
    "auditA06_freeze_crosschecked_inputs      PASS"
)

print(
    "crosschecked_inputs_v1                   PASS"
)

print(
    "benchmark02b_profile_input_features       PASS"
)

print(
    "benchmark04b_finalize_feature_schema      PASS"
)

print(
    "auditA05_temporal_modality_exclusions     PASS"
)

print(
    "benchmark05b_build_scenarios               PASS"
)

print(
    "benchmark06b_fit_input_preprocessing       PASS"
)

print(
    "benchmark07b_fit_target_representations    PASS"
)

print(
    "benchmark08b_train_validate_models         PASS"
)

print(
    "Safe summary:",
    B08_SUMMARY,
)

print(
    "Model grid decision ID:",
    summary.get(
        "model_grid_decision_id"
    ),
)
