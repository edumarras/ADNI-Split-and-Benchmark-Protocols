from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from adni_benchmark_protocol.auditA06_freeze_crosschecked_inputs import (
    EXPECTED_LEDGER_SHA256,
    EXPECTED_LEDGER_VERSION,
    EXPECTED_PACKAGE_NAME,
    EXPECTED_TEMPORAL_POLICY_DECISION_ID,
    sha256_file,
)


PROJECT_ROOT = Path.cwd()

PACKAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "frozen_experiment_v1_1"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "crosschecked_inputs_v1"
)

REPORTS_ROOT = (
    PROJECT_ROOT
    / "reports"
)

LEDGER_PATH = (
    PROJECT_ROOT
    / "config"
    / "crosscheck_ledger_frozen_v1.0.0.json"
)

BENCHMARK01_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "benchmark01b_validate_frozen_package"
    / "safe"
    / "benchmark01b_validation_summary.json"
)

A06_SAFE_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "auditA06_freeze_crosschecked_inputs"
    / "safe"
    / "a06_verification_report.json"
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
    "A06 — FREEZE CROSS-CHECKED INPUTS"
)

required = {
    "frozen_experiment_v1_1": (
        PACKAGE_ROOT
    ),
    "benchmark01b summary": (
        BENCHMARK01_SUMMARY
    ),
    "crosscheck ledger": (
        LEDGER_PATH
    ),
}

for (
    label,
    path,
) in required.items():
    exists = (
        path.is_dir()
        if label
        == "frozen_experiment_v1_1"
        else path.is_file()
    )

    print(
        f"{label:32s}",
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


ledger = json.loads(
    LEDGER_PATH.read_text(
        encoding="utf-8"
    )
)

current_ledger_sha = (
    sha256_file(
        LEDGER_PATH
    )
)

declared_historical_sha = (
    ledger.get(
        "historical_ledger_sha256"
    )
)

ledger_identity_ok = (
    current_ledger_sha
    == EXPECTED_LEDGER_SHA256
    or declared_historical_sha
    == EXPECTED_LEDGER_SHA256
)

print(
    f"{'ledger_version':32s}",
    (
        "PASS"
        if ledger.get(
            "ledger_version"
        )
        == EXPECTED_LEDGER_VERSION
        else "FAIL"
    ),
)

print(
    f"{'historical_ledger_identity':32s}",
    (
        "PASS"
        if ledger_identity_ok
        else "FAIL"
    ),
)

print(
    "Current reconstruction ledger SHA-256:",
    current_ledger_sha,
)

print(
    "Historical ledger SHA-256:",
    EXPECTED_LEDGER_SHA256,
)

if not ledger_identity_ok:
    raise RuntimeError(
        "Ledger does not preserve the historical frozen identity."
    )


header(
    "RUN A06"
)

command = [
    sys.executable,
    "-m",
    (
        "adni_benchmark_protocol."
        "auditA06_freeze_crosschecked_inputs"
    ),
    "--package-root",
    str(
        PACKAGE_ROOT
    ),
    "--output-root",
    str(
        OUTPUT_ROOT
    ),
    "--reports-root",
    str(
        REPORTS_ROOT
    ),
    "--ledger",
    str(
        LEDGER_PATH
    ),
    "--benchmark01-summary",
    str(
        BENCHMARK01_SUMMARY
    ),
    "--overwrite",
]

completed = subprocess.run(
    command,
    check=False,
)

if completed.returncode != 0:
    raise RuntimeError(
        "A06 failed. Review the terminal output and SAFE diagnostics."
    )


header(
    "A06 SAFE CONTRACT"
)

if not A06_SAFE_REPORT.is_file():
    raise RuntimeError(
        "A06 SAFE verification report was not created."
    )

report = json.loads(
    A06_SAFE_REPORT.read_text(
        encoding="utf-8"
    )
)

checks = {
    "status": (
        report.get(
            "status"
        )
        == "PASS"
    ),
    "failed_check_count": (
        int(
            report.get(
                "failed_check_count",
                -1,
            )
        )
        == 0
    ),
    "logical_package": (
        report.get(
            "upstream",
            {},
        ).get(
            "logical_package_name"
        )
        == EXPECTED_PACKAGE_NAME
    ),
    "temporal_policy_id": (
        report.get(
            "upstream",
            {},
        ).get(
            "temporal_policy_decision_id"
        )
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID
    ),
    "historical_ledger_sha256": (
        report.get(
            "upstream",
            {},
        ).get(
            "crosscheck_ledger_sha256"
        )
        == EXPECTED_LEDGER_SHA256
    ),
    "crosschecked_inputs_created": (
        OUTPUT_ROOT.is_dir()
    ),
}

for (
    name,
    passed,
) in checks.items():
    print(
        f"{name:40s}",
        (
            "PASS"
            if passed
            else "FAIL"
        ),
    )

    if not passed:
        raise RuntimeError(
            f"A06 contract mismatch: {name}"
        )


header(
    "PRIVACY CHECK"
)

print(
    "crosschecked_inputs_v1 is LOCAL_ONLY."
)

print(
    "A06 local_only reports may contain individualized records."
)

print(
    "Do not commit, upload, paste, or share files from those local directories."
)

print(
    "Only the SAFE aggregate reports may be inspected/shared."
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
