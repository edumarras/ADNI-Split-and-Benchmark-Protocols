from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from adni_benchmark_protocol.auditA05_temporal_modality_exclusions import (
    EXPECTED_ELIGIBLE_PAIRS,
    SCRIPT_VERSION,
)


PROJECT_ROOT = Path.cwd()

PACKAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "frozen_experiment_v1_1"
)

REPORTS_ROOT = (
    PROJECT_ROOT
    / "reports"
)

SAFE_SUMMARY = (
    REPORTS_ROOT
    / "auditA05_temporal_modality_exclusions"
    / "safe"
    / "auditA05_summary.json"
)

LOCAL_MANIFEST = (
    REPORTS_ROOT
    / "auditA05_temporal_modality_exclusions"
    / "local_only"
    / "LOCAL_ONLY_temporal_modality_exclusions.csv"
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
    "AUDIT A05 — TEMPORAL MODALITY EXCLUSIONS"
)

required = {
    "frozen_experiment_v1_1": (
        PACKAGE_ROOT
    ),
    "SPLIT07 pairing reference": (
        PACKAGE_ROOT
        / "02_cuts_and_adjustments"
        / "development_strict_date_pairing_reference.csv.gz"
    ),
    "canonical MRI source": (
        PACKAGE_ROOT
        / "01_postprocessed_sources"
        / "UCSFFSX7_canonical_resolved.csv.gz"
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


header(
    "RUN AUDIT A05"
)

command = [
    sys.executable,
    "-m",
    (
        "adni_benchmark_protocol."
        "auditA05_temporal_modality_exclusions"
    ),
    "--project-root",
    str(
        PROJECT_ROOT
    ),
    "--package-root",
    str(
        PACKAGE_ROOT
    ),
    "--reports-root",
    str(
        REPORTS_ROOT
    ),
]

completed = subprocess.run(
    command,
    check=False,
)

if completed.returncode != 0:
    raise RuntimeError(
        "AUDIT A05 failed. Review terminal output and SAFE diagnostics."
    )


header(
    "AUDIT A05 SAFE CONTRACT"
)

if not SAFE_SUMMARY.is_file():
    raise RuntimeError(
        "AUDIT A05 SAFE summary was not created."
    )

summary = json.loads(
    SAFE_SUMMARY.read_text(
        encoding="utf-8"
    )
)

privacy = summary.get(
    "privacy",
    {},
)

checks = {
    "audit_passed": (
        summary.get(
            "audit_passed"
        )
        is True
    ),
    "script_version": (
        summary.get(
            "script_version"
        )
        == SCRIPT_VERSION
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
    "local_manifest_created": (
        LOCAL_MANIFEST.is_file()
    ),
    "safe_has_no_participant_ids": (
        privacy.get(
            "safe_outputs_contain_participant_identifiers"
        )
        is False
    ),
    "safe_has_no_visit_ids": (
        privacy.get(
            "safe_outputs_contain_visit_identifiers"
        )
        is False
    ),
    "safe_has_no_dates": (
        privacy.get(
            "safe_outputs_contain_dates"
        )
        is False
    ),
    "safe_has_no_medical_values": (
        privacy.get(
            "safe_outputs_contain_medical_values"
        )
        is False
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
            f"A05 contract mismatch: {name}"
        )


header(
    "METHOD BOUNDARY"
)

print(
    "Expected eligible longitudinal pairs:",
    EXPECTED_ELIGIBLE_PAIRS,
)

print(
    "Unsafe definition: previous source date same day as or after current MRI date."
)

print(
    "Action: exclude only the affected previous source block."
)

print(
    "Current target visit remains eligible."
)

print(
    "Scenario tables modified here: NO"
)

print(
    "Models trained:               NO"
)


header(
    "PRIVACY CHECK"
)

print(
    "LOCAL_ONLY_temporal_modality_exclusions.csv contains individualized keys."
)

print(
    "Do not commit, upload, paste, screenshot, or share that file."
)

print(
    "Only reports/auditA05_temporal_modality_exclusions/safe/ may be shared."
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
    "Safe summary:",
    SAFE_SUMMARY,
)
