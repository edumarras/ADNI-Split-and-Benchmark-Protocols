#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK05B — construção auditada dos cenários de desenvolvimento.
#
# Objetivo
# --------
# Materializar, usando somente o pacote congelado e as decisões oficiais do
# BENCHMARK04B, os três cenários do benchmark:
#
# 1. snapshot_all
#    - todas as visitas-alvo de treino e validação;
#    - entradas não-MRI da própria visita;
#    - PTDEMOG/APOERES anexadas uma única vez por participante.
#
# 2. snapshot_matched_last
#    - uma visita-alvo atual por participante longitudinalmente elegível;
#    - a visita atual é a última visita target-complete segundo a ordenação
#      congelada do pacote;
#    - usa somente entradas não-MRI atuais.
#
# 3. longitudinal_previous_last
#    - exatamente os mesmos participantes, visitas atuais e targets de
#      snapshot_matched_last;
#    - acrescenta uma única visita não-MRI anterior comum, estritamente anterior;
#    - não busca uma visita histórica diferente para cada modalidade;
#    - não faz carry-forward;
#    - PTDEMOG/APOERES entram apenas uma vez;
#    - acrescenta gap_months.
#
# A visita anterior não precisa possuir MRI nem target completo. Ela precisa ter
# pelo menos uma feature visit-level semanticamente autorizada originalmente
# observada e ser a visita exata congelada pela política temporal do SPLIT07.
# Empates dentro da mesma onda protocolar são resolvidos somente quando intervalos
# de datas clínicas provam uma candidata estritamente posterior às concorrentes e
# estritamente anterior à visita-alvo atual. Visitas nunca são concatenadas.
#
# Privacidade e vazamento
# -----------------------
# - O estágio usa somente treino e validação para construir cenários, calcular
#   suporte e auditar transformações.
# - Nenhum valor, pareamento ou disponibilidade do teste é usado.
# - Nenhum target MRI é carregado; apenas as chaves/metadados da tabela-alvo são
#   usados para confirmar alinhamento.
# - Artefatos com RID/VISCODE2 são escritos apenas em data/processed_data e são
#   explicitamente LOCAL ONLY.
# - Relatórios safe contêm apenas contagens, nomes de features, hashes, regras e
#   estatísticas agregadas.
# - Nenhum modelo é treinado.

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "benchmark05b"
PACKAGE_NAME: Final[str] = "frozen_experiment_v1_1"
SCENARIO_VERSION: Final[str] = "2.0.0-crosschecked-b"
SUPPORT_THRESHOLD: Final[int] = 50
TARGET_COUNT: Final[int] = 323

RID: Final[str] = "RID"
VISCODE2: Final[str] = "VISCODE2"
SPLIT: Final[str] = "split"
TRAIN: Final[str] = "train"
VALIDATION: Final[str] = "validation"
DEVELOPMENT_SPLITS: Final[tuple[str, ...]] = (TRAIN, VALIDATION)

VISIT_SOURCES: Final[tuple[str, ...]] = (
    "ADAS",
    "CDR",
    "FAQ",
    "MMSE",
    "MOCA",
    "NEUROBAT",
)
PARTICIPANT_SOURCES: Final[tuple[str, ...]] = ("PTDEMOG", "APOERES")
INPUT_SOURCES: Final[tuple[str, ...]] = (*VISIT_SOURCES, *PARTICIPANT_SOURCES)

SOURCE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    **{source: (RID, VISCODE2) for source in VISIT_SOURCES},
    **{source: (RID,) for source in PARTICIPANT_SOURCES},
}

ARTIFACTS: Final[dict[str, str]] = {
    "anchor": "03_final_ready_tables/supervised_anchor_visits.csv.gz",
    "targets": "03_final_ready_tables/UCSFFSX7_targets_323.csv.gz",
    "target_catalog": "03_final_ready_tables/MRI_target_catalog_323.csv",
    "participant_split": "04_official_split/official_participant_split.csv",
}

TEMPORAL_PAIRING_REFERENCE: Final[str] = (
    "02_cuts_and_adjustments/development_strict_date_pairing_reference.csv.gz"
)
TEMPORAL_POLICY_LOCK: Final[str] = (
    "04_official_split/strict_date_temporal_policy_lock.json"
)
PACKAGE_MANIFEST: Final[str] = "04_official_split/official_frozen_manifest.json"
AUDIT_A05_SUMMARY: Final[str] = (
    "reports/auditA05_temporal_modality_exclusions/safe/auditA05_summary.json"
)
AUDIT_A05_LOCAL_MANIFEST: Final[str] = (
    "reports/auditA05_temporal_modality_exclusions/local_only/"
    "LOCAL_ONLY_temporal_modality_exclusions.csv"
)
EXPECTED_AUDIT_A05_SCRIPT_VERSION: Final[str] = "0.1.1"
TEMPORAL_EXCLUSION_FLAG_PREFIX: Final[str] = "exclude_previous__"
ALLOWED_TEMPORAL_EXCLUSION_RELATIONS: Final[frozenset[str]] = frozenset(
    {"same_day_as_current_mri", "after_current_mri"}
)
EXPECTED_TEMPORAL_EXCLUSION_ACTION: Final[str] = (
    "exclude_previous_source_block"
)
EXPECTED_TEMPORAL_POLICY_NAME: Final[str] = (
    "strict_date_interval_same_wave_previous_visit_v1"
)
EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = "90fefe5ec34d49743612"
EXPECTED_DEVELOPMENT_PARTICIPANTS: Final[dict[str, int]] = {
    TRAIN: 1_987,
    VALIDATION: 426,
}
EXPECTED_FINAL_ELIGIBLE: Final[dict[str, int]] = {
    TRAIN: 1_426,
    VALIDATION: 301,
}
EXPECTED_PAIRING_REASON_COUNTS: Final[dict[str, dict[str, int]]] = {
    # SPLIT07 materializes the two no-usable-history subcases under one
    # frozen exclusion label. The aggregate totals are unchanged:
    # train 536 + 12 = 548; validation 119 + 3 = 122.
    TRAIN: {
        "eligible_unique_previous_visit": 1_170,
        "eligible_strict_date_interval_winner": 256,
        "ambiguous_same_wave_no_strict_interval_winner": 9,
        "strict_latest_not_fully_before_current": 4,
        "only_unknown_or_non_earlier_history": 548,
    },
    VALIDATION: {
        "eligible_unique_previous_visit": 246,
        "eligible_strict_date_interval_winner": 55,
        "ambiguous_same_wave_no_strict_interval_winner": 2,
        "strict_latest_not_fully_before_current": 1,
        "only_unknown_or_non_earlier_history": 122,
    },
}

BENCHMARK01_SUMMARY: Final[str] = (
    "reports/benchmark01b_validate_frozen_package/safe/"
    "benchmark01b_validation_summary.json"
)
BENCHMARK04B_SUMMARY: Final[str] = (
    "reports/benchmark04b/safe/benchmark04b_summary.json"
)
OFFICIAL_SCHEMA: Final[str] = "config/benchmark_feature_schema_b.csv"
OFFICIAL_TRANSFORMS: Final[str] = (
    "config/benchmark_feature_transform_registry_b.csv"
)
SUPPORT_POLICY: Final[str] = "config/benchmark_support_policy_b.json"
SCHEMA_LOCK: Final[str] = "config/benchmark_schema_lock_b.json"

SCENARIO_DEFINITIONS: Final[str] = "config/benchmark_scenario_definitions_b.json"
SCENARIO_SUPPORT_MANIFEST: Final[str] = (
    "config/benchmark_scenario_support_manifest_b.csv"
)
SCENARIO_LOCK: Final[str] = "config/benchmark_scenario_lock_b.json"

CROSSCHECKED_ROOT_RELATIVE: Final[str] = "data/processed/crosschecked_inputs_v1"
CROSSCHECKED_MANIFEST: Final[str] = "crosschecked_input_manifest.json"
CROSSCHECKED_INPUT_DIR: Final[str] = "input_sources"
AGE_METADATA_RELATIVE: Final[str] = (
    "local_only/LOCAL_ONLY_ptdemog_age_derivation_metadata.csv.gz"
)
MRI_CANONICAL_RELATIVE: Final[str] = (
    "01_postprocessed_sources/UCSFFSX7_canonical_resolved.csv.gz"
)
EXPECTED_LOGICAL_PACKAGE: Final[str] = "frozen_experiment_v1_1"
EXPECTED_B04B_VERSION: Final[str] = "0.1.2"
HISTORICAL_SCHEMA_DECISION_ID_REFERENCE: Final[str] = "52fb60da4b4c2033626c61bd"
EXPECTED_SCHEMA_ROWS: Final[int] = 362
EXPECTED_INCLUDED_FEATURES: Final[int] = 212
EXPECTED_EXCLUDED_FEATURES: Final[int] = 150
EXPECTED_TRANSFORM_ROWS: Final[int] = 16

LOCAL_OUTPUT_DIR: Final[str] = "data/processed/local_only/benchmark05b_scenarios"

MULTI_SPLIT_PATTERN: Final[re.Pattern[str]] = re.compile(r"[|;]")
YEAR_PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d{4})-01-01(?:[ T].*)?\s*$"
)

IDENTITY_TRANSFORMS: Final[set[str]] = {
    "identity",
    "numeric_identity",
    "categorical_identity",
    "binary_zero_one_identity",
}
SUPPORTED_TRANSFORMS: Final[set[str]] = IDENTITY_TRANSFORMS | {
    "faq_official_item_score_map",
    "cdr_source_mode_harmonization",
    "ptdemog_pthome_harmonization",
    "extract_year_from_jan1_date_placeholder",
    "dictionary_multi_response_tokens",
    "derive_age_at_target_from_birth_year_and_current_target_date",
}


class ScenarioAbort(RuntimeError):
    # Fail-closed exception for scenario incompatibility.
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    package: Path
    crosschecked_root: Path
    crosschecked_manifest: Path
    age_metadata: Path
    mri_canonical: Path
    safe: Path
    logs: Path
    local: Path
    benchmark01_summary: Path
    benchmark04b_summary: Path
    official_schema: Path
    official_transforms: Path
    support_policy: Path
    schema_lock: Path
    scenario_definitions: Path
    scenario_support_manifest: Path
    scenario_lock: Path
    temporal_pairing_reference: Path
    temporal_policy_lock: Path
    package_manifest: Path
    temporal_exclusion_summary: Path
    temporal_exclusion_manifest: Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: Any
    observed: Any
    details: str = ""


class Checks:
    def __init__(self) -> None:
        self.items: list[Check] = []

    def add(
        self,
        name: str,
        passed: bool,
        expected: Any,
        observed: Any,
        details: str = "",
    ) -> bool:
        item = Check(name, bool(passed), expected, observed, details)
        self.items.append(item)
        logging.log(
            logging.INFO if passed else logging.ERROR,
            "%s | %s",
            "PASS" if passed else "FAIL",
            name,
        )
        if not passed:
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                name,
                short_text(expected),
                short_text(observed),
                f" | {details}" if details else "",
            )
        return bool(passed)

    @property
    def passed(self) -> bool:
        return bool(self.items) and all(item.passed for item in self.items)

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)


def short_text(value: Any, limit: int = 600) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit snapshot_all, snapshot_matched_last and "
            "longitudinal_previous_last using the frozen BENCHMARK04B rules."
        )
    )
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument(
        "--supersede-existing",
        "--supersede_existing",
        dest="supersede_existing",
        action="store_true",
        help=(
            "Archive the previous BENCHMARK05B safe/config artifacts and replace "
            "an incompatible existing scenario lock."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path(__file__).resolve().parents[2]
    package = (
        args.package_root.resolve()
        if args.package_root is not None
        else root / "data" / "processed" / PACKAGE_NAME
    )
    reports_root = (
        args.reports_root.resolve()
        if args.reports_root is not None
        else root / "reports"
    )
    processed_root = (
        args.processed_root.resolve()
        if args.processed_root is not None
        else root / "data" / "processed" / "local_only"
    )
    stage_report = reports_root / STAGE_NAME
    crosschecked_root = root / CROSSCHECKED_ROOT_RELATIVE
    return Paths(
        root=root,
        package=package,
        crosschecked_root=crosschecked_root,
        crosschecked_manifest=crosschecked_root / CROSSCHECKED_MANIFEST,
        age_metadata=crosschecked_root / AGE_METADATA_RELATIVE,
        mri_canonical=package / MRI_CANONICAL_RELATIVE,
        safe=stage_report / "safe",
        logs=stage_report / "logs",
        local=processed_root / "benchmark05b_scenarios",
        benchmark01_summary=root / BENCHMARK01_SUMMARY,
        benchmark04b_summary=root / BENCHMARK04B_SUMMARY,
        official_schema=root / OFFICIAL_SCHEMA,
        official_transforms=root / OFFICIAL_TRANSFORMS,
        support_policy=root / SUPPORT_POLICY,
        schema_lock=root / SCHEMA_LOCK,
        scenario_definitions=root / SCENARIO_DEFINITIONS,
        scenario_support_manifest=root / SCENARIO_SUPPORT_MANIFEST,
        scenario_lock=root / SCENARIO_LOCK,
        temporal_pairing_reference=package / TEMPORAL_PAIRING_REFERENCE,
        temporal_policy_lock=package / TEMPORAL_POLICY_LOCK,
        package_manifest=package / PACKAGE_MANIFEST,
        temporal_exclusion_summary=root / AUDIT_A05_SUMMARY,
        temporal_exclusion_manifest=root / AUDIT_A05_LOCAL_MANIFEST,
    )



def archive_superseded_artifacts(
    paths: Paths,
    enabled: bool,
) -> str | None:
    # Archive prior safe/config outputs without copying individualized tables.
    if not paths.scenario_lock.is_file():
        return None

    existing = read_json(paths.scenario_lock)
    existing_id = str(existing.get("scenario_decision_id", "unknown"))
    existing_version = str(existing.get("scenario_definition_version", "unknown"))

    if not enabled:
        return None

    archive = (
        paths.root
        / "reports"
        / STAGE_NAME
        / "superseded"
        / f"scenario_{existing_id}"
    )
    archive.mkdir(parents=True, exist_ok=True)

    for path in (
        paths.scenario_lock,
        paths.scenario_definitions,
        paths.scenario_support_manifest,
    ):
        if path.is_file():
            shutil.copy2(path, archive / path.name)

    if paths.safe.is_dir():
        safe_archive = archive / "safe_reports"
        safe_archive.mkdir(parents=True, exist_ok=True)
        for path in paths.safe.iterdir():
            if path.is_file():
                shutil.copy2(path, safe_archive / path.name)

    record = {
        "superseded_scenario_decision_id": existing_id,
        "superseded_scenario_definition_version": existing_version,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "individualized_local_outputs_copied": False,
        "reason": (
            "Superseded the previous scenario build after the independent "
            "temporal audit identified visit-level source blocks collected on "
            "or after the current target MRI."
        ),
    }
    (archive / "supersession_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for path in (
        paths.scenario_lock,
        paths.scenario_definitions,
        paths.scenario_support_manifest,
    ):
        path.unlink(missing_ok=True)

    return existing_id


def load_temporal_contract(
    paths: Paths,
    checks: Checks,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "temporal_pairing_reference": paths.temporal_pairing_reference,
        "temporal_policy_lock": paths.temporal_policy_lock,
        "package_manifest": paths.package_manifest,
    }
    for name, path in required.items():
        checks.add(
            f"artifact__{name}",
            path.is_file(),
            True,
            path.is_file(),
            str(path),
        )
    if not checks.passed:
        raise ScenarioAbort("SPLIT07 temporal-extension artifacts are missing.")

    lock = read_json(paths.temporal_policy_lock)
    manifest = read_json(paths.package_manifest)
    scope = lock.get("policy_scope", {})
    rules = lock.get("rules", {})
    extension = manifest.get("temporal_extension", {})

    checks.add(
        "package_logical_name",
        manifest.get("frozen_package_name") == EXPECTED_LOGICAL_PACKAGE,
        EXPECTED_LOGICAL_PACKAGE,
        manifest.get("frozen_package_name"),
    )
    checks.add(
        "temporal_policy_name",
        lock.get("policy_name") == EXPECTED_TEMPORAL_POLICY_NAME,
        EXPECTED_TEMPORAL_POLICY_NAME,
        lock.get("policy_name"),
    )
    checks.add(
        "temporal_policy_decision_id",
        lock.get("temporal_policy_decision_id")
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        lock.get("temporal_policy_decision_id"),
    )
    checks.add(
        "temporal_policy_development_only",
        set(scope.get("decision_splits", [])) == set(DEVELOPMENT_SPLITS),
        list(DEVELOPMENT_SPLITS),
        scope.get("decision_splits"),
    )
    checks.add(
        "temporal_policy_test_unused",
        scope.get("test_values_used") is False,
        False,
        scope.get("test_values_used"),
    )
    checks.add(
        "temporal_policy_no_visit_concatenation",
        rules.get("visits_concatenated") is False,
        False,
        rules.get("visits_concatenated"),
    )
    checks.add(
        "temporal_policy_no_hierarchy_fallback",
        rules.get("viscode_hierarchy_fallback") is False,
        False,
        rules.get("viscode_hierarchy_fallback"),
    )
    checks.add(
        "manifest_temporal_decision_id",
        extension.get("temporal_policy_decision_id")
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        extension.get("temporal_policy_decision_id"),
    )
    checks.add(
        "manifest_temporal_policy_name",
        extension.get("policy") == EXPECTED_TEMPORAL_POLICY_NAME,
        EXPECTED_TEMPORAL_POLICY_NAME,
        extension.get("policy"),
    )
    checks.add(
        "manifest_split_unchanged",
        manifest.get("participant_split_changed") is False,
        False,
        manifest.get("participant_split_changed"),
    )
    checks.add(
        "manifest_supervised_population_unchanged",
        manifest.get("supervised_population_changed") is False,
        False,
        manifest.get("supervised_population_changed"),
    )
    checks.add(
        "manifest_targets_unchanged",
        manifest.get("mri_targets_changed") is False,
        False,
        manifest.get("mri_targets_changed"),
    )

    critical = manifest.get("critical_unchanged_sha256", {})
    actual_anchor_hash = sha256_file(paths.package / ARTIFACTS["anchor"])
    actual_split_hash = sha256_file(paths.package / ARTIFACTS["participant_split"])
    checks.add(
        "temporal_manifest_anchor_hash",
        actual_anchor_hash == critical.get("supervised_anchor_visits.csv.gz"),
        critical.get("supervised_anchor_visits.csv.gz"),
        actual_anchor_hash,
    )
    checks.add(
        "temporal_manifest_participant_split_hash",
        actual_split_hash == critical.get("official_participant_split.csv"),
        critical.get("official_participant_split.csv"),
        actual_split_hash,
    )

    artifact_hashes = extension.get("artifact_sha256", {})
    expected_pairing_hash = artifact_hashes.get(
        "development_strict_date_pairing_reference"
    )
    actual_pairing_hash = sha256_file(paths.temporal_pairing_reference)
    checks.add(
        "temporal_pairing_reference_hash",
        actual_pairing_hash == expected_pairing_hash,
        expected_pairing_hash,
        actual_pairing_hash,
    )

    if not checks.passed:
        raise ScenarioAbort("SPLIT07 temporal contract validation failed.")
    return lock, manifest

def configure_logging(paths: Paths) -> None:
    paths.safe.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.local.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / "benchmark05b.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScenarioAbort(f"JSON root must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    ) + "\n"
    atomic_write_text(path, text)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as raw_handle:
        temp_path = Path(raw_handle.name)
    try:
        with temp_path.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                mtime=0,
            ) as gz_output:
                csv_bytes = frame.to_csv(
                    index=False,
                    lineterminator="\n",
                ).encode("utf-8")
                gz_output.write(csv_bytes)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def read_csv_text(
    path: Path,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        dtype="string",
        keep_default_na=False,
        low_memory=False,
    )
    frame.columns = [
        str(column).strip().lstrip("\ufeff") for column in frame.columns
    ]
    for column in frame.columns:
        series = frame[column].astype("string").str.strip()
        frame[column] = series.mask(series.eq(""), pd.NA)
    return frame


def normalize_rid_key(series: pd.Series, context: str) -> pd.Series:
    raw = series.astype("string").str.strip()
    raw = raw.mask(raw.eq(""), pd.NA)
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & numeric.isna()
    non_integer = numeric.notna() & numeric.ne(numeric.round())
    if invalid.any() or non_integer.any():
        raise ScenarioAbort(
            f"Invalid RID in {context}: nonnumeric={int(invalid.sum())}, "
            f"noninteger={int(non_integer.sum())}"
        )
    return numeric.round().astype("Int64").astype("string")


def normalize_viscode_key(series: pd.Series) -> pd.Series:
    output = series.astype("string").str.strip().str.casefold()
    return output.mask(output.eq(""), pd.NA)


def normalize_split_key(series: pd.Series) -> pd.Series:
    output = series.astype("string").str.strip().str.casefold()
    return output.mask(output.eq(""), pd.NA)


def temporal_exclusion_flag(source_name: str) -> str:
    return f"{TEMPORAL_EXCLUSION_FLAG_PREFIX}{source_name}"


def apply_temporal_exclusion_to_aligned(
    scenario_name: str,
    manifest: pd.DataFrame,
    aligned: pd.DataFrame,
    source_name: str,
    block: str,
    features: Sequence[str],
) -> pd.DataFrame:
    if not (
        scenario_name == "longitudinal_previous_last"
        and block == "previous_visit"
        and source_name in VISIT_SOURCES
    ):
        return aligned

    flag_column = temporal_exclusion_flag(source_name)
    require_columns(
        manifest,
        [flag_column],
        f"{scenario_name}/{block}/{source_name} temporal flags",
    )
    mask = manifest[flag_column].map(parse_bool).to_numpy(dtype=bool)
    if len(mask) != len(aligned):
        raise ScenarioAbort(
            f"Temporal exclusion mask length mismatch for {source_name}: "
            f"{len(mask)} vs {len(aligned)}"
        )
    output = aligned.copy()
    if mask.any():
        output.loc[mask, list(features)] = pd.NA
    return output


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ScenarioAbort(f"{label} is missing required columns: {missing}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise ScenarioAbort(f"Cannot interpret boolean value: {value!r}")


def normalize_code(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        return text
    if float(number).is_integer():
        return str(int(number))
    return format(number, ".15g")


def parse_multi_tokens(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [
        token.strip()
        for token in MULTI_SPLIT_PATTERN.split(text)
        if token.strip()
    ]


def parse_protocol_month(value: Any) -> float:
    # Use exactly the frozen split parser for protocol-wave ordering.
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().casefold()
    if text in {"bl", "sc", "scmri", "m00", "v01"}:
        return 0.0
    month_match = re.fullmatch(r"m(\d+)", text)
    if month_match:
        return float(int(month_match.group(1)))
    year_match = re.fullmatch(r"y(\d+)", text)
    if year_match:
        return float(int(year_match.group(1)) * 12)
    return np.nan


def parse_clinical_dates(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.eq(""), pd.NA)
    try:
        return pd.to_datetime(text, errors="coerce", format="mixed").dt.normalize()
    except (TypeError, ValueError):
        return pd.to_datetime(text, errors="coerce").dt.normalize()



def validate_upstream(
    paths: Paths,
    checks: Checks,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required_paths = {
        "package_root": paths.package,
        "crosschecked_root": paths.crosschecked_root,
        "crosschecked_manifest": paths.crosschecked_manifest,
        "age_metadata": paths.age_metadata,
        "mri_canonical": paths.mri_canonical,
        "benchmark01_summary": paths.benchmark01_summary,
        "benchmark04b_summary": paths.benchmark04b_summary,
        "official_schema_b": paths.official_schema,
        "official_transforms_b": paths.official_transforms,
        "support_policy_b": paths.support_policy,
        "schema_lock_b": paths.schema_lock,
    }
    for name, path in required_paths.items():
        checks.add(f"artifact__{name}", path.exists(), True, path.exists(), str(path))
    if not checks.passed:
        raise ScenarioAbort("Required upstream artifacts are missing.")

    benchmark01 = read_json(paths.benchmark01_summary)
    benchmark04b = read_json(paths.benchmark04b_summary)
    support_policy = read_json(paths.support_policy)
    schema_lock = read_json(paths.schema_lock)
    crosschecked_manifest = read_json(paths.crosschecked_manifest)

    checks.add(
        "benchmark01_passed",
        bool(benchmark01.get("validation_passed", False)),
        True,
        benchmark01.get("validation_passed"),
    )
    checks.add(
        "benchmark04b_passed",
        str(benchmark04b.get("status", "")).upper() == "PASS",
        "PASS",
        benchmark04b.get("status"),
    )
    checks.add(
        "benchmark04b_version",
        str(benchmark04b.get("script_version")) == EXPECTED_B04B_VERSION,
        EXPECTED_B04B_VERSION,
        benchmark04b.get("script_version"),
    )
    checks.add(
        "benchmark04b_no_failed_checks",
        int(benchmark04b.get("failed_check_count", -1)) == 0,
        0,
        benchmark04b.get("failed_check_count"),
    )
    observed_schema_decision_id = str(
        benchmark04b.get("schema_decision_id", "")
    ).strip()
    policy_schema_decision_id = str(
        support_policy.get("official_decision_id", "")
    ).strip()
    lock_schema_decision_id = str(
        schema_lock.get("official_decision_id", "")
    ).strip()

    checks.add(
        "benchmark04b_schema_decision_id_nonempty",
        bool(observed_schema_decision_id),
        True,
        bool(observed_schema_decision_id),
    )
    checks.add(
        "benchmark04b_schema_decision_chain_consistent",
        bool(observed_schema_decision_id)
        and len(
            {
                observed_schema_decision_id,
                policy_schema_decision_id,
                lock_schema_decision_id,
            }
        )
        == 1,
        True,
        {
            "benchmark04b": observed_schema_decision_id,
            "support_policy": policy_schema_decision_id,
            "schema_lock": lock_schema_decision_id,
        },
    )
    checks.add(
        "support_threshold_frozen_50",
        int(support_policy.get("selected_support_threshold_participants", -1))
        == SUPPORT_THRESHOLD,
        SUPPORT_THRESHOLD,
        support_policy.get("selected_support_threshold_participants"),
    )
    checks.add(
        "schema_lock_threshold_50",
        int(schema_lock.get("selected_support_threshold_participants", -1))
        == SUPPORT_THRESHOLD,
        SUPPORT_THRESHOLD,
        schema_lock.get("selected_support_threshold_participants"),
    )
    checks.add(
        "support_filter_removes_zero_visits",
        support_policy.get("support_filter_removes_visits") is False,
        False,
        support_policy.get("support_filter_removes_visits"),
    )
    checks.add(
        "official_decision_id_consistent",
        len(
            {
                str(benchmark04b.get("schema_decision_id", "")),
                str(support_policy.get("official_decision_id", "")),
                str(schema_lock.get("official_decision_id", "")),
            }
        )
        == 1,
        True,
        {
            "benchmark04b": benchmark04b.get("schema_decision_id"),
            "policy": support_policy.get("official_decision_id"),
            "lock": schema_lock.get("official_decision_id"),
        },
    )
    checks.add(
        "schema_lock_logical_package",
        str(schema_lock.get("logical_frozen_package")) == EXPECTED_LOGICAL_PACKAGE,
        EXPECTED_LOGICAL_PACKAGE,
        schema_lock.get("logical_frozen_package"),
    )
    checks.add(
        "schema_lock_temporal_policy",
        str(schema_lock.get("temporal_policy_decision_id"))
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        schema_lock.get("temporal_policy_decision_id"),
    )
    for key in (
        "participant_split_changed",
        "supervised_population_changed",
        "mri_targets_changed",
        "temporal_policy_changed",
        "support_threshold_changed",
    ):
        observed = schema_lock.get("invariants", {}).get(key)
        checks.add(f"schema_lock_invariant__{key}", observed is False, False, observed)

    official_hashes = schema_lock.get("artifacts", {})
    actual_hashes = {
        "benchmark_feature_schema_b.csv": sha256_file(paths.official_schema),
        "benchmark_feature_transform_registry_b.csv": sha256_file(paths.official_transforms),
        "benchmark_support_policy_b.json": sha256_file(paths.support_policy),
    }
    checks.add(
        "official_artifact_hashes_match_lock",
        official_hashes == actual_hashes,
        official_hashes,
        actual_hashes,
    )

    checks.add(
        "crosschecked_manifest_stage",
        str(crosschecked_manifest.get("stage")) == "auditA06_freeze_crosschecked_inputs",
        "auditA06_freeze_crosschecked_inputs",
        crosschecked_manifest.get("stage"),
    )
    checks.add(
        "crosschecked_manifest_output_name",
        str(crosschecked_manifest.get("output_name")) == "crosschecked_inputs_v1",
        "crosschecked_inputs_v1",
        crosschecked_manifest.get("output_name"),
    )
    checks.add(
        "crosschecked_manifest_package",
        str(crosschecked_manifest.get("upstream_frozen_package_name"))
        == EXPECTED_LOGICAL_PACKAGE,
        EXPECTED_LOGICAL_PACKAGE,
        crosschecked_manifest.get("upstream_frozen_package_name"),
    )
    checks.add(
        "crosschecked_manifest_temporal_policy",
        str(crosschecked_manifest.get("upstream_temporal_policy_decision_id"))
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        crosschecked_manifest.get("upstream_temporal_policy_decision_id"),
    )
    frozen = crosschecked_manifest.get("frozen_invariants", {})
    for key in (
        "participant_split_changed",
        "supervised_population_changed",
        "mri_targets_changed",
        "temporal_policy_changed",
        "support_threshold_changed",
    ):
        checks.add(
            f"crosschecked_invariant__{key}",
            frozen.get(key) is False,
            False,
            frozen.get(key),
        )
    checks.add(
        "crosschecked_support_threshold",
        int(frozen.get("support_threshold", -1)) == SUPPORT_THRESHOLD,
        SUPPORT_THRESHOLD,
        frozen.get("support_threshold"),
    )

    if not checks.passed:
        raise ScenarioAbort("BENCHMARK04B/crosschecked-input contract validation failed.")
    return benchmark01, benchmark04b, support_policy, schema_lock, crosschecked_manifest


def load_official_schema(paths: Paths, checks: Checks) -> pd.DataFrame:
    schema = pd.read_csv(paths.official_schema, low_memory=False)
    schema.columns = [
        str(column).strip().lstrip("\ufeff") for column in schema.columns
    ]
    required = [
        "source_name",
        "column_name",
        "full_column_name",
        "official_semantic_include",
        "official_feature_type",
        "official_encoding",
        "official_semantic_role",
        "official_value_transform_id",
        "official_support_threshold_participants",
        "official_decision_id",
        "current_in_crosschecked_inputs",
        "derived_later",
        "grain",
        "official_execution_stage",
    ]
    require_columns(schema, required, "official feature schema")

    for column in ["source_name", "column_name", "full_column_name"]:
        schema[column] = schema[column].astype("string").str.strip()
    schema["official_semantic_include"] = schema[
        "official_semantic_include"
    ].map(parse_bool)
    for column in [
        "official_feature_type",
        "official_encoding",
        "official_semantic_role",
        "official_value_transform_id",
    ]:
        schema[column] = schema[column].fillna("").astype("string").str.strip()
    schema["official_support_threshold_participants"] = pd.to_numeric(
        schema["official_support_threshold_participants"], errors="raise"
    ).astype(int)
    schema["current_in_crosschecked_inputs"] = schema[
        "current_in_crosschecked_inputs"
    ].map(parse_bool)
    schema["derived_later"] = schema["derived_later"].map(parse_bool)
    schema["grain"] = schema["grain"].fillna("").astype("string").str.strip()
    schema["official_execution_stage"] = (
        schema["official_execution_stage"].fillna("").astype("string").str.strip()
    )

    checks.add(
        "official_schema_feature_count",
        len(schema) == EXPECTED_SCHEMA_ROWS,
        EXPECTED_SCHEMA_ROWS,
        len(schema),
    )
    checks.add(
        "official_schema_included_count",
        int(schema["official_semantic_include"].sum()) == EXPECTED_INCLUDED_FEATURES,
        EXPECTED_INCLUDED_FEATURES,
        int(schema["official_semantic_include"].sum()),
    )
    checks.add(
        "official_schema_excluded_count",
        int((~schema["official_semantic_include"]).sum()) == EXPECTED_EXCLUDED_FEATURES,
        EXPECTED_EXCLUDED_FEATURES,
        int((~schema["official_semantic_include"]).sum()),
    )
    checks.add(
        "official_schema_unique_features",
        not schema.duplicated(["source_name", "column_name"]).any()
        and not schema["full_column_name"].duplicated().any(),
        True,
        bool(
            not schema.duplicated(["source_name", "column_name"]).any()
            and not schema["full_column_name"].duplicated().any()
        ),
    )
    expected_full = (
        schema["source_name"].astype(str)
        + "__"
        + schema["column_name"].astype(str)
    )
    checks.add(
        "official_schema_full_name_contract",
        bool(expected_full.eq(schema["full_column_name"].astype(str)).all()),
        True,
        bool(expected_full.eq(schema["full_column_name"].astype(str)).all()),
    )
    checks.add(
        "official_schema_sources",
        set(schema["source_name"].dropna().astype(str).unique())
        == set(INPUT_SOURCES),
        list(INPUT_SOURCES),
        sorted(schema["source_name"].dropna().astype(str).unique().tolist()),
    )
    checks.add(
        "official_schema_threshold_50",
        set(schema["official_support_threshold_participants"].unique())
        == {SUPPORT_THRESHOLD},
        [SUPPORT_THRESHOLD],
        sorted(schema["official_support_threshold_participants"].unique().tolist()),
    )
    checks.add(
        "official_schema_current_feature_count",
        int(schema["current_in_crosschecked_inputs"].sum()) == 218,
        218,
        int(schema["current_in_crosschecked_inputs"].sum()),
    )
    age_row = schema.loc[
        schema["full_column_name"].eq("PTDEMOG__AGE_AT_TARGET")
    ]
    age_ok = (
        len(age_row) == 1
        and bool(age_row["official_semantic_include"].iloc[0])
        and bool(age_row["derived_later"].iloc[0])
        and str(age_row["grain"].iloc[0]) == "visit_derived"
        and str(age_row["official_execution_stage"].iloc[0]) == "BENCHMARK05b"
    )
    checks.add(
        "official_schema_age_at_target_contract",
        age_ok,
        True,
        age_ok,
    )
    allowed_types = {"numeric", "binary", "categorical", "multi_response", "excluded"}
    observed_types = set(schema["official_feature_type"].dropna().astype(str).unique())
    checks.add(
        "official_schema_types",
        observed_types.issubset(allowed_types),
        sorted(allowed_types),
        sorted(observed_types),
    )
    unsupported = sorted(
        set(
            schema.loc[
                schema["official_semantic_include"],
                "official_value_transform_id",
            ].astype(str)
        )
        - SUPPORTED_TRANSFORMS
    )
    checks.add("official_transform_ids_supported", not unsupported, [], unsupported)
    if not checks.passed:
        raise ScenarioAbort("Official schema validation failed.")
    return schema


def load_transform_registry(
    paths: Paths,
    schema: pd.DataFrame,
    checks: Checks,
) -> pd.DataFrame:
    registry = pd.read_csv(paths.official_transforms, low_memory=False)
    registry.columns = [
        str(column).strip().lstrip("\ufeff") for column in registry.columns
    ]
    required = [
        "source_name",
        "column_name",
        "full_column_name",
        "transform_id",
        "phase_required_for_transform",
        "phase_exposed_to_model",
        "parameters_json",
        "official_decision_id",
    ]
    require_columns(registry, required, "official transform registry")
    registry["full_column_name"] = (
        registry["full_column_name"].astype("string").str.strip()
    )
    registry["transform_id"] = registry["transform_id"].astype("string").str.strip()
    registry["phase_required_for_transform"] = registry[
        "phase_required_for_transform"
    ].map(parse_bool)
    registry["phase_exposed_to_model"] = registry[
        "phase_exposed_to_model"
    ].map(parse_bool)

    parsed_parameters: list[dict[str, Any]] = []
    for raw in registry["parameters_json"].tolist():
        try:
            parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise ScenarioAbort(
                f"Invalid parameters_json in transform registry: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ScenarioAbort("Each parameters_json must decode to an object.")
        parsed_parameters.append(parsed)
    registry = registry.copy()
    registry["parameters"] = parsed_parameters

    checks.add("official_transform_registry_count", len(registry) == EXPECTED_TRANSFORM_ROWS, EXPECTED_TRANSFORM_ROWS, len(registry))
    checks.add(
        "official_transform_registry_unique",
        not registry["full_column_name"].duplicated().any(),
        True,
        bool(not registry["full_column_name"].duplicated().any()),
    )
    checks.add(
        "phase_not_exposed_to_model",
        not registry["phase_exposed_to_model"].any(),
        False,
        bool(registry["phase_exposed_to_model"].any()),
    )
    checks.add(
        "phase_not_required_by_frozen_transforms",
        not registry["phase_required_for_transform"].any(),
        False,
        bool(registry["phase_required_for_transform"].any()),
    )
    merged = registry.merge(
        schema[
            [
                "full_column_name",
                "official_value_transform_id",
                "official_semantic_include",
            ]
        ],
        on="full_column_name",
        how="left",
        validate="one_to_one",
    )
    checks.add(
        "registry_matches_official_schema",
        bool(
            merged["transform_id"]
            .astype(str)
            .eq(merged["official_value_transform_id"].astype(str))
            .all()
        ),
        True,
        bool(
            merged["transform_id"]
            .astype(str)
            .eq(merged["official_value_transform_id"].astype(str))
            .all()
        ),
    )
    checks.add(
        "registry_only_included_features",
        bool(merged["official_semantic_include"].map(parse_bool).all()),
        True,
        bool(merged["official_semantic_include"].map(parse_bool).all()),
    )
    if not checks.passed:
        raise ScenarioAbort("Official transform registry validation failed.")
    return registry


def load_anchor_and_target_keys(
    paths: Paths,
    benchmark01: Mapping[str, Any],
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor_path = paths.package / ARTIFACTS["anchor"]
    target_path = paths.package / ARTIFACTS["targets"]
    catalog_path = paths.package / ARTIFACTS["target_catalog"]
    for name, path in {
        "anchor": anchor_path,
        "targets": target_path,
        "target_catalog": catalog_path,
    }.items():
        checks.add(f"artifact__{name}", path.is_file(), True, path.is_file(), str(path))
    if not checks.passed:
        raise ScenarioAbort("Frozen core artifacts are missing.")

    anchor_columns = [
        RID,
        VISCODE2,
        SPLIT,
        "eligible_visit_position",
        "protocol_month",
        "visit_order_basis",
    ]
    anchor = read_csv_text(anchor_path, usecols=anchor_columns)
    require_columns(anchor, anchor_columns, "supervised anchor")
    anchor[RID] = normalize_rid_key(anchor[RID], "supervised anchor")
    anchor[VISCODE2] = normalize_viscode_key(anchor[VISCODE2])
    anchor[SPLIT] = normalize_split_key(anchor[SPLIT])
    anchor["eligible_visit_position"] = pd.to_numeric(
        anchor["eligible_visit_position"], errors="raise"
    ).astype(int)
    anchor["protocol_month"] = pd.to_numeric(
        anchor["protocol_month"], errors="coerce"
    )
    anchor[SPLIT] = anchor[SPLIT].astype("string").str.strip()

    target_keys = read_csv_text(
        target_path,
        usecols=[RID, VISCODE2, SPLIT, "eligible_visit_position"],
    )
    target_keys[RID] = normalize_rid_key(target_keys[RID], "target keys")
    target_keys[VISCODE2] = normalize_viscode_key(target_keys[VISCODE2])
    target_keys[SPLIT] = normalize_split_key(target_keys[SPLIT])
    target_keys["eligible_visit_position"] = pd.to_numeric(
        target_keys["eligible_visit_position"], errors="raise"
    ).astype(int)

    catalog = pd.read_csv(catalog_path, low_memory=False)
    require_columns(
        catalog,
        ["target_position_1_based", "target_column"],
        "target catalog",
    )
    catalog["target_position_1_based"] = pd.to_numeric(
        catalog["target_position_1_based"], errors="raise"
    ).astype(int)

    checks.add("anchor_unique_keys", not anchor.duplicated([RID, VISCODE2]).any(), True, bool(not anchor.duplicated([RID, VISCODE2]).any()))
    checks.add("target_keys_unique", not target_keys.duplicated([RID, VISCODE2]).any(), True, bool(not target_keys.duplicated([RID, VISCODE2]).any()))
    checks.add("target_catalog_count", len(catalog) == TARGET_COUNT, TARGET_COUNT, len(catalog))
    checks.add(
        "target_catalog_positions",
        sorted(catalog["target_position_1_based"].tolist())
        == list(range(1, TARGET_COUNT + 1)),
        f"1..{TARGET_COUNT}",
        f"count={len(catalog)}",
    )
    anchor_index = pd.MultiIndex.from_frame(anchor[[RID, VISCODE2]])
    target_index = pd.MultiIndex.from_frame(target_keys[[RID, VISCODE2]])
    checks.add(
        "anchor_target_key_alignment",
        anchor_index.equals(target_index),
        True,
        anchor_index.equals(target_index),
    )
    checks.add(
        "anchor_target_split_alignment",
        bool(anchor[SPLIT].astype(str).eq(target_keys[SPLIT].astype(str)).all()),
        True,
        bool(anchor[SPLIT].astype(str).eq(target_keys[SPLIT].astype(str)).all()),
    )

    expected_split = (
        benchmark01.get("aggregate_summary", {})
        .get("split", {})
        .get("visit_counts", {})
    )
    observed_split = {
        split: int(anchor[SPLIT].eq(split).sum())
        for split in [TRAIN, VALIDATION, "test"]
    }
    if expected_split:
        checks.add(
            "anchor_split_counts_match_benchmark01",
            observed_split == {k: int(v) for k, v in expected_split.items()},
            expected_split,
            observed_split,
        )

    development = anchor.loc[anchor[SPLIT].isin(DEVELOPMENT_SPLITS)].copy()
    checks.add(
        "development_contains_no_test",
        not development[SPLIT].eq("test").any(),
        False,
        bool(development[SPLIT].eq("test").any()),
    )
    checks.add(
        "development_nonempty_splits",
        all(development[SPLIT].eq(split).any() for split in DEVELOPMENT_SPLITS),
        list(DEVELOPMENT_SPLITS),
        development[SPLIT].value_counts().to_dict(),
    )
    if not checks.passed:
        raise ScenarioAbort("Anchor/target metadata validation failed.")
    return anchor, development, target_keys


def load_source_frames(
    paths: Paths,
    schema: pd.DataFrame,
    checks: Checks,
) -> dict[str, pd.DataFrame]:
    source_frames: dict[str, pd.DataFrame] = {}
    development_rids: set[str] | None = None

    for source_name in INPUT_SOURCES:
        included_features = schema.loc[
            schema["source_name"].eq(source_name)
            & schema["official_semantic_include"]
            & schema["current_in_crosschecked_inputs"]
            & ~schema["derived_later"],
            "column_name",
        ].astype(str).tolist()
        keys = list(SOURCE_KEYS[source_name])
        path = (
            paths.crosschecked_root
            / CROSSCHECKED_INPUT_DIR
            / f"{source_name}_model_ready.csv.gz"
        )
        checks.add(
            f"artifact__model_ready__{source_name}",
            path.is_file(),
            True,
            path.is_file(),
            str(path),
        )
        if not path.is_file():
            continue
        crosschecked_manifest = read_json(paths.crosschecked_manifest)
        expected_source_hash = str(
            crosschecked_manifest.get("input_source_sha256", {}).get(source_name, "")
        )
        actual_source_hash = sha256_file(path) if path.is_file() else ""
        checks.add(
            f"crosschecked_source_hash__{source_name}",
            bool(expected_source_hash) and actual_source_hash == expected_source_hash,
            expected_source_hash,
            actual_source_hash,
        )
        frame = read_csv_text(path, usecols=[*keys, SPLIT, *included_features])
        require_columns(frame, [*keys, SPLIT, *included_features], f"{source_name} model-ready")
        frame[RID] = normalize_rid_key(frame[RID], f"{source_name} crosschecked input")
        if VISCODE2 in frame.columns:
            frame[VISCODE2] = normalize_viscode_key(frame[VISCODE2])
        frame[SPLIT] = normalize_split_key(frame[SPLIT])
        frame = frame.loc[frame[SPLIT].isin(DEVELOPMENT_SPLITS)].copy()
        checks.add(
            f"source_unique_keys__{source_name}",
            not frame.duplicated(keys).any(),
            True,
            bool(not frame.duplicated(keys).any()),
        )
        checks.add(
            f"source_no_test__{source_name}",
            not frame[SPLIT].eq("test").any(),
            False,
            bool(frame[SPLIT].eq("test").any()),
        )
        source_frames[source_name] = frame

    if not checks.passed:
        raise ScenarioAbort("Model-ready source validation failed.")
    return source_frames


def build_visit_availability_timeline(
    source_frames: Mapping[str, pd.DataFrame],
    schema: pd.DataFrame,
    checks: Checks,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for source_name in VISIT_SOURCES:
        frame = source_frames[source_name]
        features = schema.loc[
            schema["source_name"].eq(source_name)
            & schema["official_semantic_include"]
            & schema["current_in_crosschecked_inputs"]
            & ~schema["derived_later"],
            "column_name",
        ].astype(str).tolist()
        observed_count = frame[features].notna().sum(axis=1).astype(int)
        source_rows = frame.loc[
            observed_count.gt(0), [RID, VISCODE2, SPLIT]
        ].copy()
        source_rows["source_name"] = source_name
        source_rows["observed_authorized_feature_count"] = observed_count.loc[
            observed_count.gt(0)
        ].to_numpy()
        rows.append(source_rows)

    if not rows:
        raise ScenarioAbort("No visit-level source rows were available.")
    stacked = pd.concat(rows, ignore_index=True, sort=False)
    split_conflicts = (
        stacked.groupby([RID, VISCODE2], dropna=False)[SPLIT]
        .nunique(dropna=False)
        .gt(1)
    )
    checks.add(
        "timeline_visit_split_consistency",
        not split_conflicts.any(),
        False,
        bool(split_conflicts.any()),
    )

    timeline = (
        stacked.groupby([RID, VISCODE2, SPLIT], dropna=False, sort=False)
        .agg(
            modality_count=("source_name", "nunique"),
            observed_authorized_feature_count=(
                "observed_authorized_feature_count",
                "sum",
            ),
            available_sources=(
                "source_name",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    timeline["protocol_month"] = timeline[VISCODE2].map(parse_protocol_month)
    timeline["protocol_month_known"] = timeline["protocol_month"].notna()
    timeline = timeline.sort_values(
        [RID, "protocol_month_known", "protocol_month", VISCODE2],
        ascending=[True, False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    checks.add(
        "timeline_unique_visit_keys",
        not timeline.duplicated([RID, VISCODE2]).any(),
        True,
        bool(not timeline.duplicated([RID, VISCODE2]).any()),
    )
    checks.add(
        "timeline_positive_authorized_content",
        bool(timeline["observed_authorized_feature_count"].gt(0).all()),
        True,
        bool(timeline["observed_authorized_feature_count"].gt(0).all()),
    )
    if not checks.passed:
        raise ScenarioAbort("Visit availability timeline validation failed.")
    return timeline



def load_age_derivation_table(
    paths: Paths,
    development_anchor: pd.DataFrame,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    age_meta = read_csv_text(paths.age_metadata)
    require_columns(
        age_meta,
        [RID, SPLIT, "BIRTH_YEAR_FOR_AGE", "BIRTH_YEAR_CONFLICT"],
        "A06 local age metadata",
    )
    age_meta[RID] = normalize_rid_key(age_meta[RID], "A06 age metadata")
    age_meta[SPLIT] = normalize_split_key(age_meta[SPLIT])
    age_meta = age_meta.loc[age_meta[SPLIT].isin(DEVELOPMENT_SPLITS)].copy()

    conflict = age_meta["BIRTH_YEAR_CONFLICT"].map(parse_bool)
    checks.add(
        "age_metadata_no_birth_year_conflicts",
        not conflict.any(),
        False,
        bool(conflict.any()),
    )
    checks.add(
        "age_metadata_unique_rid",
        not age_meta.duplicated(RID).any(),
        True,
        not age_meta.duplicated(RID).any(),
    )
    age_meta["__birth_year"] = pd.to_numeric(
        age_meta["BIRTH_YEAR_FOR_AGE"], errors="coerce"
    )

    # Read only the MRI date column and exact visit keys. No MRI target value is loaded.
    mri_dates = read_csv_text(
        paths.mri_canonical,
        usecols=[RID, VISCODE2, "EXAMDATE"],
    )
    mri_dates[RID] = normalize_rid_key(mri_dates[RID], "UCSFFSX7 canonical dates")
    mri_dates[VISCODE2] = normalize_viscode_key(mri_dates[VISCODE2])
    mri_dates["__target_date"] = parse_clinical_dates(mri_dates["EXAMDATE"])
    mri_dates = mri_dates.loc[:, [RID, VISCODE2, "__target_date"]]
    checks.add(
        "mri_date_table_unique_visit_keys",
        not mri_dates.duplicated([RID, VISCODE2]).any(),
        True,
        not mri_dates.duplicated([RID, VISCODE2]).any(),
    )

    dev_keys = development_anchor.loc[:, [RID, VISCODE2, SPLIT]].copy()
    dev_keys[RID] = normalize_rid_key(dev_keys[RID], "development anchor")
    dev_keys[VISCODE2] = normalize_viscode_key(dev_keys[VISCODE2])
    dev_keys[SPLIT] = normalize_split_key(dev_keys[SPLIT])

    work = dev_keys.merge(
        age_meta.loc[:, [RID, SPLIT, "__birth_year"]],
        on=[RID, SPLIT],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    work = work.merge(
        mri_dates,
        on=[RID, VISCODE2],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    work["AGE_AT_TARGET"] = (
        work["__target_date"].dt.year.astype("Float64")
        - work["__birth_year"].astype("Float64")
    )
    invalid_age = (
        work["AGE_AT_TARGET"].notna()
        & (
            work["AGE_AT_TARGET"].lt(18)
            | work["AGE_AT_TARGET"].gt(120)
            | work["AGE_AT_TARGET"].ne(work["AGE_AT_TARGET"].round())
        )
    )
    checks.add(
        "age_at_target_plausibility",
        int(invalid_age.sum()) == 0,
        0,
        int(invalid_age.sum()),
    )
    if invalid_age.any():
        raise ScenarioAbort("AGE_AT_TARGET derivation produced implausible values.")

    checks.add(
        "age_at_target_no_raw_birth_fields_exposed",
        not any(
            column in work.columns for column in ["PTDOB", "PTDOBYY"]
        ),
        True,
        not any(column in work.columns for column in ["PTDOB", "PTDOBYY"]),
    )

    # Safe reports may use only availability counts, never age values.
    age_table = work.loc[:, [RID, VISCODE2, SPLIT, "AGE_AT_TARGET"]].copy()
    local_audit = work.loc[
        :,
        [RID, VISCODE2, SPLIT, "__birth_year", "__target_date", "AGE_AT_TARGET"],
    ].copy().rename(
        columns={
            "__birth_year": "BIRTH_YEAR_FOR_AGE",
            "__target_date": "CURRENT_TARGET_MRI_EXAMDATE",
        }
    )
    return age_table, local_audit


def attach_age_at_target(
    manifest: pd.DataFrame,
    age_table: pd.DataFrame,
    checks: Checks,
    context: str,
) -> pd.DataFrame:
    out = manifest.copy()
    keys = out.loc[:, [RID, SPLIT, "current_VISCODE2"]].copy()
    keys = keys.rename(columns={"current_VISCODE2": VISCODE2})
    keys["__row_order"] = np.arange(len(keys), dtype=np.int64)
    merged = keys.merge(
        age_table,
        on=[RID, VISCODE2, SPLIT],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("__row_order", kind="stable")
    checks.add(
        f"{context}__age_alignment_row_count",
        len(merged) == len(out),
        len(out),
        len(merged),
    )
    out["AGE_AT_TARGET"] = pd.to_numeric(
        merged["AGE_AT_TARGET"], errors="coerce"
    ).astype("Float64").to_numpy()
    return out


def compute_age_support_and_audit(
    scenario_name: str,
    manifest: pd.DataFrame,
    schema: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    age_schema = schema.loc[
        schema["full_column_name"].eq("PTDEMOG__AGE_AT_TARGET")
        & schema["official_semantic_include"]
    ].copy()
    if len(age_schema) != 1:
        raise ScenarioAbort(
            "Official schema must contain exactly one included AGE_AT_TARGET row."
        )
    row = age_schema.iloc[0]
    support_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []

    for split_label in DEVELOPMENT_SPLITS:
        subset = manifest.loc[manifest[SPLIT].eq(split_label)].copy()
        raw = pd.to_numeric(subset["AGE_AT_TARGET"], errors="coerce")
        modality_rows.append(
            {
                "scenario": scenario_name,
                "block": "current_derived",
                "source_name": "PTDEMOG",
                "split": split_label,
                "rows_or_participants": int(len(subset)),
                "modality_present_count": int(raw.notna().sum()),
                "modality_present_pct": (
                    100.0 * float(raw.notna().mean()) if len(subset) else np.nan
                ),
            }
        )
        audit_rows.append(
            {
                "scenario": scenario_name,
                "block": "current_derived",
                "split": split_label,
                "source_name": "PTDEMOG",
                "column_name": "AGE_AT_TARGET",
                "full_column_name": "PTDEMOG__AGE_AT_TARGET",
                "official_feature_type": "numeric",
                "transform_id": (
                    "derive_age_at_target_from_birth_year_and_current_target_date"
                ),
                "nonmissing_input_rows": int(raw.notna().sum()),
                "nonmissing_output_rows": int(raw.notna().sum()),
                "transform_failure_rows": 0,
                "type_validation_failure_rows": 0,
                "validation_passed": True,
            }
        )

    train = manifest.loc[
        manifest[SPLIT].eq(TRAIN), [RID, "AGE_AT_TARGET"]
    ].copy()
    observed = train["AGE_AT_TARGET"].notna()
    support = int(train.loc[observed, RID].nunique())
    train_participants = int(train[RID].nunique())
    support_rows.append(
        {
            "scenario": scenario_name,
            "block": "current_derived",
            "model_feature_prefix": (
                "current__"
                if scenario_name == "longitudinal_previous_last"
                else ""
            ),
            "source_name": "PTDEMOG",
            "column_name": "AGE_AT_TARGET",
            "full_column_name": "PTDEMOG__AGE_AT_TARGET",
            "official_feature_type": str(row["official_feature_type"]),
            "official_encoding": str(row["official_encoding"]),
            "official_semantic_role": str(row["official_semantic_role"]),
            "transform_id": str(row["official_value_transform_id"]),
            "train_participant_count": train_participants,
            "train_observed_participants": support,
            "train_observed_participant_fraction": (
                support / train_participants if train_participants else np.nan
            ),
            "support_threshold_participants": SUPPORT_THRESHOLD,
            "retained_at_support_threshold": bool(
                support >= SUPPORT_THRESHOLD
            ),
            "support_rule": (
                "distinct_fit_participants_with_valid_derived_age_"
                "before_imputation_and_encoding"
            ),
        }
    )
    return (
        pd.DataFrame(support_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(modality_rows),
    )


def build_snapshot_all(development_anchor: pd.DataFrame) -> pd.DataFrame:
    output = development_anchor[
        [
            RID,
            VISCODE2,
            SPLIT,
            "eligible_visit_position",
            "protocol_month",
            "visit_order_basis",
        ]
    ].copy()
    output = output.rename(
        columns={
            VISCODE2: "current_VISCODE2",
            "eligible_visit_position": "current_eligible_visit_position",
            "protocol_month": "current_protocol_month",
            "visit_order_basis": "current_visit_order_basis",
        }
    )
    output["scenario"] = "snapshot_all"
    return output.sort_values(
        [SPLIT, RID, "current_eligible_visit_position", "current_VISCODE2"],
        kind="stable",
    ).reset_index(drop=True)


def build_matched_pairs(
    development_anchor: pd.DataFrame,
    timeline: pd.DataFrame,
    paths: Paths,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Consume the frozen SPLIT07 pairing reference; never recompute ties.
    required_columns = [
        RID,
        SPLIT,
        "current_VISCODE2",
        "current_protocol_month",
        "previous_VISCODE2",
        "previous_protocol_month",
        "pairing_status",
        "pairing_reason",
        "selection_method",
        "candidate_count_at_latest_prior_month",
        "protocol_gap_months",
        "conservative_date_gap_days",
        "median_date_gap_days",
    ]
    reference = read_csv_text(paths.temporal_pairing_reference)
    require_columns(reference, required_columns, "SPLIT07 pairing reference")
    reference = reference.loc[:, required_columns].copy()

    for column in (
        "current_protocol_month",
        "previous_protocol_month",
        "candidate_count_at_latest_prior_month",
        "protocol_gap_months",
        "conservative_date_gap_days",
        "median_date_gap_days",
    ):
        reference[column] = pd.to_numeric(reference[column], errors="coerce")

    checks.add(
        "temporal_reference_unique_participants",
        not reference.duplicated(RID).any(),
        True,
        bool(not reference.duplicated(RID).any()),
    )
    checks.add(
        "temporal_reference_development_only",
        set(reference[SPLIT].dropna().astype(str).unique())
        == set(DEVELOPMENT_SPLITS),
        list(DEVELOPMENT_SPLITS),
        sorted(reference[SPLIT].dropna().astype(str).unique().tolist()),
    )
    checks.add(
        "temporal_reference_status_vocabulary",
        set(reference["pairing_status"].dropna().astype(str).unique())
        == {"eligible", "excluded"},
        ["eligible", "excluded"],
        sorted(reference["pairing_status"].dropna().astype(str).unique().tolist()),
    )

    current_expected = (
        development_anchor.sort_values(
            [RID, "eligible_visit_position", VISCODE2],
            kind="stable",
        )
        .groupby(RID, sort=False, as_index=False)
        .tail(1)
        .loc[
            :,
            [
                RID,
                VISCODE2,
                SPLIT,
                "protocol_month",
                "eligible_visit_position",
                "visit_order_basis",
            ],
        ]
        .rename(
            columns={
                VISCODE2: "expected_current_VISCODE2",
                SPLIT: "expected_split",
                "protocol_month": "expected_current_protocol_month",
                "eligible_visit_position": "current_eligible_visit_position",
                "visit_order_basis": "current_visit_order_basis",
            }
        )
    )
    checks.add(
        "one_current_target_per_development_participant",
        not current_expected.duplicated(RID).any(),
        True,
        bool(not current_expected.duplicated(RID).any()),
    )

    reference = reference.merge(
        current_expected,
        on=RID,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    checks.add(
        "temporal_reference_all_development_participants",
        len(reference) == len(current_expected)
        and reference["expected_current_VISCODE2"].notna().all(),
        len(current_expected),
        len(reference),
    )
    checks.add(
        "temporal_reference_current_visit_matches_anchor",
        bool(
            reference["current_VISCODE2"]
            .astype(str)
            .eq(reference["expected_current_VISCODE2"].astype(str))
            .all()
        ),
        True,
        bool(
            reference["current_VISCODE2"]
            .astype(str)
            .eq(reference["expected_current_VISCODE2"].astype(str))
            .all()
        ),
    )
    checks.add(
        "temporal_reference_split_matches_anchor",
        bool(
            reference[SPLIT]
            .astype(str)
            .eq(reference["expected_split"].astype(str))
            .all()
        ),
        True,
        bool(
            reference[SPLIT]
            .astype(str)
            .eq(reference["expected_split"].astype(str))
            .all()
        ),
    )

    participant_counts = {
        split: int(reference[SPLIT].eq(split).sum())
        for split in DEVELOPMENT_SPLITS
    }
    checks.add(
        "temporal_reference_participant_counts",
        participant_counts == EXPECTED_DEVELOPMENT_PARTICIPANTS,
        EXPECTED_DEVELOPMENT_PARTICIPANTS,
        participant_counts,
    )

    observed_reason_counts = {
        split: {
            str(reason): int(count)
            for reason, count in reference.loc[
                reference[SPLIT].eq(split), "pairing_reason"
            ].value_counts().items()
        }
        for split in DEVELOPMENT_SPLITS
    }
    checks.add(
        "temporal_reference_reason_counts",
        observed_reason_counts == EXPECTED_PAIRING_REASON_COUNTS,
        EXPECTED_PAIRING_REASON_COUNTS,
        observed_reason_counts,
    )

    eligible = reference.loc[
        reference["pairing_status"].eq("eligible")
    ].copy()
    excluded = reference.loc[
        reference["pairing_status"].eq("excluded")
    ].copy()
    eligible_counts = {
        split: int(eligible[SPLIT].eq(split).sum())
        for split in DEVELOPMENT_SPLITS
    }
    checks.add(
        "temporal_reference_final_eligible_counts",
        eligible_counts == EXPECTED_FINAL_ELIGIBLE,
        EXPECTED_FINAL_ELIGIBLE,
        eligible_counts,
    )

    previous_info = timeline.rename(
        columns={
            VISCODE2: "previous_VISCODE2",
            "protocol_month": "timeline_previous_protocol_month",
            "modality_count": "previous_modality_count",
            "observed_authorized_feature_count": (
                "previous_observed_authorized_feature_count"
            ),
            "available_sources": "previous_available_sources",
        }
    )[
        [
            RID,
            "previous_VISCODE2",
            SPLIT,
            "timeline_previous_protocol_month",
            "previous_modality_count",
            "previous_observed_authorized_feature_count",
            "previous_available_sources",
        ]
    ].copy()

    eligible = eligible.merge(
        previous_info,
        on=[RID, "previous_VISCODE2", SPLIT],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    content_profiled = eligible[
        "previous_observed_authorized_feature_count"
    ].notna()
    profiled_protocol_match = bool(
        np.isclose(
            eligible.loc[
                content_profiled, "previous_protocol_month"
            ].astype(float),
            eligible.loc[
                content_profiled, "timeline_previous_protocol_month"
            ].astype(float),
            equal_nan=False,
        ).all()
    )
    checks.add(
        "profiled_previous_protocol_month_matches_timeline",
        profiled_protocol_match,
        True,
        profiled_protocol_match,
    )

    # A frozen temporal pair remains in the pair table even if the corrected
    # semantic schema leaves that historical visit with no authorized input.
    # That participant is removed from the matched scenarios only after the
    # A05 source-block exclusions have also been applied, so the A05 manifest
    # continues to match the complete frozen eligible-pair set.
    eligible["previous_observed_authorized_feature_count"] = pd.to_numeric(
        eligible["previous_observed_authorized_feature_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    eligible["previous_modality_count"] = pd.to_numeric(
        eligible["previous_modality_count"],
        errors="coerce",
    ).fillna(0).astype(int)
    eligible["previous_available_sources"] = eligible[
        "previous_available_sources"
    ].astype("string")

    pairs = pd.DataFrame(
        {
            RID: eligible[RID].astype("string"),
            SPLIT: eligible[SPLIT].astype("string"),
            "current_VISCODE2": eligible["current_VISCODE2"].astype("string"),
            "current_protocol_month": eligible["current_protocol_month"].astype(float),
            "current_eligible_visit_position": eligible[
                "current_eligible_visit_position"
            ].astype(int),
            "current_visit_order_basis": eligible[
                "current_visit_order_basis"
            ].astype("string"),
            "previous_VISCODE2": eligible["previous_VISCODE2"].astype("string"),
            "previous_protocol_month": eligible["previous_protocol_month"].astype(float),
            "gap_months": eligible["protocol_gap_months"].astype(float),
            "conservative_date_gap_days": eligible[
                "conservative_date_gap_days"
            ].astype("Float64"),
            "median_date_gap_days": eligible["median_date_gap_days"].astype("Float64"),
            "previous_modality_count": eligible["previous_modality_count"].astype(int),
            "previous_observed_authorized_feature_count": eligible[
                "previous_observed_authorized_feature_count"
            ].astype(int),
            "previous_available_sources": eligible[
                "previous_available_sources"
            ].astype("string"),
            "pairing_reason": eligible["pairing_reason"].astype("string"),
            "previous_selection_rule": eligible["selection_method"].astype("string"),
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        }
    )

    exclusions = pd.DataFrame(
        {
            RID: excluded[RID].astype("string"),
            SPLIT: excluded[SPLIT].astype("string"),
            "current_VISCODE2": excluded["current_VISCODE2"].astype("string"),
            "current_protocol_month": excluded["current_protocol_month"].astype("Float64"),
            "current_eligible_visit_position": excluded[
                "current_eligible_visit_position"
            ].astype(int),
            "current_visit_order_basis": excluded[
                "current_visit_order_basis"
            ].astype("string"),
            "exclusion_reason": excluded["pairing_reason"].astype("string"),
            "selection_method": excluded["selection_method"].astype("string"),
            "known_earlier_candidate_count": pd.Series(
                pd.NA, index=excluded.index, dtype="Int64"
            ),
            "max_prior_month_tie_count": excluded[
                "candidate_count_at_latest_prior_month"
            ].astype("Int64"),
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        }
    )

    pairs = pairs.sort_values([SPLIT, RID], kind="stable").reset_index(drop=True)
    exclusions = exclusions.sort_values([SPLIT, RID], kind="stable").reset_index(drop=True)

    checks.add(
        "pairs_unique_participants",
        not pairs.duplicated(RID).any(),
        True,
        bool(not pairs.duplicated(RID).any()),
    )
    checks.add(
        "pairs_positive_protocol_gap",
        bool(pairs["gap_months"].gt(0).all()),
        True,
        bool(pairs["gap_months"].gt(0).all()),
    )
    checks.add(
        "pairs_strict_protocol_order",
        bool(
            pairs["previous_protocol_month"]
            .astype(float)
            .lt(pairs["current_protocol_month"].astype(float))
            .all()
        ),
        True,
        bool(
            pairs["previous_protocol_month"]
            .astype(float)
            .lt(pairs["current_protocol_month"].astype(float))
            .all()
        ),
    )
    checks.add(
        "pairs_different_visit_keys",
        bool(
            pairs["previous_VISCODE2"]
            .astype(str)
            .ne(pairs["current_VISCODE2"].astype(str))
            .all()
        ),
        True,
        bool(
            pairs["previous_VISCODE2"]
            .astype(str)
            .ne(pairs["current_VISCODE2"].astype(str))
            .all()
        ),
    )
    strict_pairs = pairs.loc[
        pairs["pairing_reason"].eq("eligible_strict_date_interval_winner")
    ]
    checks.add(
        "strict_date_winners_have_positive_conservative_gap",
        bool(strict_pairs["conservative_date_gap_days"].astype(float).gt(0).all()),
        True,
        bool(strict_pairs["conservative_date_gap_days"].astype(float).gt(0).all()),
    )
    checks.add(
        "pairing_reference_accounts_for_all_development_participants",
        len(pairs) + len(exclusions) == len(reference),
        len(reference),
        len(pairs) + len(exclusions),
    )
    checks.add(
        "pair_train_validation_participants_disjoint",
        set(pairs.loc[pairs[SPLIT].eq(TRAIN), RID].astype(str)).isdisjoint(
            set(pairs.loc[pairs[SPLIT].eq(VALIDATION), RID].astype(str))
        ),
        True,
        set(pairs.loc[pairs[SPLIT].eq(TRAIN), RID].astype(str)).isdisjoint(
            set(pairs.loc[pairs[SPLIT].eq(VALIDATION), RID].astype(str))
        ),
    )
    if not checks.passed:
        raise ScenarioAbort("Frozen temporal pairing reference failed validation.")
    return pairs, exclusions

def load_temporal_modality_exclusions(
    paths: Paths,
    checks: Checks,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks.add(
        "artifact__auditA05_summary",
        paths.temporal_exclusion_summary.is_file(),
        True,
        paths.temporal_exclusion_summary.is_file(),
        str(paths.temporal_exclusion_summary),
    )
    checks.add(
        "artifact__auditA05_local_manifest",
        paths.temporal_exclusion_manifest.is_file(),
        True,
        paths.temporal_exclusion_manifest.is_file(),
        str(paths.temporal_exclusion_manifest),
    )
    if not checks.passed:
        raise ScenarioAbort("AUDIT A05 temporal exclusion artifacts are missing.")

    summary = read_json(paths.temporal_exclusion_summary)
    checks.add(
        "auditA05_passed",
        summary.get("audit_passed") is True,
        True,
        summary.get("audit_passed"),
    )
    checks.add(
        "auditA05_script_version",
        str(summary.get("script_version"))
        == EXPECTED_AUDIT_A05_SCRIPT_VERSION,
        EXPECTED_AUDIT_A05_SCRIPT_VERSION,
        summary.get("script_version"),
    )
    checks.add(
        "auditA05_no_fatal_error",
        summary.get("fatal_error") is None,
        None,
        summary.get("fatal_error"),
    )

    local_record = summary.get("local_only_manifest", {})
    expected_hash = str(local_record.get("sha256", ""))
    actual_hash = sha256_file(paths.temporal_exclusion_manifest)
    checks.add(
        "auditA05_local_manifest_hash",
        bool(expected_hash) and actual_hash == expected_hash,
        expected_hash,
        actual_hash,
    )
    checks.add(
        "auditA05_manifest_declared_local_only",
        local_record.get("contains_individualized_keys") is True
        and local_record.get("must_not_be_uploaded_or_versioned") is True,
        True,
        bool(
            local_record.get("contains_individualized_keys") is True
            and local_record.get("must_not_be_uploaded_or_versioned") is True
        ),
    )

    manifest = read_csv_text(paths.temporal_exclusion_manifest)
    required = [
        RID,
        SPLIT,
        "current_VISCODE2",
        "previous_VISCODE2",
        "source_name",
        "temporal_relation",
        "action",
    ]
    require_columns(manifest, required, "AUDIT A05 local manifest")
    manifest = manifest.loc[:, required].copy()
    manifest[RID] = normalize_rid_key(manifest[RID], "AUDIT A05")
    manifest[SPLIT] = normalize_split_key(manifest[SPLIT])
    manifest["current_VISCODE2"] = normalize_viscode_key(
        manifest["current_VISCODE2"]
    )
    manifest["previous_VISCODE2"] = normalize_viscode_key(
        manifest["previous_VISCODE2"]
    )
    manifest["source_name"] = (
        manifest["source_name"].astype("string").str.strip().str.upper()
    )
    manifest["temporal_relation"] = (
        manifest["temporal_relation"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    manifest["action"] = (
        manifest["action"].astype("string").str.strip().str.casefold()
    )

    checks.add(
        "auditA05_manifest_nonempty",
        not manifest.empty,
        True,
        not manifest.empty,
    )
    checks.add(
        "auditA05_manifest_development_only",
        set(manifest[SPLIT].dropna().astype(str).unique()).issubset(
            set(DEVELOPMENT_SPLITS)
        ),
        True,
        set(manifest[SPLIT].dropna().astype(str).unique()).issubset(
            set(DEVELOPMENT_SPLITS)
        ),
    )
    checks.add(
        "auditA05_manifest_sources_authorized",
        set(manifest["source_name"].dropna().astype(str).unique()).issubset(
            set(VISIT_SOURCES)
        ),
        True,
        set(manifest["source_name"].dropna().astype(str).unique()).issubset(
            set(VISIT_SOURCES)
        ),
    )
    checks.add(
        "auditA05_manifest_relations_authorized",
        set(
            manifest["temporal_relation"].dropna().astype(str).unique()
        ).issubset(ALLOWED_TEMPORAL_EXCLUSION_RELATIONS),
        True,
        set(
            manifest["temporal_relation"].dropna().astype(str).unique()
        ).issubset(ALLOWED_TEMPORAL_EXCLUSION_RELATIONS),
    )
    checks.add(
        "auditA05_manifest_action_authorized",
        set(manifest["action"].dropna().astype(str).unique())
        == {EXPECTED_TEMPORAL_EXCLUSION_ACTION},
        True,
        set(manifest["action"].dropna().astype(str).unique())
        == {EXPECTED_TEMPORAL_EXCLUSION_ACTION},
    )
    checks.add(
        "auditA05_manifest_unique_pair_source",
        not manifest.duplicated(
            [
                RID,
                SPLIT,
                "current_VISCODE2",
                "previous_VISCODE2",
                "source_name",
            ]
        ).any(),
        True,
        not manifest.duplicated(
            [
                RID,
                SPLIT,
                "current_VISCODE2",
                "previous_VISCODE2",
                "source_name",
            ]
        ).any(),
    )
    if not checks.passed:
        raise ScenarioAbort("AUDIT A05 temporal exclusion validation failed.")
    return manifest, summary


def apply_temporal_modality_exclusions_to_pairs(
    pairs: pd.DataFrame,
    exclusions: pd.DataFrame,
    audit_manifest: pd.DataFrame,
    source_frames: Mapping[str, pd.DataFrame],
    schema: pd.DataFrame,
    checks: Checks,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = pairs.copy().reset_index(drop=True)
    normalized = corrected[
        [RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"]
    ].copy()
    normalized[RID] = normalize_rid_key(normalized[RID], "BENCHMARK05B pairs")
    normalized[SPLIT] = normalize_split_key(normalized[SPLIT])
    normalized["current_VISCODE2"] = normalize_viscode_key(
        normalized["current_VISCODE2"]
    )
    normalized["previous_VISCODE2"] = normalize_viscode_key(
        normalized["previous_VISCODE2"]
    )
    normalized["__row_order"] = np.arange(len(normalized), dtype=np.int64)

    action_match = audit_manifest.merge(
        normalized,
        on=[RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    all_actions_match = bool(action_match["_merge"].eq("both").all())
    checks.add(
        "auditA05_all_actions_match_eligible_pairs",
        all_actions_match,
        True,
        all_actions_match,
    )
    if not all_actions_match:
        raise ScenarioAbort(
            "AUDIT A05 contains an action outside the eligible pair manifest."
        )

    corrected["temporal_source_exclusion_applied"] = False
    corrected["temporal_source_exclusion_action_count"] = 0

    for source_name in VISIT_SOURCES:
        flag_column = temporal_exclusion_flag(source_name)
        source_actions = audit_manifest.loc[
            audit_manifest["source_name"].eq(source_name),
            [RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"],
        ].copy()
        source_actions[flag_column] = True
        merged = normalized.merge(
            source_actions,
            on=[RID, SPLIT, "current_VISCODE2", "previous_VISCODE2"],
            how="left",
            validate="one_to_one",
            sort=False,
        ).sort_values("__row_order", kind="stable")
        flag = merged[flag_column].map(parse_bool).to_numpy(dtype=bool)
        corrected[flag_column] = flag
        corrected["temporal_source_exclusion_applied"] |= flag
        corrected["temporal_source_exclusion_action_count"] += flag.astype(int)

    action_flags_match = bool(
        int(corrected["temporal_source_exclusion_action_count"].sum())
        == len(audit_manifest)
    )
    checks.add(
        "auditA05_action_flags_match_manifest",
        action_flags_match,
        True,
        action_flags_match,
    )

    corrected_counts = pd.to_numeric(
        corrected["previous_observed_authorized_feature_count"],
        errors="raise",
    ).astype(int)
    pre_a05_authorized_counts = corrected_counts.copy()
    corrected_modalities = pd.to_numeric(
        corrected["previous_modality_count"], errors="raise"
    ).astype(int)
    source_sets = corrected["previous_available_sources"].map(
        lambda value: {
            token
            for token in str(value).split("|")
            if token and token != "<NA>"
        }
    )

    for source_name in VISIT_SOURCES:
        flag_column = temporal_exclusion_flag(source_name)
        mask = corrected[flag_column].map(parse_bool).to_numpy(dtype=bool)
        if not mask.any():
            continue
        features = schema.loc[
            schema["source_name"].eq(source_name)
            & schema["official_semantic_include"]
            & schema["current_in_crosschecked_inputs"]
            & ~schema["derived_later"],
            "column_name",
        ].astype(str).tolist()
        aligned = align_source_to_manifest(
            corrected,
            source_frames[source_name],
            source_name,
            "previous_visit",
            features,
        )
        observed_count = aligned[features].notna().sum(axis=1).astype(int)
        corrected_counts.loc[mask] = (
            corrected_counts.loc[mask] - observed_count.loc[mask]
        )
        for index in corrected.index[mask]:
            available = set(source_sets.at[index])
            if source_name in available:
                available.remove(source_name)
                corrected_modalities.at[index] -= 1
            source_sets.at[index] = available

    corrected["previous_observed_authorized_feature_count"] = corrected_counts
    corrected["previous_modality_count"] = corrected_modalities
    corrected["previous_available_sources"] = source_sets.map(
        lambda values: "|".join(sorted(values)) if values else pd.NA
    ).astype("string")

    nonnegative = bool(
        corrected["previous_observed_authorized_feature_count"].ge(0).all()
        and corrected["previous_modality_count"].ge(0).all()
    )
    checks.add(
        "temporal_exclusion_corrected_counts_nonnegative",
        nonnegative,
        True,
        nonnegative,
    )
    if not nonnegative:
        raise ScenarioAbort("Temporal exclusions produced negative availability counts.")

    no_safe_previous = corrected[
        "previous_observed_authorized_feature_count"
    ].le(0)
    additional = corrected.loc[no_safe_previous].copy()
    corrected = corrected.loc[~no_safe_previous].copy()

    if not additional.empty:
        original_count_for_additional = pre_a05_authorized_counts.loc[
            additional.index
        ]
        had_no_crosschecked_content = original_count_for_additional.le(0)
        exclusion_reasons = np.where(
            had_no_crosschecked_content.to_numpy(dtype=bool),
            "no_semantically_authorized_previous_input_after_crosscheck",
            "no_temporally_safe_previous_input_after_source_exclusion",
        )
        selection_methods = np.where(
            had_no_crosschecked_content.to_numpy(dtype=bool),
            "benchmark05b_crosschecked_content_filter",
            "auditA05_temporal_source_exclusion",
        )
        added_exclusions = pd.DataFrame(
            {
                RID: additional[RID].astype("string"),
                SPLIT: additional[SPLIT].astype("string"),
                "current_VISCODE2": additional["current_VISCODE2"].astype(
                    "string"
                ),
                "current_protocol_month": additional[
                    "current_protocol_month"
                ].astype("Float64"),
                "current_eligible_visit_position": additional[
                    "current_eligible_visit_position"
                ].astype(int),
                "current_visit_order_basis": additional[
                    "current_visit_order_basis"
                ].astype("string"),
                "exclusion_reason": pd.Series(
                    exclusion_reasons, index=additional.index, dtype="string"
                ),
                "selection_method": pd.Series(
                    selection_methods, index=additional.index, dtype="string"
                ),
                "known_earlier_candidate_count": pd.Series(
                    pd.NA, index=additional.index, dtype="Int64"
                ),
                "max_prior_month_tie_count": pd.Series(
                    pd.NA, index=additional.index, dtype="Int64"
                ),
                "temporal_policy_decision_id": (
                    EXPECTED_TEMPORAL_POLICY_DECISION_ID
                ),
            }
        )
        exclusions = pd.concat(
            [exclusions, added_exclusions], ignore_index=True, sort=False
        )

    corrected = corrected.sort_values([SPLIT, RID], kind="stable").reset_index(
        drop=True
    )
    exclusions = exclusions.sort_values([SPLIT, RID], kind="stable").reset_index(
        drop=True
    )
    checks.add(
        "all_retained_pairs_have_temporally_safe_previous_input",
        bool(
            corrected["previous_observed_authorized_feature_count"].gt(0).all()
        ),
        True,
        bool(
            corrected["previous_observed_authorized_feature_count"].gt(0).all()
        ),
    )
    checks.add(
        "temporal_exclusion_preserves_pair_uniqueness",
        not corrected.duplicated(RID).any(),
        True,
        not corrected.duplicated(RID).any(),
    )
    if not checks.passed:
        raise ScenarioAbort("Temporal source exclusion application failed.")
    return corrected, exclusions


def build_snapshot_matched(pairs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        RID,
        SPLIT,
        "current_VISCODE2",
        "current_protocol_month",
        "current_eligible_visit_position",
        "current_visit_order_basis",
    ]
    output = pairs[columns].copy()
    output["scenario"] = "snapshot_matched_last"
    return output.sort_values([SPLIT, RID], kind="stable").reset_index(drop=True)


def apply_scalar_transform(
    series: pd.Series,
    transform_id: str,
    parameters: Mapping[str, Any],
    full_column_name: str,
) -> tuple[pd.Series, int]:
    if transform_id in IDENTITY_TRANSFORMS or transform_id == "dictionary_multi_response_tokens":
        return series.copy(), 0

    if transform_id in {
        "faq_official_item_score_map",
        "cdr_source_mode_harmonization",
        "ptdemog_pthome_harmonization",
    }:
        normalized = series.map(normalize_code)
        mapping = {str(key): value for key, value in parameters.items()}
        transformed = normalized.map(mapping)
        unmapped = normalized.notna() & transformed.isna()
        return transformed, int(unmapped.sum())

    if transform_id == "extract_year_from_jan1_date_placeholder":
        output = pd.Series(pd.NA, index=series.index, dtype="Float64")
        failures = 0
        for index, value in series.items():
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            match = YEAR_PLACEHOLDER_PATTERN.match(str(value))
            if match is None:
                failures += 1
                continue
            output.at[index] = float(int(match.group(1)))
        return output, failures

    raise ScenarioAbort(
        f"Unsupported transform_id {transform_id!r} for {full_column_name}."
    )


def transform_and_validate_type(
    series: pd.Series,
    feature_type: str,
    transform_id: str,
    parameters: Mapping[str, Any],
    full_column_name: str,
) -> tuple[pd.Series, int, int]:
    transformed, transform_failures = apply_scalar_transform(
        series,
        transform_id,
        parameters,
        full_column_name,
    )
    type_failures = 0

    if feature_type == "numeric":
        numeric = pd.to_numeric(transformed, errors="coerce")
        type_failures = int(transformed.notna().sum() - numeric.notna().sum())
        return numeric, transform_failures, type_failures

    if feature_type == "binary":
        normalized = transformed.map(normalize_code)
        invalid = normalized.notna() & ~normalized.isin(["0", "1"])
        type_failures = int(invalid.sum())
        output = pd.to_numeric(normalized.where(~invalid), errors="coerce")
        return output, transform_failures, type_failures

    if feature_type == "categorical":
        output = transformed.astype("string")
        output = output.mask(output.str.strip().eq(""), pd.NA)
        return output, transform_failures, 0

    if feature_type == "multi_response":
        parsed_nonempty = transformed.map(lambda value: bool(parse_multi_tokens(value)))
        invalid = transformed.notna() & ~parsed_nonempty
        type_failures = int(invalid.sum())
        output = transformed.where(~invalid)
        return output, transform_failures, type_failures

    raise ScenarioAbort(
        f"Unsupported feature type {feature_type!r} for {full_column_name}."
    )


def registry_lookup(registry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["full_column_name"]): row
        for row in registry.to_dict(orient="records")
    }


def align_source_to_manifest(
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    source_name: str,
    block: str,
    features: Sequence[str],
) -> pd.DataFrame:
    manifest_work = manifest.reset_index(drop=True).copy()
    manifest_work["__row_order"] = np.arange(
        len(manifest_work), dtype=np.int64
    )

    if source_name in VISIT_SOURCES:
        visit_column = (
            "previous_VISCODE2" if block == "previous_visit" else "current_VISCODE2"
        )
        keys = manifest_work[
            [RID, SPLIT, visit_column, "__row_order"]
        ].copy().rename(columns={visit_column: VISCODE2})
        if keys.duplicated([RID, VISCODE2]).any():
            raise ScenarioAbort(
                f"Manifest contains duplicate visit keys for {source_name}/{block}."
            )
        aligned = keys.merge(
            source[[RID, VISCODE2, SPLIT, *features]],
            on=[RID, VISCODE2, SPLIT],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    else:
        keys = manifest_work[[RID, SPLIT, "__row_order"]].drop_duplicates(RID)
        aligned = keys.merge(
            source[[RID, SPLIT, *features]],
            on=[RID, SPLIT],
            how="left",
            validate="one_to_one",
            sort=False,
        )
    aligned = aligned.sort_values("__row_order", kind="stable").reset_index(
        drop=True
    )
    aligned = aligned.drop(columns=["__row_order"])
    if len(aligned) != len(keys):
        raise ScenarioAbort(
            f"Alignment row count changed for {source_name}/{block}: "
            f"{len(keys)} -> {len(aligned)}"
        )
    return aligned


def compute_support_and_transform_audit(
    scenario_name: str,
    manifest: pd.DataFrame,
    block: str,
    source_names: Sequence[str],
    source_frames: Mapping[str, pd.DataFrame],
    schema: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = registry_lookup(registry)
    support_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []

    for source_name in source_names:
        source_schema = schema.loc[
            schema["source_name"].eq(source_name)
            & schema["official_semantic_include"]
            & schema["current_in_crosschecked_inputs"]
            & ~schema["derived_later"]
        ].copy()
        features = source_schema["column_name"].astype(str).tolist()
        aligned = align_source_to_manifest(
            manifest,
            source_frames[source_name],
            source_name,
            block,
            features,
        )
        aligned = apply_temporal_exclusion_to_aligned(
            scenario_name,
            manifest,
            aligned,
            source_name,
            block,
            features,
        )

        for split_label in DEVELOPMENT_SPLITS:
            split_frame = aligned.loc[aligned[SPLIT].eq(split_label)]
            modality_present = split_frame[features].notna().any(axis=1)
            modality_rows.append(
                {
                    "scenario": scenario_name,
                    "block": block,
                    "source_name": source_name,
                    "split": split_label,
                    "rows_or_participants": int(len(split_frame)),
                    "modality_present_count": int(modality_present.sum()),
                    "modality_present_pct": (
                        100.0 * float(modality_present.mean())
                        if len(split_frame)
                        else np.nan
                    ),
                }
            )

        for row in source_schema.to_dict(orient="records"):
            column_name = str(row["column_name"])
            full_name = str(row["full_column_name"])
            feature_type = str(row["official_feature_type"])
            transform_id = str(row["official_value_transform_id"] or "identity")
            parameters: Mapping[str, Any] = {}
            if full_name in lookup:
                transform_id = str(lookup[full_name]["transform_id"])
                parameters = lookup[full_name]["parameters"]

            transformed_by_split: dict[str, pd.Series] = {}
            for split_label in DEVELOPMENT_SPLITS:
                split_index = aligned[SPLIT].eq(split_label)
                raw = aligned.loc[split_index, column_name]
                transformed, transform_failures, type_failures = (
                    transform_and_validate_type(
                        raw,
                        feature_type,
                        transform_id,
                        parameters,
                        full_name,
                    )
                )
                transformed_by_split[split_label] = transformed
                audit_rows.append(
                    {
                        "scenario": scenario_name,
                        "block": block,
                        "split": split_label,
                        "source_name": source_name,
                        "column_name": column_name,
                        "full_column_name": full_name,
                        "official_feature_type": feature_type,
                        "transform_id": transform_id,
                        "nonmissing_input_rows": int(raw.notna().sum()),
                        "nonmissing_output_rows": int(transformed.notna().sum()),
                        "transform_failure_rows": int(transform_failures),
                        "type_validation_failure_rows": int(type_failures),
                        "validation_passed": bool(
                            transform_failures == 0 and type_failures == 0
                        ),
                    }
                )

            train_aligned = aligned.loc[
                aligned[SPLIT].eq(TRAIN), [RID, column_name]
            ].copy()
            train_aligned["__observed"] = train_aligned[column_name].notna()
            support = int(
                train_aligned.loc[train_aligned["__observed"], RID].nunique()
            )
            train_participants = int(train_aligned[RID].nunique())
            support_rows.append(
                {
                    "scenario": scenario_name,
                    "block": block,
                    "model_feature_prefix": (
                        ""
                        if source_name in PARTICIPANT_SOURCES
                        or scenario_name != "longitudinal_previous_last"
                        else "previous__"
                        if block == "previous_visit"
                        else "current__"
                    ),
                    "source_name": source_name,
                    "column_name": column_name,
                    "full_column_name": full_name,
                    "official_feature_type": feature_type,
                    "official_encoding": str(row["official_encoding"]),
                    "official_semantic_role": str(row["official_semantic_role"]),
                    "transform_id": transform_id,
                    "train_participant_count": train_participants,
                    "train_observed_participants": support,
                    "train_observed_participant_fraction": (
                        support / train_participants if train_participants else np.nan
                    ),
                    "support_threshold_participants": SUPPORT_THRESHOLD,
                    "retained_at_support_threshold": bool(
                        support >= SUPPORT_THRESHOLD
                    ),
                    "support_rule": (
                        "distinct_fit_participants_with_original_feature_observed_"
                        "before_imputation_and_encoding"
                    ),
                }
            )

    return (
        pd.DataFrame(support_rows),
        pd.DataFrame(audit_rows),
        pd.DataFrame(modality_rows),
    )


def scenario_split_summary(
    snapshot_all: pd.DataFrame,
    matched: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, frame in [
        ("snapshot_all", snapshot_all),
        ("snapshot_matched_last", matched),
        ("longitudinal_previous_last", pairs),
    ]:
        for split_label in DEVELOPMENT_SPLITS:
            subset = frame.loc[frame[SPLIT].eq(split_label)]
            rows.append(
                {
                    "scenario": scenario_name,
                    "split": split_label,
                    "rows": int(len(subset)),
                    "participants": int(subset[RID].nunique()),
                    "one_row_per_participant": bool(
                        not subset.duplicated(RID).any()
                    ),
                    "current_target_visits": int(len(subset)),
                    "previous_visits": int(
                        len(subset)
                        if scenario_name == "longitudinal_previous_last"
                        else 0
                    ),
                    "test_rows_used": 0,
                }
            )
    return pd.DataFrame(rows)


def pairing_attrition_summary(
    development_anchor: pd.DataFrame,
    pairs: pd.DataFrame,
    exclusions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    current_participants = (
        development_anchor[[RID, SPLIT]].drop_duplicates(RID)
    )
    for split_label in DEVELOPMENT_SPLITS:
        total = int(current_participants[SPLIT].eq(split_label).sum())
        split_pairs = pairs.loc[pairs[SPLIT].eq(split_label)]
        for reason, count in (
            split_pairs["pairing_reason"].value_counts().sort_index().items()
        ):
            rows.append(
                {
                    "split": split_label,
                    "status": "eligible",
                    "reason": str(reason),
                    "participant_count": int(count),
                    "development_participant_count": total,
                    "participant_pct": 100.0 * int(count) / total if total else np.nan,
                }
            )
        split_exclusions = exclusions.loc[exclusions[SPLIT].eq(split_label)]
        for reason, count in (
            split_exclusions["exclusion_reason"].value_counts().sort_index().items()
        ):
            rows.append(
                {
                    "split": split_label,
                    "status": "excluded",
                    "reason": str(reason),
                    "participant_count": int(count),
                    "development_participant_count": total,
                    "participant_pct": 100.0 * int(count) / total if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def temporal_gap_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_label in DEVELOPMENT_SPLITS:
        values = pd.to_numeric(
            pairs.loc[pairs[SPLIT].eq(split_label), "gap_months"],
            errors="raise",
        ).astype(float)
        rows.append(
            {
                "split": split_label,
                "pair_count": int(len(values)),
                "gap_months_min": float(values.min()) if len(values) else np.nan,
                "gap_months_q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "gap_months_median": float(values.median()) if len(values) else np.nan,
                "gap_months_mean": float(values.mean()) if len(values) else np.nan,
                "gap_months_q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "gap_months_max": float(values.max()) if len(values) else np.nan,
                "strictly_positive": bool(values.gt(0).all()) if len(values) else False,
            }
        )
    return pd.DataFrame(rows)


def support_summary(support_manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        support_manifest.groupby(
            [
                "scenario",
                "block",
                "source_name",
                "official_feature_type",
                "retained_at_support_threshold",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="feature_count")
        .sort_values(
            [
                "scenario",
                "block",
                "source_name",
                "official_feature_type",
                "retained_at_support_threshold",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def transform_validation_summary(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby(
            ["scenario", "block", "split"],
            dropna=False,
        )
        .agg(
            audited_feature_count=("full_column_name", "nunique"),
            nonmissing_input_rows=("nonmissing_input_rows", "sum"),
            nonmissing_output_rows=("nonmissing_output_rows", "sum"),
            transform_failure_rows=("transform_failure_rows", "sum"),
            type_validation_failure_rows=("type_validation_failure_rows", "sum"),
            failed_feature_count=(
                "validation_passed",
                lambda values: int((~pd.Series(values).astype(bool)).sum()),
            ),
        )
        .reset_index()
    )


def build_scenario_definitions(
    decision_id: str,
    schema_lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scenario_definition_version": SCENARIO_VERSION,
        "scenario_decision_id": decision_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_schema_decision_id": schema_lock.get("official_decision_id"),
        "development_splits": list(DEVELOPMENT_SPLITS),
        "test_materialized": False,
        "test_policy": (
            "Apply the same frozen scenario rules to test only after model, "
            "preprocessing and target-representation configurations are locked."
        ),
        "support_threshold_participants": SUPPORT_THRESHOLD,
        "support_application": (
            "recomputed_from_distinct_train_participants_separately_by_"
            "scenario_and_temporal_block_before_encoding"
        ),
        "scenarios": {
            "snapshot_all": {
                "population": "all target-complete development visits",
                "current_visit_selection": "every supervised anchor visit",
                "visit_level_input": "same VISCODE2 as target visit",
                "participant_level_input": "PTDEMOG and APOERES once by RID",
                "visit_derived_input": "AGE_AT_TARGET once for the current target visit",
                "historical_input": False,
            },
            "snapshot_matched_last": {
                "population": (
                    "one current target visit for each participant retained after "
                    "the frozen SPLIT07 pairing, crosschecked-content requirement, and A05 temporal source exclusions"
                ),
                "current_visit_selection": (
                    "maximum frozen eligible_visit_position; no fallback to an "
                    "earlier target visit when the last target is ineligible"
                ),
                "visit_level_input": "same current VISCODE2 only",
                "participant_level_input": "PTDEMOG and APOERES once by RID",
                "visit_derived_input": "AGE_AT_TARGET once for the current target visit",
                "historical_input": False,
            },
            "longitudinal_previous_last": {
                "population": (
                    "identical participants and current target visits as "
                    "snapshot_matched_last"
                ),
                "current_block": "same current inputs as snapshot_matched_last",
                "previous_eligibility": (
                    "at least one semantically authorized and temporally safe "
                    "visit-level feature remains after crosscheck and A05 exclusions"
                ),
                "previous_selection": (
                    "consume the exact previous VISCODE2 frozen by SPLIT07; same-wave ties "
                    "are resolved only by strict non-overlapping clinical-date intervals"
                ),
                "same_previous_visit_for_all_modalities": True,
                "modality_specific_history_search": False,
                "carry_forward": False,
                "temporally_unsafe_previous_source_blocks": (
                    "represented as missing when collected on or after the "
                    "current target MRI EXAMDATE"
                ),
                "pair_retention_after_source_exclusion": (
                    "retain only if at least one temporally safe authorized "
                    "previous visit-level feature remains"
                ),
                "previous_target_complete_required": False,
                "MRI_in_predictors": False,
                "gap_feature": "current_protocol_month - previous_protocol_month",
                "participant_level_input": "PTDEMOG and APOERES once by RID",
                "visit_derived_input": "AGE_AT_TARGET once for the current target visit",
            },
        },
        "temporal_rules": {
            "protocol_month_parser": {
                "baseline_or_screening_codes": ["bl", "sc", "scmri", "m00", "v01"],
                "baseline_or_screening_month": 0,
                "month_pattern": "mNN -> NN months",
                "year_pattern": "yNN -> 12*NN months",
            },
            "unknown_protocol_codes": "not eligible as previous visits",
            "same_month_multiple_VISCODE2": (
                "consume SPLIT07: select only a strict date-interval winner; "
                "otherwise exclude from matched scenarios"
            ),
            "same_wave_visits_concatenated": False,
            "viscode_hierarchy_fallback": False,
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "MRI_exam_dates_used_for_gap": False,
            "study_phase_exposed_to_model": False,
        },
        "age_at_target_rule": {
            "source": "A06 local-only baseline birth-year metadata + current target MRI EXAMDATE",
            "formula": "calendar_year(current_target_mri_examdate) - birth_year",
            "precision": "calendar_year",
            "current_visit_only": True,
            "previous_age_feature_created": False,
            "raw_birth_fields_exposed_to_model": False,
            "safe_reports_contain_age_values": False,
        },
    }


def build_story(
    scenario_summary_frame: pd.DataFrame,
    attrition: pd.DataFrame,
    gap_summary: pd.DataFrame,
    support_manifest: pd.DataFrame,
    local_hashes: Mapping[str, str],
) -> str:
    matched_train = scenario_summary_frame.loc[
        scenario_summary_frame["scenario"].eq("snapshot_matched_last")
        & scenario_summary_frame[SPLIT].eq(TRAIN)
    ].iloc[0]
    matched_val = scenario_summary_frame.loc[
        scenario_summary_frame["scenario"].eq("snapshot_matched_last")
        & scenario_summary_frame[SPLIT].eq(VALIDATION)
    ].iloc[0]

    retained = (
        support_manifest.groupby(["scenario", "block"])[
            "retained_at_support_threshold"
        ]
        .sum()
        .astype(int)
        .reset_index(name="retained_feature_instances")
    )
    retained_lines = [
        "| Cenário | Bloco | Feature instances retidas |",
        "|---|---|---:|",
    ]
    for row in retained.itertuples(index=False):
        retained_lines.append(
            f"| `{row.scenario}` | `{row.block}` | {row.retained_feature_instances} |"
        )

    attrition_lines = [
        "| Split | Status | Motivo | Participantes |",
        "|---|---|---|---:|",
    ]
    for row in attrition.itertuples(index=False):
        attrition_lines.append(
            f"| {row.split} | {row.status} | `{row.reason}` | {row.participant_count} |"
        )

    gap_lines = [
        "| Split | Pares | Mediana (meses) | Média | Mínimo | Máximo |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in gap_summary.itertuples(index=False):
        gap_lines.append(
            f"| {row.split} | {row.pair_count} | {row.gap_months_median:.2f} | "
            f"{row.gap_months_mean:.2f} | {row.gap_months_min:.2f} | "
            f"{row.gap_months_max:.2f} |"
        )

    return f"""# BENCHMARK05B — Construção auditada dos cenários

- Status esperado após execução: **PASS**
- Dados utilizados: **treino e validação somente**
- Teste materializado: **não**
- Modelos treinados: **não**
- Threshold congelado: **50 participantes distintos do fit**
- Participantes pareados no treino: **{int(matched_train['participants'])}**
- Participantes pareados na validação: **{int(matched_val['participants'])}**

## Cenários

### `snapshot_all`

Inclui todas as visitas target-complete de treino e validação. Cada linha recebe
somente as entradas não-MRI da própria visita, além de PTDEMOG e APOERES por
participante.

### `snapshot_matched_last`

Seleciona a última visita target-complete de cada participante segundo a ordem
congelada do pacote. O participante só entra se essa visita atual possuir uma
visita não-MRI anterior elegível. Não existe fallback para uma visita-alvo mais
antiga quando a última visita não pode ser pareada.

### `longitudinal_previous_last`

Usa exatamente os mesmos participantes, visitas atuais e targets de
`snapshot_matched_last`. Acrescenta uma única visita anterior comum a todas as
modalidades, consumindo o VISCODE2 exato congelado pelo `SPLIT07`. Quando havia
mais de uma candidata na mesma onda, somente intervalos clínicos estritamente
ordenados puderam desempatar.

A visita anterior não precisa possuir MRI nem target completo. Ela precisa ter
pelo menos uma feature visit-level semanticamente autorizada observada. Não há
carry-forward e não se busca uma visita histórica diferente para cada modalidade.
Blocos de uma fonte anterior cuja data clínica ocorreu no mesmo dia ou depois da
MRI atual são representados como ausentes antes do cálculo de suporte. Um par só é
mantido se ainda restar alguma entrada anterior temporalmente segura.

## Attrition do pareamento

{chr(10).join(attrition_lines)}

## Intervalo temporal

{chr(10).join(gap_lines)}

## Suporte por cenário e bloco

A regra de 50 é recalculada usando somente participantes do treino. PTDEMOG e
APOERES formam o bloco `participant_static` e aparecem uma única vez. No cenário
longitudinal, `current_visit` e `previous_visit` têm suporte calculado
separadamente.

{chr(10).join(retained_lines)}

## Decisões conservadoras

- códigos de visita sem mês protocolar interpretável não são usados como histórico;
- `sc` e `bl` permanecem visitas exatas distintas e nunca são concatenadas;
- empates na mesma onda são resolvidos apenas por intervalos de datas estritos;
- casos ainda inconclusivos são excluídos, sem mediana, ordem lexical ou hierarquia;
- a população do cenário estático pareado é idêntica à longitudinal;
- o teste permanece fechado e será materializado somente após o lock final dos modelos.

## Artefatos local-only

Os manifests com RID/VISCODE2 foram escritos em
`data/processed/local_only/benchmark05b_scenarios/`. Eles não podem ser enviados,
publicados ou commitados.

Hashes globais dos manifests locais:

```json
{json.dumps(dict(local_hashes), ensure_ascii=False, indent=2)}
```
"""


def run_stage(
    paths: Paths,
    checks: Checks,
    superseded_scenario_decision_id: str | None,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    (
        benchmark01,
        benchmark04b,
        support_policy,
        schema_lock,
        crosschecked_manifest,
    ) = validate_upstream(paths, checks)
    temporal_lock, package_manifest = load_temporal_contract(paths, checks)
    schema = load_official_schema(paths, checks)
    registry = load_transform_registry(paths, schema, checks)
    upstream_decision_id = str(schema_lock.get("official_decision_id", ""))
    schema_decision_ids = set(
        schema["official_decision_id"].dropna().astype(str).unique().tolist()
    )
    registry_decision_ids = set(
        registry["official_decision_id"].dropna().astype(str).unique().tolist()
    )
    checks.add(
        "official_schema_decision_id_matches_lock",
        schema_decision_ids == {upstream_decision_id},
        [upstream_decision_id],
        sorted(schema_decision_ids),
    )
    checks.add(
        "official_transform_decision_id_matches_lock",
        registry_decision_ids == {upstream_decision_id},
        [upstream_decision_id],
        sorted(registry_decision_ids),
    )
    if not checks.passed:
        raise ScenarioAbort("Official schema/transform decision IDs do not match the lock.")
    _, development_anchor, target_keys = load_anchor_and_target_keys(
        paths, benchmark01, checks
    )
    age_table, age_derivation_audit = load_age_derivation_table(paths, development_anchor, checks)
    source_frames = load_source_frames(paths, schema, checks)
    timeline = build_visit_availability_timeline(source_frames, schema, checks)

    snapshot_all = build_snapshot_all(development_anchor)
    pairs, exclusions = build_matched_pairs(
        development_anchor, timeline, paths, checks
    )
    temporal_exclusion_manifest, temporal_exclusion_summary = (
        load_temporal_modality_exclusions(paths, checks)
    )
    pairs, exclusions = apply_temporal_modality_exclusions_to_pairs(
        pairs,
        exclusions,
        temporal_exclusion_manifest,
        source_frames,
        schema,
        checks,
    )
    matched = build_snapshot_matched(pairs)

    snapshot_all = attach_age_at_target(
        snapshot_all, age_table, checks, "snapshot_all"
    )
    matched = attach_age_at_target(
        matched, age_table, checks, "snapshot_matched_last"
    )
    pairs = attach_age_at_target(
        pairs, age_table, checks, "longitudinal_previous_last"
    )

    for scenario_name, frame in [
        ("snapshot_all", snapshot_all),
        ("snapshot_matched_last", matched),
        ("longitudinal_previous_last", pairs),
    ]:
        missing_age = int(frame["AGE_AT_TARGET"].isna().sum())
        checks.add(
            f"{scenario_name}__age_at_target_complete",
            missing_age == 0,
            0,
            missing_age,
        )

    snapshot_all_keys = set(
        zip(
            snapshot_all[RID].astype(str),
            snapshot_all["current_VISCODE2"].astype(str),
        )
    )
    matched_keys = set(
        zip(matched[RID].astype(str), matched["current_VISCODE2"].astype(str))
    )
    pair_current_keys = set(
        zip(pairs[RID].astype(str), pairs["current_VISCODE2"].astype(str))
    )
    target_key_set = set(
        zip(target_keys[RID].astype(str), target_keys[VISCODE2].astype(str))
    )

    checks.add(
        "snapshot_all_equals_development_anchor",
        len(snapshot_all) == len(development_anchor)
        and snapshot_all_keys
        == set(
            zip(
                development_anchor[RID].astype(str),
                development_anchor[VISCODE2].astype(str),
            )
        ),
        True,
        len(snapshot_all) == len(development_anchor),
    )
    checks.add(
        "matched_current_subset_snapshot_all",
        matched_keys.issubset(snapshot_all_keys),
        True,
        matched_keys.issubset(snapshot_all_keys),
    )
    checks.add(
        "matched_and_longitudinal_current_keys_identical",
        matched_keys == pair_current_keys,
        True,
        matched_keys == pair_current_keys,
    )
    checks.add(
        "all_scenario_current_keys_have_targets",
        snapshot_all_keys.issubset(target_key_set)
        and matched_keys.issubset(target_key_set),
        True,
        snapshot_all_keys.issubset(target_key_set)
        and matched_keys.issubset(target_key_set),
    )
    checks.add(
        "matched_one_row_per_participant",
        not matched.duplicated(RID).any(),
        True,
        bool(not matched.duplicated(RID).any()),
    )
    checks.add(
        "development_pair_splits_nonempty",
        all(pairs[SPLIT].eq(split).any() for split in DEVELOPMENT_SPLITS),
        list(DEVELOPMENT_SPLITS),
        pairs[SPLIT].value_counts().to_dict(),
    )

    support_frames: list[pd.DataFrame] = []
    transform_audits: list[pd.DataFrame] = []
    modality_frames: list[pd.DataFrame] = []

    block_specs = [
        (
            "snapshot_all",
            snapshot_all,
            "current_visit",
            VISIT_SOURCES,
        ),
        (
            "snapshot_all",
            snapshot_all,
            "participant_static",
            PARTICIPANT_SOURCES,
        ),
        (
            "snapshot_matched_last",
            matched,
            "current_visit",
            VISIT_SOURCES,
        ),
        (
            "snapshot_matched_last",
            matched,
            "participant_static",
            PARTICIPANT_SOURCES,
        ),
        (
            "longitudinal_previous_last",
            pairs,
            "current_visit",
            VISIT_SOURCES,
        ),
        (
            "longitudinal_previous_last",
            pairs,
            "previous_visit",
            VISIT_SOURCES,
        ),
        (
            "longitudinal_previous_last",
            pairs,
            "participant_static",
            PARTICIPANT_SOURCES,
        ),
    ]
    for scenario_name, manifest, block, sources in block_specs:
        logging.info(
            "Computing support and transform audit: %s / %s",
            scenario_name,
            block,
        )
        support, audit, modality = compute_support_and_transform_audit(
            scenario_name,
            manifest,
            block,
            sources,
            source_frames,
            schema,
            registry,
        )
        support_frames.append(support)
        transform_audits.append(audit)
        modality_frames.append(modality)

    for scenario_name, manifest in [
        ("snapshot_all", snapshot_all),
        ("snapshot_matched_last", matched),
        ("longitudinal_previous_last", pairs),
    ]:
        age_support, age_audit, age_modality = compute_age_support_and_audit(
            scenario_name,
            manifest,
            schema,
        )
        support_frames.append(age_support)
        transform_audits.append(age_audit)
        modality_frames.append(age_modality)

    support_manifest = pd.concat(support_frames, ignore_index=True, sort=False)
    transform_audit = pd.concat(transform_audits, ignore_index=True, sort=False)
    modality_summary = pd.concat(modality_frames, ignore_index=True, sort=False)

    included_schema = schema.loc[schema["official_semantic_include"]].copy()
    visit_feature_count = int(
        included_schema["source_name"].isin(VISIT_SOURCES).sum()
    )
    included_feature_count = int(len(included_schema))
    expected_support_rows = {
        "snapshot_all": included_feature_count,
        "snapshot_matched_last": included_feature_count,
        "longitudinal_previous_last": included_feature_count + visit_feature_count,
    }
    observed_support_rows = {
        scenario_name: int(support_manifest["scenario"].eq(scenario_name).sum())
        for scenario_name in expected_support_rows
    }
    checks.add(
        "support_manifest_feature_coverage",
        observed_support_rows == expected_support_rows,
        expected_support_rows,
        observed_support_rows,
    )
    checks.add(
        "support_manifest_excludes_semantically_excluded_features",
        set(
            support_manifest["full_column_name"].astype(str)
        ).isdisjoint(
            set(
                schema.loc[
                    ~schema["official_semantic_include"],
                    "full_column_name",
                ].astype(str)
            )
        ),
        True,
        set(
            support_manifest["full_column_name"].astype(str)
        ).isdisjoint(
            set(
                schema.loc[
                    ~schema["official_semantic_include"],
                    "full_column_name",
                ].astype(str)
            )
        ),
    )

    total_transform_failures = int(transform_audit["transform_failure_rows"].sum())
    total_type_failures = int(
        transform_audit["type_validation_failure_rows"].sum()
    )
    checks.add(
        "frozen_transforms_cover_all_development_scenario_values",
        total_transform_failures == 0,
        0,
        total_transform_failures,
    )
    checks.add(
        "scenario_values_match_frozen_feature_types",
        total_type_failures == 0,
        0,
        total_type_failures,
    )

    matched_current_support = support_manifest.loc[
        support_manifest["scenario"].eq("snapshot_matched_last")
        & support_manifest["block"].eq("current_visit")
    ].sort_values("full_column_name").reset_index(drop=True)
    long_current_support = support_manifest.loc[
        support_manifest["scenario"].eq("longitudinal_previous_last")
        & support_manifest["block"].eq("current_visit")
    ].sort_values("full_column_name").reset_index(drop=True)
    support_compare_columns = [
        "full_column_name",
        "train_observed_participants",
        "retained_at_support_threshold",
    ]
    checks.add(
        "matched_and_longitudinal_current_support_identical",
        matched_current_support[support_compare_columns].equals(
            long_current_support[support_compare_columns]
        ),
        True,
        matched_current_support[support_compare_columns].equals(
            long_current_support[support_compare_columns]
        ),
    )
    matched_static_support = support_manifest.loc[
        support_manifest["scenario"].eq("snapshot_matched_last")
        & support_manifest["block"].eq("participant_static")
    ].sort_values("full_column_name").reset_index(drop=True)
    long_static_support = support_manifest.loc[
        support_manifest["scenario"].eq("longitudinal_previous_last")
        & support_manifest["block"].eq("participant_static")
    ].sort_values("full_column_name").reset_index(drop=True)
    checks.add(
        "matched_and_longitudinal_static_support_identical",
        matched_static_support[support_compare_columns].equals(
            long_static_support[support_compare_columns]
        ),
        True,
        matched_static_support[support_compare_columns].equals(
            long_static_support[support_compare_columns]
        ),
    )

    matched_age_support = support_manifest.loc[
        support_manifest["scenario"].eq("snapshot_matched_last")
        & support_manifest["block"].eq("current_derived")
    ].sort_values("full_column_name").reset_index(drop=True)
    long_age_support = support_manifest.loc[
        support_manifest["scenario"].eq("longitudinal_previous_last")
        & support_manifest["block"].eq("current_derived")
    ].sort_values("full_column_name").reset_index(drop=True)
    checks.add(
        "matched_and_longitudinal_age_support_identical",
        matched_age_support[support_compare_columns].equals(
            long_age_support[support_compare_columns]
        ),
        True,
        matched_age_support[support_compare_columns].equals(
            long_age_support[support_compare_columns]
        ),
    )

    if not checks.passed:
        raise ScenarioAbort(
            f"Scenario validation failed with {checks.failed} failed checks."
        )

    # Local-only individualized artifacts.
    readme = """BENCHMARK05B SCENARIO ARTIFACTS — LOCAL ONLY

This directory contains individualized ADNI-derived keys, longitudinal
pairings, and visit-specific AGE_AT_TARGET values. Do not commit, upload,
publish or share any file from this directory.

- snapshot_all_development_anchor.csv.gz
- snapshot_matched_last_development_anchor.csv.gz
- longitudinal_previous_last_development_pairs.csv.gz
- development_visit_availability_timeline.csv.gz
- pairing_exclusions.csv.gz
- LOCAL_ONLY_age_at_target_derivation.csv.gz

Only train and validation are materialized. Test remains closed.
"""
    atomic_write_text(paths.local / "README_LOCAL_ONLY.txt", readme)
    local_files = {
        "snapshot_all_development_anchor.csv.gz": snapshot_all,
        "snapshot_matched_last_development_anchor.csv.gz": matched,
        "longitudinal_previous_last_development_pairs.csv.gz": pairs,
        "development_visit_availability_timeline.csv.gz": timeline,
        "pairing_exclusions.csv.gz": exclusions,
        "LOCAL_ONLY_age_at_target_derivation.csv.gz": age_derivation_audit,
    }
    for filename, frame in local_files.items():
        atomic_write_csv_gzip(frame, paths.local / filename)

    local_hashes = {
        filename: sha256_file(paths.local / filename)
        for filename in local_files
    }

    upstream_hashes = {
        "benchmark05b_script_sha256": sha256_file(Path(__file__).resolve()),
        "benchmark01_summary_sha256": sha256_file(paths.benchmark01_summary),
        "benchmark04b_summary_sha256": sha256_file(paths.benchmark04b_summary),
        "official_schema_sha256": sha256_file(paths.official_schema),
        "official_transforms_sha256": sha256_file(paths.official_transforms),
        "support_policy_sha256": sha256_file(paths.support_policy),
        "schema_lock_sha256": sha256_file(paths.schema_lock),
        "crosschecked_manifest_sha256": sha256_file(paths.crosschecked_manifest),
        "age_metadata_sha256": sha256_file(paths.age_metadata),
        "mri_canonical_sha256": sha256_file(paths.mri_canonical),
        "anchor_sha256": sha256_file(paths.package / ARTIFACTS["anchor"]),
        "target_keys_table_sha256": sha256_file(paths.package / ARTIFACTS["targets"]),
        "temporal_pairing_reference_sha256": sha256_file(
            paths.temporal_pairing_reference
        ),
        "temporal_policy_lock_sha256": sha256_file(paths.temporal_policy_lock),
        "package_manifest_sha256": sha256_file(paths.package_manifest),
        "auditA05_summary_sha256": sha256_file(
            paths.temporal_exclusion_summary
        ),
        "auditA05_local_manifest_sha256": sha256_file(
            paths.temporal_exclusion_manifest
        ),
    }
    decision_material = json.dumps(
        {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "scenario_version": SCENARIO_VERSION,
            "upstream_hashes": upstream_hashes,
            "support_threshold": SUPPORT_THRESHOLD,
            "pairing_rule": (
                "consume_SPLIT07_strict_date_interval_same_wave_previous_visit_v1"
            ),
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "previous_source_temporal_exclusion_policy": (
                "set_affected_previous_source_blocks_missing_when_collected_"
                "on_or_after_current_target_mri"
            ),
            "auditA05_local_manifest_sha256": sha256_file(
                paths.temporal_exclusion_manifest
            ),
            "age_at_target_policy": (
                "calendar_year(current_target_mri_examdate)-baseline_birth_year;"
                "current_visit_only;raw_birth_fields_not_predictors"
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    scenario_decision_id = sha256_text(decision_material)[:20]

    definitions = build_scenario_definitions(
        scenario_decision_id,
        schema_lock,
    )
    definitions["local_artifact_hashes"] = local_hashes
    definitions["upstream_hashes"] = upstream_hashes
    definitions["upstream_temporal_policy_decision_id"] = (
        EXPECTED_TEMPORAL_POLICY_DECISION_ID
    )
    definitions["supersedes_scenario_decision_id"] = (
        superseded_scenario_decision_id
    )
    definitions["temporal_source_exclusions"] = {
        "audit_stage": "auditA05_temporal_modality_exclusions",
        "audit_script_version": temporal_exclusion_summary.get(
            "script_version"
        ),
        "local_manifest_sha256": sha256_file(
            paths.temporal_exclusion_manifest
        ),
        "policy": (
            "affected previous visit-level source blocks are represented as "
            "missing when their clinical date is on or after the current "
            "target MRI EXAMDATE"
        ),
        "pair_removed_only_if_no_safe_previous_input_remains": True,
        "participant_split_changed": False,
    }

    # Official safe scenario artifacts.
    atomic_write_json(paths.scenario_definitions, definitions)
    atomic_write_csv(
        support_manifest.sort_values(
            ["scenario", "block", "source_name", "column_name"],
            kind="stable",
        ).reset_index(drop=True),
        paths.scenario_support_manifest,
    )
    scenario_lock_payload = {
        "lock_version": "2.0.0-crosschecked-b",
        "scenario_decision_id": scenario_decision_id,
        "created_at_utc": created_at,
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "scenario_definition_version": SCENARIO_VERSION,
        "upstream_schema_decision_id": schema_lock.get("official_decision_id"),
        "upstream_temporal_policy_decision_id": (
            EXPECTED_TEMPORAL_POLICY_DECISION_ID
        ),
        "supersedes_scenario_decision_id": superseded_scenario_decision_id,
        "support_threshold_participants": SUPPORT_THRESHOLD,
        "development_splits": list(DEVELOPMENT_SPLITS),
        "test_materialized": False,
        "test_used_for_scenario_development": False,
        "age_at_target": {
            "materialized": True,
            "current_visit_only": True,
            "previous_age_created": False,
            "raw_birth_fields_exposed": False,
            "derivation": "calendar_year(current_target_mri_examdate)-baseline_birth_year",
            "birth_metadata_sha256": sha256_file(paths.age_metadata),
            "mri_date_source_sha256": sha256_file(paths.mri_canonical),
        },
        "temporal_source_exclusions": {
            "audit_stage": "auditA05_temporal_modality_exclusions",
            "audit_summary_sha256": sha256_file(
                paths.temporal_exclusion_summary
            ),
            "local_manifest_sha256": sha256_file(
                paths.temporal_exclusion_manifest
            ),
            "applied_before_support_and_transform_audit": True,
            "pair_removed_only_if_no_safe_previous_input_remains": True,
            "participant_split_changed": False,
        },
        "local_artifact_hashes": local_hashes,
        "official_artifact_hashes": {
            "benchmark_scenario_definitions_b.json": sha256_file(
                paths.scenario_definitions
            ),
            "benchmark_scenario_support_manifest_b.csv": sha256_file(
                paths.scenario_support_manifest
            ),
        },
        "upstream_hashes": upstream_hashes,
        "methodological_commitment": (
            "Do not change scenario eligibility, current/previous selection, "
            "support application or ambiguity policy after test inspection."
        ),
    }
    if paths.scenario_lock.exists():
        existing_lock = read_json(paths.scenario_lock)
        if existing_lock.get("scenario_decision_id") != scenario_decision_id:
            raise ScenarioAbort(
                "An incompatible benchmark_scenario_lock_b.json already exists. "
                "Refusing to overwrite the frozen scenario decision."
            )
    atomic_write_json(paths.scenario_lock, scenario_lock_payload)

    split_summary_frame = scenario_split_summary(snapshot_all, matched, pairs)
    attrition = pairing_attrition_summary(
        development_anchor, pairs, exclusions
    )
    gap_summary = temporal_gap_summary(pairs)
    support_summary_frame = support_summary(support_manifest)
    transform_summary_frame = transform_validation_summary(transform_audit)

    atomic_write_csv(
        split_summary_frame,
        paths.safe / "benchmark05b_scenario_split_summary.csv",
    )
    atomic_write_csv(
        attrition,
        paths.safe / "benchmark05b_pairing_attrition_summary.csv",
    )
    atomic_write_csv(
        gap_summary,
        paths.safe / "benchmark05b_temporal_gap_summary.csv",
    )
    atomic_write_csv(
        modality_summary.sort_values(
            ["scenario", "block", "source_name", SPLIT], kind="stable"
        ).reset_index(drop=True),
        paths.safe / "benchmark05b_modality_availability_summary.csv",
    )
    atomic_write_csv(
        support_summary_frame,
        paths.safe / "benchmark05b_support_summary.csv",
    )
    atomic_write_csv(
        transform_summary_frame,
        paths.safe / "benchmark05b_transform_validation_summary.csv",
    )
    temporal_exclusion_safe = pd.DataFrame(
        [
            {
                "source_name": source_name,
                "exclusion_present": bool(
                    temporal_exclusion_manifest["source_name"]
                    .eq(source_name)
                    .any()
                ),
            }
            for source_name in VISIT_SOURCES
        ]
    )
    atomic_write_csv(
        temporal_exclusion_safe,
        paths.safe / "benchmark05b_temporal_source_exclusion_flags.csv",
    )

    story = build_story(
        split_summary_frame,
        attrition,
        gap_summary,
        support_manifest,
        local_hashes,
    )
    atomic_write_text(
        paths.safe / "BENCHMARK05B_SCENARIO_STORY.md",
        story,
    )

    summary_payload = {
        "benchmark05b_script_version": SCRIPT_VERSION,
        "created_at_utc": created_at,
        "stage": STAGE_NAME,
        "scenario_build_passed": True,
        "scenario_decision_id": scenario_decision_id,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": None,
        "scope": {
            "development_splits_used": list(DEVELOPMENT_SPLITS),
            "test_values_used": False,
            "test_scenarios_materialized": False,
            "target_values_loaded": False,
            "models_trained": False,
            "support_threshold_participants": SUPPORT_THRESHOLD,
            "support_computed_from_split": TRAIN,
            "support_recomputed_per_scenario_and_block": True,
            "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
            "same_wave_visits_concatenated": False,
            "temporal_source_exclusions_applied": True,
            "temporal_source_exclusions_before_support": True,
            "participant_split_changed": False,
            "age_at_target_materialized": True,
            "age_at_target_current_visit_only": True,
            "raw_birth_fields_exposed_as_predictors": False,
        },
        "scenario_counts": split_summary_frame.to_dict(orient="records"),
        "pairing_attrition": attrition.to_dict(orient="records"),
        "temporal_gap_summary": gap_summary.to_dict(orient="records"),
        "support_retained_counts": (
            support_manifest.groupby(["scenario", "block"])[
                "retained_at_support_threshold"
            ]
            .sum()
            .astype(int)
            .reset_index(name="retained_feature_instances")
            .to_dict(orient="records")
        ),
        "age_at_target": {
            "formula": "calendar_year(current_target_mri_examdate)-baseline_birth_year",
            "precision": "calendar_year",
            "raw_birth_fields_exposed": False,
            "previous_age_feature_created": False,
            "availability_counts": [
                {
                    "scenario": scenario_name,
                    "split": split_label,
                    "rows": int(len(frame.loc[frame[SPLIT].eq(split_label)])),
                    "age_observed_rows": int(
                        frame.loc[
                            frame[SPLIT].eq(split_label), "AGE_AT_TARGET"
                        ].notna().sum()
                    ),
                }
                for scenario_name, frame in [
                    ("snapshot_all", snapshot_all),
                    ("snapshot_matched_last", matched),
                    ("longitudinal_previous_last", pairs),
                ]
                for split_label in DEVELOPMENT_SPLITS
            ],
        },
        "design_features": {
            "modality_presence_indicators": (
                "one indicator per relevant source and temporal block; "
                "effective constants removed later on fit"
            ),
            "longitudinal_gap_months": True,
            "strict_date_gap_days_audited": True,
            "temporally_unsafe_previous_source_blocks_set_missing": True,
            "pair_removed_if_no_safe_previous_input_remains": True,
            "participant_static_sources_not_duplicated": list(PARTICIPANT_SOURCES),
            "age_at_target_current_visit_only": True,
        },
        "artifacts": {
            "scenario_definitions": str(
                paths.scenario_definitions.relative_to(paths.root)
            ),
            "scenario_definitions_sha256": sha256_file(
                paths.scenario_definitions
            ),
            "scenario_support_manifest": str(
                paths.scenario_support_manifest.relative_to(paths.root)
            ),
            "scenario_support_manifest_sha256": sha256_file(
                paths.scenario_support_manifest
            ),
            "scenario_lock": str(paths.scenario_lock.relative_to(paths.root)),
            "scenario_lock_sha256": sha256_file(paths.scenario_lock),
            "local_output_directory": str(paths.local.relative_to(paths.root)),
            "local_artifact_hashes": local_hashes,
            "temporal_exclusion_audit_summary_sha256": sha256_file(
                paths.temporal_exclusion_summary
            ),
            "temporal_exclusion_local_manifest_sha256": sha256_file(
                paths.temporal_exclusion_manifest
            ),
        },
        "temporal_source_exclusions": {
            "audit_stage": "auditA05_temporal_modality_exclusions",
            "audit_passed": temporal_exclusion_summary.get("audit_passed"),
            "audit_script_version": temporal_exclusion_summary.get(
                "script_version"
            ),
            "policy": (
                "set affected previous source blocks missing before support, "
                "availability and preprocessing"
            ),
            "participant_split_changed": False,
            "individualized_manifest_local_only": True,
        },
        "upstream": {
            "schema_decision_id": schema_lock.get("official_decision_id"),
            "support_threshold": support_policy.get(
                "selected_support_threshold_participants"
            ),
            "semantic_include_count": benchmark04b.get(
                "counts", {}
            ).get("included_features"),
            "semantic_exclude_count": benchmark04b.get(
                "counts", {}
            ).get("excluded_features"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "privacy": {
            "safe_outputs_contain_participant_identifiers": False,
            "safe_outputs_contain_visit_identifiers": False,
            "safe_outputs_contain_medical_values": False,
            "safe_outputs_contain_predictions": False,
            "individualized_artifacts_are_local_only": True,
            "local_artifacts_must_not_be_shared": True,
        },
        "methodological_note": (
            "Matched static and longitudinal scenarios share identical current "
            "participants, visits and targets. Previous visits are selected once "
            "per participant from semantically authorized non-MRI history, with "
            "strict protocol ordering and no modality-specific fallback. "
            "Any previous source block collected on or after the current target "
            "MRI is represented as missing before support calculation. "
            "AGE_AT_TARGET is derived locally for the current target visit only "
            "from baseline birth-year metadata and the current MRI EXAMDATE."
        ),
    }
    atomic_write_json(
        paths.safe / "benchmark05b_summary.json",
        summary_payload,
    )

    return {
        "summary_payload": summary_payload,
        "checks": checks,
    }


def write_failure_outputs(
    paths: Paths,
    checks: Checks,
    fatal_error: str,
) -> None:
    checks_frame = pd.DataFrame([asdict(item) for item in checks.items])
    atomic_write_csv(
        checks_frame,
        paths.safe / "benchmark05b_validation_checks.csv",
    )
    atomic_write_json(
        paths.safe / "benchmark05b_summary.json",
        {
            "benchmark05b_script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": STAGE_NAME,
            "scenario_build_passed": False,
            "check_count": len(checks.items),
            "failed_check_count": checks.failed,
            "fatal_error": fatal_error,
            "scope": {
                "test_values_used": False,
                "target_values_loaded": False,
                "models_trained": False,
            },
        },
    )


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    superseded_scenario_decision_id = archive_superseded_artifacts(
        paths,
        enabled=bool(args.supersede_existing),
    )
    configure_logging(paths)
    logging.info("%s v%s", STAGE_NAME, SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Frozen package: %s", paths.package)
    logging.info("Development splits only: train and validation.")
    logging.info("Test pairings and values remain closed.")
    logging.info(
        "Temporal policy: %s | decision_id=%s",
        EXPECTED_TEMPORAL_POLICY_NAME,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
    )
    logging.info("Support threshold frozen at %d participants.", SUPPORT_THRESHOLD)
    logging.info(
        "Temporal source exclusions: AUDIT A05 local manifest; applied before support."
    )

    checks = Checks()
    fatal_error: str | None = None
    try:
        run_stage(
            paths,
            checks,
            superseded_scenario_decision_id,
        )
        checks_frame = pd.DataFrame([asdict(item) for item in checks.items])
        atomic_write_csv(
            checks_frame,
            paths.safe / "benchmark05b_validation_checks.csv",
        )
        if not checks.passed:
            raise ScenarioAbort(
                f"BENCHMARK05B has {checks.failed} failed checks."
            )
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        logging.exception("BENCHMARK05B fatal error")
        write_failure_outputs(paths, checks, fatal_error)

    passed = checks.passed and fatal_error is None
    if passed:
        summary = read_json(paths.safe / "benchmark05b_summary.json")
        counts = summary.get("scenario_counts", [])
        matched = [
            row
            for row in counts
            if row.get("scenario") == "snapshot_matched_last"
        ]
        matched_counts = {
            str(row.get("split")): int(row.get("participants", 0))
            for row in matched
        }
        logging.info(
            "BENCHMARK05B PASSED | matched_train=%s | matched_validation=%s | "
            "test_materialized=false",
            matched_counts.get(TRAIN),
            matched_counts.get(VALIDATION),
        )
        logging.info(
            "Do not share data/processed/local_only/benchmark05b_scenarios/."
        )
        return 0

    logging.error(
        "BENCHMARK05B FAILED | failed_checks=%d | fatal_error=%s",
        checks.failed,
        fatal_error,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
