#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# BENCHMARK08B — treino, tuning e seleção em desenvolvimento (rerun cross-checked).
#
# Usa exclusivamente os artefatos congelados pelos BENCHMARK06B e BENCHMARK07B.
# Treina e avalia, em treino/validação somente:
#
# - mean;
# - median;
# - Ridge;
# - MultiTaskElasticNet;
# - PLSRegression;
# - ExtraTreesRegressor;
# - KNeighborsRegressor.
#
# Regras principais
# -----------------
# 1. O teste não é lido, transformado, materializado ou avaliado.
# 2. Todas as grades são escritas em config/ antes da primeira métrica.
# 3. A seleção usa validation participant-macro RMSE no espaço Full
#    padronizado de 323 targets.
# 4. Predições PCA90 são reconstruídas por PCA.inverse_transform antes das
#    métricas.
# 5. Nenhuma predição individual, erro individual, RID ou VISCODE2 é escrito
#    nos relatórios safe.
# 6. Checkpoints contêm apenas configurações e métricas agregadas.
# 7. O script suporta execução por famílias e retomada determinística.
# 8. Este estágio não abre o teste e não cria o lock final de teste; ele produz
#    candidatos selecionados para revisão e congelamento no estágio seguinte.

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


SCRIPT_VERSION: Final[str] = "0.1.0"
STAGE_NAME: Final[str] = "benchmark08b"
MODEL_POLICY_VERSION: Final[str] = "1.0.0"

HISTORICAL_PREPROCESSING_DECISION_ID_REFERENCE: Final[str] = (
    "276e092a4b9f0c122e30"
)

HISTORICAL_TARGET_DECISION_ID_REFERENCE: Final[str] = (
    "91335e09416bccaf93f1"
)

HISTORICAL_SCENARIO_DECISION_ID_REFERENCE: Final[str] = (
    "8662f66fae4b470c7e4f"
)

EXPECTED_TEMPORAL_POLICY_DECISION_ID: Final[str] = (
    "90fefe5ec34d49743612"
)

SCENARIOS: Final[tuple[str, ...]] = (
    "snapshot_all",
    "snapshot_matched_last",
    "longitudinal_previous_last",
)
TARGET_REPRESENTATIONS: Final[tuple[str, ...]] = ("full", "pca90")
MODELS: Final[tuple[str, ...]] = (
    "mean",
    "median",
    "ridge",
    "multitask_elastic_net",
    "pls",
    "extra_trees",
    "knn",
)

TARGET_GROUP_BY_SCENARIO: Final[dict[str, str]] = {
    "snapshot_all": "snapshot_all",
    "snapshot_matched_last": "matched_pair",
    "longitudinal_previous_last": "matched_pair",
}

RID: Final[str] = "RID"
SPLIT: Final[str] = "split"
CURRENT_VISCODE: Final[str] = "current_VISCODE2"
TRAIN: Final[str] = "train"
VALIDATION: Final[str] = "validation"
TARGET_COUNT: Final[int] = 323

# Grades pré-registradas. Não são expandidas automaticamente após validação.
RIDGE_ALPHAS: Final[tuple[float, ...]] = (
    100.0,
    300.0,
    1_000.0,
    3_000.0,
    10_000.0,
    30_000.0,
    100_000.0,
)
PLS_COMPONENTS: Final[tuple[int, ...]] = (2, 5, 10, 20, 40)
KNN_NEIGHBORS: Final[tuple[int, ...]] = (
    5,
    10,
    20,
    40,
    80,
    120,
    160,
    240,
    320,
)
KNN_WEIGHTS: Final[tuple[str, ...]] = ("uniform", "distance")
KNN_P: Final[int] = 2

MTEN_ALPHA_FRACTIONS: Final[tuple[float, ...]] = (
    0.3,
    0.1,
    0.03,
    0.01,
    0.003,
    0.001,
)
MTEN_L1_RATIOS: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
MTEN_MAX_ITER: Final[int] = 5_000
MTEN_TOL: Final[float] = 1e-4
MTEN_SELECTION: Final[str] = "cyclic"

EXTRA_TREES_CONFIGS: Final[tuple[dict[str, Any], ...]] = (
    {
        "n_estimators": 120,
        "max_depth": 14,
        "max_features": 0.4,
        "min_samples_leaf": 3,
    },
    {
        "n_estimators": 150,
        "max_depth": 18,
        "max_features": 0.6,
        "min_samples_leaf": 2,
    },
    {
        "n_estimators": 120,
        "max_depth": 16,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
    },
)
EXTRA_TREES_SEEDS: Final[tuple[int, ...]] = (
    2_026_002,
    2_026_003,
    2_026_004,
    2_026_005,
    2_026_006,
)

MODEL_SEED: Final[int] = 42_080_534
SELECTION_METRIC: Final[str] = "participant_macro_rmse"


class ModelStageAbort(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    root: Path
    b06_local: Path
    b07_local: Path
    b06_summary: Path
    b06_checks: Path
    b07_summary: Path
    b07_checks: Path
    preprocessing_lock: Path
    target_lock: Path
    model_grid: Path
    selection_policy: Path
    stage: Path
    safe: Path
    logs: Path
    checkpoints: Path


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
            logging.INFO if item.passed else logging.ERROR,
            "%s | %s",
            "PASS" if item.passed else "FAIL",
            item.name,
        )
        if not item.passed:
            logging.error(
                "DETAIL | %s | expected=%s | observed=%s%s",
                item.name,
                _short(expected),
                _short(observed),
                f" | {details}" if details else "",
            )
        return item.passed

    @property
    def failed(self) -> int:
        return sum(not item.passed for item in self.items)

    @property
    def passed(self) -> bool:
        return bool(self.items) and self.failed == 0


def _short(value: Any, limit: int = 500) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_csv_choice(value: str, allowed: Sequence[str], label: str) -> list[str]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError(f"Informe ao menos um {label}.")
    unknown = sorted(set(tokens) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"{label.capitalize()} desconhecidos: {unknown}. Permitidos: {list(allowed)}"
        )
    return list(dict.fromkeys(tokens))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BENCHMARK08B: tuning de modelos usando somente treino e validação, "
            "com teste fechado."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="Lista separada por vírgulas. Permite execução em etapas.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Lista separada por vírgulas.",
    )
    parser.add_argument(
        "--target-representations",
        default=",".join(TARGET_REPRESENTATIONS),
        help="full,pca90",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    args = parser.parse_args()
    args.models = parse_csv_choice(args.models, MODELS, "modelos")
    args.scenarios = parse_csv_choice(args.scenarios, SCENARIOS, "cenários")
    args.target_representations = parse_csv_choice(
        args.target_representations,
        TARGET_REPRESENTATIONS,
        "representações",
    )
    return args


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    reports_root = (
        args.reports_root.expanduser().resolve()
        if args.reports_root is not None
        else root / "reports"
    )
    stage = reports_root / STAGE_NAME
    return Paths(
        root=root,
        b06_local=root / "data/processed/local_only/benchmark06b_preprocessed_inputs",
        b07_local=root / "data/processed/local_only/benchmark07b_target_representations",
        b06_summary=root
        / "reports/benchmark06b/safe/benchmark06b_summary.json",
        b06_checks=root
        / "reports/benchmark06b/safe/benchmark06b_validation_checks.csv",
        b07_summary=root
        / "reports/benchmark07b/safe/benchmark07b_summary.json",
        b07_checks=root
        / "reports/benchmark07b/safe/benchmark07b_validation_checks.csv",
        preprocessing_lock=root / "config/benchmark_preprocessing_lock_b.json",
        target_lock=root / "config/benchmark_target_lock_b.json",
        model_grid=root / "config/benchmark_model_grid_b.json",
        selection_policy=root / "config/benchmark_model_selection_policy_b.json",
        stage=stage,
        safe=stage / "safe",
        logs=stage / "logs",
        checkpoints=stage / "safe/checkpoints",
    )


def configure_logging(paths: Paths) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(paths.logs / "benchmark08b.log", encoding="utf-8"),
        ],
        force=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().casefold()
    if token in {"true", "1", "yes", "y", "sim"}:
        return True
    if token in {"false", "0", "no", "n", "nao", "não", ""}:
        return False
    raise ValueError(f"Boolean value not understood: {value!r}")


def normalize_rid(series: pd.Series, context: str) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    invalid = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype=float))
    if invalid.any():
        raise ModelStageAbort(f"Invalid RID in {context}: count={int(invalid.sum())}")
    rounded = np.rint(numeric.to_numpy(dtype=float))
    if not np.allclose(numeric.to_numpy(dtype=float), rounded, atol=0.0, rtol=0.0):
        raise ModelStageAbort(f"Non-integer RID in {context}.")
    return pd.Series(rounded.astype(np.int64).astype(str), index=series.index, dtype="string")


def normalize_manifest(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    required = {RID, SPLIT, CURRENT_VISCODE}
    missing = required - set(frame.columns)
    if missing:
        raise ModelStageAbort(f"{context} missing manifest columns: {sorted(missing)}")
    output = frame.copy().reset_index(drop=True)
    output[RID] = normalize_rid(output[RID], context)
    output[SPLIT] = output[SPLIT].astype("string").str.strip().str.casefold()
    output[CURRENT_VISCODE] = (
        output[CURRENT_VISCODE].astype("string").str.strip().str.casefold()
    )
    if output[CURRENT_VISCODE].isna().any() or output[CURRENT_VISCODE].eq("").any():
        raise ModelStageAbort(f"Missing current VISCODE2 in {context}.")
    if output.duplicated([RID, CURRENT_VISCODE, SPLIT]).any():
        raise ModelStageAbort(f"Duplicate current target keys in {context}.")
    return output


def key_tuples(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return list(
        zip(
            frame[RID].astype(str),
            frame[CURRENT_VISCODE].astype(str),
            frame[SPLIT].astype(str),
        )
    )


def model_grid_payload(decision_ids: Mapping[str, str]) -> dict[str, Any]:
    grid = {
        "mean": [{}],
        "median": [{}],
        "ridge": [
            {
                "alpha": value,
                "solver": "lsqr",
            }
            for value in RIDGE_ALPHAS
        ],
        "multitask_elastic_net": [
            {
                "alpha_fraction": fraction,
                "l1_ratio": ratio,
                "selection": MTEN_SELECTION,
                "max_iter": MTEN_MAX_ITER,
                "tol": MTEN_TOL,
            }
            for ratio in MTEN_L1_RATIOS
            for fraction in MTEN_ALPHA_FRACTIONS
        ],
        "pls": [
            {
                "n_components": value,
                "scale": False,
                "max_iter": 500,
                "tol": 1e-6,
            }
            for value in PLS_COMPONENTS
        ],
        "extra_trees": [
            {
                **config,
                "seeds": list(EXTRA_TREES_SEEDS),
                "criterion": "squared_error",
            }
            for config in EXTRA_TREES_CONFIGS
        ],
        "knn": [
            {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "p": KNN_P,
                "metric": "minkowski",
                "algorithm": "brute",
            }
            for n_neighbors in KNN_NEIGHBORS
            for weights in KNN_WEIGHTS
        ],
    }

    policy = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "model_policy_version": MODEL_POLICY_VERSION,

        # Vincula a grade de modelos à cadeia upstream exata.
        # Qualquer mudança no BENCHMARK05, 06 ou 07 passa a gerar
        # um model_grid_decision_id diferente.
        "upstream_decision_ids": {
            "preprocessing_decision_id": decision_ids[
                "preprocessing_decision_id"
            ],
            "target_decision_id": decision_ids[
                "target_decision_id"
            ],
            "scenario_decision_id": decision_ids[
                "scenario_decision_id"
            ],
            "temporal_policy_decision_id": (
                EXPECTED_TEMPORAL_POLICY_DECISION_ID
            ),
        },

        "models": list(MODELS),
        "scenarios": list(SCENARIOS),
        "target_representations": list(
            TARGET_REPRESENTATIONS
        ),
        "selection_metric": SELECTION_METRIC,
        "selection_tie_breakers": [
            "participant_macro_mae",
            "configuration_order",
        ],
        "metric_space": (
            "standardized reconstructed Full "
            "323-dimensional target"
        ),
        "grid": grid,
        "extra_trees_selection": (
            "choose hyperparameter configuration by validation "
            "metrics of the mean prediction across five fixed-seed "
            "estimators; also report seed stability"
        ),
        "pca_evaluation": (
            "inverse PCA before every validation metric"
        ),
        "automatic_boundary_expansion": False,
        "knn_note": (
            "The neighborhood grid already includes the previously "
            "predeclared upper-bound extension from 80 through 320. "
            "No further adaptive extension."
        ),
        "test": (
            "closed; not read, transformed, materialized or evaluated"
        ),
        "bootstrap": (
            "deferred to the final locked test evaluation"
        ),
    }

    policy["model_grid_decision_id"] = (
        stable_json_hash(policy)[:20]
    )

    return policy


def selection_policy_payload(grid_decision_id: str) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "model_policy_version": MODEL_POLICY_VERSION,
        "model_grid_decision_id": grid_decision_id,
        "fit_split": TRAIN,
        "evaluation_split": VALIDATION,
        "primary_metric": SELECTION_METRIC,
        "primary_metric_definition": (
            "For each row, average squared error over 323 standardized targets; "
            "average row MSE within participant; take participant RMSE; average participants."
        ),
        "eligibility": "converged configurations only when at least one converged configuration exists",
        "tie_breakers": ["participant_macro_mae", "configuration_order"],
        "pca90_prediction_evaluation": "PCA.inverse_transform to standardized Full",
        "extra_trees": "five-seed prediction ensemble for configuration selection",
        "boundary_rule": (
            "Flag a boundary winner for review, but do not expand any grid after "
            "observing validation results."
        ),
        "test_usage": "none in BENCHMARK08B",
        "final_lock": (
            "Selected configurations are candidates only. A later explicit lock is "
            "required before refit train+validation and one-time test opening."
        ),
    }


def freeze_policy_files(
    paths: Paths,
    args: argparse.Namespace,
    decision_ids: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    grid = model_grid_payload(decision_ids)
    grid_id = str(grid["model_grid_decision_id"])
    selection = selection_policy_payload(grid_id)

    for path, payload in [(paths.model_grid, grid), (paths.selection_policy, selection)]:
        if path.exists():
            existing = read_json(path)
            if stable_json_hash(existing) != stable_json_hash(payload):
                if not args.overwrite:
                    raise ModelStageAbort(
                        f"Frozen model policy differs from current script: {path}. "
                        "Do not proceed without an intentional versioned replacement."
                    )
                atomic_write_json(path, payload)
        else:
            atomic_write_json(path, payload)
    return grid, grid_id


def load_and_validate_upstream(paths: Paths, checks: Checks) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {
        "benchmark06_summary": paths.b06_summary,
        "benchmark06_checks": paths.b06_checks,
        "benchmark07_summary": paths.b07_summary,
        "benchmark07_checks": paths.b07_checks,
        "preprocessing_lock": paths.preprocessing_lock,
        "target_lock": paths.target_lock,
        "benchmark06_local": paths.b06_local,
        "benchmark07_local": paths.b07_local,
    }
    for label, path in required.items():
        exists = path.is_file() if path.suffix else path.is_dir()
        checks.add(f"artifact__{label}", exists, True, exists, str(path))
    if not all((p.is_file() if p.suffix else p.is_dir()) for p in required.values()):
        raise ModelStageAbort("Required BENCHMARK06B/BENCHMARK07B artifacts are missing.")

    b06 = read_json(paths.b06_summary)
    b07 = read_json(paths.b07_summary)
    pre_lock = read_json(paths.preprocessing_lock)
    target_lock = read_json(paths.target_lock)

    checks.add("benchmark06_passed", bool(b06.get("preprocessing_passed")), True, b06.get("preprocessing_passed"))
    checks.add("benchmark07_passed", bool(b07.get("target_preprocessing_passed")), True, b07.get("target_preprocessing_passed"))
    checks.add("benchmark06_failed_checks", int(b06.get("failed_check_count", -1)) == 0, 0, b06.get("failed_check_count"))
    checks.add("benchmark07_failed_checks", int(b07.get("failed_check_count", -1)) == 0, 0, b07.get("failed_check_count"))

    for label, path in [("benchmark06", paths.b06_checks), ("benchmark07", paths.b07_checks)]:
        frame = pd.read_csv(path, low_memory=False)
        if "passed" not in frame.columns:
            failed = -1
        else:
            failed = int((~frame["passed"].map(parse_bool)).sum())
        checks.add(f"{label}_check_table", failed == 0, 0, failed)

    preprocessing_decision_id = str(
        b06.get("preprocessing_decision_id", "")
    ).strip()
    b07_preprocessing_decision_id = str(
        b07.get("preprocessing_decision_id", "")
    ).strip()
    pre_lock_decision_id = str(
        pre_lock.get("preprocessing_decision_id", "")
    ).strip()
    target_lock_preprocessing_id = str(
        target_lock.get("upstream_preprocessing_decision_id", "")
    ).strip()

    target_decision_id = str(
        b07.get("target_decision_id", "")
    ).strip()
    target_lock_decision_id = str(
        target_lock.get("target_decision_id", "")
    ).strip()

    scenario_decision_id = str(
        b06.get("scenario_decision_id", "")
    ).strip()
    b07_scenario_decision_id = str(
        b07.get("scenario_decision_id", "")
    ).strip()
    pre_lock_scenario_id = str(
        pre_lock.get("upstream_scenario_decision_id", "")
    ).strip()
    target_lock_scenario_id = str(
        target_lock.get("upstream_scenario_decision_id", "")
    ).strip()

    checks.add(
        "preprocessing_decision_chain_consistent",
        bool(preprocessing_decision_id)
        and len(
            {
                preprocessing_decision_id,
                b07_preprocessing_decision_id,
                pre_lock_decision_id,
                target_lock_preprocessing_id,
            }
        )
        == 1,
        True,
        {
            "benchmark06b": preprocessing_decision_id,
            "benchmark07b": b07_preprocessing_decision_id,
            "preprocessing_lock": pre_lock_decision_id,
            "target_lock_upstream": target_lock_preprocessing_id,
        },
    )
    checks.add(
        "target_decision_chain_consistent",
        bool(target_decision_id)
        and target_decision_id == target_lock_decision_id,
        True,
        {
            "benchmark07b": target_decision_id,
            "target_lock": target_lock_decision_id,
        },
    )
    checks.add(
        "scenario_decision_chain_consistent",
        bool(scenario_decision_id)
        and len(
            {
                scenario_decision_id,
                b07_scenario_decision_id,
                pre_lock_scenario_id,
                target_lock_scenario_id,
            }
        )
        == 1,
        True,
        {
            "benchmark06b": scenario_decision_id,
            "benchmark07b": b07_scenario_decision_id,
            "preprocessing_lock_upstream": pre_lock_scenario_id,
            "target_lock_upstream": target_lock_scenario_id,
        },
    )
    checks.add(
        "temporal_policy_decision_id",
        str(b07.get("temporal_policy_decision_id"))
        == EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        b07.get("temporal_policy_decision_id"),
    )
    checks.add(
        "upstream_test_closed_b06",
        not bool(b06.get("scope", {}).get("test_materialized"))
        and not bool(b06.get("scope", {}).get("test_values_used")),
        True,
        b06.get("scope"),
    )
    checks.add(
        "upstream_test_closed_b07",
        not bool(b07.get("scope", {}).get("test_materialized"))
        and not bool(b07.get("scope", {}).get("test_values_parsed")),
        True,
        b07.get("scope"),
    )

    if not checks.passed:
        raise ModelStageAbort("Upstream lock validation failed.")
    return b06, b07, pre_lock, target_lock


def verify_local_hashes(base: Path, lock: Mapping[str, Any], prefix: str, checks: Checks) -> None:
    hashes = lock.get("local_artifact_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ModelStageAbort(f"No local artifact hashes in {prefix} lock.")
    for relative, expected in hashes.items():
        path = base / str(relative)
        exists = path.is_file()
        checks.add(f"{prefix}_artifact_exists__{str(relative).replace('/', '__')}", exists, True, exists)
        if not exists:
            continue
        actual = sha256_file(path)
        checks.add(
            f"{prefix}_artifact_hash__{str(relative).replace('/', '__')}",
            actual == str(expected),
            str(expected),
            actual,
        )
    if not checks.passed:
        raise ModelStageAbort(f"{prefix} local artifact hash validation failed.")


@dataclass
class ScenarioData:
    scenario: str
    target_group: str
    x_train: sparse.csr_matrix
    x_validation: sparse.csr_matrix
    input_train_manifest: pd.DataFrame
    input_validation_manifest: pd.DataFrame
    target_train_manifest: pd.DataFrame
    target_validation_manifest: pd.DataFrame
    y_full_train: np.ndarray
    y_full_validation: np.ndarray
    y_pca_train: np.ndarray
    y_pca_validation: np.ndarray
    pca: Any


def load_scenario_data(
    paths: Paths,
    scenario: str,
    b06: Mapping[str, Any],
    b07: Mapping[str, Any],
    checks: Checks,
) -> ScenarioData:
    input_dir = paths.b06_local / scenario
    target_group = TARGET_GROUP_BY_SCENARIO[scenario]
    target_dir = paths.b07_local / target_group

    expected_files = [
        input_dir / "X_train.npz",
        input_dir / "X_validation.npz",
        input_dir / "row_manifest_train.csv.gz",
        input_dir / "row_manifest_validation.csv.gz",
target_dir / "Y_full_train.npy",
        target_dir / "Y_full_validation.npy",
        target_dir / "Y_pca90_train.npy",
        target_dir / "Y_pca90_validation.npy",
        target_dir / "row_manifest_train.csv.gz",
        target_dir / "row_manifest_validation.csv.gz",
        target_dir / "pca90.joblib",
    ]
    for path in expected_files:
        checks.add(
            f"artifact__{scenario}__{path.name}",
            path.is_file(),
            True,
            path.is_file(),
            str(path),
        )
    if not all(path.is_file() for path in expected_files):
        raise ModelStageAbort(f"Missing local matrix/target artifacts for {scenario}.")

    x_train = sparse.load_npz(input_dir / "X_train.npz").tocsr().astype(np.float32)
    x_validation = (
        sparse.load_npz(input_dir / "X_validation.npz").tocsr().astype(np.float32)
    )
    input_train_manifest = normalize_manifest(
        pd.read_csv(input_dir / "row_manifest_train.csv.gz", low_memory=False),
        f"{scenario}/input/train",
    )
    input_validation_manifest = normalize_manifest(
        pd.read_csv(input_dir / "row_manifest_validation.csv.gz", low_memory=False),
        f"{scenario}/input/validation",
    )
    target_train_manifest = normalize_manifest(
        pd.read_csv(target_dir / "row_manifest_train.csv.gz", low_memory=False),
        f"{scenario}/target/train",
    )
    target_validation_manifest = normalize_manifest(
        pd.read_csv(target_dir / "row_manifest_validation.csv.gz", low_memory=False),
        f"{scenario}/target/validation",
    )

    y_full_train = np.load(target_dir / "Y_full_train.npy", allow_pickle=False)
    y_full_validation = np.load(
        target_dir / "Y_full_validation.npy", allow_pickle=False
    )
    y_pca_train = np.load(target_dir / "Y_pca90_train.npy", allow_pickle=False)
    y_pca_validation = np.load(
        target_dir / "Y_pca90_validation.npy", allow_pickle=False
    )
    pca = joblib.load(target_dir / "pca90.joblib")

    expected_b06 = {
        str(row["scenario"]): row for row in b06.get("scenario_dimensions", [])
    }.get(scenario, {})
    expected_b07 = {
        str(row["target_group"]): row
        for row in b07.get("target_group_dimensions", [])
    }.get(target_group, {})

    checks.add(f"x_train_rows__{scenario}", x_train.shape[0] == len(input_train_manifest), len(input_train_manifest), x_train.shape[0])
    checks.add(f"x_validation_rows__{scenario}", x_validation.shape[0] == len(input_validation_manifest), len(input_validation_manifest), x_validation.shape[0])
    checks.add(f"x_train_width__{scenario}", x_train.shape[1] == int(expected_b06.get("encoded_columns_final", -1)), expected_b06.get("encoded_columns_final"), x_train.shape[1])
    checks.add(f"x_validation_width__{scenario}", x_validation.shape[1] == x_train.shape[1], x_train.shape[1], x_validation.shape[1])
    checks.add(f"x_train_finite__{scenario}", bool(np.isfinite(x_train.data).all()), True, bool(np.isfinite(x_train.data).all()))
    checks.add(f"x_validation_finite__{scenario}", bool(np.isfinite(x_validation.data).all()), True, bool(np.isfinite(x_validation.data).all()))

    checks.add(f"y_full_train_shape__{scenario}", y_full_train.shape == (x_train.shape[0], TARGET_COUNT), (x_train.shape[0], TARGET_COUNT), y_full_train.shape)
    checks.add(f"y_full_validation_shape__{scenario}", y_full_validation.shape == (x_validation.shape[0], TARGET_COUNT), (x_validation.shape[0], TARGET_COUNT), y_full_validation.shape)
    checks.add(f"y_pca_train_shape__{scenario}", y_pca_train.shape == (x_train.shape[0], int(expected_b07.get("pca90_components", -1))), (x_train.shape[0], expected_b07.get("pca90_components")), y_pca_train.shape)
    checks.add(f"y_pca_validation_shape__{scenario}", y_pca_validation.shape == (x_validation.shape[0], y_pca_train.shape[1]), (x_validation.shape[0], y_pca_train.shape[1]), y_pca_validation.shape)
    checks.add(f"y_full_finite__{scenario}", bool(np.isfinite(y_full_train).all() and np.isfinite(y_full_validation).all()), True, bool(np.isfinite(y_full_train).all() and np.isfinite(y_full_validation).all()))
    checks.add(f"y_pca_finite__{scenario}", bool(np.isfinite(y_pca_train).all() and np.isfinite(y_pca_validation).all()), True, bool(np.isfinite(y_pca_train).all() and np.isfinite(y_pca_validation).all()))

    checks.add(f"input_target_order_train__{scenario}", key_tuples(input_train_manifest) == key_tuples(target_train_manifest), True, key_tuples(input_train_manifest) == key_tuples(target_train_manifest))
    checks.add(f"input_target_order_validation__{scenario}", key_tuples(input_validation_manifest) == key_tuples(target_validation_manifest), True, key_tuples(input_validation_manifest) == key_tuples(target_validation_manifest))
    checks.add(f"train_split_only__{scenario}", set(input_train_manifest[SPLIT].astype(str)) == {TRAIN}, [TRAIN], sorted(set(input_train_manifest[SPLIT].astype(str))))
    checks.add(f"validation_split_only__{scenario}", set(input_validation_manifest[SPLIT].astype(str)) == {VALIDATION}, [VALIDATION], sorted(set(input_validation_manifest[SPLIT].astype(str))))

    if not checks.passed:
        raise ModelStageAbort(f"Scenario matrix validation failed for {scenario}.")

    return ScenarioData(
        scenario=scenario,
        target_group=target_group,
        x_train=x_train,
        x_validation=x_validation,
        input_train_manifest=input_train_manifest,
        input_validation_manifest=input_validation_manifest,
        target_train_manifest=target_train_manifest,
        target_validation_manifest=target_validation_manifest,
        y_full_train=np.asarray(y_full_train, dtype=np.float32),
        y_full_validation=np.asarray(y_full_validation, dtype=np.float32),
        y_pca_train=np.asarray(y_pca_train, dtype=np.float32),
        y_pca_validation=np.asarray(y_pca_validation, dtype=np.float32),
        pca=pca,
    )


class PeakRSSMonitor:
    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = math.nan
        self.peak_rss = math.nan

    def __enter__(self) -> "PeakRSSMonitor":
        if psutil is None:
            return self
        process = psutil.Process(os.getpid())
        self.start_rss = float(process.memory_info().rss)
        self.peak_rss = self.start_rss

        def sample() -> None:
            while not self._stop.wait(self.interval):
                with contextlib.suppress(Exception):
                    self.peak_rss = max(
                        self.peak_rss,
                        float(process.memory_info().rss),
                    )

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if psutil is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            process = psutil.Process(os.getpid())
            self.peak_rss = max(self.peak_rss, float(process.memory_info().rss))

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / (1024.0 * 1024.0) if math.isfinite(self.peak_rss) else math.nan

    @property
    def delta_mb(self) -> float:
        if not (math.isfinite(self.peak_rss) and math.isfinite(self.start_rss)):
            return math.nan
        return (self.peak_rss - self.start_rss) / (1024.0 * 1024.0)


def participant_metric_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participants: np.ndarray,
) -> pd.DataFrame:
    row_mse = np.square(y_true - y_pred).mean(axis=1)
    row_mae = np.abs(y_true - y_pred).mean(axis=1)
    frame = pd.DataFrame(
        {
            "participant": participants.astype(str),
            "row_mse": row_mse,
            "row_mae": row_mae,
        }
    )
    grouped = frame.groupby("participant", sort=False).agg(
        participant_mse=("row_mse", "mean"),
        participant_mae=("row_mae", "mean"),
    )
    grouped["participant_rmse"] = np.sqrt(grouped["participant_mse"])
    return grouped.reset_index(drop=True)


def median_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correlations: list[float] = []
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        if np.std(truth) <= 1e-12 or np.std(pred) <= 1e-12:
            continue
        value = float(np.corrcoef(truth, pred)[0, 1])
        if math.isfinite(value):
            correlations.append(value)
    return float(np.median(correlations)) if correlations else math.nan


def aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participants: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ModelStageAbort(f"Metric shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != TARGET_COUNT:
        raise ModelStageAbort(f"Metrics require (*, {TARGET_COUNT}) targets.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ModelStageAbort("Non-finite truth or prediction in metrics.")

    grouped = participant_metric_frame(y_true, y_pred, participants)
    error = y_true - y_pred
    per_target_r2 = np.asarray(
        [
            r2_score(y_true[:, j], y_pred[:, j], force_finite=True)
            for j in range(y_true.shape[1])
        ],
        dtype=float,
    )
    return {
        "n_rows": int(y_true.shape[0]),
        "n_participants": int(len(np.unique(participants))),
        "n_targets_evaluated": int(y_true.shape[1]),
        "participant_macro_rmse": float(grouped["participant_rmse"].mean()),
        "participant_macro_mae": float(grouped["participant_mae"].mean()),
        "visit_weighted_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "visit_weighted_mae": float(np.mean(np.abs(error))),
        "r2_uniform_average": float(
            r2_score(y_true, y_pred, multioutput="uniform_average", force_finite=True)
        ),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted", force_finite=True)
        ),
        "median_pearson_by_target": median_pearson(y_true, y_pred),
        "median_target_r2": float(np.median(per_target_r2)),
        "positive_target_r2_fraction": float(np.mean(per_target_r2 > 0)),
    }


def centered_alpha_base(x_train: np.ndarray, y_train: np.ndarray) -> float:
    x_centered = x_train - np.mean(x_train, axis=0, keepdims=True)
    y_centered = y_train - np.mean(y_train, axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    feature_norms = np.linalg.norm(cross, axis=1)
    value = float(np.max(feature_norms) / x_train.shape[0])
    if not math.isfinite(value) or value <= 0:
        raise ModelStageAbort(f"Invalid train-only alpha_base: {value}")
    return value


def reconstruct_prediction(
    prediction_model: np.ndarray,
    representation: str,
    pca: Any,
) -> np.ndarray:
    prediction_model = np.asarray(prediction_model, dtype=np.float64)
    if prediction_model.ndim == 1:
        prediction_model = prediction_model.reshape(-1, 1)
    if representation == "full":
        prediction = prediction_model
    elif representation == "pca90":
        prediction = np.asarray(pca.inverse_transform(prediction_model), dtype=np.float64)
    else:  # pragma: no cover
        raise ValueError(representation)
    if prediction.shape[1] != TARGET_COUNT or not np.isfinite(prediction).all():
        raise ModelStageAbort(
            f"Invalid reconstructed prediction shape/finiteness: {prediction.shape}"
        )
    return prediction


def configuration_id(model: str, params: Mapping[str, Any]) -> str:
    material = {"model": model, "params": dict(params)}
    return stable_json_hash(material)[:16]


def checkpoint_path(
    paths: Paths,
    scenario: str,
    representation: str,
    model: str,
    config_id: str,
) -> Path:
    return paths.checkpoints / scenario / representation / model / f"{config_id}.json"


def load_checkpoint(path: Path, grid_id: str) -> dict[str, Any]:
    payload = read_json(path)
    if str(payload.get("model_grid_decision_id")) != grid_id:
        raise ModelStageAbort(f"Incompatible checkpoint: {path}")
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def common_trial_context(
    scenario: str,
    representation: str,
    model: str,
    params: Mapping[str, Any],
    config_order: int,
    grid_id: str,
    data: ScenarioData,
    decision_ids: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "model_grid_decision_id": grid_id,
        "preprocessing_decision_id": decision_ids[
            "preprocessing_decision_id"
        ],
        "target_decision_id": decision_ids[
            "target_decision_id"
        ],
        "scenario_decision_id": decision_ids[
            "scenario_decision_id"
        ],
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "scenario": scenario,
        "target_group": data.target_group,
        "target_representation": representation,
        "model": model,
        "configuration_id": configuration_id(model, params),
        "configuration_order": int(config_order),
        "parameters": dict(params),
        "n_input_features": int(data.x_train.shape[1]),
        "n_model_targets": int(
            data.y_full_train.shape[1]
            if representation == "full"
            else data.y_pca_train.shape[1]
        ),
        "fit_split": TRAIN,
        "evaluation_split": VALIDATION,
        "test_used": False,
        "test_materialized": False,
    }


def evaluate_baseline(
    model: str,
    representation: str,
    data: ScenarioData,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model == "mean":
        vector = np.zeros(TARGET_COUNT, dtype=np.float64)
    elif model == "median":
        vector = np.median(data.y_full_train.astype(np.float64), axis=0)
    else:  # pragma: no cover
        raise ValueError(model)
    prediction = np.repeat(vector.reshape(1, -1), len(data.y_full_validation), axis=0)
    return prediction, {
        "fit_seconds": 0.0,
        "inference_seconds": 0.0,
        "peak_rss_mb": math.nan,
        "peak_rss_delta_mb": math.nan,
        "warning_count": 0,
        "convergence_warning_count": 0,
        "converged": True,
    }


def evaluate_ridge(
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_train = data.y_full_train if representation == "full" else data.y_pca_train
    model = Ridge(
        alpha=float(params["alpha"]),
        solver="lsqr",
        fit_intercept=True,
    )
    with PeakRSSMonitor() as memory:
        fit_start = time.perf_counter()
        model.fit(data.x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        prediction_model = model.predict(data.x_validation)
        prediction = reconstruct_prediction(prediction_model, representation, data.pca)
        inference_seconds = time.perf_counter() - inference_start
    return prediction, {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_mb": memory.peak_mb,
        "peak_rss_delta_mb": memory.delta_mb,
        "warning_count": 0,
        "convergence_warning_count": 0,
        "converged": True,
        "coefficient_frobenius_norm": float(np.linalg.norm(model.coef_)),
    }


def evaluate_pls(
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = data.x_train.toarray().astype(np.float64, copy=False)
    x_validation = data.x_validation.toarray().astype(np.float64, copy=False)
    y_train = (
        data.y_full_train.astype(np.float64, copy=False)
        if representation == "full"
        else data.y_pca_train.astype(np.float64, copy=False)
    )
    model = PLSRegression(
        n_components=int(params["n_components"]),
        scale=False,
        max_iter=int(params["max_iter"]),
        tol=float(params["tol"]),
        copy=True,
    )
    with warnings.catch_warnings(record=True) as captured, PeakRSSMonitor() as memory:
        warnings.simplefilter("always")
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        prediction_model = model.predict(x_validation)
        prediction = reconstruct_prediction(prediction_model, representation, data.pca)
        inference_seconds = time.perf_counter() - inference_start
    convergence_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in captured
    )
    n_iter = int(max(np.asarray(model.n_iter_, dtype=int))) if len(model.n_iter_) else 0
    del x_train, x_validation
    return prediction, {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_mb": memory.peak_mb,
        "peak_rss_delta_mb": memory.delta_mb,
        "warning_count": len(captured),
        "convergence_warning_count": convergence_count,
        "converged": convergence_count == 0,
        "n_iter": n_iter,
    }


def evaluate_knn(
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_train = data.y_full_train if representation == "full" else data.y_pca_train
    model = KNeighborsRegressor(
        n_neighbors=int(params["n_neighbors"]),
        weights=str(params["weights"]),
        p=int(params["p"]),
        metric=str(params["metric"]),
        algorithm=str(params["algorithm"]),
        n_jobs=n_jobs,
    )
    with PeakRSSMonitor() as memory:
        fit_start = time.perf_counter()
        model.fit(data.x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        prediction_model = model.predict(data.x_validation)
        prediction = reconstruct_prediction(prediction_model, representation, data.pca)
        inference_seconds = time.perf_counter() - inference_start
    return prediction, {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_mb": memory.peak_mb,
        "peak_rss_delta_mb": memory.delta_mb,
        "warning_count": 0,
        "convergence_warning_count": 0,
        "converged": True,
    }


def evaluate_extra_trees_ensemble(
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    y_train = data.y_full_train if representation == "full" else data.y_pca_train
    x_train = data.x_train.tocsc()
    x_validation = data.x_validation.tocsr()
    predictions: list[np.ndarray] = []
    seed_metric_rows: list[dict[str, Any]] = []
    fit_total = 0.0
    inference_total = 0.0
    peak_values: list[float] = []
    delta_values: list[float] = []
    participants = data.input_validation_manifest[RID].astype(str).to_numpy()

    for seed in params["seeds"]:
        model = ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=(
                None if params["max_depth"] is None else int(params["max_depth"])
            ),
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=int(seed),
            n_jobs=n_jobs,
            criterion=str(params["criterion"]),
            bootstrap=False,
        )
        with PeakRSSMonitor() as memory:
            fit_start = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_start
            inference_start = time.perf_counter()
            pred_model = model.predict(x_validation)
            pred = reconstruct_prediction(pred_model, representation, data.pca)
            inference_seconds = time.perf_counter() - inference_start
        predictions.append(pred)
        fit_total += fit_seconds
        inference_total += inference_seconds
        peak_values.append(memory.peak_mb)
        delta_values.append(memory.delta_mb)
        metrics = aggregate_metrics(data.y_full_validation, pred, participants)
        seed_metric_rows.append(metrics)
        del model, pred_model
        gc.collect()

    prediction = np.mean(np.stack(predictions, axis=0), axis=0)
    seed_rmse = np.asarray(
        [row["participant_macro_rmse"] for row in seed_metric_rows], dtype=float
    )
    return prediction, {
        "fit_seconds": fit_total,
        "inference_seconds": inference_total,
        "peak_rss_mb": float(np.nanmax(peak_values)) if peak_values else math.nan,
        "peak_rss_delta_mb": float(np.nanmax(delta_values)) if delta_values else math.nan,
        "warning_count": 0,
        "convergence_warning_count": 0,
        "converged": True,
        "ensemble_seed_count": len(predictions),
        "seed_participant_macro_rmse_mean": float(seed_rmse.mean()),
        "seed_participant_macro_rmse_std": float(seed_rmse.std(ddof=0)),
        "seed_participant_macro_rmse_min": float(seed_rmse.min()),
        "seed_participant_macro_rmse_max": float(seed_rmse.max()),
    }


def evaluate_mten(
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
    alpha_base: float,
    warm_model: MultiTaskElasticNet | None,
) -> tuple[np.ndarray, dict[str, Any], MultiTaskElasticNet]:
    x_train = data.x_train.toarray().astype(np.float64, copy=False)
    x_validation = data.x_validation.toarray().astype(np.float64, copy=False)
    y_train = (
        data.y_full_train.astype(np.float64, copy=False)
        if representation == "full"
        else data.y_pca_train.astype(np.float64, copy=False)
    )
    alpha_absolute = (
        alpha_base * float(params["alpha_fraction"]) / float(params["l1_ratio"])
    )
    if warm_model is None:
        model = MultiTaskElasticNet(
            alpha=alpha_absolute,
            l1_ratio=float(params["l1_ratio"]),
            fit_intercept=True,
            copy_X=True,
            max_iter=int(params["max_iter"]),
            tol=float(params["tol"]),
            warm_start=True,
            random_state=MODEL_SEED,
            selection=str(params["selection"]),
        )
    else:
        model = warm_model
        model.set_params(alpha=alpha_absolute)

    with warnings.catch_warnings(record=True) as captured, PeakRSSMonitor() as memory:
        warnings.simplefilter("always")
        fit_start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        inference_start = time.perf_counter()
        prediction_model = model.predict(x_validation)
        prediction = reconstruct_prediction(prediction_model, representation, data.pca)
        inference_seconds = time.perf_counter() - inference_start

    convergence_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in captured
    )
    coefficients = np.asarray(model.coef_, dtype=float)
    feature_norms = np.linalg.norm(coefficients, axis=0)
    active = feature_norms > 1e-12
    n_iter = int(np.max(np.asarray(model.n_iter_)))
    dual_gap = float(np.max(np.asarray(model.dual_gap_, dtype=float)))
    del x_train, x_validation
    return prediction, {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_mb": memory.peak_mb,
        "peak_rss_delta_mb": memory.delta_mb,
        "warning_count": len(captured),
        "convergence_warning_count": convergence_count,
        "converged": convergence_count == 0 and n_iter < int(params["max_iter"]),
        "max_iter_hit": n_iter >= int(params["max_iter"]),
        "n_iter": n_iter,
        "dual_gap": dual_gap,
        "alpha_base_train_only": alpha_base,
        "alpha_absolute": alpha_absolute,
        "n_active_features": int(active.sum()),
        "active_feature_fraction": float(active.mean()),
        "coefficient_frobenius_norm": float(np.linalg.norm(coefficients)),
        "coefficient_l21_norm": float(np.sum(feature_norms)),
    }, model


def evaluate_single_configuration(
    model: str,
    params: Mapping[str, Any],
    representation: str,
    data: ScenarioData,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model in {"mean", "median"}:
        return evaluate_baseline(model, representation, data)
    if model == "ridge":
        return evaluate_ridge(params, representation, data)
    if model == "pls":
        return evaluate_pls(params, representation, data)
    if model == "knn":
        return evaluate_knn(params, representation, data, n_jobs)
    if model == "extra_trees":
        return evaluate_extra_trees_ensemble(params, representation, data, n_jobs)
    raise ValueError(model)


def valid_configs(model: str, grid: Mapping[str, Any], data: ScenarioData, representation: str) -> list[dict[str, Any]]:
    configs = [dict(item) for item in grid["grid"][model]]
    if model == "pls":
        max_components = min(
            data.x_train.shape[1],
            data.y_full_train.shape[1]
            if representation == "full"
            else data.y_pca_train.shape[1],
            data.x_train.shape[0] - 1,
        )
        configs = [item for item in configs if int(item["n_components"]) <= max_components]
    elif model == "knn":
        configs = [item for item in configs if int(item["n_neighbors"]) <= data.x_train.shape[0]]
    if not configs:
        raise ModelStageAbort(
            f"No valid configurations for {model}/{data.scenario}/{representation}."
        )
    return configs


def execute_model_grid(
    paths: Paths,
    args: argparse.Namespace,
    grid: Mapping[str, Any],
    grid_id: str,
    data: ScenarioData,
    representation: str,
    model_name: str,
    decision_ids: Mapping[str, str],
) -> None:
    configs = valid_configs(model_name, grid, data, representation)
    participants = data.input_validation_manifest[RID].astype(str).to_numpy()

    if model_name != "multitask_elastic_net":
        for config_order, params in enumerate(configs):
            config_id = configuration_id(model_name, params)
            path = checkpoint_path(
                paths, data.scenario, representation, model_name, config_id
            )
            if path.exists():
                if not args.resume:
                    raise ModelStageAbort(
                        f"Checkpoint exists: {path}. Use --resume or --overwrite."
                    )
                load_checkpoint(path, grid_id)
                logging.info(
                    "RESUME | %s | %s | %s | %s",
                    data.scenario,
                    representation,
                    model_name,
                    config_id,
                )
                continue

            logging.info(
                "FIT | %s | %s | %s | config=%d/%d | %s",
                data.scenario,
                representation,
                model_name,
                config_order + 1,
                len(configs),
                params,
            )
            context = common_trial_context(
                data.scenario,
                representation,
                model_name,
                params,
                config_order,
                grid_id,
                data,
                decision_ids,
            )
            prediction, diagnostics = evaluate_single_configuration(
                model_name, params, representation, data, args.n_jobs
            )
            metrics = aggregate_metrics(
                data.y_full_validation,
                prediction,
                participants,
            )
            payload = {
                **context,
                **metrics,
                **diagnostics,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "privacy": {
                    "contains_identifiers": False,
                    "contains_individual_predictions": False,
                    "contains_individual_errors": False,
                    "contains_medical_values": False,
                },
            }
            save_checkpoint(path, payload)
            logging.info(
                "METRIC | %s | %s | %s | macro_rmse=%.6f",
                data.scenario,
                representation,
                model_name,
                payload[SELECTION_METRIC],
            )
            del prediction
            gc.collect()
        return

    # Elastic Net multitarefa: caminhos separados por l1_ratio, forte -> fraco.
    x_dense = data.x_train.toarray().astype(np.float64, copy=False)
    y_train_model = (
        data.y_full_train.astype(np.float64, copy=False)
        if representation == "full"
        else data.y_pca_train.astype(np.float64, copy=False)
    )
    alpha_base = centered_alpha_base(x_dense, y_train_model)
    del x_dense, y_train_model

    config_order_map = {
        configuration_id(model_name, params): index
        for index, params in enumerate(configs)
    }
    for ratio in MTEN_L1_RATIOS:
        path_configs = sorted(
            [item for item in configs if math.isclose(float(item["l1_ratio"]), ratio)],
            key=lambda item: float(item["alpha_fraction"]),
            reverse=True,
        )
        warm_model: MultiTaskElasticNet | None = None
        for params in path_configs:
            config_id = configuration_id(model_name, params)
            path = checkpoint_path(
                paths, data.scenario, representation, model_name, config_id
            )
            config_order = config_order_map[config_id]
            if path.exists():
                if not args.resume:
                    raise ModelStageAbort(
                        f"Checkpoint exists: {path}. Use --resume or --overwrite."
                    )
                load_checkpoint(path, grid_id)
                logging.info(
                    "RESUME | %s | %s | %s | %s",
                    data.scenario,
                    representation,
                    model_name,
                    config_id,
                )
                # A retomada não depende do estado interno anterior.
                warm_model = None
                continue

            logging.info(
                "FIT | %s | %s | %s | l1_ratio=%.3g | fraction=%.6g",
                data.scenario,
                representation,
                model_name,
                ratio,
                float(params["alpha_fraction"]),
            )
            context = common_trial_context(
                data.scenario,
                representation,
                model_name,
                params,
                config_order,
                grid_id,
                data,
                decision_ids,
            )
            prediction, diagnostics, warm_model = evaluate_mten(
                params,
                representation,
                data,
                alpha_base,
                warm_model,
            )
            metrics = aggregate_metrics(
                data.y_full_validation,
                prediction,
                participants,
            )
            payload = {
                **context,
                **metrics,
                **diagnostics,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "privacy": {
                    "contains_identifiers": False,
                    "contains_individual_predictions": False,
                    "contains_individual_errors": False,
                    "contains_medical_values": False,
                },
            }
            save_checkpoint(path, payload)
            logging.info(
                "METRIC | %s | %s | %s | macro_rmse=%.6f | converged=%s",
                data.scenario,
                representation,
                model_name,
                payload[SELECTION_METRIC],
                payload["converged"],
            )
            del prediction
            gc.collect()


def flatten_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in payload.items() if key not in {"parameters", "privacy"}}
    params = payload.get("parameters", {})
    if isinstance(params, Mapping):
        for key, value in params.items():
            if key == "seeds":
                row["parameter__seeds"] = ";".join(str(item) for item in value)
            else:
                row[f"parameter__{key}"] = value
    return row


def collect_checkpoints(paths: Paths, grid_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths.checkpoints.rglob("*.json")):
        payload = load_checkpoint(path, grid_id)
        rows.append(flatten_checkpoint(payload))
    return pd.DataFrame(rows)


def expected_configuration_count(grid: Mapping[str, Any]) -> int:
    per_combination = sum(len(grid["grid"][model]) for model in MODELS)
    return per_combination * len(SCENARIOS) * len(TARGET_REPRESENTATIONS)


def boundary_detail(model: str, selected: pd.Series, trials: pd.DataFrame) -> tuple[bool, str]:
    if model in {"mean", "median"}:
        return False, "not_applicable"
    details: list[str] = []
    if model == "ridge":
        value = float(selected["parameter__alpha"])
        if math.isclose(value, min(RIDGE_ALPHAS)):
            details.append("alpha=min")
        if math.isclose(value, max(RIDGE_ALPHAS)):
            details.append("alpha=max")
    elif model == "pls":
        value = int(selected["parameter__n_components"])
        valid = sorted(trials["parameter__n_components"].dropna().astype(int).unique())
        if value == min(valid):
            details.append("n_components=min")
        if value == max(valid):
            details.append("n_components=max")
    elif model == "knn":
        value = int(selected["parameter__n_neighbors"])
        valid = sorted(trials["parameter__n_neighbors"].dropna().astype(int).unique())
        if value == min(valid):
            details.append("n_neighbors=min")
        if value == max(valid):
            details.append("n_neighbors=max_predeclared")
    elif model == "multitask_elastic_net":
        fraction = float(selected["parameter__alpha_fraction"])
        ratio = float(selected["parameter__l1_ratio"])
        if math.isclose(fraction, min(MTEN_ALPHA_FRACTIONS)):
            details.append("alpha_fraction=weakest")
        if math.isclose(fraction, max(MTEN_ALPHA_FRACTIONS)):
            details.append("alpha_fraction=strongest")
        if math.isclose(ratio, min(MTEN_L1_RATIOS)):
            details.append("l1_ratio=min")
        if math.isclose(ratio, max(MTEN_L1_RATIOS)):
            details.append("l1_ratio=max")
    elif model == "extra_trees":
        # Grade categórica pequena: sinaliza parâmetros numéricos nas extremidades.
        for column in [
            "parameter__n_estimators",
            "parameter__max_depth",
            "parameter__min_samples_leaf",
        ]:
            values = pd.to_numeric(trials[column], errors="coerce").dropna().unique()
            selected_value = pd.to_numeric(pd.Series([selected.get(column)]), errors="coerce").iloc[0]
            if len(values) > 1 and pd.notna(selected_value):
                if math.isclose(float(selected_value), float(np.min(values))):
                    details.append(f"{column.removeprefix('parameter__')}=min")
                if math.isclose(float(selected_value), float(np.max(values))):
                    details.append(f"{column.removeprefix('parameter__')}=max")
    return bool(details), ";".join(details) if details else "interior_or_categorical"


def select_best(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trials.empty:
        return pd.DataFrame(), pd.DataFrame()
    best_rows: list[pd.Series] = []
    boundary_rows: list[dict[str, Any]] = []
    group_keys = ["scenario", "target_representation", "model"]
    for keys, group in trials.groupby(group_keys, sort=True):
        convergence = group.get(
            "converged",
            pd.Series(True, index=group.index, dtype=bool),
        ).map(parse_bool)
        converged_group = group.loc[convergence].copy()
        no_converged_configuration = converged_group.empty
        eligible = group.copy() if no_converged_configuration else converged_group
        ordered = eligible.sort_values(
            [SELECTION_METRIC, "participant_macro_mae", "configuration_order"],
            kind="mergesort",
        )
        selected = ordered.iloc[0].copy()
        boundary, detail = boundary_detail(str(keys[2]), selected, group)
        selected["selection_boundary"] = boundary
        selected["selection_boundary_detail"] = detail
        selected["selection_required_convergence"] = True
        selected["no_converged_configuration"] = no_converged_configuration
        selected["selected_configuration_converged"] = parse_bool(
            selected.get("converged", True)
        )
        best_rows.append(selected)
        boundary_rows.append(
            {
                "scenario": keys[0],
                "target_representation": keys[1],
                "model": keys[2],
                "selected_configuration_id": selected["configuration_id"],
                "selection_boundary": boundary,
                "selection_boundary_detail": detail,
                "selection_required_convergence": True,
                "no_converged_configuration": no_converged_configuration,
                "selected_configuration_converged": parse_bool(
                    selected.get("converged", True)
                ),
                "automatic_expansion_performed": False,
            }
        )
    best = pd.DataFrame(best_rows).reset_index(drop=True)
    boundary = pd.DataFrame(boundary_rows)
    return best, boundary


def model_ranking(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    output = best.copy()
    output["validation_rank"] = output.groupby(
        ["scenario", "target_representation"], sort=False
    )[SELECTION_METRIC].rank(method="min", ascending=True).astype(int)
    return output.sort_values(
        ["scenario", "target_representation", "validation_rank", "model"]
    )


def matched_longitudinal_comparison(best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (representation, model), group in best.groupby(
        ["target_representation", "model"], sort=True
    ):
        indexed = group.set_index("scenario")
        if not {
            "snapshot_matched_last",
            "longitudinal_previous_last",
        }.issubset(indexed.index):
            continue
        snapshot = indexed.loc["snapshot_matched_last"]
        longitudinal = indexed.loc["longitudinal_previous_last"]
        delta = float(longitudinal[SELECTION_METRIC] - snapshot[SELECTION_METRIC])
        rows.append(
            {
                "target_representation": representation,
                "model": model,
                "snapshot_matched_last_rmse": float(snapshot[SELECTION_METRIC]),
                "longitudinal_previous_last_rmse": float(longitudinal[SELECTION_METRIC]),
                "longitudinal_minus_snapshot_rmse": delta,
                "relative_difference_percent": float(
                    100.0 * delta / float(snapshot[SELECTION_METRIC])
                ),
                "negative_delta_favors_longitudinal": True,
                "same_target_group": True,
                "same_validation_participants": True,
            }
        )
    return pd.DataFrame(rows)


def full_pca_comparison(best: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario, model), group in best.groupby(["scenario", "model"], sort=True):
        indexed = group.set_index("target_representation")
        if not {"full", "pca90"}.issubset(indexed.index):
            continue
        full = indexed.loc["full"]
        pca = indexed.loc["pca90"]
        delta = float(pca[SELECTION_METRIC] - full[SELECTION_METRIC])
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "full_rmse": float(full[SELECTION_METRIC]),
                "pca90_rmse": float(pca[SELECTION_METRIC]),
                "pca90_minus_full_rmse": delta,
                "relative_difference_percent": float(
                    100.0 * delta / float(full[SELECTION_METRIC])
                ),
                "negative_delta_favors_pca90": True,
            }
        )
    return pd.DataFrame(rows)


def runtime_summary(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty:
        return pd.DataFrame()
    numeric = [
        "fit_seconds",
        "inference_seconds",
        "peak_rss_mb",
        "peak_rss_delta_mb",
    ]
    for column in numeric:
        trials[column] = pd.to_numeric(trials[column], errors="coerce")
    return (
        trials.groupby(["model"], sort=True)
        .agg(
            configurations_completed=("configuration_id", "count"),
            fit_seconds_total=("fit_seconds", "sum"),
            fit_seconds_median=("fit_seconds", "median"),
            inference_seconds_total=("inference_seconds", "sum"),
            peak_rss_mb_max=("peak_rss_mb", "max"),
            convergence_warning_count=("convergence_warning_count", "sum"),
        )
        .reset_index()
    )


def safe_story(
    grid_id: str,
    complete: bool,
    trial_count: int,
    expected_count: int,
    best: pd.DataFrame,
    boundary: pd.DataFrame,
) -> str:
    lines = [
        "# BENCHMARK08B — Tuning de modelos em desenvolvimento",
        "",
        f"- Versão do script: `{SCRIPT_VERSION}`",
        f"- Política de modelos: `{MODEL_POLICY_VERSION}`",
        f"- Model-grid decision ID: `{grid_id}`",
        "- Fit: **treino somente**",
        "- Seleção: **validação participant-macro RMSE**",
        "- Teste usado/materializado: **não**",
        "- Predições individuais salvas: **não**",
        f"- Execução completa: **{'sim' if complete else 'não'}**",
        f"- Configurações concluídas: **{trial_count}/{expected_count}**",
        "",
        "## Grade congelada",
        "",
        f"- Ridge: {list(RIDGE_ALPHAS)}",
        f"- PLS: {list(PLS_COMPONENTS)} componentes",
        f"- KNN: k={list(KNN_NEIGHBORS)}, pesos={list(KNN_WEIGHTS)}, p={KNN_P}",
        (
            "- MultiTaskElasticNet: alpha fractions="
            f"{list(MTEN_ALPHA_FRACTIONS)}, l1 ratios={list(MTEN_L1_RATIOS)}"
        ),
        (
            "- Extra Trees: três configurações predefinidas; cada uma avaliada "
            f"como ensemble das seeds {list(EXTRA_TREES_SEEDS)}"
        ),
        "",
        "Nenhuma grade é expandida automaticamente após observar a validação.",
        "",
    ]
    if not best.empty:
        lines.extend(
            [
                "## Melhores configurações por bloco",
                "",
                "| Cenário | Alvo | Modelo | RMSE macro | MAE macro | Borda |",
                "|---|---|---|---:|---:|---|",
            ]
        )
        for row in best.sort_values(
            ["scenario", "target_representation", "model"]
        ).itertuples(index=False):
            lines.append(
                f"| `{row.scenario}` | `{row.target_representation}` | "
                f"`{row.model}` | {row.participant_macro_rmse:.6f} | "
                f"{row.participant_macro_mae:.6f} | "
                f"{'sim' if row.selection_boundary else 'não'} |"
            )
        lines.append("")
    if not boundary.empty and bool(boundary["selection_boundary"].any()):
        lines.extend(
            [
                "## Revisão de bordas",
                "",
                "Vencedores em borda são sinalizados para interpretação, mas não disparam nova busca.",
                "",
            ]
        )
    lines.extend(
        [
            "## Privacidade",
            "",
            "Os checkpoints e relatórios contêm apenas configurações e métricas agregadas.",
            "Nenhum RID, VISCODE2, target individual, erro individual ou predição individual é exportado.",
            "",
        ]
    )
    return "\n".join(lines)


def write_checks(paths: Paths, checks: Checks) -> None:
    frame = pd.DataFrame([asdict(item) for item in checks.items])
    atomic_write_csv(frame, paths.safe / "benchmark08b_validation_checks.csv")


def aggregate_and_write(
    paths: Paths,
    grid: Mapping[str, Any],
    grid_id: str,
    checks: Checks,
    validate_only: bool,
    decision_ids: Mapping[str, str],
) -> dict[str, Any]:
    trials = collect_checkpoints(paths, grid_id)
    expected_count = expected_configuration_count(grid)
    complete = len(trials) == expected_count

    if not trials.empty:
        trials = trials.sort_values(
            ["scenario", "target_representation", "model", "configuration_order"]
        )
        best, boundary = select_best(trials)
        ranking = model_ranking(best)
        matched = matched_longitudinal_comparison(best)
        full_pca = full_pca_comparison(best)
        runtime = runtime_summary(trials.copy())
    else:
        best = pd.DataFrame()
        boundary = pd.DataFrame()
        ranking = pd.DataFrame()
        matched = pd.DataFrame()
        full_pca = pd.DataFrame()
        runtime = pd.DataFrame()

    outputs = {
        "benchmark08b_trial_metrics.csv": trials,
        "benchmark08b_best_by_model.csv": best,
        "benchmark08b_model_ranking.csv": ranking,
        "benchmark08b_boundary_review.csv": boundary,
        "benchmark08b_matched_longitudinal_comparison.csv": matched,
        "benchmark08b_full_pca90_comparison.csv": full_pca,
        "benchmark08b_runtime_summary.csv": runtime,
    }
    for filename, frame in outputs.items():
        atomic_write_csv(frame, paths.safe / filename)

    boundary_count = (
        int(boundary["selection_boundary"].sum()) if not boundary.empty else 0
    )
    summary = {
        "stage": STAGE_NAME,
        "script_version": SCRIPT_VERSION,
        "model_policy_version": MODEL_POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only_passed": checks.passed,
        "model_selection_complete": bool(complete and checks.passed and not validate_only),
        "validate_only": bool(validate_only),
        "model_grid_decision_id": grid_id,
        "preprocessing_decision_id": decision_ids[
            "preprocessing_decision_id"
        ],
        "target_decision_id": decision_ids[
            "target_decision_id"
        ],
        "scenario_decision_id": decision_ids[
            "scenario_decision_id"
        ],
        "temporal_policy_decision_id": EXPECTED_TEMPORAL_POLICY_DECISION_ID,
        "check_count": len(checks.items),
        "failed_check_count": checks.failed,
        "fatal_error": None,
        "scope": {
            "fit_split": TRAIN,
            "evaluation_split": VALIDATION,
            "test_values_used": False,
            "test_materialized": False,
            "predictions_saved": False,
            "individual_errors_saved": False,
            "bootstrap_performed": False,
            "final_configuration_lock_created": False,
        },
        "progress": {
            "completed_configurations": int(len(trials)),
            "expected_configurations": int(expected_count),
            "completion_fraction": float(len(trials) / expected_count),
            "selected_blocks_available": int(len(best)),
            "expected_selected_blocks": int(
                len(SCENARIOS) * len(TARGET_REPRESENTATIONS) * len(MODELS)
            ),
            "boundary_winner_count": boundary_count,
        },
        "selection": {
            "primary_metric": SELECTION_METRIC,
            "metric_space": "standardized reconstructed Full 323-dimensional target",
            "eligibility": "prefer only converged configurations; flag if none converged",
            "tie_breakers": ["participant_macro_mae", "configuration_order"],
            "automatic_grid_expansion": False,
            "best_by_model": best.to_dict(orient="records"),
        },
        "artifacts": {
            "model_grid": str(paths.model_grid.relative_to(paths.root)),
            "model_grid_sha256": sha256_file(paths.model_grid),
            "selection_policy": str(paths.selection_policy.relative_to(paths.root)),
            "selection_policy_sha256": sha256_file(paths.selection_policy),
            "safe_checkpoint_count": int(len(trials)),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "psutil": None if psutil is None else psutil.__version__,
        },
        "privacy": {
            "safe_outputs_contain_participant_identifiers": False,
            "safe_outputs_contain_visit_identifiers": False,
            "safe_outputs_contain_target_names": False,
            "safe_outputs_contain_individual_target_values": False,
            "safe_outputs_contain_individual_predictions": False,
            "safe_outputs_contain_individual_errors": False,
        },
    }
    atomic_write_json(paths.safe / "benchmark08b_summary.json", summary)
    atomic_write_text(
        paths.safe / "BENCHMARK08B_MODEL_SELECTION_STORY.md",
        safe_story(grid_id, complete, len(trials), expected_count, best, boundary),
    )
    write_checks(paths, checks)
    return summary


def prepare_output(paths: Paths, args: argparse.Namespace) -> None:
    if args.overwrite and paths.stage.exists():
        shutil.rmtree(paths.stage)
    paths.safe.mkdir(parents=True, exist_ok=True)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    existing_checkpoints = list(paths.checkpoints.rglob("*.json"))
    if existing_checkpoints and not args.resume and not args.overwrite:
        raise ModelStageAbort(
            "BENCHMARK08B checkpoints already exist. Use --resume to continue or "
            "--overwrite only for an intentional restart."
        )


def run_self_test() -> int:
    rng = np.random.default_rng(123)
    participants = np.asarray(["a", "a", "b", "c"])
    truth = rng.normal(size=(4, TARGET_COUNT))
    pred = truth + rng.normal(scale=0.1, size=truth.shape)
    metrics = aggregate_metrics(truth, pred, participants)
    assert metrics["participant_macro_rmse"] > 0

    x = rng.normal(size=(30, 8))
    y = rng.normal(size=(30, 5))
    alpha_base = centered_alpha_base(x, y)
    assert alpha_base > 0

    ridge = Ridge(alpha=10.0, solver="lsqr").fit(sparse.csr_matrix(x), y)
    assert ridge.predict(sparse.csr_matrix(x[:3])).shape == (3, 5)

    pls = PLSRegression(n_components=2, scale=False).fit(x, y)
    assert pls.predict(x[:3]).shape == (3, 5)

    knn = KNeighborsRegressor(n_neighbors=3, algorithm="brute").fit(
        sparse.csr_matrix(x), y
    )
    assert knn.predict(sparse.csr_matrix(x[:3])).shape == (3, 5)

    mten = MultiTaskElasticNet(
        alpha=alpha_base * 0.1 / 0.5,
        l1_ratio=0.5,
        max_iter=100,
        tol=1e-3,
    ).fit(x, y)
    assert mten.predict(x[:3]).shape == (3, 5)

    tree = ExtraTreesRegressor(n_estimators=5, random_state=1, n_jobs=1).fit(x, y)
    assert tree.predict(x[:3]).shape == (3, 5)

    print("BENCHMARK08B SELF-TEST PASSED")
    return 0


def run_stage(paths: Paths, args: argparse.Namespace, checks: Checks) -> dict[str, Any]:
    b06, b07, pre_lock, target_lock = load_and_validate_upstream(paths, checks)
    decision_ids = {
        "preprocessing_decision_id": str(
            b06["preprocessing_decision_id"]
        ).strip(),
        "target_decision_id": str(
            b07["target_decision_id"]
        ).strip(),
        "scenario_decision_id": str(
            b06["scenario_decision_id"]
        ).strip(),
    }
    grid, grid_id = freeze_policy_files(
        paths,
        args,
        decision_ids,
    )
    verify_local_hashes(paths.b06_local, pre_lock, "benchmark06", checks)
    verify_local_hashes(paths.b07_local, target_lock, "benchmark07", checks)

    data_by_scenario: dict[str, ScenarioData] = {}
    for scenario in SCENARIOS:
        data_by_scenario[scenario] = load_scenario_data(
            paths, scenario, b06, b07, checks
        )

    # Proteção do pareamento: X atual difere por histórico, mas Y e ordem devem ser idênticos.
    for split_name, manifest_attr in [
        (TRAIN, "input_train_manifest"),
        (VALIDATION, "input_validation_manifest"),
    ]:
        matched_manifest = getattr(
            data_by_scenario["snapshot_matched_last"], manifest_attr
        )
        longitudinal_manifest = getattr(
            data_by_scenario["longitudinal_previous_last"], manifest_attr
        )
        checks.add(
            f"matched_longitudinal_row_order__{split_name}",
            key_tuples(matched_manifest) == key_tuples(longitudinal_manifest),
            True,
            key_tuples(matched_manifest) == key_tuples(longitudinal_manifest),
        )
    checks.add(
        "matched_longitudinal_y_full_train_identical",
        np.array_equal(
            data_by_scenario["snapshot_matched_last"].y_full_train,
            data_by_scenario["longitudinal_previous_last"].y_full_train,
        ),
        True,
        np.array_equal(
            data_by_scenario["snapshot_matched_last"].y_full_train,
            data_by_scenario["longitudinal_previous_last"].y_full_train,
        ),
    )
    checks.add(
        "matched_longitudinal_y_full_validation_identical",
        np.array_equal(
            data_by_scenario["snapshot_matched_last"].y_full_validation,
            data_by_scenario["longitudinal_previous_last"].y_full_validation,
        ),
        True,
        np.array_equal(
            data_by_scenario["snapshot_matched_last"].y_full_validation,
            data_by_scenario["longitudinal_previous_last"].y_full_validation,
        ),
    )
    checks.add("test_not_loaded", True, True, True)

    if not checks.passed:
        raise ModelStageAbort("BENCHMARK08B pre-fit validation failed.")

    if args.validate_only:
        logging.info("Validation-only mode: no model will be fitted.")
        return aggregate_and_write(
            paths,
            grid,
            grid_id,
            checks,
            validate_only=True,
            decision_ids=decision_ids,
        )

    for scenario in args.scenarios:
        data = data_by_scenario[scenario]
        for representation in args.target_representations:
            for model_name in args.models:
                execute_model_grid(
                    paths,
                    args,
                    grid,
                    grid_id,
                    data,
                    representation,
                    model_name,
                    decision_ids,
                )

    return aggregate_and_write(
        paths,
        grid,
        grid_id,
        checks,
        validate_only=False,
        decision_ids=decision_ids,
    )


def write_failure(paths: Paths, checks: Checks, error: str) -> None:
    paths.safe.mkdir(parents=True, exist_ok=True)
    write_checks(paths, checks)
    atomic_write_json(
        paths.safe / "benchmark08b_summary.json",
        {
            "stage": STAGE_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_only_passed": False,
            "model_selection_complete": False,
            "check_count": len(checks.items),
            "failed_check_count": checks.failed,
            "fatal_error": error,
            "scope": {
                "test_values_used": False,
                "test_materialized": False,
                "individual_predictions_saved": False,
            },
        },
    )


def main() -> int:
    args = parse_args()

    if args.self_test:
        return run_self_test()

    paths = resolve_paths(args)
    checks = Checks()

    # A pasta antiga precisa ser removida antes da abertura
    # de benchmark08.log no Windows.
    try:
        prepare_output(paths, args)
    except Exception as exc:
        print(
            "BENCHMARK08B FAILED DURING OUTPUT PREPARATION | "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    configure_logging(paths)

    logging.info("%s v%s", STAGE_NAME, SCRIPT_VERSION)
    logging.info("Project root: %s", paths.root)
    logging.info("Models requested: %s", ", ".join(args.models))
    logging.info("Scenarios requested: %s", ", ".join(args.scenarios))
    logging.info(
        "Target representations requested: %s",
        ", ".join(args.target_representations),
    )
    logging.info(
        "Fit=train; selection=validation; test remains closed."
    )

    try:
        summary = run_stage(paths, args, checks)
    except Exception as exc:
        logging.exception("BENCHMARK08B fatal error")
        write_failure(
            paths,
            checks,
            f"{type(exc).__name__}: {exc}",
        )
        logging.error(
            "BENCHMARK08B FAILED | failed_checks=%d | "
            "fatal_error=%s: %s",
            checks.failed,
            type(exc).__name__,
            exc,
        )
        return 1

    progress = summary.get("progress", {})
    logging.info(
        "BENCHMARK08B %s | completed=%s/%s | "
        "test_materialized=false",
        (
            "PASSED"
            if summary.get("model_selection_complete")
            else "PARTIAL/VALIDATED"
        ),
        progress.get("completed_configurations", 0),
        progress.get("expected_configurations", 0),
    )
    logging.info("Safe reports: %s", paths.safe)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
