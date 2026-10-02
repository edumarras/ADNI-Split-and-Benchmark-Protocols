from pathlib import Path

import pytest

from adni_benchmark.source_discovery import REQUIRED_SOURCES
from adni_benchmark.source_inspection import SourceSnapshot
from adni_benchmark.source_schema import (
    SOURCE_CONTRACTS,
    SourceContract,
    SourceSchemaError,
    validate_all_source_schemas,
    validate_contract_registry,
    validate_source_schema,
)


def make_snapshot(
    source_name: str,
    columns: tuple[str, ...],
) -> SourceSnapshot:
    return SourceSnapshot(
        source_name=source_name,
        absolute_path=Path(f"{source_name}_TEST.csv"),
        relative_path=Path(f"{source_name}_TEST.csv"),
        sha256="0" * 64,
        size_bytes=100,
        encoding="utf-8-sig",
        columns=columns,
    )


def test_contract_registry_matches_required_sources() -> None:
    validate_contract_registry()

    assert set(SOURCE_CONTRACTS) == set(
        REQUIRED_SOURCES
    )


def test_all_minimum_valid_schemas_pass() -> None:
    snapshots = {
        source_name: make_snapshot(
            source_name,
            contract.required_columns,
        )
        for source_name, contract
        in SOURCE_CONTRACTS.items()
    }

    results = validate_all_source_schemas(
        snapshots
    )

    assert tuple(results.keys()) == REQUIRED_SOURCES

    assert all(
        result.is_valid
        for result in results.values()
    )


def test_extra_columns_are_allowed() -> None:
    contract = SOURCE_CONTRACTS["ADAS"]

    snapshot = make_snapshot(
        "ADAS",
        (
            *contract.required_columns,
            "EXTRA_COLUMN",
        ),
    )

    result = validate_source_schema(
        snapshot,
        contract,
    )

    assert result.is_valid
    assert result.n_source_columns == 3


def test_missing_required_column_is_reported() -> None:
    contract = SOURCE_CONTRACTS["DXSUM"]

    snapshot = make_snapshot(
        "DXSUM",
        (
            "RID",
            "VISCODE2",
        ),
    )

    result = validate_source_schema(
        snapshot,
        contract,
    )

    assert not result.is_valid

    assert result.missing_required_columns == (
        "DIAGNOSIS",
    )


def test_invalid_schema_aborts_full_validation() -> None:
    snapshots = {
        source_name: make_snapshot(
            source_name,
            contract.required_columns,
        )
        for source_name, contract
        in SOURCE_CONTRACTS.items()
    }

    snapshots["UCSFFSX7"] = make_snapshot(
        "UCSFFSX7",
        (
            "RID",
            "VISCODE2",
            "IMAGEUID",
            "EXAMDATE",
            "STATUS",
            "FSVER",
        ),
    )

    with pytest.raises(
        SourceSchemaError,
        match="OVERALLQC",
    ):
        validate_all_source_schemas(
            snapshots
        )


def test_missing_snapshot_aborts() -> None:
    snapshots = {
        source_name: make_snapshot(
            source_name,
            contract.required_columns,
        )
        for source_name, contract
        in SOURCE_CONTRACTS.items()
        if source_name != "FAQ"
    }

    with pytest.raises(
        SourceSchemaError,
        match="FAQ",
    ):
        validate_all_source_schemas(
            snapshots
        )


def test_registry_rejects_key_not_required() -> None:
    invalid_contract = SourceContract(
        source_name="TEST",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
        ),
    )

    with pytest.raises(
        SourceSchemaError,
        match="key columns",
    ):
        validate_contract_registry(
            required_sources=("TEST",),
            contracts={
                "TEST": invalid_contract,
            },
        )


def test_snapshot_contract_mismatch_aborts() -> None:
    snapshot = make_snapshot(
        "ADAS",
        (
            "RID",
            "VISCODE2",
        ),
    )

    with pytest.raises(
        SourceSchemaError,
        match="different sources",
    ):
        validate_source_schema(
            snapshot,
            SOURCE_CONTRACTS["CDR"],
        )