from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final, Mapping

from adni_benchmark.source_discovery import REQUIRED_SOURCES
from adni_benchmark.source_inspection import SourceSnapshot


class SourceSchemaError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceContract:
    source_name: str
    grain: str
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class SchemaValidationResult:
    source_name: str
    grain: str
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    n_source_columns: int

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_columns


SOURCE_CONTRACTS: Final[dict[str, SourceContract]] = {
    "DXSUM": SourceContract(
        source_name="DXSUM",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
            "DIAGNOSIS",
        ),
    ),
    "ADAS": SourceContract(
        source_name="ADAS",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "CDR": SourceContract(
        source_name="CDR",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "FAQ": SourceContract(
        source_name="FAQ",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "MMSE": SourceContract(
        source_name="MMSE",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "MOCA": SourceContract(
        source_name="MOCA",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "NEUROBAT": SourceContract(
        source_name="NEUROBAT",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
        ),
    ),
    "PTDEMOG": SourceContract(
        source_name="PTDEMOG",
        grain="participant_baseline",
        key_columns=(
            "RID",
        ),
        required_columns=(
            "RID",
            "PHASE",
            "VISCODE2",
            "PTGENDER",
        ),
    ),
    "APOERES": SourceContract(
        source_name="APOERES",
        grain="participant",
        key_columns=(
            "RID",
        ),
        required_columns=(
            "RID",
            "GENOTYPE",
        ),
    ),
    "UCSFFSX7": SourceContract(
        source_name="UCSFFSX7",
        grain="visit",
        key_columns=(
            "RID",
            "VISCODE2",
        ),
        required_columns=(
            "RID",
            "VISCODE2",
            "IMAGEUID",
            "EXAMDATE",
            "STATUS",
            "OVERALLQC",
            "FSVER",
        ),
    ),
}


def validate_contract_registry(
    required_sources: tuple[str, ...] = REQUIRED_SOURCES,
    contracts: Mapping[str, SourceContract] = SOURCE_CONTRACTS,
) -> None:
    # Every required source must have exactly one contract
    required_names = set(required_sources)
    contract_names = set(contracts)

    missing_contracts = sorted(
        required_names - contract_names
    )

    unexpected_contracts = sorted(
        contract_names - required_names
    )

    if missing_contracts or unexpected_contracts:
        raise SourceSchemaError(
            "Source contract registry mismatch. "
            f"Missing contracts: {missing_contracts}. "
            f"Unexpected contracts: {unexpected_contracts}."
        )

    for source_name, contract in contracts.items():
        # The dictionary key and the contract identity must agree
        if contract.source_name != source_name:
            raise SourceSchemaError(
                "Source contract name mismatch. "
                f"Registry key={source_name}, "
                f"contract source_name={contract.source_name}."
            )

        counts = Counter(
            contract.required_columns
        )

        duplicated_required = sorted(
            column
            for column, count in counts.items()
            if count > 1
        )

        if duplicated_required:
            raise SourceSchemaError(
                f"{source_name} has duplicated required columns: "
                f"{duplicated_required}"
            )

        # Every key must necessarily be available in the required schema
        missing_keys = sorted(
            set(contract.key_columns)
            - set(contract.required_columns)
        )

        if missing_keys:
            raise SourceSchemaError(
                f"{source_name} key columns are not included "
                f"in required_columns: {missing_keys}"
            )


def validate_source_schema(
    snapshot: SourceSnapshot,
    contract: SourceContract,
) -> SchemaValidationResult:
    # This stage checks structure only, never participant-level values
    if snapshot.source_name != contract.source_name:
        raise SourceSchemaError(
            "Snapshot and contract refer to different sources. "
            f"Snapshot={snapshot.source_name}, "
            f"contract={contract.source_name}."
        )

    available_columns = set(
        snapshot.columns
    )

    missing_required_columns = tuple(
        column
        for column in contract.required_columns
        if column not in available_columns
    )

    return SchemaValidationResult(
        source_name=snapshot.source_name,
        grain=contract.grain,
        key_columns=contract.key_columns,
        required_columns=contract.required_columns,
        missing_required_columns=missing_required_columns,
        n_source_columns=snapshot.n_columns,
    )


def validate_all_source_schemas(
    snapshots: Mapping[str, SourceSnapshot],
) -> dict[str, SchemaValidationResult]:
    # Validate the protocol definition before applying it
    validate_contract_registry()

    missing_snapshots = [
        source_name
        for source_name in REQUIRED_SOURCES
        if source_name not in snapshots
    ]

    if missing_snapshots:
        raise SourceSchemaError(
            f"Missing source snapshots: {missing_snapshots}"
        )

    results: dict[str, SchemaValidationResult] = {}

    for source_name in REQUIRED_SOURCES:
        result = validate_source_schema(
            snapshot=snapshots[source_name],
            contract=SOURCE_CONTRACTS[source_name],
        )

        results[source_name] = result

    invalid_results = [
        result
        for result in results.values()
        if not result.is_valid
    ]

    if invalid_results:
        lines = [
            "One or more sources violate their schema contract:"
        ]

        for result in invalid_results:
            missing = ", ".join(
                result.missing_required_columns
            )

            lines.append(
                f"  - {result.source_name}: "
                f"missing [{missing}]"
            )

        raise SourceSchemaError(
            "\n".join(lines)
        )

    return results