import pandas as pd

from adni_benchmark_protocol.auditA06_freeze_crosschecked_inputs import (
    APOE_ALLOWED_GENOTYPES,
    MMSE_CANONICAL_WORLD,
    RID,
    SPLIT,
    VISCODE2,
    derive_mmse_world_score,
    harmonize_adas_unable,
    harmonize_ptraccat_value,
)


def test_mmse_world_score_harmonization() -> None:
    frame = pd.DataFrame(
        {
            RID: ["XX0001", "XX0002", "XX0003"],
            VISCODE2: ["bl", "bl", "bl"],
            SPLIT: ["train", "train", "train"],
            "MMD": ["1", pd.NA, "1"],
            "MML": ["1", pd.NA, "0"],
            "MMR": ["1", pd.NA, "1"],
            "MMO": ["0", pd.NA, "0"],
            "MMW": ["1", pd.NA, "1"],
            "WORLDSCORE": [pd.NA, "3", pd.NA],
        }
    )

    transformed, stats = derive_mmse_world_score(
        frame,
        [],
    )

    assert transformed[
        MMSE_CANONICAL_WORLD
    ].tolist() == [
        4.0,
        3.0,
        3.0,
    ]

    assert stats[
        "representation_conflicts"
    ] == 0


def test_ptraccat_harmonization() -> None:
    assert harmonize_ptraccat_value(
        "8"
    )[0] == "3"

    assert harmonize_ptraccat_value(
        "9"
    )[0] == "3"

    assert harmonize_ptraccat_value(
        "1|5"
    )[0] == "6"

    assert harmonize_ptraccat_value(
        "5"
    )[0] == "5"


def test_adas_unable_implicit_conducted() -> None:
    frame = pd.DataFrame(
        {
            RID: [
                "XX0001",
                "XX0002",
            ],
            VISCODE2: [
                "bl",
                "bl",
            ],
            SPLIT: [
                "train",
                "train",
            ],
            "Q1UNABLE": [
                pd.NA,
                "2",
            ],
            "Q1SCORE": [
                "4",
                pd.NA,
            ],
        }
    )

    transformed, stats = (
        harmonize_adas_unable(
            frame,
            [],
        )
    )

    assert transformed.loc[
        0,
        "Q1UNABLE",
    ] == "0"

    assert transformed.loc[
        1,
        "Q1UNABLE",
    ] == "2"

    assert stats[
        "Q1UNABLE"
    ][
        "implicit_conducted_filled"
    ] == 1


def test_apoe_domain_has_six_categories() -> None:
    assert len(
        APOE_ALLOWED_GENOTYPES
    ) == 6
