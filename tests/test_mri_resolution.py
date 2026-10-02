import pandas as pd
import pytest

from adni_benchmark.mri_resolution import (
    MRIResolutionError,
    resolve_mri_source,
)


TARGET_COLUMNS = (
    "ST1SV",
    "ST2SV",
)


def make_frame(
    records: list[dict[str, object]],
) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        dtype="string",
    )


def make_row(
    rid: object,
    viscode2: object,
    imageuid: object,
    st1: object = "1.0",
    st2: object = "2.0",
    status: object = "PARTIAL",
) -> dict[str, object]:
    return {
        "RID": rid,
        "VISCODE2": viscode2,
        "IMAGEUID": imageuid,
        "STATUS": status,
        "ST1SV": st1,
        "ST2SV": st2,
    }


def test_unique_candidate_is_selected() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
            )
        ]
    )

    resolved, unresolved, summary = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert len(resolved) == 1
    assert unresolved.empty

    assert (
        resolved.loc[
            0,
            "__resolution_status",
        ]
        == "unique_candidate"
    )

    assert summary.resolved_rows == 1


def test_unique_maximum_target_completeness_is_selected() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
                st2=pd.NA,
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="101",
                st2="2.0",
            ),
        ]
    )

    resolved, unresolved, summary = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert len(resolved) == 1
    assert unresolved.empty

    assert resolved.loc[0, "IMAGEUID"] == "101"

    assert (
        resolved.loc[
            0,
            "__resolution_status",
        ]
        == "unique_maximum_target_completeness"
    )

    assert summary.unresolved_groups == 0


def test_equivalent_tied_candidates_use_lowest_imageuid() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="200",
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
            ),
        ]
    )

    resolved, unresolved, _ = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert unresolved.empty
    assert resolved.loc[0, "IMAGEUID"] == "100"

    assert (
        resolved.loc[
            0,
            "__resolution_status",
        ]
        == "equivalent_maximum_target_candidates"
    )


def test_non_equivalent_maximum_candidates_are_unresolved() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
                st1="1.0",
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="101",
                st1="9.0",
            ),
        ]
    )

    resolved, unresolved, summary = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert resolved.empty
    assert len(unresolved) == 1

    assert (
        unresolved.loc[
            0,
            "resolution_status",
        ]
        == "unresolved_maximum_target_conflict"
    )

    assert summary.unresolved_groups == 1


def test_status_does_not_override_target_conflict() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
                st1="1.0",
                status="COMPLETE",
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="101",
                st1="9.0",
                status="PARTIAL",
            ),
        ]
    )

    resolved, unresolved, _ = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert resolved.empty
    assert len(unresolved) == 1


def test_lower_completeness_candidate_cannot_win_from_status() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
                st2=pd.NA,
                status="COMPLETE",
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="101",
                st2="2.0",
                status="PARTIAL",
            ),
        ]
    )

    resolved, _, _ = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert resolved.loc[0, "IMAGEUID"] == "101"
    assert resolved.loc[0, "STATUS"] == "PARTIAL"


def test_missing_visit_key_is_excluded_before_resolution() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
            ),
            make_row(
                rid="2",
                viscode2=pd.NA,
                imageuid="200",
            ),
        ]
    )

    resolved, unresolved, summary = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert len(resolved) == 1
    assert unresolved.empty

    assert summary.original_rows == 2
    assert summary.valid_key_rows == 1
    assert summary.rows_with_missing_key == 1


def test_missing_imageuid_with_valid_key_aborts() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid=pd.NA,
            )
        ]
    )

    with pytest.raises(
        MRIResolutionError,
        match="missing IMAGEUID",
    ):
        resolve_mri_source(
            frame=frame,
            target_columns=TARGET_COLUMNS,
        )


def test_non_integer_imageuid_aborts() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100.5",
            )
        ]
    )

    with pytest.raises(
        MRIResolutionError,
        match="non-integer IMAGEUID",
    ):
        resolve_mri_source(
            frame=frame,
            target_columns=TARGET_COLUMNS,
        )


def test_resolution_never_coalesces_cells_between_rows() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
                st1="1",
                st2=pd.NA,
            ),
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="101",
                st1=pd.NA,
                st2="2",
            ),
        ]
    )

    resolved, unresolved, _ = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert resolved.empty
    assert len(unresolved) == 1


def test_same_imageuid_equivalent_rows_are_order_independent() -> None:
    records = [
        make_row(
            rid="1",
            viscode2="bl",
            imageuid="100",
            status="COMPLETE",
        ),
        make_row(
            rid="1",
            viscode2="bl",
            imageuid="100",
            status="PARTIAL",
        ),
    ]

    first_resolved, _, _ = resolve_mri_source(
        frame=make_frame(records),
        target_columns=TARGET_COLUMNS,
    )

    second_resolved, _, _ = resolve_mri_source(
        frame=make_frame(
            list(reversed(records))
        ),
        target_columns=TARGET_COLUMNS,
    )

    assert (
        first_resolved.loc[0, "STATUS"]
        == second_resolved.loc[0, "STATUS"]
    )


def test_resolved_keys_are_unique() -> None:
    frame = make_frame(
        [
            make_row(
                rid="1",
                viscode2="bl",
                imageuid="100",
            ),
            make_row(
                rid="2",
                viscode2="bl",
                imageuid="200",
            ),
        ]
    )

    resolved, _, _ = resolve_mri_source(
        frame=frame,
        target_columns=TARGET_COLUMNS,
    )

    assert not resolved.duplicated(
        subset=[
            "RID",
            "VISCODE2",
        ],
        keep=False,
    ).any()