from __future__ import annotations

from scripts.analyze_cutoff_trained_retrospective import (
    formula_identity_token as cutoff_formula_identity_token,
)
from scripts.summarize_synthesis_retrodiction import (
    group_first_reports,
    earliest_year_mean_sensitivity,
    identity_comparison_audit,
    legacy_hybrid_formula_identity_token,
    nominal_formula_identity_token,
    read_scored,
)


def test_current_cutoff_and_postprocessor_use_one_nominal_identity() -> None:
    cases = [
        ("LaMnO3", False),
        ("La2Mn2O6", False),
        ("Cr0.3333I2Mn0.6666", False),
        ("Cr0.6666I4Mn1.3332", False),
        ("Ba0.5000004Sr0.4999996TiO3", True),
        ("Ba1.0000008Sr0.9999992Ti2O6", True),
    ]
    for formula, is_partial in cases:
        assert nominal_formula_identity_token(formula) == cutoff_formula_identity_token(
            formula, is_partial
        )


def test_legacy_hybrid_identity_is_scale_invariant_within_each_occupancy_path() -> None:
    assert legacy_hybrid_formula_identity_token(
        "LaMnO3", False
    ) == legacy_hybrid_formula_identity_token(
        "La2Mn2O6", False
    )
    assert legacy_hybrid_formula_identity_token(
        "Ba0.5000004Sr0.4999996TiO3", True
    ) == legacy_hybrid_formula_identity_token(
        "Ba1.0000008Sr0.9999992Ti2O6", True
    )


def test_nominal_formula_identity_does_not_depend_on_occupancy_path() -> None:
    formula = "Eu2.667Mo4O16"
    assert nominal_formula_identity_token(formula) == nominal_formula_identity_token(
        formula
    )
    assert legacy_hybrid_formula_identity_token(
        formula, False
    ) != legacy_hybrid_formula_identity_token(formula, True)


def test_identity_audit_finds_cross_occupancy_split_and_hybrid_merge() -> None:
    rows = [
        {
            "formula_identity": "nominal-a",
            "cutoff_hybrid_formula_identity": "hybrid-a-full",
            "has_partial_occupancy": 0,
        },
        {
            "formula_identity": "nominal-a",
            "cutoff_hybrid_formula_identity": "hybrid-a-partial",
            "has_partial_occupancy": 1,
        },
        {
            "formula_identity": "nominal-b",
            "cutoff_hybrid_formula_identity": "hybrid-a-full",
            "has_partial_occupancy": 1,
        },
    ]
    audit = identity_comparison_audit(rows)
    assert audit["n_nominal_identities_split_by_hybrid_across_occupancy"] == 1
    assert audit["n_hybrid_identities_merging_distinct_nominal_compositions"] == 1
    assert audit["n_mixed_occupancy_nominal_identities"] == 1
    assert audit["n_partial_only_nominal_identities"] == 1


def test_first_report_tie_uses_lowest_icsd_identifier_and_audits_mixed_basin() -> None:
    rows = [
        {
            "cif_id": 9,
            "formula_identity": "A",
            "year": 2001,
            "is_in_basin": 0,
            "A_i": 2.0,
        },
        {
            "cif_id": 3,
            "formula_identity": "A",
            "year": 2001,
            "is_in_basin": 1,
            "A_i": 0.0,
        },
        {
            "cif_id": 1,
            "formula_identity": "A",
            "year": 2002,
            "is_in_basin": 1,
            "A_i": 1.0,
        },
        {
            "cif_id": 4,
            "formula_identity": "B",
            "year": 2004,
            "is_in_basin": 1,
            "A_i": 3.0,
        },
    ]
    selected, audit = group_first_reports(rows)
    assert [row["cif_id"] for row in selected] == [3, 4]
    assert audit["n_with_multiple_entries_in_earliest_year"] == 1
    assert audit["n_with_mixed_basin_status_in_earliest_year"] == 1
    sensitivity = earliest_year_mean_sensitivity(rows)
    assert sensitivity["n_formula_identities"] == 2


def test_scored_csv_preserves_sodium_nitride_formula(tmp_path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text(
        "cif_id,reduced_formula,year,assigned_community,"
        "nearest_centroid_distance,is_in_basin,A_i\n"
        "184990,NaN,2012,4,0.25,1,-0.5\n"
    )
    rows = read_scored(path)
    assert rows[0]["reduced_formula"] == "NaN"
