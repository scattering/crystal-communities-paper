from __future__ import annotations

from scripts.formula_conventions import (
    anonymous_stoichiometry_key,
    build_formula_precedent_index,
    formula_has_precedent,
    normalized_fraction_key,
    scale_invariant_formula_key,
)


def test_formula_key_is_invariant_to_formula_unit_and_element_order() -> None:
    expected = scale_invariant_formula_key("CaLa3Mn4O12")
    assert scale_invariant_formula_key("Ca2La6Mn8O24") == expected
    assert scale_invariant_formula_key("O12Mn4La3Ca") == expected


def test_anonymous_stoichiometry_follows_descending_optimade_pattern() -> None:
    assert anonymous_stoichiometry_key("MgAl2O4") == (4, 2, 1)


def test_decimal_formula_integer_and_anonymous_keys_are_scale_invariant() -> None:
    formula = "Cr0.3333I2Mn0.6666"
    doubled = "Cr0.6666I4Mn1.3332"
    assert scale_invariant_formula_key(formula) == scale_invariant_formula_key(doubled)
    assert anonymous_stoichiometry_key(formula) == anonymous_stoichiometry_key(doubled)


def test_isotope_labels_collapse_to_elemental_composition_before_reduction() -> None:
    isotope_labelled = "H1.94D1.06NaO6Se2"
    elemental = "H3NaO6Se2"
    assert scale_invariant_formula_key(isotope_labelled) == scale_invariant_formula_key(elemental)
    assert anonymous_stoichiometry_key(isotope_labelled) == anonymous_stoichiometry_key(elemental)


def test_normalized_fraction_key_is_scale_invariant() -> None:
    assert normalized_fraction_key("LaMnO3") == normalized_fraction_key("La2Mn2O6")
    # This decimal-rescaling case exposed the S1.5 attachment split: direct
    # reduced-formula strings differ although the nominal composition does not.
    assert normalized_fraction_key(
        "Cr0.3333I2Mn0.6666"
    ) == normalized_fraction_key("Cr0.6666I4Mn1.3332")


def test_hybrid_reference_keeps_ordered_and_partial_rules_separate() -> None:
    reference = build_formula_precedent_index(
        [("LaMnO3", False), ("Ba0.50Sr0.50TiO3", True)], tolerance=1e-6
    )
    assert formula_has_precedent("La2Mn2O6", reference)
    assert formula_has_precedent("Ba0.5000004Sr0.4999996TiO3", reference)
    assert not formula_has_precedent("Ba0.51Sr0.49TiO3", reference)
    assert not formula_has_precedent("Ca0.5Sr0.5TiO3", reference)


def test_partial_fraction_distance_is_componentwise() -> None:
    reference = build_formula_precedent_index(
        [("Li0.5Na0.5Cl", True)], tolerance=1e-3
    )
    assert formula_has_precedent("Li0.501Na0.499Cl", reference)
    assert not formula_has_precedent("Li0.51Na0.49Cl", reference)
