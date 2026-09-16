#!/usr/bin/env python3
"""Composition keys used by the historical-precedent analyses.

Formula strings are display values, not identifiers: equivalent compositions
can be written with different formula-unit scales or element orders.  These
helpers keep the comparison independent of those choices while retaining the
original string for tables and figures.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from typing import Iterable

import numpy as np
from pymatgen.core import Composition
from scipy.spatial import cKDTree


def _integer_amounts(formula: str) -> dict[str, int]:
    """Return the smallest rationalized integer amounts for ``formula``."""
    # Normalize before rationalization.  Calling the pymatgen method directly
    # on decimal amounts can choose different integer approximants after a
    # harmless whole-formula rescaling (for example, 0.3333:0.6666:2 versus
    # 0.6666:1.3332:4).  Fractional composition makes the documented integer
    # conversion independent of that display scale.
    parsed = Composition(str(formula).strip())
    # Collapse oxidation states and isotope labels to elemental amounts before
    # normalization.  In particular, pymatgen represents H and D as distinct
    # species with the same element symbol; applying ``fractional_composition``
    # before this collapse can count their combined fraction twice.
    composition = Composition(parsed.get_el_amt_dict()).fractional_composition
    integer_formula, _ = composition.get_integer_formula_and_factor()
    integer = Composition(integer_formula)
    amounts = {
        str(element): int(round(float(amount)))
        for element, amount in integer.get_el_amt_dict().items()
        if abs(float(amount)) > 0
    }
    divisor = 0
    for amount in amounts.values():
        divisor = gcd(divisor, abs(amount))
    if divisor == 0:
        raise ValueError(f"Formula has no nonzero amounts: {formula!r}")
    return {element: amount // divisor for element, amount in amounts.items()}


def scale_invariant_formula_key(formula: str) -> tuple[tuple[str, int], ...]:
    """Smallest normalized integer-ratio composition, sorted by element symbol."""
    return tuple(sorted(_integer_amounts(formula).items()))


def element_set_key(formula: str) -> tuple[str, ...]:
    """Alphabetically sorted set of elements in a composition."""
    return tuple(element for element, _ in scale_invariant_formula_key(formula))


def anonymous_stoichiometry_key(formula: str) -> tuple[int, ...]:
    """Element-free reduced coefficient pattern in OPTIMADE order.

    Coefficients are sorted from largest to smallest.  For example,
    ``MgAl2O4`` maps to ``(4, 2, 1)``, corresponding to ``A4B2C``.
    """
    return tuple(sorted(_integer_amounts(formula).values(), reverse=True))


def normalized_fraction_vector(formula: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Return alphabetical elements and their normalized atomic fractions."""
    amounts = Composition(str(formula).strip()).get_el_amt_dict()
    elements = tuple(sorted(str(element) for element in amounts))
    values = np.asarray([float(amounts[element]) for element in elements], dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Formula has an invalid total amount: {formula!r}")
    return elements, values / total


def normalized_fraction_key(
    formula: str, *, decimals: int = 12
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Scale-invariant identity key for grouping nominal compositions."""
    elements, fractions = normalized_fraction_vector(formula)
    return elements, tuple(float(value) for value in np.round(fractions, decimals))


@dataclass(frozen=True)
class FormulaPrecedentIndex:
    """Separate ordered and partial-occupancy ICSD composition references."""

    full_integer_keys: frozenset[tuple[tuple[str, int], ...]]
    partial_fraction_trees: dict[tuple[str, ...], cKDTree]
    tolerance: float


def build_formula_precedent_index(
    records: Iterable[tuple[str, bool]], *, tolerance: float = 1e-6
) -> FormulaPrecedentIndex:
    """Build the declared hybrid formula reference.

    Fully occupied records use exact reduced integer ratios. Partial-occupancy
    records use normalized atomic fractions within Chebyshev distance
    ``tolerance``, always requiring the same element set.
    """
    full_keys: set[tuple[tuple[str, int], ...]] = set()
    partial: dict[tuple[str, ...], set[tuple[float, ...]]] = defaultdict(set)
    for formula, is_partial in records:
        if not str(formula).strip():
            continue
        if is_partial:
            elements, fractions = normalized_fraction_vector(formula)
            partial[elements].add(tuple(float(value) for value in fractions))
        else:
            full_keys.add(scale_invariant_formula_key(formula))
    trees = {
        elements: cKDTree(np.asarray(sorted(vectors), dtype=float))
        for elements, vectors in partial.items()
    }
    return FormulaPrecedentIndex(frozenset(full_keys), trees, float(tolerance))


def formula_has_precedent(formula: str, reference: FormulaPrecedentIndex) -> bool:
    """Return whether ``formula`` matches either declared reference layer."""
    if not str(formula).strip():
        return False
    if scale_invariant_formula_key(formula) in reference.full_integer_keys:
        return True
    elements, fractions = normalized_fraction_vector(formula)
    tree = reference.partial_fraction_trees.get(elements)
    if tree is None:
        return False
    distance, _ = tree.query(fractions, k=1, p=np.inf)
    return bool(float(distance) <= reference.tolerance + 1e-15)
