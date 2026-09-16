#!/usr/bin/env python3
"""Heuristic family-name classifier for ICSD structural communities.

For each community in `functional_community_representatives_top20.csv`,
examine its top-20 centroid-nearest members, derive a (dominant space
group, dominant stoichiometry class) signature from the member set,
and look that signature up in a small static dictionary of textbook
prototype families.

The classifier is deliberately conservative:
  - it only emits a family name when both the SG and the stoich
    class are dominant (configurable thresholds),
  - it leaves the rest unlabelled rather than guessing,
  - it records evidence and confidence so the output can be reviewed
    by a chemist before it's promoted into canonical_family_names.

This is a coverage-extension tool, not a replacement for chemist
curation. It catches clean textbook families (rocksalt, perovskite,
spinel, A15, Heusler, ...) automatically; everything else stays
empty in the output and waits for human attention.

Output: a CSV with one row per community
    community,size,dominant_sg,sg_share,stoich_class,class_share,
    inferred_family,confidence,evidence
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Optional

# Anionic species. A site is treated as a cation if its symbol is not in
# this set. H is excluded because it sometimes plays cation, sometimes
# anion (hydrides vs water of hydration); treat as cation by default.
ANIONS: set[str] = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}


def parse_formula(formula: str) -> Optional[dict[str, float]]:
    """Parse a (possibly non-integer) chemical formula string into an element
    count dict. Avoid pulling in pymatgen so this script remains stdlib-only.

    Handles forms like "Cu1.5Se2Y0.84", "Ag1Li2Pb1", "Bi0.3Fe0.5Mn0.5Nd0.7O3".
    """
    import re

    if not formula:
        return None
    # Strip any leading/trailing whitespace and parens-only wrappers
    f = formula.strip()
    if not f:
        return None
    out: dict[str, float] = {}
    pattern = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]+)?")
    pos = 0
    while pos < len(f):
        m = pattern.match(f, pos)
        if not m:
            return None
        el = m.group(1)
        amt_str = m.group(2)
        amt = float(amt_str) if amt_str else 1.0
        out[el] = out.get(el, 0.0) + amt
        pos = m.end()
        if pos == m.start():  # no progress, malformed
            return None
    return out if out else None


def stoich_class(formula: str) -> Optional[str]:
    """Reduce a formula to a coarse stoichiometry class label.

    Returns one of:
        elemental, AB_int, A3B_int, A5B_int, A2B_int,
        AX, AX2, A2X3, A2X, AX3,
        A2BX_int, AB2X_int,
        ABX2, ABX3, AB2X3, AB2X4, A2BX4,
        quaternary_oxide_etc, complex
    or None if parsing fails.
    """
    elems = parse_formula(formula)
    if not elems:
        return None
    cations = {e: n for e, n in elems.items() if e not in ANIONS}
    anions = {e: n for e, n in elems.items() if e in ANIONS}
    n_cat = len(cations)
    n_ani = len(anions)
    cat_amts = sorted(cations.values(), reverse=True)
    ani_amts = sorted(anions.values(), reverse=True)
    eps = 0.18

    if n_cat == 1 and n_ani == 0:
        return "elemental"

    # Pure intermetallic (no anion species)
    if n_ani == 0:
        if n_cat == 2:
            r = cat_amts[0] / cat_amts[1]
            if abs(r - 1) < eps: return "AB_int"
            if abs(r - 3) < 3 * eps: return "A3B_int"
            if abs(r - 5) < 5 * eps: return "A5B_int"
            if abs(r - 2) < 2 * eps: return "A2B_int"
            return "AB_int_skewed"
        if n_cat >= 3:
            return "ternary_intermetallic"

    # Binary anion-bearing
    if n_cat == 1 and n_ani == 1:
        x = ani_amts[0]
        a = cat_amts[0]
        r = x / a
        if abs(r - 1) < eps: return "AX"
        if abs(r - 2) < 2 * eps: return "AX2"
        if abs(r - 1.5) < eps: return "A2X3"
        if abs(r - 3) < 3 * eps: return "AX3"
        if abs(r - 0.5) < eps: return "A2X"
        return "AX_skewed"

    # Ternary with one anion
    if n_cat == 2 and n_ani == 1:
        x = ani_amts[0]
        a, b = cat_amts[0], cat_amts[1]
        cat_total = a + b
        # Heusler-like if no anion species (handled above) or when "X" is
        # actually a metalloid sitting in the anion set (Si, Ge, Sn, Sb,
        # Bi). We keep those as cations here; pure-anion ternaries below.
        if abs(a - b) < 0.25 * cat_total and abs(x - 3 * cat_total / 2) < 0.4 * cat_total:
            return "ABX3"
        if abs(a - 2 * b) < 0.4 * cat_total or abs(b - 2 * a) < 0.4 * cat_total:
            if abs(x - 4 * (a + b) / 3) < 0.5 * cat_total:
                return "AB2X4"
        if abs(x - cat_total / 2) < 0.4 * cat_total:
            return "A2BX_int"
        if abs(x - 2 * cat_total) < 0.5 * cat_total:
            return "AB2X4"  # split bucket
        if abs(a - b) < 0.25 * cat_total and abs(x - cat_total) < 0.3 * cat_total:
            return "ABX2"
        return "ternary_anion_other"

    # Ternary intermetallic with one metalloid in anion set (rare)
    if n_cat >= 3 and n_ani == 0:
        return "ternary_intermetallic"

    # Multi-cation + 1 anion = doped perovskite / spinel / RP / etc.
    # Test AB2X4 (x/cat ≈ 4/3) BEFORE ABX3 (x/cat ≈ 3/2) and use tighter
    # windows so MgZn(FeO2)4 (8 O / 6 cations = 1.33) is not misclassified
    # as a perovskite. Doped perovskites still keep cat_total ≈ 2 → 3 O,
    # so they land at r ≈ 1.5 and stay in the ABX3 bucket.
    if n_cat >= 3 and n_ani == 1:
        x = ani_amts[0]
        cat_total = sum(cat_amts)
        r = x / cat_total if cat_total else 0.0
        if 1.20 < r < 1.45:
            return "AB2X4"
        if 1.45 <= r < 1.65:
            return "ABX3"
        if 1.85 < r < 2.15:
            return "AX2"
        if 2.30 < r < 2.70:
            return "AB2X5_RP_n2"  # n=2 Ruddlesden-Popper-like
        return "doped_anion_oxide"

    # Multi-cation + multi-anion (mixed-anion oxides, oxy-halides, ...)
    return "complex"


# (space_group, stoich_class) -> (family_name, confidence_when_dominant)
# The confidence levels reflect how diagnostic the (SG, stoich) signature is
# for the textbook family. A high-confidence pair means the SG + stoich
# alone identifies the family unambiguously; medium means the signature
# is strong but distortions / superstructures could be misclassified.
FAMILIES: dict[tuple[int, str], tuple[str, str]] = {
    # rocksalt / zincblende / wurtzite binary AX
    (225, "AX"): ("rocksalt (NaCl-type)", "high"),
    (216, "AX"): ("zincblende (ZnS-type)", "high"),
    (186, "AX"): ("wurtzite (ZnS-type)", "high"),
    (194, "AX"): ("NiAs-type", "medium"),
    # AX2 binaries
    (225, "AX2"): ("fluorite (CaF2-type)", "high"),
    (227, "AX2"): ("anti-fluorite / fluorite-related", "medium"),
    (164, "AX2"): ("CdI2 / 1T-MX2 layered", "high"),
    (194, "AX2"): ("AlB2-type / 2H-MX2", "medium"),
    (136, "AX2"): ("rutile (TiO2-type)", "high"),
    (141, "AX2"): ("anatase (TiO2-type)", "high"),
    # A2X3 binaries
    (167, "A2X3"): ("corundum (Al2O3-type)", "high"),
    # AX3 binaries
    (221, "AX3"): ("ReO3-type", "medium"),
    # ABX3 perovskites
    (221, "ABX3"): ("cubic perovskite (Pm-3m)", "high"),
    (62, "ABX3"): ("GdFeO3-tilted perovskite (Pnma)", "high"),
    (167, "ABX3"): ("LaAlO3-tilted perovskite (R-3c)", "high"),
    (140, "ABX3"): ("tetragonal perovskite (I4/mcm)", "medium"),
    (127, "ABX3"): ("tetragonal perovskite (P4/mbm)", "medium"),
    # AB2X4 spinels and stannites
    (227, "AB2X4"): ("spinel (MgAl2O4-type)", "high"),
    (216, "AB2X4"): ("stannite / chalcopyrite-related", "medium"),
    (122, "AB2X4"): ("chalcopyrite (CuFeS2-type)", "high"),
    # A2BX4 / Ruddlesden-Popper
    (139, "A2BX4"): ("K2NiF4-type (Ruddlesden-Popper n=1)", "high"),
    (139, "AB2X4"): ("K2NiF4-type (Ruddlesden-Popper n=1)", "medium"),
    # Heuslers (the ANIONS set keeps Si/Ge/Sn/As/Sb out of "anion" so
    # half/full Heuslers fall into the pure-intermetallic bucket)
    (225, "A2BX_int"): ("full Heusler (L21)", "high"),
    (216, "A2BX_int"): ("half Heusler (C1b)", "high"),
    (225, "AB_int"): ("CsCl / B2 ordered (Pm-3m via Fm-3m supercell)", "low"),
    (221, "AB_int"): ("CsCl-type (B2)", "high"),
    (229, "AB_int"): ("BCC / W-type ordered solid solution", "high"),
    (229, "AB_int_skewed"): ("BCC / W-type (off-stoichiometry)", "medium"),
    (229, "elemental"): ("BCC elemental", "high"),
    (225, "elemental"): ("FCC elemental", "high"),
    # A15 and CaCu5
    (223, "A3B_int"): ("A15 (Cr3Si / Nb3Sn-type)", "high"),
    (191, "A5B_int"): ("CaCu5-type", "high"),
    # Laves
    (227, "AB_int"): ("Laves C15 (MgCu2-type)", "high"),
    (194, "AB_int"): ("Laves C14 (MgZn2-type)", "high"),
    (194, "AB_int_skewed"): ("Laves C14-related (MgZn2-derived)", "medium"),
    # γ-brass / α-Mn cubic complex intermetallics
    (217, "AB_int"): ("γ-brass / I-43m intermetallic", "high"),
    (220, "AB_int"): ("α-Mn / I-43d cubic intermetallic", "high"),
    (217, "AB_int_skewed"): ("γ-brass-derived intermetallic", "medium"),
    (220, "AB_int_skewed"): ("α-Mn-derived intermetallic", "medium"),
    # ABX2 layered chalcogenides / oxides
    (164, "ABX2"): ("CdI2-derived ABX2 layered", "medium"),
    (194, "ABX2"): ("MgZn2-derived ABX2", "medium"),
    (12, "ABX2"): ("monoclinic ABX2 layered (C2/m)", "medium"),
    # Pyrochlores
    (227, "doped_anion_oxide"): ("pyrochlore (A2B2O7-related)", "medium"),
    (227, "complex"): ("pyrochlore-like cubic mixed-anion (Fd-3m)", "low"),

    # ----- Coverage extensions: distorted perovskite SGs that were missing
    # from the first pass. Each labels the same family with confidence
    # decreasing as the symmetry strays further from the cubic ideal.
    (14, "ABX3"): ("monoclinic distorted perovskite (P21/c)", "medium"),
    (11, "ABX3"): ("monoclinic distorted perovskite (P21/m)", "medium"),
    (33, "ABX3"): ("orthorhombic distorted perovskite (Pna21)", "medium"),
    (70, "ABX3"): ("Fddd-tilted perovskite", "low"),
    (139, "ABX3"): ("tetragonal perovskite / Ruddlesden-Popper boundary (I4/mmm)", "medium"),
    (148, "ABX3"): ("R-3 trigonal perovskite-derived", "low"),
    (160, "ABX3"): ("R3m trigonal perovskite (LiNbO3-related)", "medium"),
    (161, "ABX3"): ("R3c trigonal perovskite", "medium"),
    (12, "ABX3"): ("monoclinic distorted perovskite (C2/m)", "medium"),
    (15, "ABX3"): ("monoclinic distorted perovskite (C2/c)", "medium"),
    (2, "ABX3"): ("triclinic distorted perovskite (P-1)", "low"),

    # AB2X4 spinel-derived family at off-cubic SGs
    (141, "AB2X4"): ("tetragonally-distorted spinel (I41/amd)", "medium"),
    (139, "AB2X4"): ("tetragonal AB2X4 (I4/mmm; spinel- or hausmannite-derived)", "medium"),
    (122, "AB2X4"): ("chalcopyrite-related AB2X4 (I-42d)", "medium"),

    # Ternary intermetallic SGs that map onto well-known families. These are
    # all medium/low because the ternary_intermetallic bucket is broad.
    (225, "ternary_intermetallic"): ("full-Heusler-like (Fm-3m, ternary)", "medium"),
    (216, "ternary_intermetallic"): ("half-Heusler-like (F-43m, ternary)", "medium"),
    (223, "ternary_intermetallic"): ("A15 / Cr3Si-type (Pm-3n, ternary)", "medium"),
    (191, "ternary_intermetallic"): ("CaCu5-type / hexagonal RE-intermetallic (P6/mmm)", "medium"),
    (194, "ternary_intermetallic"): ("Laves C14-type or hexagonal intermetallic (P63/mmc)", "low"),
    (227, "ternary_intermetallic"): ("Laves C15-type or cubic intermetallic (Fd-3m)", "low"),
    (139, "ternary_intermetallic"): ("σ-phase or tetragonal intermetallic (I4/mmm)", "low"),
    (141, "ternary_intermetallic"): ("tetragonal intermetallic (I41/amd; ZrSiS-type)", "low"),
    (217, "ternary_intermetallic"): ("γ-brass / I-43m ternary intermetallic", "medium"),
    (220, "ternary_intermetallic"): ("α-Mn / I-43d ternary intermetallic", "medium"),
    (229, "ternary_intermetallic"): ("BCC ternary solid solution (Im-3m)", "medium"),
    (221, "ternary_intermetallic"): ("CsCl-type ternary (Pm-3m)", "medium"),

    # AB2X5 / Ruddlesden-Popper n=2
    (139, "AB2X5_RP_n2"): ("Ruddlesden-Popper n=2 (A3B2O7-related)", "medium"),
    (62, "AB2X5_RP_n2"): ("orthorhombic A3B2O7 RP n=2", "low"),

    # ----- Coverage extension v2: textbook families that still sat in the
    # unlabelled tail of the first heuristic pass.

    # Olivine and orthorhombic AB2X4 family. SG 62 + AB2X4 is the canonical
    # olivine signature (Mg2SiO4 / (Fe,Mg)2SiO4 / LiFePO4 cathode chemistry).
    (62, "AB2X4"): ("olivine-related (Pnma AB2X4; Mg2SiO4 / LiFePO4 cathode chemistry)", "high"),
    (14, "AB2X4"): ("monoclinic AB2X4 (P21/c; olivine-derived)", "medium"),
    (12, "AB2X4"): ("monoclinic AB2X4 (C2/m; layered olivine-derived)", "low"),
    (140, "AB2X4"): ("tetragonal AB2X4 (I4/mcm; spinel-derived)", "medium"),

    # Garnet — SG 230 (Ia-3d) cubic A3B5O12 / A3B2C3O12.
    (230, "ABX3"): ("garnet (Ia-3d A3B2C3O12 / A3B5O12-related)", "high"),
    (230, "complex"): ("garnet-related (Ia-3d cubic mixed-anion)", "medium"),
    (230, "doped_anion_oxide"): ("garnet (Ia-3d A3B2C3O12-related, doped)", "medium"),

    # Tetragonal ferroelectric perovskite (BaTiO3 / PbTiO3 / KNbO3 P4mm).
    (99, "ABX3"): ("tetragonal polar perovskite (P4mm; BaTiO3 / PbTiO3 ferroelectric family)", "high"),
    (38, "ABX3"): ("orthorhombic ferroelectric perovskite (Amm2; BaTiO3 low-T)", "medium"),

    # Iron-pnictide superconductor parent families. SG 129 (P4/nmm) is the
    # ZrCuSiAs / 1111 family (LaFeAsO etc.); SG 123 (P4/mmm) AB2X4 is the
    # ThCr2Si2 / 122 family (BaFe2As2 etc.).
    (129, "ternary_intermetallic"): ("1111-type tetragonal layered (P4/nmm; ZrCuSiAs / LaFeAsO family)", "high"),
    (129, "complex"): ("1111-type tetragonal layered mixed-anion (P4/nmm; LaFeAsO-related)", "medium"),
    (129, "ABX3"): ("1111-type tetragonal layered (P4/nmm)", "medium"),
    (123, "AB2X4"): ("122-type layered (P4/mmm; ThCr2Si2 / BaFe2As2 superconductor parent)", "high"),
    (123, "ternary_intermetallic"): ("122-type layered intermetallic (P4/mmm; ThCr2Si2)", "high"),
    (139, "AB_int"): ("122-type tetragonal (I4/mmm; CsCl-derived 122 layered intermetallic)", "low"),

    # Delafossite — SG 166 R-3m + ABX2 (CuFeO2 / CuAlO2 oxide thermoelectrics
    # and transparent conductors).
    (166, "ABX2"): ("delafossite (R-3m ABX2; CuFeO2 / CuAlO2 family)", "high"),
    (166, "ternary_intermetallic"): ("R-3m rhombohedral intermetallic (Bi2Te3 / tetradymite-related)", "medium"),
    (166, "AB_int_skewed"): ("R-3m rhombohedral intermetallic", "medium"),
    (164, "ABX2"): ("CdI2-type ABX2 (P-3m1 layered)", "medium"),

    # Cu3Au / L12 ordered cubic intermetallic.
    (221, "A3B_int"): ("Cu3Au / L12 ordered cubic (Pm-3m AB3)", "high"),

    # Heusler-related Fd-3m + A2B intermetallic.
    (227, "A2B_int"): ("Heusler-related cubic intermetallic (Fd-3m A2B)", "medium"),

    # Hexagonal perovskite (BaNiO3-type, P63/mmc).
    (194, "ABX3"): ("hexagonal perovskite (P63/mmc; BaNiO3-type)", "medium"),
    (194, "A2B_int"): ("MgZn2 Laves C14 / hexagonal A2B intermetallic", "medium"),

    # ZrNiAl / Fe2P-type hexagonal intermetallics.
    (189, "ternary_intermetallic"): ("ZrNiAl / Fe2P-type hexagonal (P-62m)", "high"),
    (189, "AB_int_skewed"): ("Fe2P-derived hexagonal intermetallic (P-62m)", "medium"),

    # Tetragonal P42/mnm intermetallic.
    (136, "ternary_intermetallic"): ("P42/mnm tetragonal intermetallic", "low"),

    # Pmmm cuprate-related orthorhombic mixed-anion oxide
    # (YBa2Cu3O7 / La2CuO4 family).
    (47, "doped_anion_oxide"): ("Pmmm orthorhombic cuprate-related oxide (YBa2Cu3O7 family)", "medium"),
    (47, "complex"): ("Pmmm orthorhombic mixed-anion (YBa2Cu3O7 / cuprate)", "medium"),
    (65, "complex"): ("Cmmm orthorhombic mixed-anion (cuprate-derived)", "low"),

    # P63/mcm hexagonal A2B intermetallic
    (193, "A2B_int"): ("P63/mcm hexagonal A2B intermetallic (FeB-type derivative)", "medium"),
    (193, "ternary_intermetallic"): ("P63/mcm hexagonal intermetallic", "low"),

    # Pnma intermetallic (CrB-type, MnP-type, etc.).
    (62, "ternary_intermetallic"): ("Pnma orthorhombic intermetallic (CrB / MnP-derived)", "medium"),
    (62, "AB_int"): ("Pnma orthorhombic AB intermetallic", "medium"),

    # P63/m apatite-related (Ca5(PO4)3OH and related).
    (176, "complex"): ("apatite-related hexagonal (P63/m; Ca5(PO4)3X-type)", "medium"),
    (176, "doped_anion_oxide"): ("apatite-related hexagonal (P63/m; Ca5(PO4)3X-type)", "medium"),

    # P-43n cubic complex (rare, Cr3Si-related extension).
    (218, "complex"): ("P-43n cubic complex (A15-related supercell)", "low"),

    # Low-symmetry catch-alls. Marked low — these only fire when the SG +
    # stoich-class signature is genuinely dominant in the member set.
    (14, "complex"): ("P21/c monoclinic mixed-anion oxide", "low"),
    (14, "doped_anion_oxide"): ("P21/c monoclinic mixed-cation oxide", "low"),
    (62, "complex"): ("Pnma orthorhombic mixed-anion oxide", "low"),
    (62, "doped_anion_oxide"): ("Pnma orthorhombic mixed-cation oxide", "low"),
    (15, "complex"): ("C2/c monoclinic mixed-anion oxide", "low"),
    (12, "complex"): ("C2/m monoclinic mixed-anion oxide", "low"),
    (2, "complex"): ("P-1 triclinic mixed-anion oxide", "low"),
    (2, "doped_anion_oxide"): ("P-1 triclinic mixed-cation oxide", "low"),
    (148, "complex"): ("R-3 trigonal mixed-anion oxide", "low"),

    # Frequently observed AB2X4 in low-symmetry settings
    (15, "AB2X4"): ("C2/c monoclinic AB2X4 (olivine-derived)", "low"),

    # Other helpful ABX2 settings
    (225, "ABX2"): ("rocksalt-derived ordered ABX2 (Fm-3m; elpasolite-related)", "medium"),
    (216, "ABX2"): ("zincblende-derived ordered ABX2 (F-43m)", "medium"),

    # Coverage extension v3: clean (SG, stoich) pairs from the top-50 tail.
    (221, "AB2X4"): ("cubic AB2X4 (Pm-3m; anti-fluorite-derived ternary)", "medium"),
    (217, "AB2X4"): ("I-43m cubic AB2X4 (γ-brass-derived ordered Heusler)", "medium"),
    (220, "AB2X4"): ("I-43d cubic AB2X4 (α-Mn-derived ordered)", "low"),
    (139, "AB2X3"): ("I4/mmm tetragonal layered AB2X3 (LiCoO2 / α-NaFeO2-related)", "medium"),
    (148, "ABX2"): ("R-3 trigonal layered ABX2 (LiCoO2-related)", "medium"),

    # Distorted-perovskite SG 167 + ABX2 (RE-intermetallic Cd2Cu / Mg-doped)
    (167, "AB_int"): ("R-3c rhombohedral intermetallic", "low"),
    (167, "AB_int_skewed"): ("R-3c rhombohedral intermetallic (off-stoich)", "low"),

    # Catch-all for SG 225 + heterogeneous oxide stuff (typically rocksalt-
    # derived superstructures or anti-fluorite-related).
    (225, "doped_anion_oxide"): ("rocksalt-derived cubic mixed-cation oxide (Fm-3m)", "low"),

    # P-42m / I-42m / P-43m heterogeneous cubic mixed-anion. Mark very low —
    # these are honest catch-alls.
    (113, "complex"): ("P-42m tetragonal mixed-anion complex", "low"),
    (215, "ABX3"): ("P-43m cubic ABX3 (anti-perovskite-related)", "medium"),

    # Beta-tridymite / cristobalite / SiO2 polymorphs (SG 152 P3121, SG 76 P31).
    (152, "AX2"): ("α-quartz / SiO2-related (P3121)", "medium"),
    (154, "AX2"): ("α-quartz right (P3221)", "medium"),

    # P63 hexagonal - several common families
    (173, "ABX3"): ("P63 hexagonal perovskite-derived", "low"),
    (185, "ABX3"): ("P63cm hexagonal perovskite (YMnO3-type)", "medium"),
    (186, "ABX3"): ("P63mc hexagonal perovskite-related (LiNbO3 / LuMnO3)", "medium"),
}


def classify_community(reps: list[dict[str, str]], sg_threshold: float = 0.4, class_threshold: float = 0.4) -> dict[str, object]:
    """Return a dict with dominant SG, dominant stoich class, inferred family,
    confidence, and supporting evidence for one community."""
    sg_counts: Counter[int] = Counter()
    class_counts: Counter[str] = Counter()
    total = 0
    for r in reps:
        try:
            sg = int(r["sym_group"])
        except (KeyError, TypeError, ValueError):
            sg = -1
        cls = stoich_class(r.get("name", "")) or "unparsed"
        if sg > 0:
            sg_counts[sg] += 1
        class_counts[cls] += 1
        total += 1

    if total == 0:
        return {"inferred_family": "", "confidence": "", "evidence": "no representatives"}

    dom_sg, dom_sg_count = sg_counts.most_common(1)[0] if sg_counts else (-1, 0)
    dom_cls, dom_cls_count = class_counts.most_common(1)[0] if class_counts else ("", 0)
    sg_share = dom_sg_count / total if total else 0.0
    class_share = dom_cls_count / total if total else 0.0

    family = ""
    confidence = ""
    if sg_share >= sg_threshold and class_share >= class_threshold:
        match = FAMILIES.get((dom_sg, dom_cls))
        if match is not None:
            family, confidence = match

    evidence_bits = [
        f"top-20 reps: dominant SG {dom_sg} ({dom_sg_count}/{total} = {sg_share:.0%}), "
        f"dominant stoichiometry class {dom_cls} ({dom_cls_count}/{total} = {class_share:.0%})",
    ]
    sg_top3 = sg_counts.most_common(3)
    cls_top3 = class_counts.most_common(3)
    evidence_bits.append(f"SG breakdown: {sg_top3}")
    evidence_bits.append(f"class breakdown: {cls_top3}")
    return {
        "dominant_sg": dom_sg,
        "sg_share": round(sg_share, 3),
        "stoich_class": dom_cls,
        "class_share": round(class_share, 3),
        "inferred_family": family,
        "confidence": confidence,
        "evidence": "; ".join(evidence_bits),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps-csv", required=True, help="functional_community_representatives_top20.csv")
    p.add_argument("--output", required=True, help="destination CSV")
    p.add_argument("--sg-threshold", type=float, default=0.4)
    p.add_argument("--class-threshold", type=float, default=0.4)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    reps_by_comm: dict[int, list[dict[str, str]]] = {}
    with Path(args.reps_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                c = int(row["community"])
            except (KeyError, TypeError, ValueError):
                continue
            reps_by_comm.setdefault(c, []).append(row)

    rows = []
    n_labelled = 0
    family_hist: Counter[str] = Counter()
    for c, reps in sorted(reps_by_comm.items()):
        result = classify_community(reps, args.sg_threshold, args.class_threshold)
        size = int(reps[0].get("community_size", "0") or 0)
        birth = int(reps[0].get("community_birth_year", "0") or 0)
        family = str(result["inferred_family"])
        if family:
            n_labelled += 1
        family_hist[family or "(unlabelled)"] += 1
        rows.append({
            "community": c,
            "size": size,
            "birth_year": birth,
            "dominant_sg": result["dominant_sg"],
            "sg_share": result["sg_share"],
            "stoich_class": result["stoich_class"],
            "class_share": result["class_share"],
            "inferred_family": family,
            "confidence": result["confidence"],
            "evidence": result["evidence"],
        })

    # sort output by community size descending so the "important" communities
    # are at the top of the CSV
    rows.sort(key=lambda r: -r["size"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "community", "size", "birth_year",
                "dominant_sg", "sg_share", "stoich_class", "class_share",
                "inferred_family", "confidence", "evidence",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {out}")
    print(f"  total communities: {len(rows)}")
    print(f"  labelled: {n_labelled} ({n_labelled / len(rows):.1%})")
    print(f"  top families:")
    for fam, n in family_hist.most_common(15):
        print(f"    {n:>4}  {fam}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
