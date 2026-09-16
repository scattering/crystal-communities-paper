from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_conventions import normalized_fraction_key  # noqa: E402
from regenerate_repaired_community_evidence import first_reports  # noqa: E402


def test_s1p5_first_reports_collapse_decimal_formula_rescaling() -> None:
    formulas = {
        1: "Cr0.3333I2Mn0.6666",
        2: "Cr0.6666I4Mn1.3332",
    }
    meta = {
        iid: {
            "reduced_formula": formula,
            "nominal_formula_identity": normalized_fraction_key(formula),
        }
        for iid, formula in formulas.items()
    }
    records = [(1, 2011, 4), (2, 2012, 5)]

    legacy, legacy_tables = first_reports(records, meta)
    corrected, corrected_tables = first_reports(
        records, meta, identity_field="nominal_formula_identity"
    )

    assert len(legacy_tables[1980]) == 2
    assert len(corrected_tables[1980]) == 1
    assert set(legacy) == {1, 2}
    assert set(corrected) == {1}
