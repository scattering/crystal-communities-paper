from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_formula_collapsed_graph as formula_graph  # noqa: E402
import analyze_tri_structural_roles as tri_roles  # noqa: E402
import compare_tri_structural_network as tri_compare  # noqa: E402
from formula_conventions import normalized_fraction_key  # noqa: E402


TRI_LOADERS = (
    formula_graph.load_tri_existing,
    tri_compare.load_tri_existing,
    tri_roles.load_tri_existing,
)
ICSD_LOADERS = (
    formula_graph.load_icsd_index,
    tri_compare.load_icsd_index,
    tri_roles.load_icsd_index,
)


def test_tri_producers_use_normalized_nominal_formula_identity(tmp_path: Path) -> None:
    formula = "Cr0.3333I2Mn0.6666"
    doubled = "Cr0.6666I4Mn1.3332"
    expected = normalized_fraction_key(formula, decimals=12)

    tri_path = tmp_path / "tri.json"
    tri_path.write_text(json.dumps({formula: {"deg": 1, "discovery": 2000}}))
    for load_tri in TRI_LOADERS:
        loaded = load_tri(tri_path)
        assert set(loaded) == {expected}

    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cif_names", "name", "publication_year"]
        )
        writer.writeheader()
        writer.writerow({"cif_names": 1, "name": formula, "publication_year": 2001})
        writer.writerow({"cif_names": 2, "name": doubled, "publication_year": 2002})

    for load_index in ICSD_LOADERS:
        loaded = load_index(index_path)
        assert loaded[1]["formula_identity"] == expected
        assert loaded[2]["formula_identity"] == expected


def test_tri_producers_reject_identity_collisions(tmp_path: Path) -> None:
    tri_path = tmp_path / "tri.json"
    tri_path.write_text(json.dumps({"LaMnO3": {}, "La2Mn2O6": {}}))
    for load_tri in TRI_LOADERS:
        try:
            load_tri(tri_path)
        except ValueError as error:
            assert "collide under normalized identity" in str(error)
        else:
            raise AssertionError("normalized TRI identity collision was not rejected")
