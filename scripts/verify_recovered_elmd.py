#!/usr/bin/env python3
"""Verify recovered ElMD cohort outputs and the original full-release NN kernel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import set_num_threads

import compute_full_gnome_elmd as full


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    down = root / "notes/feature_repair_2026_09/downstream"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-output", type=Path, required=True)
    p.add_argument("--saved-dir", type=Path, default=down / "formula_layers")
    p.add_argument("--icsd-index", type=Path, default=down / "inputs/ICSD_index.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index-sample-size", type=int, default=250)
    p.add_argument("--threads", type=int, default=10)
    args = p.parse_args()
    set_num_threads(args.threads)
    a = pd.read_csv(args.saved_dir / "nearest_icsd_elmd_records.csv")
    b = pd.read_csv(args.cohort_output / "nearest_icsd_elmd_records.csv")
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    assert a[["source", "material_id", "reduced_formula"]].equals(b[["source", "material_id", "reduced_formula"]])
    columns = {}
    for col in a:
        if col.endswith("_nearest_elmd"):
            delta = np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))
            columns[col] = {"max_abs_error": float(delta.max()), "non_bit_identical_values": int((delta != 0).sum())}
            assert np.isfinite(delta).all() and (delta <= 1e-12).all()
        elif col not in ["source", "material_id", "reduced_formula"]:
            mismatch = int((a[col].fillna("") != b[col].fillna("")).sum())
            columns[col] = {"mismatches": mismatch}
            assert mismatch == 0
    sa = pd.read_csv(args.saved_dir / "nearest_icsd_elmd_summary.csv")
    sb = pd.read_csv(args.cohort_output / "nearest_icsd_elmd_summary.csv")
    assert sa[["source", "reference"]].equals(sb[["source", "reference"]])
    summary_error = float(np.abs(sa.select_dtypes("number").to_numpy() - sb.select_dtypes("number").to_numpy()).max())
    assert summary_error <= 1e-12

    spec = importlib.util.find_spec("ElMD")
    lookup = Path(next(iter(spec.submodule_search_locations))) / "el_lookup/mod_petti.json"
    full.LOOKUP = json.loads(lookup.read_text())
    full.ICSD_INDEX = args.icsd_index
    g = a.loc[a.source == "gnome"].reset_index(drop=True)
    seed = 20260905
    selected = np.sort(np.random.default_rng(seed).choice(len(g), args.index_sample_size, replace=False))
    q = g.iloc[selected]
    pos, mass, ptr = [], [], [0]
    for formula in q.reduced_formula:
        pp, mm = full.sparse_formula(formula)
        pos.extend(pp); mass.extend(mm); ptr.append(len(pos))
    qp, qm, qptr = np.array(pos, np.int16), np.array(mass), np.array(ptr, np.int64)
    qproj = full.make_projections(qp, qm, qptr, full.funcs())
    indexed = {}
    for label, saved_label, mask in [
        ("all_index", "all_icsd", lambda d: np.ones(len(d), bool)),
        ("post1980_through2015", "post1980_through2015_icsd", lambda d: (d.publication_year > 1980) & (d.publication_year <= 2015)),
    ]:
        ref = full.load_reference(mask)
        result, meta = full.one_ref(label, qp, qm, qptr, qproj, ref)
        brute, _ = full.brute_nearest(qp, qm, qptr, *ref[:3])
        expected = q[saved_label + "_nearest_elmd"].to_numpy(float)
        saved_error = float(np.abs(result["d"] - expected).max())
        brute_error = float(np.abs(result["d"] - brute).max())
        assert saved_error <= 1e-12 and brute_error <= 1e-12
        indexed[label] = {"query_count": len(q), "reference_input_count": meta["reference_input_n"], "reference_unique_vectors": meta["reference_unique_elmd_vectors"], "max_abs_error_vs_saved_cohort": saved_error, "max_abs_error_vs_exhaustive_kernel": brute_error}
    report = {
        "status": "PASS", "cohort_rows": len(a), "row_keys_and_order_equal": True,
        "source_counts": a.source.value_counts().to_dict(), "columns": columns,
        "summary_max_abs_error": summary_error,
        "cohort_validation": json.loads((args.cohort_output / "nearest_icsd_elmd_metadata.json").read_text())["validation"],
        "full_release_kernel_check": {"sample": "Uniform sample without replacement from the saved 5,000-row GNoME cohort", "seed": seed, "gnome_cohort_positions": selected.tolist(), "references": indexed, "scope": "Checks original indexed kernel against saved cohort distances and exhaustive distances; does not rerun all 554,054 full-release queries."},
        "input_hashes": {str(path): sha(path) for path in [args.saved_dir / "nearest_icsd_elmd_records.csv", args.saved_dir / "nearest_icsd_elmd_summary.csv", args.icsd_index, lookup, Path(__file__), Path(full.__file__), root / "scripts/compute_nearest_icsd_elmd.py"]},
        "runtime": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "numba", "pymatgen", "ElMD"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "cohort_rows": len(a), "summary_max_abs_error": summary_error, "indexed_kernel": indexed}, indent=2))


if __name__ == "__main__":
    main()
