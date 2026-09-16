#!/usr/bin/env python3
"""Formula-reference and composition-stratum helpers. Regenerating references requires a separately licensed ICSD index; derived comparisons can use the released tables."""
from __future__ import annotations

import argparse
import csv
import json
from math import gcd
from pathlib import Path
from typing import Optional

ANIONS = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N"}


def parse_formula_pm(formula: str) -> Optional[dict[str, float]]:
    """pymatgen-based element->amount dict (handles parentheses, fractions)."""
    if not formula:
        return None
    try:
        from pymatgen.core import Composition

        comp = Composition(formula)
    except Exception:
        return None
    out = {str(el): float(amt) for el, amt in comp.get_el_amt_dict().items()}
    return out or None


def composition_class(formula: str, parser=parse_formula_pm) -> Optional[str]:
    """Coarse composition descriptor (anion class | n-elements bucket | ratio bucket).
    Byte-identical logic to scripts/analyze_composition_matched_ai.py."""
    elems = parser(formula)
    if not elems:
        return None
    cations = {e: n for e, n in elems.items() if e not in ANIONS}
    anions = {e: n for e, n in elems.items() if e in ANIONS}
    n_elem = len(elems)
    cat_total = sum(cations.values())
    ani_total = sum(anions.values())
    if not anions:
        anion_cls = "intermetallic"
    else:
        anion_dominant = max(anions, key=anions.get)
        anion_cls = {
            "O": "oxide", "S": "sulfide", "Se": "selenide", "Te": "telluride",
            "F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide",
            "N": "nitride",
        }.get(anion_dominant, "mixed-anion")
        if len(anions) > 1:
            anion_cls = "mixed-anion"
    if cat_total > 0 and anions:
        r = ani_total / cat_total
        if r < 0.6:
            ratio_bucket = "lo"
        elif r < 1.1:
            ratio_bucket = "11"
        elif r < 1.7:
            ratio_bucket = "15"
        elif r < 2.3:
            ratio_bucket = "20"
        elif r < 3.5:
            ratio_bucket = "30"
        else:
            ratio_bucket = "hi"
    elif anions:
        ratio_bucket = "all-anion"
    else:
        ratio_bucket = "no-anion"
    n_bucket = str(n_elem) if n_elem <= 4 else "5+"
    return f"{anion_cls}|n{n_bucket}|r{ratio_bucket}"


def anonymized_formula(formula: str, parser=parse_formula_pm) -> Optional[str]:
    """Anonymized stoichiometry (A2B1C4 style), byte-identical logic to
    scripts/analyze_composition_matched_ai.py."""
    elems = parser(formula)
    if not elems:
        return None
    amts = sorted(elems.values(), reverse=True)
    int_amts = [int(round(a * 100)) for a in amts]
    g = int_amts[0]
    for v in int_amts[1:]:
        g = gcd(g, v)
    if g == 0:
        return None
    reduced = [v // g for v in int_amts]
    return "".join(f"{chr(ord('A') + i)}{n}" for i, n in enumerate(reduced))


def parse_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--icsd-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[1990, 2000, 2010])
    ap.add_argument("--max-year", type=int, default=2015)
    args = ap.parse_args()

    from pymatgen.core import Composition

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, str]] = []  # (icsd_id, year, reduced_formula)
    failures: list[int] = []
    n_no_year = 0
    n_rows = 0
    with open(args.icsd_index, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            cid = parse_int(row.get("cif_names", ""))
            year = parse_int(row.get("publication_year", ""))
            if cid is None:
                continue
            if year is None:
                n_no_year += 1
                continue
            try:
                rf = Composition((row.get("name") or "").strip()).reduced_formula
            except Exception:
                failures.append(cid)
                continue
            if not rf:
                failures.append(cid)
                continue
            rows.append((cid, year, rf))

    summary: dict = {
        "n_index_rows": n_rows,
        "n_rows_no_year": n_no_year,
        "n_parse_failures": len(failures),
        "parse_failure_ids": failures,
        "n_rows_used": len(rows),
        "n_unique_reduced_formulas_allyear": len({rf for _, _, rf in rows}),
        "cutoffs": {},
    }

    for T in args.cutoffs:
        ref_all = sorted({rf for _, y, rf in rows if y <= T})
        ref_post1980 = sorted({rf for _, y, rf in rows if 1980 < y <= T})
        (out / f"icsd_reduced_formulas_year_le_{T}.txt").write_text("\n".join(ref_all) + "\n")
        (out / f"icsd_reduced_formulas_1980lt_year_le_{T}.txt").write_text("\n".join(ref_post1980) + "\n")
        set_all = set(ref_all)
        set_post = set(ref_post1980)
        n_post = 0
        with (out / f"post_cutoff_formula_match_T{T}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["icsd_id", "year", "formula_match_allyear", "formula_match_post1980",
                        "stratum_coarse", "stratum_anon"])
            for cid, y, rf in rows:
                if not (T < y <= args.max_year):
                    continue
                n_post += 1
                w.writerow([cid, y, int(rf in set_all), int(rf in set_post),
                            composition_class(rf) or "", anonymized_formula(rf) or ""])
        summary["cutoffs"][str(T)] = {
            "n_reference_formulas_year_le_T": len(ref_all),
            "n_reference_formulas_1980lt_year_le_T": len(ref_post1980),
            "n_post_cutoff_entries_T_lt_year_le_max": n_post,
        }
        print(f"T={T}: ref_all={len(ref_all)} ref_post1980={len(ref_post1980)} post-T entries={n_post}", flush=True)

    (out / "formula_reference_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"parse failures: {len(failures)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
