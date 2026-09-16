#!/usr/bin/env python3
"""Independent local checks of the retained factor-ablation outputs.

Reads only files retained in this directory. Recomputes every in-basin flag from
distance <= p95, recomputes own-support and exact common-support rates from the
row-level projections, compares them with results/factor_ablations.csv and with the
saved five-representation table for the two parents, and checks identifier
uniqueness and common-support containment. Writes verification/local_verification.json.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = next(p for p in HERE.parents if (p / ".git").exists())
FIVE = REPO / "notes/feature_repair_2026_09/downstream/representations/amd-external-full-map/five_representation_common_support.csv"
SOURCES = ("gnome", "mattergen", "mp", "jarvis", "alexandria")
LABELS = {"gnome": "GNoME", "mattergen": "MatterGen", "mp": "MP", "jarvis": "JARVIS", "alexandria": "Alexandria"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def flags_from(table):
    keys = [r["record_key"] for r in table]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate record keys")
    saved = np.asarray([r["in_basin"] == "True" for r in table])
    recomputed = np.asarray([float(r["nearest_centroid_distance"]) <= float(r["community_threshold_p95"]) for r in table])
    if not np.array_equal(saved, recomputed):
        raise ValueError("saved flags do not reproduce distance <= p95")
    return dict(zip(keys, saved.tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=HERE / "verification/local_verification.json")
    args = parser.parse_args()
    results = args.results.resolve()
    common = json.loads((HERE / "inputs/common_support_ids.json").read_text())["populations"]
    reported = {(r["ablation"], r["population"], r["support"]): r for r in rows(results / "factor_ablations.csv")}
    five = {(r["representation"], r["population"]): r for r in rows(FIVE)}
    report = {"inputs_sha256": {}, "maps": {}, "checks_passed": 0, "checks_failed": []}
    ablations = [d.name for d in sorted((results / "maps").iterdir()) if d.is_dir()]
    for name in ablations:
        directory = results / "maps" / name
        summary = json.loads((directory / "map_summary.json").read_text())
        entry = {"n_rows_checked": 0}
        populations = {"ICSD": directory / "icsd_full.csv.gz", **{LABELS[s]: directory / "external" / s / "projection_full.csv" for s in SOURCES}}
        icsd_rate = {}
        for population, path in populations.items():
            report["inputs_sha256"][str(path.relative_to(HERE))] = sha(path)
            flags = flags_from(rows(path))
            entry["n_rows_checked"] += len(flags)
            common_ids = common[population]["ids"]
            missing = [k for k in common_ids if k not in flags]
            if missing:
                report["checks_failed"].append(f"{name}/{population}: {len(missing)} common identifiers absent")
                continue
            own = float(np.mean(list(flags.values())))
            com = float(np.mean([flags[k] for k in common_ids]))
            for support, value, n in (("own", own, len(flags)), ("common", com, len(common_ids))):
                r = reported[(name, population, support)]
                ok = int(r["n"]) == n and abs(float(r["rate"]) - value) < 1e-12 and int(r["n_in_basin"]) == round(value * n)
                if ok:
                    report["checks_passed"] += 1
                else:
                    report["checks_failed"].append(f"{name}/{population}/{support}: reported {r['rate']} n={r['n']} vs recomputed {value} n={n}")
                if population == "ICSD":
                    icsd_rate[support] = value
                elif r["icsd_minus_source_pp"] != "":
                    gap = 100 * (icsd_rate[support] - value)
                    if abs(float(r["icsd_minus_source_pp"]) - gap) < 1e-9:
                        report["checks_passed"] += 1
                    else:
                        report["checks_failed"].append(f"{name}/{population}/{support}: gap {r['icsd_minus_source_pp']} vs {gap}")
            if name in ("parent_crystalweave", "parent_graphlet"):
                rep = "CrystalWeave" if name == "parent_crystalweave" else "CrystalNN graphlets"
                saved = five[(rep, population)]
                entry.setdefault("saved_five_way_comparison", {})[population] = {
                    "saved_common_rate": float(saved["in_basin_fraction"]), "recomputed_common_rate": com,
                    "difference_pp": 100 * (com - float(saved["in_basin_fraction"]))}
        entry["summary_verification"] = summary["verification"]
        entry["map"] = summary["map"]
        report["maps"][name] = entry
    report["n_ablations"] = len(ablations)
    report["status"] = "passed" if not report["checks_failed"] else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checks_passed", "checks_failed", "n_ablations")}, indent=1))
    for name, entry in report["maps"].items():
        if "saved_five_way_comparison" in entry:
            print(name, {p: round(v["difference_pp"], 3) for p, v in entry["saved_five_way_comparison"].items()})
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
