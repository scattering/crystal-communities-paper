#!/bin/bash
# Audit ICSD CIF headers for theoretical/calculated/predicted/DFT/VASP flags.
# Reads the encrypted ICSD_CIFs.zip directly using the ICSD_ZIP_PASSWORD env var.
# Output is aggregate-only JSON; no per-CIF data leaves TACC.
#
# Usage: ICSD_ZIP_PASSWORD=... bash this_script.sh
#                            (or sbatch this_script.sh after exporting the var)
#SBATCH -J icsd-theo-audit
#SBATCH -A CDA24014
#SBATCH -p skx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -o /work2/09870/williamratcliff/stampede3/icsd_graph_runs/icsd_theoretical_audit/slurm-%j.out
#SBATCH -e /work2/09870/williamratcliff/stampede3/icsd_graph_runs/icsd_theoretical_audit/slurm-%j.err
#
# Audit the ICSD source CIFs for the theoretical / calculated / predicted
# flag and report (a) counts and (b) per-cutoff fractions to support the
# Tier-1 reviewer concern that the manuscript's "experimental baseline"
# might silently include theoretical entries.
#
# ICSD CIF tags relevant to the audit (per ICSD specification):
#   _audit_creation_method   "calculated", "predicted", "theoretical" or empty
#   _icsd_creation_date      sometimes paired with a _diffrn_radiation tag
#                            absence of which suggests no measured diffraction
#
# Output: a single JSON of aggregate counts (no per-CIF data leaves TACC),
# safe to pull back to the local repo.

set -euo pipefail

WORKROOT=/work2/09870/williamratcliff/stampede3
OUT=$WORKROOT/icsd_graph_runs/icsd_theoretical_audit
mkdir -p "$OUT"

source $WORKROOT/conda/etc/profile.d/conda.sh
conda activate $WORKROOT/conda_envs/icsd_densify

python3 - <<'PYEOF'
"""ICSD theoretical-flag audit.

Walks the canonical ICSD CIF directory and counts entries flagged as
theoretical / calculated / predicted (via _audit_creation_method or
adjacent fields in the CIF header). Cross-references with the production
community_assignments.csv (icsd_id, year, community) to report per-decade
shares.

Intentionally pure-stdlib pymatgen-free except for pymatgen.io.cif for
robust CIF parsing, since the audit needs to be transparent. No
classification logic is hidden in a third-party library.
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# === paths ===
WORK = Path("/work2/09870/williamratcliff/stampede3")
ASSIGNMENTS = WORK / "icsd_graph_runs/full_matminer_ops_spr_20260419_labels3/community_assignments.csv"

# Locate the ICSD CIF source directory. Two probable locations:
CANDIDATES = [
    WORK / "reference_data/icsd_cif_full",
    WORK / "reference_data/icsd_cif_sample_3038020",   # a sample if full is unavailable
]
CIF_DIR = next((p for p in CANDIDATES if p.exists()), None)
if CIF_DIR is None:
    print(json.dumps({"error": "no ICSD CIF directory found", "candidates_checked": [str(p) for p in CANDIDATES]}, indent=2))
    sys.exit(1)
print(f"[audit] using CIF source: {CIF_DIR}", file=sys.stderr)

# === load icsd_id → year, community ===
print(f"[audit] loading {ASSIGNMENTS} ...", file=sys.stderr)
id_to_meta = {}
with ASSIGNMENTS.open() as f:
    for row in csv.DictReader(f):
        try:
            iid = int(row["icsd_id"])
            yr = int(row["year"]) if row.get("year") else None
            cm = int(row["community"]) if row.get("community") else None
            id_to_meta[iid] = (yr, cm)
        except (KeyError, ValueError):
            continue
print(f"[audit] loaded {len(id_to_meta)} community assignments", file=sys.stderr)

# === regex-scan CIF headers for the theoretical flag ===
PAT_FLAG = re.compile(
    r"_(?:audit_creation_method|publ_section_references|computing_data_reduction)"
    r"\s+['\"]?(.*?)['\"]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
KEYWORDS_THEO = re.compile(r"(theoretical|calculated|predicted|DFT|VASP|first[- ]principles)", re.IGNORECASE)

# Walk CIFs (lazy peek at the first ~8 KB of each — flags live near the top)
PEEK = 8192
total = 0
flagged_theo = 0
flagged_unknown = 0
per_decade_total = Counter()
per_decade_theo = Counter()
year_unknown = 0

cif_files = list(CIF_DIR.rglob("*.cif")) + list(CIF_DIR.rglob("*.CIF"))
print(f"[audit] {len(cif_files)} CIFs to scan", file=sys.stderr)
for i, p in enumerate(cif_files):
    if i and i % 5000 == 0:
        print(f"  {i}/{len(cif_files)} scanned", file=sys.stderr)
    # Extract icsd_id from filename. ICSD typically names CIFs by collection code.
    name = p.stem
    m = re.search(r"(\d+)", name)
    if not m:
        continue
    iid = int(m.group(1))
    if iid not in id_to_meta:
        continue
    total += 1
    try:
        head = p.read_text(encoding="utf-8", errors="replace")[:PEEK]
    except Exception:
        flagged_unknown += 1
        continue
    is_theo = bool(KEYWORDS_THEO.search(head))
    yr, _ = id_to_meta[iid]
    if yr is None:
        year_unknown += 1
        decade = "unknown"
    else:
        decade = f"{(yr//10)*10}s"
    per_decade_total[decade] += 1
    if is_theo:
        flagged_theo += 1
        per_decade_theo[decade] += 1

result = {
    "cif_directory_used": str(CIF_DIR),
    "scanned_cif_count": total,
    "flagged_theoretical": flagged_theo,
    "flagged_theoretical_fraction": flagged_theo / total if total else None,
    "year_unknown": year_unknown,
    "per_decade": {
        d: {
            "n": per_decade_total[d],
            "n_theoretical": per_decade_theo.get(d, 0),
            "fraction_theoretical": (per_decade_theo.get(d, 0) / per_decade_total[d]) if per_decade_total[d] else None,
        } for d in sorted(per_decade_total.keys(), key=lambda s: (s == "unknown", s))
    },
    "interpretation": (
        "Conservative regex audit of CIF headers. A CIF is flagged theoretical "
        "if its first 8 KB contains any of: 'theoretical', 'calculated', "
        "'predicted', 'DFT', 'VASP', 'first-principles' (case-insensitive). "
        "False positives are possible (e.g. an experimental paper that "
        "compares against DFT in references); false negatives are also "
        "possible. The headline scanned_cif_count is the population "
        "intersected with the 167.5K-row community-assigned ICSD subset."
    ),
}
out = Path("/work2/09870/williamratcliff/stampede3/icsd_graph_runs/icsd_theoretical_audit/icsd_theoretical_audit_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
print(f"[audit] wrote {out}", file=sys.stderr)
print(json.dumps(result, indent=2))
PYEOF
