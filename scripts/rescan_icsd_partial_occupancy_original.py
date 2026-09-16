#!/usr/bin/env python3
import csv
import getpass
import json
import re
import shlex
import sys
import time
import zipfile
from pathlib import Path


NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")


def split(line):
    try:
        return shlex.split(line, comments=True, posix=True)
    except ValueError:
        return line.split("#", 1)[0].split()


def occupancy_values(text):
    lines = text.splitlines()
    i = 0
    found = []
    while i < len(lines):
        toks = split(lines[i].strip())
        if not toks or toks[0].lower() != "loop_":
            i += 1
            continue
        i += 1
        tags = []
        while i < len(lines):
            row = split(lines[i].strip())
            if row and row[0].startswith("_"):
                tags.extend(x.lower() for x in row if x.startswith("_"))
                i += 1
            else:
                break
        try:
            col = tags.index("_atom_site_occupancy")
        except ValueError:
            continue
        values = []
        while i < len(lines):
            raw = lines[i].strip()
            row = split(raw)
            if row and (row[0].startswith("_") or row[0].lower() in {"loop_", "stop_"} or row[0].lower().startswith(("data_", "save_"))):
                break
            values.extend(row)
            i += 1
        ncol = len(tags)
        if ncol:
            found.extend(values[j] for j in range(col, len(values), ncol))
    return found


def to_number(value):
    if value in {"?", "."}:
        return None
    m = NUM.match(value)
    return float(m.group(0)) if m else None


def main():
    archive, prior_json, output = map(Path, sys.argv[1:4])
    failed = json.loads(prior_json.read_text())["failures"]
    password = getpass.getpass("ICSD archive password: ").encode()
    rows, failures = [], []
    started = time.time()
    with zipfile.ZipFile(archive) as zf:
        for rec in failed:
            try:
                raw = zf.read(rec["member"], pwd=password)
                vals = occupancy_values(raw.decode("utf-8", errors="replace"))
                parsed = [to_number(x) for x in vals]
                observed = [x for x in parsed if x is not None]
                partial = any(abs(x - 1.0) > 1e-8 for x in observed)
                rows.append((rec["icsd_id"], int(partial), len(vals),
                             len(vals) - len(observed),
                             min(observed) if observed else "",
                             max(observed) if observed else ""))
            except Exception as exc:
                failures.append({**rec, "rescan_error": f"{type(exc).__name__}: {exc}"})
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["icsd_id", "has_partial_occupancy", "n_occupancy_values",
                         "n_unknown_occupancy_values", "min_occupancy", "max_occupancy"])
        writer.writerows(rows)
    meta = {"records": len(rows), "partial": sum(r[1] for r in rows),
            "remaining_failures": failures, "elapsed_seconds": time.time()-started,
            "rule": "manual atom-site loop parser; same occupancy threshold as primary scan"}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k != "remaining_failures"}, indent=2))
    print(f"remaining_failure_count={len(failures)}")


if __name__ == "__main__":
    main()
