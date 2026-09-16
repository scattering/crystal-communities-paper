#!/usr/bin/env python3
"""Restore the frozen matching inputs from the preserved pair table.

Only the eight input columns are copied; saved outcomes, parsed structures and
timings are discarded. This packaging helper was added during source recovery,
not used to select the original candidate pairs.
"""
import argparse
import csv
from pathlib import Path

FIELDS = (
    "gnome_id", "gnome_formula", "gnome_is_train", "icsd_id",
    "icsd_formula", "icsd_year", "icsd_partial_occupancy",
    "post1980_through2015",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.saved_pairs.resolve() == args.output.resolve():
        parser.error("The output must differ from the saved result table.")
    with args.saved_pairs.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            parser.error("The saved table does not contain all eight input fields.")
        rows = [{key: row[key] for key in FIELDS} for row in reader]
    identifiers = [(r["gnome_id"], r["icsd_id"]) for r in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        parser.error("Expected a nonempty table of unique candidate/reference pairs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Restored {len(rows)} pairs with no saved outcome columns.")


if __name__ == "__main__":
    main()
