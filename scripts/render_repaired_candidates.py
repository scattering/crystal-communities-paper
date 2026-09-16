#!/usr/bin/env python3
"""Extract and render selected public candidates into explicit new destinations.

No download, API access, or ICSD CIF access is implemented. The extraction
manifest binds each public entry and emitted CIF to the selected-row identity.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from render_representative_candidates import _colors, _short, display_cell, draw_structure
from select_repaired_candidates import QUADRANTS, sha256


def candidates(path):
    df = pd.read_csv(path, keep_default_na=False)
    if len(df) != 40 or df.duplicated(["source", "record_key"]).any():
        raise ValueError("Expected forty unique selected public structures")
    for q, _, _ in QUADRANTS:
        if df[df.quadrant == q]["rank"].tolist() != list(range(1, 11)):
            raise ValueError(f"Candidate ranks are invalid: {q}")
    return df


def cif_path(root, row):
    mid = str(row.material_id)
    if Path(mid).name != mid or mid in (".", ".."):
        raise ValueError("Invalid material ID for CIF filename")
    return root / row.quadrant / f"{int(row['rank']):02d}_{mid}.cif"


def extract(args):
    from pymatgen.core import Composition, Lattice, Structure

    df = candidates(args.candidates)
    got = {}

    def keep(row, structure, source_path, source_entry, payload, original_cif=None):
        key = (row.source, row.record_key)
        if key in got:
            raise ValueError(f"Duplicate public source entry: {key}")
        expected = Composition(row.reduced_formula).fractional_composition
        if not expected.almost_equals(structure.composition.fractional_composition, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Public structure formula differs from selected formula: {key}")
        path = cif_path(args.cif_dir, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        if original_cif is not None:
            path.write_bytes(original_cif)
        else:
            structure.to(filename=str(path), fmt="cif")
        emitted = Structure.from_file(str(path))
        if len(emitted) != len(structure) or not expected.almost_equals(emitted.composition.fractional_composition, rtol=1e-5, atol=1e-6):
            raise ValueError(f"CIF serialization changed sites or composition: {key}")
        got[key] = {"quadrant": row.quadrant, "rank": int(row["rank"]), "source": row.source,
                    "material_id": row.material_id, "record_key": row.record_key, "source_path": str(source_path),
                    "source_entry": source_entry, "source_entry_sha256": hashlib.sha256(payload).hexdigest(),
                    "cif_relative_path": str(path.relative_to(args.cif_dir)), "cif_sha256": sha256(path),
                    "selected_formula": row.reduced_formula, "structure_reduced_formula": structure.composition.reduced_formula,
                    "n_sites": len(structure), "ordered": structure.is_ordered, "composition_verified": True}

    for source, path in (("GNoME", args.gnome_zip), ("MatterGen", args.mattergen_zip)):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for _, row in df[df.source == source].iterrows():
                options = [row.zip_member] if source == "MatterGen" else [
                    f"by_id/{row.material_id}.CIF", f"by_id/{row.material_id}.cif",
                    f"{row.material_id}.CIF", f"{row.material_id}.cif"]
                member = next((name for name in options if name in names), None)
                if member is None:
                    raise ValueError(f"Selected public CIF is absent: {source}/{row.record_key}")
                payload = archive.read(member)
                keep(row, Structure.from_str(payload.decode(), fmt="cif"), path, member, payload, payload)

    def json_payload(record):
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()

    wanted = {row.material_id: row for _, row in df[df.source == "MP"].iterrows()}
    with args.mp_jsonl.open() as handle:
        for line in handle:
            entry = json.loads(line)
            if entry["material_id"] in wanted:
                row = wanted.pop(entry["material_id"])
                keep(row, Structure.from_dict(entry["structure"]), args.mp_jsonl, row.material_id, json_payload(entry))
            if not wanted:
                break
    with args.jarvis_json.open() as handle:
        raw = json.load(handle)
    wanted = {row.material_id: row for _, row in df[df.source == "JARVIS"].iterrows()}
    for entry in raw:
        if entry.get("jid") in wanted:
            row = wanted.pop(entry["jid"])
            atoms = entry["atoms"]
            structure = Structure(Lattice(atoms["lattice_mat"]), atoms["elements"], atoms["coords"],
                                  coords_are_cartesian=bool(atoms.get("cartesian", False)))
            keep(row, structure, args.jarvis_json, row.material_id, json_payload(entry))
    del raw
    wanted = {row.material_id: row for _, row in df[df.source == "Alexandria"].iterrows()}
    for shard_id in (0, 19, 38):
        path = args.alexandria_dir / f"alexandria_{shard_id:05d}.json.bz2"
        with bz2.open(path, "rt") as handle:
            raw = json.load(handle)
        entries = raw["entries"] if isinstance(raw, dict) else raw
        for entry in entries:
            mid = str((entry.get("data") or {}).get("mat_id") or entry.get("entry_id"))
            if mid in wanted:
                row = wanted.pop(mid)
                keep(row, Structure.from_dict(entry["structure"]), path, mid, json_payload(entry))
        del entries, raw
        if not wanted:
            break
    missing = [(r.source, r.record_key) for _, r in df.iterrows() if (r.source, r.record_key) not in got]
    if missing:
        raise ValueError(f"Public extraction incomplete: {missing}")
    rows = [got[(r.source, r.record_key)] for _, r in df.iterrows()]
    with (args.cif_dir / "extraction_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.cif_dir / "extraction_provenance.json").write_text(json.dumps({
        "candidate_table_sha256": sha256(args.candidates), "producer_sha256": sha256(__file__),
        "n_extracted": len(rows), "scope": "Frozen public structures only; original CIF bytes retained for ZIP sources. JSON-source structures are serialized as CIF without relaxation."}, indent=2) + "\n")
    print(f"Extracted and verified {len(rows)} public structures in {args.cif_dir}")


def draw_disordered(ax, structure, colors, titles):
    """Occupancy wedges rather than silently replacing mixed sites by one atom."""
    from matplotlib.patches import Wedge
    matrix = np.array([[.906, .423, 0], [-.159, .34, .927]])
    coords = structure.cart_coords @ matrix.T
    corners_frac = np.array(list(itertools.product((0, 1), repeat=3)))
    corners = corners_frac @ structure.lattice.matrix @ matrix.T
    for i, j in itertools.combinations(range(8), 2):
        if np.sum(abs(corners_frac[i] - corners_frac[j])) == 1:
            ax.plot(*corners[[i, j]].T, color=".6", lw=.6)
    elements = set()
    for site, xy in zip(structure, coords):
        angle = 0
        for species, occupancy in sorted(site.species.items(), key=lambda x: str(x[0])):
            element = species.symbol
            elements.add(element)
            end = angle + 360 * float(occupancy)
            ax.add_patch(Wedge(xy, .35, angle, end, facecolor=colors.get(element, (.5, .5, .5)), edgecolor=".2", lw=.4))
            angle = end
        if angle < 360:
            ax.add_patch(Wedge(xy, .35, angle, 360, facecolor="white", edgecolor=".2", lw=.4))
    xy = np.vstack([coords, corners])
    ax.set(xlim=(xy[:, 0].min() - 1, xy[:, 0].max() + 1), ylim=(xy[:, 1].min() - 1, xy[:, 1].max() + 1))
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title("\n".join(titles), fontsize=7.5, linespacing=1.15)
    ax.text(.02, -.01, "Occupancy wedges; white = vacancy; " + ", ".join(sorted(elements)),
            transform=ax.transAxes, fontsize=6, va="top")


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure

    df = candidates(args.candidates)
    provenance = json.loads((args.cif_dir / "extraction_provenance.json").read_text())
    if provenance["candidate_table_sha256"] != sha256(args.candidates):
        raise ValueError("CIF extraction belongs to a different candidate selection")
    manifest = pd.read_csv(args.cif_dir / "extraction_manifest.csv", keep_default_na=False)
    checks = {(r.source, r.record_key): r for _, r in manifest.iterrows()}
    colors, summary = _colors(), []
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    for q, letter, title in QUADRANTS:
        fig, axes = plt.subplots(5, 2, figsize=(11, 11))
        for ax, (_, row) in zip(axes.ravel(), df[df.quadrant == q].iterrows()):
            path = cif_path(args.cif_dir, row)
            if checks[(row.source, row.record_key)].cif_sha256 != sha256(path):
                raise ValueError(f"Extracted CIF hash changed: {path}")
            structure = Structure.from_file(str(path))
            cell, symmetry = display_cell(structure)
            source = f"{row.source} {row.material_id}"
            if row.source == "MatterGen":
                source = f"MatterGen, {row.mattergen_family}"
            titles = [f"{row['rank']}. {row.reduced_formula} ({source})",
                      f"{symmetry} · community {row.assigned_community} · d/r$_{{c,95}}$ = {row.d_over_tau:.2f}",
                      "Community: " + _short(row.canonical_family_name, 65)]
            (draw_structure if cell.is_ordered else draw_disordered)(ax, cell, colors, titles)
            summary.append({"quadrant": letter, "rank": int(row["rank"]), "source": row.source,
                            "id": row.material_id, "record_key": row.record_key, "formula": row.reduced_formula,
                            "spg_symprec0.1": symmetry, "sites_as_distributed": len(structure),
                            "sites_drawn_cell": len(cell), "ordered": structure.is_ordered})
        fig.suptitle(f"Quadrant {letter} — {title}", fontsize=11, y=.995)
        fig.tight_layout(rect=(0, 0, 1, .98), h_pad=.8, w_pad=.4)
        path = args.figure_dir / f"candidate_structures_quadrant_{letter}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        print(f"Rendered {path}")
    pd.DataFrame(summary).to_csv(args.figure_dir / "render_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "render"):
        child = sub.add_parser(name)
        child.add_argument("--candidates", type=Path, required=True)
        child.add_argument("--cif-dir", type=Path, required=True)
        if name == "extract":
            for source in ("gnome-zip", "mattergen-zip", "jarvis-json", "mp-jsonl", "alexandria-dir"):
                child.add_argument("--" + source, type=Path, required=True)
        else:
            child.add_argument("--figure-dir", type=Path, required=True)
            child.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    (extract if args.command == "extract" else render)(args)


if __name__ == "__main__":
    main()
