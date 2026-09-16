#!/usr/bin/env python
"""Extract and render the 40 representative candidates of SI §S7.6.

Two steps, both driven by ``notes/review_2026_08/representative_candidates.csv``
(row order within a quadrant = row number in the SI tables):

  extract   pull each candidate's structure from the *public* release it came from
            and write it, as distributed, to ``<cif-dir>/<quadrant>/<rank>_<id>.cif``
  render    read those CIFs and draw one figure per quadrant (2 x 5 panels)

Public inputs (none ICSD-derived):
  GNoME       by_id.zip                 gs://gdm_materials_discovery (google-deepmind/materials_discovery)
  MatterGen   data-release/cifs.zip     github.com/microsoft/mattergen
  JARVIS-DFT  jdft_3d-12-12-2022.json   figshare file 38521619 (the ``.download`` file is a zip)
  Alexandria  alexandria_000NN.json.bz2 alexandria.icams.rub.de, PBE-3D 2025-07-02 shards 00000/00019/00038
  MP          jsonl cache of MPRester.get_structure_by_material_id (material_id, structure)

Usage:
  python scripts/render_representative_candidates.py extract --gnome-zip ... --mattergen-zip ... \
      --jarvis-zip ... --alexandria-dir ... --mp-jsonl ...
  python scripts/render_representative_candidates.py render
"""
from __future__ import annotations

import argparse
import bz2
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "notes/review_2026_08/representative_candidates.csv"
CIF_DIR = ROOT / "notes/review_2026_08/candidate_cifs"
FIG_DIR = ROOT / "resources/figures/icsd_densification"

QUADRANTS = [  # csv key, letter, short title
    ("in_basin_and_formula_match", "A", "In-basin ∧ post-1980 ICSD formula match"),
    ("frontier_and_formula_match", "B", "Frontier ∧ post-1980 ICSD formula match"),
    ("in_basin_and_no_formula_match", "C", "In-basin ∧ no post-1980 ICSD formula match"),
    ("frontier_and_no_formula_match", "D", "Frontier ∧ no post-1980 ICSD formula match"),
]


def load_candidates() -> pd.DataFrame:
    df = pd.read_csv(CANDIDATES)
    df["rank"] = df.groupby("quadrant", sort=False).cumcount() + 1
    return df


def cif_path(row) -> Path:
    return CIF_DIR / row.quadrant / f"{row['rank']:02d}_{row.material_id}.cif"


# ----------------------------------------------------------------------------- extract
def extract(args) -> None:
    from pymatgen.core import Lattice, Structure

    df = load_candidates()
    todo = {s: df[df.source == s] for s in df.source.unique()}
    got = {}

    def keep(row, s: Structure):
        p = cif_path(row)
        p.parent.mkdir(parents=True, exist_ok=True)
        s.to(filename=str(p), fmt="cif")
        got[(row.quadrant, row.material_id)] = len(s)

    if "GNoME" in todo:
        with zipfile.ZipFile(args.gnome_zip) as zf:
            for _, row in todo["GNoME"].iterrows():
                keep(row, Structure.from_str(zf.read(f"by_id/{row.material_id}.CIF").decode(), fmt="cif"))
    if "MatterGen" in todo:
        with zipfile.ZipFile(args.mattergen_zip) as zf:
            for _, row in todo["MatterGen"].iterrows():
                member = re.search(r"member `([^`]+)`", row.cif_source_for_rendering).group(1)
                keep(row, Structure.from_str(zf.read(member).decode(), fmt="cif"))
    if "JARVIS" in todo:
        with zipfile.ZipFile(args.jarvis_zip) as zf:
            name = [n for n in zf.namelist() if n.endswith(".json")][0]
            raw = {e["jid"]: e for e in json.loads(zf.read(name).decode())}
        for _, row in todo["JARVIS"].iterrows():
            a = raw[row.material_id]["atoms"]
            keep(row, Structure(Lattice(np.asarray(a["lattice_mat"])), a["elements"], np.asarray(a["coords"]),
                                coords_are_cartesian=bool(a.get("cartesian", False))))
    if "MP" in todo:
        want = set(todo["MP"].material_id)
        rows = {r.material_id: r for _, r in todo["MP"].iterrows()}
        for line in Path(args.mp_jsonl).open(encoding="utf-8"):
            d = json.loads(line)
            if d["material_id"] in want:
                keep(rows[d["material_id"]], Structure.from_dict(d["structure"]))
                want.discard(d["material_id"])
                if not want:
                    break
    if "Alexandria" in todo:
        want = set(todo["Alexandria"].material_id)
        rows = {r.material_id: r for _, r in todo["Alexandria"].iterrows()}
        for shard in sorted(Path(args.alexandria_dir).glob("alexandria_*.json.bz2")):
            if not want:
                break
            with bz2.open(shard, "rt", encoding="utf-8") as h:
                data = json.load(h)
            entries = data["entries"] if isinstance(data, dict) and "entries" in data else data
            for e in entries:
                mid = str((e.get("data") or {}).get("mat_id") or e.get("entry_id"))
                if mid in want:
                    keep(rows[mid], Structure.from_dict(e["structure"]))
                    want.discard(mid)
            print(f"{shard.name}: remaining {sorted(want)}", flush=True)

    missing = [(r.quadrant, r.material_id) for _, r in df.iterrows() if (r.quadrant, r.material_id) not in got]
    print(f"extracted {len(got)}/{len(df)} structures to {CIF_DIR}")
    if missing:
        print("MISSING:", missing)
        sys.exit(1)


# ----------------------------------------------------------------------------- render
def _colors():
    from pymatgen.vis.structure_vtk import EL_COLORS
    return {el: np.array(rgb) / 255.0 for el, rgb in EL_COLORS["Jmol"].items()}


def _radius(el: str) -> float:
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    return float(CovalentRadius.radius.get(el, 1.5))


def display_cell(s, max_sites=96):
    """Conventional standard cell when the symmetry search finds one and it stays small."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    try:
        sga = SpacegroupAnalyzer(s, symprec=0.1)
        sym = sga.get_space_group_symbol()
        conv = sga.get_conventional_standard_structure()
        if len(conv) <= max(40, int(1.2 * len(s))) and len(conv) <= max_sites:
            return conv, sym
        return s, sym
    except Exception:
        return s, "P1"


def unit_cell_atoms(s, boundary_tol=1e-8):
    """Preserve site positions and repeat only atoms on numerical cell boundaries.

    The tolerance handles floating-point representations of zero/one; it must
    not move relaxed atoms merely because they lie close to a cell face.
    """
    atoms = []
    for site in s:
        f = np.mod(site.frac_coords, 1.0)
        boundary = (f <= boundary_tol) | (1.0 - f <= boundary_tol)
        f[boundary] = 0.0
        images = [(0, 1) if on_face else (0,) for on_face in boundary]
        for shift in itertools.product(*images):
            atoms.append((site.specie.symbol, (f + shift) @ s.lattice.matrix))
    return atoms


def draw_structure(ax, s, colors, title_lines, elev_deg=22.0, azim_deg=25.0, eps=1e-8):
    from matplotlib.patches import Circle
    import matplotlib.patheffects as pe

    lat = s.lattice.matrix
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])

    corners_frac = np.array(list(itertools.product((0, 1), repeat=3)))
    corners = corners_frac @ lat

    def project0(xyz):
        p = xyz @ rz.T
        return np.column_stack([p[:, 0], np.cos(el) * p[:, 2] - np.sin(el) * p[:, 1],
                                np.sin(el) * p[:, 2] + np.cos(el) * p[:, 1]])  # X, Y, depth

    pc0 = project0(corners)
    rotate_in_plane = (np.ptp(pc0[:, 1]) > 1.1 * np.ptp(pc0[:, 0]))  # long axis horizontal (panels are wide)

    def project(xyz):
        q = project0(xyz)
        if rotate_in_plane:
            q = np.column_stack([q[:, 1], -q[:, 0], q[:, 2]])
        return q

    # Atoms inside the cell plus periodic images of atoms exactly on its faces.
    atoms = unit_cell_atoms(s, boundary_tol=eps)
    syms = [a[0] for a in atoms]
    xyz = np.array([a[1] for a in atoms])
    proj = project(xyz)
    rad = np.array([_radius(el) for el in syms])
    rad_draw = np.maximum(rad, 0.75)  # keep H and other small atoms visible
    from pymatgen.core.periodic_table import Element
    metal = {el: Element(el).is_metal for el in set(syms)}
    r_scale = 0.42 if len(atoms) <= 24 else (0.36 if len(atoms) <= 48 else 0.30)

    # Contacts: distance <= 1.2 x sum of covalent radii, drawn as two half-segments.
    items = []  # (depth, kind, payload)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = np.linalg.norm(xyz[i] - xyz[j])
            if 0.4 < d <= 1.2 * (rad[i] + rad[j]):
                mid = (proj[i] + proj[j]) / 2
                mm = metal[syms[i]] and metal[syms[j]]
                items.append(((proj[i, 2] + mid[2]) / 2, "bond", (proj[i, :2], mid[:2], syms[i], mm)))
                items.append(((proj[j, 2] + mid[2]) / 2, "bond", (proj[j, :2], mid[:2], syms[j], mm)))
    for i in range(len(atoms)):
        items.append((proj[i, 2], "atom", (proj[i, :2], r_scale * rad_draw[i], syms[i])))
    items.sort(key=lambda t: t[0])

    # cell edges
    pc = project(corners)
    for a, b in itertools.combinations(range(8), 2):
        if np.sum(np.abs(corners_frac[a] - corners_frac[b])) == 1:
            ax.plot([pc[a, 0], pc[b, 0]], [pc[a, 1], pc[b, 1]], color="0.55", lw=0.6, zorder=0)

    for k, (_, kind, payload) in enumerate(items):
        if kind == "bond":
            p, m, el, mm = payload
            ax.plot([p[0], m[0]], [p[1], m[1]], color=colors.get(el, (0.5, 0.5, 0.5)),
                    lw=0.9 if mm else 1.6, alpha=0.55 if mm else 1.0, solid_capstyle="round", zorder=1 + k)
        else:
            p, r, el = payload
            ax.add_patch(Circle(p, r, facecolor=colors.get(el, (0.5, 0.5, 0.5)), edgecolor="0.15",
                                lw=0.5, zorder=1 + k))

    allxy = np.vstack([proj[:, :2], pc[:, :2]])
    pad = 0.9
    ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title("\n".join(title_lines), fontsize=7.5, pad=2, linespacing=1.15)

    # element key, coloured, below the panel at bottom-left
    els = sorted(set(syms), key=lambda e: syms.index(e))
    x = 0.02
    for el in els:
        t = ax.text(x, -0.01, el, transform=ax.transAxes, fontsize=8, fontweight="bold",
                    color=colors.get(el, (0.5, 0.5, 0.5)), va="top", ha="left", clip_on=False)
        t.set_path_effects([pe.withStroke(linewidth=1.2, foreground="0.25")])
        x += 0.035 + 0.02 * len(el)


def _short(text: str, n: int = 44) -> str:
    text = re.sub(r"\s*\[(canonical|inferred|exemplar)\]\s*$", "", str(text))
    text = text.split(";")[0]
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pymatgen.core import Structure

    df = load_candidates()
    colors = _colors()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for key, letter, title in QUADRANTS:
        sub = df[df.quadrant == key]
        fig, axes = plt.subplots(5, 2, figsize=(11.0, 11.0))
        for ax, (_, row) in zip(axes.ravel(), sub.iterrows()):
            s = Structure.from_file(str(cif_path(row)))
            cell, sym = display_cell(s)
            src = (f"MatterGen, {row.mattergen_family}" if row.source == "MatterGen"
                   else f"{row.source} {row.material_id}")
            lines = [f"{row['rank']}. {row.reduced_formula}  ({src})",
                     f"{sym} · community {row.assigned_community} · d/τ$_c$ = {row.d_over_tau:.2f}",
                     _short(row.canonical_family_name)]
            draw_structure(ax, cell, colors, lines)
            summary.append({"quadrant": letter, "rank": int(row["rank"]), "id": row.material_id,
                            "formula": row.reduced_formula, "spg_symprec0.1": sym,
                            "sites_as_distributed": len(s), "sites_drawn_cell": len(cell)})
        fig.suptitle(f"Quadrant {letter} — {title}", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=0.8, w_pad=0.4)
        out = FIG_DIR / f"candidate_structures_quadrant_{letter}.png"
        fig.savefig(out, dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size/1e6:.2f} MB)")
    pd.DataFrame(summary).to_csv(CIF_DIR / "render_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--gnome-zip", required=True)
    e.add_argument("--mattergen-zip", required=True)
    e.add_argument("--jarvis-zip", required=True)
    e.add_argument("--alexandria-dir", required=True)
    e.add_argument("--mp-jsonl", required=True)
    r = sp.add_parser("render")
    r.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    {"extract": extract, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
