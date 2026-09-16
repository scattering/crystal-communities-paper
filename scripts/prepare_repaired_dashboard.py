#!/usr/bin/env python3
"""Bind the repaired dashboard to local saved artifacts; never fit a new basis."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = next(path for path in (REPO / "dashboard", REPO / "resources/web_demo")
                 if (path / "frozen_backend.py").is_file())
sys.path.insert(0, str(DASHBOARD))
from frozen_backend import load_bundle, sha256
from crystal_neighbors import FEATURE_VERSION

FIGURES = {
    "graph_time_ratios": "temporal_cliff_stacked_area_repaired.png",
    "prototype_collapse": "pipeline_schematic_repaired.png",
    "stepping_stone": "formula_graph_tri_comparison_repaired.png",
    "gnome_frontier": "fig3_5source_calibration_repaired.png",
    "graph_growth_gif": "synth_prior_quadrant_repaired.png",
}


def prepare(downstream, production, out_dir, pca=None, figure_dir=None):
    downstream, production, out_dir = [Path(p).resolve() for p in (downstream, production, out_dir)]
    default_figures = REPO / ("figures/icsd_densification" if DASHBOARD.name == "dashboard"
                             else "resources/figures/icsd_densification")
    figure_dir = Path(figure_dir or default_figures).resolve()
    evidence = downstream / "community_evidence"
    paths = {
        "basis": downstream / "reference_basis/projection_basis.npz",
        "pca": Path(pca).resolve() if pca else downstream / "inputs/features_pca.npy",
        "sample_assignments": production / "sample_assignments.csv",
        "assignments": production / "graph/community_assignments.csv",
        "node_events": production / "time/node_temporal_events.csv",
        "profiles": evidence / "production_full_membership_profiles.json",
        "families": evidence / "current_family_manifest.json",
        "representatives": evidence / "production_central_representatives_top20.csv",
        "layout": evidence / "community_layout.csv",
        "accessibility": downstream / "accessibility/structural_accessibility_summary.json",
    }
    paths.update({f"figure_{k}": figure_dir / v for k, v in FIGURES.items()})
    manifest = {
        "feature_version": FEATURE_VERSION,
        "observation_year": 2019,
        "scope": "Repaired production map; fixed 32-dimensional projection and checked descriptive labels. No claim of atomic identity or synthesis success.",
        "coordinate_rule": "Saved production PCA columns 1 and 2 for display; all 32 components for nearest centroid and p95 scoring.",
        "artifacts": {k: {"path": os.path.relpath(v, out_dir), "sha256": sha256(v)} for k, v in paths.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    bundle = load_bundle(target)
    report = {"feature_version": FEATURE_VERSION, "n_rows": len(bundle["rows"]),
              "n_communities": len(bundle["basis"]["communities"]), "artifacts_verified": len(paths),
              "observation_year": 2019, "projection_refit": False}
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "# Local repaired dashboard\n\n"
        "Set `ICSD_DASHBOARD_MANIFEST` to this directory's `manifest.json`, then run "
        f"`{DASHBOARD.relative_to(REPO)}/dash_app.py` with the project's Python environment. "
        "The application binds to localhost by default. This configuration does not deploy it.\n\n"
        "The manifest uses relative paths and SHA-256 hashes for every loaded artifact and figure. "
        "Startup also checks the basis's encoder hashes, production PCA/assignment hashes, row order, "
        "partition counts, descriptions, representatives and layout. Mixed or obsolete files disable scoring. "
        "Raw ICSD feature matrices and licensed CIFs are not needed or served.\n\n"
        "Distances use the saved 32-component basis. Display coordinates are its first two components; "
        "the sample changes only the displayed points. The community map has its own graph layout. "
        "Descriptions refer to full membership, not certified atomic prototypes.\n\n"
        "External historical cost uses observation year 2019 and the saved production population moments. "
        "`ICSD_DEMO_OBSERVATION_YEAR` can change the displayed year and age term together. "
        "Three communities have no dated members; their birth is shown as unknown and the cost "
        "retains the production producer's 2010 fallback, identified when applicable. "
        "This is descriptive historical context, not a calibrated probability of synthesis success.\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", required=True, type=Path)
    parser.add_argument("--production", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pca", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.downstream, args.production, args.out_dir, args.pca, args.figure_dir), indent=2))


if __name__ == "__main__":
    main()
