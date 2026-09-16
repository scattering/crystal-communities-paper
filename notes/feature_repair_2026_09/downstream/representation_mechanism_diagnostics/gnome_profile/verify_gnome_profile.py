#!/usr/bin/env python3
"""Focused regression: saved graphlet profiles must reproduce the projection partition."""
import hashlib, json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
D = ROOT / "notes/feature_repair_2026_09/downstream"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assignments_path = D / "representations/graphlet-dated-replay/graphlet/community_assignments.csv"
index_path = D / "inputs/ICSD_index.csv"
assignments = pd.read_csv(assignments_path)
expected = assignments.loc[assignments.community >= 0].groupby("community").size().to_dict()
profiles = json.load(open(HERE / "graphlet_replay_membership_profiles.json"))
observed = {int(c): p["size"] for c, p in profiles.items()}
assert observed == expected, "profile community membership differs from graphlet replay partition"

projection = pd.read_csv(D / "external_representation/graphlet/external/gnome/projection_full.csv")
assert set(projection.assigned_community.dropna().astype(int)) <= set(observed), "projected receiver lacks a profile"
provenance = json.load(open(HERE / "gnome_profile_provenance.json"))
assert provenance["inputs"][str(assignments_path.relative_to(ROOT))] == sha(assignments_path)
assert provenance["inputs"][str(index_path.relative_to(ROOT))] == sha(index_path)
assert provenance["alignment_checks"]["assignment_ids_sha256"] == provenance["alignment_checks"]["joined_ids_sha256"]
assert provenance["script_sha256"] == sha(HERE / "gnome_profile.py")
for name, digest in provenance["outputs"].items():
    assert digest == sha(HERE / name), f"stale generated output: {name}"
print(f"PASS: {len(expected)} replay communities and {len(assignments):,} assignment IDs are aligned")
