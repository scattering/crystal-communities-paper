import csv
import json
from pathlib import Path

import numpy as np
import pytest

import analyze_coordination as ac


def test_formula_clusters_connect_transitively_but_keep_sources_distinct():
    pairs = [
        {"source":"mp", "gnome_formula":"A", "comparator_formula":"X"},
        {"source":"mp", "gnome_formula":"B", "comparator_formula":"X"},
        {"source":"mp", "gnome_formula":"B", "comparator_formula":"Y"},
        {"source":"jarvis", "gnome_formula":"C", "comparator_formula":"X"},
    ]
    assert ac.pair_clusters(pairs) == [[0, 1, 2], [3]]


def test_established_formula_convention_handles_order_scale_and_parentheses():
    key = ac.scale_invariant_formula_key
    assert key("Fe2O4") == key("O2Fe")
    assert key("Ca(OH)2") == key("CaO2H2")


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="") as h:
        w=csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)


def test_failed_joins_are_attrition_and_missing_manifest_endpoint_is_error(tmp_path):
    manifest = [
        {"source":"gnome","material_id":"same","formula":"Fe2O3","n_sites":5,"n_elements":2,"chemistry_class":"oxide","was_pilot":False,"record_key":"g"},
        {"source":"mp","material_id":"same","formula":"FeO","n_sites":4,"n_elements":2,"chemistry_class":"oxide","was_pilot":False,"record_key":"m"},
    ]
    mp=tmp_path/"manifest.json"; mp.write_text(json.dumps(manifest))
    cp=tmp_path/"cn.csv"; _write_csv(cp,["source","material_id","neighbor_method","mean_cn"],
        [{"source":"gnome","material_id":"same","neighbor_method":"crystalnn","mean_cn":6}])
    pp=tmp_path/"pairs.csv"; _write_csv(pp,["analysis","source","gnome_id","comparator_id"],
        [{"analysis":"all","source":"mp","gnome_id":"same","comparator_id":"same"}])
    args=type("A",(),{"manifest":mp,"coordination":cp,"pairs":pp,"n_bootstrap":100,"seed":1})()
    result=ac.analyze(args)
    assert result["matched"]["crystalnn"]["all"]["mp"]["n_attrited"] == 1
    # Equal text IDs in different source namespaces are legitimate identities.
    bad=tmp_path/"bad.csv"; _write_csv(bad,["analysis","source","gnome_id","comparator_id"],
        [{"analysis":"all","source":"mp","gnome_id":"same","comparator_id":"absent"}])
    with pytest.raises(ValueError, match="absent from manifest"):
        ac.read_pairs(bad, ac.read_manifest(mp)[1])


def test_component_bootstrap_repeats_whole_components():
    rng=np.random.default_rng(4); clusters=[[0,1],[2]]
    for _ in range(20):
        ix=ac.resample_clusters(clusters,rng).tolist()
        assert ix.count(0) == ix.count(1)
