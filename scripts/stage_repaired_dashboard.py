#!/usr/bin/env python3
"""Stage a local temporal dashboard bundle from verified repaired derivatives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

from compare_feature_repair import read_assignments, temporal_stats, validate_rows
from crystal_neighbors import FEATURE_VERSION
from regenerate_repaired_temporal import dump, file_hash, write_csv


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Structural communities through time</title>
<style>body{font:16px system-ui,sans-serif;color:#192a3b;margin:0;background:#f5f7f9}main{max-width:1200px;margin:28px auto;padding:0 22px}h1{font-size:28px;margin-bottom:6px}.sub{color:#526373;max-width:1050px;line-height:1.5}.controls,.card{background:white;border:1px solid #dce3e9;border-radius:10px;padding:18px;margin-top:18px}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:240px}#year{font-size:26px;font-variant-numeric:tabular-nums;width:65px}canvas{display:block;width:100%;height:620px}#details{min-height:55px;line-height:1.5;border-top:1px solid #e4e9ee;padding-top:12px}.small{font-size:13px;color:#526373}a{color:#075c91}button{background:#08729e;border:0;color:white;padding:9px 14px;border-radius:6px;cursor:pointer}</style>
<main><h1>Structural communities through time</h1><p class="sub">The positions are fixed from the community graph. Move the year slider to see how many ICSD entries had been reported in each neighborhood. Circle size shows the accumulated count; color shows its first-publication year. Hover over a circle for its membership and checked description.</p>
<section class="controls"><button id="play">Play</button><label id="year"></label><input id="slider" type="range"><span id="stats"></span></section>
<section class="card"><canvas id="map"></canvas><div id="details">Select a circle to inspect a community.</div><p class="small">Only communities with at least 20 members in the full partition have graph-layout positions. The layout is schematic; the distance between plotted circles is not a calibrated structural distance. Histories use fixed final-partition labels. Entries without a publication year are omitted from temporal counts.</p></section>
<p class="small"><a href="community_time_data.json">All community histories (JSON)</a> · <a href="community_time_counts.csv">Year counts (CSV)</a> · <a href="community_labels.csv">Checked descriptive labels</a> · <a href="manifest.json">Provenance</a></p></main>
<script>const D=__DATA__;const slider=document.querySelector('#slider'),canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');slider.min=D.min_year;slider.max=D.max_year;slider.value=D.max_year;let drawn=[],timer=null;
const visible=D.communities.filter(c=>c.layout);const xs=visible.map(c=>c.layout.x),ys=visible.map(c=>c.layout.y);const minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
function count(c,y){return c.history.reduce((s,[t,n])=>s+(t<=y?n:0),0)}
function draw(){const y=+slider.value,r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=r.width*dpr;canvas.height=r.height*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,r.width,r.height);document.querySelector('#year').textContent=y;drawn=[];let total=0,nc=0;visible.forEach(c=>{const n=count(c,y);if(!n)return;total+=n;nc++;const x=35+(c.layout.x-minx)/(maxx-minx)*(r.width-70),z=35+(maxy-c.layout.y)/(maxy-miny)*(r.height-70),radius=2+Math.sqrt(n)/7;const t=(c.birth_year-D.min_year)/(D.max_year-D.min_year);ctx.beginPath();ctx.arc(x,z,radius,0,2*Math.PI);ctx.fillStyle=`hsla(${215-170*t},65%,42%,.63)`;ctx.fill();drawn.push({c,n,x,y:z,r:Math.max(radius,5)})});document.querySelector('#stats').textContent=`${nc.toLocaleString()} visible communities · ${total.toLocaleString()} dated entries`;}
canvas.addEventListener('mousemove',e=>{const b=canvas.getBoundingClientRect(),x=e.clientX-b.left,y=e.clientY-b.top;let hit=null,dist=Infinity;drawn.forEach(p=>{const d=Math.hypot(x-p.x,y-p.y);if(d<=p.r&&d<dist){hit=p;dist=d}});if(hit){const c=hit.c;document.querySelector('#details').textContent=`Community ${c.community}: ${c.description||'no curated description'}. ${hit.n.toLocaleString()} entries through ${slider.value}; ${c.size.toLocaleString()} total members, ${c.n_undated} undated; first year ${c.birth_year}. ${c.description?'Description checked against full-membership chemistry and space groups; not a certified prototype or function.':''}`;}});
slider.addEventListener('input',draw);window.addEventListener('resize',draw);document.querySelector('#play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play';return}if(+slider.value>=D.max_year)slider.value=D.min_year;document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{slider.value=+slider.value+1;draw();if(+slider.value>=D.max_year){clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play'}},120)});draw();</script></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--community-evidence", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ap = args.run_dir / "graph/community_assignments.csv"
    tp = args.run_dir / "time/graph_time_summary.json"
    rows = read_assignments(ap)
    summary = json.loads(tp.read_text())
    validate_rows(rows, temporal_stats(summary), ap)
    all_sizes, histories = Counter(), defaultdict(Counter)
    for _, year, c in rows:
        if c >= 0:
            all_sizes[c] += 1
            if year is not None:
                histories[c][year] += 1
    with (args.community_evidence / "community_layout.csv").open(newline="") as f:
        layouts = {int(r["community"]): {"x": float(r["x"]), "y": float(r["y"])} for r in csv.DictReader(f)}
    with (args.community_evidence / "renaissance_top20_checked_descriptors.csv").open(newline="") as f:
        descriptions = {int(r["community"]): r for r in csv.DictReader(f)}
    communities, year_rows, labels = [], [], []
    for c, size in sorted(all_sizes.items()):
        h = histories[c]
        label = descriptions.get(c, {})
        communities.append({"community": c, "size": size, "birth_year": min(h) if h else None,
                            "n_undated": size-sum(h.values()), "history": sorted(h.items()),
                            "description": label.get("description"), "layout": layouts.get(c)})
        year_rows += [{"community": c, "year": y, "n_entries": n} for y, n in sorted(h.items())]
        if label:
            labels.append({"kind": "graph_community", "id": c, "canonical_family_name": label["description"],
                           "confidence": "checked descriptive chemistry/SG", "evidence": label["space_group_evidence"],
                           "notes": label["status"]})
    years = [year for _, year, _ in rows if year is not None]
    data = {"feature_version": FEATURE_VERSION, "min_year": min(years), "max_year": max(years),
            "n_successful": len(rows), "n_communities": len(communities), "n_with_layout": len(layouts), "communities": communities}
    dump(out / "community_time_data.json", data)
    write_csv(out / "community_time_counts.csv", year_rows)
    write_csv(out / "community_labels.csv", labels)
    write_csv(out / "community_families_inferred.csv", [{"community": "", "inferred_family": ""}])
    dump(out / "community_prototype_labels.json", [])
    dump(out / "top_communities.json", sorted(communities, key=lambda r: (-r["size"], r["community"])))
    (out / "temporal_viewer.html").write_text(HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    with args.index.open(newline="") as f:
        index = {int(r["cif_names"]): r for r in csv.DictReader(f)}
    with (args.community_evidence / "production_central_representatives_top20.csv").open(newline="") as f:
        reps = list(csv.DictReader(f))
    for r in reps:
        meta = index[int(r["icsd_id"])]
        r.update({"name": r["formula"], "publication_year": r["year"], "sym_group": r["space_group"],
                  "Bravais": meta.get("Bravais", ""), "a": meta.get("a", ""), "b": meta.get("b", ""), "c": meta.get("c", "")})
    write_csv(out / "community_representatives.csv", reps)
    sources = [ap, tp, args.run_dir / "time/node_temporal_events.csv", args.community_evidence / "community_layout.csv"]
    for src in sources:
        shutil.copy2(src, out / src.name)
    dump(out / "manifest.json", {"feature_version": FEATURE_VERSION,
         "source_sha256": {str(p): file_hash(p) for p in sources+[args.index]},
         "producer_sha256": file_hash(Path(__file__)),
         "n_successful": len(rows), "n_communities": len(communities), "n_with_layout": len(layouts),
         "scope": "Local temporal viewer and new labels/representatives/layout; not deployed. Upload scoring still requires the version-matched raw feature matrix and fitted projection objects, not this temporal bundle."})
    (out / "README.md").write_text("# Local temporal dashboard bundle\n\nOpen temporal_viewer.html directly; it embeds its data and needs no server or network. The fixed layout covers communities with at least20 total members. The JSON/CSV files contain all community histories, including smaller communities.\n\nThe canonical-label CSV contains only checked descriptive chemistry/SG labels. Empty inferred/prototype files prevent accidental historical-label fallback when configuring a later app. The representatives use the existing Dash loader schema. Raw CIFs are not included.\n\nThis bundle is not a complete upload-scoring deployment: that also needs the compatible raw features and frozen projection objects. Do not point ICSD_FEATURES_PATH at a saved PCA array. The existing app's narrative text and figure URLs also require review before deployment.\n")
    print(json.dumps({k: data[k] for k in ("n_successful", "n_communities", "n_with_layout")}, indent=2))


if __name__ == "__main__":
    main()
