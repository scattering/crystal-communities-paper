# Web Demo Scaffold

This directory now contains two layers of the planned public/demo surface for the ICSD structural-history project:

1. `index.html`
- a static landing page
- useful as a lightweight front door and figure hub

2. `dash_app.py`
- the first interactive app scaffold
- intended for integration into an existing Dash deployment such as `danse2`

## Why Dash here

Although a standalone `Streamlit` app would be faster for a one-off demo, Dash is the better fit if this should become another page inside an existing `danse2`-style multi-page site.

Benefits of the current Dash direction:

- easier integration into an existing Dash ecosystem
- easier reuse of shared navigation / theming / deployment patterns
- straightforward path to a CIF-upload scoring page

## Current state

The app currently has:

- an `Overview` route
- a `Score a CIF` route
- a CIF-upload component
- real upload/scoring feedback
- embedded manuscript-supporting figures

The scoring backend is now wired in.

Current scoring behavior:

- parse an uploaded CIF with `pymatgen`
- compute the real structural embedding with `build_structure_embedding(...)`
- project the structure into the frozen ICSD map
- return:
  - nearest structural basin / community
  - centroid distance
  - community `p95` threshold
  - in-basin vs frontier-like classification
  - structural accessibility score `A_i`
  - 2D placement on a sampled historical background

## Next integration step

The next real milestone is to adapt this self-contained app into the actual DANSE2 page/module structure and configure the frozen-map artifact paths in the deployed environment.

Required environment variables:

- `ICSD_FEATURES_PATH`
- `ICSD_COMMUNITY_ASSIGNMENTS_PATH`
- `ICSD_NODE_EVENTS_PATH`

Remaining work after that:

- run one real CIF end-to-end inside DANSE2
- improve malformed-CIF error handling
- add output polish such as nearest exemplars or community labels

## Likely deployment pattern

If this is deployed on a modest shared server:

- no GPU should be needed
- `2–4` CPU cores and `8–16 GB` RAM should be sufficient

The main requirement is simply to preload the frozen map artifacts in memory rather than recomputing them on every request.
