#!/usr/bin/env python3
"""Stepping-stone effect as a Sankey: a thick river of new reduced formulas
flows into "joins existing basin" while a tiny trickle diverts to "new basin"
or to the inverted "basin-after-formula" case.

Three flows are hard-wired from the analysis (TRI shared-formula subset,
N = 16,582):

  joins existing basin       13,739  (82.9%)
  co-births a basin           2,682  (16.2%)
  formula precedes basin        161  ( 1.0%)

Output: PNG via plotly + kaleido.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True)
    p.add_argument("--n-existing", type=int, default=13739)
    p.add_argument("--n-cobirth", type=int, default=2682)
    p.add_argument("--n-inverted", type=int, default=161)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    total = args.n_existing + args.n_cobirth + args.n_inverted
    pct_existing = 100.0 * args.n_existing / total
    pct_cobirth = 100.0 * args.n_cobirth / total
    pct_inverted = 100.0 * args.n_inverted / total

    # Sankey nodes:
    # 0  source: "New reduced formula in ICSD"
    # 1  joins existing basin
    # 2  co-births a basin
    # 3  formula precedes basin
    labels = [
        f"<b>New reduced formula appears in ICSD</b><br>(N = {total:,})",
        f"<b>Joins existing structural basin</b><br>{args.n_existing:,} formulas · {pct_existing:.1f}%",
        f"<b>Co-births a basin</b><br>{args.n_cobirth:,} formulas · {pct_cobirth:.1f}%",
        f"<b>Formula precedes basin</b><br>{args.n_inverted:,} formulas · {pct_inverted:.1f}%",
    ]
    node_colors = ["#2c4a55", "#0f6d61", "#b56200", "#8a2f2f"]
    link_colors = ["rgba(15,109,97,0.55)", "rgba(181,98,0,0.55)", "rgba(138,47,47,0.65)"]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node={
            "label": labels,
            "color": node_colors,
            "pad": 30,
            "thickness": 28,
            "line": {"color": "white", "width": 0.5},
        },
        link={
            "source": [0, 0, 0],
            "target": [1, 2, 3],
            "value": [args.n_existing, args.n_cobirth, args.n_inverted],
            "color": link_colors,
            "label": [
                f"82.9% — the stepping-stone rule",
                f"16.2% — co-birth",
                f"1.0% — chemistry precedes structure",
            ],
            "hovertemplate": "%{label}<br>%{value:,} formulas<extra></extra>",
        },
    ))

    fig.update_layout(
        title={
            "text": "<b>The stepping-stone rule</b>",
            "x": 0.5, "xanchor": "center",
        },
        font={"family": "Helvetica, Arial, sans-serif", "size": 12, "color": "#222"},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 30, "r": 30, "t": 60, "b": 30},
        width=args.width,
        height=args.height,
        annotations=[{
            "text": (
                "When a new reduced formula first appears in ICSD, "
                "it overwhelmingly enters a structural basin that already exists."
            ),
            "showarrow": False,
            "x": 0.5, "y": 1.06, "xref": "paper", "yref": "paper",
            "xanchor": "center", "font": {"size": 11, "color": "#5b6672"},
        }],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import plotly.io as pio
    pio.write_image(fig, str(out), format="png", width=args.width, height=args.height, scale=2)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
