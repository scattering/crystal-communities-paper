#!/usr/bin/env python3
"""Small wiring check for the companion repository.

This is deliberately not a miniature data bundle. It only verifies that
the checkout can compile the scripts and render a data-free figure. The
full numerical reproduction still requires the Zenodo artifacts described
in docs/HOW_TO_REPRODUCE.md.
"""
from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/tmp/crystal-communities-paper-smoke",
        help="Directory for temporary smoke-test outputs.",
    )
    return parser.parse_args()


def run(name: str, command: list[str]) -> bool:
    proc = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"{status}  {name}")
    if proc.returncode != 0:
        print(proc.stdout.strip())
    return proc.returncode == 0


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    compiled = compileall.compile_dir(REPO / "scripts", quiet=1)
    print(f"{'PASS' if compiled else 'FAIL'}  compile scripts/")
    ok = ok and compiled

    ok = ok and run(
        "render pipeline schematic",
        [
            sys.executable,
            "scripts/make_fig_pipeline_schematic.py",
            "--output",
            str(out_dir / "pipeline_schematic.png"),
        ],
    )

    if ok:
        print(f"\nSmoke outputs written to {out_dir}")
        print("Full data-dependent reproduction still requires the Zenodo bundle.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
