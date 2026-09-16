#!/usr/bin/env python3
"""Run the unchanged external projector with spawn instead of Linux fork.

The Alexandria JSON candidate pool leaves a large parent address space even
after unselected records are released. Spawn avoids inheriting that address
space in each feature worker. Cohorts, features, caches and outputs are unchanged.
"""
import multiprocessing

from regenerate_external_projection import main


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())
