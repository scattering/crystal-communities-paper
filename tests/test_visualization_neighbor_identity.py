"""Visualization edges preserve duplicate-row neighbors and original row mapping."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_community_layout as layout
import icsd_graph_connectivity_view as connectivity


class VisualizationNeighborIdentityTests(unittest.TestCase):
    def finder(self):
        finder = Mock()
        finder.fit.return_value = finder
        finder.kneighbors.return_value = (
            np.array([[0., 0.], [0., 0.], [0., 2.]]),
            np.array([[0, 1], [0, 1], [2, 0]]),
        )
        return finder

    def test_connectivity_keeps_mutual_duplicate_and_maps_subset_ids(self):
        with patch.object(connectivity, "NearestNeighbors", return_value=self.finder()):
            edges = connectivity.build_edges(
                np.array([[9.], [0.], [8.], [0.], [2.]]),
                np.array([False, True, False, True, True]), k=1, mutual_knn=True,
            )
        self.assertEqual(edges, [(1, 3, 1.)])

    def test_layout_counts_both_returned_duplicate_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(features="unused.npy", community_assignments="unused.csv",
                output=str(Path(directory) / "layout.csv"), k=1, min_community_size=1,
                top_communities=0, seed=42, iterations=1)
            def positions(graph, **kwargs):
                self.assertEqual(graph[10][20]["weight"], 2)
                self.assertEqual(graph[10][30]["weight"], 1)
                return {c: (float(c), 0.) for c in graph}
            with (patch.object(layout, "parse_args", return_value=args),
                  patch.object(layout.np, "load", return_value=np.array([[0.], [0.], [2.]])),
                  patch.object(layout, "load_assignments", return_value=[10, 20, 30]),
                  patch.object(layout, "NearestNeighbors", return_value=self.finder()),
                  patch.object(layout.nx, "spring_layout", side_effect=positions)):
                self.assertEqual(layout.main(), 0)


if __name__ == "__main__":
    unittest.main()
