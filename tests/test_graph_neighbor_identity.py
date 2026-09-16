"""Duplicate vectors must not lose true neighbours or introduce self-loops."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import icsd_graph_community_postprocess as partition
import icsd_graph_time_evolution as temporal


class GraphNeighborIdentityTests(unittest.TestCase):
    def build_both(self, points, indices, distances, k, mutual):
        graphs = []
        for module in (partition, temporal):
            finder = Mock()
            finder.kneighbors.return_value = (
                np.asarray(distances, dtype=float), np.asarray(indices, dtype=int)
            )
            with patch.object(module, "NearestNeighbors", return_value=finder) as constructor:
                if module is partition:
                    graph, _ = module.build_weighted_graph(np.asarray(points), k, mutual, 1)
                else:
                    graph = module.build_weighted_graph(np.asarray(points), k, mutual)
                constructor.assert_called_once_with(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
            self.assertEqual(list(nx.selfloop_edges(graph)), [])
            graphs.append(graph)
        self.assertEqual(set(graphs[0].edges), set(graphs[1].edges))
        for u, v in graphs[0].edges:
            self.assertAlmostEqual(graphs[0][u][v]["weight"], graphs[1][u][v]["weight"])
        return graphs[0]

    def test_duplicate_pair_with_self_second_keeps_mutual_connection(self):
        # This exact neighbour ordering is returned by sklearn for these rows.
        graph = self.build_both(
            [[0.], [0.], [2.]], [[0, 1], [0, 1], [2, 0]],
            [[0., 0.], [0., 0.], [0., 2.]], k=1, mutual=True,
        )
        self.assertEqual(set(graph.edges), {(0, 1)})
        self.assertEqual(graph[0][1]["weight"], 1.)

    def test_self_absent_from_tied_candidates_keeps_first_k_other_rows(self):
        # Row 3 is absent from its k+1 tied candidates; it must keep only 0,1.
        for mutual in (False, True):
            with self.subTest(mutual=mutual):
                graph = self.build_both(
                    [[0.]] * 4, [[0, 1, 2]] * 4, [[0., 0., 0.]] * 4,
                    k=2, mutual=mutual,
                )
                expected = {(0, 1), (0, 2), (1, 2)}
                if not mutual:
                    expected |= {(0, 3), (1, 3)}
                self.assertEqual(set(graph.edges), expected)

    def test_untied_rows_preserve_gaussian_weights_and_distance_alignment(self):
        graph = self.build_both(
            [[0.], [1.], [3.]], [[0, 1], [1, 0], [2, 1]],
            [[0., 1.], [0., 1.], [0., 2.]], k=1, mutual=False,
        )
        self.assertEqual(set(graph.edges), {(0, 1), (1, 2)})
        self.assertAlmostEqual(graph[0][1]["weight"], math.exp(-1))
        self.assertAlmostEqual(graph[1][2]["weight"], math.exp(-4))


if __name__ == "__main__":
    unittest.main()
