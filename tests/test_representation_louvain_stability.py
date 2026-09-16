import importlib.util
from pathlib import Path
import sys
import unittest

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "notes/feature_repair_2026_09/downstream/analyze_representations.py"
sys.path.insert(0, str(ANALYZER.parent))
SPEC = importlib.util.spec_from_file_location("representation_analysis_louvain_test", ANALYZER)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def canonical(partition):
    return sorted(tuple(sorted(members)) for members in partition)


class StableLouvainTests(unittest.TestCase):
    def test_zero_move_tolerance_matches_networkx_fixture(self):
        graph = nx.karate_club_graph()
        expected = nx.community.louvain_communities(
            graph, weight="weight", resolution=1.0, seed=42,
        )
        actual, diagnostics = analysis.stable_louvain_communities(
            graph, weight="weight", resolution=1.0, seed=42,
            move_gain_tolerance=0.0,
        )
        self.assertEqual(canonical(actual), canonical(expected))
        self.assertGreaterEqual(diagnostics["levels"], 1)

    def test_default_tolerance_is_deterministic_and_auditable(self):
        graph = nx.karate_club_graph()
        first, first_diagnostics = analysis.stable_louvain_communities(graph, seed=42)
        second, second_diagnostics = analysis.stable_louvain_communities(graph, seed=42)
        self.assertEqual(canonical(first), canonical(second))
        self.assertEqual(first_diagnostics, second_diagnostics)
        accepted_minima = [
            level["minimum_accepted_gain"]
            for level in first_diagnostics["level_diagnostics"]
            if level["minimum_accepted_gain"] is not None
        ]
        self.assertTrue(accepted_minima)
        self.assertTrue(all(
            gain > analysis.LOUVAIN_NODE_MOVE_GAIN_TOLERANCE
            for gain in accepted_minima
        ))

    def test_sweep_cap_raises_instead_of_silently_truncating(self):
        self.assertEqual(analysis.LOUVAIN_MAX_LOCAL_MOVE_SWEEPS, 100)
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1.0)
        partition = [{0}, {1}]
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            analysis._stable_louvain_one_level(
                graph,
                graph.size(weight="weight"),
                partition,
                resolution=1.0,
                random_state=nx.utils.create_py_random_state(42),
                max_sweeps=1,
            )

    def test_invalid_sweep_cap_is_rejected(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            analysis._stable_louvain_one_level(
                graph,
                graph.size(weight="weight"),
                [{0}, {1}],
                resolution=1.0,
                random_state=nx.utils.create_py_random_state(42),
                max_sweeps=0,
            )

    def test_sub_tolerance_move_is_rejected(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1.0)
        resolution = 2.0 - 1e-15
        test_tolerance = 1e-12
        candidate_gain = 1.0 - resolution / 2.0
        self.assertGreater(candidate_gain, 0.0)
        self.assertLess(candidate_gain, test_tolerance)
        partition, _, improved, diagnostics = analysis._stable_louvain_one_level(
            graph,
            graph.size(weight="weight"),
            [{0}, {1}],
            resolution=resolution,
            random_state=nx.utils.create_py_random_state(42),
            move_gain_tolerance=test_tolerance,
        )
        self.assertFalse(improved)
        self.assertEqual(canonical(partition), [(0,), (1,)])
        self.assertEqual(diagnostics["accepted_moves"], 0)
        self.assertIsNone(diagnostics["minimum_accepted_gain"])
        _, _, zero_tolerance_improved, zero_tolerance_diagnostics = (
            analysis._stable_louvain_one_level(
                graph,
                graph.size(weight="weight"),
                [{0}, {1}],
                resolution=resolution,
                random_state=nx.utils.create_py_random_state(42),
                move_gain_tolerance=0.0,
            )
        )
        self.assertTrue(zero_tolerance_improved)
        self.assertGreater(zero_tolerance_diagnostics["accepted_moves"], 0)

    def test_public_weight_name_is_normalized_to_internal_weight(self):
        graph = nx.karate_club_graph()
        for index, (left, right) in enumerate(graph.edges()):
            graph[left][right]["affinity"] = index * index
        expected = nx.community.louvain_communities(
            graph, weight="affinity", resolution=1.0, seed=42,
        )
        unweighted = nx.community.louvain_communities(
            graph, weight=None, resolution=1.0, seed=42,
        )
        actual, _ = analysis.stable_louvain_communities(
            graph, weight="affinity", resolution=1.0, seed=42,
            move_gain_tolerance=0.0,
        )
        actual_unweighted, _ = analysis.stable_louvain_communities(
            graph, weight=None, resolution=1.0, seed=42,
            move_gain_tolerance=0.0,
        )
        self.assertNotEqual(canonical(expected), canonical(unweighted))
        self.assertEqual(canonical(actual), canonical(expected))
        self.assertEqual(canonical(actual_unweighted), canonical(unweighted))


if __name__ == "__main__":
    unittest.main()
