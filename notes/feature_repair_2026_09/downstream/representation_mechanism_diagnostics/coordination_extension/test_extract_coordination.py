"""Focused physical/periodic-image checks for coordination extraction."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "coordination_extractor", HERE / "extract_coordination.py"
)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(extractor)


def supercell(structure: Structure) -> Structure:
    expanded = structure.copy()
    expanded.make_supercell([2, 2, 2])
    return expanded


class CoordinationPhysicsTests(unittest.TestCase):
    def test_simple_cubic_periodic_images_and_supercell_invariance(self):
        primitive = Structure(Lattice.cubic(3.0), ["Al"], [[0, 0, 0]])
        for method in extractor.METHODS:
            with self.subTest(method=method):
                summary = extractor.summarize_neighbors(primitive, method)
                expanded = extractor.summarize_neighbors(supercell(primitive), method)
                self.assertAlmostEqual(summary["mean_cn"], 6.0, places=8)
                self.assertAlmostEqual(expanded["mean_cn"], 6.0, places=8)
                self.assertEqual(summary["n_contacts"], 6)
                self.assertEqual(expanded["n_contacts"], 8 * 6)

                # Every neighbor of a one-site primitive cell has the central
                # site index, but represents one of six distinct lattice images.
                entries = extractor.gf._structure_neighbor_info(
                    primitive, neighbor_method=method
                )[0]
                identities = []
                for entry in entries:
                    self.assertEqual(int(entry["site_index"]), 0)
                    shift = np.asarray(entry["site"].frac_coords) - primitive[0].frac_coords
                    image = tuple(np.rint(shift).astype(int))
                    self.assertTrue(np.allclose(shift, image, rtol=0, atol=1e-6))
                    self.assertNotEqual(image, (0, 0, 0))
                    identities.append(image)
                self.assertEqual(len(identities), len(set(identities)))
                self.assertEqual(len(identities), 6)

    def test_fcc_coordination_and_supercell_invariance(self):
        conventional = Structure(
            Lattice.cubic(4.0),
            ["Al"] * 4,
            [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
        )
        for method in extractor.METHODS:
            with self.subTest(method=method):
                summary = extractor.summarize_neighbors(conventional, method)
                expanded = extractor.summarize_neighbors(supercell(conventional), method)
                self.assertAlmostEqual(summary["mean_cn"], 12.0, places=8)
                self.assertAlmostEqual(expanded["mean_cn"], 12.0, places=8)
                self.assertEqual(summary["n_contacts"], 4 * 12)
                self.assertEqual(expanded["n_contacts"], 32 * 12)

    def test_bcc_soft_crystalnn_and_radius_screened_voronoi(self):
        lattice = Lattice.cubic(2.87)
        positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
        small_radius = Structure(lattice, ["B", "B"], positions)
        large_radius = Structure(lattice, ["K", "K"], positions)

        crystal_small = extractor.summarize_neighbors(small_radius, "crystalnn")
        crystal_large = extractor.summarize_neighbors(large_radius, "crystalnn")
        # CrystalNN is species-independent here and gives a soft coordination
        # between conventional bcc first-shell CN=8 and Voronoi face count=14.
        self.assertAlmostEqual(crystal_small["mean_cn"], crystal_large["mean_cn"], places=10)
        self.assertGreater(crystal_small["mean_cn"], 8.0)
        self.assertLess(crystal_small["mean_cn"], 14.0)
        self.assertAlmostEqual(crystal_small["mean_cn"], 10.689581241954368, places=8)

        voronoi_small = extractor.summarize_neighbors(small_radius, "voronoinn")
        voronoi_large = extractor.summarize_neighbors(large_radius, "voronoinn")
        # At fixed geometry, the 1.5*(r_i+r_j) screen retains only the eight
        # first-shell neighbors for B, but all 14 Voronoi faces for K.
        self.assertAlmostEqual(voronoi_small["mean_cn"], 8.0, places=8)
        self.assertAlmostEqual(voronoi_large["mean_cn"], 14.0, places=8)


if __name__ == "__main__":
    unittest.main()
