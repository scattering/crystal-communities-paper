"""Physical regression checks for the repaired production representation."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core import Composition, Lattice, Species, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import icsd_densify_worker as worker
from crystal_neighbors import geometry_crystalnn


def bcc():
    mixture = {Species("Nb", 0): 0.75, Species("In", 0): 0.25}
    return Structure(Lattice.cubic(3.326), [mixture, mixture], [[0, 0, 0], [.5, .5, .5]])


def a15():
    return Structure.from_spacegroup(
        223, Lattice.cubic(5.28),
        [{Species("In", 0): .25, Species("Sn", 0): .75}, Species("Nb", 0)],
        [[0, 0, 0], [.25, 0, .5]],
    )


class ProductionFeaturesTest(unittest.TestCase):
    def test_chemical_properties_ignore_oxidation_annotations(self):
        plain = worker.element_vector_from_species(Composition("Fe"))
        mixed = worker.element_vector_from_species(Composition({Species("Fe", 2): .4, Species("Fe", 3): .6}))
        np.testing.assert_allclose(plain, mixed, atol=1e-12)
        self.assertEqual(plain[0], 26)
        self.assertAlmostEqual(worker.element_vector_from_species(bcc()[0].species)[0], 43)

    def test_disordered_bcc_has_geometry_and_three_rounds_propagate(self):
        s = bcc()
        diagnostics = {}
        x0 = worker.build_structure_embedding(s, 0)
        x3 = worker.build_structure_embedding(s, 3, diagnostics=diagnostics)
        self.assertEqual(x3.shape, (213,))
        self.assertTrue(np.isfinite(x3).all())
        self.assertGreater(np.count_nonzero(x3[7:68]), 0)
        # Symmetry-equivalent sites receive the same neighbour vector each round.
        np.testing.assert_allclose(x3[:136], 8 * x0[:136], atol=1e-10)
        np.testing.assert_allclose(x3[204:], x0[204:], atol=1e-12)
        self.assertFalse(diagnostics["ordered"])
        self.assertEqual(diagnostics["sites_with_unrepresented_cn_mass"], 0)

    def test_annotation_and_mixture_order_do_not_change_embedding(self):
        s = a15()
        plain = s.copy()
        plain.remove_oxidation_states()
        reordered = Structure(s.lattice, [dict(reversed(list(site.species.items()))) for site in s], s.frac_coords)
        expected = worker.build_structure_embedding(s, 3)
        np.testing.assert_allclose(worker.build_structure_embedding(plain, 3), expected, atol=1e-10)
        np.testing.assert_allclose(worker.build_structure_embedding(reordered, 3), expected, atol=1e-10)

    def test_propagation_weights_match_weighted_crystalnn(self):
        s = bcc()
        cnn = geometry_crystalnn()
        info = worker.weighted_nn_from_nndata(cnn.get_nn_data(s, 0), weighted_cn=True)
        expected = cnn.get_nn_info(s, 0)
        key = lambda e: (int(e["site_index"]), tuple(e["image"]))
        actual_weights = {key(e): e["weight"] for e in info}
        expected_weights = {key(e): e["weight"] for e in expected if e["weight"] > 0}
        self.assertEqual(actual_weights.keys(), expected_weights.keys())
        np.testing.assert_allclose(list(actual_weights.values()), [expected_weights[k] for k in actual_weights])
        self.assertTrue(any(0 < w < 1 for w in actual_weights.values()))

    def test_pooled_site_features_are_supercell_invariant(self):
        s = a15()
        supercell = s * [2, 1, 1]
        np.testing.assert_allclose(
            worker.build_structure_embedding(s, 3)[:204],
            worker.build_structure_embedding(supercell, 3)[:204], rtol=1e-8, atol=1e-8,
        )

    def test_neighbor_failure_is_not_a_successful_zero_vector(self):
        cnn = geometry_crystalnn()
        with patch.object(cnn, "get_nn_data", side_effect=RuntimeError("deliberate neighbour failure")):
            with self.assertRaisesRegex(RuntimeError, "deliberate neighbour failure"):
                worker.build_structure_embedding(bcc(), 3, cnn=cnn)

    def test_missing_matminer_is_explicit(self):
        with patch.object(worker, "CrystalNNFingerprint", None):
            with self.assertRaises(ImportError):
                worker.build_structure_embedding(bcc(), 3)

    def test_fast_geometry_uses_physical_periodic_distance(self):
        s = Structure(Lattice.cubic(3), ["Nb"], [[0, 0, 0]])
        cnn = geometry_crystalnn()
        geom, neighbors = worker.local_geometry_vector(s, 0, cnn, None, 12, "fast_local")
        self.assertTrue(neighbors)
        self.assertAlmostEqual(geom[1], 3)
        self.assertGreater(geom[1], 0)


if __name__ == "__main__":
    unittest.main()
