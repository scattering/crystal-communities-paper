"""Crystal drawings must preserve relaxed sites near periodic cell faces."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from render_representative_candidates import unit_cell_atoms


class CandidateRenderGeometryTests(unittest.TestCase):
    def test_relaxed_upper_face_coordinates_are_preserved(self):
        # This O position previously moved by ~0.22 A in the pyrophosphate panel.
        structure = Structure(
            Lattice.from_parameters(6.5752, 9.6319, 11.2315, 94.2, 96.1, 106.8),
            ["O"], [[0.98155887, 0.98448738, 0.30066777]],
        )
        before = structure.frac_coords.copy()
        atoms = unit_cell_atoms(structure)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0][0], "O")
        np.testing.assert_allclose(atoms[0][1], structure[0].coords, atol=1e-12)
        np.testing.assert_array_equal(structure.frac_coords, before)

    def test_near_lower_face_site_has_no_extra_outside_image(self):
        structure = Structure(Lattice.cubic(8), ["La"], [[.006472, .25, .7]])
        atoms = unit_cell_atoms(structure)
        self.assertEqual(len(atoms), 1)
        np.testing.assert_allclose(atoms[0][1], structure[0].coords, atol=1e-12)

    def test_exact_boundary_images_preserve_species_and_periodicity(self):
        structure = Structure(Lattice.cubic(4), ["Si", "Ba"], [[0, 0, 0], [.5, .5, .5]])
        atoms = unit_cell_atoms(structure)
        self.assertEqual(Counter(el for el, _ in atoms), {"Si": 8, "Ba": 1})
        silicon = {tuple(xyz) for el, xyz in atoms if el == "Si"}
        self.assertEqual(silicon, {(x, y, z) for x in (0., 4.) for y in (0., 4.) for z in (0., 4.)})
        ba = next(xyz for el, xyz in atoms if el == "Ba")
        np.testing.assert_allclose([np.linalg.norm(np.array(xyz) - ba) for xyz in silicon],
                                   np.sqrt(12), atol=1e-12)

    def test_numerical_roundoff_at_faces_is_canonicalized(self):
        lattice = Lattice.from_parameters(4, 5, 6, 80, 95, 110)
        structure = Structure(lattice, ["Si"], [[-1e-12, 1 + 1e-12, 1e-12]])
        atoms = unit_cell_atoms(structure)
        self.assertEqual(len(atoms), 8)
        frac = lattice.get_fractional_coords(np.array([xyz for _, xyz in atoms]))
        self.assertTrue(np.all((np.abs(frac) < 1e-12) | (np.abs(frac - 1) < 1e-12)))
        for _, xyz in atoms:
            periodic_error = lattice.get_distance_and_image(structure[0].frac_coords,
                                                             lattice.get_fractional_coords(xyz))[0]
            self.assertLess(periodic_error, 2e-11)


if __name__ == "__main__":
    unittest.main()
