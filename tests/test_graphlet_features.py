"""Physical and representation-invariance checks for periodic graphlets."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Species, Structure
from pymatgen.core.local_env import CrystalNN, VoronoiNN

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "graphlet_compare"))

import graphlet_features as gf


def fcc():
    return Structure(
        Lattice.cubic(3.6), ["Cu"] * 4,
        [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
    )


def bcc():
    return Structure(Lattice.cubic(3.0), ["Fe"] * 2, [[0, 0, 0], [.5, .5, .5]])


def a15():
    return Structure.from_spacegroup(
        223, Lattice.cubic(5.3), ["Sn", "Nb"], [[0, 0, 0], [.25, 0, .5]],
    )


def shared_edges(raws):
    """Fixed common bins away from exact symmetry-angle boundaries."""
    result = {}
    for name in gf.REGISTRY.all:
        vals = [v for raw in raws for v, _ in raw[name]]
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1.0)
        result[name] = [lo - .1137 * span, hi + .0713 * span]
    return result


class GraphletFeaturesTests(unittest.TestCase):
    def assert_same_histograms(self, raws):
        for edges in (shared_edges(raws), gf.global_bin_edges(dict(enumerate(raws)))):
            baseline = gf.histogram_tensor(raws[0], edges)
            np.testing.assert_allclose(baseline.sum(axis=1), 1.0, atol=1e-6)
            for raw in raws[1:]:
                np.testing.assert_allclose(
                    gf.histogram_tensor(raw, edges), baseline, atol=1e-6, rtol=1e-6,
                )

    def test_primitive_fcc_keeps_periodic_bonds_and_all_angles(self):
        raw = gf.all_raw_features(fcc().get_primitive_structure())
        distances = [v for v, _ in raw["f2_distance"]]
        self.assertEqual(len(distances), 12)
        np.testing.assert_allclose(distances, 3.6 / np.sqrt(2), atol=1e-8)
        angles = [v for v, _ in raw["f3_cos_angle"]]
        self.assertEqual(len(angles), 66)
        self.assertEqual(
            set(np.round(angles, 7)), {-1.0, -.5, 0.0, .5},
        )

    def test_primitive_bcc_has_physical_shells_and_angles(self):
        raw = gf.all_raw_features(bcc().get_primitive_structure())
        distances = np.array([v for v, _ in raw["f2_distance"]])
        self.assertEqual(np.count_nonzero(np.isclose(distances, np.sqrt(3) * 1.5)), 8)
        self.assertEqual(np.count_nonzero(np.isclose(distances, 3.0)), 6)
        angles = np.array([v for v, _ in raw["f3_cos_angle"]])
        self.assertEqual(len(angles), 91)
        self.assertTrue(np.any(np.isclose(angles, -1.0)))
        self.assertTrue(np.any(np.isclose(angles, 1 / 3)))
        self.assertTrue(np.any(np.isclose(angles, -1 / 3)))

    def test_neighbor_method_factories_are_explicit_and_distinct(self):
        self.assertIsInstance(gf.make_neighbor_finder("crystalnn"), CrystalNN)
        self.assertIsInstance(gf.make_neighbor_finder("voronoinn"), VoronoiNN)
        voronoi_settings = gf.neighbor_settings("voronoinn")
        self.assertEqual(voronoi_settings["constructor"], {
            "tol": 0.0,
            "targets": None,
            "cutoff": 13.0,
            "allow_pathological": False,
            "weight": "solid_angle",
            "extra_nn_info": False,
            "compute_adj_neighbors": False,
        })
        self.assertEqual(voronoi_settings["distance_screen_multiplier"], 1.5)
        self.assertEqual(voronoi_settings["retained_neighbor_weight"], "uniform")
        self.assertNotEqual(
            gf.GRAPHLET_FEATURE_VERSIONS["crystalnn"],
            gf.GRAPHLET_FEATURE_VERSIONS["voronoinn"],
        )
        with self.assertRaisesRegex(ValueError, "Unknown graphlet neighbour method"):
            gf.make_neighbor_finder("not-a-method")

    def test_default_method_is_exactly_the_existing_crystalnn_path(self):
        structure = a15()
        self.assertEqual(
            gf.all_raw_features(structure),
            gf.all_raw_features(structure, neighbor_method="crystalnn"),
        )

    def test_voronoinn_bcc_is_uniform_and_distinct_from_crystalnn(self):
        bcc_primitive = bcc().get_primitive_structure()
        voronoi_nn = gf._structure_neighbor_info(
            bcc_primitive, neighbor_method="voronoinn",
        )[0]
        distances = np.asarray([
            np.linalg.norm(n["site"].coords - bcc_primitive[0].coords)
            for n in voronoi_nn
        ])
        self.assertEqual(len(voronoi_nn), 14)
        self.assertEqual(np.count_nonzero(np.isclose(distances, np.sqrt(3) * 1.5)), 8)
        self.assertEqual(np.count_nonzero(np.isclose(distances, 3.0)), 6)
        self.assertTrue(all(float(n["weight"]) == 1.0 for n in voronoi_nn))

        crystal = gf.all_raw_features(bcc_primitive, neighbor_method="crystalnn")
        voronoi = gf.all_raw_features(bcc_primitive, neighbor_method="voronoinn")
        for feature in gf.REGISTRY.first_order:
            self.assertEqual(crystal[feature], voronoi[feature])
        voronoi_shells = gf._histogram(
            voronoi["f2_distance"], 2.5, 3.1, num_bins=2,
            feature_name="f2_distance",
        )
        crystal_shells = gf._histogram(
            crystal["f2_distance"], 2.5, 3.1, num_bins=2,
            feature_name="f2_distance",
        )
        np.testing.assert_allclose(voronoi_shells, [8 / 14, 6 / 14])
        self.assertFalse(np.allclose(crystal_shells, voronoi_shells))

        conventional = bcc()
        conventional_nn = gf._structure_neighbor_info(
            conventional, neighbor_method="voronoinn",
        )
        occurrences = set()
        for center, neighbors in enumerate(conventional_nn):
            for neighbor in neighbors:
                displacement = tuple(np.round(
                    neighbor["site"].coords - conventional[center].coords, 8,
                ))
                occurrences.add((center, int(neighbor["site_index"]), displacement))
        self.assertEqual(len(occurrences), 28)
        for center, neighbor, displacement in occurrences:
            reverse = tuple(-np.asarray(displacement))
            self.assertIn((neighbor, center, reverse), occurrences)
        self.assertEqual(
            len(gf.all_raw_features(
                conventional, neighbor_method="voronoinn",
            )["f2_distance"]),
            28,
        )

    def test_voronoinn_disorder_preserves_geometry_and_occupancy_mass(self):
        lattice = Lattice.cubic(3.326)
        positions = [[0, 0, 0], [.5, .5, .5]]
        ordered = Structure(lattice, ["Nb"] * 2, positions)
        mixed = Structure(lattice, [{"Nb": .75, "In": .25}] * 2, positions)
        partial = Structure(lattice, [{"Nb": .60, "In": .20}] * 2, positions)

        def signature(structure):
            result = []
            for center, neighbors in enumerate(gf._structure_neighbor_info(
                    structure, neighbor_method="voronoinn")):
                for neighbor in neighbors:
                    displacement = neighbor["site"].coords - structure[center].coords
                    result.append((
                        center, int(neighbor["site_index"]),
                        tuple(np.round(displacement, 8)), float(neighbor["weight"]),
                    ))
            return sorted(result)

        self.assertEqual(signature(ordered), signature(mixed))
        self.assertEqual(signature(ordered), signature(partial))
        for site_index in range(2):
            self.assertAlmostEqual(
                gf._effective_atomic_radius(mixed[site_index], site_index),
                gf._effective_atomic_radius(partial[site_index], site_index),
            )

        raw_ordered = gf.all_raw_features(ordered, neighbor_method="voronoinn")
        raw_mixed = gf.all_raw_features(mixed, neighbor_method="voronoinn")
        raw_partial = gf.all_raw_features(partial, neighbor_method="voronoinn")
        self.assertEqual(
            len(raw_mixed["f2_distance"]), 4 * len(raw_ordered["f2_distance"]),
        )
        ordered_mass = sum(weight for _, weight in raw_ordered["f2_distance"])
        mixed_mass = sum(weight for _, weight in raw_mixed["f2_distance"])
        partial_mass = sum(weight for _, weight in raw_partial["f2_distance"])
        self.assertAlmostEqual(mixed_mass, ordered_mass)
        self.assertAlmostEqual(partial_mass, .64 * ordered_mass)

    def test_voronoinn_radius_screen_boundary_and_empty_failure(self):
        structure = Structure(
            Lattice.cubic(20), [{"Mg": .95, "Al": .05}], [[0, 0, 0]],
        )
        radius = gf._effective_atomic_radius(structure[0], 0)
        self.assertAlmostEqual(radius, 1.4875)
        cutoff = 1.5 * (radius + radius)

        def candidate(distance):
            site = PeriodicSite(
                structure[0].species, [distance, 0, 0], structure.lattice,
                coords_are_cartesian=True, to_unit_cell=False,
            )
            return {"site": site, "site_index": 0, "weight": 0.25}

        inside, outside = candidate(cutoff), candidate(cutoff + 1e-6)
        retained = gf._screen_voronoi_neighbors(
            structure, 0, [inside, outside], [radius],
        )
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["weight"], 1.0)

        finder = Mock()
        finder.get_nn_info.return_value = [outside]
        with self.assertRaisesRegex(ValueError, "No geometric neighbours"):
            gf._structure_neighbor_info(
                structure, finder, neighbor_method="voronoinn",
            )

    def test_voronoinn_periodic_images_are_distinct_and_cell_invariant(self):
        primitive = fcc().get_primitive_structure()
        neighbors = gf._structure_neighbor_info(
            primitive, neighbor_method="voronoinn",
        )[0]
        images = [tuple(int(v) for v in n["image"]) for n in neighbors]
        self.assertEqual(len(images), 12)
        self.assertEqual(len(set(images)), 12)
        self.assertTrue(all(int(n["site_index"]) == 0 for n in neighbors))
        self.assertTrue(all(
            np.linalg.norm(n["site"].coords - primitive[0].coords) > 0
            for n in neighbors
        ))
        raw_primitive = gf.all_raw_features(primitive, neighbor_method="voronoinn")
        self.assertEqual(len(raw_primitive["f2_distance"]), 12)
        self.assertEqual(len(raw_primitive["f3_cos_angle"]), 66)
        self.assert_same_histograms([
            raw_primitive,
            gf.all_raw_features(fcc(), neighbor_method="voronoinn"),
            gf.all_raw_features(fcc() * [2, 1, 1], neighbor_method="voronoinn"),
        ])

    def test_voronoinn_histograms_and_cdfs_are_normalized(self):
        raw = gf.all_raw_features(a15(), neighbor_method="voronoinn")
        edges = shared_edges([raw])
        histograms = gf.histogram_tensor(raw, edges)
        np.testing.assert_allclose(histograms.sum(axis=1), 1.0, atol=1e-7)
        cdfs = np.cumsum(histograms, axis=1)
        self.assertTrue(np.all(np.diff(cdfs, axis=1) >= -1e-7))
        np.testing.assert_allclose(cdfs[:, -1], 1.0, atol=1e-7)

        duplicated = {name: list(values) * 2 for name, values in raw.items()}
        scaled = {
            name: [(value, 7.0 * weight) for value, weight in values]
            for name, values in raw.items()
        }
        np.testing.assert_allclose(
            gf.histogram_tensor(duplicated, edges), histograms, atol=1e-7,
        )
        np.testing.assert_allclose(
            gf.histogram_tensor(scaled, edges), histograms, atol=1e-7,
        )

    def test_cell_origin_and_site_order_invariance(self):
        for name, structure in (("FCC", fcc()), ("BCC", bcc()), ("A15", a15())):
            with self.subTest(prototype=name):
                shifted = structure.copy()
                shifted.translate_sites(
                    list(range(len(structure))), [.371, .419, .287],
                    frac_coords=True, to_unit_cell=True,
                )
                reverse = list(reversed(range(len(structure))))
                reordered = Structure(
                    structure.lattice,
                    [structure[i].species for i in reverse],
                    [structure[i].frac_coords for i in reverse],
                )
                variants = [
                    structure, structure.get_primitive_structure(),
                    structure * [2, 1, 1], shifted, reordered,
                ]
                self.assert_same_histograms([gf.all_raw_features(s) for s in variants])

    def test_unordered_triplet_arms_ignore_neighbor_list_order(self):
        structure = a15()
        nn = gf._structure_neighbor_info(structure)
        raw = gf.all_raw_features(structure)
        reverse_raw = dict(raw)
        reverse_raw.update(gf._accumulate_third_order(
            structure, [list(reversed(neighbours)) for neighbours in nn],
        ))
        self.assert_same_histograms([raw, reverse_raw])
        short = [v for v, _ in raw["f3_d_ij"]]
        long = [v for v, _ in raw["f3_d_jk"]]
        self.assertTrue(all(a <= b for a, b in zip(short, long)))

    def test_disordered_bcc_and_a15_retain_geometry(self):
        structures = [
            Structure(
                Lattice.cubic(3.326), [{"Nb": .75, "In": .25}] * 2,
                [[0, 0, 0], [.5, .5, .5]],
            ),
            Structure.from_spacegroup(
                223, Lattice.cubic(5.28), [{"In": .25, "Sn": .75}, "Nb"],
                [[0, 0, 0], [.25, 0, .5]],
            ),
        ]
        for structure in structures:
            with self.subTest(formula=structure.formula):
                raw = gf.all_raw_features(structure)
                self.assertTrue(raw["f2_distance"])
                self.assertTrue(raw["f3_cos_angle"])
                self.assertTrue(all(v > 0 for v, _ in raw["f2_distance"]))
                self.assert_same_histograms([raw])

    def test_oxidation_annotations_and_mixed_valence_do_not_lose_occupancy(self):
        neutral = bcc()
        decorated = neutral.copy()
        decorated.add_oxidation_state_by_element({"Fe": 0})
        mixed = Structure(
            neutral.lattice,
            [{Species("Fe", 2): .5, Species("Fe", 3): .5}] * 2,
            neutral.frac_coords,
        )
        self.assertTrue(all(props["Fe"][0] == 1.0 for props in gf._site_props(mixed)))
        self.assert_same_histograms([
            gf.all_raw_features(s) for s in (neutral, decorated, mixed)
        ])

    def test_neighbor_failures_are_not_silently_empty(self):
        failing = Mock()
        failing.get_nn_info.side_effect = ValueError("test neighbor failure")
        with self.assertRaisesRegex(ValueError, "test neighbor failure"):
            gf._structure_neighbor_info(bcc(), failing)
        empty = Mock()
        empty.get_nn_info.return_value = []
        with self.assertRaises(ValueError):
            gf._structure_neighbor_info(bcc(), empty)

    def test_missing_and_zero_mass_features_fail_with_the_feature_name(self):
        raw = gf.all_raw_features(fcc())
        edges = shared_edges([raw])
        for name, invalid in (
            ("f2_distance", []),
            ("f3_cos_angle", [(0.0, 0.0)]),
            ("f1_electron_affinity", []),
        ):
            with self.subTest(feature=name):
                damaged = dict(raw)
                damaged[name] = invalid
                with self.assertRaisesRegex(ValueError, name):
                    gf.histogram_tensor(damaged, edges)


if __name__ == "__main__":
    unittest.main()
