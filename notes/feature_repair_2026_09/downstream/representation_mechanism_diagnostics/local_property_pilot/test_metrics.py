import unittest

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.stats import wasserstein_distance

from notes.feature_repair_2026_09.downstream.representation_mechanism_diagnostics.local_property_pilot import metrics

gf = metrics.gf


def nacl():
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, 0, 0]],
    )


class LocalPropertyMetricsTest(unittest.TestCase):
    def test_continuous_distributions_match_graphlet_raw_features(self):
        structure = nacl()
        neighbor_info = gf._structure_neighbor_info(structure)
        props = metrics.ordered_site_properties(structure)
        observed = metrics.distributions(props, metrics.fixed_edges(structure, neighbor_info))
        expected = {
            **{f"f1/{p}": values for p, values in gf._accumulate_first_order(structure).items()},
            **{
                f"{family}/{prop}": values
                for name, values in gf._accumulate_second_order(structure, neighbor_info).items()
                if name != "f2_distance"
                for family, prop in [next(
                    (family, name[len(family) + 1:])
                    for family in ("f2_mean", "f2_absdiff")
                    if name.startswith(family + "_")
                )]
            },
        }
        for channel, (actual_values, actual_weights) in observed.items():
            pairs = expected[channel]
            expected_values = np.asarray([value for value, _ in pairs])
            expected_weights = np.asarray([weight for _, weight in pairs], dtype=float)
            expected_weights /= expected_weights.sum()
            np.testing.assert_allclose(actual_values, expected_values)
            np.testing.assert_allclose(actual_weights, expected_weights)

    def test_periodic_multiedges_and_same_base_site_loops_are_retained(self):
        structure = Structure(Lattice.cubic(3.0), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
        neighbor_info = [
            [
                {"site_index": 0, "weight": 0.25},
                {"site_index": 0, "weight": 0.75},
                {"site_index": 1},
            ],
            [],
        ]
        edges = metrics.fixed_edges(structure, neighbor_info)
        np.testing.assert_array_equal(edges.i, [0, 0, 0])
        np.testing.assert_array_equal(edges.j, [0, 0, 1])
        np.testing.assert_allclose(edges.w, [0.25, 0.75, 1.0])

        props = metrics.ordered_site_properties(structure)
        shuffled = metrics.shuffled_properties(props, permutation=[1, 0])
        z_column = metrics.PROPERTY_NAMES.index("Z")
        self.assertEqual(shuffled[edges.i[0], z_column], shuffled[edges.j[0], z_column])

    def test_shuffle_preserves_f1_but_can_change_bonded_differences(self):
        structure = Structure(
            Lattice.cubic(4.0), ["Na", "Na", "Cl"],
            [[0, 0, 0], [.4, 0, 0], [.8, 0, 0]],
        )
        props = metrics.ordered_site_properties(structure)
        edges = metrics.FixedEdges(
            np.asarray([0, 1]), np.asarray([1, 0]), np.ones(2),
        )
        before = metrics.distributions(props, edges)
        after = metrics.distributions(
            metrics.shuffled_properties(props, permutation=[0, 2, 1]), edges,
        )
        for prop in metrics.PROPERTY_NAMES:
            a_values, a_weights = before[f"f1/{prop}"]
            b_values, b_weights = after[f"f1/{prop}"]
            a_order = np.argsort(a_values, kind="stable")
            b_order = np.argsort(b_values, kind="stable")
            np.testing.assert_array_equal(a_values[a_order], b_values[b_order])
            np.testing.assert_array_equal(a_weights[a_order], b_weights[b_order])
            self.assertEqual(
                wasserstein_distance(a_values, b_values, a_weights, b_weights), 0.0,
            )
        self.assertTrue(np.all(before["f2_absdiff/Z"][0] == 0.0))
        self.assertTrue(np.all(after["f2_absdiff/Z"][0] > 0.0))

    def test_nonfinite_values_are_excluded_and_missing_mass_reported(self):
        props = np.tile(np.arange(10.0), (3, 1))
        props[1, 0] = np.nan
        edges = metrics.FixedEdges(
            np.asarray([0, 1, 2]), np.asarray([1, 2, 0]), np.asarray([2.0, 3.0, 5.0]),
        )
        raw, missing = metrics.distributions(props, edges, return_missing_mass=True)
        values, weights = raw["f1/Z"]
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(weights.sum(), 1.0)
        self.assertEqual(missing["f1/Z"], 1.0)
        self.assertEqual(missing["f2_mean/Z"], 5.0)
        np.testing.assert_allclose(raw["f2_mean/Z"][1].sum(), 1.0)

    def test_distance_matches_scipy_and_uses_one_scale_per_property(self):
        structure = nacl()
        props = metrics.ordered_site_properties(structure)
        edges = metrics.fixed_edges(structure, gf._structure_neighbor_info(structure))
        raw_a = metrics.distributions(props, edges)
        raw_b = metrics.distributions(metrics.shuffled_properties(
            props, permutation=np.roll(np.arange(len(props)), 1),
        ), edges)
        scales = {
            channel: float(index + 1)
            for index, channel in enumerate(metrics.CHANNEL_NAMES)
        }
        scales["f2_absdiff/Z"] = 0.0
        observed = metrics.distance(raw_a, raw_b, scales)
        for channel, result in observed.items():
            av, aw = raw_a[channel]
            bv, bw = raw_b[channel]
            expected = wasserstein_distance(av, bv, aw, bw) / scales[channel]
            self.assertTrue(np.isfinite(result))
            self.assertAlmostEqual(result, expected)
        self.assertNotIn("f2_absdiff/Z", observed)

        property_fallback = {
            prop: float(index + 1) for index, prop in enumerate(metrics.PROPERTY_NAMES)
        }
        self.assertEqual(set(metrics.distance(raw_a, raw_b, property_fallback)), set(metrics.CHANNEL_NAMES))


if __name__ == "__main__":
    unittest.main()
