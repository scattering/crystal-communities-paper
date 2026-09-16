"""Focused regression: identical records must have identical fitted/external coordinates."""
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault('FA_STAGE', str(HERE.parents[4]))
os.environ.setdefault('FA_OUT', tempfile.gettempdir())
os.environ.setdefault('FA_COMMON_IDS', str(HERE / 'inputs/common_support_ids.json'))
import run_factor_ablations as driver


class FrozenPCARegression(unittest.TestCase):
    def test_reference_and_external_coordinates_match_for_each_pca_family(self):
        rng = np.random.default_rng(42)
        production = {'raw': rng.normal(size=(2200, 213)), 'saved_pca': None}
        graphlet = {'cdf': rng.normal(size=(600, 1280)).astype(np.float32)}
        graphlet['cdf'][:, -1] = 1  # also exercise zero-variance column removal
        for name, spec in driver.ABLATIONS.items():
            if spec['coordinates'] not in ('crystalweave_pca', 'cdf_pca'):
                continue
            family = 'graphlet' if spec['coordinates'] == 'cdf_pca' else 'crystalweave'
            raw = graphlet['cdf'] if family == 'graphlet' else production['raw']
            indices = np.array([0, 17, 59, len(raw)-1])
            ext = {family: {'gnome': ([str(i) for i in indices], raw[indices].copy())}}
            with self.subTest(name=name):
                reference, external, record, _ = driver.build_coordinates(name, production, graphlet, ext)
                np.testing.assert_allclose(reference[indices], external['gnome'][1], rtol=1e-12, atol=1e-12)
                self.assertEqual(record['pca_solver'], 'randomized' if family == 'graphlet' else 'covariance_eigh')
                self.assertEqual(record['coordinate_convention'], 'frozen_transform_for_all_populations')
                self.assertEqual(record['pca_reference_retransform_max_abs_error'], 0)


if __name__ == '__main__':
    unittest.main()
