"""Manufactured diagnostic checks; no model or simulation calls."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "coverage_analysis", Path(__file__).with_name("analyze_saved_coverage.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoverageChecks(unittest.TestCase):
    def test_nearby_is_not_exact_coverage(self):
        row = module.coverage([0, 0, 0], [[1e-12, 0, 0]])
        self.assertFalse(row["exact_encoded_coordinate_member"])
        self.assertEqual(row["nearest_selected_site_distance_mm"], 1e-9)

    def test_duplicate_sites_count_as_one_exact_coordinate(self):
        row = module.coverage([1, 2, 3], [[1, 2, 3], [1, 2, 3]])
        self.assertTrue(row["exact_encoded_coordinate_member"])
        self.assertEqual(row["nearest_selected_site_distance_mm"], 0)

    def test_unavailable_part_has_no_invented_nearest_site(self):
        self.assertEqual(module.coverage([0, 0, 0], []), {
            "exact_encoded_coordinate_member": False,
            "nearest_selected_site_distance_mm": None})

    def test_nonfinite_coordinate_is_refused(self):
        with self.assertRaises(ValueError):
            module.coverage([math.nan, 0, 0], [[0, 0, 0]])

    def test_repeated_bands_do_not_inflate_unique_witnesses(self):
        row = {"query_index": 5, "part": "ring", "flip_type": "away",
               "query_object_m": [1, 2, 3], "band_mm": 2}
        result = module.unique_witnesses([row, {**row, "band_mm": 5}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["observed_bands_mm"], [2, 5])
        self.assertEqual(row["band_mm"], 2)

    def test_same_index_different_anatomy_is_separate(self):
        row = {"query_index": 5, "part": "ring", "flip_type": "away",
               "query_object_m": [1, 2, 3], "band_mm": 2}
        self.assertEqual(len(module.unique_witnesses(
            [row, {**row, "part": "little"}])), 2)

    def test_inconsistent_index_coordinate_is_refused(self):
        row = {"query_index": 5, "part": "ring", "flip_type": "away",
               "query_object_m": [1, 2, 3], "band_mm": 2}
        with self.assertRaises(ValueError):
            module.unique_witnesses([row, {**row, "query_object_m": [4, 5, 6]}])

    def test_soft_l1_formula_matches_zero_and_scalar(self):
        self.assertEqual(module.robust_cost([0, 0], .2), 0)
        self.assertAlmostEqual(module.robust_cost([.3], .2),
                               .04 * (math.sqrt(1 + (.3 / .2) ** 2) - 1))

    def test_equal_sse_does_not_determine_soft_l1_cost(self):
        a, b = [1., 0.], [math.sqrt(.5), math.sqrt(.5)]
        self.assertAlmostEqual(sum(v*v for v in a), sum(v*v for v in b))
        self.assertNotAlmostEqual(module.robust_cost(a, .2), module.robust_cost(b, .2))


if __name__ == "__main__":
    unittest.main()
