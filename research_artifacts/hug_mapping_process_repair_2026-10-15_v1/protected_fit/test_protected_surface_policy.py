"""Synthetic policy tests: no scene, geometry, inference or physics imports."""
import copy
import unittest
import numpy as np
from protected_surface_policy import protected_surface_constraint_policy


def reference(prox=(True, False), ori=(True, False, False), valid=(True, True, False)):
    p, o = len(prox), len(ori)
    return {'proximity': {'passed': np.array(prox, bool)},
            'exact_orientation': {'passed': np.array(ori, bool)},
            'exact_orientation_fields': {'valid': np.array(valid, bool)},
            'literal_surface_pass': np.array(tuple(prox) + tuple(ori), bool),
            'surface_margin': np.full(p + 5 * o, .1)}


class PolicyTests(unittest.TestCase):
    def test_exact_block_mask_and_counts(self):
        p = protected_surface_constraint_policy(reference())
        expected = np.array([1, 0] + [1, 0, 0] * 3 + [1, 1, 0] * 2, bool)
        np.testing.assert_array_equal(p['hard_mask'], expected)
        np.testing.assert_array_equal(p['slack_coefficients'], (~expected).astype(float))
        self.assertEqual((p['hard_count'], p['soft_count']), (8, 9))
        self.assertEqual(len(p['row_reasons']), 17)
        self.assertEqual(len(p['row_block_labels']), 17)

    def test_zero_offsets_retain_every_existing_search_buffer(self):
        p = protected_surface_constraint_policy(reference())
        nominal = np.r_[np.full(2, 1e-6), np.full(3, 1e-8), np.full(12, 1e-6)]
        np.testing.assert_array_equal(p['offset'], np.zeros(17))
        np.testing.assert_array_equal(p['search_buffer_per_raw_row'], nominal)
        self.assertTrue(p['existing_inward_search_buffers_retained_on_every_row'])

    def test_slack_cannot_purchase_protected_violation(self):
        p = protected_surface_constraint_policy(reference())
        current = np.full(17, -2.)
        g = current + p['offset'] + p['slack_coefficients'] * 100.
        self.assertTrue(np.all(g[p['hard_mask']] < 0.))
        self.assertTrue(np.all(g[~p['hard_mask']] > 0.))

    def test_slack_derivative_zero_hard_one_soft(self):
        p = protected_surface_constraint_policy(reference())
        current = np.linspace(-.2, .3, 17)
        def g(t): return current + p['offset'] + p['slack_coefficients'] * t
        fd = (g(.3 + 1e-6) - g(.3 - 1e-6)) / 2e-6
        np.testing.assert_allclose(fd, p['slack_coefficients'], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(fd[p['hard_mask']], np.zeros(p['hard_count']))

    def test_eligible_failed_orientation_guidance_hard_sign_soft(self):
        p = protected_surface_constraint_policy(reference((), (False,), (True,)))
        np.testing.assert_array_equal(p['hard_mask'], [False, False, False, True, True])

    def test_pooled_only_no_invented_original_union(self):
        p = protected_surface_constraint_policy(reference((False,), (False,), (False,)))
        self.assertEqual(p['hard_count'], 0)
        self.assertIsNone(p['reference_hard_adjusted_margin_min'])
        np.testing.assert_array_equal(p['offset'], np.zeros(6))
        np.testing.assert_array_equal(p['slack_coefficients'], np.ones(6))

    def test_exact_boundary_reference_refused_by_stricter_search_target(self):
        r = reference((True,), (), ())
        r['surface_margin'][0] = -1e-9 / .001
        # Literal near equality may pass, but does not satisfy the declared
        # stricter optimizer target. Refuse this start; do not silently drop
        # protection or change any final scientific threshold.
        with self.assertRaisesRegex(ValueError, 'violates'):
            protected_surface_constraint_policy(r)
        r['surface_margin'][0] = 0.
        p = protected_surface_constraint_policy(r)
        self.assertEqual(p['reference_adjusted_surface_margin'][0], 0.)
        self.assertEqual(p['offset'][0], 0.)

    def test_strict_sign_boundary_still_needs_exact_gate(self):
        p = protected_surface_constraint_policy(reference((), (True,), (True,)))
        # Move a protected native-away row to sign_margin==0. The optimizer's
        # retained inward search buffer refuses it, and the unchanged literal
        # sign_margin>0 check also fails.
        native_sign, candidate_score, deadzone = -1., -.01, .01
        sign_margin = native_sign * candidate_score - deadzone
        raw = np.array([sign_margin - 1e-8, .1, .1, .1, .1])
        self.assertEqual((raw + p['offset'])[0], -1e-8)
        self.assertFalse(sign_margin > 0.)
        self.assertTrue(p['strict_far_sign_and_boundary_require_literal_final_gate'])

    def test_native_away_uses_existing_signed_margin(self):
        p = protected_surface_constraint_policy(reference((), (True,), (True,)))
        for candidate_score, expected in [(-.4, True), (.4, False)]:
            signed_margin = -candidate_score - .01
            raw = np.array([signed_margin - 1e-8, .1, .1, .1, .1])
            self.assertEqual(bool((raw + p['offset'])[0] >= 0), expected)

    def test_zero_orientation_and_empty_json_vectors(self):
        r = reference((True, False), (), ())
        r['exact_orientation']['passed'] = []
        r['exact_orientation_fields']['valid'] = []
        p = protected_surface_constraint_policy(r)
        np.testing.assert_array_equal(p['hard_mask'], [True, False])
        self.assertEqual(p['orientation_count'], 0)

    def test_detached_masks_do_not_change_with_source(self):
        r = reference()
        p = protected_surface_constraint_policy(r)
        before = copy.deepcopy(p)
        r['literal_surface_pass'][:] = False
        r['exact_orientation_fields']['valid'][:] = False
        r['surface_margin'][:] = 9.
        for key in ('hard_mask', 'offset', 'slack_coefficients',
                    'protected_literal_surface_rows', 'protected_orientation_eligible_rows'):
            np.testing.assert_array_equal(p[key], before[key])

    def test_bad_pass_mask_refused(self):
        r = reference(); r['literal_surface_pass'][0] = False
        with self.assertRaises(ValueError): protected_surface_constraint_policy(r)

    def test_pass_without_eligibility_refused(self):
        r = reference(); r['exact_orientation_fields']['valid'][0] = False
        with self.assertRaises(ValueError): protected_surface_constraint_policy(r)

    def test_nonboolean_and_mismatched_masks_refused(self):
        for key, value in [('literal_surface_pass', [1, 0, 1, 0, 0]),
                           ('literal_surface_pass', [True]),
                           ('literal_surface_pass', [True, False, np.nan, False, False])]:
            with self.subTest(value=value):
                r = reference(); r[key] = value
                with self.assertRaises(ValueError): protected_surface_constraint_policy(r)

    def test_nonfinite_or_wrong_shape_surface_refused(self):
        for value in [np.zeros(16), np.zeros((17, 1)), np.full(17, np.nan), np.full(17, np.inf)]:
            with self.subTest(value=value):
                r = reference(); r['surface_margin'] = value
                with self.assertRaises(ValueError): protected_surface_constraint_policy(r)

    def test_hard_infeasible_reference_refused_without_mask_filter(self):
        r = reference(); r['surface_margin'][0] = -.1
        with self.assertRaisesRegex(ValueError, 'violates'): protected_surface_constraint_policy(r)


if __name__ == '__main__': unittest.main(verbosity=2)
