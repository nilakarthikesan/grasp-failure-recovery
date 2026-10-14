"""Independent analytic strip-area checks; no grasp assets or evaluation data."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

sys.path.insert(0, '/private/tmp/hug_sampling_checks_2026-10-14')
from cell_intervals import triangle_cells, pair_intervals


def clip(poly, x, keep_lower):
    output = []
    if not len(poly):
        return np.empty((0, 2))
    for i, b in enumerate(poly):
        a = poly[i - 1]
        keep_a = (a[0] <= x) if keep_lower else (a[0] >= x)
        keep_b = (b[0] <= x) if keep_lower else (b[0] >= x)
        if keep_a != keep_b:
            u = (x - a[0]) / (b[0] - a[0])
            output.append(a + u * (b - a))
        if keep_b:
            output.append(b)
    return np.asarray(output).reshape(-1, 2)


def strip_area(triangle, low, high):
    polygon = clip(clip(triangle, low, False), high, True)
    if len(polygon) < 3:
        return 0.
    following = np.roll(polygon, -1, axis=0)
    cross_sum = (polygon[:, 0] * following[:, 1]
                 - polygon[:, 1] * following[:, 0]).sum()
    return abs(cross_sum / 2) * 1e6


def main():
    rng = np.random.default_rng(26410)
    checked = 0
    maximum_violation = 0.
    for k in range(100):
        triangle = rng.uniform(-.006, .006, size=(3, 2))
        ab = triangle[1] - triangle[0]
        ac = triangle[2] - triangle[0]
        cross = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(cross) < 1e-7:
            continue
        tau = rng.uniform(.0003, .002)
        center = rng.uniform(-.002, .002)
        nlow, nhigh = -tau, tau
        mlow, mhigh = center - tau, center + tau
        tri3 = np.column_stack([triangle, np.zeros(3)])[None]
        _, q, weights, radius, _, _, _ = triangle_cells(tri3, .0005)
        bounds = pair_intervals(abs(q[:, 0]), abs(q[:, 0] - center),
                                radius, weights, tau, 1e-10)
        native = strip_area(triangle, nlow, nhigh)
        mapped = strip_area(triangle, mlow, mhigh)
        shared = (strip_area(triangle, max(nlow, mlow), min(nhigh, mhigh))
                  if min(nhigh, mhigh) >= max(nlow, mlow) else 0.)
        known = {'native_area': native, 'mapped_area': mapped,
                 'shared_area': shared, 'union_area': native + mapped - shared,
                 'lost_native_area': native - shared,
                 'new_mapped_area': mapped - shared}
        for name, exact in known.items():
            low, high = bounds[name]['lower_mm2'], bounds[name]['upper_mm2']
            violation = max(low - exact, exact - high, 0)
            maximum_violation = max(maximum_violation, violation)
            assert violation < 1e-8, (k, name, exact, low, high)
        for key, numerator, denominator in (
            ('iou_interval', shared, native + mapped - shared),
            ('native_retention_interval', shared, native),
            ('mapped_correspondence_interval', shared, mapped)
        ):
            value = bounds[key]
            if value['denominator_definitely_nonzero']:
                assert denominator > 0
                assert (value['lower'] - 1e-10 <= numerator / denominator
                        <= value['upper'] + 1e-10), (k, key, value)
            else:
                assert value['lower'] is None and value['upper'] is None
        checked += 1
    script = Path(__file__)
    helper = Path(triangle_cells.__code__.co_filename).resolve()
    report = {
        'schema': 'independent_hug_sampling_review_v1',
        'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
        'source_cell_library_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
        'method': 'Independent exact polygon clipping of native/mapped plane-proximity strips in 100 deterministic manufactured triangle cases; compare all six area set operations and conditional ratios',
        'seed': 26410,
        'case_count': checked,
        'distance_numerical_allowance_m': 1e-10,
        'test_comparison_rounding_allowance_mm2': 1e-8,
        'max_interval_violation_mm2': maximum_violation,
        'passed': True,
        'no_grasp_inputs_or_physics_or_eval_data': True,
        'scope': 'Finite analytic manufactured cross-check; not a uniform floating-point error certificate',
        'publication_reproduction': 'Keep cell_intervals.py beside the review script or add its published directory to sys.path, preserving its hash. No original grasp assets required.'
    }
    script.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
