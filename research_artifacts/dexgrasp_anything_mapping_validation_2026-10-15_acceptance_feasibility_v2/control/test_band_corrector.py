import numpy as np
import pytest
from band_corrector import band_violations


def test_identity_has_no_band_or_occupancy_violation():
    fields = np.array([[0., .0005, .001, .002, .003, .004, .0015, -.001],
                       [.004, .003, .002, .001, .0005, 0., .0025, .001]])
    assert np.all(band_violations(fields, fields) == 0)


def test_added_lost_and_occupancy_errors_are_detected():
    source = np.full((1, 8), .004)
    source[0, 0] = .0001
    target = source.copy()
    target[0, 0] = .003
    target[0, 1] = .0001
    target[0, 7] = -.001
    violations = band_violations(source, target)
    assert np.count_nonzero(violations) == 7
    assert violations[-1] > .001


def test_zero_occupancy_boundary_requires_zero_at_fitting_query():
    source = np.full((1, 8), .004)
    source[0, 7] = 0.
    target = source.copy()
    target[0, 7] = -.001
    assert band_violations(source, target)[-1] == .001


def test_nonfinite_and_missing_fields_refused():
    with pytest.raises(ValueError):
        band_violations(np.zeros((1, 7)), np.zeros((1, 7)))
    source = np.zeros((1, 8)); source[0, 0] = np.nan
    with pytest.raises(ValueError):
        band_violations(source, source)
