"""Conservative native-proximity preflight for ContactGen imports.

This rejects a provably empty native 2 mm proximity set. Overlapping bounding
boxes are inconclusive: this helper never certifies contact or mapping fidelity.
All rejection comparisons use exact rationals of the supplied binary64 vertex
coordinates. The displayed square root is a diagnostic, not the decision.
"""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


CONTACTGEN_NATIVE_PROXIMITY_M = Fraction(1, 500)


@dataclass(frozen=True)
class NativeSeparationCheck:
    squared_lower_bound_m2: Fraction
    threshold_m: Fraction

    @property
    def certified_separated(self) -> bool:
        return self.squared_lower_bound_m2 > self.threshold_m**2

    @property
    def status(self) -> str:
        return "certified_separated" if self.certified_separated else "inconclusive"

    @property
    def displayed_lower_bound_m(self) -> float:
        return math.sqrt(float(self.squared_lower_bound_m2))


def native_object_separation_check(
    native_vertices_object_m,
    object_vertices_object_m,
    *,
    threshold_m: Fraction = CONTACTGEN_NATIVE_PROXIMITY_M,
) -> NativeSeparationCheck:
    """Bound separation of every triangle spanned by the supplied vertices.

    A triangle lies in its vertex bounding box, so separation of the two boxes
    is a lower bound on all triangle-to-triangle distances. This is sufficient
    for rejection, not sufficient for admission when the boxes overlap.
    """
    if not isinstance(threshold_m, Fraction) or threshold_m <= 0:
        raise ValueError("threshold_m must be a positive exact Fraction")
    arrays = []
    for name, values in (
        ("native_vertices_object_m", native_vertices_object_m),
        ("object_vertices_object_m", object_vertices_object_m),
    ):
        value = np.asarray(values, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a nonempty finite Nx3 array")
        arrays.append(value)
    a, b = arrays
    squared = Fraction(0)
    for axis in range(3):
        alo = Fraction.from_float(float(a[:, axis].min()))
        ahi = Fraction.from_float(float(a[:, axis].max()))
        blo = Fraction.from_float(float(b[:, axis].min()))
        bhi = Fraction.from_float(float(b[:, axis].max()))
        gap = max(Fraction(0), blo - ahi, alo - bhi)
        squared += gap * gap
    return NativeSeparationCheck(squared, threshold_m)
