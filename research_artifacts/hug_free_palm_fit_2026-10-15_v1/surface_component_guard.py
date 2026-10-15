"""Frozen, native-relative direction selection for static development candidates.

This guard adds no mapping acceptance claim. It preserves every supplied part,
site and positive weight, uses the existing invalid-site residual 2, and refuses
both weighted direction-SSE increases and original-valid site invalidation.
No missing mass is redistributed and no native negative score is replaced.
"""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np

from grasp_failure_prediction.evaluation.constrained_surface_refinement import (
    DIRECTION_SCORE_ENDPOINT_ALLOWANCE,
)


@dataclass(frozen=True)
class _Part:
    name: str
    available: bool
    native: np.ndarray
    weights: np.ndarray
    baseline_valid: np.ndarray
    baseline_sse: float


def _array(value, name):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    result.setflags(write=False)
    return result


def _fields(entry, native, weights):
    if not isinstance(entry, Mapping) or entry.get("available") is not True:
        raise ValueError("available direction part missing or marked unavailable")
    score = _array(entry.get("candidate_score"), "candidate_score")
    valid = np.asarray(entry.get("candidate_valid"))
    if score.shape != native.shape or valid.shape != native.shape or valid.dtype.kind != "b":
        raise ValueError("candidate score/boolean validity shape differs from frozen sites")
    if np.any(np.abs(score) > 1.0 + DIRECTION_SCORE_ENDPOINT_ALLOWANCE):
        raise ValueError("candidate score exceeds the inherited cosine endpoint allowance")
    if "native_score" in entry and not np.array_equal(_array(entry["native_score"], "native_score"), native):
        raise ValueError("native score targets changed")
    if "weights" in entry and not np.array_equal(_array(entry["weights"], "weights"), weights):
        raise ValueError("fixed site weights changed")
    valid = np.array(valid, dtype=bool, copy=True)
    valid.setflags(write=False)
    delta = np.where(valid, score-native, 2.0)
    with np.errstate(over="raise", invalid="raise"):
        sse = float(np.dot(weights, delta*delta))
    if not np.isfinite(sse):
        raise ValueError("direction SSE is not representable")
    return score, valid, sse


@dataclass(frozen=True)
class FixedDirectionComponentGuard:
    """Construct through freeze_direction_component_guard; input arrays copied.

    assess accepts the existing per-part normal_diagnostics mapping. Each
    available entry supplies candidate_score and a boolean candidate_valid
    array. Optional native_score/weights fields must equal the frozen arrays.
    Malformed candidate evidence returns an explicit refusal, never a reduced
    site denominator. Invalid-site scores must remain finite even though their
    contribution uses the literal residual 2. This is stricter input hygiene,
    not a change to the inherited scientific orientation rule.
    """
    _parts: tuple

    @property
    def reference_part_names(self):
        return tuple(part.name for part in self._parts)

    def assess(self, candidate_by_part):
        reasons, results = [], {}
        if not isinstance(candidate_by_part, Mapping):
            return {"accepted": False, "refusal_reasons": ["candidate_parts_not_a_mapping"],
                    "parts": {}, "scientific_mapping_acceptance": False}
        expected = set(self.reference_part_names)
        extras = set(candidate_by_part)-expected
        if extras:
            reasons.append("unexpected_direction_parts")
        for part in self._parts:
            entry = candidate_by_part.get(part.name)
            if entry is None:
                reasons.append(f"{part.name}:missing_direction_part")
                results[part.name] = {"accepted": False, "evidence_valid": False,
                                      "reason": "missing_direction_part"}
                continue
            if not part.available:
                ok = isinstance(entry, Mapping) and entry.get("available") is False
                if ok:
                    for name in ("candidate_score", "candidate_valid", "native_score", "weights"):
                        if name in entry and np.asarray(entry[name]).size:
                            ok = False
                if not ok:
                    reasons.append(f"{part.name}:unavailable_reference_part_changed")
                results[part.name] = {"accepted": bool(ok), "available": False,
                                      "fixed_site_count": 0, "fixed_weight_mass": 0.0,
                                      "preservation_status": "UNAVAILABLE"}
                continue
            try:
                _, valid, sse = _fields(entry, part.native, part.weights)
            except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
                reasons.append(f"{part.name}:invalid_direction_evidence")
                results[part.name] = {"accepted": False, "evidence_valid": False, "reason": str(exc),
                                      "fixed_site_count": len(part.native),
                                      "fixed_weight_mass": float(part.weights.sum())}
                continue
            invalidated = part.baseline_valid & ~valid
            increased = sse > part.baseline_sse
            if invalidated.any():
                reasons.append(f"{part.name}:original_valid_sites_invalidated")
            if increased:
                reasons.append(f"{part.name}:direction_sse_increased")
            results[part.name] = {"accepted": bool(not increased and not invalidated.any()),
                "available": True, "evidence_valid": True,
                "fixed_site_count": len(part.native), "fixed_weight_mass": float(part.weights.sum()),
                "baseline_direction_sse": part.baseline_sse, "candidate_direction_sse": sse,
                "direction_sse_change": sse-part.baseline_sse,
                "direction_sse_nonincrease": bool(not increased),
                "original_valid_site_count": int(part.baseline_valid.sum()),
                "original_valid_sites_invalidated_count": int(invalidated.sum()),
                "original_valid_sites_invalidated_indices": np.flatnonzero(invalidated).tolist(),
                "candidate_invalid_site_count": int((~valid).sum()),
                "candidate_invalid_fixed_mass": float(part.weights[~valid].sum()),
                "invalid_site_residual": 2.0}
        if not any(part.available for part in self._parts):
            reasons.append("no_available_direction_parts")
        return {"schema": "fixed_direction_component_guard_v1", "accepted": not reasons,
                "refusal_reasons": reasons, "parts": results,
                "sse_increase_allowance": 0.0, "invalid_site_residual": 2.0,
                "weights_renormalized": False, "missing_part_mass_redistributed": False,
                "scientific_mapping_acceptance": False}


def freeze_direction_component_guard(reference_by_part, baseline_by_part):
    """Freeze native_score/weights and original validity/SSE for every part.

    reference entries have available plus native_score and weights when
    available. baseline entries use candidate_score and candidate_valid. All
    arrays are copied; the exact weighted SSE arithmetic is dot(w, delta**2),
    with delta=2 at invalid sites. This separate selector uses zero SSE
    increase allowance. It does not revise historical fitting objectives.
    """
    if (not isinstance(reference_by_part, Mapping) or not reference_by_part
        or not isinstance(baseline_by_part, Mapping)
        or set(reference_by_part) != set(baseline_by_part)):
        raise ValueError("nonempty reference and baseline must have identical part names")
    parts = []
    for name, reference in reference_by_part.items():
        if not isinstance(name, str) or not name or not isinstance(reference, Mapping):
            raise ValueError("named direction reference mappings required")
        available = reference.get("available")
        if not isinstance(available, bool):
            raise ValueError("each reference part requires boolean available")
        baseline = baseline_by_part[name]
        if not available:
            if not isinstance(baseline, Mapping) or baseline.get("available") is not False:
                raise ValueError("unavailable baseline part differs from reference")
            for entry in (reference, baseline):
                for key in ("native_score", "weights", "candidate_score", "candidate_valid"):
                    if key in entry and np.asarray(entry[key]).size:
                        raise ValueError("unavailable direction parts must have no site arrays")
            empty = np.empty(0); empty.setflags(write=False)
            empty_valid = np.empty(0, bool); empty_valid.setflags(write=False)
            parts.append(_Part(name, False, empty, empty, empty_valid, 0.0))
            continue
        native = _array(reference.get("native_score"), "native_score")
        weights = _array(reference.get("weights"), "weights")
        if (not native.size or weights.shape != native.shape or np.any(weights <= 0)
            or np.any(np.abs(native) > 1.0 + DIRECTION_SCORE_ENDPOINT_ALLOWANCE)):
            raise ValueError("fixed native scores and positive same-shape weights required")
        if not np.isfinite(weights.sum()):
            raise ValueError("fixed weight mass is not representable")
        _, valid, sse = _fields(baseline, native, weights)
        parts.append(_Part(name, True, native, weights, valid, sse))
    return FixedDirectionComponentGuard(tuple(parts))
