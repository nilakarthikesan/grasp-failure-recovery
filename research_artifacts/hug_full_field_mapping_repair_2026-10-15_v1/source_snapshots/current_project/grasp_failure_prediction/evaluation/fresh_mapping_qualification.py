"""Observational helpers for a finite static retargeter qualification.

The solver wrapper records its existing outputs/statuses without modifying the
objective, initialization or clamping. Gate values are supplied by the frozen
driver; this module does not select defaults or perform physical execution.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np


def capture_solve(retargeter, grasp):
    """Observe all raw SeqRetargeting returns and actual optimizer statuses.

    The production pose is still returned by the unmodified project retargeter.
    A refusal retains the partial raw trace and exception, rather than being
    converted to a fabricated pose or physical failure label.
    """
    sequence = retargeter._retargeting
    original = sequence.retarget
    raw, statuses = [], []
    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        raw.append(np.asarray(value, dtype=float).copy())
        reader = getattr(sequence.optimizer.opt, 'last_optimize_result', None)
        status = reader() if callable(reader) else None
        statuses.append(int(status) if isinstance(status, (int, np.integer)) and not isinstance(status, (bool, np.bool_)) else None)
        return value
    sequence.retarget = observe
    try:
        pose = retargeter.retarget(grasp)
    except Exception as exc:
        return None, dict(status='refused', exception_type=type(exc).__name__, message=str(exc),
                          raw_refinement_qpos_rad=[v.tolist() for v in raw], optimizer_status_codes=statuses)
    finally:
        sequence.retarget = original
    provenance = asdict(pose.retargeting_provenance) if pose.retargeting_provenance is not None else None
    return pose, dict(status='returned', joint_names=list(pose.joint_names), qpos_rad=pose.qpos.tolist(),
                      raw_refinement_qpos_rad=[v.tolist() for v in raw], optimizer_status_codes=statuses,
                      max_final_clipping_rad=float(np.max(abs(raw[-1]-pose.qpos))) if raw else None,
                      provenance=provenance)


def compare_commands(reference, candidate, tolerance):
    """Compare named output commands; refusals remain explicit non-passes."""
    if reference['status'] != 'returned' or candidate['status'] != 'returned':
        return dict(comparable=False, passed=False, bit_exact=False, max_difference_rad=None)
    if reference['joint_names'] != candidate['joint_names']:
        return dict(comparable=False, passed=False, bit_exact=False, max_difference_rad=None)
    a, b = np.asarray(reference['qpos_rad']), np.asarray(candidate['qpos_rad'])
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(comparable=False, passed=False, bit_exact=False, max_difference_rad=None)
    difference = float(np.max(abs(a-b)))
    return dict(comparable=True, passed=difference <= tolerance,
                bit_exact=bool(np.array_equal(a,b)), max_difference_rad=difference)


def independent_local_operator_landmarks(grasp):
    """Scalar form of the declared transpose-rotation MANO/operator convention.

    Float-valued source rotation orthogonality defects are measured separately
    by the driver; this does not silently orthogonalize those source values.
    """
    output = np.empty((21,3))
    basis = np.array([[0.,-1.,0.],[0.,0.,1.],[-1.,0.,0.]])
    for i in range(21):
        local = [sum((grasp.landmarks_3d[i,k]-grasp.T_camera_wrist[k,3])*
                     grasp.T_camera_wrist[k,j] for k in range(3)) for j in range(3)]
        output[i] = [sum(basis[j,k]*local[k] for k in range(3)) for j in range(3)]
    return output


def settings_gate(record, expected):
    provenance = record.get('provenance')
    statuses = record.get('optimizer_status_codes', [])
    if record.get('status') != 'returned' or provenance is None:
        return False
    fields = ('profile_id','vector_scaling','low_pass_filter_enabled',
              'temporal_norm_delta','reset_before_proposal','refinement_calls')
    return bool(all(provenance.get(key) == expected[key] for key in fields)
                and provenance.get('config_sha256') == expected['config_sha256']
                and len(record['raw_refinement_qpos_rad']) == expected['refinement_calls']
                and len(statuses) == expected['refinement_calls']
                and all(type(status) is int and status > 0 for status in statuses))
