"""Independent arithmetic readback of the ORIGINAL object visual transform.

The original visual_mesh callable is invoked once and its exact return objects
are retained for the solver. The two vertex-by-3x3 products are also evaluated
with optimize=False einsum and scalar math.fsum at EVERY vertex/coordinate.
Coordinate comparisons use 1e-14 m; signed-volume comparisons use 1e-14 m^3
and require the same nonzero sign. These are numerical comparisons only, never
geometry, native-fidelity, eligibility or fingertip acceptance allowances.

No MuJoCo/project imports, model construction, forward calls or distance calls.
Finite work is limited to 250,000 vertices and 500,000 faces; larger inputs
refuse rather than selecting a subset. Actual inputs/outputs are saved once,
including partial evidence on an arithmetic refusal, with outer hash bindings.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import warnings

import numpy as np

COORDINATE_LIMIT_M = 1e-14
VOLUME_LIMIT_M3 = 1e-14
MAX_VERTICES = 250_000
MAX_FACES = 500_000
SCHEMA = "hug_original_visual_transform_arithmetic_v3"


class VisualTransformRefused(RuntimeError):
    def __init__(self, reason, receipt_binding=None):
        super().__init__(reason)
        self.receipt_binding = receipt_binding


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _finite(value, name):
    a = np.asarray(value)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite " + name)
    return a


def _measured(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value, error = None, None
        try:
            with np.errstate(divide="warn", over="warn", invalid="warn", under="ignore"):
                value = fn()
        except Exception as exc:
            error = exc
    notices = Counter((w.category.__name__, str(w.message)) for w in caught)
    return value, [dict(category=k[0], message=k[1], count=v) for k, v in sorted(notices.items())], error


def _inputs(meta, model, data, object_position, object_rotation):
    visuals = meta["disabled_visual_geoms"]
    if len(visuals) != 1 or type(visuals[0]["id"]) is not int:
        raise ValueError("Exactly one integer visual geom ID required")
    g = visuals[0]["id"]
    if not 0 <= g < len(model.geom_dataid):
        raise ValueError("Visual geom ID out of range")
    mid = int(model.geom_dataid[g])
    if not 0 <= mid < len(model.mesh_vertadr):
        raise ValueError("Visual geometry must reference one compiled mesh")
    va, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    if not 0 < nv <= MAX_VERTICES or not 0 < nf <= MAX_FACES:
        raise ValueError("Complete visual mesh exceeds declared finite work domain")
    raw_vertices = np.array(model.mesh_vert[va:va + nv], copy=True)
    raw_faces = np.array(model.mesh_face[fa:fa + nf], copy=True)
    vertices = raw_vertices.astype(float); faces = raw_faces.astype(int)
    if vertices.shape != (nv, 3) or faces.shape != (nf, 3):
        raise ValueError("Malformed compiled visual mesh")
    if np.any(faces < 0) or np.any(faces >= nv) or not np.array_equal(raw_faces, faces):
        raise ValueError("Visual face indices are not literal valid vertex indices")
    arrays = dict(raw_compiled_vertices=raw_vertices, raw_compiled_faces=raw_faces,
        vertices=vertices, faces=faces,
        geom_rotation=np.array(data.geom_xmat[g], dtype=float, copy=True).reshape(3, 3),
        geom_position=np.array(data.geom_xpos[g], dtype=float, copy=True),
        object_position=np.array(object_position, dtype=float, copy=True),
        object_rotation=np.array(object_rotation, dtype=float, copy=True))
    if any(arrays[n].shape != shape for n, shape in (("geom_position", (3,)), ("object_position", (3,)), ("object_rotation", (3, 3)))):
        raise ValueError("Malformed original visual transform operands")
    for name, value in arrays.items():_finite(value, name)
    return arrays, dict(geom_id=g, mesh_id=mid, vertex_count=nv, face_count=nf)


def _scalar_product(a, b):
    """Three products per coordinate, full matrix; no BLAS or sampling."""
    out = np.empty((len(a), 3), dtype=float)
    for row in range(len(a)):
        for col in range(3):
            out[row, col] = math.fsum(float(a[row, k]) * float(b[k, col]) for k in range(3))
    return _finite(out, "scalar three-product sum")


def _volume_scalar(triangles):
    def determinant(t):
        a, b, c = t
        return math.fsum((
            float(a[0]) * math.fsum((float(b[1]) * float(c[2]), -float(b[2]) * float(c[1]))),
            float(a[1]) * math.fsum((float(b[2]) * float(c[0]), -float(b[0]) * float(c[2]))),
            float(a[2]) * math.fsum((float(b[0]) * float(c[1]), -float(b[1]) * float(c[0])))))
    value = math.fsum(determinant(t) for t in triangles) / 6.
    if not math.isfinite(value):raise ValueError("Nonfinite scalar signed volume")
    return value


def _difference(a, b, limit, name):
    a, b = _finite(a, name + " first"), _finite(b, name + " second")
    if a.shape != b.shape:raise ValueError("Shape mismatch in " + name)
    with np.errstate(over="raise", invalid="raise"):
        maximum = float(np.max(np.abs(a - b), initial=0.))
    if not math.isfinite(maximum) or maximum > limit:
        raise ValueError(name + " numerical comparison exceeds declared limit")
    return maximum


def checked_visual_mesh(original_callable, meta, model, data, object_position, object_rotation, *, evidence_dir, case_index):
    """Return original (tri, norm, metadata), then a path/hash receipt binding."""
    if type(case_index) is not int:raise ValueError("Literal case index required")
    out = Path(evidence_dir); out.mkdir(parents=True, exist_ok=True)
    stem = f"case_{case_index}_visual_transform"
    npz, report = out / (stem + ".npz"), out / (stem + ".json")
    if npz.exists() or report.exists():raise FileExistsError("Visual arithmetic evidence is append-only")
    arrays, info = {}, {}
    receipt = dict(schema_version=SCHEMA, case_index=case_index, status="HARD_STOP",
        coordinate_comparison_limit_m=COORDINATE_LIMIT_M, volume_comparison_limit_m3=VOLUME_LIMIT_M3,
        maximum_vertices=MAX_VERTICES, maximum_faces=MAX_FACES, original_output_retained=False,
        numerical_comparison_only=True, scientific_acceptance_allowance=False,
        additional_forward_calls=0, additional_distance_calls=0, failure=None)
    returned = None; error = None
    try:
        arrays, info = _inputs(meta, model, data, object_position, object_rotation)
        source = inspect.getsourcefile(original_callable)
        if source is None or not Path(source).is_file():raise ValueError("Original visual callable source must be identifiable")
        receipt["original_callable"] = dict(source_path=str(Path(source).resolve()), source_sha256=_sha(source),
            qualified_name=original_callable.__qualname__)
        returned, notices, original_error = _measured(lambda: original_callable(meta, model, data, object_position, object_rotation))
        receipt["original_callable_warnings"] = notices
        if original_error is not None:raise original_error
        after, after_info = _inputs(meta, model, data, object_position, object_rotation)
        if after_info != info or any(not np.array_equal(arrays[k], after[k]) for k in arrays):
            raise ValueError("Original visual callable changed its captured operands")
        if not isinstance(returned, tuple) or len(returned) != 3:raise ValueError("Original visual return contract changed")
        tri, norm, original_metadata = returned
        arrays["original_triangles"] = np.array(tri, dtype=float, copy=True)
        arrays["original_normals"] = np.array(norm, dtype=float, copy=True)
        if arrays["original_triangles"].shape != (info["face_count"], 3, 3) or arrays["original_normals"].shape != (info["face_count"], 3):
            raise ValueError("Original triangle/normal shape changed")
        _finite(arrays["original_triangles"], "original triangles"); _finite(arrays["original_normals"], "original normals")
        original_volume = float(original_metadata["signed_volume_m3"])
        if not math.isfinite(original_volume) or original_volume == 0:raise ValueError("Original signed volume must be finite and nonzero")
        receipt["original_signed_volume_m3"] = original_volume
        v, g, t, p, r, f = (arrays[k] for k in ("vertices", "geom_rotation", "geom_position", "object_position", "object_rotation", "faces"))
        def replay():
            first = v @ g.T
            centered = first + t - p
            return first, centered, centered @ r
        replayed, replay_warnings, replay_error = _measured(replay)
        receipt["matmul_replay_warnings"] = replay_warnings
        if replay_error is not None:raise replay_error
        for name, value in zip(("matmul_first", "matmul_centered", "matmul_object_vertices"), replayed, strict=True):arrays[name] = _finite(value, name)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            arrays["einsum_first"] = np.einsum("ij,kj->ik", v, g, optimize=False)
            arrays["einsum_centered"] = arrays["einsum_first"] + t - p
            arrays["einsum_object_vertices"] = np.einsum("ij,jk->ik", arrays["einsum_centered"], r, optimize=False)
            arrays["scalar_first"] = _scalar_product(v, g.T)
            arrays["scalar_centered"] = arrays["scalar_first"] + t - p
            arrays["scalar_object_vertices"] = _scalar_product(arrays["scalar_centered"], r)
        comparisons = {}
        for stage in ("first", "centered", "object_vertices"):
            for left, right in (("matmul", "einsum"), ("matmul", "scalar"), ("einsum", "scalar")):
                key = stage + "_" + left + "_vs_" + right
                comparisons[key] = _difference(arrays[left + "_" + stage], arrays[right + "_" + stage], COORDINATE_LIMIT_M, key)
        for method in ("matmul", "einsum", "scalar"):
            candidate = arrays[method + "_object_vertices"][f]
            comparisons["original_triangles_vs_" + method] = _difference(arrays["original_triangles"], candidate, COORDINATE_LIMIT_M, "original triangles vs " + method)
        receipt["maximum_coordinate_differences_m"] = comparisons
        volumes = {method: _volume_scalar(arrays[method + "_object_vertices"][f]) for method in ("matmul", "einsum", "scalar")}
        volumes["original_triangles_scalar"] = _volume_scalar(arrays["original_triangles"])
        receipt["independent_signed_volumes_m3"] = volumes
        for name, volume in volumes.items():
            if math.copysign(1., volume) != math.copysign(1., original_volume) or volume == 0:
                raise ValueError("Original signed volume sign changed in " + name)
            _difference(np.asarray(volume), np.asarray(original_volume), VOLUME_LIMIT_M3, "signed volume " + name)
        receipt.update(status="VERIFIED_ORIGINAL_VISUAL_TRANSFORM_ARITHMETIC", original_output_retained=True, all_vertices_and_coordinates_checked=True,
            first_product_coordinates=3 * len(v), second_product_coordinates=3 * len(v), all_face_coordinates_compared=True,
            maximum_signed_volume_difference_m3=max(abs(value - original_volume) for value in volumes.values()))
    except Exception as exc:
        error = exc; receipt["failure"] = dict(type=type(exc).__name__, message=str(exc))
    receipt.update(mesh=info, array_hashes={k: _array_hash(v) for k, v in arrays.items()})
    with npz.open("xb") as handle:np.savez_compressed(handle, **arrays)
    receipt["operands_and_outputs"] = dict(path=str(npz.resolve()), sha256=_sha(npz))
    with report.open("x") as handle:json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False); handle.write("\n")
    binding = dict(path=str(report.resolve()), sha256=_sha(report), schema_version=SCHEMA,
        operands_and_outputs=dict(receipt["operands_and_outputs"]))
    if error is not None:raise VisualTransformRefused(str(error), binding) from error
    return (*returned, binding)


def verify_receipt(binding):
    """Read-only outer byte/array identity verification; no geometry replay."""
    if not isinstance(binding, dict) or _sha(binding["path"]) != binding["sha256"]:
        raise VisualTransformRefused("Visual arithmetic receipt bytes changed")
    receipt = json.loads(Path(binding["path"]).read_text())
    if receipt["schema_version"] != SCHEMA or receipt["status"] != "VERIFIED_ORIGINAL_VISUAL_TRANSFORM_ARITHMETIC":
        raise VisualTransformRefused("Visual arithmetic receipt is not complete")
    reference = receipt["operands_and_outputs"]
    if reference != binding["operands_and_outputs"] or _sha(reference["path"]) != reference["sha256"]:
        raise VisualTransformRefused("Visual arithmetic operand bytes changed")
    with np.load(reference["path"], allow_pickle=False) as arrays:
        if set(arrays.files) != set(receipt["array_hashes"]):raise VisualTransformRefused("Visual arithmetic array inventory changed")
        if any(_array_hash(arrays[k]) != h for k, h in receipt["array_hashes"].items()):
            raise VisualTransformRefused("Visual arithmetic array values changed")
    return receipt
