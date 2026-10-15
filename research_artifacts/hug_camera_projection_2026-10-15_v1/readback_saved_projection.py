"""Independent scalar/source-dtype/rigid-camera readback of frozen v1 evidence.

No simulator import, native reconstruction, renderer, generator, or fitting.
The extra rigid check was requested before inspection; it uses v1's existing
1e-10 compiled-camera arithmetic allowance without changing the v1 gate.
"""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parent
out = root / "independent_readback.json"
if out.exists():
    raise FileExistsError(out)
protocol = json.loads((root / "protocol.json").read_text())
report = json.loads((root / "results.json").read_text())
assert sha(root / "protocol.json") == report["protocol_sha256"]
for filename, expected in protocol["bindings"].items():
    assert sha(filename) == expected, filename

rows = []
rigid_allowance = protocol["tolerances"]["compiled_camera_max_abs"]
for case, measured in zip(protocol["cases"], report["cases"], strict=True):
    assert case["index"] == measured["index"]
    observation = Path(case["observation"])
    native = pickle.loads(Path(case["native"]).read_bytes())
    xyz = np.asarray(native["landmarks_3d"])
    k = np.load(observation / "intrinsics.npy", allow_pickle=False)
    # Exact upstream order: multiply first, then add epsilon to product z.
    product = xyz @ k.T
    upstream = product[:, :2] / (product[:, 2:3] + 1e-8)
    scalar = np.array([[(float(k[0, 0])*float(x) + float(k[0, 1])*float(y) + float(k[0, 2])*float(z)) / (float(z) + 1e-8),
                        (float(k[1, 0])*float(x) + float(k[1, 1])*float(y) + float(k[1, 2])*float(z)) / (float(z) + 1e-8)]
                       for x, y, z in xyz])
    saved = np.asarray(native["landmarks_2d"])
    assert xyz.shape == (21, 3) and saved.shape == (21, 2) and np.isfinite(saved).all()
    t = np.load(observation / "T_world_camera.npy", allow_pickle=False)
    rotation = t[:3, :3]
    orthogonality = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.)
    rigid_pass = orthogonality <= rigid_allowance and determinant_error <= rigid_allowance
    assert rigid_pass
    assert np.array_equal(upstream, saved)
    scalar_error = float(np.max(np.abs(scalar-saved)))
    assert scalar_error <= protocol["tolerances"]["stored_2d_max_abs_px"]
    rows.append({"index": case["index"], "landmarks_3d_dtype": str(xyz.dtype),
                 "intrinsics_dtype": str(k.dtype), "projection_product_dtype": str(product.dtype),
                 "stored_2d_dtype": str(saved.dtype), "upstream_source_formula_exact": True,
                 "independent_scalar_max_abs_px": scalar_error,
                 "camera_rotation_orthogonality_max_abs": orthogonality,
                 "camera_rotation_determinant_abs_error": determinant_error,
                 "camera_rigid_pass": rigid_pass})

summary = {"case_count": len(rows), "protocol_sha256": sha(root / "protocol.json"),
           "results_sha256": sha(root / "results.json"), "readback_source_sha256": sha(__file__),
           "upstream_formula": "proj = points_3d @ K.T; proj[:, :2] / (proj[:, 2:3] + 1e-8)",
           "upstream_formula_exact_case_count": sum(r["upstream_source_formula_exact"] for r in rows),
           "max_independent_scalar_abs_px": max(r["independent_scalar_max_abs_px"] for r in rows),
           "rigid_camera_check": {"additional_check_separate_from_frozen_v1": True,
               "allowance": rigid_allowance, "allowance_source": "existing frozen compiled_camera_max_abs; coordinator explicitly requested rigid check before readback",
               "pass_count": sum(r["camera_rigid_pass"] for r in rows),
               "max_orthogonality_abs": max(r["camera_rotation_orthogonality_max_abs"] for r in rows),
               "max_determinant_abs_error": max(r["camera_rotation_determinant_abs_error"] for r in rows)},
           "parts": {}, "cases": rows, "physics_steps": 0, "static_forwards": 0,
           "source_bindings_unchanged": True, "independent_observed_2d_ground_truth": False}
for part in ("landmarks_3d", "mesh_vertices"):
    first = report["cases"][0]["parts"][part]
    summary["parts"][part] = {
        "maximum_errors": {key: max(c["parts"][part][key] for c in report["cases"])
                           for key, value in first.items() if isinstance(value, float)},
        "support_totals": {key: sum(c["parts"][part]["image_support"][key] for c in report["cases"])
                           for key, value in first["image_support"].items()
                           if isinstance(value, int) and not isinstance(value, bool)}}
summary["maximum_errors"] = {key: max(c["errors"][key] for c in report["cases"])
                             for key in report["cases"][0]["errors"]}
for filename, expected in protocol["bindings"].items():
    assert sha(filename) == expected, filename
out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
