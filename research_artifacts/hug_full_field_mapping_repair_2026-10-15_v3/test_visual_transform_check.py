"""Manufactured visual arithmetic controls, no actual model or native scene."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("manufactured_visual_transform_check", Path(__file__).with_name("visual_transform_check.py"))
check = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(check)


def synthetic():
    # Closed tetrahedron, nonzero signed volume; manufactured nonidentity poses.
    vertices = np.array([[0., 0., 0.], [.05, 0., 0.], [0., .04, 0.], [0., 0., .03]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    cg, sg = np.cos(.31), np.sin(.31); co, so = np.cos(-.23), np.sin(-.23)
    g = np.array([[cg, -sg, 0.], [sg, cg, 0.], [0., 0., 1.]])
    r = np.array([[co, 0., so], [0., 1., 0.], [-so, 0., co]])
    model = SimpleNamespace(geom_dataid=np.array([0]), mesh_vertadr=np.array([0]), mesh_vertnum=np.array([4]),
        mesh_faceadr=np.array([0]), mesh_facenum=np.array([4]), mesh_vert=vertices, mesh_face=faces)
    data = SimpleNamespace(geom_xmat=g.reshape(1, 9), geom_xpos=np.array([[.2, -.1, .3]]))
    return dict(meta=dict(disabled_visual_geoms=[dict(id=0)]), model=model, data=data,
                object_position=np.array([.19, -.11, .29]), object_rotation=r)


def original_visual(meta, model, data, object_position, object_rotation):
    v = model.mesh_vert.astype(float); f = model.mesh_face.astype(int)
    vertices = (v @ data.geom_xmat[0].reshape(3, 3).T + data.geom_xpos[0] - object_position) @ object_rotation
    triangles = vertices[f]; normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    volume = float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6)
    return triangles, normals, dict(signed_volume_m3=volume)


def run(tmp_path, original=original_visual, **overrides):
    inputs = synthetic(); inputs.update(overrides)
    return check.checked_visual_mesh(original, **inputs, evidence_dir=tmp_path, case_index=20)


def test_full_vertex_and_volume_readback_keeps_original_return_objects(tmp_path):
    inputs = synthetic(); returned = []
    def observed(*args):
        value = original_visual(*args); returned.append(value); warnings.warn("manufactured finite warning", RuntimeWarning); return value
    tri, norm, metadata, binding = run(tmp_path, observed)
    assert tri is returned[0][0] and norm is returned[0][1] and metadata is returned[0][2]
    receipt = check.verify_receipt(binding)
    assert receipt["original_output_retained"] and receipt["all_vertices_and_coordinates_checked"]
    assert receipt["first_product_coordinates"] == receipt["second_product_coordinates"] == 12
    assert receipt["coordinate_comparison_limit_m"] == 1e-14
    assert receipt["volume_comparison_limit_m3"] == 1e-14
    assert not receipt["scientific_acceptance_allowance"]
    assert receipt["original_callable_warnings"] == [dict(category="RuntimeWarning", message="manufactured finite warning", count=1)]
    assert max(receipt["maximum_coordinate_differences_m"].values()) < 1e-14
    with np.load(binding["operands_and_outputs"]["path"], allow_pickle=False) as z:
        assert z["scalar_object_vertices"].shape == (4, 3)
        assert z["raw_compiled_vertices"].dtype == np.float32
        np.testing.assert_array_equal(z["original_triangles"], tri)


@pytest.mark.parametrize("change", ["single_coordinate", "volume", "sign", "normal_nonfinite", "tri_nonfinite"])
def test_outlier_or_bad_original_output_refuses_with_partial_hashed_evidence(tmp_path, change):
    def bad(*args):
        tri, norm, metadata = original_visual(*args)
        if change == "single_coordinate":tri[-1, -1, -1] += 2e-14
        elif change == "volume":metadata["signed_volume_m3"] += 2e-14
        elif change == "sign":metadata["signed_volume_m3"] *= -1
        elif change == "normal_nonfinite":norm[0, 0] = np.nan
        else:tri[0, 0, 0] = np.inf
        return tri, norm, metadata
    with pytest.raises(check.VisualTransformRefused) as caught:run(tmp_path, bad)
    binding = caught.value.receipt_binding
    assert check._sha(binding["path"]) == binding["sha256"]
    receipt = json.loads(Path(binding["path"]).read_text())
    assert receipt["status"] == "HARD_STOP" and receipt["failure"]
    assert not receipt["original_output_retained"]
    assert check._sha(binding["operands_and_outputs"]["path"]) == binding["operands_and_outputs"]["sha256"]
    with pytest.raises(check.VisualTransformRefused):check.verify_receipt(binding)


def test_captured_operand_mutation_refuses(tmp_path):
    def bad(meta, model, data, pos, rotation):
        value = original_visual(meta, model, data, pos, rotation)
        data.geom_xpos[0, 0] += 1e-6
        return value
    with pytest.raises(check.VisualTransformRefused, match="changed its captured operands"):run(tmp_path, bad)


def test_overflow_is_not_a_finite_warning_exception(tmp_path):
    inputs = synthetic()
    inputs["model"].mesh_vert = np.full((4, 3), 1e308)
    inputs["data"].geom_xmat = np.full((1, 9), 10.)
    with pytest.raises(check.VisualTransformRefused) as caught:
        check.checked_visual_mesh(original_visual, **inputs, evidence_dir=tmp_path, case_index=20)
    receipt = json.loads(Path(caught.value.receipt_binding["path"]).read_text())
    assert receipt["status"] == "HARD_STOP"
    assert receipt["original_callable_warnings"]
    with np.load(receipt["operands_and_outputs"]["path"], allow_pickle=False) as z:
        assert z["vertices"].shape == (4, 3)


def test_warning_before_original_exception_preserved(tmp_path):
    def bad(*args):
        warnings.warn("manufactured before refusal", RuntimeWarning)
        raise ArithmeticError("manufactured original failure")
    with pytest.raises(check.VisualTransformRefused) as caught:run(tmp_path, bad)
    receipt = json.loads(Path(caught.value.receipt_binding["path"]).read_text())
    assert receipt["original_callable_warnings"][0]["message"] == "manufactured before refusal"
    assert receipt["failure"]["type"] == "ArithmeticError"


@pytest.mark.parametrize("kind", ["receipt", "operand"])
def test_outer_receipt_and_operand_byte_tampering_refused(tmp_path, kind):
    *_, binding = run(tmp_path)
    target = binding["path"] if kind == "receipt" else binding["operands_and_outputs"]["path"]
    with Path(target).open("ab") as handle:handle.write(b"tampered")
    with pytest.raises(check.VisualTransformRefused, match="bytes changed"):check.verify_receipt(binding)


def test_no_overwrite_or_silent_mesh_sampling(tmp_path):
    run(tmp_path)
    with pytest.raises(FileExistsError):run(tmp_path)
    inputs = synthetic(); inputs["model"].mesh_vertnum[0] = check.MAX_VERTICES + 1
    with pytest.raises(check.VisualTransformRefused, match="finite work domain"):
        check.checked_visual_mesh(original_visual, **inputs, evidence_dir=tmp_path / "large", case_index=20)


def test_zero_extra_forward_or_distance_apis(tmp_path):
    # The helper has no dependency on a MuJoCo/project module. The manufactured
    # model exposes only literal array operands and cannot forward or query.
    *_, binding = run(tmp_path)
    receipt = check.verify_receipt(binding)
    assert receipt["additional_forward_calls"] == receipt["additional_distance_calls"] == 0
