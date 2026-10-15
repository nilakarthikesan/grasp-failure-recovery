from pathlib import Path
import difflib
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');H=Path(__file__).resolve().parent
s=(W/'src/grasp_failure_prediction/evaluation/object_assets.py').read_text();(H/'object_assets.py.before').write_text(s)
helper='''def _canonicalize_full_inertia(inertial: ET.Element) -> None:
    """Preserve a validated source tensor in an explicit principal frame.

    MuJoCo's small-matrix eigensolver can stop before a supplied full tensor
    round-trips at double precision. Convert only the copied object's tensor,
    using a symmetric eigensolver and full-precision XML numbers. This changes
    its numerical representation, not its mass or intended inertia tensor.
    Original source bytes and the content hash remain untouched. An existing
    diagonal inertia and its explicitly declared orientation are left alone.
    """
    if "fullinertia" not in inertial.attrib:
        return
    values = np.asarray([float(v) for v in inertial.attrib["fullinertia"].split()])
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("full object inertia must contain six finite values")
    xx, yy, zz, xy, xz, yz = values
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    principal, axes = np.linalg.eigh(tensor)
    if not np.all(np.isfinite(principal)) or np.any(principal <= 0):
        raise ValueError("full object inertia must be positive definite")
    order = np.argsort(principal)[::-1]
    principal, axes = principal[order], axes[:, order]
    # Eigenvector signs are arbitrary. A proper frame is required for a
    # quaternion; flipping one eigenvector leaves the reconstructed tensor.
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(axes).reshape(9))
    del inertial.attrib["fullinertia"]
    inertial.set("diaginertia", " ".join(format(float(x), ".17g") for x in principal))
    inertial.set("quat", " ".join(format(float(x), ".17g") for x in quaternion))


'''
s=s.replace('Native XML mesh and inertial frames are preserved.', 'Native XML mesh frames and inertia tensors are preserved.')
s=s.replace('def append_object_geometry(',helper+'def append_object_geometry(')
s=s.replace('    body.set("name", "object")\n', '    body.set("name", "object")\n    inertial = body.find("inertial")\n    assert inertial is not None\n    _canonicalize_full_inertia(inertial)\n')
(H/'object_assets.py.proposed').write_text(s)
t=(W/'tests/test_object_assets.py').read_text();(H/'test_object_assets.py.before').write_text(t)
t=t.replace('for field in ("body_mass", "body_inertia", "body_ipos"):', 'for field in ("body_mass", "body_ipos"):')
t=t.replace('    # The existing actuation conversion serializes a compiled model through\n    # MuJoCo XML, which rounds quaternions to about six significant figures.\n    np.testing.assert_allclose(runner.model.body_iquat[body], native.body_iquat[1], atol=1e-6)', '''    # Principal-frame eigenvector signs are not unique. Compare the intended
    # source tensor, rather than requiring an arbitrary eigenframe quaternion.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, runner.model.body_iquat[body])
    actual_tensor = rotation.reshape(3, 3) @ np.diag(runner.model.body_inertia[body]) @ rotation.reshape(3, 3).T
    source_tensor = np.array([[.00025, .00001, -.00002],
                              [.00001, .00035, .000005],
                              [-.00002, .000005, .00030]])
    np.testing.assert_allclose(actual_tensor, source_tensor, atol=1e-12, rtol=0)''')
append='''

@pytest.mark.parametrize("tensor", [
    np.array([[.000691456, -3.51381e-6, 3.23894e-6],
              [-3.51381e-6, .000675951, 4.26748e-6],
              [3.23894e-6, 4.26748e-6, .000180772]]),
    np.array([[.003, .0002, -.0001], [.0002, .004, .0003], [-.0001, .0003, .005]]),
    np.eye(3) * .003,
    np.diag([.003, .003, .004]),
    np.array([[.003, 0, 0], [0, .00364, -.00048], [0, -.00048, .00336]]),
])
def test_full_inertia_precision_preserves_source_tensor_and_identity(asset, tensor):
    from xml.etree import ElementTree as ET
    from grasp_failure_prediction.evaluation.object_assets import append_object_geometry

    tree = ET.parse(asset)
    inertial = tree.getroot().find("worldbody/body/inertial")
    values = tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    inertial.set("fullinertia", " ".join(format(float(x), ".17g") for x in values))
    tree.write(asset)
    original = asset.read_bytes()
    geometry = load_object_geometry(asset)
    original_hash = geometry.content_hash
    scene = ET.Element("mujoco")
    ET.SubElement(scene, "compiler", angle="radian")
    body = append_object_geometry(scene, ET.SubElement(scene, "worldbody"), geometry)
    copied = body.find("inertial")
    assert "fullinertia" not in copied.attrib
    assert copied.get("mass") == inertial.get("mass")
    assert copied.get("pos") == inertial.get("pos")
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[1])
    actual = rotation.reshape(3, 3) @ np.diag(model.body_inertia[1]) @ rotation.reshape(3, 3).T
    np.testing.assert_allclose(actual, tensor, atol=1e-12, rtol=0)
    assert asset.read_bytes() == original == geometry.source_xml
    assert load_object_geometry(asset).content_hash == original_hash


def test_diagonal_inertia_precision_keeps_explicit_frame(asset):
    from xml.etree import ElementTree as ET
    from grasp_failure_prediction.evaluation.object_assets import append_object_geometry

    tree = ET.parse(asset)
    inertial = tree.getroot().find("worldbody/body/inertial")
    del inertial.attrib["fullinertia"]
    inertial.set("diaginertia", ".00025 .00035 .00030")
    inertial.set("quat", "0.9238795325112867 0 0 0.3826834323650898")
    tree.write(asset)
    geometry = load_object_geometry(asset)
    scene = ET.Element("mujoco")
    copied = append_object_geometry(scene, ET.SubElement(scene, "worldbody"), geometry)
    assert copied.find("inertial").attrib == inertial.attrib


@pytest.mark.parametrize("full", ["-.001 .001 .001 0 0 0", ".001 .001 .003 0 0 0", "nan .001 .001 0 0 0"])
def test_inertia_precision_does_not_bypass_original_physical_validation(asset, full):
    from xml.etree import ElementTree as ET

    tree = ET.parse(asset)
    tree.getroot().find("worldbody/body/inertial").set("fullinertia", full)
    tree.write(asset)
    with pytest.raises(ValueError):
        load_object_geometry(asset)
'''
t+=append;(H/'test_object_assets.py.proposed').write_text(t)
diff=''.join(difflib.unified_diff((H/'object_assets.py.before').read_text().splitlines(True),s.splitlines(True),fromfile='a/src/grasp_failure_prediction/evaluation/object_assets.py',tofile='b/src/grasp_failure_prediction/evaluation/object_assets.py'))
diff+=''.join(difflib.unified_diff((H/'test_object_assets.py.before').read_text().splitlines(True),t.splitlines(True),fromfile='a/tests/test_object_assets.py',tofile='b/tests/test_object_assets.py'))
(H/'production.patch').write_text(diff)
