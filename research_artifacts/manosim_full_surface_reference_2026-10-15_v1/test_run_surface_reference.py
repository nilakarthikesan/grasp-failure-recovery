"""Manufactured orchestration contracts; no compilation, forwards or native query."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('manufactured_locked_surface_runner', HERE/'run_surface_reference.py')
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)


def manufactured_model():
    s = driver.surface(); names = list(reversed(s.JOINT_NAMES))
    qa, va, types, q, v = [], [], [], 0, 0
    for name in names:
        slide = name in s.SLIDE_NAMES; qa.append(q); va.append(v); types.append(2 if slide else 1)
        q += 1 if slide else 4; v += 1 if slide else 3
    model = SimpleNamespace(nq=67, nv=51, nu=51, njnt=19,
        jnt_qposadr=np.array(qa), jnt_dofadr=np.array(va), jnt_type=np.array(types))
    data = SimpleNamespace(qpos=np.zeros(67), qvel=np.ones(51), qacc=np.zeros(51), ctrl=np.zeros(51),
                           warning=SimpleNamespace(number=np.zeros(8, int)), time=0.)
    engine = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_JOINT=1),
                             mjtJoint=SimpleNamespace(mjJNT_SLIDE=2, mjJNT_BALL=1),
                             mj_id2name=lambda m, kind, j: names[j])
    named = {name: .025 if name in s.SLIDE_NAMES else [1., 0., 0., 0.] for name in s.JOINT_NAMES}
    return model, data, engine, dict(named_qpos=named)


def test_exact_saved_qpos_is_staged_by_actual_names_types_and_disjoint_addresses():
    model, data, engine, literal = manufactured_model()
    addresses = driver.stage_saved_named(model, data, literal, engine)
    assert len(addresses) == 19 and np.array_equal(data.qvel, np.zeros(51))
    named = driver.actual_named(model, data, engine)
    for name, value in literal['named_qpos'].items(): np.testing.assert_array_equal(named[name], value)
    # The manufactured model deliberately reverses the source joint order.
    assert addresses[0]['name'] == 'mano_joint_15'
    np.testing.assert_array_equal(data.ctrl, np.zeros(51))


@pytest.mark.parametrize('change', ['dimension', 'joint_type', 'alias', 'overlap', 'negative', 'nonfinite'])
def test_staging_refuses_changed_model_or_literal_state(change):
    model, data, engine, literal = manufactured_model()
    if change == 'dimension': model.nu = 50
    elif change == 'joint_type': model.jnt_type[0] = 3
    elif change == 'alias': engine.mj_id2name = lambda *a: 'some_joint'
    elif change == 'overlap': model.jnt_qposadr[1] = model.jnt_qposadr[0]
    elif change == 'negative': model.jnt_qposadr[0] = -1
    elif change == 'nonfinite': literal['named_qpos']['wrist_tx'] = np.nan
    with pytest.raises(ValueError): driver.stage_saved_named(model, data, literal, engine)


def test_all_budgets_refuse_before_overshoot_and_keep_two_ordered_scans():
    b = driver.Budget(2)
    for key, limit in [('models', 1), ('data', 1), ('forwards', 2)]:
        for _ in range(limit): b.operation(key)
        with pytest.raises(RuntimeError, match='before call'): b.operation(key)
        assert getattr(b, key) == limit
    with pytest.raises(RuntimeError): b.scan(driver.STAGES[1])
    b.scan(driver.STAGES[0]); b.query(3104); b.scan(driver.STAGES[1]); b.query(3104)
    with pytest.raises(RuntimeError): b.query(1)
    with pytest.raises(RuntimeError): b.scan(driver.STAGES[1])
    assert b.tests == 6208 and b.stages == list(driver.STAGES)


@pytest.mark.parametrize('change', ['warning_counter', 'python_warning', 'time', 'qacc', 'ctrl'])
def test_static_forward_numerical_integrity_guard_retains_attempt(change):
    _, data, _, _ = manufactured_model(); data.qvel[:] = 0
    b = driver.Budget(1); calls = []
    def fake(model, d):
        calls.append(True)
        if change == 'warning_counter': d.warning.number[0] += 1
        elif change == 'python_warning': __import__('warnings').warn('manufactured numerical warning')
        elif change == 'time': d.time = .001
        elif change == 'qacc': d.qacc[0] = np.nan
        elif change == 'ctrl': d.ctrl[0] = 1
    with pytest.raises(ValueError, match='integrity'):
        driver.static_forward(None, data, fake, b)
    assert calls == [True] and b.forwards == 1


def test_forward_precheck_refuses_unclean_state_before_actual_call():
    _, data, _, _ = manufactured_model(); data.warning.number[0] = 1
    b = driver.Budget(1); calls = []
    with pytest.raises(ValueError): driver.static_forward(None, data, lambda *a: calls.append(True), b)
    assert not calls and b.forwards == 0


def test_new_xml_retains_skeleton_and_inertias_only_disables_original_geoms(tmp_path):
    meshes = ''.join(f'<mesh name="bone{i}" file="bone{i}.obj"/>' for i in range(16))
    bodies = ''.join(f'<body name="mano_body_{i}"><inertial pos="0 0 0" mass=".025" diaginertia="1 1 1"/>'
                     f'<joint name="j{i}"/><geom type="mesh" mesh="bone{i}" pos=".01 0 0"/></body>' for i in range(16))
    for i in range(16): (tmp_path/f'bone{i}.obj').write_text('manufactured path only; no mesh parsing')
    base = tmp_path/'source.xml'; base.write_text(f'<mujoco><asset>{meshes}</asset><worldbody>{bodies}</worldbody><actuator><motor joint="j0"/></actuator></mujoco>')
    text, receipt = driver.scene_xml(base, tmp_path/'candidate.xml')
    parsed, original = ET.fromstring(text), ET.parse(base).getroot()
    assert len(parsed.findall('.//body/geom')) == 16
    assert all(g.get('contype') == '0' and g.get('conaffinity') == '0' for g in parsed.findall('.//body/geom'))
    assert all(Path(g.get('file')).is_absolute() for g in parsed.findall('./asset/mesh'))
    assert ET.tostring(parsed.find('./actuator')) == ET.tostring(original.find('./actuator'))
    for current, old in zip(parsed.findall('.//body/inertial'), original.findall('.//body/inertial'), strict=True):
        assert current.attrib == old.attrib
    assert receipt['body_joint_inertia_actuator_parameters_changed'] is False
    assert len(parsed.findall('./include')) == 1 and parsed.find('./include').get('file') == str(tmp_path/'candidate.xml')
    base.write_text(base.read_text().replace('<asset>', '<include file="hidden.xml"/><asset>'))
    with pytest.raises(ValueError): driver.scene_xml(base, tmp_path/'candidate.xml')


def test_prior_binding_conflict_cannot_be_overwritten_with_current_source(tmp_path):
    h = driver.helpers(); path = tmp_path/'source.py'; path.write_text('original')
    prior = dict(bindings={str(path): h.sha(path)})
    with pytest.raises(ValueError, match='closures conflict'):
        driver.merged_bindings(prior, dict(bindings={str(path): '0'*64}), [])
    path.write_text('changed')
    with pytest.raises(ValueError, match='historical bound'):
        driver.merged_bindings(prior, dict(bindings={}), [path])


def test_not_qualified_primitive_radius_fails_closed_before_any_native_preparation(tmp_path):
    report = tmp_path/'primitive.json'; report.write_text('{}')
    helper = tmp_path/'helper.py'; helper.write_text('def qualified_radius_policy(*a):\n    raise ValueError("NOT_QUALIFIED")\n')
    with pytest.raises(ValueError, match='NOT_QUALIFIED'):
        driver.radius_selection(report, helper)


def test_positive_radius_is_not_misread_as_uninflated_triangle_surface(tmp_path):
    report = tmp_path/'primitive.json'; report.write_text('{}')
    helper = tmp_path/'helper.py'; helper.write_text('def qualified_radius_policy(*a):\n    return '+repr(dict(
        schema_version='manosim_qualified_flex_radius_selection_v1', radius_id='positive_1um', radius_m=1e-6,
        undocumented_zero_radius=False, native_surface_gate_still_required=True))+'\n')
    with pytest.raises(ValueError, match='zero-radius'):
        driver.radius_selection(report, helper)


def test_primitive_started_identity_is_required_even_with_completed_report(tmp_path):
    report = tmp_path/'report.json'; protocol = tmp_path/'protocol.json'; helper = tmp_path/'helper.py'
    protocol.write_text('{}')
    h = driver.helpers()
    report.write_text(json.dumps(dict(protocol=str(protocol), protocol_sha256=h.sha(protocol), freeze_commit='frozen',
        status='COMPLETED_PRIMITIVE_QUALIFICATION', failure=None, forbidden_attempts=[])))
    helper.write_text('def validate(*a,**k): return {},"frozen"\n')
    (tmp_path/'STARTED.json').write_text(json.dumps(dict(protocol_sha256='wrong', freeze_commit='frozen', status='STARTED_ONCE')))
    with pytest.raises(ValueError, match='STARTED'):
        driver.primitive_prior(report, helper)


class Rules:
    @staticmethod
    def field_status(native, mapped, radius):
        required = np.ones((*native.shape, 3), bool)
        preserved = np.repeat((native == mapped)[:, :, None], 3, axis=2)
        return dict(required=required, preserved=preserved), dict(required=int(required.sum()))


def test_scan_seals_geometry_and_charges_query_before_manufactured_kernel(tmp_path):
    case = driver.Case(20, {}, tmp_path, driver.Budget(1)); calls = []
    geometry = dict(triangles_object_m=np.zeros((1552,3,3)), anatomical_part_index=np.arange(1552)%6)
    actual = dict(query_object_m=np.ones((1,3)), native_distance_by_part_m=np.ones((1,6)),
                  cell_radius_m=np.zeros(1), required=np.ones((1,6,3),bool))
    def refuse(*a): raise RuntimeError('manufactured quota before kernel')
    case.budget.query = refuse
    kernel = SimpleNamespace(nearest_parts=lambda *a: calls.append(True))
    with pytest.raises(RuntimeError, match='quota'):
        case.scan(driver.STAGES[0], geometry, actual, Rules, kernel)
    assert not calls and len(case.artifact_rows) == 1
    assert Path(case.artifact_rows[0]['path']).is_file()


def test_scan_uses_all_faces_fixed_denominator_and_no_admission(tmp_path):
    case = driver.Case(20, {}, tmp_path, driver.Budget(1)); calls = []
    geometry = dict(triangles_object_m=np.zeros((1552,3,3)), anatomical_part_index=np.arange(1552)%6)
    actual = dict(query_object_m=np.ones((1,3)), native_distance_by_part_m=np.ones((1,6)),
                  cell_radius_m=np.zeros(1), required=np.ones((1,6,3),bool))
    def query(points, tri, labels, n):
        calls.append((len(points), len(tri), n)); return np.ones((1,6)), None, None
    row, distance, fields = case.scan(driver.STAGES[0], geometry, actual, Rules, SimpleNamespace(nearest_parts=query))
    assert calls == [(1,1552,6)] and case.budget.tests == 1552
    assert all(row[key] is False for key in driver.surface().FLAGS)
    assert len(case.artifact_rows) == 2 and int(fields['required'].sum()) == 18


def test_earlier_completed_array_receipts_rechecked_before_final_publication(tmp_path):
    h, s = driver.helpers(), driver.surface()
    tracked, raw = tmp_path/'tracked', tmp_path/'raw'
    tracked.mkdir(); raw.mkdir()
    array = raw/'readback.npz'; np.savez_compressed(array, manufactured=np.arange(3))
    value = dict(artifacts=[s.binding(array, 'actual_creation_time_array')])
    result, raw_result = tracked/'result.json', raw/'result.json'
    result.write_text(json.dumps(value)); raw_result.write_bytes(result.read_bytes())
    rows = [dict(index=20, status='COMPLETED', result=str(result), raw_result=str(raw_result),
                 result_sha256=h.sha(result), raw_result_sha256=h.sha(raw_result))]
    assert len(driver.verify_completed_outputs(rows, tracked, raw)) == 1
    np.savez_compressed(array, manufactured=np.arange(4))
    with pytest.raises(ValueError, match='artifact changed'):
        driver.verify_completed_outputs(rows, tracked, raw)
