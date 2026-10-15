"""Prospectively freeze primitive-only MuJoCo flex checks; never use native assets.

Preparation and the manufactured tests do not import or construct MuJoCo models.
Only the committed `run`/internal worker stage may compile the twelve declared XMLs.
All numerical tolerances below concern these manufactured arithmetic/contact controls,
not the HUG surface-fidelity rule and not physical contact realism.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = 'manosim_flex_primitive_qualification_v1'
PROTOCOL_NAME = 'protocol_v2.json'
CONTROLS_NAME = 'controls_v2.json'
UNUSED_FREEZE = 'b0e9db979853222c3ce35b8e24fb3ba210794dd3'
MUJOCO_VERSION = '3.3.7'
MUJOCO_REVISION = 'f1d45bd5422c74beddfb0d1deb590a02583d21de'
RADII = (('zero', 0.0), ('positive_1um', 1e-6))
ATTACHMENTS = ('single_body', 'three_bodies')
FRAMES = ('identity', 'rigid_nonzero')
SIDES = (1, -1)
PLACEMENTS = (('penetrating', -2e-6), ('touching', 0.0), ('separated', 2e-6))
VERTICES = np.array([[0., 0., 0.], [.04, 0., 0.], [0., .04, 0.]], dtype=np.float64)
FACES = np.array([[0, 1, 2]], dtype=np.int64)
SPHERE_RADIUS = .005
TOLERANCES = dict(vertex_m=1e-14, contact_m=1e-10, normal=1e-10,
                  translation_jacobian=1e-10, rigid_jacobian=1e-10)
BUDGET = dict(model_compile_attempts=12, data_construction_attempts=12,
              explicit_forward_attempts=84, max_contacts_per_probe=2,
              maximum_seconds=120, outer_timeout_seconds=130,
              physics_steps=0, explicit_kinematics_calls=0,
              explicit_collision_calls=0, native_model_calls=0,
              native_assets=0, training_calls=0)
FORBIDDEN_IMPORTS = ('torch', 'manotorch', 'manopth', 'mano', 'hug', 'transformers',
                     'grasp_failure_prediction')
FORBIDDEN_ENGINE = ('mj_step', 'mj_step1', 'mj_step2', 'mj_kinematics', 'mj_fwdPosition',
                    'mj_collision', 'mj_geomDistance', 'mj_ray', 'mj_rayMesh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    def finite(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError('Nonfinite JSON value')
        if isinstance(obj, dict):
            for v in obj.values(): finite(v)
        elif isinstance(obj, list):
            for v in obj: finite(v)
    finite(value)
    return value


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def frame(name):
    if name == 'identity':
        return np.eye(3), np.zeros(3), np.array([1., 0., 0., 0.])
    if name != 'rigid_nonzero': raise ValueError('Unknown rigid frame')
    axis = np.array([1., -2., 3.]); axis /= np.linalg.norm(axis)
    angle = .63; k = np.array([[0., -axis[2], axis[1]],
                              [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    rotation = np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)
    return rotation, np.array([.123, -.164, .232]), np.r_[math.cos(angle/2), axis*math.sin(angle/2)]


def cases():
    rows = [dict(case_id=f'{rid}__{attachment}__{transform}', radius_id=rid,
                 radius_m=radius, attachment=attachment, frame=transform, geometry='triangle')
            for rid, radius in RADII for attachment in ATTACHMENTS for transform in FRAMES]
    rows += [dict(case_id=f'{rid}__disconnected__{transform}', radius_id=rid,
        radius_m=radius, attachment='single_body', frame=transform, geometry='disconnected')
        for rid, radius in RADII for transform in FRAMES]
    return rows


def geometry(case):
    if case['geometry'] == 'triangle': return VERTICES.copy(), FACES.copy()
    return np.array([[-.04, -.01, 0.], [-.02, -.01, 0.], [-.03, .01, 0.],
                     [.02, -.01, 0.], [.04, -.01, 0.], [.03, .01, 0.]]), np.array([[0, 1, 2], [3, 4, 5]])


def probes(case):
    rotation, translation, _ = frame(case['frame'])
    vertices, faces = geometry(case)
    specifications = [(placement, gap, 0, vertices.mean(axis=0)) for placement, gap in PLACEMENTS]
    if case['geometry'] == 'disconnected':
        specifications = [('penetrating', -2e-6, i, vertices[face].mean(axis=0)) for i, face in enumerate(faces)]
        specifications.append(('gap_clearance', -2e-6, None, vertices.mean(axis=0)))
    return [dict(probe_id=f'{placement}__triangle_{triangle}__side_{side:+d}', placement=placement,
                 side=side, expected_gap_m=gap, triangle_index=triangle, surface_center_local=center.tolist(),
                 sphere_center_world=(rotation @ (center + np.array([0., 0.,
                     side*(SPHERE_RADIUS + case['radius_m'] + gap)])) + translation).tolist())
            for side in SIDES for placement, gap, triangle, center in specifications]


def numbers(values):
    return ' '.join(format(float(x), '.17g') for x in np.ravel(values))


def make_xml(case):
    if case not in cases(): raise ValueError('Only the twelve declared cases are allowed')
    rotation, translation, quat = frame(case['frame'])
    source_vertices, source_faces = geometry(case)
    vertices = source_vertices @ rotation.T + translation
    bodies = []
    if case['attachment'] == 'single_body':
        bodies.append(f'<body name="triangle" pos="{numbers(translation)}" quat="{numbers(quat)}">'
            '<freejoint name="triangle_joint"/><inertial pos="0 0 0" mass=".03" '
            'diaginertia=".00001 .00001 .00001"/></body>')
        owners = 'triangle'; local = source_vertices
    else:
        for i, vertex in enumerate(vertices):
            bodies.append(f'<body name="vertex_{i}" pos="{numbers(vertex)}" quat="{numbers(quat)}">'
                f'<freejoint name="vertex_joint_{i}"/><inertial pos="0 0 0" mass=".01" '
                'diaginertia=".00001 .00001 .00001"/></body>')
        owners = 'vertex_0 vertex_1 vertex_2'; local = np.zeros((3, 3))
    initial = rotation @ np.array([.04/3, .04/3, .02]) + translation
    bodies.append(f'<body name="sphere" pos="{numbers(initial)}" quat="{numbers(quat)}">'
        '<freejoint name="sphere_joint"/><geom name="probe" type="sphere" size=".005" '
        'mass=".01" contype="1" conaffinity="1" condim="1" margin="0" gap="0"/></body>')
    return ('<mujoco model="primitive_flex_radius">'
        '<compiler angle="radian"/><option gravity="0 0 0" jacobian="dense" cone="elliptic"/>'
        '<worldbody>' + ''.join(bodies) + '</worldbody><deformable>'
        f'<flex name="triangle_flex" dim="2" radius="{case["radius_m"]:.17g}" body="{owners}" '
        f'vertex="{numbers(local)}" element="{numbers(source_faces)}"><contact contype="1" conaffinity="1" '
        'condim="1" margin="0" gap="0" internal="false" selfcollide="none"/>'
        '</flex></deformable></mujoco>')


def validate_xml(case, xml):
    if case not in cases() or xml != make_xml(case):
        raise ValueError('XML differs from the exact declared synthetic model')
    root = ET.fromstring(xml)
    if any(x.tag in ('include', 'asset', 'plugin', 'extension', 'actuator')
           or 'file' in x.attrib for x in root.iter()):
        raise ValueError('No assets, includes, plugins or actuators are permitted')
    flex = root.find('./deformable/flex')
    if flex is None or flex.attrib['dim'] != '2' or flex.attrib['element'] != numbers(geometry(case)[1]):
        raise ValueError('Exact ordered original triangles required')
    return xml


def runtime():
    return dict(python=platform.python_version(), executable=str(Path(sys.executable).resolve()),
                numpy=importlib.metadata.version('numpy'), mujoco=importlib.metadata.version('mujoco'))


def runtime_files():
    files = {Path(sys.executable).resolve()}
    for package in ('numpy', 'mujoco'):
        distribution = importlib.metadata.distribution(package)
        for member in distribution.files or ():
            path = Path(distribution.locate_file(member)).resolve()
            if path.is_file() and path.suffix != '.pyc' and '__pycache__' not in path.parts:
                files.add(path)
    return sorted(files)


def source_files():
    return [HERE/name for name in ('run_qualification.py', 'test_run_qualification.py',
                                 'README.md', 'primary_sources.json', CONTROLS_NAME,
                                 'protocol.json', 'controls.json')]


def prior_unused():
    return dict(path=str((HERE/'protocol.json').resolve()), sha256=sha(HERE/'protocol.json'),
                freeze_commit=UNUSED_FREEZE, execution_status='NEVER_EXECUTED')


def prepare(path):
    path = Path(path).resolve()
    if path.exists(): raise FileExistsError('Never replace a prepared declaration')
    if path.parent != HERE or path.name != PROTOCOL_NAME:
        raise ValueError('Use the one new primitive protocol path')
    if (HERE/'run_v1').exists(): raise FileExistsError('No preparation after any primitive execution attempt')
    current = runtime()
    if current['mujoco'] != MUJOCO_VERSION: raise ValueError('Exactly MuJoCo3.3.7 required')
    controls = read(HERE/CONTROLS_NAME)
    if controls.get('exit_code') != 0 or controls.get('integration_calls') != 0:
        raise ValueError('Successful manufactured controls receipt required')
    required = source_files() + runtime_files()
    bindings = {str(file.resolve()): sha(file) for file in required}
    protocol = dict(schema_version=SCHEMA, preparation_version=2, prior_unused_declaration=prior_unused(),
        prepared_utc=datetime.now(timezone.utc).isoformat(),
        runtime=current, upstream_revision=MUJOCO_REVISION, cases=cases(),
        probes={c['case_id']: probes(c) for c in cases()}, budgets=BUDGET, tolerances=TOLERANCES,
        xml_sha256={c['case_id']: hashlib.sha256(make_xml(c).encode()).hexdigest() for c in cases()},
        bindings=bindings, tracked_sources=[str(file.resolve()) for file in source_files()],
        output_root=str((HERE/'run_v1').resolve()), no_native_assets=True,
        undocumented_zero_radius_qualified=False, HUG_fidelity_accepted=False,
        articulated_MANO_dynamics_validated=False, collection_or_training_allowed=False)
    write(path, protocol)
    return protocol


def validate(protocol, commit, *, require_head=True):
    protocol = Path(protocol).resolve(); p = read(protocol)
    if (protocol != HERE/PROTOCOL_NAME or p.get('schema_version') != SCHEMA
        or p.get('preparation_version') != 2 or p.get('prior_unused_declaration') != prior_unused()
        or p.get('runtime') != runtime() or p.get('upstream_revision') != MUJOCO_REVISION
        or p.get('cases') != cases() or p.get('probes') != {c['case_id']: probes(c) for c in cases()}
        or p.get('budgets') != BUDGET or p.get('tolerances') != TOLERANCES
        or p.get('output_root') != str((HERE/'run_v1').resolve())
        or p.get('no_native_assets') is not True):
        raise ValueError('Primitive protocol/runtime/finite-case declaration changed')
    for key in ('undocumented_zero_radius_qualified', 'HUG_fidelity_accepted',
                'articulated_MANO_dynamics_validated', 'collection_or_training_allowed'):
        if p.get(key) is not False: raise ValueError('Protocol cannot predeclare success or admission')
    expected_sources = sorted(str(f.resolve()) for f in source_files())
    if sorted(p.get('tracked_sources', [])) != expected_sources:
        raise ValueError('Exact primitive source inventory required')
    required = sorted(str(f.resolve()) for f in source_files()+runtime_files())
    if sorted(p.get('bindings', {})) != required: raise ValueError('Exact runtime/source closure required')
    for name, value in p['bindings'].items():
        if sha(name) != value: raise ValueError('Bound input changed: '+name)
    expected_xml = {c['case_id']: hashlib.sha256(validate_xml(c, make_xml(c)).encode()).hexdigest() for c in cases()}
    if p.get('xml_sha256') != expected_xml: raise ValueError('Synthetic XML changed')
    frozen = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', commit+'^{commit}'], text=True).strip()
    if require_head and subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() != frozen:
        raise ValueError('HEAD must remain at the declared primitive freeze')
    for name in expected_sources+[str(protocol)]:
        relative = str(Path(name).relative_to(ROOT))
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', frozen+':'+relative])
        if hashlib.sha256(blob).hexdigest() != sha(name):
            raise ValueError('Protocol/source differs from exact committed Git blob')
    return p, frozen


class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN_IMPORTS:
            raise RuntimeError('Native/model/project import forbidden: '+fullname)
        return None


class EngineBudget:
    def __init__(self, output):
        self.output = Path(output); self.start = time.monotonic()
        self.counts = {key: 0 for key in ('model_compile_attempts', 'data_construction_attempts',
            'explicit_forward_attempts', 'physics_steps', 'explicit_kinematics_calls',
            'explicit_collision_calls', 'native_model_calls', 'training_calls')}
        self.forbidden_attempts = []

    def check_time(self):
        if time.monotonic()-self.start > BUDGET['maximum_seconds']:
            raise RuntimeError('Primitive wall-time budget exceeded')

    def charge(self, name):
        self.check_time()
        if name not in self.counts or self.counts[name] >= BUDGET[name]:
            self.forbidden_attempts.append(name); self.save()
            raise RuntimeError('Undeclared or exhausted operation: '+name)
        self.counts[name] += 1; self.save()

    def save(self):
        target = self.output/'operation_progress.json'; temporary = self.output/'operation_progress.tmp'
        temporary.write_text(json.dumps(dict(counts=self.counts, forbidden_attempts=self.forbidden_attempts,
            elapsed_s=time.monotonic()-self.start), sort_keys=True, allow_nan=False)+'\n')
        temporary.replace(target)

    def reject(self, name):
        self.forbidden_attempts.append(name); self.save()
        raise RuntimeError('Forbidden engine operation: '+name)


class CompileRejected(Exception):
    pass


def finite_state(data):
    for key in ('qpos', 'qvel', 'qacc', 'ctrl'):
        if not np.isfinite(np.asarray(getattr(data, key))).all():
            raise RuntimeError('Nonfinite '+key)
    if not math.isfinite(float(data.time)): raise RuntimeError('Nonfinite time')


@contextmanager
def engine_scope(module, budget, allowed_xml):
    original_model, original_data, original_forward = module.MjModel, module.MjData, module.mj_forward
    saved = {name: getattr(module, name) for name in FORBIDDEN_ENGINE if hasattr(module, name)}
    model_ids = set(); data_pairs = set(); remaining = list(allowed_xml)
    class ModelFactory:
        @staticmethod
        def from_xml_string(xml, assets=None):
            if assets is not None or not remaining or xml != remaining[0]:
                budget.reject('undeclared_model_XML_or_assets')
            remaining.pop(0); budget.charge('model_compile_attempts')
            try: result = original_model.from_xml_string(xml)
            except Exception as error:
                raise CompileRejected(type(error).__name__+': '+str(error)) from error
            model_ids.add(id(result)); budget.check_time(); return result
    def data_factory(model):
        if id(model) not in model_ids: budget.reject('undeclared_model_data')
        budget.charge('data_construction_attempts'); result = original_data(model)
        data_pairs.add((id(model), id(result))); return result
    def forward(model, data):
        if (id(model), id(data)) not in data_pairs: budget.reject('undeclared_forward_pair')
        finite_state(data); before_time = float(data.time)
        before_warnings = np.array([row.number for row in data.warning], dtype=np.int64)
        budget.charge('explicit_forward_attempts')
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always'); result = original_forward(model, data)
        finite_state(data); budget.check_time()
        after_warnings = np.array([row.number for row in data.warning], dtype=np.int64)
        if emitted or not np.array_equal(before_warnings, after_warnings):
            raise RuntimeError('Static forward emitted a numerical/runtime warning')
        if float(data.time) != before_time: raise RuntimeError('Static forward changed time')
        return result
    module.MjModel = ModelFactory; module.MjData = data_factory; module.mj_forward = forward
    for name in saved: setattr(module, name, lambda *a, _name=name, **k: budget.reject(_name))
    try: yield
    finally:
        module.MjModel = original_model; module.MjData = original_data; module.mj_forward = original_forward
        for name, value in saved.items(): setattr(module, name, value)


def barycentric(point, vertices):
    a, b, c = vertices
    edge1, edge2, delta = b-a, c-a, point-a
    gram = np.array([[edge1@edge1, edge1@edge2], [edge1@edge2, edge2@edge2]])
    uv = np.linalg.solve(gram, np.array([edge1@delta, edge2@delta]))
    return np.array([1-uv.sum(), uv[0], uv[1]])


def independent_normal_jacobian(model, data, case, normal, contact_position, vertices):
    """Rigid vertex velocity, then triangle barycentric blending; no engine Jacobian API.

    Only free-joint translation columns are a required three-owner check. Rotational
    mismatch is retained: MuJoCo uses a common contact position for owner Jacobians.
    """
    weights = barycentric(contact_position, vertices)
    expected = np.zeros(model.nv)
    names = ['triangle']*3 if case['attachment'] == 'single_body' else [f'vertex_{i}' for i in range(3)]
    joints = ['triangle_joint']*3 if case['attachment'] == 'single_body' else [f'vertex_joint_{i}' for i in range(3)]
    translation_columns = []
    for weight, vertex, name, joint in zip(weights, vertices, names, joints):
        bid = int(model.body(name).id); adr = int(model.joint(joint).dofadr[0])
        expected[adr:adr+3] += weight*normal
        rotation = np.asarray(data.xmat[bid]).reshape(3, 3)
        for k in range(3):
            expected[adr+3+k] += weight*(normal @ np.cross(rotation[:, k], vertex-data.xpos[bid]))
        translation_columns.extend(range(adr, adr+3))
    sphere = int(model.body('sphere').id); adr = int(model.joint('sphere_joint').dofadr[0])
    expected[adr:adr+3] -= normal; translation_columns.extend(range(adr, adr+3))
    rotation = np.asarray(data.xmat[sphere]).reshape(3, 3)
    for k in range(3):
        expected[adr+3+k] -= normal @ np.cross(rotation[:, k], contact_position-data.xpos[sphere])
    return expected, sorted(set(translation_columns)), weights


def geometry_readback(model, data, case):
    source_vertices, source_faces = geometry(case); n = len(source_vertices); nf = len(source_faces)
    expected_nv = 12 if case['attachment'] == 'single_body' else 24
    if model.nv != expected_nv or model.nq != expected_nv//6*7:
        raise RuntimeError('Free owner/sphere degrees of freedom were removed or changed')
    if model.nflex != 1 or int(model.flex_dim[0]) != 2 or int(model.flex_vertnum[0]) != n or int(model.flex_elemnum[0]) != nf:
        raise RuntimeError('Compiled primitive topology changed')
    if float(model.flex_radius[0]) != case['radius_m']: raise RuntimeError('Compiled radius changed')
    faces = np.asarray(model.flex_elem)[int(model.flex_elemdataadr[0]):int(model.flex_elemdataadr[0])+3*nf].reshape(nf, 3).copy()
    vertices = np.asarray(data.flexvert_xpos)[int(model.flex_vertadr[0]):int(model.flex_vertadr[0])+n].copy()
    owners = np.asarray(model.flex_vertbodyid)[int(model.flex_vertadr[0]):int(model.flex_vertadr[0])+n].copy()
    expected_owners = [int(model.body('triangle').id)]*n if case['attachment'] == 'single_body' else [int(model.body(f'vertex_{i}').id) for i in range(3)]
    if not np.array_equal(faces, source_faces) or not np.array_equal(owners, expected_owners):
        raise RuntimeError('Compiled face order or owner binding changed')
    if not np.isfinite(vertices).all(): raise RuntimeError('Nonfinite compiled vertex')
    return vertices, faces, owners


def artifact(path, label):
    return dict(path=str(Path(path).resolve()), sha256=sha(path), bytes=Path(path).stat().st_size, label=label)


def measure_case(module, case, output):
    """The only model/forward scientific work, reachable only after run freeze validation."""
    output = Path(output); xml = validate_xml(case, make_xml(case))
    (output/'model.xml').write_text(xml)
    try:
        model = module.MjModel.from_xml_string(xml)
    except CompileRejected as error:
        return dict(case=case, status='COMPILE_REJECTED', compile_exception=dict(type=type(error).__name__, message=str(error)),
                    probes=[], artifacts=[artifact(output/'model.xml', 'exact_synthetic_XML')])
    data = module.MjData(model); module.mj_forward(model, data)
    rotation, translation, _ = frame(case['frame']); source_vertices, source_faces = geometry(case)
    expected_vertices = source_vertices@rotation.T+translation
    observed, observed_faces, owners = geometry_readback(model, data, case)
    q0 = np.asarray(data.qpos).copy(); sphere_adr = int(model.joint('sphere_joint').qposadr[0])
    if not np.array_equal(data.qvel, np.zeros(model.nv)) or float(data.time) != 0.:
        raise RuntimeError('Static initial state is not stationary/timezero')
    arrays = dict(expected_vertices=expected_vertices, observed_vertices=[observed], expected_faces=source_faces,
                  observed_faces=observed_faces, owner_body_ids=owners)
    records = []; qpos_records = [data.qpos.copy()]; qvel_records = [data.qvel.copy()]
    warning_records = [np.array([w.number for w in data.warning], dtype=np.int64)]
    for i, probe in enumerate(probes(case)):
        data.qpos[:] = q0; data.qpos[sphere_adr:sphere_adr+3] = probe['sphere_center_world']
        data.qvel[:] = 0.; module.mj_forward(model, data)
        vertices, faces, current_owners = geometry_readback(model, data, case)
        arrays['observed_vertices'].append(vertices); qpos_records.append(data.qpos.copy()); qvel_records.append(data.qvel.copy())
        warning_records.append(np.array([w.number for w in data.warning], dtype=np.int64))
        if not np.array_equal(faces, observed_faces) or not np.array_equal(owners, current_owners):
            raise RuntimeError('Topology or ownership changed during static probes')
        unchanged = np.ones(model.nq, dtype=bool); unchanged[sphere_adr:sphere_adr+3] = False
        if not np.array_equal(data.qpos[unchanged], q0[unchanged]) or not np.array_equal(data.qvel, np.zeros(model.nv)):
            raise RuntimeError('Static forward changed positions or velocities')
        count = int(data.ncon)
        if count > BUDGET['max_contacts_per_probe']: raise RuntimeError('Unexpected contact count exceeds artifact bound')
        rows = []
        dense_jac = np.asarray(data.efc_J).reshape(int(data.nefc), model.nv)
        for j in range(count):
            contact = data.contact[j]
            if (np.asarray(contact.geom).tolist() != [int(model.geom('probe').id), -1]
                or np.asarray(contact.flex).tolist() != [-1, 0] or not 0 <= int(contact.elem[1]) < len(source_faces)
                or int(contact.dim) != 1):
                raise RuntimeError('Unexpected contact identity or dimensionality')
            normal = np.asarray(contact.frame).reshape(3, 3)[0].copy(); position = np.asarray(contact.pos).copy()
            address = int(contact.efc_address)
            if address >= int(data.nefc): raise RuntimeError('Invalid contact constraint address')
            active = address >= 0
            # At exact zero gap, collision can retain an excluded contact with no
            # constraint row. Touch presence/activity are descriptive, not a failure.
            engine_jac = dense_jac[address].copy() if active else np.zeros(model.nv)
            expected_jac, columns, weights = independent_normal_jacobian(model, data, case, normal, position,
                vertices[source_faces[int(contact.elem[1])]])
            row = dict(distance_m=float(contact.dist), position_world=position.tolist(), normal_world=normal.tolist(),
                engine_normal_jacobian=engine_jac.tolist(), expected_vertex_normal_jacobian=expected_jac.tolist(),
                translation_columns=columns, barycentric_weights=weights.tolist(), efc_address=address,
                constraint_active=active)
            if not all(np.isfinite(np.asarray(row[key])).all() for key in ('position_world', 'normal_world',
                'engine_normal_jacobian', 'expected_vertex_normal_jacobian', 'barycentric_weights')) or not math.isfinite(row['distance_m']):
                raise RuntimeError('Nonfinite contact/Jacobian')
            rows.append(row)
            arrays[f'probe_{i}_contact_{j}_engine_normal_jacobian'] = engine_jac
            arrays[f'probe_{i}_contact_{j}_expected_vertex_normal_jacobian'] = expected_jac
        records.append(dict(probe=probe, contact_count=count, contacts=rows))
    arrays['observed_vertices'] = np.array(arrays['observed_vertices'])
    arrays['qpos'] = np.array(qpos_records); arrays['qvel'] = np.array(qvel_records)
    arrays['warning_counters'] = np.array(warning_records)
    np.savez_compressed(output/'readback.npz', **arrays)
    result = dict(case=case, status='COMPILED_AND_MEASURED', probes=records,
                  vertex_max_error_m=float(np.max(np.abs(arrays['observed_vertices']-expected_vertices))),
                  exact_faces=True, exact_owner_bindings=True, nv=int(model.nv),
                  artifacts=[artifact(output/'model.xml', 'exact_synthetic_XML'), artifact(output/'readback.npz', 'primitive_readback')])
    result['checks'] = check_case(result)
    return result


def check_case(result):
    case = result['case']
    if case not in cases() or result.get('status') != 'COMPILED_AND_MEASURED':
        raise ValueError('Only measured exact primitive cases can receive contact checks')
    if [r.get('probe') for r in result.get('probes', [])] != probes(case):
        raise ValueError('Exact six ordered probe placements required')
    errors = []; max_distance = max_point = max_normal = max_translation = max_rotation = 0.
    if not math.isfinite(result['vertex_max_error_m']) or result['vertex_max_error_m'] > TOLERANCES['vertex_m']:
        errors.append('vertex_readback')
    if result.get('exact_faces') is not True or result.get('exact_owner_bindings') is not True:
        errors.append('topology_or_owner_readback')
    rotation, translation, _ = frame(case['frame'])
    for record in result['probes']:
        p = record['probe']; rows = record['contacts']; count = record['contact_count']
        if isinstance(count, bool) or count != len(rows) or count > BUDGET['max_contacts_per_probe']:
            raise ValueError('Malformed bounded contact inventory')
        if (p['placement'] == 'penetrating' and count != 1) or (p['placement'] in ('separated', 'gap_clearance') and count != 0):
            errors.append(p['probe_id']+':contact_presence')
        if p['placement'] == 'touching' and count > 1: errors.append(p['probe_id']+':touch_duplicates')
        for row in rows:
            pos = np.asarray(row['position_world']); normal = np.asarray(row['normal_world'])
            engine = np.asarray(row['engine_normal_jacobian']); expected = np.asarray(row['expected_vertex_normal_jacobian'])
            if pos.shape != (3,) or normal.shape != (3,) or engine.shape != (result['nv'],) or expected.shape != engine.shape:
                raise ValueError('Malformed contact/Jacobian shape')
            if not all(np.isfinite(x).all() for x in (pos, normal, engine, expected)) or not math.isfinite(row['distance_m']):
                raise ValueError('Nonfinite saved primitive contact')
            center = np.asarray(p['surface_center_local'])
            expected_point = rotation@(center+np.array([0., 0., p['side']*(case['radius_m']+p['expected_gap_m']/2)]))+translation
            max_distance = max(max_distance, abs(row['distance_m']-p['expected_gap_m']))
            max_point = max(max_point, float(np.max(np.abs(pos-expected_point))))
            max_normal = max(max_normal, float(np.max(np.abs(normal+p['side']*rotation[:, 2]))))
            columns = row['translation_columns']
            if columns != sorted(set(columns)) or not columns or min(columns) < 0 or max(columns) >= result['nv']:
                raise ValueError('Malformed translation-column declaration')
            active = row.get('constraint_active')
            if not isinstance(active, bool): raise ValueError('Explicit contact constraint activity required')
            if p['placement'] == 'penetrating' and not active:
                errors.append(p['probe_id']+':missing_active_constraint')
            if not active: continue
            max_translation = max(max_translation, float(np.max(np.abs(engine[columns]-expected[columns]))))
            rotational = sorted(set(range(result['nv']))-set(columns))
            max_rotation = max(max_rotation, float(np.max(np.abs(engine[rotational]-expected[rotational]))))
    for name, value, limit in (('distance', max_distance, TOLERANCES['contact_m']),
        ('point', max_point, TOLERANCES['contact_m']), ('normal', max_normal, TOLERANCES['normal']),
        ('translation_jacobian', max_translation, TOLERANCES['translation_jacobian'])):
        if value > limit: errors.append(name)
    if case['attachment'] == 'single_body' and max_rotation > TOLERANCES['rigid_jacobian']:
        errors.append('single_body_rigid_rotation_jacobian')
    return dict(status='PASS' if not errors else 'FAIL', failures=errors,
        max_distance_error_m=max_distance, max_contact_point_error_m=max_point,
        max_normal_error=max_normal, max_translation_jacobian_error=max_translation,
        max_vertex_kinematic_rotation_jacobian_error=max_rotation,
        three_owner_rotation_is_diagnostic_only=case['attachment'] == 'three_bodies',
        touching_contact_presence_is_descriptive=True)


def compare_frames(identity, transformed):
    if identity['case']['frame'] != 'identity' or transformed['case']['frame'] != 'rigid_nonzero':
        raise ValueError('Ordered identity/nonzero frame pair required')
    errors = []; distance = point = normal = 0.
    r, t, _ = frame('rigid_nonzero')
    for left, right in zip(identity['probes'], transformed['probes'], strict=True):
        # Touch count is explicitly not an invariant requirement at a strict zero boundary.
        if left['probe']['placement'] != 'touching' and left['contact_count'] != right['contact_count']:
            errors.append(left['probe']['probe_id']+':contact_presence')
        if left['contact_count'] != right['contact_count']: continue
        for a, b in zip(left['contacts'], right['contacts'], strict=True):
            distance = max(distance, abs(a['distance_m']-b['distance_m']))
            point = max(point, float(np.max(np.abs(np.asarray(a['position_world'])-r.T@(np.asarray(b['position_world'])-t)))))
            normal = max(normal, float(np.max(np.abs(np.asarray(a['normal_world'])-r.T@np.asarray(b['normal_world'])))))
    if distance > TOLERANCES['contact_m']: errors.append('distance')
    if point > TOLERANCES['contact_m']: errors.append('point')
    if normal > TOLERANCES['normal']: errors.append('normal')
    return dict(status='PASS' if not errors else 'FAIL', failures=errors,
                max_distance_error_m=distance, max_point_error_m=point, max_normal_error=normal)


def summarize(case_results):
    if [x['case'] for x in case_results] != cases(): raise ValueError('All twelve planned cases must be accounted for')
    by_id = {row['case']['case_id']: row for row in case_results}; assessments = []
    for radius_id, radius in RADII:
        selected = [x for x in case_results if x['case']['radius_id'] == radius_id]
        pairs = []
        for geometry_id, attachment in (('triangle', 'single_body'), ('triangle', 'three_bodies'), ('disconnected', 'single_body')):
            pair = [next(x for x in selected if x['case']['attachment'] == attachment
                and x['case']['geometry'] == geometry_id and x['case']['frame'] == transform) for transform in FRAMES]
            check = compare_frames(*pair) if all(x['status'] == 'COMPILED_AND_MEASURED' for x in pair) else dict(status='UNAVAILABLE')
            pairs.append(dict(geometry=geometry_id, attachment=attachment, comparison=check))
        compiled = sum(x['status'] == 'COMPILED_AND_MEASURED' for x in selected)
        passed = compiled == 6 and all(check_case(x)['status'] == 'PASS' for x in selected) and all(x['comparison']['status'] == 'PASS' for x in pairs)
        assessments.append(dict(radius_id=radius_id, radius_m=radius, planned_cases=6, compiled_cases=compiled,
            measured_cases=compiled, case_ids=[x['case']['case_id'] for x in selected], frame_comparisons=pairs,
            status='BOUNDED_PRIMITIVE_CONTROLS_PASS' if passed else 'NOT_QUALIFIED',
            zero_radius_documented=False if radius == 0 else None,
            scope='Synthetic static sphere/one-triangle cases only; no native surface or articulated dynamics admission'))
    return assessments


def verify_artifacts(rows, output):
    output = Path(output).resolve(); seen = set()
    for row in rows:
        path = Path(row['path']); resolved = path.resolve()
        if (set(row) != {'path', 'sha256', 'bytes', 'label'} or not path.is_absolute() or path.is_symlink()
            or resolved in seen or not resolved.is_relative_to(output) or not resolved.is_file()
            or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(output))
            or isinstance(row['bytes'], bool) or row['bytes'] <= 0 or resolved.stat().st_size != row['bytes'] or sha(resolved) != row['sha256']):
            raise ValueError('Output artifact changed, escaped, duplicated or aliased')
        seen.add(resolved)
    if {p.resolve() for p in output.rglob('*.npz')} != {p for p in seen if p.suffix == '.npz'}:
        raise ValueError('Unindexed or missing primitive array artifact')


def worker(protocol, commit, output):
    p, frozen = validate(protocol, commit); output = Path(output).resolve()
    if output != Path(p['output_root']): raise ValueError('Exact frozen output root required')
    started = read(output/'STARTED.json')
    if started != dict(protocol_sha256=sha(protocol), freeze_commit=frozen, status='STARTED_ONCE'):
        raise ValueError('Matching once-only launcher marker required')
    if (output/'operation_progress.json').exists() or (output/'report.json').exists():
        raise FileExistsError('Never restart an attempted primitive worker')
    budget = EngineBudget(output); budget.save(); import_guard = ForbiddenImports(); sys.meta_path.insert(0, import_guard)
    results = []; failure = None; active_case = None
    try:
        import mujoco
        if mujoco.__version__ != MUJOCO_VERSION or str(Path(mujoco.__file__).resolve()) not in p['bindings']:
            raise RuntimeError('Imported MuJoCo version/path differs from bound runtime')
        for name, module in tuple(sys.modules.items()):
            if name.split('.')[0] in ('numpy', 'mujoco') and getattr(module, '__file__', None):
                path = str(Path(module.__file__).resolve())
                if path not in p['bindings']: raise RuntimeError('Runtime module outside frozen distribution: '+path)
        with engine_scope(mujoco, budget, [make_xml(c) for c in cases()]):
            for case in cases():
                active_case = case
                folder = output/case['case_id']; folder.mkdir()
                result = measure_case(mujoco, case, folder)
                write(folder/'result.json', result); result['result_artifact'] = artifact(folder/'result.json', 'primitive_case_result')
                results.append(result)
                active_case = None
        validate(protocol, frozen)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        sys.meta_path.remove(import_guard); budget.save()
    if active_case is not None:
        folder = output/active_case['case_id']
        results.append(dict(case=active_case, status='HARD_STOPPED_PARTIAL', probes=[],
            artifacts=[artifact(file, 'failed_partial_uninterpreted') for file in sorted(folder.glob('*')) if file.is_file()]))
    for case in cases()[len(results):]: results.append(dict(case=case, status='NOT_RUN_AFTER_HARD_STOP', probes=[], artifacts=[]))
    artifacts = [row for result in results for row in result['artifacts']]
    artifacts += [result['result_artifact'] for result in results if 'result_artifact' in result]
    if failure is None:
        try: verify_artifacts(artifacts, output)
        except Exception as error: failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    report = dict(schema_version=SCHEMA, status='HARD_STOPPED' if failure else 'COMPLETED_PRIMITIVE_QUALIFICATION',
        protocol=str(Path(protocol).resolve()), protocol_sha256=sha(protocol), freeze_commit=frozen,
        cases=results, radius_assessments=[] if failure else summarize(results), operation_counts=budget.counts,
        forbidden_attempts=budget.forbidden_attempts, elapsed_s=time.monotonic()-budget.start,
        failure=failure, artifacts=artifacts, HUG_fidelity_accepted=False,
        articulated_MANO_dynamics_validated=False, collection_or_training_allowed=False)
    write(output/'report.json', report)
    return report


def run(protocol, commit):
    p, frozen = validate(protocol, commit); output = Path(p['output_root'])
    output.mkdir(exist_ok=False)
    write(output/'STARTED.json', dict(protocol_sha256=sha(protocol), freeze_commit=frozen, status='STARTED_ONCE'))
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--protocol', str(protocol), '--freeze-commit', frozen]
    try:
        with (output/'worker.log').open('x') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=BUDGET['outer_timeout_seconds'])
        if process.returncode != 0 or not (output/'report.json').is_file():
            raise RuntimeError('Primitive worker did not publish a complete report; see worker.log')
        report = read(output/'report.json'); validate(protocol, frozen)
        if report['status'] != 'COMPLETED_PRIMITIVE_QUALIFICATION': raise RuntimeError('Primitive worker hard-stopped; report retained')
        return report
    except BaseException as error:
        write(output/'launcher_failure.json', dict(type=type(error).__name__, message=str(error),
            operation_counts=read(output/'operation_progress.json') if (output/'operation_progress.json').exists() else None,
            note='Missing/partial work is unknown or attempted; never interpreted as zero successful calls.'))
        raise


def qualified_radius_policy(report_path, expected_sha256, radius_id):
    """Read-only primitive capability receipt; never admits native collection by itself."""
    report_path = Path(report_path).resolve()
    if sha(report_path) != expected_sha256: raise ValueError('Qualification report hash changed')
    report = read(report_path)
    if (report.get('schema_version') != SCHEMA or report.get('status') != 'COMPLETED_PRIMITIVE_QUALIFICATION'
        or report.get('failure') is not None or report.get('forbidden_attempts') != []
        or report.get('HUG_fidelity_accepted') is not False or report.get('collection_or_training_allowed') is not False):
        raise ValueError('Completed non-admitting primitive qualification required')
    if sha(report['protocol']) != report['protocol_sha256']: raise ValueError('Qualification protocol changed')
    protocol = read(report['protocol'])
    if protocol.get('cases') != cases() or protocol.get('budgets') != BUDGET or protocol.get('tolerances') != TOLERANCES:
        raise ValueError('Qualification method changed')
    verify_artifacts(report['artifacts'], report_path.parent)
    for case in report['cases']:
        receipt = case.get('result_artifact')
        if receipt is None: raise ValueError('Completed case creation-time scalar receipt required')
        actual = dict(case); actual.pop('result_artifact')
        if read(receipt['path']) != actual: raise ValueError('Report case differs from original scalar receipt')
        if receipt not in report['artifacts']: raise ValueError('Case receipt omitted from indexed artifacts')
    assessments = summarize(report['cases'])
    if report.get('radius_assessments') != assessments: raise ValueError('Qualification summary differs from measurements')
    row = next((x for x in assessments if x['radius_id'] == radius_id), None)
    if row is None or row['status'] != 'BOUNDED_PRIMITIVE_CONTROLS_PASS':
        raise ValueError('This radius did not pass all declared primitive controls')
    return dict(schema_version='manosim_qualified_flex_radius_selection_v1', radius_id=radius_id, radius_m=row['radius_m'],
        qualification_receipt=dict(path=str(report_path), sha256=expected_sha256, bytes=report_path.stat().st_size),
        qualification_case_ids=row['case_ids'], undocumented_zero_radius=row['radius_m'] == 0.,
        scope=row['scope'], native_surface_gate_still_required=True,
        articulated_MANO_dynamics_validated=False, collection_or_training_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', '_worker'))
    parser.add_argument('--protocol', type=Path, default=HERE/PROTOCOL_NAME)
    parser.add_argument('--freeze-commit')
    args = parser.parse_args()
    if args.stage == 'prepare': result = prepare(args.protocol)
    else:
        if not args.freeze_commit: parser.error('--freeze-commit is required before any model operation')
        result = run(args.protocol, args.freeze_commit) if args.stage == 'run' else worker(args.protocol, args.freeze_commit, HERE/'run_v1')
    print(json.dumps(dict(status=result.get('status', 'PREPARED_NOT_EXECUTED'), schema_version=SCHEMA)))


if __name__ == '__main__': main()
