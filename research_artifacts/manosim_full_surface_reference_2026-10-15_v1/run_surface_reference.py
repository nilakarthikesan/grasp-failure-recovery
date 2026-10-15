"""Separately frozen, static locked-articulation complete native skin reference.

Prepare reads/hashes only. Actual compilation/forwards/queries are available
only to a child of a committed protocol; physics and model generation stay
blocked. Historical native/ablation source and results are never modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import warnings
from xml.etree import ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABLATION = ROOT/'research_artifacts/manosim_geometry_ablation_2026-10-15_v1'
NATIVE = ROOT/'research_artifacts/manosim_native_transfer_2026-10-15_v2'
PRIMITIVE = ROOT/'research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1'
SCHEMA = 'manosim_locked_native_full_surface_reference_v1'
INDICES = (20, 28)
STAGES = ('native_reference', 'compiled_locked_native_surface')
ABLATION_PROTOCOL_SHA = 'f5d80b06df40bb900861d4fa310e999e099ed468bb527e16af0838c8a671dbf7'
ABLATION_READBACK_SHA = '964e55f90bdaea6af2f2b316ac36aa19e0e508bbdb038737b0fb3bc36d96c546'
BUDGET = dict(cases=2, maximum_seconds_per_case=600, maximum_models_per_case=1,
    maximum_data_per_case=1, maximum_forwards_per_case=2, maximum_full_field_scans_per_case=2,
    maximum_triangle_point_tests_per_query_cell=3104, query_chunk_points=256,
    physics_steps=0, MANO_model_calls=0, HUG_generator_calls=0, retarget_calls=0,
    predictor_fits=0, task_outcomes=0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helpers():
    return load('locked_surface_prior_harness', ABLATION/'run_ablation.py')


def surface():
    return load('locked_surface_pure_candidate', HERE/'surface_candidate.py')


def completed_ablation():
    h = helpers()
    ppath, apath = ABLATION/'protocol_v2.json', ABLATION/'run_v1/independent_readback.json'
    if h.sha(ppath) != ABLATION_PROTOCOL_SHA or h.sha(apath) != ABLATION_READBACK_SHA:
        raise ValueError('Exact completed nine-stage ablation required')
    p, a, rpath = h.read(ppath), h.read(apath), ABLATION/'run_v1/report.json'
    report = h.read(rpath)
    if (report.get('status') != 'COMPLETED_SAVED_GEOMETRY_PILOT'
            or report.get('protocol_sha256') != ABLATION_PROTOCOL_SHA
            or a.get('status') != 'VERIFIED_COMPLETED_SAVED_GEOMETRY_ARRAYS'
            or a.get('protocol_sha256') != ABLATION_PROTOCOL_SHA
            or a.get('report_sha256') != h.sha(rpath)
            or a.get('freeze_commit') != report.get('freeze_commit')
            or a.get('input_binding_count') != 1914
            or a.get('all_bindings_verified_before') is not True
            or a.get('all_bindings_verified_after') is not True
            or len(p.get('bindings', {})) != 1914
            or p.get('indices') != list(INDICES)):
        raise ValueError('Independent completed ablation identity/closure changed')
    h.verify(p['bindings'])
    native, native_evidence = h.completed_native()
    inventory = a['artifact_inventory']
    for path in inventory: h.exact_inventory_file(path, inventory)
    evidence = {ppath.resolve(), apath.resolve(), rpath.resolve(), *native_evidence,
                *(Path(path).resolve() for path in inventory)}
    if [row['index'] for row in report['cases']] != list(INDICES):
        raise ValueError('Completed ablation case order changed')
    for row in report['cases']:
        path = Path(row['result']); result = h.read(path)
        if (row.get('status') != 'COMPLETED' or row.get('returncode') != 0
                or h.sha(path) != row.get('result_sha256')
                or result.get('status') != 'COMPLETED_SAVED_GEOMETRY_DIAGNOSTIC'
                or result.get('protocol_sha256') != ABLATION_PROTOCOL_SHA
                or result.get('freeze_commit') != report['freeze_commit']):
            raise ValueError('Ablation completed result differs')
        evidence.add(path.resolve())
        for artifact in result['artifacts']:
            h.exact_inventory_file(artifact['path'], inventory)
            if h.sha(artifact['path']) != artifact['sha256']:
                raise ValueError('Prior write-time geometry artifact changed')
            evidence.add(Path(artifact['path']).resolve())
    return p, native, sorted(evidence)


def _recorded_artifact(result, path):
    path = Path(path).resolve()
    matches = [row for row in result['artifacts'] if Path(row['path']).resolve() == path]
    if len(matches) != 1:
        raise ValueError('Required saved input lacks a unique write-time artifact')
    surface().verify_binding(matches[0])
    return dict(matches[0])


def case_spec(index, prior, native):
    h, s = helpers(), surface()
    nrun, arun = Path(prior['native_run']), Path(prior['output_root'])
    nr, ar = h.read(nrun/str(index)/'result.json'), h.read(arun/str(index)/'result.json')
    input_paths = dict(decoded_reference=nrun/str(index)/'decoded_native_reference.npz',
        kinematic_readback=nrun/str(index)/'mesh_kinematic_readback.npz',
        literal_pose=nrun/str(index)/'mesh_literal_pose.json',
        membership_provenance=Path(prior['raw_root'])/str(index)/'membership_provenance.npz')
    inputs = {name: _recorded_artifact(ar if name == 'membership_provenance' else nr, path)
              for name, path in input_paths.items()}
    shape = native['generated_assets']['shape_asset']
    inputs['shape_asset'] = s.binding(shape, 'literal_native_shape_from_completed_build')
    return dict(schema_version='manosim_full_surface_saved_case_spec_v1', source_inventory_index=index,
        native_sha256=nr['source_receipt']['native_sha256'],
        expected_shape_sha256=native['generated_assets']['expected_shape_sha256'], inputs=inputs,
        prior_evidence={name: s.binding(path, name) for name, path in dict(
            native_protocol=NATIVE/'native_protocol_v2.json', native_independent=NATIVE/'run_v1/independent_readback_v2.json',
            native_case=nrun/str(index)/'result.json', ablation_protocol=ABLATION/'protocol_v2.json',
            ablation_independent=ABLATION/'run_v1/independent_readback.json', ablation_case=arun/str(index)/'result.json').items()})


def radius_selection(report, helper_path):
    h = helpers(); module = load('locked_surface_qualified_radius', helper_path)
    policy = module.qualified_radius_policy(Path(report), h.sha(report), 'zero')
    if (policy.get('schema_version') != 'manosim_qualified_flex_radius_selection_v1'
            or policy.get('radius_id') != 'zero' or policy.get('radius_m') != 0.
            or policy.get('undocumented_zero_radius') is not True
            or policy.get('native_surface_gate_still_required') is not True):
        raise ValueError('Only separately qualified zero-radius triangle reference supported')
    # Positive radius needs a different actual boundary evaluator. Never treat
    # rounded triangles as the uninflated triangle surface in this wrapper.
    surface().verify_binding(policy['qualification_receipt'])
    return policy


def primitive_prior(report_path, helper_path):
    """Validate actual primitive launcher, source/runtime freeze and all outputs."""
    h = helpers(); module = load('locked_surface_primitive_closure', helper_path)
    report_path = Path(report_path).resolve(); report = h.read(report_path)
    protocol = Path(report['protocol']).resolve()
    pp, frozen = module.validate(protocol, report['freeze_commit'], require_head=False)
    if (h.sha(protocol) != report['protocol_sha256'] or report.get('status') != 'COMPLETED_PRIMITIVE_QUALIFICATION'
            or report.get('failure') is not None or report.get('forbidden_attempts') != []):
        raise ValueError('Actual completed primitive freeze required')
    marker = report_path.parent/'STARTED.json'
    if h.read(marker) != dict(protocol_sha256=h.sha(protocol), freeze_commit=frozen, status='STARTED_ONCE'):
        raise ValueError('Primitive STARTED/protocol/freeze identity changed')
    module.verify_artifacts(report['artifacts'], report_path.parent)
    if any(row.get('status') not in ('COMPILED_AND_MEASURED', 'COMPILE_REJECTED') for row in report['cases']):
        raise ValueError('All twelve primitive cases must be explicitly accounted for')
    compiled = sum(row['status'] == 'COMPILED_AND_MEASURED' for row in report['cases'])
    expected = dict(model_compile_attempts=12, data_construction_attempts=compiled,
                    explicit_forward_attempts=compiled*7, physics_steps=0,
                    explicit_kinematics_calls=0, explicit_collision_calls=0, native_model_calls=0, training_calls=0)
    if report.get('operation_counts') != expected:
        raise ValueError('Completed twelve-case primitive operation counts differ')
    paths = {report_path, protocol, marker, *(Path(row['path']).resolve() for row in report['artifacts'])}
    return pp, paths


def source_inventory(specs, primitive_report, primitive_helper, evidence, primitive_paths):
    required = {Path(__file__).resolve(), HERE/'surface_candidate.py', HERE/'README.md', HERE/'controls_v2.json',
                *HERE.glob('test_*.py'), Path(primitive_report).resolve(), Path(primitive_helper).resolve(),
                *Path(primitive_helper).parent.glob('*.py'), *Path(primitive_helper).parent.glob('*.md'),
                *evidence, *primitive_paths}
    for spec in specs.values():
        required.update(Path(row['path']) for row in list(spec['inputs'].values()) + list(spec['prior_evidence'].values()))
    return sorted(path.resolve() for path in required)


def merged_bindings(prior, primitive, required):
    h = helpers(); bindings = dict(prior['bindings'])
    for path, value in primitive['bindings'].items():
        if path in bindings and bindings[path] != value:
            raise ValueError('Prior scientific/source closures conflict: ' + path)
        bindings[path] = value
    for path in required:
        key, value = str(path), h.sha(path)
        if key in bindings and bindings[key] != value:
            raise ValueError('Current measurement source differs from historical bound bytes: ' + key)
        bindings[key] = value
    return bindings


def roots(p, output):
    tracked = (HERE/'run_v1').resolve()
    raw = (ROOT/'runs/manosim_full_surface_reference_2026-10-15_v1/run_v1/raw').resolve()
    if str(tracked) != p['output_root'] or str(raw) != p['raw_root'] or Path(output).resolve() != tracked:
        raise ValueError('Prospectively fixed tracked/raw roots required')
    return tracked, raw


def prepare(path, primitive_report, primitive_helper=PRIMITIVE/'run_qualification.py'):
    h = helpers(); path = Path(path)
    if path.exists(): raise FileExistsError('Never overwrite a surface declaration')
    prior, native, evidence = completed_ablation(); h.bootstrap(native)
    policy = radius_selection(primitive_report, primitive_helper)
    primitive_report, primitive_helper = Path(primitive_report).resolve(), Path(primitive_helper).resolve()
    specs = {str(index): case_spec(index, prior, native) for index in INDICES}
    p = dict(schema_version=SCHEMA, prepared_utc=datetime.now(timezone.utc).isoformat(),
        indices=list(INDICES), stages=list(STAGES), budgets=dict(BUDGET), runtime=h.runtime(),
        prior_ablation_protocol=str((ABLATION/'protocol_v2.json').resolve()),
        prior_ablation_sha256=ABLATION_PROTOCOL_SHA, prior_ablation_readback_sha256=ABLATION_READBACK_SHA,
        native_v3_protocol=native['v3_protocol'], radius_policy=policy,
        primitive_report=str(primitive_report), primitive_helper=str(primitive_helper),
        cases=specs, field_rules=prior['field_rules'], geometry_base=prior['geometry_base'],
        native_run=prior['native_run'], ablation_run=prior['output_root'], ablation_raw=prior['raw_root'],
        base_xml=native['generated_assets']['variant_xmls']['mesh'],
        output_root=str((HERE/'run_v1').resolve()),
        raw_root=str((ROOT/'runs/manosim_full_surface_reference_2026-10-15_v1/run_v1/raw').resolve()),
        fixed_articulation=True, reference_reproduction_limit_m=1e-12,
        face_normal_component_limit=1e-10, scientific_acceptance_changed=False,
        **surface().FLAGS)
    pp, primitive_paths = primitive_prior(primitive_report, primitive_helper)
    required = source_inventory(specs, primitive_report, primitive_helper, evidence, primitive_paths)
    bindings = merged_bindings(prior, pp, required)
    p['bindings'] = bindings
    p['measurement_sources'] = sorted(str(Path(file).resolve()) for file in required)
    p['tracked_sources'] = h.source_partitions(bindings, p['measurement_sources'],
        sorted(set(prior['tracked_sources']) | set(pp['tracked_sources'])), p['raw_root'])
    h.verify(bindings); h.write(path, p)
    return p


def validate(protocol, freeze_commit, output):
    h = helpers(); p = h.read(protocol)
    if (p.get('schema_version') != SCHEMA or p.get('indices') != list(INDICES)
            or p.get('stages') != list(STAGES) or p.get('budgets') != BUDGET
            or p.get('prior_ablation_sha256') != ABLATION_PROTOCOL_SHA
            or p.get('prior_ablation_readback_sha256') != ABLATION_READBACK_SHA
            or p.get('fixed_articulation') is not True
            or p.get('scientific_acceptance_changed') is not False
            or p.get('reference_reproduction_limit_m') != 1e-12
            or p.get('face_normal_component_limit') != 1e-10):
        raise ValueError('Frozen full-surface method/budgets changed')
    h.verify(p['bindings']); prior, native, evidence = completed_ablation(); h.bootstrap(native)
    if h.runtime() != p['runtime']:
        raise ValueError('Runtime changed after surface freeze')
    if p['cases'] != {str(i): case_spec(i, prior, native) for i in INDICES}:
        raise ValueError('Saved inputs/shape/frame/native membership changed')
    if p['radius_policy'] != radius_selection(p['primitive_report'], p['primitive_helper']):
        raise ValueError('Primitive radius selection changed')
    pp, primitive_paths = primitive_prior(p['primitive_report'], p['primitive_helper'])
    required = source_inventory(p['cases'], p['primitive_report'], p['primitive_helper'], evidence, primitive_paths)
    if (p.get('measurement_sources') != sorted(map(str, required))
            or p['bindings'] != merged_bindings(prior, pp, required)
            or p['tracked_sources'] != h.source_partitions(p['bindings'], p['measurement_sources'],
                sorted(set(prior['tracked_sources']) | set(pp['tracked_sources'])), p['raw_root'])):
        raise ValueError('Full source/runtime/data closure or selected Git partition changed')
    for key in surface().FLAGS:
        if p.get(key) is not False: raise ValueError('Surface reference cannot admit production')
    if (p.get('field_rules') != prior['field_rules'] or p.get('geometry_base') != prior['geometry_base']
            or p.get('base_xml') != native['generated_assets']['variant_xmls']['mesh']):
        raise ValueError('Frozen evaluator or original generated XML changed')
    for path in evidence:
        if p['bindings'].get(str(path)) != h.sha(path): raise ValueError('Prior evidence omitted')
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', freeze_commit+'^{commit}'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('HEAD must remain at the exact surface freeze')
    for name in sorted(set(p['tracked_sources']) | {str(Path(protocol).resolve())}):
        path = Path(name).resolve(); relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', commit+':'+relative])
        if hashlib.sha256(blob).hexdigest() != h.sha(path): raise ValueError('Tracked source/protocol Git blob changed')
    roots(p, output)
    return p, commit


def scene_xml(base_path, include_path):
    """Only absolutize meshes/disable old collision bits/add the new include."""
    base_path = Path(base_path).resolve(); root = ET.parse(base_path).getroot()
    if root.tag != 'mujoco' or root.findall('.//include') or root.findall('.//deformable'):
        raise ValueError('Standalone original generated native XML required')
    meshes = root.findall('./asset/mesh')
    if len(meshes) != 16: raise ValueError('All original sixteen visual meshes required')
    for mesh in meshes:
        file = (base_path.parent/mesh.get('file')).resolve()
        if not file.is_file(): raise ValueError('Original bound mesh missing')
        mesh.set('file', str(file))
    bodies = {node.get('name'): node for node in root.findall('.//body')}
    if set(bodies) != {f'mano_body_{i}' for i in range(16)}:
        raise ValueError('Original sixteen native bodies required')
    before = {}
    for name, body in bodies.items():
        before[name] = [ET.tostring(node, encoding='unicode') for node in body if node.tag != 'geom' and node.tag != 'body']
        geoms = body.findall('./geom')
        if len(geoms) != 1 or geoms[0].get('type') != 'mesh':
            raise ValueError('Exactly sixteen original mesh geoms required')
        geoms[0].set('contype', '0'); geoms[0].set('conaffinity', '0')
    ET.SubElement(root, 'include', file=str(Path(include_path).resolve()))
    return ET.tostring(root, encoding='unicode')+'\n', dict(disabled_original_collision_geoms=16,
        original_body_joint_inertial_elements_sha256=helpers().digest(before),
        original_actuator_elements_sha256=helpers().digest([ET.tostring(x, encoding='unicode') for x in root.findall('./actuator')]),
        old_visual_meshes_retained=True, body_joint_inertia_actuator_parameters_changed=False)


class Budget:
    def __init__(self, query_count):
        self.query_count = query_count; self.models = self.data = self.forwards = 0
        self.stages = []; self.tests = 0

    def operation(self, kind):
        limit = {'models': 1, 'data': 1, 'forwards': 2}[kind]
        if getattr(self, kind) >= limit: raise RuntimeError('Static '+kind+' budget exhausted before call')
        setattr(self, kind, getattr(self, kind)+1)

    def scan(self, stage):
        if len(self.stages) >= 2 or stage != STAGES[len(self.stages)]:
            raise RuntimeError('Two fixed ordered full fields exhausted before query')
        self.stages.append(stage)

    def query(self, count):
        if type(count) is not int or count <= 0 or not self.stages or self.tests+count > self.query_count*3104:
            raise RuntimeError('Triangle-point budget exhausted before query')
        self.tests += count


def stage_saved_named(model, data, literal, mujoco):
    s = surface(); names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    if (model.nq, model.nv, model.nu, model.njnt) != (67, 51, 51, 19) or set(names) != set(s.JOINT_NAMES):
        raise ValueError('Original67/51/51 native state/actuator inventory changed')
    values = literal['named_qpos']; s.assert_locked_articulation(values, values)
    qcovered, vcovered, address = set(), set(), []
    for j, name in enumerate(names):
        slide = name in s.SLIDE_NAMES
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_SLIDE if slide else mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError('Actual named native joint type changed')
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]); nq, nv = (1, 1) if slide else (4, 3)
        qr, vr = set(range(q, q+nq)), set(range(v, v+nv))
        if q < 0 or v < 0 or qr & qcovered or vr & vcovered or q+nq > model.nq or v+nv > model.nv:
            raise ValueError('Named qpos/dof slices overlap or escape state')
        qcovered |= qr; vcovered |= vr
        data.qpos[q:q+nq] = np.asarray(values[name]).reshape(nq)
        address.append(dict(name=name, qpos_address=q, dof_address=v, type=int(model.jnt_type[j])))
    if qcovered != set(range(67)) or vcovered != set(range(51)):
        raise ValueError('Named state coverage is incomplete')
    data.qvel[:] = 0
    return address


def actual_named(model, data, mujoco):
    result = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j); q = int(model.jnt_qposadr[j])
        result[name] = float(data.qpos[q]) if name in surface().SLIDE_NAMES else data.qpos[q:q+4].copy()
    return result


def static_forward(model, data, original_forward, budget):
    if data.time != 0 or np.any(data.warning.number) or not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.ctrl)):
        raise ValueError('Clean finite zero-time state required before static forward')
    warning_numbers = data.warning.number.copy(); ctrl = data.ctrl.copy()
    budget.operation('forwards')
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always'); original_forward(model, data)
    if (data.time != 0 or not np.array_equal(data.warning.number, warning_numbers) or not np.array_equal(data.ctrl, ctrl)
            or emitted or not all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc, data.ctrl))):
        raise ValueError('Static forward warning/nonfinite/time/control integrity failure')


class Case:
    def __init__(self, index, p, raw, budget):
        self.index, self.p, self.raw, self.budget = index, p, Path(raw), budget
        self.artifact_rows = []; self.source_receipt = dict(index=index, fixed_articulation=True, **surface().FLAGS)

    def save(self, name, values, label):
        path = self.raw/name
        with path.open('xb') as f: np.savez_compressed(f, **values)
        row = surface().binding(path, label); self.artifact_rows.append(row); return row

    def scan(self, stage, geometry, actual, rules, kernel):
        tri, labels = geometry['triangles_object_m'], geometry['anatomical_part_index']
        if tri.shape != (1552, 3, 3) or labels.shape != (1552,) or not np.isfinite(tri).all():
            raise ValueError('Complete finite1552-face geometry required')
        g = self.save(stage+'_geometry.npz', geometry, stage+'_sealed_prequery_geometry')
        self.budget.scan(stage); distances = np.empty_like(actual['native_distance_by_part_m'])
        for start in range(0, len(distances), 256):
            points = actual['query_object_m'][start:start+256]
            self.budget.query(len(points)*1552)
            distances[start:start+len(points)] = kernel.nearest_parts(points, tri, labels, 6)[0]
        if not np.isfinite(distances).all(): raise ValueError('Nonfinite complete field')
        fields, summary = rules.field_status(actual['native_distance_by_part_m'], distances, actual['cell_radius_m'])
        if int(fields['required'].sum()) != int(actual['required'].sum()): raise ValueError('Native relationship denominator changed')
        artifact = self.save(stage+'_field.npz', dict(query_object_m=actual['query_object_m'],
            cell_radius_m=actual['cell_radius_m'], native_distance_by_part_m=actual['native_distance_by_part_m'],
            mapped_distance_by_part_m=distances, **fields), stage+'_complete_fixed_field')
        return dict(stage_id=stage, geometry_artifact=g, field_artifact=artifact, summary=summary,
                    physical_outcome=None, **surface().FLAGS), distances, fields

    def run(self, original_model, original_data, original_forward, mujoco, reinstall):
        h, s, p = helpers(), surface(), self.p
        old_case = h.read(Path(p['ablation_run'])/str(self.index)/'result.json')
        paths = {Path(row['path']).name: row['path'] for row in old_case['artifacts']}
        reference = s.arrays(paths['native_reference_geometry.npz'])
        actual = s.arrays(paths['native_reference_field.npz'])
        kernel = load('locked_surface_frozen_distance_kernel', p['geometry_base']); reinstall()
        kernel.load_case = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('No historical target-hand constructor'))
        rules = load('locked_surface_frozen_field_rules', p['field_rules']); reinstall()
        import numba
        numba.set_num_threads(2)
        row, distance, fields = self.scan(STAGES[0], reference, actual, rules, kernel)
        old_module = load('locked_surface_saved_reference_control', ABLATION/'ablation_case.py')
        control = old_module.reference_control(distance, actual['native_distance_by_part_m'], actual['cell_radius_m'], rules, actual)
        self.source_receipt['reference_control'] = control
        if not control['passed']: raise ValueError('Reference reproducibility failed before candidate construction')
        prepared, prep_binding = s.prepare_saved_candidate(self.index, p['cases'][str(self.index)], self.raw/'candidate', radius_policy=p['radius_policy'])
        self.artifact_rows.extend(prepared['artifacts'] + [prep_binding])
        expected = s.arrays(self.raw/'candidate/candidate_surface.npz')
        nr = h.read(Path(p['native_run'])/str(self.index)/'result.json')
        pos, rot = np.asarray(nr['source_receipt']['object_reference_world_position_m']), np.asarray(nr['source_receipt']['object_reference_world_rotation'])
        frozen_tri = np.einsum('ni,ij->nj', expected['native_vertices_world_m']-pos, rot, optimize=False)[expected['native_faces']]
        if (not np.array_equal(frozen_tri, reference['triangles_object_m'])
                or not np.array_equal(expected['native_face_labels'], reference['anatomical_part_index'])
                or not np.array_equal(expected['native_face_ids'], reference['native_face_id'])):
            raise ValueError('Candidate changed the frozen source surface/face membership')
        xml, assembly = scene_xml(p['base_xml'], self.raw/'candidate/candidate_flex.xml')
        xml_path = self.raw/'compiled_scene.xml'
        with xml_path.open('x') as f: f.write(xml)
        self.artifact_rows.append(s.binding(xml_path, 'new_scene_only_old_collision_bits_disabled_and_flex_added'))
        self.source_receipt['scene_assembly'] = assembly
        self.budget.operation('models'); model = original_model.from_xml_path(str(xml_path))
        self.budget.operation('data'); data = original_data(model)
        static_forward(model, data, original_forward, self.budget)
        literal = h.read(p['cases'][str(self.index)]['inputs']['literal_pose']['path'])
        address = stage_saved_named(model, data, literal, mujoco)
        static_forward(model, data, original_forward, self.budget)
        named = actual_named(model, data, mujoco)
        for name in s.JOINT_NAMES:
            if not np.array_equal(np.asarray(named[name]), np.asarray(literal['named_qpos'][name])):
                raise ValueError('Actual saved named qpos changed during static forward')
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s.OWNER_BODY)
        if body < 0 or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, 0) != s.FLEX_NAME:
            raise ValueError('Actual compiled owner/flex name changed')
        if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
            raise ValueError('Old mesh collisions remain active beside native skin')
        values, readback = s.compiled_surface_readback(model, data, expected, owner_body_id=body,
            radius_m=p['radius_policy']['radius_m'], saved_named=literal['named_qpos'], actual_named=named)
        kin = s.arrays(p['cases'][str(self.index)]['inputs']['kinematic_readback']['path'])
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'mano_body_{i}') for i in range(16)]
        if min(ids) < 0: raise ValueError('Native body readback inventory changed')
        if not all(np.isfinite(value).all() for value in (data.xpos[ids], data.xmat[ids], data.xanchor)):
            raise ValueError('Nonfinite actual native body/anchor readback')
        body_error = max(float(np.max(np.abs(data.xpos[ids]-kin['compiled_body_origins_world_m']))),
                         float(np.max(np.abs(data.xmat[ids].reshape(16,3,3)-kin['compiled_world_body_rotations']))))
        if body_error > 1e-12: raise ValueError('New scene changed original saved native body frames')
        values.update(compiled_body_origins_world_m=data.xpos[ids].copy(), compiled_world_body_rotations=data.xmat[ids].reshape(16,3,3).copy(),
                      compiled_joint_anchors_world_m=data.xanchor.copy(), literal_qpos=data.qpos.copy(), literal_qvel=data.qvel.copy(),
                      retained_ctrl=data.ctrl.copy(), body_mass=model.body_mass.copy(), body_inertia=model.body_inertia.copy(),
                      flex_radius=model.flex_radius.copy(), flex_margin=model.flex_margin.copy(), flex_gap=model.flex_gap.copy())
        rb = self.save('compiled_native_surface_readback.npz', values, 'actual_compiled_skin_vertices_topology_normals_and_native_body_state')
        self.source_receipt.update(candidate_receipt=prepared, compiled_readback=readback,
            compiled_readback_artifact=rb, named_joint_addresses=address, maximum_saved_body_frame_error=body_error)
        tri = np.einsum('ni,ij->nj', values['compiled_vertices_world_m']-pos, rot, optimize=False)[values['compiled_faces']]
        geometry = dict(triangles_object_m=tri, anatomical_part_index=expected['native_face_labels'],
                        native_face_id=expected['native_face_ids'], source_vertex_ids=expected['native_faces'])
        last, final_distance, final_fields = self.scan(STAGES[1], geometry, actual, rules, kernel)
        return dict(schema_version=SCHEMA, source_inventory_index=self.index, stages=[row, last],
            reference_control=control, compiled_readback=readback, fixed_articulation=True,
            surface_fidelity_status='FAIL' if (final_fields['definite_lost']|final_fields['definite_added']).any() else 'INCOMPLETE',
            maximum_distance_delta_from_reference_m=float(np.max(np.abs(final_distance-distance))),
            full_orientation_status='NOT_RUN', corresponding_face_normals_only=True,
            continuous_surface_equivalence='NOT_PROVED', physical_outcome=None, **s.FLAGS)


def worker(protocol, freeze_commit, output, index):
    p, commit = validate(protocol, freeze_commit, output); h, s = helpers(), surface()
    tracked, raw = roots(p, output); directory, raw_case = tracked/str(index), raw/str(index)
    directory.mkdir(parents=True, exist_ok=False); raw_case.mkdir(parents=True, exist_ok=False)
    import mujoco
    original_model, original_data, original_forward = mujoco.MjModel, mujoco.MjData, mujoco.mj_forward
    forbidden = {key: 0 for key in h.FORBIDDEN_KEYS}
    reinstall = lambda: h.install_guards(forbidden)
    reinstall(); finder = h.ForbiddenImports(forbidden); sys.meta_path.insert(0, finder)
    actual = s.arrays(Path(p['ablation_raw'])/str(index)/'native_reference_field.npz')
    budget = Budget(len(actual['query_object_m'])); case = Case(index, p, raw_case, budget)
    failure, measurement, artifacts, partial = None, None, [], None; started = time.perf_counter()
    try:
        measurement = case.run(original_model, original_data, original_forward, mujoco, reinstall)
        if (budget.models, budget.data, budget.forwards, budget.stages, budget.tests) != (1, 1, 2, list(STAGES), budget.query_count*3104):
            raise ValueError('Actual static counts disagree with fixed pre-work guards')
        if any(forbidden.values()): raise ValueError('Forbidden model/physics/generation/training operation attempted')
        artifacts = h.artifact_index((directory, raw_case), case.artifact_rows, complete=True)
        validate(protocol, freeze_commit, output)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        (directory/'exception.txt').write_text(traceback.format_exc())
        partial = h.partial_artifact_receipt((directory, raw_case), case)
    if finder in sys.meta_path: sys.meta_path.remove(finder)
    result = dict(schema_version=SCHEMA, source_inventory_index=index, protocol_sha256=h.sha(protocol), freeze_commit=commit,
        status='COMPLETED_LOCKED_SURFACE_STATIC_REFERENCE' if failure is None else 'HARD_STOP',
        elapsed_s=time.perf_counter()-started, failure=failure, measurement=measurement,
        guarded_counts=dict(models=budget.models, data=budget.data, forwards=budget.forwards,
            full_field_stages=budget.stages, triangle_point_tests=budget.tests), forbidden_counts=forbidden,
        artifacts=artifacts, partial_artifacts=partial, source_receipt=case.source_receipt,
        physical_outcome=None, **s.FLAGS)
    h.write(raw_case/'result.json', result)
    with (directory/'result.json').open('xb') as f: f.write((raw_case/'result.json').read_bytes())
    return int(failure is not None)


def run(protocol, freeze_commit, output):
    p, commit = validate(protocol, freeze_commit, output); h = helpers(); tracked, raw = roots(p, output)
    if tracked.exists() or raw.exists(): raise FileExistsError('Never retry/overwrite a surface run')
    tracked.mkdir(parents=True); raw.mkdir(parents=True)
    h.write(tracked/'started.json', dict(protocol_sha256=h.sha(protocol), freeze_commit=commit, budgets=BUDGET))
    rows = []
    for index in INDICES:
        validate(protocol, commit, output); log = tracked/f'case_{index}.log'; started = time.perf_counter()
        with log.open('x') as f:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--protocol', str(Path(protocol).resolve()),
                '--freeze-commit', commit, '--output', str(tracked), '--index', str(index)], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = child.wait(timeout=600)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait(); code = None
        result = tracked/str(index)/'result.json'; raw_result = raw/str(index)/'result.json'
        good = (code == 0 and result.is_file() and raw_result.is_file() and result.read_bytes() == raw_result.read_bytes()
                and h.read(result).get('status') == 'COMPLETED_LOCKED_SURFACE_STATIC_REFERENCE')
        rows.append(dict(index=index, status='COMPLETED' if good else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR', returncode=code,
            elapsed_s=time.perf_counter()-started, log=str(log), log_sha256=h.sha(log), result=str(result) if result.exists() else None,
            result_sha256=h.sha(result) if result.exists() else None, raw_result=str(raw_result) if raw_result.exists() else None,
            raw_result_sha256=h.sha(raw_result) if raw_result.exists() else None))
        if not good: break
    seen = {row['index'] for row in rows}; rows.extend(dict(index=i, status='NOT_RUN') for i in INDICES if i not in seen)
    validate(protocol, commit, output)
    verify_completed_outputs(rows, tracked, raw)
    report = dict(schema_version=SCHEMA, protocol_sha256=h.sha(protocol), freeze_commit=commit, cases=rows,
        status='COMPLETED_LOCKED_NATIVE_SURFACE_PILOT' if all(row['status'] == 'COMPLETED' for row in rows) else 'HARD_STOP',
        physical_outcome=None, **surface().FLAGS)
    h.write(tracked/'report.json', report); print(json.dumps(report), flush=True)
    return int(report['status'] == 'HARD_STOP')


def verify_completed_outputs(rows, tracked, raw):
    """Recheck earlier child write-time receipts before publishing the report."""
    h = helpers(); artifacts = []
    for row in rows:
        if row['status'] != 'COMPLETED': continue
        result, raw_result = Path(row['result']), Path(row['raw_result'])
        if (h.sha(result) != row['result_sha256'] or h.sha(raw_result) != row['raw_result_sha256']
                or result.read_bytes() != raw_result.read_bytes()):
            raise ValueError('Earlier completed result changed before final publication')
        artifacts.extend(h.read(result)['artifacts'])
    return h.artifact_index((Path(tracked), Path(raw)), artifacts,
                            complete=all(row['status'] == 'COMPLETED' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    q = commands.add_parser('prepare'); q.add_argument('--protocol', type=Path, required=True)
    q.add_argument('--primitive-report', type=Path, required=True); q.add_argument('--primitive-helper', type=Path, default=PRIMITIVE/'run_qualification.py')
    for name in ('run', 'worker'):
        q = commands.add_parser(name); q.add_argument('--protocol', type=Path, required=True)
        q.add_argument('--freeze-commit', required=True); q.add_argument('--output', type=Path, required=True)
        if name == 'worker': q.add_argument('--index', type=int, choices=INDICES, required=True)
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.protocol, a.primitive_report, a.primitive_helper); return 0
    return run(a.protocol, a.freeze_commit, a.output) if a.command == 'run' else worker(a.protocol, a.freeze_commit, a.output, a.index)


if __name__ == '__main__': raise SystemExit(main())
