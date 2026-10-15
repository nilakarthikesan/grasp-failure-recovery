"""Frozen current-scene full-field distance diagnostic; never a mapping PASS.

Preparation reads existing arrays/metadata only. Execution compiles the current
registered scene and evaluates literal individual component surfaces. It never
loads a historical MJB or the historical constraint-engine Case constructor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = Path('/Users/nilakarthikesan/grasp-failure-prediction')
SCHEMA = 'hug_current_full_field_distance_diagnostic_v1'
PARTS = ('palm', 'index', 'middle', 'ring', 'little', 'thumb')
INDICES = (20, 28)
BANDS = (1, 2, 5)
ALLOWANCE = 1e-8
MAX_SECONDS_PER_CASE = 600
MAX_FORWARDS_PER_CASE = 48
CHUNK = 256
OLD = ROOT / 'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
WITNESS = ROOT / 'research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/current_geometry_witness'
COUNTS_FILE = ROOT / 'research_artifacts/hug_surface_constraints_2026-10-15_v1/witness_provenance/cached_failure_row_counts.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(plain(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def verify(bindings):
    for name, expected in bindings.items():
        if sha(name) != expected: raise ValueError('Bound bytes changed: ' + name)


def classifications(distance, radius, bands=BANDS, allowance=ALLOWANCE):
    """Literal inherited inclusive near / strict far whole-cell predicates."""
    d = np.asarray(distance, float)
    r = np.asarray(radius, float)
    if d.ndim != 2 or d.shape[1] != 6 or r.shape != (len(d),):
        raise ValueError('Six anatomical parts and aligned cell radii required')
    if not np.isfinite(d).all() or not np.isfinite(r).all() or np.any(d < 0) or np.any(r < 0):
        raise ValueError('Finite nonnegative distances/radii required')
    if tuple(bands) != BANDS or allowance != ALLOWANCE:
        raise ValueError('Frozen bands and allowance may not be changed')
    error = r[:, None, None] + allowance
    b = np.asarray(bands)[None, None, :] / 1000.
    low = np.maximum(0., d[:, :, None] - error)
    high = d[:, :, None] + error
    near = high <= b
    far = low > b
    return dict(near=near, far=far, ambiguous=~(near | far), low=low, high=high)


def field_status(native_distance, mapped_distance, radius, *, baseline_mapped=None, historical_mapped=None):
    n = classifications(native_distance, radius)
    m = classifications(mapped_distance, radius)
    required = n['near'] | n['far']
    preserved = (n['near'] & m['near']) | (n['far'] & m['far'])
    lost = n['near'] & m['far']
    added = n['far'] & m['near']
    unresolved = required & ~preserved & ~(lost | added)
    arrays = dict(native_near=n['near'], native_far=n['far'], native_ambiguous=n['ambiguous'],
                  required=required, preserved=preserved, definite_lost=lost,
                  definite_added=added, unresolved_required=unresolved)
    if baseline_mapped is not None:
        baseline = field_status(native_distance, baseline_mapped, radius)[0]
        arrays['lost_protected_passing'] = baseline['preserved'] & ~preserved
        arrays['new_definite_violation'] = (lost | added) & ~(baseline['definite_lost'] | baseline['definite_added'])
        arrays['repaired_original_definite'] = (baseline['definite_lost'] | baseline['definite_added']) & preserved
    if historical_mapped is not None:
        historical = field_status(native_distance, historical_mapped, radius)[0]
        old = historical['definite_lost'] | historical['definite_added']
        arrays['historical_definite_mismatch'] = old
        arrays['historical_mismatch_still_definite'] = old & (lost | added)
        arrays['historical_mismatch_now_preserved'] = old & preserved
        arrays['historical_mismatch_now_unresolved'] = old & unresolved
    summary = {name: int(value.sum()) for name, value in arrays.items()}
    summary.update(native_definite_relationships_pass=bool(np.all(~required | preserved)),
                   full_strict_mapping_status='FAIL' if np.any(lost | added) else 'INCOMPLETE',
                   full_orientation_status='NOT_RUN', continuous_coverage_status='NOT_PROVED',
                   no_mapping_pass_issued=True)
    return arrays, summary


def constraint_rows(native_distance, radius):
    """Complete deterministic native-definite inventory, never a capped sample."""
    n = classifications(native_distance, radius)
    cell, part, band = np.nonzero(n['near'] | n['far'])
    return dict(cell_index=cell, part_index=part, band_index=band,
                required_near=n['near'][cell, part, band],
                band_m=np.asarray(BANDS, float)[band]/1000.)


def runtime_identity():
    import mujoco, scipy
    return dict(python=platform.python_version(), numpy=np.__version__,
                mujoco=mujoco.__version__, scipy=scipy.__version__, executable=sys.executable)


def prepare(protocol_path):
    """No project import, model construction, FK, geometry query or solve."""
    old = read(WITNESS / 'protocol.json')
    sampling = read(OLD / 'sampling/protocol.json')
    counts = read(COUNTS_FILE)
    if sampling['bands_mm'] != list(BANDS) or sampling['primary_allowance_m'] != ALLOWANCE:
        raise ValueError('Original scientific scale changed')
    snapshot_sources = {
        'scene_runtime.py': '/private/tmp/hug_current_scene_bridge_2026-10-15_v1/scene_runtime.py',
        'xml_reference.py': '/private/tmp/hug_current_execution_bridge_2026-10-15_v1/xml_reference.py',
        'native_reference.py': '/private/tmp/hug_current_scene_bridge_2026-10-15_v1/native_reference.py',
        'audit_other_surfaces.py': str(ROOT / 'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py'),
        'witness_check.py': str(WITNESS / 'witness_check.py'),
        'cell_intervals.py': str(OLD / 'sampling/cell_intervals.py'),
    }
    snapshots = {}
    for name, source in snapshot_sources.items():
        path = HERE / 'source_snapshots' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = Path(source).read_bytes()
        if path.exists() and path.read_bytes() != data: raise ValueError('Snapshot already differs')
        if not path.exists(): path.write_bytes(data)
        snapshots[name] = dict(source=source, sha256=sha(source), snapshot=str(path), snapshot_sha256=sha(path))
    # Preserve actual imported current source bytes, even for unrelated existing
    # dirty files transitively imported by case_runner. Never overwrite them.
    # The separately prepared30D adapter is neither imported nor exercised by
    # v1 and must not couple this frozen audit to concurrent v2 development.
    project_sources = sorted(path for path in (ROOT / 'src/grasp_failure_prediction').rglob('*')
        if path.is_file() and path.suffix in {'.py','.yaml','.yml','.json'} and path.name!='surface_pose.py')
    for source in project_sources:
        path = HERE / 'source_snapshots/current_project' / source.relative_to(ROOT / 'src')
        path.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if path.exists() and path.read_bytes() != data: raise ValueError('Current snapshot already differs')
        if not path.exists(): path.write_bytes(data)
        snapshots[str(source.relative_to(ROOT))] = dict(source=str(source), sha256=sha(source), snapshot=str(path), snapshot_sha256=sha(path))
    input_paths = {str(WITNESS / 'protocol.json'), str(COUNTS_FILE), str(OLD / 'sampling/protocol.json'),
                   str(OLD / 'acceptance/acceptance_rule.json'), old['inventory'], old['source_urdf']}
    # Existing hand/object/runtime assets remain read-only and hash-bound; no
    # licensed asset or historical binary is copied into this diagnostic.
    for name in old['bindings']:
        if '/third_party/' in name or ('/runs/' in name and Path(name).suffix in {'.xml','.obj','.stl','.dae','.json'}):
            input_paths.add(name)
    cases = []
    inventory = read(old['inventory'])
    for index in INDICES:
        case = next(c for c in sampling['cases'] if c['source_inventory_index'] == index)
        source = inventory['modern_cases'][index]
        input_paths.update((source['native']['path'], source['resolved_case_path'], source['frame_files']['T_world_camera.npy']['path'], source['frame_files']['scene_contract.json']['path']))
        poses = [p for p in old['poses'] if p['source_inventory_index'] == index]
        if [p['pose_id'] for p in poses]!=[f'original_{index}',f'experimental_{index}']:
            raise ValueError('Exactly original then experimental saved poses required')
        for pose in poses:
            input_paths.add(pose['historical_cap_report'])
            input_paths.add(str(WITNESS / 'cases' / (pose['pose_id'] + '.json')))
        grids = []
        for density in case['densities_mm']:
            name = f"{case['case_id']}_density{density}mm"
            target = Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral') / (name + '_full_current.npz')
            cache = Path('/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces') / (name + '_queries.npz')
            binding = target.with_suffix('.binding.json')
            b = read(binding)
            if sha(target) != b['ephemeral_npz_sha256'] or sha(cache) != b['original_cache_sha256']:
                raise ValueError('Original full-grid cache binding changed')
            with np.load(target, allow_pickle=False) as t, np.load(cache, allow_pickle=False) as n:
                q = t['query_object_m']; r = t['cell_radius_m']; nd = n['native_distance_by_part_m'][:,:6]
                if not np.array_equal(q, n['query_object_m']): raise ValueError('Native/cell query order differs')
                original = field_status(nd, t['current_distance_by_part_m'][:,:6], r)[1]
                expected = next(c for c in counts['cases'] if c['source_inventory_index']==index)['densities'][str(density)]['failure_rows']
                if original['definite_lost'] + original['definite_added'] != expected:
                    raise ValueError('Full original mismatch inventory does not reproduce')
                grids.append(dict(density_mm=density, native_cache=str(cache), original_cache=str(target),
                    cache_binding=str(binding), query_count=len(q), required_relationships=len(constraint_rows(nd,r)['cell_index']),
                    original_definite_mismatch_tuples=expected))
            input_paths.update(map(str,(target,cache,binding)))
        cases.append(dict(source_inventory_index=index, native_sha256=case['native_sha256'],
                          poses=poses, grids=grids))
    binding = {name:sha(name) for name in sorted(input_paths)}
    binding.update({v['source']:v['sha256'] for v in snapshots.values()})
    binding.update({v['snapshot']:v['snapshot_sha256'] for v in snapshots.values()})
    binding[str(Path(__file__).resolve())] = sha(__file__)
    test = ROOT / 'tests/test_hug_full_field_mapping.py'
    binding[str(test)] = sha(test)
    value = dict(schema_version=SCHEMA, prepared_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases, source_snapshots=snapshots, bindings=binding, runtime=runtime_identity(),
        inventory=old['inventory'], source_urdf=old['source_urdf'], runtime_project_root=str(RUNTIME),
        dependency_dirs=[str(RUNTIME/'runs/hug_integration_check'/n) for n in ('hug-deps','deps','deps/cmeel.prefix/lib/python3.12/site-packages')],
        budgets=dict(maximum_seconds_per_case=MAX_SECONDS_PER_CASE,maximum_forwards_per_case=MAX_FORWARDS_PER_CASE,
                     query_chunk_size=CHUNK,cases=2,poses=4,retarget_solves=0,physics_steps=0,inference_calls=0),
        scientific_rule_sha256=sha(OLD/'acceptance/acceptance_rule.json'), bands_mm=list(BANDS), numerical_allowance_m=ALLOWANCE,
        geometry='CURRENT registered ActuatedShadowRunner; same-part literal component surfaces, not exposed/contact boundary',
        coverage='ALL native-definite near/far relationships on complete original grids; native-ambiguous rows retained; no orientation claim',
        solver_status='BLOCKED_CURRENT_SCENE_30D_ADAPTER_NOT_QUALIFIED',
        solver_reason='Historical feasible Case constructs old geometry and recomputes caps there. No old overlay Case or capped178 solve is run.',
        thresholds_changed=False,historical_caps_rebased=False,requires_committed_protocol_and_snapshots=True,
        source_binding_policy='Current sources hash-bound, source snapshots committed; dirty input sources are preserved, not staged by this tool.')
    write(protocol_path,value)
    return value


def committed(protocol_path, p, freeze_commit):
    actual = subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'], text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=actual:
        raise ValueError('Run from exact declared freeze commit')
    paths = [Path(protocol_path).resolve(),Path(__file__).resolve(),ROOT/'tests/test_hug_full_field_mapping.py']
    paths += [Path(v['snapshot']) for v in p['source_snapshots'].values()]
    relative = [str(path.relative_to(ROOT)) for path in paths]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*relative],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*relative],check=True)
    return actual


def validate_case_plan(p,old,sampling,counts):
    for case in p['cases']:
        index=case['source_inventory_index']
        expected=[pose for pose in old['poses'] if pose['source_inventory_index']==index]
        if (case['poses']!=expected or [pose['pose_id'] for pose in expected]!=[f'original_{index}',f'experimental_{index}']):
            raise ValueError('Original then experimental saved pose definitions required')
        original=next(c for c in sampling['cases'] if c['source_inventory_index']==index)
        counted=next(c for c in counts['cases'] if c['source_inventory_index']==index)
        if case['native_sha256']!=original['native_sha256'] or [g['density_mm'] for g in case['grids']]!=original['densities_mm']:
            raise ValueError('Full original grid domain/native identity required')
        for grid in case['grids']:
            name=f"{original['case_id']}_density{grid['density_mm']}mm"
            expected_paths=dict(native_cache=str(Path('/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces')/(name+'_queries.npz')),
                original_cache=str(Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral')/(name+'_full_current.npz')),
                cache_binding=str(Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral')/(name+'_full_current.binding.json')))
            for key,value in expected_paths.items():
                if grid[key]!=value or value not in p['bindings']:raise ValueError('Full grid input identity/closure changed')
            historical=counted['densities'][str(grid['density_mm'])]
            if grid['query_count']!=historical['query_count'] or grid['original_definite_mismatch_tuples']!=historical['failure_rows']:
                raise ValueError('Original full-field row count changed')


def validate(protocol_path, freeze_commit):
    p = read(protocol_path)
    if p['schema_version']!=SCHEMA or p['bands_mm']!=list(BANDS) or p['numerical_allowance_m']!=ALLOWANCE:
        raise ValueError('Wrong diagnostic declaration')
    if [c['source_inventory_index'] for c in p['cases']]!=list(INDICES) or p['budgets']!=dict(
        maximum_seconds_per_case=600,maximum_forwards_per_case=48,query_chunk_size=256,cases=2,poses=4,retarget_solves=0,physics_steps=0,inference_calls=0):
        raise ValueError('Changed finite audit budget or cohort')
    validate_case_plan(p,read(WITNESS/'protocol.json'),read(OLD/'sampling/protocol.json'),read(COUNTS_FILE))
    if p['scientific_rule_sha256']!='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4':
        raise ValueError('Strict scientific rule changed')
    verify(p['bindings'])
    if runtime_identity()!=p['runtime']: raise ValueError('Runtime changed')
    return p, committed(protocol_path,p,freeze_commit)


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(protocol_path, freeze_commit, output, index):
    p,commit=validate(protocol_path,freeze_commit)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    import mujoco
    counts=dict(forwards=0,physics_steps=0,retarget_solves=0,inference_calls=0,collider_point_queries=0)
    def forbidden(kind):
        def fail(*a,**kw): counts[kind]+=1;raise RuntimeError('Static full-field audit forbids '+kind)
        return fail
    for name in ('mj_step','mj_step1','mj_step2'): setattr(mujoco,name,forbidden('physics_steps'))
    original_forward=mujoco.mj_forward
    def forward(*a,**kw):
        counts['forwards']+=1
        if counts['forwards']>MAX_FORWARDS_PER_CASE: raise RuntimeError('Static forward budget exhausted')
        return original_forward(*a,**kw)
    mujoco.mj_forward=forward
    sys.path.insert(0,str(ROOT/'src'));sys.path.extend(p['dependency_dirs']);sys.path.insert(0,str(HERE/'source_snapshots'))
    # Import pinned mathematical kernels without historical model constructors.
    import scene_runtime,xml_reference,native_reference,witness_check,cell_intervals
    a=load_module('full_field_geometry',HERE/'source_snapshots/audit_other_surfaces.py')
    for name in ('mj_step','mj_step1','mj_step2'): setattr(mujoco,name,forbidden('physics_steps'))
    a.load_case=forbidden('inference_calls')
    if Path(a.URDF).resolve()!=Path(p['source_urdf']).resolve():raise ValueError('Geometry library uses another source hand')
    from grasp_failure_prediction.evaluation import case_runner,retargeting,pose_validation,runner as runners
    if 'grasp_failure_prediction.evaluation.surface_pose' in sys.modules:raise ValueError('v1 must not import the separately developed30D adapter')
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    retargeting.ShadowHandRetargeter.retarget=forbidden('retarget_solves')
    case_runner._run_case_inference=forbidden('inference_calls')
    case_runner.run_case=forbidden('inference_calls')
    for cls in (runners.AdroitShadowRunner,runners.ActuatedShadowRunner):
        cls.execute=forbidden('physics_steps');cls._advance=forbidden('physics_steps')
    import numba
    numba.set_num_threads(2)
    assert numba.get_num_threads()==2
    assert Path(pose_validation.__file__).resolve()==ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py'
    case=next(c for c in p['cases'] if c['source_inventory_index']==index)
    source=read(p['inventory'])['modern_cases'][index]
    if source['native']['sha256']!=case['native_sha256'] or sha(source['native']['path'])!=case['native_sha256']:
        raise ValueError('Native proposal identity changed')
    import pickle
    native=pickle.loads(Path(source['native']['path']).read_bytes())
    camera=np.load(source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
    native_tasks=np.asarray(native['landmarks_3d'])@camera[:3,:3].T+camera[:3,3]
    native_tasks=native_tasks[[4,8,12,16,20,2,6,10,14,18]]
    payload=read(source['resolved_case_path'])['case'];contract=read(source['frame_files']['scene_contract.json']['path'])
    runtime,resolved,_=scene_runtime.build_current_runner(payload,p['runtime_project_root'],Path(p['source_urdf']).parent.parent)
    if resolved.environment.runner!='grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner': raise ValueError('Wrong current runner')
    scene_runtime.reset_captured_scene(runtime,payload,contract)
    m,d=runtime.model,runtime.data
    source_xml=xml_reference.read_urdf(p['source_urdf']);validator=pose_validation.ShadowPoseValidator(Path(p['source_urdf']).parent.parent)
    object_position=d.xpos[runtime._object_body_id].copy();object_rotation=d.xmat[runtime._object_body_id].reshape(3,3).copy()
    baseline={};records=[];started=time.perf_counter()
    for pose in case['poses']:
        validate(protocol_path,freeze_commit)
        names=pose['joint_names'];q=np.asarray(pose['qpos_rad'],float)
        limits=xml_reference.source_limit_rows(source_xml,names,q)
        if not all(row['literal_pass'] for row in limits): raise ValueError('Literal source joint limit violated')
        if pose['pose_id'].startswith('original_'):
            grasp=load_hug_prediction(Path(source['native']['path']))
            mapped=retargeting.RetargetedHandPose(tuple(names),q,np.zeros((21,3)),np.zeros((10,3)),np.asarray(native['T_camera_wrist']))
            pos,quat=case_runner._hug_world_palm_pose(grasp,mapped,validator,Path(source['native']['path']).parent)
            fixed=xml_reference.forward_xml(source_xml,names,q)
            expected=native_reference.expected_native_palm(camera,native['T_camera_wrist'],fixed['palm'])['palm_world']
            actual=xml_reference.transform(pos,xml_reference.quaternion_matrix_wxyz(quat))
            if np.max(np.abs(actual-expected))>1e-12: raise ValueError('Independent native frame disagrees')
            rootpos,rootquat,_=scene_runtime.stage_named_palm(runtime,names,q,actual)
        else:
            rootpos=np.asarray(pose['root_position_m']);rootquat=np.asarray(pose['root_quaternion_wxyz'])
            scene_runtime.stage_named_root(runtime,names,q,rootpos,rootquat)
        carrier=xml_reference.transform(rootpos,xml_reference.quaternion_matrix_wxyz(rootquat))
        fk=xml_reference.forward_xml(source_xml,names,q,carrier_world=carrier)
        palm_error=float(np.max(np.abs(fk['palm']-xml_reference.transform(d.xpos[runtime._palm_body_id],d.xmat[runtime._palm_body_id].reshape(3,3)))))
        if palm_error>1e-12: raise ValueError('Independent current/source FK disagrees')
        if d.time!=0 or np.max(np.abs(d.xpos[runtime._object_body_id]-object_position))>1e-12 or np.max(np.abs(d.xmat[runtime._object_body_id].reshape(3,3)-object_rotation))>1e-12:
            raise ValueError('Object state/time changed during static placement')
        errors=pose_validation._link_positions(m,d,Path(p['source_urdf']))-native_tasks
        caps=np.asarray(pose['historical_tip_caps_m'])
        if not np.array_equal(caps,read(pose['historical_cap_report'])['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m']):
            raise ValueError('Historical cap rebasing forbidden')
        norms=np.sqrt(np.einsum('ij,ij->i',errors[:5],errors[:5]))
        forearm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm')
        hand=[]
        for g in range(m.ngeom):
            body=int(m.geom_bodyid[g])
            if witness_check.descendant(m.body_parentid,body,forearm) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                if m.geom_margin[g]!=0 or m.geom_gap[g]!=0: raise ValueError('Nonzero hand margin/gap')
                rep=a.geom_rep(m,d,g,object_rotation,object_position)
                rep['part']=witness_check.anatomical_part(rep['body']);hand.append(rep)
        if len(hand)!=45: raise ValueError('Current hand collider inventory changed')
        pose_record=dict(pose_id=pose['pose_id'],source_inventory_index=index,native_sha256=case['native_sha256'],
            qpos_rad=q,root_position_m=rootpos,root_quaternion_wxyz=rootquat,source_limit_rows=limits,
            palm_source_fk_max_abs_error=palm_error,original_tip_caps_m=caps,tip_error_m=norms,
            all_literal_tip_caps_pass=bool(np.all(norms<=caps)),mean_tip_guard_pass=bool(norms.mean()<=.010),
            historical_caps_rebased=False,grids=[],hand_geometry_metadata=[{k:g[k] for k in ('id','body','part','kind','representation','mesh')} for g in hand])
        for grid in case['grids']:
            verify(p['bindings'])
            with np.load(grid['original_cache'],allow_pickle=False) as old, np.load(grid['native_cache'],allow_pickle=False) as native_cache:
                query=old['query_object_m'].copy();radius=old['cell_radius_m'].copy();area=old['area_mm2'].copy()
                nd=native_cache['native_distance_by_part_m'][:,:6].copy();historical=old['current_distance_by_part_m'][:,:6].copy()
                if not np.array_equal(query,native_cache['query_object_m']) or not np.array_equal(area,native_cache['area_mm2']): raise ValueError('Cached grid alignment changed')
            mapped=np.empty((len(query),6))
            for j,part in enumerate(PARTS):
                colliders=[g for g in hand if g['part']==part]
                if not colliders: raise ValueError('Missing anatomical collider')
                for start in range(0,len(query),CHUNK):
                    points=query[start:start+CHUNK];distance=np.full(len(points),np.inf)
                    for g in colliders:
                        signed,_,_=a.geom_query(g,points)
                        if not np.isfinite(signed).all(): raise ValueError('Nonfinite actual surface distance')
                        distance=np.minimum(distance,np.abs(signed));counts['collider_point_queries']+=len(points)
                    mapped[start:start+len(points),j]=distance
                print(json.dumps(dict(progress='part_complete',index=index,pose=pose['pose_id'],part=part)),flush=True)
            key=grid['density_mm']
            if not pose['pose_id'].startswith('original_') and key not in baseline:
                raise ValueError('Experimental comparison requires original-current full-field baseline')
            masks,summary=field_status(nd,mapped,radius,baseline_mapped=baseline.get(key),historical_mapped=historical)
            if pose['pose_id'].startswith('original_'): baseline[key]=mapped.copy()
            ledger=constraint_rows(nd,radius)
            if len(ledger['cell_index'])!=grid['required_relationships']: raise ValueError('Full relationship inventory changed')
            arrays=out/(pose['pose_id']+f'_density{key}mm.npz')
            with arrays.open('xb') as f: np.savez_compressed(f,query_object_m=query,cell_radius_m=radius,area_mm2=area,
                native_distance_by_part_m=nd,current_distance_by_part_m=mapped,historical_distance_by_part_m=historical,
                **masks,**{'constraint_'+k:v for k,v in ledger.items()})
            # Literal inherited full area bounds, all parts/bands, including
            # native-ambiguous cells omitted from definite optimization rows.
            intervals=[dict(part=part,band_mm=band,**cell_intervals.pair_intervals(nd[:,j],mapped[:,j],radius,area,band/1000.,ALLOWANCE))
                       for j,part in enumerate(PARTS) for band in BANDS]
            pose_record['grids'].append(dict(density_mm=key,query_count=len(query),summary=summary,
                arrays_path=str(arrays),arrays_sha256=sha(arrays),area_intervals=intervals))
        write(out/(pose['pose_id']+'.json'),pose_record);records.append(pose_record)
    validate(protocol_path,freeze_commit)
    if any(counts[k] for k in ('physics_steps','retarget_solves','inference_calls')): raise RuntimeError('Forbidden computation attempted')
    max_queries=sum(g['query_count'] for g in case['grids'])*45*len(case['poses'])
    if counts['collider_point_queries']>max_queries:raise RuntimeError('Finite full-field query budget exceeded')
    result=dict(schema_version=SCHEMA,source_inventory_index=index,freeze_commit=commit,protocol_sha256=sha(protocol_path),
        counts=counts,poses=records,runtime_s=time.perf_counter()-started,geometry_source='CURRENT_REGISTERED_SCENE',
        historical_geometry_loaded=False,full_orientation_status='NOT_RUN',solver_status=p['solver_status'],
        production_promotion=False,full_strict_mapping_status='FAIL' if any(g['summary']['full_strict_mapping_status']=='FAIL' for pose in records for g in pose['grids']) else 'INCOMPLETE')
    write(out/'result.json',result)


def run(protocol_path,freeze_commit,output):
    p,commit=validate(protocol_path,freeze_commit)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'started.json',dict(protocol_sha256=sha(protocol_path),freeze_commit=commit))
    cases=[]
    for index in INDICES:
        validate(protocol_path,freeze_commit)
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--protocol',str(Path(protocol_path).resolve()),
                 '--freeze-commit',commit,'--output',str(out/str(index)),'--index',str(index)]
        log=out/f'case_{index}.log'
        started=time.perf_counter()
        with log.open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=child.wait(timeout=MAX_SECONDS_PER_CASE)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
        result=out/str(index)/'result.json'
        status='COMPLETED' if code==0 and result.exists() else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR'
        cases.append(dict(index=index,status=status,returncode=code,elapsed_s=time.perf_counter()-started,
                          log_path=str(log),log_sha256=sha(log),result_path=str(result) if result.exists() else None,
                          result_sha256=sha(result) if result.exists() else None))
        if status!='COMPLETED': break
    completed={r['index'] for r in cases}
    cases.extend(dict(index=index,status='NOT_RUN',reason='earlier_case_hard_stop') for index in INDICES if index not in completed)
    validate(protocol_path,freeze_commit)
    report=dict(schema_version=SCHEMA,protocol_sha256=sha(protocol_path),freeze_commit=commit,cases=cases,
        status='COMPLETED_BOUNDED_DIAGNOSTIC' if all(r['status']=='COMPLETED' for r in cases) else 'HARD_STOP',
        solver_status=p['solver_status'],mapping_acceptance=False,thresholds_changed=False,historical_caps_rebased=False,
        full_orientation_status='NOT_RUN',no_new_inference_physics_or_training=True)
    write(out/'report.json',report)
    print(json.dumps(report),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,required=True)
    for name in ('run','worker'):
        q=sub.add_parser(name);q.add_argument('--protocol',type=Path,required=True)
        q.add_argument('--freeze-commit',required=True);q.add_argument('--output',type=Path,required=True)
        if name=='worker':q.add_argument('--index',type=int,choices=INDICES,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args.protocol)
    elif args.command=='run':run(args.protocol,args.freeze_commit,args.output)
    else:worker(args.protocol,args.freeze_commit,args.output,args.index)


if __name__=='__main__':main()
