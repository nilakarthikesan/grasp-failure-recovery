"""Bounded actual-operand diagnostic; no surface distances or mapping acceptance.

prepare only reads existing inputs. run requires a committed declaration/script;
it stages four saved roots verbatim in the original current-scene source snapshot.
"""
from __future__ import annotations
import argparse
from collections import Counter
import contextlib
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.util
import io
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

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1'
SCHEMA = 'hug_actual_halfspace_matmul_diagnosis_v1'
INDICES = (20, 28)
PARTS = ('palm', 'index', 'middle', 'ring', 'little', 'thumb')
CHUNK = 256
MAX_FORWARDS = 48
MAX_SECONDS = 180
SCALAR_SAMPLES = 16


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Bound input/source changed: ' + path)


def measured(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with np.errstate(divide='warn', over='warn', invalid='warn', under='ignore'):
            value = fn()
    counts = Counter((w.category.__name__, str(w.message)) for w in caught)
    return value, [dict(category=k[0], message=k[1], count=v) for k, v in sorted(counts.items())]


def sample_indices(size):
    count = min(SCALAR_SAMPLES, size)
    if count <= 1:
        return list(range(count))
    return [i * (size - 1) // (count - 1) for i in range(count)]


def halfspace_comparison(points, equations):
    """Every output, then a fixed evenly spaced scalar subset; no BLAS reference."""
    points = np.asarray(points, dtype=float)
    equations = np.asarray(equations, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or equations.ndim != 2 or equations.shape[1] != 4:
        raise ValueError('Three coordinates and four plane coefficients required')
    if not len(points) or not len(equations) or not np.isfinite(points).all() or not np.isfinite(equations).all():
        raise ValueError('Nonempty finite operands required')
    normals, offsets = equations[:, :3], equations[:, 3]
    lengths = np.sqrt(np.einsum('ij,ij->i', normals, normals, optimize=False))
    unit_error = float(np.max(np.abs(lengths - 1.)))
    if unit_error > 1e-12:
        raise ValueError('Hull normals must be unit length within 1e-12')
    value, wm = measured(lambda: points @ normals.T + offsets)
    explicit, we = measured(lambda: np.einsum('ij,kj->ik', points, normals, optimize=False) + offsets)
    if not np.isfinite(value).all() or not np.isfinite(explicit).all():
        raise FloatingPointError('Nonfinite halfspace output')
    reference = []
    indices = sample_indices(value.size)
    for flat in indices:
        row, col = divmod(flat, len(normals))
        scalar = math.fsum(float(points[row, k]) * float(normals[col, k]) for k in range(3)) + float(offsets[col])
        if not math.isfinite(scalar):
            raise FloatingPointError('Nonfinite scalar reference')
        reference.append(dict(flat_index=flat, row=row, plane=col, scalar_m=scalar,
            matmul_error_m=float(value[row, col] - scalar), einsum_error_m=float(explicit[row, col] - scalar)))
    vm, ve = np.max(value, axis=1), np.max(explicit, axis=1)
    bound = 3 * float(np.max(np.abs(points))) * float(np.max(np.abs(normals))) + float(np.max(np.abs(offsets)))
    return dict(shape=list(value.shape), outputs=int(value.size), points_max_abs_m=float(np.max(np.abs(points))),
        normals_max_abs=float(np.max(np.abs(normals))), offsets_max_abs_m=float(np.max(np.abs(offsets))),
        normal_length_max_error=unit_error, arithmetic_absolute_bound_m=bound,
        arithmetic_bound_below_float64_max=bool(math.isfinite(bound) and bound < np.finfo(float).max / 16),
        operands_finite=True, outputs_finite=True, dtype=value.dtype.str,
        points_strides=list(points.strides), normal_transpose_strides=list(normals.T.strides),
        matmul_warnings=wm, einsum_warnings=we, exact_output_disagreements=int(np.count_nonzero(value != explicit)),
        max_matmul_einsum_difference_m=float(np.max(np.abs(value - explicit))),
        inside_mask_disagreements=int(np.count_nonzero((vm <= 1e-10) != (ve <= 1e-10))),
        inside_matmul_count=int(np.count_nonzero(vm <= 1e-10)), inside_einsum_count=int(np.count_nonzero(ve <= 1e-10)),
        hull_predicate_disagreements=int(np.count_nonzero((vm < 2e-8) != (ve < 2e-8))),
        hull_all_vertices_matmul=bool(np.max(vm) < 2e-8), hull_all_vertices_einsum=bool(np.max(ve) < 2e-8),
        matmul_max_residual_m=float(np.max(vm)), einsum_max_residual_m=float(np.max(ve)),
        minimum_inside_threshold_margin_m=float(np.min(np.abs(ve - 1e-10))),
        scalar_samples=reference, scalar_sample_count=len(reference),
        max_matmul_scalar_difference_m=max(abs(r['matmul_error_m']) for r in reference),
        max_einsum_scalar_difference_m=max(abs(r['einsum_error_m']) for r in reference))


def runtime_identity():
    return dict(executable=str(Path(sys.executable).resolve()), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, mujoco=mujoco.__version__, platform=platform.platform())


def prepare(path):
    old = read(OLD / 'protocol.json')
    if Path(path).exists():
        raise FileExistsError(path)
    # Reuse pinned scene source/config snapshots, not today's evolving source.
    snapshots = {v['snapshot']: v['snapshot_sha256'] for v in old['source_snapshots'].values()}
    original_sources = {v['source'] for v in old['source_snapshots'].values()}
    bindings = {name: expected for name, expected in old['bindings'].items()
                if name not in original_sources and '/tests/' not in name}
    bindings.update(snapshots)
    bindings[str(OLD / 'protocol.json')] = sha(OLD / 'protocol.json')
    cases = []
    for case in old['cases']:
        index = case['source_inventory_index']
        if index not in INDICES:
            raise ValueError('Unexpected original case')
        poses = []
        for pose_id in (f'original_{index}', f'experimental_{index}'):
            saved = OLD / 'run_v1' / str(index) / (pose_id + '.json')
            value = read(saved)
            joint_names = next(p['joint_names'] for p in case['poses'] if p['pose_id'] == pose_id)
            if value['pose_id'] != pose_id or value['source_inventory_index'] != index:
                raise ValueError('Saved pose identity mismatch')
            poses.append(dict(pose_id=pose_id, path=str(saved), joint_names=joint_names,
                qpos_rad=value['qpos_rad'], root_position_m=value['root_position_m'], root_quaternion_wxyz=value['root_quaternion_wxyz']))
            bindings[str(saved)] = sha(saved)
        cases.append(dict(source_inventory_index=index, poses=poses, grids=case['grids']))
    if [c['source_inventory_index'] for c in cases] != list(INDICES):
        raise ValueError('Exactly two ordered scenes required')
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    test = HERE / 'test_halfspace_products.py'
    bindings[str(test)] = sha(test)
    verify(bindings)
    value = dict(schema_version=SCHEMA, prepared_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings,
        cases=cases, original_protocol=str(OLD / 'protocol.json'), inventory=old['inventory'], source_urdf=old['source_urdf'],
        scene_snapshot=str(OLD / 'source_snapshots'), runtime_project_root=old['runtime_project_root'],
        dependency_dirs=old['dependency_dirs'], runtime=runtime_identity(),
        budgets=dict(forwards=MAX_FORWARDS, seconds=MAX_SECONDS, cases=2, poses=4, physics_steps=0,
            retarget_solves=0, inference_calls=0, distance_queries=0, training_calls=0, query_chunk_size=CHUNK),
        operations=['line40 hull vertex halfspaces for all active hand meshes/boxes',
                    'line104 duplicate mesh hull metadata halfspaces',
                    'line128 query halfspaces for six anatomical parts mesh colliders only'],
        reference='optimize=False einsum every output; math.fsum of exactly three scalar products plus offset at <=16 evenly spaced flat indices per matrix',
        predicates=dict(inside_inclusive_m=1e-10, hull_strict_m=2e-8, unit_normal_absolute_allowance=1e-12),
        controls=dict(seed=20261015, cube=True, random_shapes=[[8,12],[64,124],[256,124],[305,192],[256,192],[64,128],[256,256]],
            normal_layouts=['original four-column plane view','C-contiguous normal copy']),
        root_placement='Exactly saved run_v1 q/root JSON, no palm inference/reanchor',
        transform='Production matmul placement and query-local transform; independent optimize=False einsum comparison of every transform output',
        hull_scope='Same SciPy ConvexHull of actual compiled collider vertices; no nearest triangle, signed distance, full-field or orientation evaluation',
        interpretation='Numerical operand diagnostic only; no native fidelity acceptance or unchanged-distance claim',
        requires_committed_declaration_and_source=True)
    write(path, value)
    return value


def validate(path, commit):
    p = read(path)
    if p['schema_version'] != SCHEMA or p['budgets'] != dict(forwards=MAX_FORWARDS, seconds=MAX_SECONDS, cases=2, poses=4,
            physics_steps=0, retarget_solves=0, inference_calls=0, distance_queries=0, training_calls=0, query_chunk_size=CHUNK):
        raise ValueError('Changed diagnostic quota')
    verify(p['bindings'])
    if p['runtime'] != runtime_identity():
        raise ValueError('Runtime identity changed')
    full = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', commit + '^{commit}'], text=True).strip()
    for file in (Path(path).resolve(), Path(__file__).resolve(), HERE / 'test_halfspace_products.py'):
        relative = str(file.relative_to(ROOT))
        contents = subprocess.check_output(['git', '-C', str(ROOT), 'show', full + ':' + relative])
        if hashlib.sha256(contents).hexdigest() != sha(file):
            raise ValueError('Declaration/source differs from frozen Git blob: ' + relative)
    return p, full


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controls():
    rng = np.random.default_rng(20261015)
    cube = np.array([[x,y,z] for x in (-.02,.02) for y in (-.03,.03) for z in (-.04,.04)])
    rows = [dict(name='analytic_cube', **halfspace_comparison(cube, ConvexHull(cube).equations))]
    for count, faces in ((8,12),(64,124),(256,124),(305,192),(256,192),(64,128),(256,256)):
        points = rng.uniform(-.2, .2, (count,3))
        normals = rng.normal(size=(faces,3))
        normals /= np.sqrt(np.einsum('ij,ij->i', normals, normals, optimize=False))[:,None]
        eq = np.column_stack((normals, rng.uniform(-.1,.1,faces)))
        rows.append(dict(name=f'manufactured_{count}_{faces}_strided', **halfspace_comparison(points, eq)))
        # Explicitly assemble C-contiguous three-column normals in this separate control.
        copied = np.ascontiguousarray(eq[:,:3])
        value, notices = measured(lambda: points @ copied.T + eq[:,3])
        explicit = np.einsum('ij,kj->ik', points, copied, optimize=False) + eq[:,3]
        rows.append(dict(name=f'manufactured_{count}_{faces}_contiguous', shape=list(value.shape), outputs=int(value.size),
            normal_transpose_strides=list(copied.T.strides), matmul_warnings=notices,
            max_matmul_einsum_difference_m=float(np.max(np.abs(value-explicit))),
            outputs_finite=bool(np.isfinite(value).all() and np.isfinite(explicit).all()),
            inside_mask_disagreements=int(np.count_nonzero((value.max(1)<=1e-10)!=(explicit.max(1)<=1e-10)))))
    return rows


def summary(rows):
    sums = ('outputs','exact_output_disagreements','inside_mask_disagreements','hull_predicate_disagreements','scalar_sample_count')
    maxima = ('max_matmul_einsum_difference_m','max_matmul_scalar_difference_m','max_einsum_scalar_difference_m',
              'points_max_abs_m','normals_max_abs','offsets_max_abs_m','normal_length_max_error','arithmetic_absolute_bound_m')
    result = dict(matrices=len(rows), **{key:sum(r.get(key,0) for r in rows) for key in sums},
        **{key:max((r.get(key,0) for r in rows), default=0) for key in maxima})
    for field in ('matmul_warnings','einsum_warnings'):
        counter = Counter()
        for row in rows:
            for warning in row.get(field,[]):
                counter[(warning['category'],warning['message'])] += warning['count']
        result[field] = [dict(category=k[0],message=k[1],count=v) for k,v in sorted(counter.items())]
    return result


def execute(p, rows, counts, deadline, completed):
    def forbidden(kind):
        def fail(*args, **kwargs):
            counts[kind] += 1
            raise RuntimeError('Diagnostic forbids ' + kind)
        return fail
    for api in ('mj_step', 'mj_step1', 'mj_step2'):
        setattr(mujoco, api, forbidden('physics_steps'))
    real_forward = mujoco.mj_forward
    def forward(*args, **kwargs):
        counts['forwards'] += 1
        if counts['forwards'] > MAX_FORWARDS or time.perf_counter() > deadline:
            raise RuntimeError('Diagnostic static/time budget exhausted')
        return real_forward(*args, **kwargs)
    mujoco.mj_forward = forward
    snapshot = Path(p['scene_snapshot'])
    sys.path.extend(p['dependency_dirs'])
    sys.path.insert(0, str(snapshot / 'current_project'))
    scene = load_module('matmul_scene_runtime', snapshot / 'scene_runtime.py')
    witness = load_module('matmul_witness_check', snapshot / 'witness_check.py')
    from grasp_failure_prediction.evaluation import case_runner, retargeting, runner as runners
    retargeting.ShadowHandRetargeter.retarget = forbidden('retarget_solves')
    case_runner._run_case_inference = forbidden('inference_calls')
    case_runner.run_case = forbidden('inference_calls')
    for cls in (runners.AdroitShadowRunner, runners.ActuatedShadowRunner):
        cls.execute = forbidden('physics_steps')
        cls._advance = forbidden('physics_steps')
    if Path(runners.__file__).resolve() != snapshot / 'current_project/grasp_failure_prediction/evaluation/runner.py':
        raise RuntimeError('Wrong scene source snapshot')
    inventory = read(p['inventory'])['modern_cases']
    for case in p['cases']:
        index = case['source_inventory_index']
        source = inventory[index]
        payload = read(source['resolved_case_path'])['case']
        contract = read(source['frame_files']['scene_contract.json']['path'])
        runtime, resolved, _ = scene.build_current_runner(payload, p['runtime_project_root'], Path(p['source_urdf']).parent.parent)
        if resolved.environment.runner != 'grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner':
            raise RuntimeError('Unexpected scene runner')
        scene.reset_captured_scene(runtime, payload, contract)
        m, d = runtime.model, runtime.data
        objpos = d.xpos[runtime._object_body_id].copy()
        objrot = d.xmat[runtime._object_body_id].reshape(3,3).copy()
        forearm = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'forearm'))
        if forearm < 0:
            raise ValueError('Missing forearm')
        for pose in case['poses']:
            verify(p['bindings'])
            saved = read(pose['path'])
            for name in ('qpos_rad','root_position_m','root_quaternion_wxyz'):
                if saved[name] != pose[name]:
                    raise ValueError('Saved pose changed')
            scene.stage_named_root(runtime, pose['joint_names'], np.asarray(pose['qpos_rad']),
                np.asarray(pose['root_position_m']), np.asarray(pose['root_quaternion_wxyz']))
            if d.time != 0 or not np.array_equal(d.xpos[runtime._object_body_id], objpos) or not np.array_equal(d.xmat[runtime._object_body_id].reshape(3,3), objrot):
                raise RuntimeError('Object pose/time changed')
            expected = {g['id']:g for g in saved['hand_geometry_metadata']}
            hand = [g for g in range(m.ngeom) if witness.descendant(m.body_parentid,int(m.geom_bodyid[g]),forearm)
                    and (m.geom_contype[g] or m.geom_conaffinity[g])]
            if len(hand) != 45 or set(hand) != set(expected):
                raise ValueError('Hand collider inventory changed')
            for geom in hand:
                if time.perf_counter() > deadline:
                    raise TimeoutError('Diagnostic time exhausted')
                kind, mesh_id = int(m.geom_type[geom]), int(m.geom_dataid[geom])
                body = mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[geom]))
                part = witness.anatomical_part(body)
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    va,nv = int(m.mesh_vertadr[mesh_id]),int(m.mesh_vertnum[mesh_id])
                    vertices = m.mesh_vert[va:va+nv].astype(float)
                    if hashlib.sha256(vertices.tobytes()).hexdigest() != expected[geom]['mesh']['compiled_vertices_sha256']:
                        raise ValueError('Compiled mesh vertices differ from v1')
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    size = m.geom_size[geom]
                    vertices = np.array([[x,y,z] for x in (-size[0],size[0]) for y in (-size[1],size[1]) for z in (-size[2],size[2])])
                else:
                    continue
                eq = ConvexHull(vertices).equations
                common = dict(case=index,pose=pose['pose_id'],geom=geom,body=body,part=part)
                rows.append(dict(**common,site='old_kernel_line40_hull',**halfspace_comparison(vertices,eq)))
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    rows.append(dict(**common,site='old_kernel_line104_mesh_metadata',**halfspace_comparison(vertices,eq)))
                if kind != mujoco.mjtGeom.mjGEOM_MESH or part not in PARTS:
                    continue
                # Exactly the old production transforms, with separate independent references.
                rotation, wr = measured(lambda: objrot.T @ d.geom_xmat[geom].reshape(3,3))
                pos, wp = measured(lambda: (d.geom_xpos[geom] - objpos) @ objrot)
                rotation_ref = np.einsum('ji,jk->ik', objrot, d.geom_xmat[geom].reshape(3,3), optimize=False)
                pos_ref = np.einsum('j,ji->i', d.geom_xpos[geom]-objpos, objrot, optimize=False)
                if not np.isfinite(rotation).all() or not np.isfinite(pos).all():
                    raise FloatingPointError('Nonfinite actual transform')
                transform_error = max(float(np.max(abs(rotation-rotation_ref))),float(np.max(abs(pos-pos_ref))))
                for grid in case['grids']:
                    with np.load(grid['original_cache'],allow_pickle=False) as data:
                        query = data['query_object_m'].copy()
                    if len(query) != grid['query_count']:
                        raise ValueError('Original grid changed')
                    for start in range(0,len(query),CHUNK):
                        if time.perf_counter() > deadline:
                            raise TimeoutError('Diagnostic time exhausted')
                        points = query[start:start+CHUNK]
                        local, wl = measured(lambda: (points-pos) @ rotation)
                        local_ref = np.einsum('ij,jk->ik', points-pos, rotation, optimize=False)
                        if not np.isfinite(local).all():
                            raise FloatingPointError('Nonfinite actual query transform')
                        rows.append(dict(**common,site='old_kernel_line128_query',density_mm=grid['density_mm'],query_start=start,
                            transform_warnings=wr+wp+wl,transform_max_difference=max(transform_error,float(np.max(abs(local-local_ref)))),
                            **halfspace_comparison(local,eq)))
            completed.append([index,pose['pose_id']])
            print(json.dumps(dict(progress='pose_complete',case=index,pose=pose['pose_id'],matrices=len(rows),forwards=counts['forwards'])),flush=True)


def run(path, commit, output):
    p, full = validate(path, commit)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    rows, control_rows, completed = [], [], []
    counts = dict(forwards=0,physics_steps=0,retarget_solves=0,inference_calls=0,distance_queries=0,training_calls=0)
    status, failure = 'COMPLETED_BOUNDED_OPERAND_DIAGNOSTIC', None
    def timeout(*args):
        raise TimeoutError('180 second wall-clock hard stop')
    prior = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        control_rows = controls()
        execute(p,rows,counts,started+MAX_SECONDS,completed)
        validate(path,full)
        if any(counts[k] for k in counts if k != 'forwards'):
            raise RuntimeError('Forbidden operation attempted')
    except Exception:
        status,failure = 'HARD_STOP',traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,prior)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    result = dict(schema_version=SCHEMA,status=status,freeze_commit=full,protocol_sha256=sha(path),counts=counts,
        elapsed_s=time.perf_counter()-started,actual=summary(rows),controls=summary(control_rows),numpy_configuration=stream.getvalue(),
        failure=failure,mapping_acceptance=False,distance_replay=False,poses_completed=completed,
        poses_not_run_or_partial=[[c['source_inventory_index'],pose['pose_id']] for c in p['cases'] for pose in c['poses']
            if [c['source_inventory_index'],pose['pose_id']] not in completed],
        interpretation='This checks these halfspace operands only; returned distance fields and other kernels are not certified.')
    with (out/'actual_matrices.json.gz').open('xb') as handle:
        handle.write(gzip.compress(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0))
    write(out/'manufactured_controls.json',control_rows)
    write(out/'report.json',result)
    print(json.dumps(result),flush=True)
    if status == 'HARD_STOP':
        raise RuntimeError(failure)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    q = sub.add_parser('prepare');q.add_argument('--protocol',type=Path,required=True)
    q = sub.add_parser('run');q.add_argument('--protocol',type=Path,required=True);q.add_argument('--freeze-commit',required=True);q.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        p=prepare(args.protocol);print(json.dumps(dict(protocol=str(args.protocol),sha256=sha(args.protocol),cases=len(p['cases']),physics_steps=0)))
    else:
        run(args.protocol,args.freeze_commit,args.output)


if __name__ == '__main__':
    main()
