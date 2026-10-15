#!/usr/bin/env python3
"""Frozen static exposed-boundary diagnosis on the original 32 HUG poses.

prepare binds existing inputs, the previous comparison, and all new source.
run never solves a grasp, integrates physics, produces labels, or modifies a
model. Existing Shadow scalar measurements are linked without replaying it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import types

# Load base runtime before external directories, which include another NumPy.
import numpy as np
import mujoco


def load_driver(path):
    spec = importlib.util.spec_from_file_location('frozen_native_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('protocol exists; choose a new versioned directory')
    driver = load_driver(project/'scripts/audit_hug_native_mano.py')
    previous = project/'research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement'
    old = json.loads((previous/'protocol.json').read_text())
    # The historical run's serialization-only fix is explicitly documented.
    bindings = {}
    for path, digest in old['bindings'].items():
        actual = previous/'audit_at_measurement.py' if path.endswith('/scripts/audit_hug_native_mano.py') else Path(path)
        if driver.sha(actual) != digest:
            raise ValueError(f'historical binding changed: {actual}')
        bindings[str(actual)] = digest
    source = [Path(__file__).resolve(), project/'scripts/audit_hug_native_mano.py',
              project/'src/grasp_failure_prediction/evaluation/static_boundary_audit.py',
              project/'src/grasp_failure_prediction/evaluation/native_mano.py',
              project/'tests/test_static_boundary_audit.py']
    (output/'source').mkdir()
    for path in source:
        bindings[str(path)] = driver.sha(path)
        shutil.copyfile(path, output/'source'/path.name)
    for path in [previous/'protocol.json', previous/'results.json', *sorted((previous/'cases').glob('case_*.json'))]:
        bindings[str(path)] = driver.sha(path)
    protocol = dict(schema='hug_exposed_boundary_static_diagnosis_v1', project=str(project),
                    hug=old['hug'], dependency_dirs=old['dependency_dirs'], inventory=old['inventory'],
                    previous_measurement=str(previous), previous_protocol_sha256=driver.sha(previous/'protocol.json'),
                    bindings=bindings, case_count=32, source_snapshot='source/',
                    physics_steps_allowed=0, retarget_solves_allowed=0, inference_allowed=False,
                    numerical_exposure_tolerance_m=1e-9, skin_side_witness_offset_m=1e-6,
                    normal_region_m=.005, descriptive_near_bands_m=[.001, .002, .005],
                    sample_sizes=[128, 512, 2048],
                    sampling='128=every fourth original512;512=literal prior probes;2048=original512 plus1536 independent fixed area-quantile/Halton probes; no hand-dependent selection',
                    native_geometry='unchanged native_mano_static_capsule_hull_v1; whole-hand union of compiled convex palm and15capsules',
                    skin_sign='closed oriented-edge topology plus generalized winding0/1 and distance>1um; self-intersection absence not certified',
                    normals='nearest skin face interior, local winding outside/inside witnesses at+/-1um; unique exposed regular native component; each ambiguous case unknown',
                    penetration='union membership; exact per-component depths give lower bound, six exterior-axis witnesses give upper bound; not exact inside union SDF',
                    acceptance='diagnostic only; no new geometry acceptance threshold; no default/profile changes',
                    limitations=['finite probes do not certify continuous surface fidelity', 'nested density sensitivity is not a confidence interval',
                                 'winding occupancy does not certify an injective physical skin', 'object visual surface preferred as in historical comparison',
                                 'no physical contact dynamics or controller validation', 'all32 are development inputs; no qualification on new grasps',
                                 'whole-union quantities differ from prior anatomical component quantities'])
    driver.write(output/'protocol.json', protocol)
    print(json.dumps({'protocol':str(output/'protocol.json'), 'sha256':driver.sha(output/'protocol.json')}))


def object_triangles(model, data, driver):
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object')
    if obj < 0:
        raise ValueError('captured object missing')
    ids = [g for g in range(model.ngeom) if driver.descendant(model, int(model.geom_bodyid[g]), obj)]
    visual = [g for g in ids if not (model.geom_contype[g] or model.geom_conaffinity[g])]
    output = []
    for g in visual or ids:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and visual:
            mid = int(model.geom_dataid[g])
            v, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            f, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            vertices = driver.transform(model.mesh_vert[v:v+nv], data.geom_xmat[g].reshape(3,3), data.geom_xpos[g])
            output.append(vertices[model.mesh_face[f:f+nf]])
        else:
            output.append(driver.geometry(model, data, g)['triangles'])
    return np.concatenate(output)


def additional_probes(triangles, count):
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)/2
    ids = np.searchsorted(np.cumsum(area), (np.arange(count)+.5)*area.sum()/count)
    def inverse(n, base):
        result, denom = 0., 1.
        while n:
            n, remainder = divmod(n, base)
            denom *= base
            result += remainder/denom
        return result
    u = np.sqrt([inverse(k+513, 2) for k in range(count)])
    v = np.array([inverse(k+513, 3) for k in range(count)])
    return ((1-u[:,None])*triangles[ids,0]+(u*(1-v))[:,None]*triangles[ids,1]
            +(u*v)[:,None]*triangles[ids,2])


def numeric_stats(values):
    values = np.asarray(values)
    if not len(values):
        return {'count':0, 'mean':None, 'p95':None, 'max':None}
    if not np.isfinite(values).all():
        raise ValueError('nonfinite values in scalar summary')
    return {'count':len(values), 'mean':float(values.mean()), 'p95':float(np.quantile(values,.95)), 'max':float(values.max())}


def summary(indices, skin, proxy, protocol):
    sd = skin['distance'][indices]
    inside = proxy['inside'][indices]
    resolved = skin['occupancy_resolved'][indices]
    si = skin['inside'][indices]
    exterior = (~inside) & (~proxy['boundary'][indices])
    # Normals refer to each representation's closest point, not a shared
    # correspondence or a proof that the normals belong to the same anatomy.
    normals = skin['normal_valid'][indices] & proxy['normal_valid'][indices]
    normal_near = sd <= protocol['normal_region_m']
    normals &= normal_near
    cosines = np.einsum('ij,ij->i',skin['normals'][indices][normals],proxy['normals'][indices][normals])
    angles = np.degrees(np.arccos(np.clip(cosines,-1,1)))
    distance_error = abs(proxy['exterior_distance'][indices][exterior]-sd[exterior])
    rows=[]
    for band in protocol['descriptive_near_bands_m']:
        # Interior distance is an interval, so near-band assignment is unknown
        # when the interval crosses a threshold, never filled with component min.
        pd=proxy['exterior_distance'][indices]
        pl=proxy['penetration_lower'][indices]
        pu=proxy['penetration_upper'][indices]
        pnear=(exterior & (pd <= band)) | (inside & (pu <= band)) | proxy['boundary'][indices]
        pfar=(exterior & (pd > band)) | (inside & (pl > band))
        ref=sd <= band
        rows.append(dict(band_m=band,skin_near=int(ref.sum()),proxy_definitely_near=int(pnear.sum()),
                         proxy_definitely_far=int(pfar.sum()),proxy_near_unknown=int((~(pnear|pfar)).sum()),
                         definitely_lost=int((ref&pfar).sum()),definitely_added=int((~ref&pnear).sum())))
    return dict(probe_count=len(indices), skin_occupancy_unknown=int((~resolved).sum()),
                skin_winding_inside=int(si.sum()),proxy_union_inside=int(inside.sum()),
                proxy_boundary_unknown=int(proxy['boundary'][indices].sum()),
                added_proxy_penetration=int((resolved & ~si & inside).sum()),
                lost_skin_penetration=int((resolved & si & exterior).sum()),
                buried_component_projections=int(proxy['closest_component_projection_buried'][indices].sum()),
                penetration_lower_m=numeric_stats(proxy['penetration_lower'][indices][inside]),
                penetration_upper_m=numeric_stats(proxy['penetration_upper'][indices][inside]),
                penetration_bound_width_m=numeric_stats((proxy['penetration_upper'][indices]-proxy['penetration_lower'][indices])[inside]),
                exterior_union_distance_error_m=numeric_stats(distance_error),
                normal_region_count=int(normal_near.sum()),normals_compared=int(normals.sum()),
                normal_region_unknown=int((normal_near & ~normals).sum()),
                normal_angle_degrees=numeric_stats(angles),normal_reversals=int((cosines < 0).sum()),
                near_bands=rows)


def run(args):
    output=Path(args.output).resolve()
    if (output/'cases').exists() or (output/'results.json').exists():
        raise FileExistsError('measurements exist; never overwrite')
    protocol_path=output/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    driver=load_driver(Path(protocol['project'])/'scripts/audit_hug_native_mano.py')
    protocol_hash=driver.sha(protocol_path)
    def verify():
        if driver.sha(protocol_path) != protocol_hash:
            raise ValueError('protocol changed')
        for path,digest in protocol['bindings'].items():
            if driver.sha(path) != digest:
                raise ValueError(f'changed bound input/source: {path}')
    verify()
    sys.path.insert(0,str(Path(protocol['project'])/'src'))
    sys.path.extend(protocol['dependency_dirs'])
    import torch
    import trimesh
    from grasp_failure_prediction.evaluation.native_mano import NativeManoGeometry, NativeManoStaticModel
    from grasp_failure_prediction.evaluation.static_boundary_audit import Capsule,ConvexSolid,SolidUnion,oriented_skin,skin_query
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
    counts=dict(static_forwards=0, physics_steps=0,retarget_solves=0,inference_calls=0)
    def no_step(*a,**kw):
        counts['physics_steps']+=1
        raise RuntimeError('static audit forbids integration')
    for name in ('mj_step','mj_step1','mj_step2'):
        setattr(mujoco,name,no_step)
    def no_solve(*a,**kw):
        counts['retarget_solves']+=1
        raise RuntimeError('static audit forbids refitting')
    ShadowHandRetargeter.retarget=no_solve
    original_forward=mujoco.mj_forward
    def forward(*a,**kw):
        counts['static_forwards']+=1
        return original_forward(*a,**kw)
    mujoco.mj_forward=forward
    package=types.ModuleType('hug')
    package.__path__=[str(Path(protocol['hug'])/'src')]
    sys.modules['hug']=package
    from hug.models.mano import MANO
    torch.set_num_threads(2)
    mano=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
    weights=mano.mano_layer.th_weights.detach().numpy()
    faces=mano.mano_layer.get_mano_closed_faces().detach().numpy()
    inventory=json.loads(Path(protocol['inventory']).read_text())['modern_cases']
    if len(inventory) != protocol['case_count']:
        raise ValueError('case count changed')
    (output/'cases').mkdir()
    models={}
    records=[]
    for index,case in enumerate(inventory):
        raw=pickle.loads(Path(case['native']['path']).read_bytes())
        params=np.concatenate([np.asarray(raw['t']).reshape(3),np.asarray(raw['R_6d']).reshape(6),np.asarray(raw['pose_6d']).reshape(90)]).astype(np.float32)
        beta=np.asarray(raw['shape'],np.float32).reshape(1,10)
        with torch.no_grad():
            exact=mano(torch.from_numpy(params[None]),betas=torch.from_numpy(beta))
        # Reconstruction proves these are the saved source poses, not a new
        # inference. No generator network or checkpoint is loaded.
        error=float(np.linalg.norm(exact['vertices'][0].numpy()+params[:3]-np.asarray(raw['mesh_vertices']),axis=1).max())
        if error > 1e-6 or not np.array_equal(faces,raw['mesh_faces']):
            raise ValueError('saved MANO reconstruction changed')
        key=beta.tobytes()
        if key not in models:
            with torch.no_grad():
                neutral=mano.mano_layer(torch.zeros((1,48)),torch.from_numpy(beta))
            vertices=neutral.verts[0].numpy()
            geom=NativeManoGeometry(neutral.transforms_abs[0,:,:3,3].numpy(),driver.PARENTS,vertices,faces,weights,vertices[driver.TIPS],np.array([15,3,6,12,9]))
            models[key]=NativeManoStaticModel(geom)
        native=models[key]
        camera=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        world=driver.transform(np.asarray(raw['mesh_vertices']),camera[:3,:3],camera[:3,3])
        wrist=driver.transform(params[:3][None],camera[:3,:3],camera[:3,3])[0]
        native.set_pose(wrist,camera[:3,:3]@exact['R_3x3'][0].numpy(),exact['pose_3x3'][0].numpy())
        readback=native.readback()
        solids=[ConvexSolid(readback.palm_vertices_world)]
        solids.extend(Capsule(start,end,radius) for (start,end),radius in zip(readback.capsule_endpoints_world,readback.capsule_radii,strict=True))
        union=SolidUnion(solids)
        captured=mujoco.MjModel.from_binary_path(case['scene_mjb']['path'])
        data=mujoco.MjData(captured)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:
            data.qpos[:]=z['qpos']
        mujoco.mj_forward(captured,data)
        original,meta=driver.object_probes(captured,data)
        objtri=object_triangles(captured,data,driver)
        probes=np.concatenate([original,additional_probes(objtri,1536)])
        if len(np.unique(probes,axis=0)) != 2048:
            raise ValueError('non-distinct probes')
        triangles,topology=oriented_skin(world,faces)
        skin=skin_query(probes,triangles,eligible=topology['winding_eligible'],side_offset=protocol['skin_side_witness_offset_m'])
        proxy=union.query(probes,tolerance=protocol['numerical_exposure_tolerance_m'])
        # Exterior nearest component projections must be on the exposed union.
        exterior=~proxy['inside'] & ~proxy['boundary']
        if proxy['closest_component_projection_buried'][exterior].any():
            raise AssertionError('invalid exterior exposed-boundary claim')
        indices={128:np.arange(0,512,4),512:np.arange(512),2048:np.arange(2048)}
        rec=dict(index=index,object_id=case['object_id'],native_sha256=case['native']['sha256'],
                 reconstruction_max_error_m=error,skin_topology=topology,object_geometry=meta,
                 sampling=[summary(indices[n],skin,proxy,protocol) for n in protocol['sample_sizes']],
                 frozen_shadow_case_sha256=driver.sha(Path(protocol['previous_measurement'])/'cases'/f'case_{index:02}.json'),
                 physics_steps=0,new_eval_data=False)
        driver.write(output/'cases'/f'case_{index:02}.json',rec)
        records.append(rec)
        assert data.time == native.data.time == 0
        print(json.dumps(dict(index=index,object_id=case['object_id'],**rec['sampling'][-1])),flush=True)
    verify()
    result=dict(schema=protocol['schema'],protocol_sha256=protocol_hash,case_count=len(records),counts=counts,
                runtime=dict(numpy=np.__version__,mujoco=mujoco.__version__,torch=torch.__version__,trimesh=trimesh.__version__),
                all_input_source_bindings_unchanged=True,native_collision_fidelity_accepted=False,
                physical_execution_qualified=False,original_acceptance_changed=False,new_eval_data=False,
                shadow_replayed=False,shadow_reference_sha256=driver.sha(Path(protocol['previous_measurement'])/'results.json'),
                case_reports=[dict(index=c['index'],sha256=driver.sha(output/'cases'/f"case_{c['index']:02}.json")) for c in records],
                totals=[])
    for position,n in enumerate(protocol['sample_sizes']):
        rows=[c['sampling'][position] for c in records]
        additive=['probe_count','skin_occupancy_unknown','skin_winding_inside','proxy_union_inside','proxy_boundary_unknown',
                  'added_proxy_penetration','lost_skin_penetration','buried_component_projections','normal_region_count',
                  'normals_compared','normal_region_unknown','normal_reversals']
        total={key:sum(r[key] for r in rows) for key in additive}
        total['probes_per_case']=n
        total['max_penetration_lower_m']=max((r['penetration_lower_m']['max'] or 0) for r in rows)
        total['max_penetration_upper_m']=max((r['penetration_upper_m']['max'] or 0) for r in rows)
        total['max_penetration_bound_width_m']=max((r['penetration_bound_width_m']['max'] or 0) for r in rows)
        total['max_exterior_union_distance_error_m']=max(r['exterior_union_distance_error_m']['max'] for r in rows)
        total['near_bands']=[{key:sum(r['near_bands'][b][key] for r in rows) for key in rows[0]['near_bands'][b] if key!='band_m'}|{'band_m':band} for b,band in enumerate(protocol['descriptive_near_bands_m'])]
        result['totals'].append(total)
    driver.write(output/'results.json',result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--project')
    args=parser.parse_args()
    if args.action=='prepare':
        if not args.project:
            parser.error('prepare needs --project')
        prepare(args)
    else:
        run(args)


if __name__=='__main__':
    main()
