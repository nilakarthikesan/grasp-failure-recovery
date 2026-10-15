#!/usr/bin/env python3
"""Compare saved HUG native MANO collision proxy with saved Shadow commands.

Static FK/geometry only. `prepare` binds every input/source before `run`.
No generator, grasp execution, integration, outcome labels, or training.
External MANO/HUG assets stay external; reports contain scalar measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import types
import warnings

import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

PARTS = ('palm', 'thumb', 'index', 'middle', 'ring', 'little')
BONE_PART = np.array([0,2,2,2,3,3,3,5,5,5,4,4,4,1,1,1])
PARENTS = np.array([-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14])
TIPS = np.array([745,317,444,556,673])

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def transform(points, rotation, position):
    return np.einsum('ij,nj->ni', rotation, points, optimize=False)+position

def triangle_distance(points, triangles):
    """Unsigned distance to all triangles, using bounded temporary memory."""
    import trimesh
    if len(triangles)==0:
        raise ValueError('Empty triangle surface')
    result=[]
    for start in range(0,len(points),32):
        q=points[start:start+32]
        tri=np.tile(triangles,(len(q),1,1))
        query=np.repeat(q,len(triangles),axis=0)
        closest=trimesh.triangles.closest_point(tri,query)
        delta=closest-query
        ds=np.einsum('ij,ij->i',delta,delta).reshape(len(q),-1)
        result.extend(np.sqrt(ds.min(axis=1)))
    return np.asarray(result)

def box_triangles(size):
    verts=np.array([[x,y,z] for x in (-size[0],size[0])
                    for y in (-size[1],size[1]) for z in (-size[2],size[2])])
    return verts[ConvexHull(verts).simplices]

def geometry(model,data,g):
    kind=int(model.geom_type[g]); pos=data.geom_xpos[g].copy()
    rot=data.geom_xmat[g].reshape(3,3).copy(); size=model.geom_size[g].copy()
    if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
        mid=int(model.geom_dataid[g]); a=int(model.mesh_vertadr[mid]); n=int(model.mesh_vertnum[mid])
        verts=transform(model.mesh_vert[a:a+n],rot,pos)
        return {'kind':'triangles','triangles':verts[ConvexHull(verts).simplices]}
    if kind==int(mujoco.mjtGeom.mjGEOM_BOX):
        return {'kind':'triangles','triangles':transform(box_triangles(size).reshape(-1,3),rot,pos).reshape(-1,3,3)}
    if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_CYLINDER),int(mujoco.mjtGeom.mjGEOM_SPHERE)):
        return {'kind':{int(mujoco.mjtGeom.mjGEOM_CAPSULE):'capsule',int(mujoco.mjtGeom.mjGEOM_CYLINDER):'cylinder',int(mujoco.mjtGeom.mjGEOM_SPHERE):'sphere'}[kind],
                'position':pos,'rotation':rot,'size':size}
    raise ValueError(f'Unsupported measured geometry {kind}')

def surface_distance(points, geometries):
    distances=[]
    for g in geometries:
        if g['kind']=='triangles':
            d=triangle_distance(points,g['triangles'])
        else:
            p=np.einsum('ji,nj->ni',g['rotation'],points-g['position'],optimize=False)
            r=float(g['size'][0]); half=float(g['size'][1])
            if g['kind']=='capsule':
                p[:,2]-=np.clip(p[:,2],-half,half)
                d=np.abs(np.sqrt(np.einsum('ij,ij->i',p,p))-r)
            elif g['kind']=='sphere':
                d=np.abs(np.sqrt(np.einsum('ij,ij->i',p,p))-r)
            else:
                radial=np.sqrt(np.einsum('ij,ij->i',p[:,:2],p[:,:2]))
                q=np.column_stack((radial-r,np.abs(p[:,2])-half))
                outside=np.maximum(q,0.)
                signed=np.sqrt(np.einsum('ij,ij->i',outside,outside))+np.minimum(np.max(q,axis=1),0.)
                d=np.abs(signed)
        distances.append(d)
    if not distances:raise ValueError('No colliders in anatomical part')
    return np.min(distances,axis=0)

def stats(values,weights=None):
    values=np.asarray(values,float)
    return {'mean_m':float(np.average(values,weights=weights)),
            'p95_m':float(np.quantile(values,.95)), 'max_m':float(values.max()),
            'count':len(values), 'p95_is_unweighted':True}

def descendant(model,body,ancestor):
    while body:
        if body==ancestor:return True
        body=int(model.body_parentid[body])
    return False

def shadow_part(name):
    for prefix,part in [('th',1),('ff',2),('mf',3),('rf',4),('lf',5)]:
        if name.startswith(prefix):return part
    if name=='palm':return 0
    return None

def object_probes(model,data):
    obj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
    if obj<0:raise ValueError('Missing captured object')
    ids=[g for g in range(model.ngeom) if descendant(model,int(model.geom_bodyid[g]),obj)]
    visual=[g for g in ids if not(model.geom_contype[g] or model.geom_conaffinity[g])]
    ids=visual if visual else ids
    triangles=[]
    for g in ids:
        if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and visual:
            mid=int(model.geom_dataid[g]);v=int(model.mesh_vertadr[mid]);nv=int(model.mesh_vertnum[mid]);f=int(model.mesh_faceadr[mid]);nf=int(model.mesh_facenum[mid])
            vertices=transform(model.mesh_vert[v:v+nv],data.geom_xmat[g].reshape(3,3),data.geom_xpos[g])
            triangles.append(vertices[model.mesh_face[f:f+nf]])
        else:triangles.append(geometry(model,data,g)['triangles'])
    tri=np.concatenate(triangles)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.sqrt(np.einsum('ij,ij->i',cross,cross))/2
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('Invalid object triangles')
    # Fixed area-quantile faces with deterministic Halton barycentric locations.
    # Sampling is independent of hand errors; no continuous-coverage claim.
    ids=np.searchsorted(np.cumsum(area),(np.arange(512)+.5)*area.sum()/512)
    def radical_inverse(n,base):
        result=0.;denom=1.
        while n:
            n,remainder=divmod(n,base);denom*=base;result+=remainder/denom
        return result
    u=np.sqrt([radical_inverse(k,2) for k in range(1,513)])
    v=np.asarray([radical_inverse(k,3) for k in range(1,513)])
    points=(1-u[:,None])*tri[ids,0]+(u*(1-v))[:,None]*tri[ids,1]+(u*v)[:,None]*tri[ids,2]
    return points,{'probe_count':512,'unique_points':len(np.unique(points,axis=0)),'unique_triangle_count':len(np.unique(ids)),
                               'reference':'captured_object_visual' if visual else 'captured_object_collision'}

def prepare(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    root=Path(args.project).resolve(); hug=Path(args.hug).resolve()
    inventory=root/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
    cases=json.loads(inventory.read_text())['modern_cases']
    paths=[inventory,Path(__file__).resolve(),root/'src/grasp_failure_prediction/evaluation/native_mano.py',
           root/'src/grasp_failure_prediction/evaluation/pose_validation.py',root/'src/grasp_failure_prediction/evaluation/retargeting.py',
           hug/'src/models/mano.py',hug/'src/utils/transform_utils.py',hug/'src/utils/data_keys.py',
           hug/'assets/mano_models/models/MANO_RIGHT.pkl',
           root/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json']
    for x in cases:
        paths.extend([Path(x['native']['path']),Path(x['stored_shadow_command']['path']),Path(x['scene_mjb']['path'])])
        paths.extend(Path(x['frame_files'][f]['path']) for f in ('T_world_camera.npy','scene_state.npz','scene_contract.json'))
        for frame in ('T_world_camera.npy','scene_state.npz','scene_contract.json'):
            if sha(x['frame_files'][frame]['path'])!=x['frame_files'][frame]['sha256']:raise ValueError(f'Frame inventory binding mismatch {frame}')
        for field in ('native','stored_shadow_command','scene_mjb'):
            if sha(x[field]['path'])!=x[field]['sha256']:raise ValueError(f'Inventory binding mismatch {field}')
    paths.extend((root/'third_party/dex-urdf/robots/hands/shadow_hand').rglob('*.obj'))
    paths.extend((root/'third_party/dex-urdf/robots/hands/shadow_hand').glob('*.urdf'))
    for dep in args.dependency_dir:
        for rel in ('manotorch/manolayer.py','manotorch/utils/geometry.py','trimesh/triangles.py'):
            if (Path(dep)/rel).exists():paths.append(Path(dep)/rel)
    protocol={'schema':'hug_native_mano_static_comparison_v1','project':str(root),'hug':str(hug),
              'dependency_dirs':args.dependency_dir,'inventory':str(inventory),
              'bindings':{str(p.resolve()):sha(p) for p in paths},'case_count':32,
              'numerical_import_tolerance_m':1e-6,'near_bands_m':[.001,.002,.005],
              'selection':'all32 existing saved proposals, no success filtering',
              'metrics':'same-part native triangle-centroid to collider boundary; identical512 object area-quantile/Halton barycentric probes to native triangles and each proxy; pointwise bands only',
              'limitations':['component surface distances can include buried interfaces','no continuous coverage or normals certificate','no dynamic/contact-force validation','no equivalence threshold assigned for proxy geometry','32 development poses, no held-out evidence'],
              'native_proxy':'neutral source shape;15capsules RMS owner-vertex radius and convex palm, fixed before posed errors; not official HUG simulator reproduction',
              'physics_steps_allowed':0,'native_inference_allowed':False,'original_criteria_changed':False}
    write(out/'protocol.json',protocol);print(json.dumps({'protocol':str(out/'protocol.json'),'sha256':sha(out/'protocol.json')}))

def run(args):
    out=Path(args.output);pp=out/'protocol.json';p=json.loads(pp.read_text());ph=sha(pp)
    if (out/'results.json').exists() or (out/'cases').exists():raise FileExistsError('Never overwrite measurements')
    def verify():
        if sha(pp)!=ph:raise ValueError('Protocol changed')
        for path,digest in p['bindings'].items():
            if sha(path)!=digest:raise ValueError(f'Changed binding {path}')
    verify();sys.path.insert(0,str(Path(p['project'])/'src'));sys.path.extend(p['dependency_dirs'])
    import torch
    import trimesh
    from grasp_failure_prediction.evaluation.native_mano import NativeManoGeometry,NativeManoStaticModel
    from grasp_failure_prediction.evaluation import pose_validation as pv
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
    counts={'static_forwards':0,'physics_steps':0,'retarget_solves':0,'new_native_proposals':0}
    def no_step(*a,**kw):
        counts['physics_steps']+=1;raise RuntimeError('No new eval data: integration forbidden')
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,no_step)
    def no_retarget(*a,**kw):
        counts['retarget_solves']+=1;raise RuntimeError('Only saved Shadow commands allowed')
    ShadowHandRetargeter.retarget=no_retarget
    old_forward=mujoco.mj_forward
    def forward(*a,**kw):counts['static_forwards']+=1;return old_forward(*a,**kw)
    mujoco.mj_forward=forward
    package=types.ModuleType('hug');package.__path__=[str(Path(p['hug'])/'src')];sys.modules['hug']=package
    from hug.models.mano import MANO
    torch.set_num_threads(2)
    mano=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
    weights=mano.mano_layer.th_weights.detach().numpy(); faces=mano.mano_layer.get_mano_closed_faces().detach().numpy()
    groupweights=np.stack([weights[:,BONE_PART==i].sum(axis=1) for i in range(6)],axis=1)
    faceparts=groupweights[faces].sum(axis=1).argmax(axis=1)
    inventory=json.loads(Path(p['inventory']).read_text())['modern_cases'];assert len(inventory)==32
    urdf=Path(p['project'])/'third_party/dex-urdf/robots/hands'
    sm=pv.load_shadow_model(urdf,floating_root=True);sd=mujoco.MjData(sm)
    models={};records=[];(out/'cases').mkdir()
    for index,c in enumerate(inventory):
        raw=pickle.loads(Path(c['native']['path']).read_bytes())
        params=np.concatenate([np.asarray(raw['t']).reshape(3),np.asarray(raw['R_6d']).reshape(6),np.asarray(raw['pose_6d']).reshape(90)]).astype(np.float32)
        beta=np.asarray(raw['shape'],np.float32).reshape(1,10)
        with torch.no_grad():
            exact=mano(torch.from_numpy(params[None]),betas=torch.from_numpy(beta))
            local=exact['pose_3x3'][0].numpy(); globalR=exact['R_3x3'][0].numpy()
            # Source kinematic transforms, with no new generator sample.
            rots=torch.cat([exact['R_3x3'][:,None],exact['pose_3x3']],dim=1)
            kin=mano.mano_layer.skinning_layer(rots,torch.from_numpy(beta))
        rebuilt=exact['vertices'][0].numpy()+params[:3]
        native_vertices=np.asarray(raw['mesh_vertices'],float)
        native_joints=np.asarray(raw['landmarks_3d'],float)
        reconstructed_joints=exact['landmarks_3d'][0].numpy()+params[:3]
        reconstruction={'max_vertex_error_m':float(np.linalg.norm(rebuilt-native_vertices,axis=1).max()),
                        'max_landmark_error_m':float(np.linalg.norm(reconstructed_joints-native_joints,axis=1).max()),
                        'faces_exact':bool(np.array_equal(faces,raw['mesh_faces']))}
        key=beta.tobytes()
        if key not in models:
            with torch.no_grad():neutral=mano.mano_layer(torch.zeros((1,48)),torch.from_numpy(beta))
            verts=neutral.verts[0].numpy();joints=neutral.transforms_abs[0,:,:3,3].numpy()
            ng=NativeManoGeometry(joints,PARENTS,verts,faces,weights,verts[TIPS],np.array([15,3,6,12,9]))
            models[key]=NativeManoStaticModel(ng)
        nm=models[key]
        camera=np.load(c['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        world_vertices=transform(native_vertices,camera[:3,:3],camera[:3,3])
        world_joints=transform(native_joints,camera[:3,:3],camera[:3,3])
        wrist=transform(params[:3][None],camera[:3,:3],camera[:3,3])[0]
        rotation=np.einsum('ij,jk->ik',camera[:3,:3],globalR,optimize=False)
        nm.set_pose(wrist,rotation,local);rb=nm.readback()
        source_bones=kin['transforms_abs'][0,:,:3,3].numpy()+params[:3]
        source_bones=transform(source_bones,camera[:3,:3],camera[:3,3])
        bone_errors=np.linalg.norm(rb.joint_positions_world-source_bones,axis=1)
        native_tips=np.linalg.norm(rb.tip_positions_world-world_joints[[4,8,12,16,20]],axis=1)
        # Replay literal saved Shadow commands; no endpoint or controller change.
        with np.load(c['stored_shadow_command']['path'],allow_pickle=False) as z:
            row=c['stored_shadow_command']['row_index'];names=z['joint_names'].tolist();q=z['nominal_target_rad'][row]
            rp=z['root_target_position_m'][row];rq=z['root_target_quaternion_wxyz'][row]
        sd.qpos[:]=sm.qpos0;sd.qvel[:]=0
        for name,value in zip(names,q,strict=True):
            j=mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_JOINT,name);assert j>=0
            sd.qpos[int(sm.jnt_qposadr[j])]=value
        j=mujoco.mj_name2id(sm,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');a=int(sm.jnt_qposadr[j])
        sd.qpos[a:a+7]=np.concatenate([rp,rq]);mujoco.mj_forward(sm,sd)
        shadow_tips=np.linalg.norm(pv._link_positions(sm,sd,urdf/'shadow_hand/shadow_hand_right.urdf')[:5]-world_joints[[4,8,12,16,20]],axis=1)
        shadow=[[] for _ in PARTS];native=[[] for _ in PARTS]
        for g in range(sm.ngeom):
            if not(sm.geom_contype[g] or sm.geom_conaffinity[g]):continue
            part=shadow_part(mujoco.mj_id2name(sm,mujoco.mjtObj.mjOBJ_BODY,int(sm.geom_bodyid[g])))
            if part is not None:shadow[part].append(geometry(sm,sd,g))
        for g in range(nm.model.ngeom):
            # Geometry ordering is not used: resolve owning named MANO body.
            bname=mujoco.mj_id2name(nm.model,mujoco.mjtObj.mjOBJ_BODY,int(nm.model.geom_bodyid[g]))
            bone=int(bname.rsplit('_',1)[-1]);native[int(BONE_PART[bone])].append(geometry(nm.model,nm.data,g))
        old=mujoco.MjModel.from_binary_path(c['scene_mjb']['path']);od=mujoco.MjData(old)
        with np.load(c['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:od.qpos[:]=z['qpos']
        mujoco.mj_forward(old,od);probes,probe_meta=object_probes(old,od)
        tri=world_vertices[faces];centers=tri.mean(axis=1);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);areas=np.sqrt(np.einsum('ij,ij->i',cross,cross))/2
        partrows=[]
        for part,name in enumerate(PARTS):
            mask=faceparts==part;target=triangle_distance(probes,tri[mask])
            n_dist=surface_distance(probes,native[part]);s_dist=surface_distance(probes,shadow[part])
            row={'part':name,'native_triangle_count':int(mask.sum()),
                 'native_mesh_to_mano_proxy':stats(surface_distance(centers[mask],native[part]),areas[mask]),
                 'native_mesh_to_shadow':stats(surface_distance(centers[mask],shadow[part]),areas[mask]),
                 'object_probe_distance_error_mano':stats(np.abs(n_dist-target)),
                 'object_probe_distance_error_shadow':stats(np.abs(s_dist-target)),
                 'pointwise_band_changes':[]}
            for band in p['near_bands_m']:
                ref=target<=band
                row['pointwise_band_changes'].append({'band_m':band,'native_near':int(ref.sum()),
                    'mano_lost':int((ref&(n_dist>band)).sum()),'mano_added':int((~ref&(n_dist<=band)).sum()),
                    'shadow_lost':int((ref&(s_dist>band)).sum()),'shadow_added':int((~ref&(s_dist<=band)).sum())})
            partrows.append(row)
        rec={'index':index,'object_id':c['object_id'],'native_sha256':c['native']['sha256'],
             'source_reconstruction':reconstruction,'pose_normalization':nm.pose_diagnostics,'bone_fk_error':stats(bone_errors),
             'mano_rigid_tip_errors_m':native_tips.tolist(),'shadow_tip_errors_m':shadow_tips.tolist(),
             'object_probes':probe_meta,'parts':partrows,'physics_steps':0,
             'native_pose_roundtrip_scope':'15balljoints and freewrist, static only; collider errors reported independently',
             'bone_numerical_import_pass':bool(bone_errors.max()<=p['numerical_import_tolerance_m']),
             'native_reconstruction_pass':bool(reconstruction['max_vertex_error_m']<=p['numerical_import_tolerance_m'] and reconstruction['max_landmark_error_m']<=p['numerical_import_tolerance_m'] and reconstruction['faces_exact'])}
        write(out/'cases'/f'case_{index:02}.json',rec);records.append(rec)
        assert nm.data.time==sd.time==od.time==0
        print(json.dumps({'index':index,'bone_max_m':float(bone_errors.max()),'mano_tip_mean_m':float(native_tips.mean()),'shadow_tip_mean_m':float(shadow_tips.mean())}),flush=True)
    verify()
    report={'schema':p['schema'],'protocol_sha256':ph,'case_count':len(records),'counts':counts,
            'all_native_reconstructions_pass':all(c['native_reconstruction_pass'] for c in records),
            'all_native_bone_imports_pass':all(c['bone_numerical_import_pass'] for c in records),
            'max_native_bone_error_m':max(c['bone_fk_error']['max_m'] for c in records),
            'mano_tip_errors':stats(np.array([c['mano_rigid_tip_errors_m'] for c in records]).ravel()),
            'shadow_tip_errors':stats(np.array([c['shadow_tip_errors_m'] for c in records]).ravel()),
            'cases_with_lower_mean_mano_tip_error':int(sum(np.mean(c['mano_rigid_tip_errors_m'])<np.mean(c['shadow_tip_errors_m']) for c in records)),
            'object_summary':[], 'case_reports':[{'index':c['index'],'sha256':sha(out/'cases'/f"case_{c['index']:02}.json")} for c in records],
            'runtime':{'numpy':np.__version__,'mujoco':mujoco.__version__,'torch':torch.__version__,'trimesh':trimesh.__version__},
            'native_collision_fidelity_accepted':False,'dynamic_validation':False,'new_eval_data':False,
            'original_criteria_changed':False,'source_bindings_unchanged':True}
    for obj in dict.fromkeys(c['object_id'] for c in records):
        cs=[c for c in records if c['object_id']==obj]
        report['object_summary'].append({'object_id':obj,'cases':len(cs),
            'mano_tip_mean_m':float(np.mean([c['mano_rigid_tip_errors_m'] for c in cs])),
            'shadow_tip_mean_m':float(np.mean([c['shadow_tip_errors_m'] for c in cs])),
            'part_mean_object_probe_error_mano_m':float(np.mean([x['object_probe_distance_error_mano']['mean_m'] for c in cs for x in c['parts']])),
            'part_mean_object_probe_error_shadow_m':float(np.mean([x['object_probe_distance_error_shadow']['mean_m'] for c in cs for x in c['parts']])),
            'part_case_pairs_mano_surface_mean_better':sum(x['native_mesh_to_mano_proxy']['mean_m']<x['native_mesh_to_shadow']['mean_m'] for c in cs for x in c['parts'])})
    write(out/'results.json',report);print(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--project');parser.add_argument('--hug')
    parser.add_argument('--dependency-dir',action='append',default=[])
    args=parser.parse_args()
    if args.action=='prepare':
        if not args.project or not args.hug:parser.error('prepare requires --project and --hug')
        prepare(args)
    else:run(args)

if __name__=='__main__':main()
