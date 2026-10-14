"""Reproducible HUG-only geometric diagnostic; no physics or learned inference.

Reuses a hash-verified setup prefix of the previous preserved audit, never its
measurement/writing section. All new writes stay next to this script.
"""
from pathlib import Path
import hashlib, json, time, sys
import numpy as np

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD = ROOT / 'research_artifacts/hug_mapping_confirmation_2026-10-14'
PROTOCOL = json.loads((OUT / 'protocol.json').read_text())
PROTOCOL_HASH = hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
manifest = json.loads((OLD / 'manifest.json').read_text())
for name, expected in manifest['files_sha256'].items():
    assert sha(OLD/name) == expected, (name, 'changed prior evidence')
setup_path = OLD / 'hug_surface_confirmation.py'
assert sha(setup_path) == PROTOCOL['reused_setup_script_sha256']
setup = setup_path.read_text().split('half=np.array([.025,.025,.025]);spacing=.002')[0]
# The prefix only loads assets, computes static FK, and builds collider shapes.
setup = setup.replace("OUT=Path('/private/tmp/grasp-hug-confirmation-2026-10-14')", "OUT=Path(__file__).resolve().parent")
exec(compile(setup, str(setup_path) + ':setup-only', 'exec'), globals())
assert sha(NATIVE/'proposal.pkl') == PROTOCOL['native_proposal_sha256']
assert sha(ASSET) == PROTOCOL['mano_asset_sha256']
initial_obj_world_pos = data.xpos[obj_id].copy()
initial_obj_world_R = data.xmat[obj_id].copy()
initial_qpos = data.qpos.copy()
assert data.time == 0

# Relabel native triangles by vertex-majority grouped skinning, as frozen above.
vertex_label = np.argmax(group_weights, axis=1)
face_vertex_label = vertex_label[native_faces]
counts = np.column_stack([(face_vertex_label == j).sum(axis=1) for j in range(len(PARTS))])
majority = counts.max(axis=1)
ties = (counts == majority[:, None]).sum(axis=1) > 1
means = group_weights[native_faces].mean(axis=1)
native_face_parts = np.argmax(np.where(counts == majority[:, None], means, -np.inf), axis=1)
mixed_faces = (counts > 0).sum(axis=1) > 1
all_parts = PARTS + ['forearm_carrier', 'wrist_carrier']
for g in geometries:
    if g['body'] == 'forearm': g['part'] = 'forearm_carrier'
    elif g['body'] == 'wrist': g['part'] = 'wrist_carrier'

object_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == obj_id and
                (model.geom_contype[g] or model.geom_conaffinity[g])]
assert len(object_geoms) == 1, object_geoms
og = object_geoms[0]
assert model.geom_type[og] == mujoco.mjtGeom.mjGEOM_BOX
half = model.geom_size[og].copy()
assert np.allclose(half, [.025,.025,.025], atol=1e-12, rtol=0)
object_geom_center = to_obj(data.geom_xpos[og])
object_geom_R = R_obj.T @ data.geom_xmat[og].reshape(3,3)
assert np.allclose(object_geom_center, 0, atol=1e-12)
assert np.allclose(object_geom_R, np.eye(3), atol=1e-12)
def compatible(geom):
    return bool((int(model.geom_contype[geom]) & int(model.geom_conaffinity[og])) or
                (int(model.geom_contype[og]) & int(model.geom_conaffinity[geom])))
excluded_body_pairs = []
for signature in model.exclude_signature:
    encoded = int(signature)
    excluded_body_pairs.append([encoded >> 16, encoded & 65535])
explicit_pairs = [{'geom1':int(model.pair_geom1[i]),'geom2':int(model.pair_geom2[i])} for i in range(model.npair)]
for g in geometries:
    bid = int(model.geom_bodyid[g['geom']])
    g['compatible_with_object'] = compatible(g['geom'])
    g['body_excluded_with_object'] = sorted([bid,obj_id]) in [sorted(pair) for pair in excluded_body_pairs]
active_geometries = [g for g in geometries if g['compatible_with_object'] and not g['body_excluded_with_object']]
assert len(active_geometries) == len(geometries), 'Prior comparison included geometries rejected by object filters'
assert not any(og in [p['geom1'],p['geom2']] for p in explicit_pairs), 'Explicit object pair overrides need treatment'

def cube_sdf(points):
    delta=np.abs(points)-half
    return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0)

def actual_geom_query(g, points):
    """Return signed per-geom distance, surface point and outward normal."""
    p=(points@R_obj.T+t_obj-g['pos'])@g['R']; kind=g['kind']; size=g['size']
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        length=np.linalg.norm(p,axis=1); sd=length-size[0]
        normal=p/np.maximum(length[:,None],1e-15)
        normal[length < 1e-15] = [1,0,0]
        q=p-sd[:,None]*normal
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        delta=np.abs(p)-size; outside=np.maximum(delta,0); od=np.linalg.norm(outside,axis=1)
        sd=od+np.minimum(delta.max(axis=1),0)
        normal=outside*np.sign(p)/np.maximum(od[:,None],1e-15)
        inside=od==0; axis=np.argmax(delta[inside],axis=1); normal[inside]=0
        row=np.flatnonzero(inside); normal[row,axis]=np.where(p[row,axis]>=0,1.,-1.)
        q=p-sd[:,None]*normal
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(p[:,:2],axis=1)
        rdir=p[:,:2]/np.maximum(radial[:,None],1e-15); rdir[radial < 1e-15]=[1,0]
        delta=np.column_stack([radial-size[0],np.abs(p[:,2])-size[1]])
        outside=np.maximum(delta,0); od=np.linalg.norm(outside,axis=1)
        sd=od+np.minimum(delta.max(axis=1),0)
        normal=np.column_stack([rdir*outside[:,0:1],np.sign(p[:,2])*outside[:,1]])/np.maximum(od[:,None],1e-15)
        inside=od==0; side=delta[:,0]>=delta[:,1]; normal[inside]=0
        normal[inside&side,:2]=rdir[inside&side]
        normal[inside&~side,2]=np.where(p[inside&~side,2]>=0,1.,-1.)
        q=p-sd[:,None]*normal
    else:
        q,index,distance=nearest_triangles(p,g['tri_local'],batch=64)
        in_geom=np.max(p@g['equations'][:,:3].T+g['equations'][:,3],axis=1)<=1e-10
        sd=np.where(in_geom,-distance,distance)
        normal=triangle_data(g['tri_local'])[0][index]
    return sd, to_obj(q@g['R'].T+g['pos']), normal@g['R'].T@R_obj

# Independent analytic sanity checks exercise both face interiors and edge cases.
sanity_points=np.array([[.2,.3,.4],[.5,.5,.2],[-.1,-.1,0],[0,0,1],[1,0,0]])
sanity_tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]]])
sanity_q,_,sanity_d=nearest_triangles(sanity_points,sanity_tri)
expected_q=np.array([[.2,.3,0],[.5,.5,0],[0,0,0],[0,0,0],[1,0,0]])
assert np.allclose(sanity_q,expected_q,atol=1e-14)
assert np.allclose(sanity_d,np.linalg.norm(sanity_points-expected_q,axis=1),atol=1e-14)
assert np.allclose(np.linalg.norm(native_normals,axis=1),1,atol=1e-12)

def summary(values):
    if len(values)==0: return None
    return {'min':float(np.min(values)), 'mean':float(np.mean(values)),
            'median':float(np.median(values)), 'p95':float(np.percentile(values,95)), 'max':float(np.max(values))}
def coverage(n,s,area):
    union=n|s; inter=n&s
    return {'native_count':int(n.sum()),'shadow_count':int(s.sum()),'intersection_count':int(inter.sum()),
            'union_count':int(union.sum()),'native_only_count':int((n&~s).sum()),'shadow_only_count':int((s&~n).sum()),
            'native_area_mm2':float(n.sum()*area),'shadow_area_mm2':float(s.sum()*area),
            'intersection_area_mm2':float(inter.sum()*area),'union_area_mm2':float(union.sum()*area),
            'lost_native_area_mm2':float((n&~s).sum()*area),'new_shadow_area_mm2':float((s&~n).sum()*area),
            'jaccard':float(inter.sum()/union.sum()) if union.any() else None,
            'native_retention':float(inter.sum()/n.sum()) if n.any() else None,
            'shadow_overlap':float(inter.sum()/s.sum()) if s.any() else None}
FACES=[('+X',0,1),('-X',0,-1),('+Y',1,1),('-Y',1,-1),('+Z',2,1),('-Z',2,-1)]
results={}
for spacing_mm in PROTOCOL['grid_spacing_mm']:
    tick=time.perf_counter(); spacing=spacing_mm/1000; nside=round(.05/spacing)
    uv=(np.arange(nside)+.5)*spacing-.025; nface=nside*nside
    grids=[]
    for label,axis,sign in FACES:
        other=[i for i in range(3) if i!=axis]; grid=np.zeros((nface,3));grid[:,axis]=sign*.025
        u,v=np.meshgrid(uv,uv,indexing='xy');grid[:,other[0]]=u.ravel();grid[:,other[1]]=v.ravel();grids.append(grid)
    query=np.concatenate(grids); N=len(query); area=spacing_mm**2
    print(json.dumps({'progress':'grid_started','spacing_mm':spacing_mm,'queries':N}),flush=True)
    native_d_by_part=np.full((N,len(all_parts)),np.inf)
    native_q_by_part=np.full((N,len(all_parts),3),np.nan)
    native_i_by_part=np.full((N,len(all_parts)),-1,dtype=int)
    for part in range(len(PARTS)):
        idx=np.flatnonzero(native_face_parts==part)
        q,i,d=nearest_triangles(query,native_triangles[idx],batch=64)
        native_d_by_part[:,part]=d; native_q_by_part[:,part]=q; native_i_by_part[:,part]=idx[i]
    native_part=np.argmin(native_d_by_part,axis=1); rows=np.arange(N)
    native_distance=native_d_by_part[rows,native_part]
    native_closest=native_q_by_part[rows,native_part]
    native_index=native_i_by_part[rows,native_part]
    native_normal=native_normals[native_index]
    native_inside=inside_mesh(query,native_triangles,batch=64)
    shadow_d_by_part=np.full_like(native_d_by_part,np.inf)
    shadow_q_by_part=np.full_like(native_q_by_part,np.nan)
    shadow_norm_by_part=np.full_like(native_q_by_part,np.nan)
    shadow_inside=np.zeros(N,dtype=bool)
    for g in active_geometries:
        # Skip a collider only if its exact bounds are >5mm from every cube point.
        lo=g['tri_object'].reshape(-1,3).min(axis=0); hi=g['tri_object'].reshape(-1,3).max(axis=0)
        gap=np.maximum(np.maximum(lo-half,-half-hi),0)
        if g['kind']==mujoco.mjtGeom.mjGEOM_MESH and np.linalg.norm(gap)>.005: continue
        sd,q,norm=actual_geom_query(g,query);shadow_inside|=sd<0
        p=all_parts.index(g['part']); d=np.abs(sd);update=d<shadow_d_by_part[:,p]
        shadow_d_by_part[update,p]=d[update];shadow_q_by_part[update,p]=q[update];shadow_norm_by_part[update,p]=norm[update]
    shadow_part=np.argmin(shadow_d_by_part,axis=1)
    shadow_distance=shadow_d_by_part[rows,shadow_part]
    shadow_closest=shadow_q_by_part[rows,shadow_part]
    shadow_normal=shadow_norm_by_part[rows,shadow_part]
    exterior=(~native_inside)&(~shadow_inside)&(native_distance>1e-8)&(shadow_distance>1e-8)
    n_cube_inside=cube_sdf(native_closest)<-1e-10
    s_cube_inside=cube_sdf(shadow_closest)<-1e-10
    angle=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',native_normal,shadow_normal),-1,1)))
    separation=np.linalg.norm(native_closest-shadow_closest,axis=1)*1000
    level={'spacing_mm':spacing_mm,'query_count':N,'cell_area_mm2':area,'native_mixed_triangle_query_count':int(mixed_faces[native_index].sum()),
           'native_tied_triangle_query_count':int(ties[native_index].sum()),'bands':{}}
    for band_mm in PROTOCOL['diagnostic_bands_mm']:
        band=band_mm/1000
        ng=native_inside|(native_distance<=band); sg=shadow_inside|(shadow_distance<=band)
        band_result={'occupied_or_near':coverage(ng,sg,area),'faces':{},'independent_per_part_unsigned':{},'exterior_nearest_labels':{}}
        for j,part in enumerate(all_parts):
            npart=native_d_by_part[:,j]<=band; spart=shadow_d_by_part[:,j]<=band
            band_result['independent_per_part_unsigned'][part]=coverage(npart,spart,area)
        nne=(~native_inside)&(native_distance>1e-8)&(native_distance<=band)
        sne=(~shadow_inside)&(shadow_distance>1e-8)&(shadow_distance<=band)
        common=exterior&(native_distance<=band)&(shadow_distance<=band)
        same=common&(native_part==shadow_part); different=common&~(native_part==shadow_part)
        def paired(sel):
            return {'query_count':int(sel.sum()),'area_mm2':float(sel.sum()*area),
                    'normal_angle_deg':summary(angle[sel]),'closest_point_separation_mm':summary(separation[sel]),
                    'native_closest_inside_cube_count':int((sel&n_cube_inside).sum()),
                    'shadow_closest_inside_cube_count':int((sel&s_cube_inside).sum()),
                    'both_closest_inside_cube_count':int((sel&n_cube_inside&s_cube_inside).sum()),
                    'one_closest_inside_cube_count':int((sel&(n_cube_inside^s_cube_inside)).sum()),
                    'neither_closest_inside_cube_count':int((sel&~n_cube_inside&~s_cube_inside).sum()),
                    'native_mixed_triangle_count':int((sel&mixed_faces[native_index]).sum()),
                    'native_tied_triangle_count':int((sel&ties[native_index]).sum()),
                    'neither_inside_cube_normals_deg':summary(angle[sel&~n_cube_inside&~s_cube_inside]),
                    'neither_inside_cube_separation_mm':summary(separation[sel&~n_cube_inside&~s_cube_inside])}
        band_result['exterior_nearest_labels']={
            'native_exterior_near_count':int(nne.sum()),'shadow_exterior_near_count':int(sne.sum()),
            'common_both_exterior_near_count':int(common.sum()),'common_same_label':paired(same),
            'common_different_label':paired(different),
            'native_near_shadow_not_exterior_near_count':int((nne&~sne).sum()),
            'shadow_near_native_not_exterior_near_count':int((sne&~nne).sum()),
            'common_label_confusion_counts':{part:{other:int((common&(native_part==j)&(shadow_part==k)).sum()) for k,other in enumerate(all_parts)} for j,part in enumerate(all_parts)},
            'common_same_label_by_part':{part:paired(same&(native_part==j)) for j,part in enumerate(all_parts)}}
        for fi,(name,axis,sign) in enumerate(FACES):
            face=np.zeros(N,dtype=bool);face[fi*nface:(fi+1)*nface]=True
            band_result['faces'][name]={'occupied_or_near':coverage(ng[face],sg[face],area),
                'independent_per_part_unsigned':{part:coverage((native_d_by_part[face,j]<=band),(shadow_d_by_part[face,j]<=band),area) for j,part in enumerate(all_parts)},
                'common_same_label':paired(same&face),'common_different_label':paired(different&face)}
        level['bands'][str(band_mm)]=band_result
    level['runtime_seconds']=time.perf_counter()-tick
    results[str(spacing_mm)]=level
    (OUT/f'grid_{spacing_mm}mm.json').write_text(json.dumps(level,indent=2)+'\n')
    print(json.dumps({'progress':'grid_completed','spacing_mm':spacing_mm,'seconds':level['runtime_seconds'],
                      'global_IoU':{band:body['occupied_or_near']['jaccard'] for band,body in level['bands'].items()}}),flush=True)

baseline=json.loads((OLD/'hug_surface_confirmation.json').read_text())
baseline_checks={}
for band in [2,5]:
    old=baseline['global_coverage'][f'{band}mm']; new=results['2']['bands'][str(band)]['occupied_or_near']
    pairs={'native_occupied_or_near_area_mm2':'native_area_mm2','shadow_occupied_or_near_area_mm2':'shadow_area_mm2',
           'intersection_area_mm2':'intersection_area_mm2','union_area_mm2':'union_area_mm2','jaccard':'jaccard'}
    differences={k:float(new[v]-old[k]) for k,v in pairs.items()}
    assert max(abs(value) for value in differences.values())<1e-8, differences
    baseline_checks[str(band)]={'passed':True,'new_minus_old':differences}
assert data.time==0
assert np.array_equal(initial_qpos,data.qpos)
assert np.array_equal(object_qpos,data.qpos[object_adr:object_adr+7])
assert np.array_equal(initial_obj_world_pos,data.xpos[obj_id])
assert np.array_equal(initial_obj_world_R,data.xmat[obj_id])
assert sha(OUT/'protocol.json')==PROTOCOL_HASH
for name,expected in manifest['files_sha256'].items():assert sha(OLD/name)==expected
inputs={str(p):sha(p) for p in [PFILE,NATIVE/'proposal.pkl',NATIVE/'scene.mjb',NATIVE/'scene_state.npz',NATIVE/'T_world_camera.npy',ASSET,setup_path,OLD/'manifest.json',OLD/'hug_native_reconstruction.json']}
report={'protocol_sha256':PROTOCOL_HASH,'script_sha256':sha(__file__),'input_sha256':inputs,
 'scope':PROTOCOL['scope'],'runtime_seconds':time.perf_counter()-START,
 'runtime_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__},
 'invariants':{'physics_steps':0,'guarded_apis':['mj_step','mj_step1','mj_step2'],'time_s':float(data.time),
    'object_qpos_unchanged':True,'object_world_pose_unchanged':True,'all_static_qpos_unchanged':True,
    'old_artifact_manifest_hashes_match':True,'native_and_asset_hashes_match':True,'protocol_unchanged':True},
 'object_collider':{'geom_id':int(og),'half_size_m':half.tolist(),'center_object_frame_m':object_geom_center.tolist(),
    'rotation_object_frame':object_geom_R.tolist(),'contype':int(model.geom_contype[og]),'conaffinity':int(model.geom_conaffinity[og]),
    'npair':int(model.npair),'explicit_pairs':explicit_pairs,'nexclude':int(model.nexclude),'excluded_body_pairs':excluded_body_pairs},
 'shadow_geoms':[{'id':g['geom'],'body':g['body'],'part':g['part'],'compatible_with_object':g['compatible_with_object'],
    'body_excluded_with_object':g['body_excluded_with_object']} for g in geometries],
 'native_triangle_labels':{'groups':GROUPS,'triangle_count':len(native_triangles),'mixed_vertex_labels':int(mixed_faces.sum()),
    'three_way_ties':int(ties.sum()),'count_by_part':{part:int((native_face_parts==j).sum()) for j,part in enumerate(PARTS)}},
 'analytic_nearest_triangle_sanity':{'cases':len(sanity_points),'passed':True},'baseline_checks':baseline_checks,
 'grids':results,'limitations':PROTOCOL['limitations'],
 'interpretation':'Diagnostic bands and grid refinement measure geometric agreement only. Exact recovery/frame checks do not imply hand equivalence. Contact fidelity remains unconfirmed; no native/target physical contact forces, task labels, calibration, or prediction changes.'}
(OUT/'hug_surface_refinement.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'completed':True,'seconds':report['runtime_seconds'],'report':str(OUT/'hug_surface_refinement.json')}),flush=True)
