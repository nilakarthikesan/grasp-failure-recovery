"""Independent analytic-box verification of recorded MugB full-source-URDF-wrist cover leaves.

No branch/search/model inference, physics step or new object-query selection.
"""
from pathlib import Path
import hashlib,json,gzip,pickle,warnings,itertools,importlib.util,time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import chumpy
from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_body_id,_link_positions
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose,camera_landmarks_to_wrist

OUT=Path(__file__).parent
COUNTS={'actual':0,'selftests':0};TEST=False
def forbid(*args,**kwargs):
    COUNTS['selftests' if TEST else 'actual']+=1
    raise RuntimeError('Independent palm-bound check forbids physics integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbid)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ident(m,kind,name):
    result=mujoco.mj_name2id(m,kind,name);assert result>=0,name;return result
def array_sha(value):return hashlib.sha256(np.asarray(value,dtype='<f8').tobytes()).hexdigest()
def nearest_triangle_distance(point,triangles):
    """Plane projection inside a facet plus exact three segment candidates."""
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];ab=b-a;ac=c-a
    n=np.cross(ab,ac);nn=np.einsum('ij,ij->i',n,n);assert np.all(nn>0)
    ap=point-a;dot=np.einsum('ij,ij->i',ap,n)
    projected=point-dot[:,None]/nn[:,None]*n;v=projected-a
    d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
    d20=np.einsum('ij,ij->i',v,ab);d21=np.einsum('ij,ij->i',v,ac);den=d00*d11-d01*d01;assert np.all(den>0)
    u=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;inside=(u>=0)&(w>=0)&(u+w<=1)
    squared=np.where(inside,dot*dot/nn,np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start;t=np.clip(np.einsum('ij,ij->i',point-start,edge)/np.einsum('ij,ij->i',edge,edge),0,1)
        delta=point-(start+t[:,None]*edge);squared=np.minimum(squared,np.einsum('ij,ij->i',delta,delta))
    return float(np.sqrt(squared.min()))

def main():
    global TEST
    started=time.perf_counter();pp=OUT/'protocol.json';p=read(pp);ph=sha(pp)
    assert sha(__file__)==p['script_sha256']
    assert not (OUT/'report.json').exists()
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    TEST=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Physics guard selftest failed')
    TEST=False
    bound=read(p['bound_report']);bp=read(p['bound_protocol']);case_result=next(c for c in bound['cases'] if c['source_inventory_index']==28)
    assert len(case_result['covered_leaves'])==p['leaf_midpoint_count']==171 and not case_result['unresolved_leaves']
    assert bound['protocol_sha256']==sha(p['bound_protocol'])
    inventory=read(p['inventory'])['modern_cases'];case=inventory[28]
    assert case_result['native_sha256']==case['native']['sha256']
    m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
    with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
    stored=case['stored_shadow_command'];names=stored['joint_names'];row=stored['row_index']
    with np.load(stored['path'],allow_pickle=False) as z:
        assert z['joint_names'].tolist()==names;q0=z['nominal_target_rad'][row].copy();rootpos=z['root_target_position_m'][row].copy();rootquat=z['root_target_quaternion_wxyz'][row].copy()
    ja=np.asarray([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
    rootadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);objadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')])
    palm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');object_body=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');forearm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm')
    d.qpos[ja]=q0;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat;d.qvel[:]=0
    if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
    mujoco.mj_forward(m,d);nominal=d.qpos.copy();anchor=d.xpos[palm].copy();original_Rp=d.xmat[palm].reshape(3,3).copy();opos=d.xpos[object_body].copy();oR=d.xmat[object_body].reshape(3,3).copy();object_q=nominal[objadr:objadr+7].copy()
    prior=json.loads(gzip.decompress(Path(p['prior_preflight']).read_bytes()));prior_case=next(c for c in prior['cases'] if c['source_inventory_index']==28)
    assert prior['protocol_sha256']==p['prior_preflight_protocol_sha256']
    assert array_sha(nominal)==prior_case['scene_state_binding']['original_qpos_float64_sha256']==prior_case['scene_state_binding']['selected_qpos_float64_sha256']
    assert prior_case['engineering_guard_independent_audit']['source_preflight_status']=='PASS'
    grasp=load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);T_wrist=T_wc@grasp.T_camera_wrist
    anchor_error=float(np.linalg.norm(anchor-T_wrist[:3,3]));assert anchor_error<p['native_anchor_comparison_m']
    validator=ShadowPoseValidator(Path(p['urdf']).parent.parent)
    local_landmarks=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
    validator.set_pose(RetargetedHandPose(tuple(names),q0,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist))
    fixedp=_body_id(validator.model,'palm');fixed_R=validator.data.xmat[fixedp].reshape(3,3).copy();Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
    source_R_error=float(np.max(np.abs(original_Rp-Rop@fixed_R)));assert source_R_error<p['source_rotation_comparison_maxabs']
    source_links=_link_positions(validator.model,validator.data,validator.source_urdf);expected=(source_links-validator.data.xpos[fixedp])@Rop.T+T_wrist[:3,3]
    scene_links=_link_positions(m,d,validator.source_urdf);source_link_error=float(np.linalg.norm(scene_links-expected,axis=1).max());assert source_link_error<p['source_link_comparison_m']
    def descendant(body,root):
        while body>0:
            if body==root:return True
            body=int(m.body_parentid[body])
        return False
    def anatomical_part(body):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,body)
        return 'forearm_carrier' if name=='forearm' else 'wrist_carrier' if name=='wrist' else 'index' if name.startswith('ff') else 'middle' if name.startswith('mf') else 'ring' if name.startswith('rf') else 'little' if name.startswith('lf') else 'thumb' if name.startswith('th') else 'palm'
    hand=[g for g in range(m.ngeom) if descendant(int(m.geom_bodyid[g]),forearm) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    palm_geoms=[g for g in hand if anatomical_part(int(m.geom_bodyid[g]))=='palm']
    object_geoms=[g for g in range(m.ngeom) if descendant(int(m.geom_bodyid[g]),object_body) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert palm_geoms==case_result['palm_geom_ids']==list(range(8,16)) and object_geoms and m.npair==0
    assert all(int(m.geom_bodyid[g])==palm and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX for g in palm_geoms)
    exclusions=[sorted([int(s)>>16,int(s)&65535]) for s in m.exclude_signature]
    compatibility=[]
    for g in palm_geoms:
        for o in object_geoms:
            valid=bool((int(m.geom_contype[g])&int(m.geom_conaffinity[o])) or (int(m.geom_contype[o])&int(m.geom_conaffinity[g])))
            excluded=sorted([int(m.geom_bodyid[g]),int(m.geom_bodyid[o])]) in exclusions
            assert valid and not excluded;compatibility.append({'palm_geom':g,'object_geom':o,'mask_compatible':valid,'body_excluded':excluded})
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)));radius_records=[]
    for g in palm_geoms:
        matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,m.geom_quat[g]);local_R=matrix.reshape(3,3)
        corners=(signs*m.geom_size[g])@local_R.T+m.geom_pos[g];raw=float(np.linalg.norm(corners,axis=1).max())
        original_record=next(r for r in case_result['radius_geometry_records'] if r['geom_id']==g)
        assert np.array_equal(m.geom_size[g],original_record['size']) and np.array_equal(m.geom_pos[g],original_record['geom_local_position_m']) and np.array_equal(m.geom_quat[g],original_record['geom_local_quaternion_wxyz'])
        assert abs(raw-original_record['raw_maximum_vertex_radius_m'])<=p['reproduction_arithmetic_tolerance_m']
        radius_records.append({'geom_id':g,'corners_palm_frame_m':corners.tolist(),'independent_exact_box_corner_radius_m':raw})
    radius=max(r['independent_exact_box_corner_radius_m'] for r in radius_records);rho=case_result['palm_radius_upper_m']
    assert radius<rho and abs(radius-case_result['palm_radius_raw_maximum_m'])<=p['reproduction_arithmetic_tolerance_m']
    assert abs((rho-radius)-bp['rho_rounding_allowance_m'])<=p['reproduction_arithmetic_tolerance_m']
    ancestors=[];body=palm
    while body>0:
        for j in range(int(m.body_jntadr[body]),int(m.body_jntadr[body]+m.body_jntnum[body])):
            if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE:continue
            assert m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and abs(np.linalg.norm(m.jnt_axis[j])-1)<1e-14
            ancestors.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j))
        body=int(m.body_parentid[body])
    assert ancestors==case_result['palm_ancestor_joints']==['WRJ1','WRJ2']
    physical={j.get('name'):[float(j.find('limit').get(k)) for k in ('lower','upper')] for j in ET.parse(p['urdf']).getroot().findall('joint') if j.find('limit') is not None}
    wi=np.asarray([names.index(n) for n in ['WRJ1','WRJ2']],int)
    limits=np.asarray([physical[n] for n in names]);domain=limits[wi].copy();assert p['domain_mode']=='full_source_URDF_WRJ_limits'
    assert np.array_equal(domain,case_result['wrist_domain_rad']) and np.array_equal(q0[wi],case_result['wrist_q0_rad'])
    witnesses=case_result['existing_witnesses'];frozen=read(p['witness_file'])['witnesses']
    assert witnesses==[w for w in frozen if w['native_sha256']==case_result['native_sha256'] and w['anatomical_part']=='palm'] and len(witnesses)==2
    vs=[g for g in range(m.ngeom) if descendant(int(m.geom_bodyid[g]),object_body) and not (m.geom_contype[g] or m.geom_conaffinity[g])];assert len(vs)==1
    g=vs[0];mesh=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[mesh]);nv=int(m.mesh_vertnum[mesh]);fa=int(m.mesh_faceadr[mesh]);nf=int(m.mesh_facenum[mesh])
    vertices=m.mesh_vert[va:va+nv].astype(float);faces=m.mesh_face[fa:fa+nf].astype(int)
    object_vertices=(vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-opos)@oR;tri=object_vertices[faces]
    spec=importlib.util.spec_from_file_location('frozen_independent_cell_construction',p['cell_library']);cells=importlib.util.module_from_spec(spec);spec.loader.exec_module(cells)
    children,centroids,areas,radii,diameter,faceids,depths=cells.triangle_cells(tri,.002)
    source=pickle.loads(Path(case['native']['path']).read_bytes())
    native_vertices=(np.asarray(source['mesh_vertices'])@T_wc[:3,:3].T+T_wc[:3,3]-opos)@oR;native_faces=np.asarray(source['mesh_faces'],int)
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(Path(p['mano_asset']).read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);groups=[[0],[1,2,3],[4,5,6],[10,11,12],[7,8,9],[13,14,15]]
    group_weights=np.column_stack([weights[:,ids].sum(1) for ids in groups]);vlabel=np.argmax(group_weights,axis=1);labels=vlabel[native_faces]
    counts=np.column_stack([(labels==j).sum(1) for j in range(6)]);major=counts.max(1);means=group_weights[native_faces].mean(1)
    triangle_labels=np.argmax(np.where(counts==major[:,None],means,-np.inf),axis=1);native_tri=native_vertices[native_faces][triangle_labels==0]
    witness_records=[]
    for w in witnesses:
        i=w['child_cell_index'];q=np.asarray(w['centroid_object_m']);assert np.array_equal(centroids[i],q)
        assert faceids[i]==w['original_visual_face_index_after_degenerate_exclusion'] and abs(areas[i]-w['cell_area_mm2'])<=p['reproduction_arithmetic_tolerance_m']
        assert abs(radii[i]-w['centroid_radius_m'])<=p['reproduction_arithmetic_tolerance_m']
        nd=nearest_triangle_distance(q,native_tri);assert abs(nd-w['native_centroid_distance_m'])<=p['reproduction_arithmetic_tolerance_m']
        native_upper=nd+radii[i]+max(w['numerical_allowance_m'],bp['effective_numeric_allowance_m']);assert native_upper<=w['band_mm']/1000
        witness_records.append({'child_cell_index':i,'native_distance_independent_m':nd,'cell_radius_reconstructed_m':float(radii[i]),'cell_area_mm2':float(areas[i]),'native_upper_with_100nm_allowance_m':native_upper,'band_m':w['band_mm']/1000})
    qpoints=np.asarray([w['centroid_object_m'] for w in witnesses]);world_points=qpoints@oR.T+opos;records=[];max_difference=0.;max_anchor_error=0.
    for leaf in case_result['covered_leaves']:
        box=np.asarray(leaf['wrist_box_rad']);center=box.mean(axis=1);values=q0.copy();values[wi]=center
        assert np.all(center>=domain[:,0]) and np.all(center<=domain[:,1])
        d.qpos[:]=nominal;d.qpos[ja]=values;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat;mujoco.mj_forward(m,d)
        correction=rootpos-(d.xpos[palm]-anchor);d.qpos[rootadr:rootadr+3]=correction
        if m.nmocap:d.mocap_pos[0]=correction;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d)
        anchor_delta=float(np.linalg.norm(d.xpos[palm]-anchor));max_anchor_error=max(max_anchor_error,anchor_delta)
        assert anchor_delta<=p['anchor_hold_numeric_m'] and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],object_q)
        assert np.array_equal(d.qpos[rootadr+3:rootadr+7],rootquat) and np.array_equal(d.xpos[object_body],opos) and np.array_equal(d.xmat[object_body].reshape(3,3),oR)
        allowed=set(ja[wi].tolist()+list(range(rootadr,rootadr+3)))
        assert all(d.qpos[j]==nominal[j] for j in range(m.nq) if j not in allowed)
        distances=[]
        for g in palm_geoms:
            local=(world_points-d.geom_xpos[g])@d.geom_xmat[g].reshape(3,3);delta=np.abs(local)-m.geom_size[g]
            unsigned=np.abs(np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(delta.max(axis=1),0));distances.append(unsigned)
        all_distances=np.min(np.stack(distances),axis=0);j=leaf['witness_index'];dist=float(all_distances[j]);difference=abs(dist-leaf['centroid_distance_at_midpoint_m']);max_difference=max(max_difference,difference)
        assert difference<=p['reproduction_arithmetic_tolerance_m']
        half=(box[:,1]-box[:,0])/2;motion=rho*float(half.sum());w=witnesses[j]
        lower=dist-motion-w['centroid_radius_m']-max(w['numerical_allowance_m'],bp['effective_numeric_allowance_m'])-bp['extra_transform_arithmetic_allowance_m'];margin=lower-w['band_mm']/1000
        assert margin>0 and abs(margin-leaf['strict_margin_m'])<=p['reproduction_arithmetic_tolerance_m']
        records.append({'path':leaf['path'],'wrist_midpoint_rad':center.tolist(),'witness_index':j,'independent_box_union_surface_distance_m':dist,'recorded_distance_difference_m':difference,'independent_lower_m':lower,'independent_strict_margin_m':margin,'root_quaternion_fixed':True,'only_wrist_and_required_root_translation_changed':True})
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    assert sha(pp)==ph and sha(__file__)==p['script_sha256'] and COUNTS['actual']==0
    report={'schema':'hug_palm_full_wrist_bound_independent_geometry_verification_v2','result':'PASS','protocol_sha256':ph,'script_sha256':sha(__file__),'bound_report_sha256':sha(p['bound_report']),
        'bound_protocol_sha256':sha(p['bound_protocol']),'source_inventory_index':28,'native_sha256':case_result['native_sha256'],'initial_scene_qpos_float64_sha256':array_sha(nominal),'matches_prior_independent_complete_scene_hash':True,
        'native_wrist_palm_position_error_m':anchor_error,'source_URDF_palm_rotation_maxabs':source_R_error,'source_URDF_scene_max_link_error_m':source_link_error,
        'palm_geom_ids':palm_geoms,'all_palm_geoms_are_direct_rigid_boxes':True,'collision_compatibility':compatibility,'palm_ancestor_hinges':ancestors,
        'independent_palm_radius_raw_m':radius,'declared_palm_radius_upper_m':rho,'independent_box_corner_radius_records':radius_records,
        'native_witness_cell_verification':witness_records,'recorded_leaf_midpoints_verified':len(records),'maximum_midpoint_distance_difference_m':max_difference,
        'minimum_independently_recomputed_strict_margin_m':min(r['independent_strict_margin_m'] for r in records),'maximum_leaf_palm_anchor_hold_error_m':max_anchor_error,
        'root_orientation_fixed_and_translation_uniquely_anchor_determined':True,'leaves':records,'guards':COUNTS,'physics_steps':0,'new_inference':False,'new_eval_data':False,'new_query_sites':False,
        'runtime_seconds':time.perf_counter()-started,'scope':'Only fixed recorded171 MugB leaves across full source-URDF WRJ limits, fixed original root rotation/native palm position and actual eight palm boxes. Conditional uniform100nm distance+100nm transform arithmetic allowances. No bound search rerun, arbitrary free-root impossibility or all-HUG-source claim.',
        'full_surface_HUG_mapping_accepted':False,'native_failure_label_preservation':'NOT_TESTED','input_sha256':p['input_sha256']}
    with (OUT/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ['result','recorded_leaf_midpoints_verified','maximum_midpoint_distance_difference_m','minimum_independently_recomputed_strict_margin_m','initial_scene_qpos_float64_sha256','runtime_seconds']},indent=2))

if __name__=='__main__':main()
