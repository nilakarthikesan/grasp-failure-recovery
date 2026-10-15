"""Conditional palm witness cover on a fixed two-wrist engineering domain.

This is a static development diagnostic, not a grasp, physics run, outcome label,
or universal robot-reachability proof. All geometry witnesses pre-exist.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BASE=Path(__file__).parent
ACTUAL_STEPS=0
def forbidden(*args,**kwargs):
    global ACTUAL_STEPS
    ACTUAL_STEPS+=1
    raise RuntimeError('Static reachability diagnostics forbid integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def ident(m,k,n):
    i=mujoco.mj_name2id(m,k,n);assert i>=0,n;return i

def verify_partition(domain,leaves):
    paths=[tuple(x['path']) for x in leaves];assert len(set(paths))==len(paths)
    known=set(paths)
    for leaf,path in zip(leaves,paths,strict=True):
        assert leaf['depth']==len(path)
        assert not any(path[:i] in known for i in range(len(path)))
        box=np.asarray(domain,float).copy()
        for bit in path:
            assert bit in (0,1);axis=int(np.argmax(box[:,1]-box[:,0]));mid=box.mean(axis=1)[axis]
            box[axis,1-bit]=mid
        assert np.array_equal(box,np.asarray(leaf['wrist_box_rad']))
    maxdepth=max(map(len,paths));assert sum(1<<(maxdepth-len(path)) for path in paths)==1<<maxdepth
    return {'prefix_free':True,'all_rectangles_reconstructed_exactly':True,'binary_partition_fraction_exactly_one':True,'leaf_count':len(leaves)}

def manufactured_controls():
    domain=np.array([[0.,1.],[0.,1.]])
    leaves=[{'path':[0],'depth':1,'wrist_box_rad':[[0.,.5],[0.,1.]]},{'path':[1,0],'depth':2,'wrist_box_rad':[[.5,1.],[0.,.5]]},{'path':[1,1],'depth':2,'wrist_box_rad':[[.5,1.],[.5,1.]]}]
    assert verify_partition(domain,leaves)['binary_partition_fraction_exactly_one']
    for damaged in (leaves[:-1],leaves+[leaves[0]]):
        try:verify_partition(domain,damaged)
        except AssertionError:pass
        else:raise AssertionError('Malformed manufactured partition accepted')
    return {'one_complete_partition_accepted':True,'one_gap_and_one_duplicate_rejected':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,default=BASE/'protocol.json');args=ap.parse_args()
    assert not (BASE/'report.json').exists() and not list(BASE.glob('*_bound.json')),'Refuse overwrite of a completed or partial run'
    p=read(args.protocol);ph=sha(args.protocol);assert sha(__file__)==p['implementation_sha256']
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    a=load('reachability_frozen_geometry',p['geometry_library'])
    # Imported historical library also installs its static integration guard.
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
    inventory=read(p['inventory'])['modern_cases'];source=read(p['witness_file'])['witnesses']
    xr=ET.parse(p['urdf']).getroot();physical={j.get('name'):[float(j.find('limit').get(x)) for x in ('lower','upper')] for j in xr.findall('joint') if j.find('limit') is not None}
    controls_summary=manufactured_controls();results=[];started=time.perf_counter()
    for case_index in p['case_indices']:
        case=inventory[case_index];w=[x for x in source if x['native_sha256']==case['native']['sha256'] and x['anatomical_part']=='palm'];assert w
        for x in w:
            assert x['relationship_change']=='lost_native'
            assert x['native_centroid_distance_m']+x['centroid_radius_m']+max(x['numerical_allowance_m'],p['effective_numeric_allowance_m'])<=x['band_mm']/1000
            cache=Path(p['query_cache_root'])/(x['case_id']+'_density'+str(x['density_mm'])+'mm_queries.npz')
            assert sha(cache)==x['original_query_cache_sha256']
            with np.load(cache,allow_pickle=False) as z:
                i=x['child_cell_index'];assert np.array_equal(z['query_object_m'][i],np.asarray(x['centroid_object_m']))
                assert abs(float(z['native_distance_by_part_m'][i,0])-x['native_centroid_distance_m'])<1e-12
                assert abs(float(z['area_mm2'][i])-x['cell_area_mm2'])<1e-12
        m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
        c=case['stored_shadow_command'];names=c['joint_names'];row=c['row_index']
        with np.load(c['path'],allow_pickle=False) as z:
            assert z['joint_names'].tolist()==names;q0=z['nominal_target_rad'][row].copy();rootpos=z['root_target_position_m'][row].copy();rootquat=z['root_target_quaternion_wxyz'][row].copy()
        ja=np.array([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        rootadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);objadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);palm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');obj=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')
        d.qpos[ja]=q0;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat;d.qvel[:]=0
        if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d);nominal=d.qpos.copy();anchor=d.xpos[palm].copy();opos=d.xpos[obj].copy();oR=d.xmat[obj].reshape(3,3).copy();object_q=nominal[objadr:objadr+7].copy()
        forearm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm')
        def descends(b,root):
            while b>0:
                if b==root:return True
                b=int(m.body_parentid[b])
            return False
        def frozen_part(b):
            n=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)
            return 'forearm_carrier' if n=='forearm' else 'wrist_carrier' if n=='wrist' else 'index' if n.startswith('ff') else 'middle' if n.startswith('mf') else 'ring' if n.startswith('rf') else 'little' if n.startswith('lf') else 'thumb' if n.startswith('th') else 'palm'
        palm_geoms=[g for g in range(m.ngeom) if descends(int(m.geom_bodyid[g]),forearm) and (m.geom_contype[g] or m.geom_conaffinity[g]) and frozen_part(int(m.geom_bodyid[g]))=='palm'];assert palm_geoms
        assert all(int(m.geom_bodyid[g])==palm for g in palm_geoms),'Frozen palm surfaces must all rigidly attach to palm'
        base=[a.geom_rep(m,d,g,oR,opos) for g in palm_geoms];Rp=d.xmat[palm].reshape(3,3).copy()
        assert all(g['kind'] in (mujoco.mjtGeom.mjGEOM_MESH,mujoco.mjtGeom.mjGEOM_BOX) for g in base),'Only exact box/convex mesh palm certificates implemented'
        local_vertices=[];radius_records=[]
        for g in base:
            world=g['tri'].reshape(-1,3)@d.geom_xmat[g['id']].reshape(3,3).T+d.geom_xpos[g['id']]
            vv=(world-anchor)@Rp;local_vertices.append(vv);r=float(np.linalg.norm(vv,axis=1).max());imax=int(np.argmax(np.linalg.norm(vv,axis=1)))
            radius_records.append({'geom_id':g['id'],'body':g['body'],'kind':g['kind'],'size':g['size'].tolist(),'geom_local_position_m':m.geom_pos[g['id']].tolist(),'geom_local_quaternion_wxyz':m.geom_quat[g['id']].tolist(),'contype':g['contype'],'conaffinity':g['conaffinity'],'compiled_local_triangle_vertices_sha256':hashlib.sha256(g['tri'].tobytes()).hexdigest(),'palm_frame_unique_vertices_m':np.unique(vv,axis=0).tolist(),'raw_maximum_vertex_radius_m':r,'maximum_vertex_palm_frame_m':vv[imax].tolist()})
        rho=float(max(np.linalg.norm(v,axis=1).max() for v in local_vertices))+p['rho_rounding_allowance_m']
        wi=np.array([names.index(n) for n in p['wrist_joint_names']],int);assert len(wi)==2
        limits=np.asarray([physical[n] for n in names]);lower=np.maximum(limits[:,0],q0-p['maximum_joint_delta_rad']);upper=np.minimum(limits[:,1],q0+p['maximum_joint_delta_rad']);domain=np.column_stack([lower[wi],upper[wi]])
        # Analytic relation check: palm chain has exactly these moving ancestors.
        ancestors=[];body=palm
        while body>0:
            for j in range(int(m.body_jntadr[body]),int(m.body_jntadr[body]+m.body_jntnum[body])):
                if int(m.jnt_type[j])!=int(mujoco.mjtJoint.mjJNT_FREE):
                    assert int(m.jnt_type[j])==int(mujoco.mjtJoint.mjJNT_HINGE)
                    assert abs(np.linalg.norm(m.jnt_axis[j])-1)<1e-14
                    ancestors.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j))
            body=int(m.body_parentid[body])
        assert sorted(ancestors)==sorted(p['wrist_joint_names']),ancestors
        qpoints=np.asarray([x['centroid_object_m'] for x in w]);radii=np.asarray([x['centroid_radius_m'] for x in w]);eps=np.asarray([max(x['numerical_allowance_m'],p['effective_numeric_allowance_m']) for x in w]);bands=np.asarray([x['band_mm']/1000 for x in w])
        calls=0
        def pose(values):
            nonlocal calls
            d.qpos[:]=nominal;d.qpos[ja]=values;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d);corrected=rootpos-(d.xpos[palm]-anchor);d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d);calls+=1
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],object_q)
            assert np.linalg.norm(d.xpos[palm]-anchor)<1e-12
            assert np.array_equal(d.xpos[obj],opos) and np.array_equal(d.xmat[obj].reshape(3,3),oR)
            updated=[]
            for g in base:
                gg=g.copy();gi=g['id'];gg['R']=oR.T@d.geom_xmat[gi].reshape(3,3);gg['pos']=(d.geom_xpos[gi]-opos)@oR;gg['tri_object']=gg['tri']@gg['R'].T+gg['pos'];updated.append(gg)
            return updated
        def distances(x):
            q=q0.copy();q[wi]=x;geoms=pose(q)
            return np.min(np.stack([np.abs(a.geom_query(g,qpoints)[0]) for g in geoms]),axis=0)
        # Fixed deterministic controls check finger independence and analytic motion cap;
        # they are regression checks, not the mathematical source of the bound.
        controls=[];geoms0=pose(q0)
        for frac in p['control_fractions']:
            qq=lower+float(frac)*(upper-lower);geoms=pose(qq)
            qsame=q0.copy();qsame[wi]=qq[wi];gsame=pose(qsame)
            independence=max(float(np.max(np.abs(x['tri_object']-y['tri_object']))) for x,y in zip(geoms,gsame,strict=True));assert independence<1e-14
            movement=max(float(np.linalg.norm(x['tri_object']-y['tri_object'],axis=2).max()) for x,y in zip(geoms,geoms0,strict=True))
            bound=rho*float(np.abs(qq[wi]-q0[wi]).sum());assert movement<=bound+p['control_numeric_allowance_m']
            controls.append({'fraction':frac,'finger_independence_maxabs_m':independence,'max_corresponded_palm_vertex_movement_m':movement,'analytic_upper_m':bound})
        stack=[(domain.copy(),0,())];leaves=[];unresolved=[];nodes=0;minimum_margin=np.inf;stop_reason='stack_exhausted'
        while stack:
            if nodes>=p['maximum_nodes']:
                unresolved.extend({'path':list(pp),'depth':dd,'wrist_box_rad':bb.tolist(),'not_evaluated_due_to_budget':True} for bb,dd,pp in stack);stack=[];stop_reason='node_budget';break
            box,depth,path=stack.pop();nodes+=1;center=box.mean(axis=1);half=(box[:,1]-box[:,0])/2
            dist=distances(center);motion=rho*float(half.sum());margin=dist-motion-radii-eps-p['extra_transform_arithmetic_allowance_m']-bands
            eligible=np.flatnonzero(margin>0)
            if len(eligible):
                j=int(eligible[np.argmax(margin[eligible])]);minimum_margin=min(minimum_margin,float(margin[j]));leaves.append({'path':list(path),'depth':depth,'wrist_box_rad':box.tolist(),'witness_index':j,'centroid_distance_at_midpoint_m':float(dist[j]),'palm_surface_motion_bound_m':motion,'mapped_all_q_all_cell_distance_lower_m':float(dist[j]-motion-radii[j]-eps[j]-p['extra_transform_arithmetic_allowance_m']),'band_m':float(bands[j]),'strict_margin_m':float(margin[j])});continue
            if depth>=p['maximum_subdivision_depth']:
                unresolved.append({'path':list(path),'depth':depth,'wrist_box_rad':box.tolist(),'best_existing_witness_margin_m':float(margin.max())});continue
            axis=int(np.argmax(box[:,1]-box[:,0]));mid=center[axis];left=box.copy();right=box.copy();left[axis,1]=mid;right[axis,0]=mid
            stack.append((right,depth+1,path+(1,)));stack.append((left,depth+1,path+(0,)))
        area=lambda bb:float(np.prod(np.asarray(bb)[:,1]-np.asarray(bb)[:,0]))
        domain_area=area(domain);cert_area=sum(area(x['wrist_box_rad']) for x in leaves);unres_area=sum(area(x['wrist_box_rad']) for x in unresolved);assert abs(cert_area+unres_area-domain_area)<p['rectangle_area_check_allowance_rad2']
        assert nodes<=p['maximum_nodes'];partition=verify_partition(domain,leaves+unresolved)
        result={'case_id':w[0]['case_id'],'source_inventory_index':case_index,'native_sha256':case['native']['sha256'],'wrist_joint_names':p['wrist_joint_names'],'wrist_q0_rad':q0[wi].tolist(),'wrist_domain_rad':domain.tolist(),'palm_ancestor_joints':ancestors,'palm_geom_ids':palm_geoms,'palm_radius_upper_m':rho,'radius_method':'Maximum Euclidean norm of vertices of all actual compiled convex palm geom hull triangles about fixed palm anchor; convexity bounds whole surfaces.','controls':controls,'existing_witnesses':w,'nodes':nodes,'geometry_pose_calls':calls,'stop_reason':stop_reason,'certified_node_count':len(leaves),'unresolved_node_count':len(unresolved),'wrist_rectangle_area_rad2':domain_area,'certified_rectangle_area_rad2':cert_area,'unresolved_rectangle_area_rad2':unres_area,'minimum_leaf_strict_margin_m':float(minimum_margin) if len(leaves) else None,'status':'CURRENT_DOMAIN_STRICT_PALM_PRESERVATION_IMPOSSIBLE_CONDITIONAL' if not unresolved else 'UNRESOLVED','covered_leaves':leaves,'unresolved_leaves':unresolved,'scope_does_not_cover_all_URDF_robot_poses':True,'finger_constraints_ignored_enlarges_domain_conservatively':True}
        result['radius_geometry_records']=radius_records;result['palm_radius_raw_maximum_m']=float(max(x['raw_maximum_vertex_radius_m'] for x in radius_records));result['palm_radius_padding_m']=p['rho_rounding_allowance_m'];result['effective_distance_allowance_m']=p['effective_numeric_allowance_m'];result['extra_transform_arithmetic_allowance_m']=p['extra_transform_arithmetic_allowance_m'];result['partition_verification']=partition;result['strict_node_budget_verified']=True
        results.append(result);write(BASE/(w[0]['case_id']+'_bound.json'),result)
        print(json.dumps({k:result[k] for k in ('case_id','status','nodes','certified_node_count','unresolved_node_count','minimum_leaf_strict_margin_m')}),flush=True)
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    assert sha(args.protocol)==ph and sha(__file__)==p['implementation_sha256'];assert ACTUAL_STEPS==0
    write(BASE/'report.json',{'schema':p['schema'],'protocol_sha256':ph,'implementation_sha256':sha(__file__),'manufactured_partition_controls':controls_summary,'all_input_hashes_unchanged':True,'physics_steps':0,'new_eval_data':False,'new_inference':False,'new_query_sites':False,'training':False,'runtime_seconds':time.perf_counter()-started,'cases':results,'limitations':p['limitations']})
if __name__=='__main__':main()
