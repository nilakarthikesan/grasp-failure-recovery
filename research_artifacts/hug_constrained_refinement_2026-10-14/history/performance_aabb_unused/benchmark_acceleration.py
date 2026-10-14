"""Versioned static parity/timing only; does not run the constrained solver."""
from pathlib import Path
import hashlib, importlib.util, json, pickle, time, warnings
import numpy as np
import mujoco
import numba
from visual_membership_aabb import VisualMembershipAABB

HERE = Path(__file__).parent
ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
V1 = Path('/private/tmp/hug_constrained_refinement_2026-10-14')
COUNTS = {'actual': 0, 'selftests': 0, 'testing': False}
def blocked(*args, **kwargs):
    COUNTS['selftests' if COUNTS['testing'] else 'actual'] += 1
    raise RuntimeError('Static performance benchmark forbids integration')
def guard():
    for name in ('mj_step', 'mj_step1', 'mj_step2'): setattr(mujoco, name, blocked)
guard()
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
direction = load('frozen_direction_for_acceleration', Path('/private/tmp/hug_constrained_validation_2026-10-14/exact_direction.py'))
refresh = load('frozen_rigid_refresh_for_acceleration', V1/'geometry_helpers.py')
guard()
a, o = direction.a, direction.o
original = o.visual_membership
accelerated = VisualMembershipAABB(original)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def equal(x,y):
    if isinstance(x,np.ndarray): return isinstance(y,np.ndarray) and x.dtype==y.dtype and np.array_equal(x,y,equal_nan=True)
    if isinstance(x,dict): return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(xx,yy) for xx,yy in zip(x,y,strict=True))
    return x==y
def timed(fn, reps=2):
    times=[]; value=None
    for _ in range(reps):
        start=time.perf_counter();value=fn();times.append(time.perf_counter()-start)
    return value, {'seconds':times,'median_seconds':float(np.median(times))}

def fixture():
    xy=np.array([[0,0],[.04,0],[.04,.01],[.01,.01],[.01,.04],[0,.04]])
    verts=np.r_[np.column_stack([xy,np.full(6,-.005)]),np.column_stack([xy,np.full(6,.005)])]
    top=np.array([[0,1,3],[1,2,3],[0,3,5],[3,4,5]])
    faces=[*(top+6),*top[:,::-1]]
    for i in range(6):
        j=(i+1)%6;faces.extend([[i,j,i+6],[j,j+6,i+6]])
    tri=verts[np.asarray(faces)];tau=1e-7
    points=np.array([[.005,.005,0],[.02,.02,0],[.02,.005,.005],[-tau*.5,.005,0],
        [-tau*2,.005,0],[-tau*.75,-tau*.75,0],[.02,.005,.005+tau],
        [.02,.005,.005+np.nextafter(tau,np.inf)],[.1,.1,.1]])
    tests=[]
    for name,q in [('concave_boundary_inside_outside',points),('all_outside',points[-1:]),('empty',points[:0])]:
        old=original(q,tri,{'object_boundary_tolerance_m':tau});new=accelerated(q,tri,{'object_boundary_tolerance_m':tau})
        assert equal(old,new),name
        tests.append({'fixture':name,'points':len(q),'flags_and_diagnostics_exact':True,'shortcut':accelerated.last_audit})
    return tests

def main():
    started=time.perf_counter();p=read(HERE/'protocol.json');ph=sha(HERE/'protocol.json')
    for path,h in p['source_sha256'].items(): assert sha(path)==h,path
    protected=read(V1/'protected_hashes.json')
    for path,h in protected.items():assert sha(path)==h,path
    COUNTS['testing']=True
    for name in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,name)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Guard failed')
    COUNTS['testing']=False
    numba.set_num_threads(1)
    fixture_report=fixture()
    # Compile both unchanged geometry kernels before timing real input sets.
    dummy=np.array([[[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]]])
    original(np.array([[0.1,0.1,0.1]]),dummy,{'object_boundary_tolerance_m':1e-7})
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory=read(prior/'existing_hug_inventory.json');fk={c['native_sha256']:c for c in read(prior/'static_saved_hug_fk.json')['cases']}
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);grouped=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    normal_p=read(ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json')
    v1=read(V1/'protocol.json')
    import xml.etree.ElementTree as ET
    physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
    records=[]
    for planned in v1['cases']:
        index=planned['source_inventory_index'];case=inventory['modern_cases'][index]
        assert case['native']['sha256']==planned['native_sha256']
        meta,nt,nlab,nnorm,hand,objs,m,d,initial,objadr,opos,oR=a.load_case(case,grouped,fk)
        vt,_,_=o.visual_mesh(meta,m,d,opos,oR);names=case['stored_shadow_command']['joint_names']
        q0=np.array([meta['recovered_original_targets'][n] for n in names]);limits=np.asarray([physical[n] for n in names])
        nominal=d.qpos.copy();adr=np.array([m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');ob=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        rootpos=np.asarray(meta['root_target_world_position_m']);rootquat=np.asarray(meta['root_target_world_quaternion_wxyz']);source=pickle.loads(Path(case['native']['path']).read_bytes())
        T=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);target=(T@np.asarray(source['T_camera_wrist']))[:3,3]
        caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        cp=Path(v1['query_cache_root'])/f'{caseid}_density2mm_queries.npz';fp=Path(v1['original_full_distance_root'])/f'{caseid}_density2mm_full_current.npz'
        assert sha(cp)==v1['query_cache_sha256'][str(cp)] and sha(fp)==v1['full_cache_sha256'][str(fp)]
        with np.load(cp,allow_pickle=False) as f:c={k:f[k].copy() for k in f.files}
        with np.load(fp,allow_pickle=False) as f:full={k:f[k].copy() for k in f.files}
        o.visual_membership=original
        refs,coverage=direction.freeze_direction_sites(c['query_object_m'],c['area_mm2'],c['visual_triangle_outward_normal'],nt,nlab,nnorm,hand,vt,normal_p,c['native_distance_by_part_m'],c['native_inside'],c['native_nearest_part'],full,bins_per_part=64)
        available=[part for part in a.PARTS[:6] if refs[part]['available']]
        allsites=np.concatenate([refs[part]['source_candidate_indices'] for part in available]);bins=np.concatenate([refs[part]['query_indices'] for part in available]);expected=np.concatenate([np.full(len(refs[part]['query_indices']),a.PARTS.index(part),int) for part in available])
        native_cp=np.concatenate([refs[part]['native_closest_object_m'] for part in available]);base_cp=np.concatenate([refs[part]['baseline_closest_object_m'] for part in available])
        membership=[]
        for name,points in [('native_reference_closest',native_cp),('baseline_reference_closest',base_cp)]:
            old,to=timed(lambda:original(points,vt,normal_p),p['timing_repeats']);new,tn=timed(lambda:accelerated(points,vt,normal_p),p['timing_repeats']);assert equal(old,new)
            membership.append({'domain':name,'point_count':len(points),'flags_and_diagnostics_exact':True,'original':to,'accelerated':tn,'speedup':to['median_seconds']/tn['median_seconds'],'shortcut':accelerated.last_audit.copy()})
        o.visual_membership=accelerated
        fastrefs,fastcoverage=direction.freeze_direction_sites(c['query_object_m'],c['area_mm2'],c['visual_triangle_outward_normal'],nt,nlab,nnorm,hand,vt,normal_p,c['native_distance_by_part_m'],c['native_inside'],c['native_nearest_part'],full,bins_per_part=64)
        assert equal(refs,fastrefs) and equal(coverage,fastcoverage)
        poses=[('original',0.)]
        if index in p['perturb_inventory_indices']:poses.append(('fixed_plus',p['perturb_offset_rad']))
        for pose_name,offset in poses:
            q=np.clip(q0+offset*np.where(np.arange(24)%2==0,1.,-1.),limits[:,0],limits[:,1]);d.qpos[:]=nominal;d.qpos[adr]=q;d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat;mujoco.mj_forward(m,d)
            corrected=rootpos.copy() if np.array_equal(q,q0) else rootpos-(d.xpos[palm]-target);d.qpos[rootadr:rootadr+3]=corrected
            if m.nmocap:d.mocap_pos[0]=corrected;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d);assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);assert np.array_equal(d.xpos[ob],opos) and np.array_equal(d.xmat[ob].reshape(3,3),oR)
            gh=refresh.refresh_colliders(hand,m,d,opos,oR);fields=[];reference=None
            for label,threads,fn in [('original_1thread',1,original),('original_4threads',4,original),('accelerated_1thread',1,accelerated),('accelerated_4threads',4,accelerated)]:
                numba.set_num_threads(threads);o.visual_membership=fn
                result,timing=timed(lambda:direction.mapped_direction_fields(c['query_object_m'][bins],c['visual_triangle_outward_normal'][bins],gh,vt,normal_p,expected_part=expected),p['timing_repeats'])
                if reference is None:reference=result
                else:assert equal(reference,result),(caseid,pose_name,label)
                fields.append({'variant':label,'threads':threads,'timing':timing,'all_fields_masks_scores_and_diagnostics_exact':True,'eligible_bin_count':int(result['valid'].sum())})
            numba.set_num_threads(1);o.visual_membership=original
            # All original eligible sites, not just quantile bins, receive exact
            # full-field equality under the accelerated one-thread path.
            all_expected=c['native_nearest_part'][allsites]
            old=direction.mapped_direction_fields(c['query_object_m'][allsites],c['visual_triangle_outward_normal'][allsites],gh,vt,normal_p,expected_part=all_expected)
            o.visual_membership=accelerated
            new=direction.mapped_direction_fields(c['query_object_m'][allsites],c['visual_triangle_outward_normal'][allsites],gh,vt,normal_p,expected_part=all_expected)
            assert equal(old,new),(caseid,pose_name,'all_eligible')
            records.append({'case':caseid,'source_inventory_index':index,'native_sha256':case['native']['sha256'],'pose':pose_name,'offset_rad':offset,'qpos_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'bin_count':len(bins),'all_original_eligible_query_count':len(allsites),'all_original_eligible_fields_exact':True,'reference_sites_and_coverage_exact':True,'membership_benchmarks':membership if pose_name=='original' else [],'normal_field_variants':fields,'time_s':float(d.time),'object_pose_unchanged':True})
            write(HERE/'benchmark.partial.json',{'protocol_sha256':ph,'records':records,'fixtures':fixture_report})
            print(json.dumps({'completed_case_pose':caseid+' '+pose_name,'variant_seconds':{f['variant']:f['timing']['median_seconds'] for f in fields},'all_fields_exact':True}),flush=True)
        for path,h in meta['input_sha256'].items():assert sha(path)==h,path
    o.visual_membership=original;numba.set_num_threads(1)
    for path,h in p['source_sha256'].items():assert sha(path)==h,path
    for path,h in protected.items():assert sha(path)==h,path
    assert COUNTS['actual']==0 and sha(HERE/'protocol.json')==ph
    report={'schema':'hug_exact_visual_membership_acceleration_benchmark_v1','protocol_sha256':ph,'fixtures':fixture_report,'records':records,'runtime_seconds':time.perf_counter()-started,'guards':COUNTS,'source_and_protected_hashes_unchanged':True,'physics_steps':0,'new_inference':False,'new_eval_data':False,'constrained_solver_run':False,'ongoing_V1_modified':False}
    write(HERE/'benchmark.json',report);print(json.dumps({'completed_records':len(records),'all_exact':True,'runtime_seconds':report['runtime_seconds']}))
if __name__=='__main__':main()
