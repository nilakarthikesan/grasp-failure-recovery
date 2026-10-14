"""Independent fixed-grid HUG candidate diagnostics; never integrates time.

Inputs are existing frozen native proposals and candidate joint targets. All
cell/direction parameters remain from the previous pre-frozen rule. A better
objective or an unobserved flip never becomes a strict mapping PASS.
"""
from pathlib import Path
import argparse, gzip, hashlib, importlib.util, json, pickle, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_link_positions,_body_id
from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose,camera_landmarks_to_wrist,reorder_for_mujoco

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ori=load('frozen_orientation',OLD/'orientation/check_orientation.py')
a=ori.a
cells=load('frozen_cells',OLD/'sampling/cell_intervals.py')
COUNTS={'selftest':0,'actual':0};TEST=False
def forbid(*args,**kwargs):
    COUNTS['selftest' if TEST else 'actual']+=1
    raise RuntimeError('Independent candidate mapping validation forbids physics integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbid)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def full_fields(hand,q):
    shape=(len(q),len(a.PARTS));dist=np.full(shape,np.inf);closest=np.full(shape+(3,),np.nan);normal=np.full_like(closest,np.nan);geom=np.full(shape,-1,np.int64);inside=np.zeros(len(q),bool)
    for g in hand:
        signed,cp,n=a.geom_query(g,q);inside|=signed<0;part=a.PARTS.index(g['part']);u=np.abs(signed);upd=u<dist[:,part]
        dist[upd,part]=u[upd];closest[upd,part]=cp[upd];normal[upd,part]=n[upd];geom[upd,part]=g['id']
    assert np.isfinite(dist).all() and np.isfinite(closest).all() and np.isfinite(normal).all()
    return {'current_distance_by_part_m':dist,'current_closest_by_part_m':closest,'current_normal_by_part':normal,'nearest_geom_by_part':geom,'shadow_inside':inside}

def status(proximity,orientation,rule,eps):
    checks=[];normals=[]
    for band in rule['bands_mm']:
        for part in rule['anatomical_parts']:
            x=proximity[str(eps)]['bands'][str(band)]['independent_parts'][part]
            lo=x['lost_native_area']['lower_mm2'];new=x['new_mapped_area']['lower_mm2'];lu=x['lost_native_area']['upper_mm2'];nu=x['new_mapped_area']['upper_mm2']
            state='FAIL' if lo>0 or new>0 else ('PASS' if lu==0 and nu==0 else 'UNRESOLVED')
            checks.append({'part':part,'band_mm':band,'status':state,'lost_area_lower_mm2':lo,'added_area_lower_mm2':new,'lost_area_upper_mm2':lu,'added_area_upper_mm2':nu})
        s=orientation['bands'][str(band)]['common_same_label']['pointwise_sign_check'];count=sum(x['query_count'] for x in s['flip_denominators'].values())
        normals.append({'band_mm':band,'pointwise_status':'FAIL' if count else 'UNRESOLVED','reliable_sampled_sign_reversal_count':count,'flip_strata':s['flip_denominators'],'continuous_orientation_coverage_status':'UNRESOLVED'})
    prox='FAIL' if any(x['status']=='FAIL' for x in checks) else ('PASS' if all(x['status']=='PASS' for x in checks) else 'UNRESOLVED')
    normal='FAIL' if any(x['pointwise_status']=='FAIL' for x in normals) else 'UNRESOLVED'
    return {'proximity_status_conditional':prox,'pointwise_orientation_status':normal,'strict_relationship_status_conditional':'FAIL' if prox=='FAIL' or normal=='FAIL' else 'UNRESOLVED','proximity_checks':checks,'orientation_checks':normals,'full_physical_contact_equivalence':'UNRESOLVED','failure_label_preservation':'NOT_TESTED','independent_unseen_object_validation':'NOT_TESTED'}

def main():
    global TEST
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent);ap.add_argument('--case-index',type=int,nargs='*');ap.add_argument('--skip-orientation',action='store_true',help='Separate sampling-only pilot; strict orientation remains NOT_TESTED');args=ap.parse_args();args.out.mkdir(exist_ok=True);tic=time.perf_counter()
    protocol=read(args.out/'protocol.json');ph=sha(args.out/'protocol.json');assert sha(__file__)==protocol['script_sha256']
    for file,h in protocol['fixed_files_sha256'].items():assert sha(file)==h,file
    cp=sha(args.candidate);candidate=read(args.candidate);assert candidate['schema'] in ('hug_surface_refinement_candidate_v1','hug_surface_refinement_candidate_v2');v2=candidate['schema'].endswith('_v2');targets={c['native_sha256']:c for c in candidate['cases']};assert len(targets)==len(candidate['cases'])
    p=read(OLD/'orientation/protocol.json');sp=read(OLD/'sampling/protocol.json');rule=read(OLD/'acceptance/acceptance_rule.json')
    current=read(OLD/'sampling/sampling_uncertainty.json');oldori=json.loads(gzip.decompress((OLD/'orientation/orientation_report_full.json.gz').read_bytes()));cm={x['native_sha256']:x for x in current['cases']};om={x['native_sha256']:x for x in oldori['cases']}
    acceptance=read(OLD/'acceptance/acceptance_application.json');am={(x['native_sha256'],str(x['density_mm'])):x for x in acceptance['records']}
    previous=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inventory=read(previous/'existing_hug_inventory.json');fk={c['native_sha256']:c for c in read(previous/'static_saved_hug_fk.json')['cases']}
    protected=read(OLD/'protected_input_sha256.json')
    for file,h in protected.items():assert sha(file)==h,file
    xml=ET.parse(a.URDF).getroot();ranges={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in xml.findall('joint') if j.find('limit') is not None}
    TEST=True
    for name in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,name)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Guard failed')
    TEST=False
    write(args.out/'sanity_report.json',ori.sanity(p))
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    w=np.asarray(asset['weights']);grouped=np.column_stack([w[:,ids].sum(1) for ids in a.GROUPS.values()]);records=[];validator=ShadowPoseValidator(a.URDF.parent.parent)
    for selected in p['cases']:
        index=selected['source_inventory_index']
        if args.case_index is not None and index not in args.case_index:continue
        case=inventory['modern_cases'][index];h=case['native']['sha256'];assert h==selected['native_sha256'];target=targets[h];assert target['source_inventory_index']==index
        meta,tri,labels,nn,hand,objs,m,d,initial,objadr,pos,R=a.load_case(case,grouped,fk);palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');original_palm_pos=d.xpos[palm].copy();original_palm_R=d.xmat[palm].reshape(3,3).copy();original_q=d.qpos.copy();rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);bound=[]
        assert target['joint_names']==meta['stored_command']['joint_names'];qv=np.asarray(target['qpos_rad'],float);assert qv.shape==(24,) and np.isfinite(qv).all()
        for name,value in zip(target['joint_names'],qv,strict=True):
            low,high=ranges[name];assert low-1e-9<=value<=high+1e-9,(name,value,low,high)
            j=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);sl,sh=m.jnt_range[j];difference=max(abs(sl-low),abs(sh-high));assert sl-difference-1e-9<=value<=sh+difference+1e-9
            qa=int(m.jnt_qposadr[j]);delta=float(value-d.qpos[qa]);assert abs(delta)<=protocol['maximum_joint_delta_rad']+1e-9,(name,delta);d.qpos[qa]=value
            if not v2 and name in ('rh_WRJ1','rh_WRJ2'):assert delta==0,(name,delta)
            bound.append({'joint':name,'delta_from_original_rad':delta,'source_URDF_range_rad':[low,high],'scene_range_rad':[float(sl),float(sh)],'scene_representation_difference_rad':float(difference),'strict_scene_overflow_rad':float(max(sl-value,value-sh,0))})
        rootpos=np.asarray(target['root_target_position_m'],float);rootquat=np.asarray(target['root_target_quaternion_wxyz'],float);assert rootpos.shape==(3,) and rootquat.shape==(4,) and np.isfinite(rootpos).all() and np.isfinite(rootquat).all();assert abs(np.linalg.norm(rootquat)-1)<1e-8
        grasp=load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);T_wrist=T_wc@grasp.T_camera_wrist;Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        local_landmarks=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T;pose=RetargetedHandPose(tuple(target['joint_names']),qv,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose)
        fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();fixed_palm_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf);desired_R=Rop@fixed_palm_R
        if not v2:assert np.array_equal(rootpos,original_q[rootadr:rootadr+3]) and np.array_equal(rootquat,original_q[rootadr+3:rootadr+7])
        d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
        if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d)
        palm_error=float(np.linalg.norm(d.xpos[palm]-T_wrist[:3,3]));rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desired_R)));expected_links=(fixed_links-fixed_palm)@Rop.T+T_wrist[:3,3];scene_links=_link_positions(m,d,validator.source_urdf);link_error=float(np.linalg.norm(scene_links-expected_links,axis=1).max())
        assert palm_error<1e-7 and rotation_error<1e-6 and link_error<1.1e-7,(palm_error,rotation_error,link_error)
        if not v2:assert np.array_equal(d.xpos[palm],original_palm_pos) and np.array_equal(d.xmat[palm].reshape(3,3),original_palm_R)
        if 'target_palm_world_position_m' in target:assert np.linalg.norm(np.asarray(target['target_palm_world_position_m'])-T_wrist[:3,3])<1e-12
        if 'target_palm_world_rotation' in target:assert np.max(np.abs(np.asarray(target['target_palm_world_rotation'])-desired_R))<1e-12
        if 'T_world_operator_rotation' in target:assert np.max(np.abs(np.asarray(target['T_world_operator_rotation'])-Rop))<1e-12
        assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);nominal=d.qpos.copy();newhand=[]
        for g in hand:gg=a.geom_rep(m,d,g['id'],R,pos);gg['part']=g['part'];newhand.append(gg)
        vt,on,_=ori.visual_mesh(meta,m,d,pos,R);caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        rec={'source_inventory_index':index,'native_sha256':h,'case_id':caseid,'candidate_joint_range_checks':bound,'root_policy':{'native_wrist_to_palm_position_error_m':palm_error,'declared_world_operator_times_base_palm_rotation_maxabs':rotation_error,'independent_fixed_URDF_vs_scene_max_link_error_m':link_error,'V1_fixed_root_WRJ_palm_bitwise_unchanged':not v2,'V2_q_dependent_carrier_preserving_native_palm_position':v2,'not_an_anatomy_equivalence_certificate':True},'input_sha256':meta['input_sha256'],'resolutions':{}}
        for density in selected['densities_mm']:
            cache=Path(p['original_query_cache_root'])/f'{caseid}_density{density}mm_queries.npz';assert sha(cache)==p['query_cache_sha256'][str(cache)]
            with np.load(cache,allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
            child,q,area,radius,diameter,faceids,depths=cells.triangle_cells(vt,density/1000);assert np.array_equal(q,old['query_object_m']);assert np.array_equal(area,old['area_mm2']);objnorm=np.repeat(on,4**np.asarray(depths),axis=0);assert np.allclose(objnorm,old['visual_triangle_outward_normal'],rtol=0,atol=1e-14)
            print(json.dumps({'starting':caseid,'density_mm':density,'query_count':len(q)}),flush=True)
            tick=time.perf_counter();full=full_fields(newhand,q);nd=old['native_distance_by_part_m'];sd=full['current_distance_by_part_m'];proximity={}
            for eps in sp['numerical_allowances_m']:
                result={'conditional_on_uniform_numeric_allowance_m':eps,'bands':{}}
                for band in rule['bands_mm']:
                    result['bands'][str(band)]={'whole_hand_unsigned_proximity':cells.pair_intervals(nd.min(1),sd.min(1),radius,area,band/1000,eps),'independent_parts':{part:cells.pair_intervals(nd[:,j],sd[:,j],radius,area,band/1000,eps) for j,part in enumerate(a.PARTS)}}
                proximity[str(eps)]=result
            orientation=None;check=None
            if not args.skip_orientation:
                orientation=ori.inspect_orientation(q,area,objnorm,nd,old['native_inside'],old['native_nearest_part'],tri,labels,nn,newhand,vt,full,p)
                check=status(proximity,orientation,rule,sp['primary_allowance_m'])
            rec['resolutions'][str(density)]={'query_cache_sha256':sha(cache),'query_count':len(q),'conditional_sampling':proximity,'orientation':orientation,'frozen_acceptance_application':check,'baseline_acceptance':am[h,str(density)],'baseline_conditional_sampling':cm[h]['resolutions'][str(density)]['numerical_allowances'],'baseline_orientation':om[h]['resolutions'][str(density)],'runtime_seconds':time.perf_counter()-tick}
            assert np.array_equal(d.qpos,nominal) and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            write(args.out/f'{caseid}_validation.json',rec)
            print(json.dumps({'completed':caseid,'density_mm':density,'seconds':time.perf_counter()-tick,'strict_status':check['strict_relationship_status_conditional'] if check else 'ORIENTATION_NOT_TESTED'}),flush=True)
        for file,expected in meta['input_sha256'].items():assert sha(file)==expected,file
        records.append(rec)
    for file,h in protected.items():assert sha(file)==h,file
    for file,h in protocol['fixed_files_sha256'].items():assert sha(file)==h,file
    assert sha(args.candidate)==cp and sha(args.out/'protocol.json')==ph and sha(__file__)==protocol['script_sha256'];assert COUNTS['actual']==0
    report={'schema':'hug_candidate_independent_static_validation_v2','candidate_sha256':cp,'candidate_path':str(args.candidate),'protocol_sha256':ph,'script_sha256':sha(__file__),'all_source_geometry_and_protected_hashes_unchanged':True,'physics_steps':0,'guards':COUNTS,'new_eval_data':False,'new_inference':False,'training':False,'development_cases_only':True,'skip_orientation_pilot':args.skip_orientation,'runtime_seconds':time.perf_counter()-tic,'cases':records,'limitations':protocol['limitations']}
    raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();name='candidate_validation'+('_'+','.join(map(str,args.case_index)) if args.case_index else '')+('_sampling_only' if args.skip_orientation else '')+'.json.gz';(args.out/name).write_bytes(gzip.compress(raw,mtime=0));write(args.out/(name+'.binding.json'),{'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':sha(args.out/name),'case_count':len(records),'protocol_sha256':ph,'candidate_sha256':cp})
    print(json.dumps({'report':str(args.out/name),'seconds':time.perf_counter()-tic,'cases':len(records)}),flush=True)
if __name__=='__main__':main()
