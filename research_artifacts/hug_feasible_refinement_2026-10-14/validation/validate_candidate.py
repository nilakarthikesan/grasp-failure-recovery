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

def verify_recorded_selection(target, physical_ranges, protocol):
    audit=target['solver_audit']; original=np.asarray(target['original_qpos_rad'],float)
    bounds=np.asarray([physical_ranges[n] for n in target['joint_names']],float)
    low=np.maximum(bounds[:,0],original-protocol['maximum_joint_delta_rad'])
    high=np.minimum(bounds[:,1],original+protocol['maximum_joint_delta_rad'])
    assert audit['schema']=='hug_feasible_only_static_pattern_refinement_v1'
    assert audit['config']['maximum_pattern_sweeps']==0
    assert audit['config']['maximum_full_geometry_evaluations']==4
    assert audit['feasibility_error_allowance_m']==0
    assert audit['final_callbacks_fresh_outside_cache'] and not audit['mapping_fidelity_acceptance']
    assert np.array_equal(original,np.asarray(audit['original_qpos_rad']))
    assert np.array_equal(bounds,np.asarray(audit['physical_bounds_rad']))
    assert np.array_equal(np.column_stack([low,high]),np.asarray(audit['bounded_correction_domain_rad']))
    original_vectors=np.asarray(audit['baseline']['tip_error_vectors_m'],float)
    assert original_vectors.shape==(5,3) and np.isfinite(original_vectors).all()
    original_norms=np.sqrt(np.einsum('ij,ij->i',original_vectors,original_vectors))
    baseline_cost=float(audit['baseline']['objective'])
    assert np.isfinite(baseline_cost) and baseline_cost>=0
    mean_limit=protocol['existing_engineering_mean_tip_guard_m']
    improvement=protocol['minimum_objective_improvement']

    def checked_q(q):
        v=np.asarray(q,float)
        assert v.shape==(24,) and np.isfinite(v).all()
        assert np.all(v>=low) and np.all(v<=high)
        return v

    def check_assessment(row):
        errors=np.asarray(row['tip_error_vectors_m'],float)
        assert errors.shape==(5,3) and np.isfinite(errors).all()
        norm=np.sqrt(np.einsum('ij,ij->i',errors,errors))
        assert np.array_equal(norm,np.asarray(row['tip_error_norm_m']))
        assert np.array_equal(norm-original_norms,np.asarray(row['tip_error_increase_m']))
        caps=norm<=original_norms; mean=float(norm.mean()); feasible=bool(caps.all() and mean<=mean_limit)
        assert caps.tolist()==row['individual_tip_nonworsening_pass']
        assert mean==row['mean_fingertip_error_m']
        assert bool(mean<=mean_limit)==row['strict_existing_mean_guard_pass']
        assert feasible==row['exact_computed_engineering_feasible']
        return feasible

    assert check_assessment(audit['baseline'])
    seeds=np.asarray(audit['predeclared_seed_qpos_rad'],float)
    assert seeds.shape==(1,24)
    seed=checked_q(seeds[0]);assert np.array_equal(seed,np.asarray(target['predeclared_seed_qpos_rad']))
    assert np.array_equal(seed,np.asarray(target['raw_candidate_qpos_rad']))
    assert target['seed_source_cohort_sha256']==audit['predeclared_seed_source_sha256']
    proposals=audit['proposals'];accepted=audit['accepted_candidates']
    assert len(proposals)<=1 and len(accepted)<=1
    chosen=None
    for number,row in enumerate(proposals,1):
        assert row['sequence']==number and row['sweep']==0 and row['seed_index']==0
        assert row['direction']=='predeclared_seed' and row['radius_rad'] is None
        assert np.array_equal(np.asarray(row['anchor_qpos_rad']),original)
        assert np.array_equal(checked_q(row['proposal_qpos_rad']),seed)
        if 'proposal_assessment' in row:check_assessment(row['proposal_assessment'])
        if 'restored_assessment' in row:check_assessment(row['restored_assessment'])
        for backtrack in row.get('restoration_backtracking',[]):
            checked_q(backtrack['qpos_rad']);check_assessment(backtrack['assessment'])
        if 'full_assessment' not in row:
            assert not row['accepted'];continue
        candidate=checked_q(row['candidate_qpos_rad']);assessment=row['full_assessment']
        assert check_assessment(assessment)
        cost=float(assessment['objective']);assert np.isfinite(cost) and cost>=0
        improved=bool(cost<baseline_cost-improvement)
        assert improved==row['accepted']
        if improved:chosen=(candidate,cost,row['sequence'])
    if chosen is None:
        assert accepted==[]
    else:
        assert len(accepted)==1
        assert np.array_equal(np.asarray(accepted[0]['qpos_rad']),chosen[0])
        assert accepted[0]['objective']==chosen[1] and accepted[0]['proposal_sequence']==chosen[2]
    refused=audit['callback_or_geometry_refusal'] is not None
    if refused or chosen is None:
        expected=original; source='original_fallback'; chosen_sequence=None; cost=baseline_cost
    else:
        expected=chosen[0];source='predeclared_seed_after_FK_restore';cost=chosen[1];chosen_sequence=chosen[2]
    returned=checked_q(target['qpos_rad'])
    assert np.array_equal(returned,expected) and np.array_equal(returned,np.asarray(audit['returned_qpos_rad']))
    assert audit['selected_candidate_source']==source and audit['selected_major_iteration'] is None
    assert bool(source!='original_fallback')==audit['surface_candidate_returned']
    assert bool(source=='original_fallback')==audit['exact_original_fallback']
    final=audit['final_assessment'];assert final==audit['final'] and check_assessment(final)
    assert abs(final['objective']-cost)<1e-10
    assert abs(final['objective']-target['selected_fresh_diagnostics']['objective'])<1e-10
    events=audit['events'];counts=audit['counters']
    assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
    full_events=[e for e in events if e['kind']=='full_callback']
    fk_events=[e for e in events if e['kind']=='FK_callback']
    assert len(full_events)==counts['full']<=audit['config']['maximum_full_geometry_evaluations']
    assert len(fk_events)==counts['FK']<=audit['config']['maximum_FK_evaluations']
    assert full_events[-1]['fresh'] and full_events[-1]['terminal']
    assert fk_events[-1]['fresh'] and fk_events[-1]['terminal']
    assert np.array_equal(np.asarray(full_events[-1]['qpos_rad']),returned)
    assert np.array_equal(np.asarray(fk_events[-1]['qpos_rad']),returned)
    assert target['full_callback_count_including_fresh_final']==counts['full']
    assert audit['full_FK_agreement_maxabs_m']<=audit['config']['full_FK_agreement_allowance_m']
    return {'recorded_seed_restoration_selection_verified':True,'selected_candidate_source':source,
        'selected_proposal_sequence':chosen_sequence,'recorded_seed_proposal_count':len(proposals),
        'historically_accepted_seed_count':len(accepted),'fresh_final_refusal_forces_original':refused,
        'exact_computed_all_five_caps_verified_in_recorded_assessments':True,
        'norm_arithmetic':'sqrt(einsum(ij,ij->i)) on independently recovered vectors',
        'feasibility_error_allowance_m':0.,'full_and_FK_terminal_events_fresh_verified':True,
        'objective_values_not_independently_recomputed_for_all_trace_poses':True,
        'not_scientific_mapping_acceptance':True}

def main():
    global TEST
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent);ap.add_argument('--case-index',type=int,nargs='*');ap.add_argument('--skip-orientation',action='store_true',help='Separate sampling-only pilot; strict orientation remains NOT_TESTED');ap.add_argument('--preflight-only',action='store_true',help='Check every original/final source pose first, without costly surface metrics');args=ap.parse_args();args.out.mkdir(exist_ok=True);tic=time.perf_counter()
    protocol=read(args.out/'protocol.json');ph=sha(args.out/'protocol.json');assert sha(__file__)==protocol['script_sha256']
    for file,h in protocol['fixed_files_sha256'].items():assert sha(file)==h,file
    assert protocol['execution_authorized_by_root_review'], 'Frozen root review required before audit execution'
    cp=sha(args.candidate);candidate=read(args.candidate);assert candidate['schema']==protocol['candidate_schema'];v3=False;v2=True
    targets={c['native_sha256']:c for c in candidate['cases']};assert len(targets)==len(candidate['cases'])==6
    assert candidate['protocol_sha256']==protocol['candidate_protocol_expected_sha256']
    candidate_protocol_file=args.candidate.parent/'protocol.json';assert sha(candidate_protocol_file)==candidate['protocol_sha256']
    candidate_protocol=read(candidate_protocol_file)
    assert candidate['method_version']==2 and candidate['method_stage']=='predeclared_seed_only'
    assert candidate_protocol['method_version']==2 and candidate_protocol['method_stage']=='predeclared_seed_only'
    assert candidate_protocol['solver_configuration']['maximum_pattern_sweeps']==0
    assert sha(args.candidate.parent/'run_seed_stage.py')==candidate['script_sha256']
    assert sha(args.candidate.parent/'feasible_surface_refinement.py')==candidate['solver_implementation_sha256']
    assert sha(Path(candidate_protocol['direction_helper_path']))==candidate['direction_implementation_sha256']
    for file,h in candidate_protocol['implementation_sha256'].items():assert sha(file)==h,file
    for file,h in candidate_protocol['source_sha256'].items():assert sha(file)==h,file
    if 'merge_script_sha256' in candidate:
        assert sha(protocol['candidate_merge_script_path'])==candidate['merge_script_sha256']
    cohorts=candidate['cohort_provenance'];assert len(cohorts)==1
    cohort=cohorts[0];assert cohort['cohort_case_indices']==protocol['case_indices']
    assert sha(cohort['path'])==cohort['sha256'];cohort_report=read(cohort['path'])
    assert cohort_report['schema']=='hug_feasible_seed_stage_surface_candidate_cohort_v2'
    assert cohort_report['protocol_sha256']==candidate['protocol_sha256']
    assert cohort_report['method_version']==2 and cohort_report['method_stage']=='predeclared_seed_only'
    assert cohort_report['cohort_case_indices']==protocol['case_indices']
    assert cohort_report['guards']['actual']==0 and cohort_report['guards']['selftests']==3
    assert cohort_report['physics_steps']==0 and cohort_report['protected_hashes_unchanged']
    assert len(cohort_report['cases'])==6 and candidate['guards']['actual']==0 and candidate['guards']['selftests']==3
    assert {x['native_sha256'] for x in cohort_report['cases']}==set(targets)
    assert all(x==targets[x['native_sha256']] for x in cohort_report['cases'])
    for target in targets.values():
        seed_binding=candidate_protocol['seed_bindings'][str(target['source_inventory_index'])]
        assert sha(seed_binding['old_cohort_path'])==seed_binding['old_cohort_sha256']
        old_case=read(seed_binding['old_cohort_path'])['cases'][0]
        assert old_case['native_sha256']==target['native_sha256']
        assert old_case['joint_names']==target['joint_names']
        assert np.array_equal(np.asarray(old_case['original_qpos_rad']),np.asarray(target['original_qpos_rad']))
        assert np.array_equal(np.asarray(old_case['raw_candidate_qpos_rad']),np.asarray(target['predeclared_seed_qpos_rad']))
        assert target['seed_source_cohort_sha256']==seed_binding['old_cohort_sha256']
    if args.preflight_only:
        assert args.case_index is None, 'All six preflight cases must complete together'
    else:
        preflight_path=args.out/'candidate_validation_preflight_only.json.gz'
        preflight_binding=read(args.out/'candidate_validation_preflight_only.json.gz.binding.json')
        preflight_raw=gzip.decompress(preflight_path.read_bytes());preflight=json.loads(preflight_raw)
        assert sha(preflight_path)==preflight_binding['gzip_sha256']
        assert hashlib.sha256(preflight_raw).hexdigest()==preflight_binding['uncompressed_sha256']
        assert preflight['protocol_sha256']==ph and preflight['candidate_sha256']==cp
        assert preflight['preflight_only'] and not preflight['skip_orientation_pilot']
        assert preflight['physics_steps']==0 and preflight['guards']['actual']==0
        assert preflight['all_source_geometry_and_protected_hashes_unchanged']
        assert sorted(x['source_inventory_index'] for x in preflight['cases'])==protocol['case_indices']
        assert all(x['engineering_guard_independent_audit']['source_preflight_status']=='PASS' for x in preflight['cases'])
    p=read(OLD/'orientation/protocol.json');sp=read(OLD/'sampling/protocol.json');rule=read(OLD/'acceptance/acceptance_rule.json')
    current=read(OLD/'sampling/sampling_uncertainty.json');oldori=json.loads(gzip.decompress((OLD/'orientation/orientation_report_full.json.gz').read_bytes()));cm={x['native_sha256']:x for x in current['cases']};om={x['native_sha256']:x for x in oldori['cases']}
    acceptance=read(OLD/'acceptance/acceptance_application.json');am={(x['native_sha256'],str(x['density_mm'])):x for x in acceptance['records']}
    previous=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inventory=read(previous/'existing_hug_inventory.json');fk={c['native_sha256']:c for c in read(previous/'static_saved_hug_fk.json')['cases']}
    assert sha(args.out/'protected_hashes.json')==protocol['protected_snapshot_sha256'];protected=read(args.out/'protected_hashes.json');assert len(protected)==protocol['protected_file_count']
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
        meta,tri,labels,nn,hand,objs,m,d,initial,objadr,pos,R=a.load_case(case,grouped,fk);palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');original_palm_pos=d.xpos[palm].copy();original_palm_R=d.xmat[palm].reshape(3,3).copy();original_q=d.qpos.copy();original_scene_links=_link_positions(m,d,a.URDF).copy();rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);bound=[]
        assert target['joint_names']==meta['stored_command']['joint_names'];qv=np.asarray(target['qpos_rad'],float);assert qv.shape==(24,) and np.isfinite(qv).all()
        guard_audit=None;assert np.array_equal(np.asarray(target['original_qpos_rad']),np.asarray([meta['recovered_original_targets'][n] for n in target['joint_names']]));selection_audit=verify_recorded_selection(target,ranges,protocol)
        if v3:
            original=np.asarray([meta['recovered_original_targets'][n] for n in target['joint_names']]);assert np.array_equal(original,np.asarray(target['original_qpos_rad']));rawq=np.asarray(target['raw_v2_qpos_rad']);selection=target['engineering_guard'];alpha=selection['selected_alpha'];assert alpha in protocol['engineering_guard_alphas'];expected=original if alpha==0 else original+alpha*(rawq-original);assert np.array_equal(qv,expected)
            attempted=selection['attempted_alphas'];assert [x['alpha'] for x in attempted]==protocol['engineering_guard_alphas'][:len(attempted)];baseline=selection['baseline'];bcost=baseline['objective'];limit=protocol['existing_engineering_mean_tip_guard_m'];min_improvement=protocol['engineering_minimum_objective_improvement'];accepted=[]
            for attempt in attempted:
                aa=attempt['alpha'];ss=attempt['assessment'];passed=ss['mean_fingertip_error_m']<=limit;improved=ss['objective']<bcost-min_improvement
                if aa==0:improved=False
                assert passed==attempt['passed_existing_engineering_guard'] and improved==attempt['objective_improved'];assert bool(passed and improved)==attempt['accepted'];accepted.append(attempt['accepted'])
            assert not any(accepted[:-1]);assert (alpha==0 and not any(accepted)) or (alpha>0 and accepted[-1] and attempted[-1]['alpha']==alpha)
            guard_audit={'joint_interpolation_exact':True,'declared_alpha_order_exact':True,'first_largest_acceptable_recorded_interpolant_verified':True,'objective_decision_uses_same_recorded_global_baseline':True,'global_objective_implementation_independently_reviewed':True,'full_objective_values_not_independently_recomputed_here':True,'selected_alpha':alpha,'existing_engineering_guard_m':limit,'does_not_mean_mapping_acceptance':True}
        for name,value in zip(target['joint_names'],qv,strict=True):
            low,high=ranges[name];assert low-1e-9<=value<=high+1e-9,(name,value,low,high)
            j=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);sl,sh=m.jnt_range[j];difference=max(abs(sl-low),abs(sh-high));assert sl-difference-1e-9<=value<=sh+difference+1e-9
            qa=int(m.jnt_qposadr[j]);delta=float(value-d.qpos[qa]);assert abs(delta)<=protocol['maximum_joint_delta_rad']+1e-9,(name,delta);d.qpos[qa]=value
            if not v2 and name.endswith(('WRJ1','WRJ2')):assert delta==0,(name,delta)
            bound.append({'joint':name,'delta_from_original_rad':delta,'source_URDF_range_rad':[low,high],'scene_range_rad':[float(sl),float(sh)],'scene_representation_difference_rad':float(difference),'strict_scene_overflow_rad':float(max(sl-value,value-sh,0))})
        rootpos=np.asarray(target['root_target_position_m'],float);rootquat=np.asarray(target['root_target_quaternion_wxyz'],float);assert rootpos.shape==(3,) and rootquat.shape==(4,) and np.isfinite(rootpos).all() and np.isfinite(rootquat).all();assert abs(np.linalg.norm(rootquat)-1)<1e-8
        grasp=load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);T_wrist=T_wc@grasp.T_camera_wrist;Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        local_landmarks=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
        original_pose=RetargetedHandPose(tuple(target['joint_names']),np.asarray(target['original_qpos_rad']),local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(original_pose)
        original_fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();original_fixed_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();original_fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf)
        original_expected_links=(original_fixed_links-original_fixed_palm)@Rop.T+T_wrist[:3,3]
        original_source_scene_link_error=float(np.linalg.norm(original_scene_links-original_expected_links,axis=1).max())
        original_source_palm_rotation_error=float(np.max(np.abs(original_palm_R-Rop@original_fixed_R)))
        original_source_palm_position_error=float(np.linalg.norm(original_palm_pos-T_wrist[:3,3]))
        assert original_source_scene_link_error<1.1e-7 and original_source_palm_rotation_error<1e-6 and original_source_palm_position_error<1e-7
        pose=RetargetedHandPose(tuple(target['joint_names']),qv,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose)
        fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();fixed_palm_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf);desired_R=Rop@fixed_palm_R
        if not v2:assert np.array_equal(rootpos,original_q[rootadr:rootadr+3]) and np.array_equal(rootquat,original_q[rootadr+3:rootadr+7])
        d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
        if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d)
        palm_error=float(np.linalg.norm(d.xpos[palm]-T_wrist[:3,3]));rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desired_R)));expected_links=(fixed_links-fixed_palm)@Rop.T+T_wrist[:3,3];scene_links=_link_positions(m,d,validator.source_urdf);link_error=float(np.linalg.norm(scene_links-expected_links,axis=1).max())
        assert palm_error<1e-7 and rotation_error<1e-6 and link_error<1.1e-7,(palm_error,rotation_error,link_error)
        native_world=grasp.landmarks_3d@T_wc[:3,:3].T+T_wc[:3,3];native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])];original_errors=np.linalg.norm(original_scene_links-native_tasks,axis=1);candidate_errors=np.linalg.norm(scene_links-native_tasks,axis=1)
        assert np.max(np.abs(original_errors-np.asarray(target['baseline_fresh_diagnostics']['task_link_error_norm_m'])))<1e-12;assert np.max(np.abs(candidate_errors-np.asarray(target['selected_fresh_diagnostics']['task_link_error_norm_m'])))<1e-12
        if 'original_task_link_error_m' in target:assert np.array_equal(np.asarray(target['original_task_link_error_m']),np.asarray(target['baseline_fresh_diagnostics']['task_link_error_norm_m']))
        if 'candidate_task_link_error_m' in target:assert np.array_equal(np.asarray(target['candidate_task_link_error_m']),np.asarray(target['selected_fresh_diagnostics']['task_link_error_norm_m']))
        original_tip_vectors=original_scene_links[:5]-native_tasks[:5];candidate_tip_vectors=scene_links[:5]-native_tasks[:5]
        original_cap_norms=np.sqrt(np.einsum('ij,ij->i',original_tip_vectors,original_tip_vectors))
        candidate_cap_norms=np.sqrt(np.einsum('ij,ij->i',candidate_tip_vectors,candidate_tip_vectors))
        assert protocol['individual_fingertip_arithmetic_allowance_m']==0
        assert np.max(np.abs(original_cap_norms-np.asarray(target['authoritative_original_tip_caps_sqrt_einsum_m'])))<1e-12
        assert np.max(np.abs(candidate_cap_norms-np.asarray(target['selected_tip_norms_sqrt_einsum_m'])))<1e-12
        actual_mean=float(candidate_cap_norms.mean());assert np.all(candidate_cap_norms<=original_cap_norms),{'errors':candidate_cap_norms.tolist(),'original_caps':original_cap_norms.tolist()};assert actual_mean<=protocol['existing_engineering_mean_tip_guard_m']
        guard_audit={'native_landmark_errors_independently_recomputed':True,'original_task_link_errors_m':original_errors.tolist(),'candidate_task_link_errors_m':candidate_errors.tolist(),
            'original_tip_caps_sqrt_einsum_m':original_cap_norms.tolist(),'candidate_tip_norms_sqrt_einsum_m':candidate_cap_norms.tolist(),
            'norm_arithmetic':'sqrt(einsum(ij,ij->i)); np.linalg.norm is a separate diagnostic',
            'individual_fingertip_arithmetic_allowance_m':0.,'each_final_fingertip_nonworsening_exact_computed':True,
            'fingertip_error_change_m':(candidate_cap_norms-original_cap_norms).tolist(),'actual_mean_fingertip_error_m':actual_mean,
            'strict_actual_mean_guard_m':protocol['existing_engineering_mean_tip_guard_m'],'actual_mean_guard_pass':True,
            'original_source_URDF_vs_scene_max_link_error_m':original_source_scene_link_error,
            'original_source_URDF_vs_scene_palm_rotation_maxabs':original_source_palm_rotation_error,
            'original_native_wrist_to_palm_position_error_m':original_source_palm_position_error,
            'source_preflight_status':'PASS','selected_candidate_source':target['solver_audit']['selected_candidate_source'],
            'recorded_solver_selection':selection_audit,'fallback_original_target_exact':bool(np.array_equal(qv,np.asarray(target['original_qpos_rad']))),
            'does_not_mean_scientific_mapping_acceptance':True}
        if v3:
            native_world=grasp.landmarks_3d@T_wc[:3,:3].T+T_wc[:3,3];native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])];errors=np.linalg.norm(scene_links-native_tasks,axis=1);computed_mean=float(errors[:5].mean());final=target['final_assessment'];assert np.max(np.abs(errors-np.asarray(final['task_link_error_m'])))<1e-12;assert abs(computed_mean-final['mean_fingertip_error_m'])<1e-12;assert computed_mean<=protocol['existing_engineering_mean_tip_guard_m'];guard_audit.update({'native_landmark_error_independently_recomputed':True,'task_link_errors_m':errors.tolist(),'mean_fingertip_error_m':computed_mean,'existing_engineering_guard_status':'PASS'})
        if not v2:assert np.array_equal(d.xpos[palm],original_palm_pos) and np.array_equal(d.xmat[palm].reshape(3,3),original_palm_R)
        if 'target_palm_world_position_m' in target:assert np.linalg.norm(np.asarray(target['target_palm_world_position_m'])-T_wrist[:3,3])<1e-12
        if 'target_palm_world_rotation' in target:assert np.max(np.abs(np.asarray(target['target_palm_world_rotation'])-desired_R))<1e-12
        if 'T_world_operator_rotation' in target:assert np.max(np.abs(np.asarray(target['T_world_operator_rotation'])-Rop))<1e-12
        assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);nominal=d.qpos.copy();newhand=[]
        for g in hand:gg=a.geom_rep(m,d,g['id'],R,pos);gg['part']=g['part'];newhand.append(gg)
        vt,on,_=ori.visual_mesh(meta,m,d,pos,R);caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        rec={'source_inventory_index':index,'native_sha256':h,'case_id':caseid,'candidate_joint_range_checks':bound,'engineering_guard_independent_audit':guard_audit,'root_policy':{'native_wrist_to_palm_position_error_m':palm_error,'declared_world_operator_times_base_palm_rotation_maxabs':rotation_error,'independent_fixed_URDF_vs_scene_max_link_error_m':link_error,'V1_fixed_root_WRJ_palm_bitwise_unchanged':not v2,'V2_q_dependent_carrier_preserving_native_palm_position':v2,'not_an_anatomy_equivalence_certificate':True},'input_sha256':meta['input_sha256'],'resolutions':{}}
        if args.preflight_only:
            records.append(rec);assert d.time==0 and np.array_equal(d.qpos,nominal) and np.array_equal(d.qpos[objadr:objadr+7],initial);print(json.dumps({'preflight':caseid,'passed':True,'actual_mean_tip_mm':actual_mean*1000}),flush=True);continue
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
    report={'schema':'hug_feasible_seed_candidate_independent_static_validation_v2','candidate_sha256':cp,'candidate_path':str(args.candidate),'protocol_sha256':ph,'script_sha256':sha(__file__),'all_source_geometry_and_protected_hashes_unchanged':True,'protected_source_count':len(protected),'physics_steps':0,'guards':COUNTS,'new_eval_data':False,'new_inference':False,'training':False,'development_cases_only':True,'preflight_only':args.preflight_only,'skip_orientation_pilot':args.skip_orientation,'runtime_seconds':time.perf_counter()-tic,'cases':records,'limitations':protocol['limitations']}
    raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();name='candidate_validation'+('_'+','.join(map(str,args.case_index)) if args.case_index else '')+('_sampling_only' if args.skip_orientation else '')+('_preflight_only' if args.preflight_only else '')+'.json.gz';(args.out/name).write_bytes(gzip.compress(raw,mtime=0));write(args.out/(name+'.binding.json'),{'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':sha(args.out/name),'case_count':len(records),'protocol_sha256':ph,'candidate_sha256':cp})
    print(json.dumps({'report':str(args.out/name),'seconds':time.perf_counter()-tic,'cases':len(records)}),flush=True)
if __name__=='__main__':main()
