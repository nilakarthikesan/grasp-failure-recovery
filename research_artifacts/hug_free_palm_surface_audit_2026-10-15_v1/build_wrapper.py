from pathlib import Path
import ast,hashlib,json
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent
old=(ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14/validation/validate_candidate.py').read_text()
header=old[:old.index('def verify_recorded_selection')]
start=old.index('        for density in selected[\'densities_mm\']:')
stop=old.index('        for file,expected in meta[\'input_sha256\'].items():',start)
metrics=old[start:stop]
(OUT/'inherited_metric_block.py.txt').write_text(metrics)
main='''def main():
    global TEST
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent);ap.add_argument('--case-index',type=int,nargs='*');ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args();args.skip_orientation=False;args.out.mkdir(exist_ok=True);tic=time.perf_counter()
    protocol=read(args.out/'protocol.json');ph=sha(args.out/'protocol.json');assert sha(__file__)==protocol['script_sha256']
    assert str(args.candidate)==protocol['candidate_path']
    for file,h in protocol['fixed_files_sha256'].items():assert sha(file)==h,file
    assert protocol['execution_authorized_by_root_review'], 'Frozen root review required before audit execution'
    cp=sha(args.candidate);assert cp==protocol['candidate_sha256'];candidate=read(args.candidate);assert candidate['schema']==protocol['candidate_schema']
    targets={c['native_sha256']:c for c in candidate['cases']};assert len(targets)==len(candidate['cases'])==6
    assert candidate['protocol_sha256']==protocol['candidate_protocol_expected_sha256']
    candidate_protocol_file=args.candidate.parent/'protocol.json';assert sha(candidate_protocol_file)==candidate['protocol_sha256']
    candidate_protocol=read(candidate_protocol_file)
    for file,h in protocol['candidate_bound_files_sha256'].items():assert sha(file)==h,file
    if args.preflight_only:
        assert args.case_index is None, 'All six preflight cases must complete together'
    else:
        assert args.case_index in protocol['allowed_full_audit_cohorts']
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
    assert sha(OLD/'acceptance/acceptance_rule.json')=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
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
    write(args.out/('sanity_report_'+('all' if args.case_index is None else '_'.join(map(str,args.case_index)))+'.json'),ori.sanity(p))
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    w=np.asarray(asset['weights']);grouped=np.column_stack([w[:,ids].sum(1) for ids in a.GROUPS.values()]);records=[];validator=ShadowPoseValidator(a.URDF.parent.parent)
    for selected in p['cases']:
        index=selected['source_inventory_index']
        if args.case_index is not None and index not in args.case_index:continue
        case=inventory['modern_cases'][index];h=case['native']['sha256'];assert h==selected['native_sha256'];target=targets[h];assert target['source_inventory_index']==index
        role=target['audited_pose_role'];assert role in protocol['allowed_audited_pose_roles']
        assert isinstance(target['guarded_mapper_selected_fallback'],bool) and isinstance(target['direction_guard_pass'],bool) and isinstance(target['diagnostic_pose_not_promoted'],bool)
        if role=='tip_feasible_translated_diagnostic_not_selected':assert target['diagnostic_pose_not_promoted'] and target['guarded_mapper_selected_fallback']
        audit_role={'audited_pose_role':role,'guarded_mapper_selected_fallback':target['guarded_mapper_selected_fallback'],'direction_guard_pass':target['direction_guard_pass'],'diagnostic_pose_not_promoted':target['diagnostic_pose_not_promoted'],'reason':target.get('audit_selection_reason'),'selection_source':target.get('audit_selection_source')}

        meta,tri,labels,nn,hand,objs,m,d,initial,objadr,pos,R=a.load_case(case,grouped,fk);palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');original_palm_pos=d.xpos[palm].copy();original_palm_R=d.xmat[palm].reshape(3,3).copy();original_q=d.qpos.copy();original_scene_links=_link_positions(m,d,a.URDF).copy();rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);bound=[]
        assert target['joint_names']==meta['stored_command']['joint_names'];qv=np.asarray(target['qpos_rad'],float);assert qv.shape==(24,) and np.isfinite(qv).all()
        original_values=np.asarray([meta['recovered_original_targets'][n] for n in target['joint_names']]);assert np.array_equal(np.asarray(target['original_qpos_rad']),original_values)
        for name,value,q0 in zip(target['joint_names'],qv,original_values,strict=True):
            low,high=ranges[name];assert low-1e-9<=value<=high+1e-9,(name,value,low,high)
            j=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);sl,sh=m.jnt_range[j];difference=max(abs(sl-low),abs(sh-high));assert sl-difference-1e-9<=value<=sh+difference+1e-9
            qa=int(m.jnt_qposadr[j]);delta=float(value-d.qpos[qa]);assert abs(delta)<=protocol['maximum_joint_delta_rad']+1e-9,(name,delta);d.qpos[qa]=value
            bound.append({'joint':name,'delta_from_original_rad':delta,'source_URDF_range_rad':[low,high],'scene_range_rad':[float(sl),float(sh)],'scene_representation_difference_rad':float(difference),'strict_scene_overflow_rad':float(max(sl-value,value-sh,0))})
        rootpos=np.asarray(target['root_target_position_m'],float);rootquat=np.asarray(target['root_target_quaternion_wxyz'],float);assert rootpos.shape==(3,) and rootquat.shape==(4,) and np.isfinite(rootpos).all() and np.isfinite(rootquat).all();assert abs(np.linalg.norm(rootquat)-1)<1e-8
        assert np.array_equal(rootquat,original_q[rootadr+3:rootadr+7]), 'Root orientation must stay original in this new translation-only placement method'
        grasp=load_hug_prediction(case['native']['path']);T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);T_wrist=T_wc@grasp.T_camera_wrist;Rop=T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        local_landmarks=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
        original_pose=RetargetedHandPose(tuple(target['joint_names']),original_values,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(original_pose)
        original_fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();original_fixed_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();original_fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf)
        original_expected_links=(original_fixed_links-original_fixed_palm)@Rop.T+T_wrist[:3,3]
        original_source_scene_link_error=float(np.linalg.norm(original_scene_links-original_expected_links,axis=1).max());original_source_palm_rotation_error=float(np.max(np.abs(original_palm_R-Rop@original_fixed_R)));original_source_palm_position_error=float(np.linalg.norm(original_palm_pos-T_wrist[:3,3]))
        assert original_source_scene_link_error<1.1e-7 and original_source_palm_rotation_error<1e-6 and original_source_palm_position_error<1e-7
        pose=RetargetedHandPose(tuple(target['joint_names']),qv,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose)
        fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();fixed_palm_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf);desired_R=Rop@fixed_palm_R
        shift=np.asarray(target['palm_displacement_world_m'],float);assert shift.shape==(3,) and np.isfinite(shift).all();anchor=T_wrist[:3,3]+shift
        if role=='tip_feasible_translated_diagnostic_not_selected':assert np.any(shift!=0)
        assert np.linalg.norm(np.asarray(target['target_palm_world_position_m'])-anchor)<1e-12
        d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
        if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
        mujoco.mj_forward(m,d)
        palm_error=float(np.linalg.norm(d.xpos[palm]-anchor));rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desired_R)));expected_links=(fixed_links-fixed_palm)@Rop.T+anchor;scene_links=_link_positions(m,d,validator.source_urdf);link_error=float(np.linalg.norm(scene_links-expected_links,axis=1).max())
        assert palm_error<=1e-12 and rotation_error<1e-6 and link_error<1.1e-7,(palm_error,rotation_error,link_error)
        if 'target_palm_world_rotation' in target:assert np.max(np.abs(np.asarray(target['target_palm_world_rotation'])-d.xmat[palm].reshape(3,3)))<1e-12
        native_world=grasp.landmarks_3d@T_wc[:3,:3].T+T_wc[:3,3];native_tasks=native_world[np.array([4,8,12,16,20,2,6,10,14,18])];original_errors=np.linalg.norm(original_scene_links-native_tasks,axis=1);candidate_errors=np.linalg.norm(scene_links-native_tasks,axis=1)
        assert np.max(np.abs(original_errors-np.asarray(target['baseline_fresh_diagnostics']['task_link_error_norm_m'])))<1e-12;assert np.max(np.abs(candidate_errors-np.asarray(target['selected_fresh_diagnostics']['task_link_error_norm_m'])))<1e-12
        original_tip_vectors=original_scene_links[:5]-native_tasks[:5];candidate_tip_vectors=scene_links[:5]-native_tasks[:5];original_cap_norms=np.sqrt(np.einsum('ij,ij->i',original_tip_vectors,original_tip_vectors));candidate_cap_norms=np.sqrt(np.einsum('ij,ij->i',candidate_tip_vectors,candidate_tip_vectors))
        assert protocol['individual_fingertip_arithmetic_allowance_m']==0
        assert np.max(np.abs(original_cap_norms-np.asarray(target['authoritative_original_tip_caps_sqrt_einsum_m'])))<1e-12
        assert np.max(np.abs(candidate_cap_norms-np.asarray(target['selected_tip_norms_sqrt_einsum_m'])))<1e-12
        frozen_caps=np.asarray(target['authoritative_original_tip_caps_sqrt_einsum_m']);assert np.all(candidate_cap_norms<=frozen_caps), 'Exact original frozen caps cannot be rebased by independent reconstruction tolerance'
        actual_mean=float(candidate_cap_norms.mean());assert np.all(candidate_cap_norms<=original_cap_norms),{'errors':candidate_cap_norms.tolist(),'original_caps':original_cap_norms.tolist()};assert actual_mean<=protocol['existing_engineering_mean_tip_guard_m']
        guard_audit={'native_landmark_errors_independently_recomputed':True,'original_task_link_errors_m':original_errors.tolist(),'candidate_task_link_errors_m':candidate_errors.tolist(),'original_tip_error_vectors_m':original_tip_vectors.tolist(),'candidate_tip_error_vectors_m':candidate_tip_vectors.tolist(),
            'original_tip_caps_sqrt_einsum_m':original_cap_norms.tolist(),'frozen_authoritative_tip_caps_sqrt_einsum_m':frozen_caps.tolist(),'candidate_tip_norms_sqrt_einsum_m':candidate_cap_norms.tolist(),'norm_arithmetic':'sqrt(einsum(ij,ij->i)); np.linalg.norm is a separate diagnostic','individual_fingertip_arithmetic_allowance_m':0.,'each_final_fingertip_nonworsening_exact_computed':True,'fingertip_error_change_m':(candidate_cap_norms-original_cap_norms).tolist(),'actual_mean_fingertip_error_m':actual_mean,'strict_actual_mean_guard_m':protocol['existing_engineering_mean_tip_guard_m'],'actual_mean_guard_pass':True,
            'original_source_URDF_vs_scene_max_link_error_m':original_source_scene_link_error,'original_source_URDF_vs_scene_palm_rotation_maxabs':original_source_palm_rotation_error,'original_native_wrist_to_palm_position_error_m':original_source_palm_position_error,'source_preflight_status':'PASS','fallback_original_complete_pose_exact':bool(np.array_equal(d.qpos,original_q)),'does_not_mean_scientific_mapping_acceptance':True}
        if role=='original_fallback_no_feasible_nonzero':assert np.array_equal(d.qpos,original_q) and np.all(shift==0) and target['guarded_mapper_selected_fallback']
        assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);nominal=d.qpos.copy();newhand=[]
        for g in hand:gg=a.geom_rep(m,d,g['id'],R,pos);gg['part']=g['part'];newhand.append(gg)
        vt,on,_=ori.visual_mesh(meta,m,d,pos,R);caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        rec={'source_inventory_index':index,'native_sha256':h,'case_id':caseid,'candidate_joint_range_checks':bound,'engineering_guard_independent_audit':guard_audit,'audit_selection_role':audit_role,
            'root_policy':{'translated_anchor_world_m':anchor.tolist(),'palm_displacement_world_m':shift.tolist(),'palm_displacement_norm_m':float(np.linalg.norm(shift)),'translated_anchor_to_palm_position_error_m':palm_error,'declared_world_operator_times_base_palm_rotation_maxabs':rotation_error,'independent_fixed_URDF_vs_scene_max_link_error_m':link_error,'root_orientation_bitwise_original':True,'q_dependent_carrier_and_explicit_palm_translation':True,'not_an_anatomy_equivalence_certificate':True},'input_sha256':meta['input_sha256'],'full_scene_qpos_sha256':hashlib.sha256(nominal.tobytes()).hexdigest(),'resolutions':{}}
        if args.preflight_only:
            np.save(args.out/f'case_{index}_full_qpos.npy',nominal,allow_pickle=False)
            records.append(rec);assert d.time==0 and np.array_equal(d.qpos,nominal) and np.array_equal(d.qpos[objadr:objadr+7],initial);print(json.dumps({'preflight':caseid,'passed':True,'actual_mean_tip_mm':actual_mean*1000,'palm_displacement_mm':float(np.linalg.norm(shift))*1000}),flush=True);continue
        assert np.array_equal(nominal,np.load(args.out/f'case_{index}_full_qpos.npy',allow_pickle=False))
'''
ending='''        for file,expected in meta['input_sha256'].items():assert sha(file)==expected,file
        records.append(rec)
    for file,h in protected.items():assert sha(file)==h,file
    for file,h in protocol['fixed_files_sha256'].items():assert sha(file)==h,file
    for file,h in protocol['candidate_bound_files_sha256'].items():assert sha(file)==h,file
    assert sha(args.candidate)==cp and sha(args.out/'protocol.json')==ph and sha(__file__)==protocol['script_sha256'];assert COUNTS['actual']==0
    report={'schema':'hug_free_palm_candidate_independent_static_validation_v1','candidate_sha256':cp,'candidate_path':str(args.candidate),'protocol_sha256':ph,'script_sha256':sha(__file__),'all_source_geometry_and_protected_hashes_unchanged':True,'protected_source_count':len(protected),'physics_steps':0,'guards':COUNTS,'new_eval_data':False,'new_inference':False,'training':False,'development_cases_only':True,'preflight_only':args.preflight_only,'skip_orientation_pilot':False,'runtime_seconds':time.perf_counter()-tic,'cases':records,'limitations':protocol['limitations']}
    raw=(json.dumps(report,indent=2,allow_nan=False)+'\\n').encode();name='candidate_validation'+('_'+','.join(map(str,args.case_index)) if args.case_index else '')+('_preflight_only' if args.preflight_only else '')+'.json.gz';(args.out/name).write_bytes(gzip.compress(raw,mtime=0));write(args.out/(name+'.binding.json'),{'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'gzip_sha256':sha(args.out/name),'case_count':len(records),'protocol_sha256':ph,'candidate_sha256':cp})
    print(json.dumps({'report':str(args.out/name),'seconds':time.perf_counter()-tic,'cases':len(records)}),flush=True)
if __name__=='__main__':main()
'''
new=header+main+metrics+ending
ast.parse(new)
(OUT/'validate_candidate.py').write_text(new)
parity={'source':str(ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14/validation/validate_candidate.py'),'source_sha256':hashlib.sha256(old.encode()).hexdigest(),'density_metric_block_byte_identical':metrics==new[new.index('        for density in selected[\'densities_mm\']:'):new.index('        for file,expected in meta[\'input_sha256\'].items():')],'full_fields_and_status_byte_identical':header==new[:new.index('def main():')],'new_script_sha256':hashlib.sha256(new.encode()).hexdigest()}
(OUT/'wrapper_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
print(parity)
