"""Draft wrapper changes only candidate binding, exact-cap/selection preflight."""
from pathlib import Path
import ast, difflib, hashlib, json

OUT=Path(__file__).parent
OLD=Path('/private/tmp/hug_constrained_validation_runtime_v2_1_2026-10-14')
STAGE=OUT.parent/'seed_stage'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    old=(OLD/'validate_candidate.py').read_text();new=old
    start=new.index('def verify_recorded_selection(');end=new.index('\ndef main():',start)
    new=new[:start]+(OUT/'selection_audit.py').read_text().split('def verify_recorded_selection',1)[-1].join(['def verify_recorded_selection',''])+new[end:]
    # The leading import/docstring of selection_audit is intentionally omitted;
    # its only dependency is the wrapper's existing NumPy import.
    start=new.index("    cp=sha(args.candidate);")
    end=new.index("    p=read(OLD/'orientation/protocol.json')",start)
    binding="""    assert protocol['execution_authorized_by_root_review'], 'Frozen root review required before audit execution'
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
"""
    new=new[:start]+binding+new[end:]
    new=new.replace("assert len(protected)==114","assert len(protected)==protocol['protected_file_count']")
    # Add a separately evaluated original source-URDF FK check before candidate
    # source FK. This cannot change the later density/surface computations.
    needle=";pose=RetargetedHandPose(tuple(target['joint_names']),qv,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose)"
    assert needle in new
    replacement="""
        original_pose=RetargetedHandPose(tuple(target['joint_names']),np.asarray(target['original_qpos_rad']),local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(original_pose)
        original_fixed_palm=validator.data.xpos[_body_id(validator.model,'palm')].copy();original_fixed_R=validator.data.xmat[_body_id(validator.model,'palm')].reshape(3,3).copy();original_fixed_links=_link_positions(validator.model,validator.data,validator.source_urdf)
        original_expected_links=(original_fixed_links-original_fixed_palm)@Rop.T+T_wrist[:3,3]
        original_source_scene_link_error=float(np.linalg.norm(original_scene_links-original_expected_links,axis=1).max())
        original_source_palm_rotation_error=float(np.max(np.abs(original_palm_R-Rop@original_fixed_R)))
        original_source_palm_position_error=float(np.linalg.norm(original_palm_pos-T_wrist[:3,3]))
        assert original_source_scene_link_error<1.1e-7 and original_source_palm_rotation_error<1e-6 and original_source_palm_position_error<1e-7
        pose=RetargetedHandPose(tuple(target['joint_names']),qv,local_landmarks,np.zeros((10,3)),grasp.T_camera_wrist);validator.set_pose(pose)"""
    new=new.replace(needle,replacement)
    start=new.index("        allowance=protocol['individual_fingertip_arithmetic_allowance_m'];")
    end=new.index('\n        if v3:',start)
    exact="""        original_tip_vectors=original_scene_links[:5]-native_tasks[:5];candidate_tip_vectors=scene_links[:5]-native_tasks[:5]
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
            'does_not_mean_scientific_mapping_acceptance':True}"""
    new=new[:start]+exact+new[end:]
    new=new.replace("'hug_constrained_candidate_independent_static_validation_v1'","'hug_feasible_seed_candidate_independent_static_validation_v2'")
    ast.parse(new)
    (OUT/'validate_candidate.py').write_text(new)
    (OUT/'wrapper.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(OLD/'validate_candidate.py'),tofile=str(OUT/'validate_candidate.py'))))
    oldtree,newtree=ast.parse(old),ast.parse(new)
    functions={}
    for name in ['full_fields','status']:
        oldnode=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        newnode=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        oldtext=ast.get_source_segment(old,oldnode);newtext=ast.get_source_segment(new,newnode)
        assert oldtext==newtext
        functions[name]={'literal_bytes_equal':True,'sha256':hashlib.sha256(oldtext.encode()).hexdigest()}
    startold=old.index('        for density in selected[\'densities_mm\']:')
    endold=old.index("        for file,expected in meta['input_sha256'].items()",startold)
    startnew=new.index('        for density in selected[\'densities_mm\']:')
    endnew=new.index("        for file,expected in meta['input_sha256'].items()",startnew)
    assert old[startold:endold]==new[startnew:endnew]
    parity={'schema':'hug_feasible_seed_independent_wrapper_literal_parity_v2','old_wrapper_sha256':sha(OLD/'validate_candidate.py'),
        'draft_wrapper_sha256':sha(OUT/'validate_candidate.py'),'unchanged_functions':functions,
        'entire_density_query_proximity_cell_orientation_status_measurement_block_literal_bytes_equal':True,
        'density_measurement_block_sha256':hashlib.sha256(new[startnew:endnew].encode()).hexdigest(),
        'actual_audit_executed':False,'source_preflight_executed':False,'root_freeze_review_required':True}
    (OUT/'wrapper_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    p=json.loads((OLD/'protocol.json').read_text());p.update({
        'schema':'hug_feasible_seed_independent_static_validation_protocol_v2_DRAFT','frozen_utc':None,
        'script_sha256':sha(OUT/'validate_candidate.py'),'candidate_schema':'hug_feasible_seed_stage_surface_candidate_v2',
        'candidate_protocol_expected_sha256':'b0e095b39897c6c2d337f53b0c26d82ef5bd6b7632d3c21f60dd9d8e574a98c6',
        'candidate_path':str(STAGE/'candidate_targets.json'),'candidate_merge_script_path':None,
        'execution_authorized_by_root_review':False,'protected_file_count':89,
        'individual_fingertip_arithmetic_allowance_m':0.,
        'scope':'Allsix same saved nativegrasps; independently recompute original/final sourceURDF and capturedscene FK, exactcomputed all5caps and strictmean10mm. Allsix preflight mustfinish before any of the original6coarse+3fine surfaceaudits. No newinference/steps/evaldata.',
        'selection_trace_audit':'Declaredone immutableoldrawseed percase, no pattern sweeps; independently audit recorded proposal/cap/cost/acceptance/fallback/budgets/freshfinalevents. No objective recomputation atallhistorical poses.',
        'amendment':'Only newcandidate schema/provenance/seedselection and exactcap preflight changed. full_fields/status and entire density query/proximity/cell/orientation block byte-identical to previous independent wrapper.',
        'protected_snapshot_sha256':None})
    for key in ['candidate_runtime_protocol_version','candidate_runtime_amendment','candidate_allowed_cohorts','runtime_amendment','engineering_guard_alphas']:
        p.pop(key,None)
    p['limitations'][-1]='Per-tip original-norm nonworsening is an engineering guard. Exactcomputed arithmetic and 1e-12 full/FK comparison are not continuous surface/contact/failure-label equivalence certificates. No feasibility margin is added.'
    p['fixed_files_sha256']={k:v for k,v in p['fixed_files_sha256'].items() if 'protected_hashes.json' not in k and 'hug_constrained_refinement_runtime_v2_1' not in k}
    p['fixed_files_sha256'][str(STAGE/'protocol.json')]=p['candidate_protocol_expected_sha256']
    p['fixed_files_sha256'][str(OUT/'selection_audit.py')]=sha(OUT/'selection_audit.py')
    protected_path=STAGE/'protected_hashes.json'
    if protected_path.exists():
        (OUT/'protected_hashes.json').write_bytes(protected_path.read_bytes());p['protected_snapshot_sha256']=sha(OUT/'protected_hashes.json')
        p['protected_file_count']=len(json.loads(protected_path.read_text()))
    (OUT/'protocol.DRAFT.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(parity))

if __name__=='__main__':main()
