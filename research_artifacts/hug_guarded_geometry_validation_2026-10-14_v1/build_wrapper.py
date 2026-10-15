"""Prepare an independent audit; no geometry or candidate execution occurs here."""
from pathlib import Path
import ast, difflib, hashlib, json

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent
PREVIOUS=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14/validation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    old=(PREVIOUS/'validate_candidate.py').read_text(); new=old
    start=new.index('def verify_recorded_selection('); end=new.index('\ndef main():',start)
    new=new[:start]+"from selection_audit import verify_recorded_selection\n"+new[end:]
    start=new.index("    cp=sha(args.candidate);");end=new.index("    p=read(OLD/'orientation/protocol.json')",start)
    binding='''    cp=sha(args.candidate);candidate=read(args.candidate);assert candidate['schema']==protocol['candidate_schema'];v3=False;v2=True
    targets={c['native_sha256']:c for c in candidate['cases']};assert len(targets)==len(candidate['cases'])==6
    assert candidate['protocol_sha256']==protocol['candidate_protocol_expected_sha256']
    candidate_protocol_file=args.candidate.parent/'protocol.json';assert sha(candidate_protocol_file)==candidate['protocol_sha256']
    candidate_protocol=read(candidate_protocol_file)
    assert candidate['method_version']==protocol['candidate_method_version'] and candidate['method_stage']==protocol['candidate_method_stage']
    assert candidate_protocol['method_version']==protocol['candidate_method_version'] and candidate_protocol['method_stage']==protocol['candidate_method_stage']
    assert sha(args.candidate.parent/'run_guarded_geometry.py')==candidate['script_sha256']
    assert sha(args.candidate.parent/'feasible_surface_refinement.py')==candidate['solver_implementation_sha256']
    assert sha(args.candidate.parent/'exact_direction.py')==candidate['direction_implementation_sha256']
    assert sha(args.candidate.parent/'merge_guarded_geometry.py')==candidate['merge_script_sha256']
    for file,h in candidate_protocol['implementation_sha256'].items():assert sha(file)==h,file
    for file,h in candidate_protocol['source_sha256'].items():assert sha(file)==h,file
    assert candidate['physics_steps']==0 and candidate['guards']['actual']==0 and candidate['guards']['selftests']==6
    assert not candidate['new_generator_proposal'] and not candidate['new_eval_data'] and not candidate['training'] and not candidate['production_changes']
    assert sorted(c['source_inventory_index'] for c in candidate['cases'])==protocol['case_indices']
    assert candidate['cohort_case_indices']==protocol['case_indices']
    assert candidate['strict_acceptance_rule_unchanged_sha256']=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
    assert candidate['protected_hashes_unchanged'] and candidate['original_inputs_and_implementations_unchanged']
    cohorts=candidate['cohort_provenance'];assert len(cohorts)==2
    assert [c['cohort_case_indices'] for c in cohorts]==candidate_protocol['allowed_cohorts']==[[8,16,24],[12,20,28]]
    cohort_cases=[]
    for cohort in cohorts:
        assert sha(cohort['path'])==cohort['sha256'];report=read(cohort['path'])
        assert report['schema']=='hug_guarded_geometry_surface_candidate_cohort_v1'
        assert report['protocol_sha256']==candidate['protocol_sha256']
        assert report['method_version']==3 and report['method_stage']==candidate['method_stage']
        assert report['cohort_case_indices']==cohort['cohort_case_indices']
        assert [c['source_inventory_index'] for c in report['cases']]==cohort['cohort_case_indices']
        assert report['guards']['actual']==0 and report['guards']['selftests']==3 and report['physics_steps']==0
        assert not report['new_eval_data'] and not report['new_generator_proposal'] and not report['training']
        assert report['protected_hashes_unchanged'] and report['original_inputs_and_implementations_unchanged']
        assert report['strict_acceptance_rule_unchanged_sha256']==candidate['strict_acceptance_rule_unchanged_sha256']
        for key in ['script_sha256','solver_implementation_sha256','direction_implementation_sha256']:
            assert report[key]==candidate[key]
        cohort_cases+=report['cases']
    assert len(cohort_cases)==len({c['source_inventory_index'] for c in cohort_cases})==len({c['native_sha256'] for c in cohort_cases})==6
    assert sorted(cohort_cases,key=lambda c:c['source_inventory_index'])==candidate['cases']
    assert {c['source_inventory_index']:c['native_sha256'] for c in candidate['cases']}=={c['source_inventory_index']:c['native_sha256'] for c in candidate_protocol['cases']}
    for target in candidate['cases']:
        seed=candidate_protocol['seed_bindings'][str(target['source_inventory_index'])]
        assert sha(seed['old_cohort_path'])==seed['old_cohort_sha256']
        old_case=read(seed['old_cohort_path'])['cases'][0]
        assert old_case['native_sha256']==target['native_sha256'] and old_case['joint_names']==target['joint_names']
        assert np.array_equal(np.asarray(old_case['original_qpos_rad']),np.asarray(target['original_qpos_rad']))
        assert np.array_equal(np.asarray(old_case['raw_candidate_qpos_rad']),np.asarray(target['predeclared_seed_qpos_rad']))
        assert target['seed_source_cohort_sha256']==seed['old_cohort_sha256']
    if args.preflight_only:
        assert args.case_index is None, 'All six preflight cases must complete together'
    else:
        assert not args.skip_orientation, 'Full audit cannot omit orientation'
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
'''
    new=new[:start]+binding+new[end:]
    # A separate branch reuses original numerical reports only after exact scene
    # identity. The new-pose measurement block remains literally unchanged.
    needle="        for density in selected['densities_mm']:"
    assert new.count(needle)==1
    state='''        object_body=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')
        assert np.array_equal(d.xpos[object_body],pos) and np.array_equal(d.xmat[object_body].reshape(3,3),R)
        rec['scene_state_binding']={'original_qpos_float64_sha256':hashlib.sha256(np.asarray(original_q,dtype='<f8').tobytes()).hexdigest(),
            'selected_qpos_float64_sha256':hashlib.sha256(np.asarray(d.qpos,dtype='<f8').tobytes()).hexdigest(),
            'qpos24_exact_original':bool(np.array_equal(qv,np.asarray(target['original_qpos_rad']))),
            'entire_scene_qpos_exact_original_including_root':bool(np.array_equal(d.qpos,original_q)),
            'time_s':float(d.time),'object_qpos_exact_original':bool(np.array_equal(d.qpos[objadr:objadr+7],initial)),
            'object_world_position_and_rotation_exact_original':True}
'''
    before_preflight="        if args.preflight_only:\n            records.append(rec)"
    assert before_preflight in new
    new=new.replace(before_preflight,state+before_preflight)
    reuse='''        if np.array_equal(d.qpos,original_q):
            assert np.array_equal(qv,np.asarray(target['original_qpos_rad'])) and d.time==0
            assert np.array_equal(rootpos,original_q[rootadr:rootadr+3]) and np.array_equal(rootquat,original_q[rootadr+3:rootadr+7])
            for file,expected in meta['input_sha256'].items():assert sha(file)==expected,file
            for density in selected['densities_mm']:
                key=str(density);cache=Path(p['original_query_cache_root'])/f'{caseid}_density{density}mm_queries.npz'
                assert sha(cache)==p['query_cache_sha256'][str(cache)]
                prior_sampling=cm[h]['resolutions'][key]['numerical_allowances'];prior_orientation=om[h]['resolutions'][key]
                with np.load(cache,allow_pickle=False) as f:
                    assert len(f['query_object_m'])==prior_orientation['query_count']
                    assert f['query_object_m'].dtype==np.float64 and np.isfinite(f['query_object_m']).all()
                check=status(prior_sampling,prior_orientation,rule,sp['primary_allowance_m'])
                rec['resolutions'][key]={'query_cache_sha256':sha(cache),'query_count':prior_orientation['query_count'],
                    'conditional_sampling':prior_sampling,'orientation':prior_orientation,'frozen_acceptance_application':check,
                    'baseline_acceptance':am[h,key],'baseline_conditional_sampling':prior_sampling,'baseline_orientation':prior_orientation,
                    'runtime_seconds':0.0,'audit_provenance':{'measurement_mode':'REUSED_EXACT_ORIGINAL_SCENE_STATE_NO_GEOMETRY_RERUN',
                        'fresh_source_FK_preflight_this_run':True,'new_surface_or_orientation_measurement_executed':False,
                        'original_sampling_report_sha256':sha(OLD/'sampling/sampling_uncertainty.json'),
                        'original_orientation_report_sha256':sha(OLD/'orientation/orientation_report_full.json.gz'),
                        'sampling_protocol_sha256':sha(OLD/'sampling/protocol.json'),'orientation_protocol_sha256':sha(OLD/'orientation/protocol.json'),
                        'acceptance_rule_sha256':sha(OLD/'acceptance/acceptance_rule.json'),'native_sha256':h,
                        'input_sha256':meta['input_sha256'],'original_and_selected_scene_qpos_float64_sha256':rec['scene_state_binding']['original_qpos_float64_sha256']}}
            records.append(rec)
            assert np.array_equal(d.qpos,nominal) and d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            write(args.out/f'{caseid}_validation.json',rec)
            print(json.dumps({'reused_original_audit':caseid,'densities_mm':selected['densities_mm'],'geometry_rerun':False}),flush=True)
            continue
'''
    new=new.replace(needle,reuse+needle)
    new=new.replace("'hug_feasible_seed_candidate_independent_static_validation_v2'","'hug_guarded_geometry_candidate_independent_static_validation_v1'")
    ast.parse(new)
    (OUT/'validate_candidate.py').write_text(new)
    (OUT/'wrapper.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(PREVIOUS/'validate_candidate.py'),tofile=str(OUT/'validate_candidate.py'))))
    oldtree,newtree=ast.parse(old),ast.parse(new);functions={}
    for name in ['full_fields','status']:
        oldnode=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        newnode=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        oldtext=ast.get_source_segment(old,oldnode);newtext=ast.get_source_segment(new,newnode)
        assert oldtext==newtext
        functions[name]={'literal_bytes_equal':True,'sha256':hashlib.sha256(oldtext.encode()).hexdigest()}
    startold=old.index(needle);endold=old.index("        for file,expected in meta['input_sha256'].items()",startold)
    startnew=new.rindex(needle);endnew=new.index("        for file,expected in meta['input_sha256'].items()",startnew)
    assert old[startold:endold]==new[startnew:endnew]
    parity={'schema':'hug_guarded_geometry_independent_wrapper_literal_parity_v1','old_wrapper_sha256':sha(PREVIOUS/'validate_candidate.py'),
        'draft_wrapper_sha256':sha(OUT/'validate_candidate.py'),'unchanged_functions':functions,
        'entire_new_pose_density_query_proximity_cell_orientation_status_measurement_block_literal_bytes_equal':True,
        'density_measurement_block_sha256':hashlib.sha256(new[startnew:endnew].encode()).hexdigest(),
        'actual_audit_executed':False,'source_preflight_executed':False,'root_freeze_review_required':True}
    (OUT/'wrapper_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
    p=json.loads((PREVIOUS/'protocol.json').read_text());p.update({
        'schema':'hug_guarded_geometry_independent_static_validation_protocol_v1_DRAFT','frozen_utc':None,
        'script_sha256':sha(OUT/'validate_candidate.py'),'candidate_schema':'hug_guarded_geometry_surface_candidate_v1',
        'candidate_protocol_expected_sha256':'d7d489505bd707278a3b0eb0aaedd447182af101ab60b10f40ea0daa221c9ed0',
        'candidate_method_version':3,'candidate_method_stage':'fixed_direction_guarded_bounded_pattern_v1',
        'candidate_path':'/private/tmp/hug_guarded_geometry_2026-10-14_v1/candidate_targets.json',
        'execution_authorized_by_root_review':False,
        'scope':'Same six saved development grasps. Fresh all-six original/final source-URDF and scene FK, source/palm/root/joint/cap/time0 preflight; unchanged original six-coarse/three-fine audit only for new scene q. Literal unchanged original scene state can reuse bound original reports with explicit no rerun.',
        'selection_trace_audit':'Independent recorded feasible pattern-selection and per-part direction guard arithmetic. Not an objective replay at every historical pose.',
        'amendment':'Candidate provenance/selector and exact-original scene-state audit reuse only. Literal scientific full_fields/status and entire new-pose density loop unchanged.',
        'protected_snapshot_sha256':sha(OUT/'protected_hashes.json'),
        'exact_original_reuse_policy':'Require bitwise equality of entire captured scene qpos including root, joint target equality, fresh all-six FK/source preflight, all input/asset/protocol/query-cache hashes. Explicitly report REUSED_EXACT_ORIGINAL_SCENE_STATE_NO_GEOMETRY_RERUN. Internal q24 identity alone is insufficient.',
    })
    p.pop('candidate_merge_script_path',None)
    p['fixed_files_sha256']={k:v for k,v in p['fixed_files_sha256'].items() if 'hug_feasible_refinement_2026-10-14/' not in k}
    fit_path=Path(p['candidate_path']).parent/'protocol.json';fit=json.loads(fit_path.read_text())
    assert sha(fit_path)==p['candidate_protocol_expected_sha256']
    p['candidate_solver_configuration']=fit['solver_configuration']
    p['fixed_files_sha256'][str(fit_path)]=sha(fit_path)
    p['fixed_files_sha256'][str(OUT/'selection_audit.py')]=sha(OUT/'selection_audit.py')
    p['fixed_files_sha256'][str(OUT/'run_validation.py')]=sha(OUT/'run_validation.py')
    p['execution_launcher_sha256']=sha(OUT/'run_validation.py')
    p['root_review']='PENDING final candidate-provenance/independent trace verifier review. Scientific byte parity and exact-scene reuse plan previously approved. No geometry execution until new final freeze.'
    p['limitations'].append('No disappearance of observed sign reversals or cell witnesses certifies full continuous orientation/contact or native outcome equivalence; finite selector budgets do not establish a global impossibility.')
    (OUT/'protocol.DRAFT.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(parity,indent=2))

if __name__=='__main__':main()
