"""Read-only arithmetic/hash replay; no simulator, numerical solver or geometry imports."""
from pathlib import Path
import hashlib,json,math,struct
D=Path('/private/tmp/hug_se3_independent_review_2026-10-15_v1')
S=Path('/private/tmp/hug_se3_mapping_2026-10-15_v1')
R=Path('/private/tmp/hug_retargeter_state_audit_2026-10-15_v1')
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
checks=[]
def check(name,predicate):
 assert predicate,name
 checks.append(name)
bindings={}
def bind(p):
 p=Path(p);bindings[str(p)]=sha(p);return p
p=read(bind(S/'preflight_protocol_v2.json'));k=read(bind(S/'case20_kinematic_controls_v2.json'));receipt=read(bind(S/'kinematic_implementation_receipt.json'))
for label in ['input_sha256','implementation_sha256']:
 for path,h in p[label].items():check('SE3 preflight source '+path,sha(path)==h)
for path,h in receipt['source_bindings'].items():check('kinematic receipt source '+path,sha(path)==h)
check('kinematic pass',k['passed'] is True and receipt['passed'] is True)
check('kinematic protocol',k['protocol_sha256']==sha(S/'preflight_protocol_v2.json'))
check('three fixed records',len(k['records'])==len(p['jacobian_states'])==3)
errors=[]
for r,state in zip(k['records'],p['jacobian_states']):
 check(r['label']+' fixed state',r['label']==state['label'] and r['state_scaled']==state['state_scaled'])
 diffs=[[[abs(a-b) for a,b in zip(aa,bb)] for aa,bb in zip(ar,br)] for ar,br in zip(r['analytic'],r['finite_difference'])]
 check(r['label']+' shape',len(diffs)==5 and all(len(t)==3 and all(len(v)==30 for v in t) for t in diffs))
 maxerr=max(z for t in diffs for v in t for z in v);errors.append(maxerr)
 check(r['label']+' max replay',maxerr==r['maxabs_Jacobian_difference_m_per_scaled_state'])
 check(r['label']+' original tolerance',maxerr<=p['jacobian_tolerance_m_per_scaled_state'])
 for key,lo,hi in [('joint_block_maxabs',0,24),('translation_block_maxabs',24,27),('rotation_block_maxabs',27,30)]:
  check(r['label']+key,max(z for t in diffs for v in t for z in v[lo:hi])==r[key])
check('zero rotation original identity','original:zero_rotation_all_qpos_bitwise' in k['checks'] and 'original:zero_rotation_all_body_poses_bitwise' in k['checks'])
check('zero rotation pooled identity','pooled:zero_rotation_all_qpos_bitwise' in k['checks'] and 'pooled:zero_rotation_all_body_poses_bitwise' in k['checks'])
check('kinematic scope',k['surface_callbacks']==k['physics_steps']==0 and k['new_fit'] is False and k['new_eval_data'] is False)
check('390 author forwards',k['actual_mj_forward_calls']==390)
check('final 16 tests','16 passed' in bind(R/'implementation_tests_final.log').read_text())
check('13 math tests','13 passed' in bind(S/'manufactured_tests.log').read_text())
check('8 solver tests','8 passed' in bind(S/'solver_controls.log').read_text())
for n in ['se3_math.py','se3_case.py','se3_solver.py','run_pilot.py','test_se3_solver.py','test_se3_math.py']:bind(S/n)
for n in ['src/grasp_failure_prediction/evaluation/retargeting.py','tests/test_retargeting.py']:bind(W/n)
for n in ['retargeting.py.proposed','test_retargeting.py.proposed','implementation_change_plan.json']:bind(R/n)
check('final implementation matches proposal',sha(W/'src/grasp_failure_prediction/evaluation/retargeting.py')==sha(R/'retargeting.py.proposed'))
check('final tests match proposal',sha(W/'tests/test_retargeting.py')==sha(R/'test_retargeting.py.proposed'))
c=read(bind(R/'corrected_profile/results.json'));cp=read(bind(R/'corrected_profile/protocol.json'));bind(R/'corrected_profile/run_corrected_probe.py');bind(R/'corrected_profile/run.log')
check('corrected protocol',c['protocol_sha256']==sha(R/'corrected_profile/protocol.json'))
for path,h in cp['input_sha256'].items():check('corrected profile source '+path,sha(path)==h)
rows={v['label']:v for v in c['mappings']}
check('five corrected mappings',len(rows)==5 and len(c['mappings'])==5)
for label,rel in c['order_invariance'].items():
 lhs=rows[label]['qpos_rad'];rhs=rows[rel['reference']]['qpos_rad']
 check(label+' exact pose bytes',struct.pack('<'+'d'*len(lhs),*lhs)==struct.pack('<'+'d'*len(rhs),*rhs))
 check(label+' zero difference',max(abs(a-b) for a,b in zip(lhs,rhs))==rel['maximum_absolute_joint_difference_rad']==0 and rel['bitwise_equal'] is True)
for label,row in rows.items():
 prov=row['retargeting_provenance']
 check(label+' profile provenance',prov['profile_id']=='independent_static_shadow_vector_v2' and prov['temporal_norm_delta']==0 and prov['reset_before_proposal'] is True and prov['refinement_calls']==20)
 check(label+' status provenance',len(prov['optimizer_status_codes'])==20 and all(type(code) is int and code>0 for code in prov['optimizer_status_codes']))
for label,row in c['fresh_profile_comparison'].items():
 check(label+' worsened tips arithmetic',sum(x>0 for x in row['corrected_minus_legacy_fingertip_error_m'])==row['tips_worsened_in_source_urdf_FK'])
 check(label+' acceptance not tested',row['full_mapping_acceptance']=='NOT_TESTED' and row['source_urdf_original_tip_nonworsening_all'] is False)
check('corrected scope',c['physics_steps']==c['surface_queries']==0 and c['new_model_inference'] is False and c['new_eval_data'] is False and c['new_default_selection'] is False)
bind(D/'source_review.json');bind(D/'finalize_review.py')
out={
 'schema':'hug_se3_and_static_profile_final_independent_review_v1',
 'passed':True,'scope':'Source review plus independent saved-array and hash arithmetic; readiness for the separately frozen bounded SE3 pilot, not mapping validation.',
 'source_review_status':'No blocking code issue. This final receipt supersedes the preliminary receipt statements that the saved-case preflight remained pending, only 12 tests existed, and static-profile status refusal remained absent.',
 'source_bindings':bindings,'pure_read_only_checks_passed':len(checks),'checks':checks,
 'SE3':{'full_30D_parameterization':True,'rotation_composition':'Exp(world omega) left multiplies original root; actual palm offset is compensated by root translation','zero_omega_exact_historical_state':True,'maximum_saved_FD_discrepancy_m_per_scaled_state':max(errors),'maximum_rigid_identity_discrepancy_m':max(r['rigid_task_identity_maxabs_m'] for r in k['records']),'author_static_forward_calls':390,'reviewer_forward_calls':0,'rotation_domain':'Each rotation-vector component is +/-15 degrees; the norm can reach sqrt(3)*15 degrees. This is a search bound, not an acceptance threshold.'},
 'solver_review':{'original_tip_caps_and_mean_and_domain_preserved':True,'complete_30D_cache_key':True,'raw_probes_not_selection_candidates':True,'protected_orientation_masks_preserved':True,'fresh_terminal_rank_and_full_predicate_vectors_required':True,'fresh_pooled_fallback_provenance_preserved':True,'budget':'1800 raw and 60 exact calls, counting raw inside exact; two reserved exact calls. Two starts, 25 iterations each. The 180-second cooperative search stop excludes reconstruction and reserved terminal work; an in-flight callback may finish afterward.'},
 'static_profile':{'optional_profile':'independent_static_shadow_vector_v2','production_default':'legacy_shadow_vector_v1','norm_delta':0.0,'reset_before_each_proposal':True,'negative_zero_missing_or_noninteger_status_refused':True,'positive_budget_status_recorded_without_convergence_claim':True,'repository_tests_passed':16,'saved_reused_A_B_A_exact_to_fresh':True,'individual_tips_worsened_vs_fresh_legacy':{k:v['tips_worsened_in_source_urdf_FK'] for k,v in c['fresh_profile_comparison'].items()},'full_mapping_acceptance':'NOT_TESTED'},
 'historical_scientific_status':'Unchanged nine strict FAIL results; no new mapping PASS and no production mapper promotion.',
 'limitations':['Manufactured and one-case derivative controls test implementation correctness, not all-state or all-case accuracy.','Original native targets and acceptance criteria remain fixed; future target geometry must use each optimized root quaternion.','The SE3 pilot and optional static profile are distinct versioned paths. Neither proves morphology equivalence.','Positive optimizer termination codes do not establish global convergence or scientific validity.'],
 'reviewer_new_geometry_queries':0,'reviewer_new_FK':0,'reviewer_new_fit':False,'new_native_inference':False,'physics_steps':0,'new_eval_data':False}
(D/'final_review.json').write_text(json.dumps(out,indent=2)+'\n')
manifest={'schema':'hug_se3_independent_review_manifest_v1','files':{str(p.relative_to(D)):sha(p) for p in sorted(D.iterdir()) if p.is_file() and p.name!='manifest.json'},'scope':'Read-only implementation review; previous frozen evidence directories unchanged.'}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'final_review_sha256':sha(D/'final_review.json'),'manifest_sha256':sha(D/'manifest.json')}))
