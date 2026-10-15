"""Read saved JSON and hash bytes only; no pickle decoding, geometry or simulator calls."""
from pathlib import Path
import json,hashlib,subprocess,collections,copy
from jsonschema import Draft202012Validator
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
MAIN=Path('/Users/nilakarthikesan/grasp-failure-prediction')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
inv=read(OUT/'inventory_exposure.json'); identities={r['native_sha256'] for r in inv['modern_cases']}
b=ROOT/'research_artifacts'
paths={'native_contract':b/'hug_orientation_uncertainty_acceptance_2026-10-14/native_contract/native_contract.json','native_protocol':b/'hug_orientation_uncertainty_acceptance_2026-10-14/native_contract/protocol.json','original_FK':b/'hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json','compiled_contract':Path('/private/tmp/hug_collision_contract_review_2026-10-15_v1/compiled_contract.json')}
nat=read(paths['native_contract']);fk=read(paths['original_FK']);comp=read(paths['compiled_contract']);nprot=read(paths['native_protocol'])
assert {c['native_sha256'] for c in nat['cases']}==identities=={c['native_sha256'] for c in fk['cases']}
assert sum(c['passed'] is True for c in nat['cases'])==32
assert nat['protocol_sha256']==sha(paths['native_protocol'])
assert [c['source_index'] for c in comp['cases']]==[8,12,16,20,24,28]
assert comp['passed'] is True
for c in comp['cases']:
 r=inv['modern_cases'][c['source_index']]
 for k in ['native_sha256','object_id','geometry_content_hash','observation_group_id']:assert c[k]==r[k]
alljoint=[q for c in fk['cases'] for q in c['joint_limit_checks']]
assert len(alljoint)==32*24 and all(q['violation_rad']==0 for q in alljoint)
fknums={k:max(c[k] for c in fk['cases']) for k in ['native_landmark_wrist_vs_T_camera_wrist_translation_error_m','object_position_vs_scene_contract_error_m','object_rotation_vs_scene_contract_max_abs_error','palm_position_vs_native_wrist_translation_error_m','fixed_urdf_vs_stored_scene_link_position_max_error_m','palm_rotation_vs_native_operator_times_local_palm_rotation_max_abs_error','world_vs_object_link_error_max_delta_m','mean_tip_world_error_m','maximum_tip_world_error_m']}
manifest={'schema':'hug_current_static_claim_manifest_v1','status':'SCOPED_SAVED_EVIDENCE_REPLAY_NOT_NEW_GEOMETRIC_VALIDATION','bindings':{str(p):sha(p) for p in paths.values()},'inventory_binding':sha(OUT/'inventory_exposure.json'),'identity_agreement':{'native_contract_and_original_FK_same32_native_hashes':True,'compiled_contract_exact_subset_indices':[8,12,16,20,24,28]},'supported':[{'claim':'Saved native MANO decoding reconstructs the saved export for these32 sources.','receipt':'native_contract','passed_cases':32,'vertices_per_case':778,'landmarks_per_case':21,'closed_faces_per_case':1552,'maximum_vertex_error_m':nat['max_vertex_error_m'],'maximum_landmark_error_m':nat['max_landmark_error_m'],'maximum_rotation_element_error':nat['max_rotation_element_error'],'all_face_indices_equal':nat['all_face_indices_equal'],'original_numeric_allowances':{'vertex_landmark_m':nprot['vertex_landmark_tolerance_m'],'rotation_element':nprot['rotation_element_tolerance']},'does_not_establish':'Correct physical human grasp, new inference, contact fidelity, or grasp outcome.'},{'claim':'Original nominal Shadow command implementation, captured object/camera/world frame consistency, and independent fixed-URDF versus stored-scene FK were measured on all32.','receipt':'original_FK','cases':32,'joint_limit_comparisons':len(alljoint),'joint_limit_violations':0,'all_time_zero':fk['global_checks']['all_time_zero'],'all_initial_object_qpos_unchanged':fk['global_checks']['all_initial_object_qpos_unchanged'],'maximum_residuals':fknums,'mean_tip_guard_m':0.01,'cases_meeting_existing_mean_tip_guard':sum(c['mean_tip_world_error_m']<=.01 for c in fk['cases']),'guard_basis':'Historical researcher-selected engineering guard; not paper-calibrated fidelity acceptance.','does_not_establish':fk['wrist_palm_policy_limitation'],'limits':fk['limitations']},{'claim':'Compiled scene collider representation and configuration checked for exact six original stored scenes, including geometry transforms, mesh hull representation, masks and exclusions.','receipt':'compiled_contract','reported_checks':comp['checks'],'reported_pass':comp['passed'],'indices':[c['source_index'] for c in comp['cases']],'simulator_version':comp['mujoco_version'],'recorded_static_forward_calls':comp['actual_static_forward_calls'],'recorded_physics_steps':comp['physics_steps'],'recorded_hand_object_distance_queries':comp['new_hand_object_distance_queries'],'does_not_establish':'No fresh pairwise contact/dynamics verification; no proof that visual/compiled/source surfaces coincide; no exposed-union equivalence to simulator contacts.'}], 'unknown_or_unsupported':['Source native geometry and Shadow morphology/contact correspondence are not exactly preserved; historical nine strict failures remain.','The declared source wrist-to-Shadow-palm anchoring is numerically implemented but not an established anatomical identity.','Literal-part, exposed-union, visible-object and compiled-object measures describe different geometric domains; reference diagnosis is ongoing.','Static realization does not establish dynamic tracking or native MANO versus Shadow outcome equivalence.','No genuinely untouched object-level validation set has been established within this scoped saved-proposal inventory.','No generalization or failure-prediction adequacy has been demonstrated by these static receipts.'],'claim_levels':{'C1_declared_pipeline':'PARTLY_SUPPORTED_IN_LISTED_STATIC_SCOPE; a complete frozen pipeline readiness receipt is not issued here','C2_historical_strict_native_preservation':'FAIL_UNCHANGED','C3_approximate_native_fidelity':'NOT_ASSESSED_NO_JUSTIFIED_BUDGET','C4_Shadow_prediction_and_generator_diversity':'NOT_TESTED','C5_native_outcome_equivalence':'NOT_ESTABLISHED_NOT_REQUIRED_FOR_C1_OR_C4'},'read_only_replay':{'new_FK':0,'new_geometry_queries':0,'physics_steps':0,'new_inference':False,'new_eval_data':False}}
write('current_claim_manifest.json',manifest)
# Additional pickle file names prevent conflating raw observations with saved proposals.
allpaths=[Path(p) for p in subprocess.check_output(['rg','--files','-uuu',str(MAIN/'runs'),str(ROOT/'runs')],text=True).splitlines() if p.endswith('.pkl') and not p.endswith('/proposal.pkl')]
obs=[p for p in allpaths if '/hug/data/hug_bench/' in str(p) or '/hug_observation_pilot/medium_1.pkl' in str(p)]
apple=[p for p in allpaths if '/hug_benchmark_review_2026-10-09/apple/' in str(p)]
rep=MAIN/'runs/hug_benchmark_review_2026-10-09/review_report.json';r=read(rep)
assert len(apple)==10 and len(r['all_sample_grasp_checks'])==10 and all(c['grasp_withheld'] for c in r['all_sample_grasp_checks'])
assert len(obs)==19
up=MAIN/'runs/hug_integration_check/hug/README.md'
walk=ROOT/'docs/archive/HUG_STEP_1_WALKTHROUGH.md'
write('extra_file_disposition.json',{'schema':'saved_nonproposal_pickle_disposition_v1','scope':'File-name inventory in same two runs roots; roles from saved reports/upstream README, no deserialization. Not an exhaustive home-directory search.','nonproposal_pickle_count':len(allpaths),'by_basename':dict(collections.Counter(p.name for p in allpaths)),'upstream_observation_examples':{'count':len(obs),'classification':'18 upstream scene observation examples plus copied medium_1; requires new authorized inference to obtain HUG proposals, not an existing grasp holdout.','files':[{'path':str(p),'sha256':sha(p)} for p in sorted(obs)],'README_binding':{'path':str(up),'sha256':sha(up),'lines':'Usage says sample inputs of one image from each HUG-Bench scene; app --save-pred creates predictions.'}},'apple_validation_inputs':{'count':10,'saved_report_grasp_labels_withheld':True,'files':[{'path':str(p),'sha256':sha(p)} for p in sorted(apple)],'report_binding':{'path':str(rep),'sha256':sha(rep)},'historical_walkthrough_binding':{'path':str(walk),'sha256':sha(walk)},'classification':'Observation inputs with grasp labels withheld, not saved generated native grasps; apple geometry already part of development.'},'other_file_disposition':'Remaining files named simulation.pkl/cube.pkl/MANO/hand labels or unrelated WidowX carrot recordings were not promoted to HUG native proposal holdouts. Existing source inventory identifies proposal.pkl files separately. Unclassified pickle contents are not asserted to contain no useful data.','new_inference':False,'new_geometry_queries':0})
# Stronger documentary guards; scientific thresholds remain unset.
s=read(OUT/'case_record.schema.json')
s['allOf'][0]['then']['properties']['metrics']={'minItems':1}
s['properties']['prediction_adequacy']['allOf']=[{'if':{'properties':{'status':{'const':'ASSESSED_WITH_LIMITATIONS'}}},'then':{'properties':{'analysis_protocol_sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'},'heldout_evidence_sha256':{'type':'string','pattern':'^[a-f0-9]{64}$'}}}}]
s['properties']['task_outcome']['allOf'][1]['then']['properties']['label']={'enum':['TASK_SUCCESS','TASK_FAILURE','INDETERMINATE']}
write('case_record.schema.json',s);Draft202012Validator.check_schema(s)
# Produce synthetic records recursively only to test schema gates; these are never eval data.
def base(s):
 if 'const' in s:return s['const']
 if 'enum' in s:return s['enum'][0]
 if 'anyOf' in s:
  return next((None for x in s['anyOf'] if x.get('type')=='null'),base(s['anyOf'][0]))
 t=s.get('type')
 if isinstance(t,list):return None if 'null' in t else base({**s,'type':t[0]})
 if t=='object':return {k:base(s.get('properties',{}).get(k,{})) for k in s.get('required',[])}
 if t=='array':return [base(s.get('items',{})) for i in range(s.get('minItems',0))]
 if t=='boolean':return False
 if t=='number' or t=='integer':return 0
 if t=='string':return '0'*64 if 'pattern' in s else 'SYNTHETIC_SCHEMA_CONTROL'
 return {}
x=base(s);x['correctness']['status']='NOT_TESTED';x['historical_strict']['status']='NOT_RUN';x['historical_strict']['fresh_or_prior']='NOT_RUN';v=Draft202012Validator(s);tests=[]
def check(name,data,valid):
 errors=list(v.iter_errors(data));assert (not errors)==valid,(name,[e.message for e in errors]);tests.append({'name':name,'expected_valid':valid,'passed':True})
check('unassessed synthetic record valid',x,True)
y=copy.deepcopy(x);y['task_outcome']['label']='TASK_SUCCESS';check('no execution cannot have success label',y,False)
y=copy.deepcopy(x);y['task_outcome']['execution_status']='MAPPING_REFUSED';y['task_outcome']['label']='TASK_FAILURE';check('mapping refusal is not binary task failure',y,False)
y=copy.deepcopy(x);y['task_outcome']['execution_status']='EXECUTED';y['task_outcome']['label']='TASK_FAILURE';check('executed label needs execution and pose hashes',y,False)
y=copy.deepcopy(x);y['task_outcome'].update(execution_status='EXECUTED',label='TASK_FAILURE',execution_protocol_sha256='1'*64,executed_pose_sha256='2'*64);check('bound executed synthetic failure valid',y,True)
y=copy.deepcopy(x);y['approximate_native_fidelity']['status']='PASS';check('native fidelity cannot pass without budget and independent evidence',y,False)
y=copy.deepcopy(x);y['pipeline_readiness']['status']='STATIC_CHECKED_WITH_MEASURED_DISTORTION';check('pipeline readiness needs correctness and receipt',y,False)
y=copy.deepcopy(x);y['pipeline_readiness'].update(status='STATIC_CHECKED_WITH_MEASURED_DISTORTION',receipt_sha256='3'*64);y['correctness']['status']='PASS';check('measured distortion readiness needs metric records',y,False)
y['metrics']=[base(s['properties']['metrics']['items'])];check('scoped pipeline can pass with no native fidelity budget',y,True)
y=copy.deepcopy(x);y['prediction_adequacy']['status']='ASSESSED_WITH_LIMITATIONS';check('prediction assessment needs protocol and heldout receipts',y,False)
y=copy.deepcopy(x);y['historical_strict']['rule_sha256']='9'*64;check('historical scientific rule identity cannot be rebased',y,False)
write('schema_self_validation.json',{'passed':True,'validator':'jsonschema Draft202012Validator','schema_sha256':sha(OUT/'case_record.schema.json'),'controls':tests,'scope':'Synthetic schema checks only; no actual pose evaluated or example written into eval data.','synthetic_case_records_persisted':False,'new_geometry_queries':0})
p=read(OUT/'protocol_design.json');p['implementation_order_now'][-1]='Keep optional approximate-native-fidelity acceptance unset until justified budget and independent evidence exist; C1 pipeline correctness and C4 Shadow prediction claims follow their separate requirements.';p['current_claim_manifest_sha256']=sha(OUT/'current_claim_manifest.json');p['extra_saved_file_disposition_sha256']=sha(OUT/'extra_file_disposition.json');write('protocol_design.json',p)
print(json.dumps({'claim_manifest':str(OUT/'current_claim_manifest.json'),'native_contract_cases':32,'joint_limit_comparisons':len(alljoint),'mean_tip_guard_passes':manifest['supported'][1]['cases_meeting_existing_mean_tip_guard'],'schema_controls':len(tests),'maximum_existing_mean_tip_error_m':fknums['mean_tip_world_error_m'],'maximum_existing_individual_tip_error_m':fknums['maximum_tip_world_error_m'],'new_geometry':False},indent=2))
