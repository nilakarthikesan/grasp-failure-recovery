"""Pure saved-array/hash freeze; no models, FK, distance or normal queries."""
from pathlib import Path
import gzip,hashlib,json,numpy as np,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(gzip.decompress(Path(p).read_bytes()) if Path(p).suffix=='.gz' else Path(p).read_bytes())
bridge=read('/private/tmp/hug_current_execution_bridge_2026-10-15_v1/protocol.json')
invpath=Path(bridge['inventory_path']);inventory=read(invpath)
savedpath=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json');saved=read(savedpath)
prior=read('/private/tmp/hug_free_palm_fit_2026-10-15_v1/protocol.json')
bindings={}
def bind(path):
 p=Path(path).resolve();assert p.is_file(),p;bindings[str(p)]=sha(p)
# Rebind actual current dependencies; do not copy stale prior hashes as a current claim.
for path in bridge['bindings']:
 if Path(path).is_file():bind(path)
for path in (ROOT/'src').rglob('*.py'):bind(path)
for path in (ROOT/'src').rglob('*.yaml'):bind(path)
for path in HERE.glob('*.py'):bind(path)
for path in ('/private/tmp/hug_feasibility_restoration_2026-10-15_v1/exact_direction.py','/private/tmp/hug_feasibility_restoration_2026-10-15_v1/constraint_core.py','/private/tmp/hug_feasibility_restoration_2026-10-15_v1/geometry_helpers.py','/private/tmp/hug_current_scene_bridge_2026-10-15_v1/scene_runtime.py','/private/tmp/hug_current_scene_bridge_2026-10-15_v1/native_reference.py','/private/tmp/hug_current_execution_bridge_2026-10-15_v1/xml_reference.py') :bind(path)
orientation=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json'
for path in (orientation,orientation.with_name('check_orientation.py'),ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py',savedpath,invpath):bind(path)
# Saved-row extraction already binds native caches; retain all existing source pointers.
for path,h in saved['input_sha256'].items():assert sha(path)==h;bind(path)
oldreport=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json');bind(oldreport)
poses=[];order=[20,28,8,12,16,24]
for idx in order:
 source=inventory['modern_cases'][idx];old=next(c for c in bridge['cases'] if c['source_inventory_index']==idx)
 for path,h in old['input_bindings'].items():assert sha(path)==h;bind(path)
 bind(source['resolved_case_path'])
 # Resolve all object MJCF/mesh files consumed by current registered runner.
 payload=read(source['resolved_case_path'])['case'];geometry=payload['object'].get('geometry')
 if geometry:
  mjcf=(BASE/geometry['mjcf_path']).resolve();bind(mjcf);xml=ET.parse(mjcf).getroot();compiler=xml.find('compiler');meshdir=compiler.get('meshdir','.')
  for mesh in xml.findall('asset/mesh'):bind(mjcf.parent/meshdir/mesh.get('file'))
 cap=prior['prior_original_cap_bindings'][str(idx)];assert sha(cap['path'])==cap['sha256'];bind(cap['path'])
 capvalues=read(cap['path'])['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m']
 pi=[i for i,r in enumerate(saved['whole_cell_rows']) if r['source_inventory_index']==idx]
 oi=[i for i,r in enumerate(saved['orientation_rows']) if r['source_inventory_index']==idx]
 common={'source_inventory_index':idx,'native_sha256':old['native_sha256'],'joint_names':old['joint_names'],
 'original_qpos_rad':old['qpos_rad'],'historical_tip_caps_m':capvalues,'historical_cap_report':cap['path'],
 'proximity_source_indices':pi,'orientation_source_indices':oi,'proximity_count':len(pi),'orientation_count':len(oi),
 'expected_object_collider_count':{8:2,12:2,16:7,20:7,24:15,28:15}[idx]}
 poses.append({**common,'pose_id':f'original_{idx}','role':'original_current_nominal','qpos_rad':old['qpos_rad'],
 'root_position_m':old['root_position_m'],'root_quaternion_wxyz':old['root_quaternion_wxyz'],'historical_witness_report':str(oldreport)})
 if idx in (20,28):
  candidate=Path(f'/private/tmp/hug_feasible_eligibility_2026-10-15_v1/pilot/{idx}/result.json');bind(candidate);j=read(candidate)
  assert j['native_sha256']==old['native_sha256']
  q=np.asarray(old['qpos_rad'])+.025*np.asarray(j['returned_state_scaled'])[:24]
  poses.append({**common,'pose_id':f'experimental_{idx}','role':'latest_eligibility_guidance_saved_command','qpos_rad':q.tolist(),
   'root_position_m':j['returned_exact']['root_target_position_m'],'root_quaternion_wxyz':j['returned_exact']['root_target_quaternion_wxyz'],
   'candidate_path':str(candidate),'candidate_sha256':sha(candidate),'historical_witness_report':str(candidate)})
for path in (HERE/'manufactured_controls.log',HERE/'summary_controls.log',HERE/'import_preflight.json'):
 if path.exists():bind(path)
p={'schema':'hug_current_geometry_fixed_witness_protocol_v1','status':'DRAFT_NOT_EXECUTABLE','bindings':bindings,
 'inventory':str(invpath),'saved_witnesses':str(savedpath),'orientation_protocol':str(orientation),'source_urdf':bridge['source_urdf'],
 'runtime_project_root':str(BASE),'current_pose_source':str(ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py'),
 'poses':poses,'runtime':{'numpy':'2.2.6','mujoco':'3.3.7','numba_threads':2,'executable':str(BASE/'.venv/bin/python'),'launch':'env -u PYTHONPATH <executable> launch.py; NumPy/MuJoCo first, legacy dependency paths appended'},
 'frame_assertion_m':1e-12,'rotation_assertion':1e-12,'expected_hand_collider_count':45,
 'finite_budget':{'poses':8,'unique_existing_native_grasps':6,'proximity_rows':178,'orientation_rows':134,'retained_row_evaluations':sum(x['proximity_count']+x['orientation_count'] for x in poses),'maximum_static_forwards':70,'optimizations':0,'native_geometry_queries':0,'physics_steps':0},
 'scope':'Current complete representation witnesses: six original named q commands use actual current native-to-palm helper and root inversion, with independent XML/native frame checks and historical root delta. Two latest experimental q/root commands replay literally. No new retargeting solve or anatomical-equivalence certificate.',
 'geometry':'Current registered ActuatedShadowRunner via current MjSpec. Same-label individual component surfaces plus all45 active hand colliders including forearm/wrist in exact nearest/eligibility. Frozen old object visual triangles used only as the unchanged criterion reference; current object frame checked.',
 'historical_reference':'Same immutable saved source rows/queries/native signs and bounds. Existing historical measurements compared; original comparison includes frame and geometry changes, experimental comparison holds literal root fixed. No rerun of old hand queries.',
 'tip_policy':'Report literal original tip caps and inclusive10mm mean on current FK, without rebasing or discarding a pose for old cap failure. Diagnostic reporting is not acceptance.',
 'coverage':'178 retained original whole-cell examples out of44558 historical failed tuples, plus134 repeated orientation evidence rows; fixed312 rows across6 cases. Experimental20/28 repeat53 rows. No full-grid or unseen-grasp claim.',
 'unchanged_scientific_predicates':'Exact inherited constraint_core proximity lower/upper and native-direction eligibility with original bands, signs, margins, tie/burial/object checks; numerical frame gates are engineering assertions only.',
 'new_eval_data':False,'new_native_grasps':False,'production_promotion':False}
(HERE/'protocol.draft.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
print(json.dumps({'draft_sha256':sha(HERE/'protocol.draft.json'),'bindings':len(bindings),'poses':len(poses),'rows':p['finite_budget']['retained_row_evaluations']}))
