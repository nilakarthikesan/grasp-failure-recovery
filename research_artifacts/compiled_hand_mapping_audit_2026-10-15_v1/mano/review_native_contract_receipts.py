from pathlib import Path
import hashlib,json
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');P=Path('/Users/nilakarthikesan/grasp-failure-prediction');O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
hp=W/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/native_contract/native_contract.json';h=read(hp)
assert len(h['cases'])==32
for r in h['cases']:
 assert sha(r['native_path'])==r['native_sha256'] and r['passed']
 assert r['max_vertex_error_m']==r['max_landmark_error_m']==r['max_rotation_element_error']==0 and r['closed_face_indices_equal']
rows=[]
fixtures=read(O/'target_cases_v1.json')['cases']
for f in fixtures:
 if f['generator']!='ContactGen':continue
 n=Path(f['native_fk_reference']['path']);p=n.parent.parent/'native_reconstruction.json'
 if f['case_id']=='smoke_cube_seed42':p=P/'runs/contactgen_native_run_2026-10-14_v1/download_cn4nexx8e9fxjl/contactgen_diagnostics/native_reconstruction.json'
 r=read(p);s=r['samples'][0]
 assert r['bindings']['native_bundle']['sha256'].removeprefix('sha256:')==sha(n)
 assert r['status']=='PASS' and s['numerical_replay_status']=='PASS'
 for key in ['vertices_error_m','full_frames_error_mixed_units','wrist_error_m']:
  assert s[key]['exact_array_equal_float32'] and s[key]['max_abs']==0
 rows.append({'case_id':f['case_id'],'reconstruction_report':{'path':str(p),'sha256':sha(p)},'native_bundle_sha256':sha(n),'all778_vertices_and16frames_exact_float32':True,'shaped_root_plus_translation_exact_float32':True})
sources=[P/'runs/hug_integration_check/hug/src/models/mano.py',P/'runs/hug_integration_check/hug-deps/manotorch/manolayer.py',W/'research_artifacts/contactgen_mapping_validation_2026-10-14_235056/source_reference/manopth/manolayer.py',W/'src/grasp_failure_prediction/proposals/contactgen.py',W/'src/grasp_failure_prediction/evaluation/retargeting.py',W/'src/grasp_failure_prediction/integrations/hug_frames.py']
out={'schema_version':'mano_native_representation_receipt_readback_v1','readback_only':True,'new_mano_forward_calls':0,'native_generator_calls':0,'retarget_optimizer_calls':0,'physics_steps':0,'source_bindings':{str(p):sha(p) for p in sources},'hug':{'cases':32,'report':{'path':str(hp),'sha256':sha(hp)},'all32_native_3d_exports_exact_reconstruction':True,'configuration':'right MANO, full 15x3 axis-angle derived from6D, no PCA, flat mean, wrist center0, shape10, meters, 778 vertices/1552 closed faces','source_translation_semantics':'wrist-to-camera translation; native landmarks[0] and wrist transform origin agree','adapter_uses_saved_native_3d_geometry':True},'contactgen':{'cases':13,'configuration':'right MANO PCA26, nonflat mean, shape10, meters,778 vertices/1538 faces/open wrist','source_translation_semantics':'parameter translation added to shaped root; frame0 origin includes shaped J0(beta). Adapter correctly uses frame0 rather than translation parameter.','centering_semantics':'restore arithmetic processed-mesh vertex mean before applying mesh-to-object placement','landmark_semantics':'16 source bone-frame origins plus right-hand vertex tips745/317/444/556/673, reorder wrist/thumb/index/middle/ring/little','receipts':rows},'interpretation':'Native representation reconstruction and adapter coordinate conventions are supported by saved byte-bound source receipts and current independent frame/FK checks. MANO-to-Shadow anatomical or contact equivalence does not follow. No new source-hand reconstruction or source model execution was needed.'}
p=O/'native_contract_receipt_review_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'hug':32,'contactgen':13}))
