"""Freeze one explicit-seed coverage supplement before any actual rank."""
from pathlib import Path
import datetime,hashlib,json
OUT=Path(__file__).parent
OLD=Path('/private/tmp/hug_free_palm_review_2026-10-15_v1/diagnostic_selection')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert not (OUT/'execution_started.json').exists() and not (OUT/'protocol.json').exists()
old=read(OLD/'protocol.json');inputs=dict(old['input_sha256'])
controls=read(OUT/'synthetic_policy_checks.json');assert controls['passed'] and controls['count']==14
for path in (OUT/'select_explicit_proposal.py',OUT/'test_policy.py',OUT/'synthetic_policy_checks.json',Path(__file__),OLD/'protocol.json',OLD/'policy.json',OLD/'candidate_targets.json',OLD/'selection_report.json',Path('/private/tmp/hug_free_palm_fit_tiny_shift_review_2026-10-15.json'),Path('/private/tmp/hug_free_palm_intentional_seed_provenance_2026-10-15_v1/existing_three_provenance.json')):
    inputs[str(path)]=sha(path)
assert inputs['/private/tmp/hug_free_palm_fit_tiny_shift_review_2026-10-15.json']=='dc3d7f1a451652256cca963c65187ccd9c50f3c63d116967c964ba361d4a334a'
for path,expected in inputs.items():assert sha(path)==expected,path
policy={
 'schema':'hug_explicit_translation_seed_coverage_policy_v1',
 'freeze_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'case_indices':[8,12,16],
 'reason':'Original no-minimum-shift diagnostic policy selected joint-only THJ5 proposal descendants with1.165--3.284nm restoration drift for these three cases. User asks per-grasp palm-position tests. Root authorized ONE separate provenance-based coverage supplement, retaining original policy/results.',
 'timing':'Post-hoc coverage correction after observing original selected poses/provenance; not held-out evidence. No supplemental actual ranking/FK/surface measurement occurred before freeze. Original surface audit runs independently and its results do not enter eligibility or ranking.',
 'eligibility':'Only fully scored candidates descending from frozen predeclared_seed proposals in V1(26) or V2(56). Require exact seed-index/state equality, seed-stage metadata, original24joint coordinates in requested seed, explicitly nonzero requested world translation BEFORErestoration, actual final nonzero displacement, literal original5caps+10mmmean+originaldomain. No positive minimum final movement or baseline objective improvement threshold.',
 'provenance':'Bind immutable proposal candidate_guard metadata_callback_sequence to full callback index/gzip/raw/digests; require full27state, full tip vectors, objective and guard decision/refusals match exactly. Retain complete generating proposal and seed label.',
 'comparability':'Same protocol configurations, percase native/source/targets/sites/weights/baseline/domain; complete objective recomputed asf_scale*sum(hypot(f_scale,residuals)-f_scale).',
 'selection':'Minimum comparable objective, then original method order(V1,V2), then callbacksequence, then earliest explicit generating proposal. Exactstate dedup retains earliest matching origin and all additional eligible origins. No selection using surface acceptance, orientation witnesses, physical outcomes or post-freeze adjustments.',
 'independent_FK':'Reuse already frozen independent source/action/scene FK helper, require literal historicalcaps, strict10mmmean, fixed originalroot quaternion, qdependent palm compensation, unchangedobjects/zero time and callback reproduction.',
 'scientific_scope':'Supplemental three diagnoses use unchanged existing5grids. Keep previous9diagnostic audits separate; do not rewrite primary fallback or extra direction refusals; no production promotion. All are previously inspected development cases.',
 'stopping_rule':'Exactly one selection supplement; no further fitting, tuning or selection loops regardless outcome.',
 'new_eval_data':False,'new_fitting':False,'new_surface_measurement_by_selector':False,'new_inference':False,'physics_steps':0,
 'script_sha256':sha(OUT/'select_explicit_proposal.py'),'synthetic_policy_checks_sha256':sha(OUT/'synthetic_policy_checks.json'),
 'scientific_acceptance_rule_sha256':old['scientific_acceptance_rule_sha256']
}
with (OUT/'policy.json').open('x') as f:json.dump(policy,f,indent=2);f.write('\n')
inputs[str(OUT/'policy.json')]=sha(OUT/'policy.json')
protocol={'schema':'hug_explicit_translation_seed_selection_inputs_v1','freeze_utc':policy['freeze_utc'],'policy_path':str(OUT/'policy.json'),'policy_sha256':sha(OUT/'policy.json'),'script_sha256':sha(OUT/'select_explicit_proposal.py'),'input_sha256':inputs,'methods':old['methods'],'case_indices':[8,12,16],'inventory_path':old['inventory_path'],'historical_candidate_path':old['historical_candidate_path'],'source_urdf':old['source_urdf'],'original_diagnostic_candidate_sha256':sha(OLD/'candidate_targets.json'),'scope':'One post-hoc explicit seed-origin coverage supplement before ranking; previous selected diagnostics and both primary fit outputs unchanged.','no_more_selection_loops':True,'new_eval_data':False,'physics_steps':0}
with (OUT/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
print(json.dumps({'policy_sha256':sha(OUT/'policy.json'),'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(OUT/'select_explicit_proposal.py'),'input_count':len(inputs)}))
