from pathlib import Path
import gzip,hashlib,json
import numpy as np

ROOT=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1')
OUT=Path(__file__).parent
DIAG=Path('/private/tmp/hug_free_palm_review_2026-10-15_v1/diagnostic_selection/compact_review_table.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
files=[ROOT/'candidate_cohort_8_16_24.json',ROOT/'candidate_cohort_12_20_28.json',ROOT/'protocol.json',DIAG]
initial={str(p):sha(p.read_bytes()) for p in files}
cases={c['source_inventory_index']:c for f in files[:2] for c in json.loads(f.read_text())['cases']}
rows={r['source_inventory_index']:r for r in json.loads(DIAG.read_text())['table']}
results=[]
checks=0
def check(b):
 global checks
 assert b
 checks+=1
for i,seq,seed_index,shift in [(20,19,17,[0.,0.,-.01]),(24,17,15,[0.,-.01,0.]),(28,17,15,[0.,-.01,0.])]:
 c=cases[i];r=rows[i]
 check(r['selected_method']=='free_palm_v1' and r['callback_sequence']==seq)
 cb=c['keyed_direction_evidence']['callbacks'][seq-1]
 check(cb['callback_sequence']==seq)
 x=np.asarray(cb['state_scaled'],dtype=np.float64)
 candidates=[p for p in c['solver_audit']['proposals'] if p.get('candidate_guard',{}).get('metadata_callback_sequence')==seq]
 check(len(candidates)==1)
 p=candidates[0]
 check(np.asarray(p['candidate_state_scaled'],dtype=np.float64).tobytes()==x.tobytes())
 check(p['direction']=='predeclared_seed' and p['sweep']==0 and p['seed_index']==seed_index)
 proposal=np.asarray(p['proposal_state_scaled'],dtype=np.float64)
 check(proposal.shape==(27,) and np.array_equal(proposal[:24],np.zeros(24)))
 check(np.array_equal(proposal[24:]*.005,np.asarray(shift)))
 check(np.asarray(c['solver_audit']['predeclared_seed_state_scaled'][seed_index],dtype=np.float64).tobytes()==proposal.tobytes())
 tracepath=Path(cb['trace_path']); compressed=tracepath.read_bytes();raw=gzip.decompress(compressed)
 check(sha(compressed)==cb['trace_compressed_sha256'])
 check(sha(raw)==cb['trace_uncompressed_sha256'])
 trace=json.loads(raw)
 check(trace['callback_sequence']==seq and np.asarray(trace['state_scaled'],dtype=np.float64).tobytes()==x.tobytes())
 check(sha(x.tobytes())==trace['q_float64_sha256']==p['candidate_guard']['q_float64_sha256'])
 residual=np.asarray(trace['residuals'],dtype=np.float64)
 check(sha(residual.tobytes())==trace['residuals_float64_sha256']==p['candidate_guard']['residuals_float64_sha256'])
 fs=c['solver_audit']['config']['f_scale'];cost=float(fs*np.sum(np.hypot(fs,residual)-fs))
 check(cost==p['full_assessment']['objective']==r['candidate_objective'])
 tips=np.asarray(trace['fingertip_error_vectors_m'],dtype=np.float64)
 check(sha(tips.tobytes())==trace['fingertips_float64_sha256']==p['candidate_guard']['fingertips_float64_sha256'])
 norms=np.sqrt(np.einsum('ij,ij->i',tips,tips));caps=np.asarray(c['authoritative_original_tip_caps_sqrt_einsum_m'])
 check(bool(np.all(norms<=caps)) and float(np.mean(norms))<.010)
 check(np.array_equal(x[24:]*.005,np.asarray(r['palm_displacement_world_m'])))
 check(p['refusal']=='candidate_guard_refused' and not p['candidate_guard']['accepted'])
 results.append({
  'source_inventory_index':i,'callback_sequence':seq,'proposal_sequence':p['sequence'],'seed_index_zero_based':seed_index,'seed_label':c['solver_audit']['seed_labels'][seed_index],
  'proposed_palm_displacement_world_m':shift,'requested_displacement_norm_m':float(np.linalg.norm(proposal[24:]*.005)),
  'final_palm_displacement_world_m':(x[24:]*.005).tolist(),'final_displacement_norm_m':float(np.linalg.norm(x[24:]*.005)),
  'same_state_via_guard_metadata_and_callback_and_candidate_bytes':True,'restoration':p['restoration'],'original_caps_and_mean_pass_computed_arithmetic':True,
  'candidate_objective':cost,'extra_direction_guard_pass':False,'extra_direction_guard_refusals':p['candidate_guard']['refusal_reasons'],
  'trace_binding':{k:cb[k] for k in ['trace_path','trace_compressed_sha256','trace_uncompressed_sha256']},
  'intentional_nonzero_translation_before_restoration':True,'no_new_selection_or_ranking':True,
 })
for p in files:check(sha(p.read_bytes())==initial[str(p)])
result={'schema':'hug_existing_three_intentional_palm_provenance_check_v1','scope':'Read-only verification of three previously chosen diagnostics. No fitting, ranking, new geometry queries, physics or eval data.','passed':True,'checks':checks,'source_bindings':initial,'script_sha256':sha(Path(__file__).read_bytes()),'cases':results,'limitations':'Verifies saved computed arithmetic and exact provenance; not an independent source/FK or surface rerun. Original guard refusals and fallback remain unchanged.'}
target=OUT/'existing_three_provenance.json';target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(target),'sha256':sha(target.read_bytes()),'checks':checks,'passed':True}))
