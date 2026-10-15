from pathlib import Path
import gzip,hashlib,json
import numpy as np

OUT=Path(__file__).parent
DIAG=Path('/private/tmp/hug_explicit_palm_selection_2026-10-15_v1/candidate_targets.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
checkcount=0
def check(b):
 global checkcount
 assert b
 checkcount+=1
check(sha(DIAG.read_bytes())=='4b428a5247dfc0d9ce342e333a404af31826e0af48771e9bcde5057afc166ba9')
selection=json.loads(DIAG.read_text());rows={r['source_inventory_index']:r for r in selection['cases']}
expected={8:('fine_translation_v2',2,53,52,51,.00015625,[1,-1,-1]),12:('free_palm_v1',1,23,22,21,.005,[1,-1,-1]),16:('fine_translation_v2',2,50,49,48,.00015625,[1,1,1])}
files=[DIAG,DIAG.parent/'protocol.json',DIAG.parent/'policy.json'];check(set(rows)==set(expected))
check(sha((DIAG.parent/'protocol.json').read_bytes())==selection['protocol_sha256'])
for v in [1,2]:
 root=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v'+str(v))
 files.extend([root/'candidate_cohort_8_16_24.json',root/'candidate_cohort_12_20_28.json',root/'protocol.json'])
initial={str(p):sha(p.read_bytes()) for p in files}
results=[]
for i,(method,v,seq,pseq,seed_index,length,signs) in expected.items():
 root=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v'+str(v));cohort=root/('candidate_cohort_12_20_28.json' if i==12 else 'candidate_cohort_8_16_24.json')
 source=json.loads(cohort.read_text());c=next(c for c in source['cases'] if c['source_inventory_index']==i);r=rows[i]
 check(source['protocol_sha256']==sha((root/'protocol.json').read_bytes()))
 check(r['selection_source']['method_name']==method and r['selection_source']['callback_sequence']==seq and r['selection_source']['proposal_sequence']==pseq)
 cb=c['keyed_direction_evidence']['callbacks'][seq-1];check(cb['callback_sequence']==seq)
 x=np.asarray(cb['state_scaled'],dtype=np.float64)
 candidates=[p for p in c['solver_audit']['proposals'] if p.get('candidate_guard',{}).get('metadata_callback_sequence')==seq and p['sequence']==pseq]
 check(len(candidates)==1);p=candidates[0]
 check(np.asarray(p['candidate_state_scaled'],dtype=np.float64).tobytes()==x.tobytes()==np.asarray(r['selected_state_scaled'],dtype=np.float64).tobytes())
 check(p['direction']=='predeclared_seed' and p['sweep']==0 and p['seed_index']==seed_index)
 proposal=np.asarray(p['proposal_state_scaled'],dtype=np.float64)
 check(proposal.shape==(27,) and np.array_equal(proposal[:24],np.zeros(24)))
 check(bool(np.any(proposal[24:]!=0.)) and bool(np.any(x[24:]!=0.)))
 check(np.array_equal(np.sign(proposal[24:]),np.asarray(signs)))
 check(abs(float(np.linalg.norm(proposal[24:]*.005))-length)<1e-18)
 check(np.asarray(c['solver_audit']['predeclared_seed_state_scaled'][seed_index],dtype=np.float64).tobytes()==proposal.tobytes())
 origin=r['explicit_translation_proposal_origin'];check(origin['complete_generating_proposal']==p)
 check(origin['proposal_seed_label']==c['solver_audit']['seed_labels'][seed_index])
 check(np.asarray(origin['proposal_state_scaled'],dtype=np.float64).tobytes()==proposal.tobytes())
 check(r['selection_source']['source_trace_binding']==cb)
 tracepath=Path(cb['trace_path']);compressed=tracepath.read_bytes();raw=gzip.decompress(compressed)
 check(sha(compressed)==cb['trace_compressed_sha256']);check(sha(raw)==cb['trace_uncompressed_sha256'])
 trace=json.loads(raw)
 check(trace['callback_sequence']==seq and np.asarray(trace['state_scaled'],dtype=np.float64).tobytes()==x.tobytes())
 check(sha(x.tobytes())==trace['q_float64_sha256']==p['candidate_guard']['q_float64_sha256'])
 residual=np.asarray(trace['residuals'],dtype=np.float64)
 check(sha(residual.tobytes())==trace['residuals_float64_sha256']==p['candidate_guard']['residuals_float64_sha256'])
 fs=c['solver_audit']['config']['f_scale'];cost=float(fs*np.sum(np.hypot(fs,residual)-fs))
 check(cost==p['full_assessment']['objective']==r['selection_source']['objective']==r['selected_fresh_diagnostics']['objective'])
 tips=np.asarray(trace['fingertip_error_vectors_m'],dtype=np.float64)
 check(sha(tips.tobytes())==trace['fingertips_float64_sha256']==p['candidate_guard']['fingertips_float64_sha256'])
 norms=np.sqrt(np.einsum('ij,ij->i',tips,tips));caps=np.asarray(c['authoritative_original_tip_caps_sqrt_einsum_m'])
 check(bool(np.all(norms<=caps)) and float(np.mean(norms))<.010)
 check(np.array_equal(caps,np.asarray(r['authoritative_original_tip_caps_sqrt_einsum_m'])))
 check(np.array_equal(x[24:]*.005,np.asarray(r['palm_displacement_world_m'])))
 check(np.array_equal(np.asarray(c['original_qpos_rad'])+.025*x[:24],np.asarray(r['qpos_rad'])))
 check(p['refusal']=='candidate_guard_refused' and not p['candidate_guard']['accepted'])
 check(r['diagnostic_pose_not_promoted'] and r['guarded_mapper_selected_fallback'])
 guard=r['extra_direction_guard'];sourceguard=p['candidate_guard']
 check(guard['accepted']==sourceguard['accepted'] and guard['refusal_reasons']==sourceguard['refusal_reasons'])
 for part,summary in guard['parts'].items():
  for key,value in summary.items():check(value==sourceguard['parts'][part][key])
 results.append({'source_inventory_index':i,'method':method,'callback_sequence':seq,'proposal_sequence':pseq,'seed_index_zero_based':seed_index,'seed_label':c['solver_audit']['seed_labels'][seed_index],'proposed_palm_displacement_world_m':(proposal[24:]*.005).tolist(),'requested_displacement_norm_m':float(np.linalg.norm(proposal[24:]*.005)),'final_palm_displacement_world_m':(x[24:]*.005).tolist(),'final_displacement_norm_m':float(np.linalg.norm(x[24:]*.005)),'same_state_via_guard_metadata_callback_candidate_and_selection_bytes':True,'restoration':p['restoration'],'original_caps_and_mean_pass_computed_arithmetic':True,'candidate_objective':cost,'extra_direction_guard_pass':False,'extra_direction_guard_refusals':p['candidate_guard']['refusal_reasons'],'trace_binding':{k:cb[k] for k in ['trace_path','trace_compressed_sha256','trace_uncompressed_sha256']},'intentional_nonzero_translation_before_restoration':True,'no_new_selection_or_ranking':True})
for p in files:check(sha(p.read_bytes())==initial[str(p)])
result={'schema':'hug_supplement_three_intentional_palm_provenance_check_v1','scope':'Read-only verification of3 reviewer-selected supplemental diagnostics. No fitting, ranking, new geometry queries, physics or eval data.','passed':True,'checks':checkcount,'source_bindings':initial,'script_sha256':sha(Path(__file__).read_bytes()),'cases':results,'checker_schema_adjustment':'Initial checker expected the compact independent guard summary to equal the full source dictionary. These deliberately have different schemas; compare all shared summary fields and full copied proposal instead. No scientific data or selected candidate changed.','limitations':'Verifies saved computed arithmetic and exact provenance; not an independent source/FK or surface rerun. Original guard refusals and fallbacks remain unchanged.'}
target=OUT/'supplement_three_provenance.json';target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(target),'sha256':sha(target.read_bytes()),'checks':checkcount,'passed':True}))
