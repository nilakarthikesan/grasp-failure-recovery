"""Independent saved-array and selection replay; no geometry or solver imports."""
from pathlib import Path
import gzip,hashlib,json,collections
import numpy as np
D=Path(__file__).resolve().parent;S=Path('/private/tmp/hug_se3_mapping_2026-10-15_v1');W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
protocol=read(S/'pilot_protocol.json');prior=read(protocol['pooled_candidates']);witness=read('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json');orientation=read(W/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json')
checks=collections.Counter();bindings={str(S/'pilot_protocol.json'):sha(S/'pilot_protocol.json')};out=[]
def check(kind,pred):
 assert pred,kind
 checks[kind]+=1
def equal(kind,a,b,eps=0):
 a=np.asarray(a);b=np.asarray(b);check(kind,a.shape==b.shape and (np.array_equal(a,b) if eps==0 else np.allclose(a,b,rtol=0,atol=eps)))
for p,h in protocol['bindings'].items():check('source hash',sha(p)==h)
for index in [20,28]:
 path=S/f'pilot/{index}/result.json';result=read(path);r=result['solver_receipt'];bindings[str(path)]=sha(path);trace=Path(r['trace_path']);bindings[str(trace)]=sha(trace)
 check('trace hash',sha(trace)==r['trace_sha256']);check('protocol hash',result['protocol_sha256']==sha(S/'pilot_protocol.json'))
 case=next(c for c in prior['cases'] if c['source_inventory_index']==index);caps=np.asarray(case['authoritative_original_tip_caps_sqrt_einsum_m']);bounds=np.r_[np.asarray(case['bounds_scaled']),np.tile([-np.pi/12/.025,np.pi/12/.025],(3,1))]
 prox=[a for a in witness['whole_cell_rows'] if a['source_inventory_index']==index];ori=[a for a in witness['orientation_rows'] if a['source_inventory_index']==index];P=len(prox)
 radius=np.array([a['row']['centroid_radius_m'] for a in prox]);allow=np.array([a['row']['numerical_allowance_m'] for a in prox]);band=np.array([a['row']['band_mm']/1000 for a in prox]);near=np.array([a['row']['relationship_change']=='lost_native' for a in prox]);sign=np.sign([a['row']['native_facing_compatibility'] for a in ori]);oband=np.array([a['band_mm']/1000 for a in ori]);dead=orientation['facing_dot_numerical_deadzone'];boundary=orientation['query_boundary_tolerance_m']
 with gzip.open(trace,'rt') as f:events=[json.loads(line) for line in f]
 by={e['event']:e for e in events};count=collections.Counter(e['kind'] for e in events);raw_by_state={};reference=None;eligible=[];major={};terminals={};max_normdiff=0.
 def key(x):return np.asarray(x,dtype=np.float64).tobytes()
 def rawcheck(v):
  nonlocal_placeholder=None
  x=np.asarray(v['state_scaled']);check('finite 30D state',x.shape==(30,) and np.isfinite(x).all())
  norms=np.sqrt(np.einsum('ij,ij->i',np.asarray(v['task_error_vectors_m'])[:5],np.asarray(v['task_error_vectors_m'])[:5]));equal('tip norms replay',norms,v['tip_status']['norms_m'],2e-17)
  # Gate remains literal using authoritative saved norms and original caps, not the replay allowance.
  recorded=np.asarray(v['tip_status']['norms_m']);ip=recorded<=caps;mp=float(recorded.mean())<=.010
  equal('individual cap predicates',ip,v['tip_status']['individual_pass']);check('mean cap predicate',mp==v['tip_status']['mean_pass']);check('tip predicate',bool(ip.all() and mp)==v['tip_status']['passed'])
  d=np.asarray(v['part_distance_m'])[:P];error=radius+allow;lo=np.maximum(0.,d-error);hi=d+error;margin=np.where(near,band-hi,lo-band);passed=np.where(near,hi<=band,lo>band)
  for n,a in [('lower_m',lo),('upper_m',hi),('margin_m',margin),('passed',passed)]:equal('proximity '+n,a,v['proximity'][n])
  equal('strict far boundary',(~near)&(lo==band),v['proximity']['strict_far_boundary'])
  equal('proximity raw surrogate',(np.where(near,margin,d-error-band)-1e-9)/.001,v['proximity']['solver_margin'])
  score=np.asarray(v['raw_score']);dist=np.asarray(v['part_distance_m'])[P:];sg=sign*score-dead;bg=oband-dist;qg=dist-boundary
  for n,a in [('sign_margin',sg),('band_margin_m',bg),('query_boundary_margin_m',qg),('raw_passed',(sg>0)&(dist<=oband)&(dist>boundary)),('solver_margin',np.r_[sg-1e-8,(bg-1e-9)/.001,(qg-1e-9)/.001])]:equal('raw orientation '+n,a,v['raw_orientation'][n])
  equal('surface surrogate assembly',np.r_[v['proximity']['solver_margin'],v['raw_orientation']['solver_margin'],(np.asarray(v['raw_nearest_part_margin_m'])-1e-9)/.001,(np.asarray(v['raw_query_exterior_margin_m'])-1e-9)/.001],v['surface_margin'])
  raw_by_state[key(x)]=v
 def exactcheck(v):
  rawcheck(v);x=np.asarray(v['state_scaled']);f=v['exact_orientation_fields'];score=np.asarray(f['score']);dist=np.asarray(f['distance_m']);valid=np.asarray(f['valid'],bool);sg=sign*score-dead;bg=oband-dist;qg=dist-boundary;op=(sg>0)&(dist<=oband)&(dist>boundary)&valid
  for n,a in [('sign_margin',sg),('band_margin_m',bg),('query_boundary_margin_m',qg),('passed',op)]:equal('exact orientation '+n,a,v['exact_orientation'][n])
  passed=np.r_[v['proximity']['passed'],op];margin=np.r_[np.asarray(v['proximity']['margin_m'])/.001,sg,bg/.001,qg/.001]
  equal('full literal predicates',passed,v['literal_surface_pass']);equal('exact rank margins',margin,v['exact_surface_margin_without_optimizer_buffers'])
  rank=[int(not passed.all()),int((~valid).sum()),max(0.,float(np.max(-margin,initial=0))),int((~passed).sum()),float(np.dot(x,x))]
  equal('rank first four components',rank[:4],v['rank'][:4]);equal('rank state norm',rank[4],v['rank'][4],1e-12)
  check('witness pass conjunction',bool(passed.all())==v['all_witness_constraints_pass']);check('exact tip and bounds',v['in_domain'] is True and v['tip_status']['passed'] is True and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]));equal('rotation decode',x[27:]*.025,v['world_root_delta_rotation_vector_rad'])
 for seq,e in enumerate(events,1):
  check('trace sequence',e['event']==seq);kind=e['kind']
  if kind=='raw_callback':
   rawcheck(e['value']);check('30D state hash',hashlib.sha256(key(e['value']['state_scaled'])).hexdigest()==e['state_sha256'])
  elif kind=='raw_cache_hit':
   src=by[e['source_event']];check('cache source type',src['kind']=='raw_callback' and src['event']<e['event']);equal('cache full state',e['state_scaled'],src['value']['state_scaled']);check('cache state hash',hashlib.sha256(key(e['state_scaled'])).hexdigest()==e['state_sha256'])
  elif kind=='optimizer_major_iterate':major[(e['start_index'],e['major_iteration'])]=e['state_scaled']
  elif kind=='optimizer_terminal':terminals[e['start_index']]=e['terminal_state_scaled']
  elif kind=='constraint_jacobian':
   x=np.asarray(e['state_scaled']);base=raw_by_state[key(x)];mat=np.asarray(e['matrix']);check('Jacobian shape',mat.shape==(6+len(base['surface_margin']),31));equal('Jacobian mean tip row',mat[5,:30],mat[:5,:30].mean(0),2e-15);equal('epigraph columns',mat[:6,30],np.zeros(6));equal('surface slack column',mat[6:,30],np.ones(len(base['surface_margin'])))
   for j,span in enumerate(e['one_sided_spans_scaled']):
    probe=x.copy();probe[j]+=span;v=raw_by_state[key(probe)];equal('surface FD column',(np.asarray(v['surface_margin'])-base['surface_margin'])/span,mat[6:,j])
  elif kind=='selection_refusal_before_orientation':
   v=raw_by_state[key(e['state_scaled'])];equal('prefilter tip status',v['tip_status']['individual_pass'],e['tip_status']['individual_pass']);check('prefilter real refusal',not e['in_domain'] or not e['tip_status']['passed']);check('orientation not computed',e['orientation_not_computed'] is True)
  elif kind=='exact_checkpoint':
   v=e['value'];exactcheck(v)
   if reference is None:check('first is pooled reference',e['reason']=='fixed_pooled_selection_reference');reference=v
   valid=np.asarray(v['exact_orientation_fields']['valid']);oldvalid=np.asarray(reference['exact_orientation_fields']['valid']);passed=np.asarray(v['literal_surface_pass']);oldpass=np.asarray(reference['literal_surface_pass']);protected=bool(valid[oldvalid].all() and passed[oldpass].all())
   s=next(a for a in r['exact_checkpoint_summaries'] if a['event']==e['event']);check('protected mask replay',protected==s['preserves_reference_eligible_and_passing_rows']);equal('checkpoint rank',v['rank'],s['rank']);equal('checkpoint state',v['state_scaled'],s['state_scaled'])
   reason=e['reason'];x=v['state_scaled']
   if reason=='fixed_pooled_selection_reference' or reason=='start_1':equal('pooled start identity',x,list(case['selected_state_scaled'])+[0.,0.,0.])
   elif reason=='original_baseline' or reason=='start_0':equal('original start identity',x,np.zeros(30))
   elif '_major_' in reason:
    a,b=reason.split('_major_');equal('major source identity',x,major[(int(a.split('_')[1]),int(b))])
   elif reason.endswith('_terminal'):equal('terminal source identity',x,terminals[int(reason.split('_')[1])])
   else:raise AssertionError('Unexpected selectable source '+reason)
   if protected:eligible.append(e)
  elif kind=='protected_reference_refusal':check('protected refusal exists',not next(a for a in r['exact_checkpoint_summaries'] if a['event']==e['source_event'])['preserves_reference_eligible_and_passing_rows'])
  elif kind=='selected_fresh_terminal':exactcheck(e['value']);check('fresh selected source exists',e['selected_source_event'] in by and by[e['selected_source_event']]['kind']=='exact_checkpoint')
 winner=min(eligible,key=lambda e:tuple(e['value']['rank']));check('stable lexicographic selection',winner['event']==r['selected_source_exact_checkpoint']['event']);check('eligible count',len(eligible)==r['counts']['selection_eligible_exact_checks']);check('no fresh fallback',r['refused_preterminal_selection'] is None)
 fresh=by[r['actual_returned_fresh_event']];check('fresh selected identity',fresh['value']==result['selected_exact']);check('fresh full value repeats selected exact',fresh['value']==winner['value']);equal('returned state',result['state_scaled'],winner['value']['state_scaled']);equal('retained pooled state',result['state_scaled'],list(case['selected_state_scaled'])+[0.,0.,0.])
 c=r['counts'];check('raw count',count['raw_callback']+count['exact_checkpoint']+count['selected_fresh_terminal']==c['raw_callback_executions']);check('exact count',count['exact_checkpoint']+count['selected_fresh_terminal']==c['exact_checkpoint_executions']);check('Jacobian count',count['constraint_jacobian']==c['analytic_tip_jacobian_executions']);check('cache count',count['raw_cache_hit']==c['raw_cache_hits']);check('trace count',len(events)==c['trace_events']);check('budget',c['raw_callback_executions']<=1800 and c['exact_checkpoint_executions']<=60);check('zero physics',result['physics_steps']==0 and not result['new_eval_data'] and not result['new_native_grasps'])
 check('all four terminal iteration counts',len(r['optimizer_starts'])==2 and all(a['status']==9 and not a['success'] and a['nit']==a['major_iterations_observed']==25 for a in r['optimizer_starts']))
 out.append({'index':index,'selected_role':'unchanged pooled reference; no new pose accepted','rank':result['selected_exact']['rank'],'failed_proximity_rows':int((~np.asarray(result['selected_exact']['proximity']['passed'])).sum()),'failed_orientation_rows':int((~np.asarray(result['selected_exact']['exact_orientation']['passed'])).sum()),'invalid_orientation_rows':int((~np.asarray(result['selected_exact']['exact_orientation_fields']['valid'])).sum()),'counts':c,'runtime_seconds':result['runtime_s'],'trace_event_counts':dict(count),'all_witness_constraints_pass':False})
report={'schema':'hug_se3_pilot_independent_saved_trace_review_v1','passed':True,'bindings':bindings,'total_arithmetic_identity_checks':sum(checks.values()),'checks_by_kind':dict(checks),'cases':out,'conclusion':'The finite two-case pilot returns both prior reference poses unchanged. Every selectable exact checkpoint and protected-row decision was replayed; no eligible improvement was found. Four optimizer starts reached their fixed iteration limit. This is not an infeasibility proof.','scope':'All saved raw/exact predicate arithmetic, finite-difference surface columns, complete-state cache identity, selection eligibility, budget and trace provenance; no independent new FK or geometry. Source-only kinematic review is separately bound.','reviewer_new_FK':0,'reviewer_new_geometry':0,'reviewer_new_fit':False,'new_eval_data':False,'historical_nine_strict_FAIL_unchanged':True}
(D/'pilot_output_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'checks':sum(checks.values()),'sha256':sha(D/'pilot_output_review.json'),'cases':out},indent=2))
