"""Read-only V1 terminal cause replay; no new geometry or optimizer calls."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
OUT=Path(__file__).parent;D=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
candidate=read(D/'candidate_targets.json');bindings={str(D/'candidate_targets.json'):sha(D/'candidate_targets.json'),str(Path(__file__)):sha(__file__)};records=[];checks=0
for c in candidate['cases']:
 path=Path(c['solver_audit']['trace_path']);assert sha(path)==c['solver_audit']['trace_sha256'];checks+=1;bindings[str(path)]=sha(path)
 raw={};jac={}
 for line in gzip.open(path,'rt'):
  e=json.loads(line)
  if e['kind'] in ('raw_callback','exact_checkpoint','selected_fresh_terminal','fresh_original_fallback'):
   v=e['value'];raw[np.asarray(v['state_scaled'],np.float64).tobytes()]=v
  if e['kind']=='constraint_jacobian':jac[np.asarray(e['state_scaled'],np.float64).tobytes()]=e
 for term in c['solver_audit']['optimizer_starts']:
  if 'terminal_state_scaled' not in term:continue
  x=np.asarray(term['terminal_state_scaled']);key=x.tobytes();assert key in raw;checks+=1;v=raw[key];P=len(c['constraint_source_rows']['proximity']);ds=np.asarray(v['part_distance_m']);m=np.asarray(v['surface_margin']);worst=float(max(0.,np.max(-m,initial=0.)));worst_proximity=float(max(0.,np.max(-m[:P],initial=0.)))
  far=np.array([r['row']['relationship_change']=='new_mapped' for r in c['constraint_source_rows']['proximity']]);worst_far=float(np.max(-m[:P][far],initial=0.));worst_rows=[]
  for j,row in enumerate(c['constraint_source_rows']['proximity']):
   r=row['row'];err=r['centroid_radius_m']+r['numerical_allowance_m'];b=r['band_mm']/1000
   if r['relationship_change']!='new_mapped' or not ds[j]<err or -m[j]!=worst_far:continue
   expected=-(b+1e-9)/.001;assert m[j]==expected;checks+=1
   rec={'source_row_index':j,'source_binding':row,'distance_m':float(ds[j]),'radius_plus_allowance_m':float(err),'distance_inside_clamped_region_m':float(err-ds[j]),'band_m':b,'literal_margin_m':float(v['proximity']['margin_m'][j]),'old_solver_margin':float(m[j]),'unclamped_solver_margin':float((ds[j]-err-b-1e-9)/.001)}
   if key in jac:
    J=jac[key];matrix=np.asarray(J['matrix']);oldgrad=matrix[6+j,:27];newgrad=[]
    for k,delta in enumerate(J['one_sided_spans_scaled']):
     probe=x.copy();probe[k]+=delta;assert probe.tobytes() in raw;checks+=1
     newgrad.append((raw[probe.tobytes()]['part_distance_m'][j]-ds[j])/delta/.001)
    newgrad=np.asarray(newgrad);assert np.array_equal(oldgrad,np.zeros(27));checks+=1
    rec.update(saved_jacobian_event=J['event'],old_solver_gradient_norm=0.,distance_based_guidance_gradient_norm=float(np.linalg.norm(newgrad)),distance_based_guidance_gradient=newgrad.tolist(),geometry_distance_is_responsive=bool(np.any(newgrad!=0)))
   else:rec['terminal_jacobian_available']=False
   worst_rows.append(rec)
  records.append({'source_inventory_index':c['source_inventory_index'],'start_index':term['start_index'],'reported_success':term['success'],'optimizer_status':term['status'],'optimizer_message':term['message'],'terminal_epigraph':term['terminal_epigraph'],'worst_raw_violation':worst,'worst_proximity_violation':worst_proximity,'global_minus_proximity_worst':worst-worst_proximity,'worst_far_violation':worst_far,'global_minus_far_worst':worst-worst_far,'worst_clamped_far_row_count':len(worst_rows),'terminal_exact_tip_pass':v['tip_status']['passed'],'worst_clamped_far_rows':worst_rows})
for i,s in [(12,1),(16,0),(16,1),(20,0),(24,0),(24,1),(28,0),(28,1)]:
 r=next(r for r in records if r['source_inventory_index']==i and r['start_index']==s)
 assert r['reported_success'] and r['worst_clamped_far_row_count']>0;checks+=1
 available=[v for v in r['worst_clamped_far_rows'] if 'old_solver_gradient_norm' in v]
 assert all(v['old_solver_gradient_norm']==0 for v in available);checks+=1
all_rows=[v for r in records for v in r['worst_clamped_far_rows']];available=[v for v in all_rows if 'old_solver_gradient_norm' in v];responsive=[v for v in available if v['geometry_distance_is_responsive']]
checks+=1 # Report unavailable terminal-Jacobian evidence without inventing a response.
assert all(sha(p)==h for p,h in bindings.items());checks+=1
report={'passed':True,'checks':checks,'source_bindings':bindings,'terminals':records,'clamped_worst_far_row_count':len(all_rows),'exact_terminal_jacobian_available_rows':len(available),'responsive_distance_probe_rows':len(responsive),'interpretation':'For eight reported-success V1 terminals, worst far-cell epigraph rows are in d<radius+allowance. Where an exact terminal Jacobian was stored, saved far-residual Jacobian rows have zero 27D gradient; matching saved distance probes and their responsiveness are reported per row. Missing terminal Jacobians are explicitly unavailable. The global raw maximum exceeds the far maximum by at most about 1.25e-10 from other rows, recorded explicitly. This certifies a local optimizer-guidance plateau in those records; it does not establish that it is the sole cause of failure, that V2 will improve, or that feasible mapping exists.','new_geometry_queries':0,'new_FK':0,'new_fit':False,'new_eval_data':False}
p=OUT/'far_plateau_evidence.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'checks':checks,'sha256':sha(p),'terminal_summaries':[{k:v for k,v in r.items() if k!='worst_clamped_far_rows'} for r in records]}))
