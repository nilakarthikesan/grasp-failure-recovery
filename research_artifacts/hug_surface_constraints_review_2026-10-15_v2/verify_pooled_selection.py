"""Independent pure-data replay of the frozen two-version pooling rule."""
from pathlib import Path
import hashlib,json
O=Path(__file__).parent;D=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection')
read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=0
p=read(D/'protocol.json');c=read(D/'candidate_targets.json');policy=read(D/'policy_v3.json')
assert c['protocol_sha256']==sha(D/'protocol.json') and p['policy_sha256']==sha(D/'policy_v3.json');checks+=1
assert p['selection_rule']==policy['selection_rule'];checks+=1
for path,h in p['input_sha256'].items():assert sha(Path(path))==h,path;checks+=1
paths=[Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json'),Path('/private/tmp/hug_surface_constraints_2026-10-15_v2/candidate_targets.json')];inputs=[read(x) for x in paths];maps=[{r['source_inventory_index']:r for r in v['cases']} for v in inputs]
rows=[]
for r in c['cases']:
 i=r['source_inventory_index'];options=[m[i] for m in maps];ranks=[tuple(x['selected_constraint_diagnostics']['rank']) for x in options]
 which=0 if ranks[0]<=ranks[1] else 1
 source=r['selection_source'];assert source['pooled_source_version']==['V1','V2'][which];checks+=1
 assert source['source_candidate_path']==str(paths[which]) and source['source_candidate_sha256']==sha(paths[which]);checks+=1
 assert source['source_protocol_sha256']==inputs[which]['protocol_sha256'];checks+=1
 assert source['V1_rank']==list(ranks[0]) and source['V2_rank']==list(ranks[1]) and source['stable_V1_tie']==(ranks[0]==ranks[1]);checks+=1
 expected=options[which].copy();expected['selection_source']=source;assert r==expected;checks+=1
 assert r['selected_constraint_diagnostics']['tip_status']['passed'] and r['selected_constraint_diagnostics']['in_domain'];checks+=1
 rows.append({'source_inventory_index':i,'selected_version':['V1','V2'][which],'exact_rank_tie':ranks[0]==ranks[1],'V1_rank':ranks[0],'V2_rank':ranks[1],'palm_displacement_world_m':r['palm_displacement_world_m'],'palm_displacement_norm_m':r['palm_displacement_norm_m'],'retained_proximity_pass_count':sum(r['selected_constraint_diagnostics']['proximity']['passed']),'retained_proximity_row_count':len(r['selected_constraint_diagnostics']['proximity']['passed']),'retained_orientation_pass_count':sum(r['selected_constraint_diagnostics']['exact_orientation']['passed']),'retained_orientation_row_count':len(r['selected_constraint_diagnostics']['exact_orientation']['passed'])})
assert sorted(r['source_inventory_index'] for r in rows)==[8,12,16,20,24,28];checks+=1
report={'passed':True,'checks':checks,'source_bindings':{str(x):sha(x) for x in [D/'protocol.json',D/'candidate_targets.json',D/'policy_v3.json',*paths,Path(__file__)]},'cases':rows,'scope':'Pure-data replay of frozen exact five-component rank with V1 ties. All selected full records except explicit pooling provenance are identical to previously independently source/FK/cap-verified source candidates. No new query, fit, simulation, evaluation data or full-grid outcome selection.','mapping_acceptance':False}
f=O/'pooled_selection_review.json';assert not f.exists();f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'checks':checks,'report_sha256':sha(f),'rows':rows}))
