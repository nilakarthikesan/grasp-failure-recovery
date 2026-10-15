"""Read-only post-fit audit of recorded major/terminal refusal events."""
from pathlib import Path
import gzip,hashlib,json
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((OUT/'candidate_targets.json').read_text());results=[]
for c in r['cases']:
    path=Path(c['solver_audit']['trace_path']);caps=c['authoritative_original_tip_caps_sqrt_einsum_m'];rows=[];terminals={};counts={'tip_cap':0,'mean':0,'domain':0}
    with gzip.open(path,'rt') as f:
        for line in f:
            e=json.loads(line)
            if e['kind']=='selection_refusal_before_orientation' and ('_major_' in e['reason'] or e['reason'].endswith('_terminal')):
                t=e['tip_status'];ex=[n-cap for n,cap in zip(t['norms_m'],caps)];row={'event':e['event'],'reason':e['reason'],'in_domain':e['in_domain'],'tip_cap_excess_m':ex,'maximum_tip_cap_excess_m':max(ex),'individual_tip_pass':t['individual_pass'],'mean_tip_error_m':t['mean_m'],'mean_gate_pass':t['mean_pass'],'orientation_not_computed':True}
                rows.append(row);counts['tip_cap']+=any(x>0 for x in ex);counts['mean']+=not t['mean_pass'];counts['domain']+=not e['in_domain']
                if e['reason'].endswith('_terminal'):terminals[e['reason']]=row
            elif e['kind']=='exact_checkpoint' and e['reason'].endswith('_terminal'):
                v=e['value'];terminals[e['reason']]={'event':e['event'],'reason':e['reason'],'in_domain':v['in_domain'],'tip_status':v['tip_status'],'rank':v['rank'],'failed_proximity_rows':sum(not x for x in v['proximity']['passed']),'failed_orientation_rows':sum(not x for x in v['exact_orientation']['passed']),'invalid_orientation_rows':sum(not x for x in v['exact_orientation_fields']['valid']),'orientation_not_computed':False}
    starts=[]
    for s in c['solver_audit']['optimizer_starts']:
        small={k:v for k,v in s.items() if k!='terminal_state_scaled'};small['terminal_record']=terminals.get('start_'+str(s['start_index'])+'_terminal');starts.append(small)
    results.append({'index':c['source_inventory_index'],'trace_sha256':sha(path),'major_and_terminal_refusals':rows,'refusal_counts':counts,'refused_major_or_terminal_events':len(rows),'optimizer_starts':starts,'maximum_recorded_tip_cap_excess_m':max([z['maximum_tip_cap_excess_m'] for z in rows],default=0),'selected_exact_reason':c['solver_audit']['selected_source_exact_checkpoint']['reason'],'selected_role':c['audited_pose_role']})
p=OUT/'optimizer_refusal_summary.json';assert not p.exists();p.write_text(json.dumps({'schema':'hug_saved_trace_optimizer_refusals_v1','candidate_sha256':sha(OUT/'candidate_targets.json'),'script_sha256':sha(__file__),'cases':results,'no_new_geometry_or_optimization':True,'status_success_is_optimizer_not_mapping_success':True},indent=2)+'\n')
print(json.dumps({'report':str(p),'sha256':sha(p),'summary':[{'index':x['index'],'counts':x['refusal_counts'],'selected_reason':x['selected_exact_reason']} for x in results]}))
