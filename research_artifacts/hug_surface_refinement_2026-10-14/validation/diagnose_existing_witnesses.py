"""Recheck all 178 pre-existing counterexample cells at the guarded targets.

No new cells are selected, no solver is run and no geometry is changed. A
previous witness ceasing to be definite is not a preservation PASS.
"""
from pathlib import Path
import importlib.util,json,pickle,warnings,time
import numpy as np
import mujoco
B=Path(__file__).parent;spec=importlib.util.spec_from_file_location('validation_wrapper',B/'validate_candidate.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);a=v.a
sha=v.sha;read=v.read;write=v.write
source=v.OLD/'sampling/strict_preservation_witnesses.json';candidate=Path('/private/tmp/hug_surface_refinement_2026-10-14/guarded_candidate_targets.json');protocol=B/'witness_recheck_protocol.json';p=read(protocol);ph=sha(protocol);assert sha(__file__)==p['script_sha256'];assert sha(source)==p['existing_witnesses_sha256'];assert sha(candidate)==p['candidate_sha256'];old=read(source);targets={c['native_sha256']:c for c in read(candidate)['cases']}
prior=v.ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inventory=read(prior/'existing_hug_inventory.json');fk={c['native_sha256']:c for c in read(prior/'static_saved_hug_fk.json')['cases']};protected=read(v.OLD/'protected_input_sha256.json')
for f,h in protected.items():assert sha(f)==h,f
with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
weights=np.asarray(asset['weights']);grouped=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()]);records=[]
for h,t in targets.items():
    case=inventory['modern_cases'][t['source_inventory_index']];assert h==case['native']['sha256'];meta,tri,labels,norm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(case,grouped,fk);rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);w=[x for x in old['witnesses'] if x['native_sha256']==h];q=np.asarray([x['centroid_object_m'] for x in w]);assert len(q)>0
    for name,val in zip(t['joint_names'],t['qpos_rad'],strict=True):d.qpos[int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name)])]=val
    d.qpos[rootadr:rootadr+3]=t['root_target_position_m'];d.qpos[rootadr+3:rootadr+7]=t['root_target_quaternion_wxyz']
    if m.nmocap:d.mocap_pos[0]=t['root_target_position_m'];d.mocap_quat[0]=t['root_target_quaternion_wxyz']
    mujoco.mj_forward(m,d);assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);nominal=d.qpos.copy();new=[]
    for g in hand:gg=a.geom_rep(m,d,g['id'],R,pos);gg['part']=g['part'];new.append(gg)
    f=v.full_fields(new,q);nd,_,_=a.nearest_parts(q,tri,labels,len(a.PARTS));rows=[]
    for j,x in enumerate(w):
        pi=a.PARTS.index(x['anatomical_part']);native=float(nd[j,pi]);assert abs(native-x['native_centroid_distance_m'])<1e-12;eps=x['numerical_allowance_m'];radius=x['centroid_radius_m'];band=x['band_mm']/1000;mapped=float(f['current_distance_by_part_m'][j,pi]);ni=[max(0,native-radius-eps),native+radius+eps];mi=[max(0,mapped-radius-eps),mapped+radius+eps];nn=ni[1]<=band;nf=ni[0]>band;mn=mi[1]<=band;mf=mi[0]>band
        status='DEFINITE_LOST_NATIVE' if nn and mf else ('DEFINITE_ADDED_MAPPED' if mn and nf else ('DEFINITE_EQUAL_NEAR' if nn and mn else ('DEFINITE_EQUAL_FAR' if nf and mf else 'UNRESOLVED_CELL')))
        rows.append({'original_witness':x,'guarded_centroid_distance_m':mapped,'guarded_cell_distance_interval_m':mi,'native_cell_distance_interval_m':ni,'guarded_relationship_status':status,'retained_strict_counterexample':status.startswith('DEFINITE_LOST') or status.startswith('DEFINITE_ADDED'),'no_longer_definite_does_not_mean_pass':True})
    records.extend(rows);assert np.array_equal(d.qpos,nominal) and d.time==0
    for f,hsh in meta['input_sha256'].items():assert sha(f)==hsh,f
for f,h in protected.items():assert sha(f)==h,f
assert sha(protocol)==ph and sha(candidate)==p['candidate_sha256'] and sha(source)==p['existing_witnesses_sha256'];assert v.COUNTS['actual']==0
counts={s:sum(r['guarded_relationship_status']==s for r in records) for s in sorted({r['guarded_relationship_status'] for r in records})};summary=[]
for caseid in sorted({r['original_witness']['case_id'] for r in records}):
    rr=[r for r in records if r['original_witness']['case_id']==caseid];summary.append({'case_id':caseid,'existing_witness_count':len(rr),'retained_counterexample_count':sum(r['retained_strict_counterexample'] for r in rr),'by_part':{part:sum(r['retained_strict_counterexample'] and r['original_witness']['anatomical_part']==part for r in rr) for part in a.PARTS[:6]}})
report={'schema':'hug_guarded_preexisting_witness_recheck_v1','protocol_sha256':ph,'script_sha256':sha(__file__),'candidate_sha256':sha(candidate),'old_witness_report_sha256':sha(source),'fixed_old_witness_count':len(records),'status_counts':counts,'cases':summary,'witnesses':records,'uniform_numeric_allowance_conditional':True,'physics_steps':0,'guards_actual_attempts':v.COUNTS['actual'],'new_eval_data':False,'new_queries_selected':False,'all_source_geometry_hashes_unchanged':True,'scope':'Recheck only pre-existing178 development witness cells, not exhaustive candidatewitness selection or a validation PASS.'};write(B/'existing_witness_recheck.json',report);print(json.dumps({k:r for k,r in report.items() if k not in ('witnesses',)},indent=2))
