"""Read-only saved-file/hash/JSON inventory; never deserialize native pickles or query geometry."""
from pathlib import Path
import json,hashlib,subprocess,collections,datetime
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');MAIN=Path('/Users/nilakarthikesan/grasp-failure-prediction');OUT=Path(__file__).parent
files={
'inventory':ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json',
'fk':ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json',
'palm':ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/palm/palm_anchor_report.json',
'reference':ROOT/'research_artifacts/hug_geometry_reference_validation_2026-10-14/reference/hug_reference_comparison_compact.json',
'native_contract':ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/native_contract/protocol.json',
'orientation':ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json',
'cube_surface':ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/surface/protocol.json',
'fit':Path('/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json'),
'acceptance':ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json',
'criteria':ROOT/'docs/HUG_CRITERIA_ALIGNMENT.md',
'static_protocol':ROOT/'docs/HUG_STATIC_VALIDATION_PROTOCOL.md',
'confirmation':ROOT/'docs/HUG_MAPPING_CONFIRMATION.md',
'paper_review':Path('/private/tmp/hug_criteria_source_review_2026-10-15_v1/source_review.md'),
'paper_bindings':Path('/private/tmp/hug_criteria_source_review_2026-10-15_v1/source_bindings.json')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,v):(OUT/name).write_text(json.dumps(v,indent=2)+'\n')
d={k:read(p) for k,p in files.items() if str(p).endswith('.json')};inv=d['inventory'];modern=inv['modern_cases'];hashset={x['native']['sha256'] for x in modern}
assert len(modern)==len(hashset)==32
sets={k:{x['native_sha256'] for x in d[k]['cases']} for k in ['fk','palm','reference','native_contract','orientation','fit']}
for k in ['fk','palm','reference','native_contract']:assert sets[k]==hashset
assert sets['orientation']==sets['fit'] and len(sets['fit'])==6
cubehash=d['cube_surface']['native_proposal_sha256'];assert cubehash in hashset
# Explicitly inventory both project run roots again; no other directory is implied searched.
paths=[Path(x) for x in subprocess.check_output(['rg','--files','-uuu',str(MAIN/'runs'),str(ROOT/'runs')],text=True).splitlines() if x.endswith('/proposal.pkl')]
known={Path(x['path']):x for x in inv['all_primary_native_proposal_entries']+inv['all_worktree_native_proposal_entries']}
assert set(paths)==set(known),'Saved proposal set changed; investigate before claiming completeness'
current=[]
for p in sorted(paths):
 h=sha(p);assert h==known[p]['sha256'];current.append({'path':str(p),'sha256':h,'file_bytes':p.stat().st_size,'recorded_native_inference_report_verified':known[p].get('recorded_proposal_hash_matches'), 'modern_source':h in hashset})
rows=[]
for i,c in enumerate(modern):
 h=c['native']['sha256'];full=h in sets['orientation'];cube=h==cubehash
 rows.append({'source_inventory_index':i,'native_sha256':h,'object_id':c['object_id'],'geometry_content_hash':c['geometry_content_hash'],'observation_group_id':c['observation_group_id'],'generation_seed':c['native']['seed'],'input_tensor_hash':c['native']['input_tensor_hash'],'case_path':c['native']['path'],'source_inference_report_path':c['native']['inference_report_path'],'source_inference_report_sha256':c['native']['inference_report_sha256'],'recorded_episode_count':c['existing_episode_count_for_same_source'],'exposure':{'native_3d_contract':h in sets['native_contract'],'point_FK':h in sets['fk'],'wrist_palm_anatomical_proxies':h in sets['palm'],'current_released_metric_reference_comparison':h in sets['reference'],'detailed_cube_surface':cube,'noncube_surface_direction_sampling':full,'repeated_surface_placement_fitting':h in sets['fit']},'prospective_role':'EXPOSED_DEVELOPMENT','new_object_holdout':False,'new_observation_holdout':False,'blind_native_grasp_holdout':False,'no_located_detailed_surface_fit_or_full_surface_audit':not(full or cube),'interpretation':'May support additional same-object development stress checks after a new metric freeze; absence of detailed surface fitting is not independent holdout status.'})
legacy_by_hash={}
for p,e in known.items():
 if e['sha256'] in hashset:continue
 x=legacy_by_hash.setdefault(e['sha256'],{'native_sha256':e['sha256'],'paths':[],'reported_object_names':set(),'prior_family_roots':set(),'holdout_status':'UNESTABLISHED_LEGACY_NOT_ADMISSIBLE_AS_BLIND_HOLDOUT','reason':'Existing pilot/closure/review histories and weaker scene/provenance bindings; no no-tuning exposure proof.'})
 x['paths'].append(str(p));x['reported_object_names'].add(e.get('object_name'));x['prior_family_roots'].add('/'.join(p.relative_to(MAIN/'runs').parts[:2]))
for x in legacy_by_hash.values():x['reported_object_names']=sorted(x['reported_object_names'],key=lambda z:str(z));x['prior_family_roots']=sorted(x['prior_family_roots'])
groups=[]
for geom in sorted({r['geometry_content_hash'] for r in rows}):
 g=[r for r in rows if r['geometry_content_hash']==geom]
 groups.append({'geometry_content_hash':geom,'object_ids':sorted({r['object_id'] for r in g}),'observation_groups':sorted({r['observation_group_id'] for r in g}),'indices':[r['source_inventory_index'] for r in g],'all_previously_exposed':True,'unseen_geometry':False})
assert len(groups)==4 and len({r['observation_group_id'] for r in rows})==8
summary={'saved_proposal_files_two_roots':len(paths),'unique_native_pickle_hashes':len(set(x['sha256'] for x in current)),'well_bound_modern_sources':32,'legacy_unique_hashes':len(legacy_by_hash),'well_bound_geometry_groups':4,'well_bound_observation_groups':8,'all32_prior_point_anchor_reference_exposure':True,'repeated_surface_fit_cases':len(sets['fit']),'additional_prior_detailed_cube_case':1,'no_located_detailed_surface_study_cases':sum(r['no_located_detailed_surface_fit_or_full_surface_audit'] for r in rows),'certified_unused_native_grasp_holdouts':0,'available_unseen_object_holdouts':0,'recorded_modern_episode_count':sum(r['recorded_episode_count'] for r in rows),'episode_variants_are_not_unique_native_grasps':True}
write('inventory_exposure.json',{'schema':'hug_practical_validation_existing_inventory_exposure_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Read-only current file-name/SHA verification in two named runs roots plus saved JSON/doc exposure. No native pickle decoding, geometry, FK, outcomes read or new data. Zero certified holdouts means none established by this scoped audit, not a claim that every legacy file was proven used for every fit.','input_bindings':{str(p):sha(p) for p in files.values()},'summary':summary,'geometry_groups':groups,'modern_cases':rows,'legacy_unique_sources':list(legacy_by_hash.values()),'current_native_file_hashes':current,'data_split_rule':'Connected source-geometry identity including aliases/decompositions/transformed copies is the outer split unit; observation groups/seeds/native-hash duplicates stay with the geometry. Input tensor hashes identify inference provenance, not independent objects.','no_new_eval_data':True,'new_inference':False,'new_geometry_queries':0,'new_FK':0,'physics_steps':0})
excerpts=[]
for key,start,end in [('static_protocol',90,109),('static_protocol',320,360),('confirmation',146,159),('confirmation',226,233),('confirmation',370,380)]:
 p=files[key];lines=p.read_text().splitlines();excerpts.append({'path':str(p),'sha256':sha(p),'start_line':start,'end_line':end,'excerpt':'\n'.join(lines[start-1:end])})
write('exposure_documentary_evidence.json',{'excerpts':excerpts,'input_binding_policy':'Snapshot hashes and exact excerpts; no claim mutable shared docs stay byte-identical after this audit.'})
print(json.dumps(summary,indent=2))
