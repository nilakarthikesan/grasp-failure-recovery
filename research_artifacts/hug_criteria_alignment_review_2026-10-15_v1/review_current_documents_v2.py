"""Read saved normal fields; bind the added Milk B sentence without geometry queries."""
from pathlib import Path
import json,hashlib,datetime
OUT=Path(__file__).parent;DOC=Path('/private/tmp/hug_criteria_alignment_root_2026-10-15_v1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v1=OUT/'final_root_review.json';assert sha(v1)=='586b277d57bcbc9e4f29e046080032e4f256f742f1c87826327cfb12f3216082'
prior=read(v1);cp=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1/candidate_targets.json');candidate=read(cp)
assert sha(cp)=='b0a6993006daf7aba9c1ee4d8898892a99dac19a8906f76e7fb1b85295832757'
case=next(x for x in candidate['cases'] if x['source_inventory_index']==20);source=case['constraint_source_rows']['orientation'][2];row=source['row'];native=row['native_facing_compatibility'];assert native>1e-6
scores=[]
for r in case['restoration_endpoint_results']:
 f=r['exact_surface_check']['exact_orientation_fields'];score=f['score'][2]
 assert f['valid'][2] is True and score< -1e-6 and not any(mask[2] for mask in f['reason_masks'].values())
 dot=-sum(a*b for a,b in zip(f['outward_normal'][2],row['object_outward_normal']))
 assert abs(dot-score)<1e-15
 scores.append({'endpoint_start_index':r['source_endpoint']['start_index'],'native_score':native,'mapped_score':score,'recomputed_dot_score':dot,'exact_valid':True,'invalid_reason_flags_all_false':True})
assert len(scores)==2
review=Path('/private/tmp/hug_feasibility_restoration_review_2026-10-15_v1/final_review.json');direction=Path('/private/tmp/hug_feasibility_restoration_direction_review_2026-10-15_v1/reliable_facing_reversal.json')
assert sha(review)=='f68da88a7b6ea8de7ef2af8e10c5c1f7a1e0114216614b891236145e7e7d5a20' and read(review)['passed'] is True
assert sha(direction)=='97467033bc23086f270b2971561099442886fff100d9226701ecb38a052685c1'
docs=[DOC/n for n in ['criteria_alignment.md','paper_scope_paragraph.tex','paper_result_paragraph.tex']]
assert sha(docs[1])==prior['reviewed_document_bindings'][str(docs[1])]
assert 'A still-eligible milk-box B sample reverses its native-relative facing sign in each of its two restored endpoints.' in docs[0].read_text()
assert 'One still-eligible milk-box B sample reverses its native-relative facing sign' in docs[2].read_text()
result={'schema':'hug_current_root_documents_local_provenance_and_result_review_v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'prior_review_retained':{'path':str(v1),'sha256':sha(v1)},'reviewed_document_bindings':{str(p):sha(p) for p in docs},'evidence_bindings':{str(p):sha(p) for p in [cp,review,direction]},'scope':'Narrow addendum to V1: confirm the new still-eligible Milk B facing-sign reversal sentence against saved exact fields and independent numerical review. Previous local/result review conclusions remain; external-paper facts remain separate root source review responsibility.','blocking_findings':[],'case_index':20,'retained_orientation_row_index':2,'source_query_index':row['query_index'],'query_object_m':row['query_object_m'],'anatomical_part':row['native_part'],'source_band_mm':source['band_mm'],'source_reference':source,'verified_endpoint_scores':scores,'interpretation':'The same saved object query has native facing score +0.27928109956560787 and negative mapped scores in both restored endpoints, with all exact eligibility flags valid. This is pointwise retained-row evidence, not a new full-grid audit or a continuous area certificate. Previously infeasible endpoint orientations were not measured by the same exact check, so projection causality remains unclaimed.','new_geometry_queries':0,'new_FK':0,'physics_steps':0,'new_eval_data':False,'historical_result_reclassification':False}
p=OUT/'final_root_review_v2.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
files=[{'path':x.name,'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='manifest_v2.json']
(OUT/'manifest_v2.json').write_text(json.dumps({'schema':'hug_current_root_document_review_manifest_v2','files':files,'prior_manifest_retained_unchanged':True},indent=2)+'\n')
assert sha(v1)=='586b277d57bcbc9e4f29e046080032e4f256f742f1c87826327cfb12f3216082'
assert sha(OUT/'manifest.json')=='cf49516e913392e4c186e3f7f406e5b21fe29423c97e5b726a2d78482eda1c91'
print(json.dumps({'passed':True,'review_sha256':sha(p),'manifest_v2_sha256':sha(OUT/'manifest_v2.json'),'scores':scores},indent=2))
