"""Extract already recorded counterexamples; no fitting or geometry queries."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json
import numpy as np
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
PREV=Path('/private/tmp/hug_free_palm_surface_audit_2026-10-15_v1')
EPHEM=Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral')
CACHE=Path('/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def load(p):return json.loads(gzip.decompress(Path(p).read_bytes())) if str(p).endswith('.gz') else read(p)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
sp=read(OLD/'sampling/protocol.json');op=read(OLD/'orientation/protocol.json');cases={c['native_sha256']:c for c in sp['cases']}
reports=[('original_mapping',OLD/'orientation/orientation_report_full.json.gz'),('initial_translated_diagnostics',PREV/'candidate_validation_8,12,16.json.gz'),('initial_translated_diagnostics',PREV/'candidate_validation_20,24,28.json.gz'),('explicit_translation_supplement',PREV/'supplemental_explicit_translation/candidate_validation_8,12,16.json.gz')]
paths=[Path(__file__),OLD/'sampling/strict_preservation_witnesses.json',OLD/'sampling/cell_intervals.py',OLD/'sampling/protocol.json',OLD/'orientation/protocol.json',OLD/'acceptance/acceptance_rule.json']+[p for _,p in reports]
for c in sp['cases']:
 for density in c['densities_mm']:
  name=f'{c["case_id"]}_density{density}mm';paths.extend([EPHEM/(name+'_full_current.npz'),EPHEM/(name+'_full_current.binding.json'),CACHE/(name+'_queries.npz')])
inputs={str(p):sha(p) for p in paths}
assert inputs[str(OLD/'acceptance/acceptance_rule.json')]=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
protocol={'schema':'hug_existing_witness_extraction_protocol_v1','utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':inputs,'scope':'All retained saved whole-cell and orientation counterexamples, original + previous translated diagnostics. No ranking, fitting, model/FK, new grid or closest-surface queries. Duplicate rows across bands/audits retained.','scientific_acceptance_unchanged':True}
assert not (OUT/'witness_extraction_protocol.json').exists();write(OUT/'witness_extraction_protocol.json',protocol)
loaded={};checks=0
def arrays(native,density):
 global checks
 key=(native,int(density))
 if key not in loaded:
  c=cases[native];name=f'{c["case_id"]}_density{int(density)}mm';ep=EPHEM/(name+'_full_current.npz');cp=CACHE/(name+'_queries.npz');binding=read(EPHEM/(name+'_full_current.binding.json'))
  assert sha(ep)==binding['ephemeral_npz_sha256'];assert sha(cp)==binding['original_cache_sha256'];assert sha(cp)==op['query_cache_sha256'][str(cp)]
  with np.load(ep,allow_pickle=False) as f:e={k:f[k].copy() for k in f.files}
  with np.load(cp,allow_pickle=False) as f:q={k:f[k].copy() for k in f.files}
  assert np.array_equal(e['query_object_m'],q['query_object_m']) and np.array_equal(e['area_mm2'],q['area_mm2'])
  loaded[key]=(e,q,cp,ep);checks+=5
 return loaded[key]
parts=['palm','index','middle','ring','little','thumb','forearm_carrier','wrist_carrier']
whole=[];source=OLD/'sampling/strict_preservation_witnesses.json'
for number,row in enumerate(read(source)['witnesses']):
 e,q,cp,ep=arrays(row['native_sha256'],row['density_mm']);ix=row['child_cell_index'];part=parts.index(row['anatomical_part']);assert np.array_equal(e['query_object_m'][ix],row['centroid_object_m']);assert e['cell_radius_m'][ix]==row['centroid_radius_m'];assert e['area_mm2'][ix]==row['cell_area_mm2'];assert q['native_distance_by_part_m'][ix,part]==row['native_centroid_distance_m'];checks+=4
 whole.append({'source_inventory_index':cases[row['native_sha256']]['source_inventory_index'],'source_report':str(source),'source_report_sha256':inputs[str(source)],'source_json_pointer':f'/witnesses/{number}','original_full_field_cache_path':str(ep),'original_full_field_cache_sha256':inputs[str(ep)],'row':row})
orientation=[];counts={}
for role,path in reports:
 report=load(path)
 for ci,c in enumerate(report['cases']):
  native=c['native_sha256'];index=cases[native]['source_inventory_index']
  for density,res in c['resolutions'].items():
   ori=res if role=='original_mapping' else res['orientation'];e,q,cp,ep=arrays(native,density)
   for band,data in ori['bands'].items():
    signed=data['common_same_label']['pointwise_sign_check']
    for direction,rows in signed['witnesses'].items():
     for wi,row in enumerate(rows):
      ix=row['query_index'];part=parts.index(row['native_part']);assert row['native_part']==row['mapped_part'];assert np.array_equal(q['query_object_m'][ix],row['query_object_m']);assert abs(q['native_distance_by_part_m'][ix,part]-row['native_unsigned_distance_m'])<1e-12;checks+=3
      pointer=f'/cases/{ci}/resolutions/{density}/'+('' if role=='original_mapping' else 'orientation/')+f'bands/{band}/common_same_label/pointwise_sign_check/witnesses/{direction}/{wi}'
      orientation.append({'source_inventory_index':index,'native_sha256':native,'case_id':c['case_id'],'density_mm':int(density),'band_mm':int(band),'direction':direction,'source_audit_role':role,'source_report':str(path),'source_report_sha256':inputs[str(path)],'source_json_pointer':pointer,'original_query_cache_path':str(cp),'original_query_cache_sha256':inputs[str(cp)],'original_full_field_cache_path':str(ep),'original_full_field_cache_sha256':inputs[str(ep)],'cell_radius_m':float(e['cell_radius_m'][ix]),'cell_area_mm2':float(e['area_mm2'][ix]),'source_face_id':int(e['source_face_id'][ix]),'row':row})
      counts[role]=counts.get(role,0)+1
for p,h in inputs.items():assert sha(p)==h
result={'schema':'hug_complete_saved_counterexample_extraction_v1','extraction_protocol_sha256':sha(OUT/'witness_extraction_protocol.json'),'input_sha256':inputs,'whole_cell_rows':whole,'orientation_rows':orientation,'whole_cell_row_count':len(whole),'orientation_row_count':len(orientation),'orientation_counts_by_prior_audit':counts,'source_identity_checks':checks,'coverage_limits':['Whole-cell rows are largest-area retained definite counterexamples per case/density/part/band/direction, not all failing cells.','Orientation rows are capped original saved examples per stratum (maximum3); duplicates across bands and studies retained. No new witness search or reranking.','Full original cache arrays support separately declared deterministic training constraints; native labels and exact grids remain fixed.','Turning definite mismatch into ambiguity does not establish preservation. No pointwise reversal remains UNRESOLVED for continuous orientation.'],'new_geometry_queries':0,'new_FK':0,'new_fit':False,'new_eval_data':False}
write(OUT/'saved_witnesses.json',result);print(json.dumps({k:result[k] for k in ('whole_cell_row_count','orientation_row_count','orientation_counts_by_prior_audit','source_identity_checks')},indent=2))
