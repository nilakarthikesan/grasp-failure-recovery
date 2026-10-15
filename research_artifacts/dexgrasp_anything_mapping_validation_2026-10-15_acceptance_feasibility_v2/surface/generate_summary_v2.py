"""Correct only v1 summary's mm² leaf labels, preserving measured evidence."""
from pathlib import Path
import hashlib,json,copy
HERE=Path(__file__).resolve().parent
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
 old=json.loads((HERE/'summary_v1.json').read_text());new=copy.deepcopy(old)
 source=json.loads((HERE/'visual_results_v1/global_6_report.json').read_text())['strict_encoded_nearset_acceptance']['by_band_and_part']['1.0']
 new['primary_row6_at_1mm_band_mm2']={part:{name:{key.replace('_m2','_mm2'):value*1e6 for key,value in bounds.items()} for name,bounds in source[part]['bounds_intersection'].items()} for part in ['index','ring']}
 for part,metrics in new['primary_row6_at_1mm_band_mm2'].items():
  for name,bounds in metrics.items():
   assert set(bounds)=={'lower_mm2','upper_mm2'}
   original=source[part]['bounds_intersection'][name]
   for bound in ['lower','upper']:
    assert bounds[bound+'_mm2']==original[bound+'_m2']*1e6
    assert bounds[bound+'_mm2']==old['primary_row6_at_1mm_band_mm2'][part][name][bound+'_m2']
 assert new['primary_row6_at_1mm_band_mm2']['index']['lost_native']['lower_mm2']==50.24999999999999
 assert new['primary_row6_at_1mm_band_mm2']['ring']['lost_native']['lower_mm2']==66.9375
 for key in old:
  if key!='primary_row6_at_1mm_band_mm2':assert new[key]==old[key]
 new['schema_version']='dga_raw_surface_summary_v2'
 new['reporting_correction']={'superseded_summary':{'path':'summary_v1.json','sha256':sha(HERE/'summary_v1.json')},'scope':'V1 had mm²-converted values under misleading lower_m2/upper_m2 leaf names inside the mm2 block. V2 labels them lower_mm2/upper_mm2. No numerical values, measurements, outcomes, criteria, or source evidence changed.','generator_sha256':sha(__file__),'units_regression_status':'PASS_ALL8_LEAF_VALUES_RECOUNTED_FROM_ORIGINAL_METRE_SQUARED_REPORT'}
 return new
if __name__=='__main__':
 new=build()
 with (HERE/'summary_v2.json').open('x') as f:json.dump(new,f,indent=2);f.write('\n')
 print(json.dumps({'status':'PASS_UNITS_REGRESSION_NO_MEASUREMENT_RERUN','summary_v2_sha256':sha(HERE/'summary_v2.json')}))
