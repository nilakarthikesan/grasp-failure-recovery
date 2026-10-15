"""Count exact existing primary-allowance failure rows using cached arrays only."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14')
EPHEM=Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral');CACHE=Path('/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sp=read(ROOT/'sampling/protocol.json');wp=ROOT/'sampling/strict_preservation_witnesses.json';retained=read(wp)['witnesses'];eps=sp['primary_allowance_m'];assert eps==1e-8
paths=[Path(__file__),wp,ROOT/'sampling/protocol.json',ROOT/'acceptance/acceptance_rule.json']
for c in sp['cases']:
 for density in c['densities_mm']:
  n=f'{c["case_id"]}_density{density}mm';paths.extend([EPHEM/(n+'_full_current.npz'),EPHEM/(n+'_full_current.binding.json'),CACHE/(n+'_queries.npz')])
inputs={str(p):sha(p) for p in paths};assert inputs[str(ROOT/'acceptance/acceptance_rule.json')]=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
p={'schema':'hug_cached_whole_cell_failure_row_count_protocol_v1','utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':inputs,'numerical_allowance_m':eps,'scope':'All existing nine original pose grids; count lost=Ndef & ~Mposs and added=Mdef & ~Nposs. No queries, cell construction, fitting or FK. Row counts across nested bands/parts/grids are not unique physical cells or area.'}
assert not (OUT/'cached_failure_count_protocol.json').exists();write(OUT/'cached_failure_count_protocol.json',p)
parts=['palm','index','middle','ring','little','thumb'];records=[];totals={'lost_native':0,'new_mapped':0};covered=0
keys=[(w['native_sha256'],w['density_mm'],w['anatomical_part'],w['band_mm'],w['relationship_change'],w['child_cell_index']) for w in retained];assert len(keys)==len(set(keys))==178
for c in sp['cases']:
 native=c['native_sha256'];crec={'source_inventory_index':c['source_inventory_index'],'case_id':c['case_id'],'native_sha256':native,'densities':{},'failure_rows':0,'retained_rows':0}
 for density in c['densities_mm']:
  name=f'{c["case_id"]}_density{density}mm';ep=EPHEM/(name+'_full_current.npz');cp=CACHE/(name+'_queries.npz');binding=read(EPHEM/(name+'_full_current.binding.json'));assert sha(ep)==binding['ephemeral_npz_sha256'] and sha(cp)==binding['original_cache_sha256']
  with np.load(ep,allow_pickle=False) as f:r=f['cell_radius_m'].copy();sd=f['current_distance_by_part_m'].copy()
  with np.load(cp,allow_pickle=False) as f:nd=f['native_distance_by_part_m'].copy()
  rows=[];count=0;kept=0
  for j,part in enumerate(parts):
   for band in sp['bands_mm']:
    b=band/1000;error=r+eps;nD=nd[:,j]+error<=b;nP=np.maximum(0,nd[:,j]-error)<=b;mD=sd[:,j]+error<=b;mP=np.maximum(0,sd[:,j]-error)<=b
    for kind,mask in [('lost_native',nD & ~mP),('new_mapped',mD & ~nP)]:
     n=int(mask.sum());matching=[k for k in keys if k[:5]==(native,density,part,band,kind)]
     assert all(mask[k[5]] for k in matching);covered+=len(matching);kept+=len(matching);count+=n;totals[kind]+=n
     rows.append({'part':part,'band_mm':band,'relationship_change':kind,'definite_failing_cell_rows':n,'retained_example_rows':len(matching),'omitted_rows':n-len(matching)})
  crec['densities'][str(density)]={'query_count':len(r),'failure_rows':count,'retained_rows':kept,'omitted_rows':count-kept,'strata':rows};crec['failure_rows']+=count;crec['retained_rows']+=kept
 crec['omitted_rows']=crec['failure_rows']-crec['retained_rows'];records.append(crec)
assert covered==178
for path,h in inputs.items():assert sha(path)==h
result={'schema':'hug_cached_whole_cell_failure_rows_v1','protocol_sha256':sha(OUT/'cached_failure_count_protocol.json'),'script_sha256':sha(__file__),'inputs_sha256':inputs,'total_definite_failure_rows':sum(totals.values()),'failure_rows_by_relationship':totals,'retained_example_rows':178,'omitted_definite_failure_rows':sum(totals.values())-178,'cases':records,'scope':'Original pose only; counts of case/density/part/band/change/cell tuples. Nested bands, distinct parts and coarse/fine grids overlap; do not interpret as distinct physical failure sites or contact area.','new_geometry_queries':0,'new_grid_construction':0,'new_FK':0,'new_fit':False,'new_eval_data':False};write(OUT/'cached_failure_row_counts.json',result);print(json.dumps({k:result[k] for k in ('total_definite_failure_rows','failure_rows_by_relationship','retained_example_rows','omitted_definite_failure_rows')},indent=2));print(json.dumps([{'index':c['source_inventory_index'],'failure_rows':c['failure_rows'],'retained':c['retained_rows'],'omitted':c['omitted_rows']} for c in records],indent=2))
