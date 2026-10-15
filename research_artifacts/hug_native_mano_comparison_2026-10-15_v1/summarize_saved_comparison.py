"""Recover aggregate summary after export type error. Reads scalars only."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
BASE=Path(__file__).resolve().parent
OUT=BASE/'measurement'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((OUT/'protocol.json').read_text())
old=OUT/'audit_at_measurement.py'
sourcepath=next(p for p in protocol['bindings'] if p.endswith('scripts/audit_hug_native_mano.py'))
assert sha(old)==protocol['bindings'][sourcepath]
for p,h in protocol['bindings'].items():
 if p!=sourcepath:assert sha(p)==h,p
records=[json.loads((OUT/'cases'/f'case_{i:02d}.json').read_text()) for i in range(32)]
assert [c['index'] for c in records]==list(range(32))
assert all(c['physics_steps']==0 for c in records)
# Extract only the original aggregate construction for an exact saved-values replay.
source=old.read_text();start=source.index("    report={'schema':p['schema']");end=source.index("    write(out/'results.json',report)",start)
block='\n'.join(line[4:] for line in source[start:end].splitlines())
spec=importlib.util.spec_from_file_location('original_audit',old);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Version:
 def __init__(self,x):self.__version__=x
context={'p':protocol,'ph':sha(OUT/'protocol.json'),'records':records,'counts':{'static_forwards_reconstructed_from_completed_calls':97,'physics_steps':0,'retarget_solves':0,'new_native_proposals':0},'stats':module.stats,'np':np,'mujoco':Version('3.3.7'),'torch':Version('2.11.0'),'trimesh':Version('4.12.2'),'sha':sha,'out':OUT}
exec(compile(block,'original_saved_value_aggregation','exec'),context)
r=context['report'];r['cases_with_lower_mean_mano_tip_error']=int(r['cases_with_lower_mean_mano_tip_error'])
r['recovery']={'reason':'all32casefiles complete; final summary export raised NumPy int64 serialization error','no_measurement_rerun':True,'original_script_sha256':sha(old),'postprocessor_sha256':sha(__file__),'counts_note':'97forward calls derived from32completecases x3 plus1modelneutralinitialization; counters were not exported before the summary error','runtime_note':'version values read from the same available runtime during recovery; original final metadata was not exported'}
# Read installed runtime versions directly; no HUG, MANO or simulator calls.
import sys,mujoco
sys.path.extend(protocol['dependency_dirs'])
import torch,trimesh
r['runtime'].update(numpy=np.__version__,mujoco=mujoco.__version__,torch=torch.__version__,trimesh=trimesh.__version__)
r['surface_comparison']=[]
for part in module.PARTS:
 rows=[next(x for x in c['parts'] if x['part']==part) for c in records]
 r['surface_comparison'].append({'part':part,'native_triangle_centroid_boundary_error_mano_mean_m':float(np.mean([x['native_mesh_to_mano_proxy']['mean_m'] for x in rows])),'native_triangle_centroid_boundary_error_shadow_mean_m':float(np.mean([x['native_mesh_to_shadow']['mean_m'] for x in rows])),'cases_mano_mean_better':sum(x['native_mesh_to_mano_proxy']['mean_m']<x['native_mesh_to_shadow']['mean_m'] for x in rows),'pointwise_bands':[{'band_m':b,**{key:sum(next(v for v in x['pointwise_band_changes'] if v['band_m']==b)[key] for x in rows) for key in ('native_near','mano_lost','mano_added','shadow_lost','shadow_added')}} for b in protocol['near_bands_m']]})
target=OUT/'results.json';assert not target.exists();target.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['case_reports','surface_comparison']},indent=2))
print(json.dumps(r['surface_comparison'],indent=2))
