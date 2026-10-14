from pathlib import Path
import json,pickle,warnings
import numpy as np
import audit_other_surfaces as a
out=Path(__file__).parent;root=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');prior=root/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inventory=json.loads((prior/'existing_hug_inventory.json').read_text());p=json.loads((out/'protocol_v2.json').read_text());fk=json.loads((prior/'static_saved_hug_fk.json').read_text());fm={c['native_sha256']:c for c in fk['cases']}
with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
w=np.asarray(asset['weights']);gw=np.column_stack([w[:,ids].sum(1) for ids in a.GROUPS.values()]);seen=set();records=[]
for planned in p['cases']:
 c=inventory['modern_cases'][planned['source_inventory_index']]
 if c['object_id'] in seen:continue
 seen.add(c['object_id']);meta,tri,labels,norm,hand,objs,*_=a.load_case(c,gw,fm)
 for density in [2,1]:
  q,area,n,gid,qm=a.object_queries(objs,density);probe=a.object_membership(q+1e-7*n,objs,tol=-1e-10)
  records.append({'object':c['object_id'],'density_mm':density,'retained_v2_queries':len(q),'outward_probe_occluded_query_count':int(probe.sum()),'outward_probe_occluded_area_mm2':float(area[probe].sum()),'retained_v2_area_mm2':float(area.sum())})
a.write(out/'outward_probe_before_v3.json',{'check':'Read-only geometry check of retained v2 collision-hull centroid queries. Outward probe +1e-7m*source outward normal inside another hull reveals abutting internal faces. No hand surface measurements changed.','v2_protocol_sha256':a.sha(out/'protocol_v2.json'),'script_sha256':a.sha(__file__),'physics_steps':0,'results':records});print(json.dumps(records,indent=2))
