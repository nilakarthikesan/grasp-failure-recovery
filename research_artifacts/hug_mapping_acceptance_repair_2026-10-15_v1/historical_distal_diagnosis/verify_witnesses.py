"""Saved-array witness recount using math.fsum; no BLAS, MuJoCo or hull construction."""
from pathlib import Path
from xml.etree import ElementTree as ET
import json,hashlib,math
import numpy as np
H=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((H/'result.json').read_text());p=json.loads((H/'protocol.json').read_text());u=Path(p['source_urdf']);xml=ET.parse(u).getroot();out=[]
def source(name,kind):
 node=next(e for e in xml.findall('link') if e.get('name')==name).find(kind);o=node.find('origin');xyz=[0,0,0] if o is None else list(map(float,o.get('xyz','0 0 0').split()));rpy=[0,0,0] if o is None else list(map(float,o.get('rpy','0 0 0').split()));assert rpy==[0,0,0]
 m=node.find('geometry/mesh');scale=list(map(float,m.get('scale','1 1 1').split()));q=u.parent/m.get('filename');return [[float(t)*scale[k]+xyz[k] for k,t in enumerate(s.split()[1:4])] for s in q.read_text().splitlines() if s.startswith('v ')]
with np.load(H/'local_arrays.npz',allow_pickle=False) as arrays:
 for r in d['rows']:
  old=arrays[r['saved_array_key']].tolist()
  for kind,key in [('collision','historical_to_declared_collision'),('visual','historical_to_source_visual')]:
   raw=source(r['link'],kind)
   for direction,A,B in [('first_to_second',old,raw),('second_to_first',raw,old)]:
    w=r[key][direction];plane=w['target_outward_plane'];normal=math.sqrt(math.fsum(x*x for x in plane[:3]));dot=lambda x:math.fsum([*(x[k]*plane[k] for k in range(3)),plane[3]])/normal
    value=dot(w['query_point']);assert abs(value-w['max_signed_outside_support_plane_m'])<1e-16
    point=A[w['query_source_vertex_index']];assert point==w['query_point']
    target_support=max(dot(x) for x in B);assert target_support<1e-14
    out.append({'scene_sha256':r['scene_sha256'],'link':r['link'],'target_kind':kind,'direction':direction,'replayed_signed_witness_m':value,'target_max_signed_plane_residual_m':target_support,'shifted_to_actual_target_vertices_lower_witness_m':max(0.,value-max(0.,target_support))})
report={'passed':True,'schema':'hug_distal_saved_plane_witness_recount_v1','bindings':{str(H/'result.json'):sha(H/'result.json'),str(H/'local_arrays.npz'):sha(H/'local_arrays.npz'),str(H/'protocol.json'):sha(H/'protocol.json'),str(Path(__file__)):sha(__file__)},'witness_count':len(out),'rows':out,'maximum_target_support_residual_m':max(r['target_max_signed_plane_residual_m'] for r in out),'new_mujoco_loads':0,'new_FK_calls':0,'new_hull_queries':0,'not_directed_rounding_certificate':True}
(H/'witness_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'sha256':sha(H/'witness_review.json'),'witnesses':len(out),'max_target_support_residual_m':report['maximum_target_support_residual_m']}))
