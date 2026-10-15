from pathlib import Path
import hashlib,json,importlib.util,sys,warnings,traceback,difflib
import numpy as np
import mujoco
warnings.simplefilter('error',RuntimeWarning)
def denied(*a,**k):raise RuntimeError('Static object compiler only; dynamics forbidden')
for n in ['mj_step','mj_step1','mj_step2','mj_integratePos']:setattr(mujoco,n,denied)
from grasp_failure_prediction.evaluation import object_assets as old
P=Path(__file__).parent;B=P.parent
sha=lambda p:'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path(old.__file__);before=sha(source);text=source.read_text()
a='''        vertices.append(native.mesh_vert[start : start + count] @ rotation.reshape(3, 3).T
                        + native.geom_pos[geom_id])'''
b='''        vertices.append(np.einsum(
            "ij,nj->ni", rotation.reshape(3, 3),
            native.mesh_vert[start : start + count].astype(np.float64), optimize=False,
        ) + native.geom_pos[geom_id])'''
assert text.count(a)==1
candidate=P/'object_assets_candidate.py';candidate.write_text(text.replace(a,b))
(P/'object_assets_einsum_candidate.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),candidate.read_text().splitlines(True),fromfile='a/src/grasp_failure_prediction/evaluation/object_assets.py',tofile='b/src/grasp_failure_prediction/evaluation/object_assets.py')))
name='grasp_failure_prediction.evaluation._unidex_object_assets_candidate'
spec=importlib.util.spec_from_file_location(name,candidate);new=importlib.util.module_from_spec(spec);sys.modules[name]=new;spec.loader.exec_module(new)
rows=[]
for c in json.loads((B/'native_inventory_prepare/conditioning_protocol_v2.json').read_text())['cases']:
 path=B/'conditioning_inputs_v1'/c['case_id']/'geometry/object.xml';row={'case_id':c['case_id'],'geometry_path':str(path)}
 for key,module in [('current',old),('candidate',new)]:
  try:
   value=module.load_object_geometry(path);row[key]={'status':'PASS','geometry_content_hash':value.content_hash,'bounds':value.local_collision_bounds_m}
  except Exception as e:row[key]={'status':'FAIL','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 m=mujoco.MjModel.from_xml_path(str(path));verts=[]
 for g in range(m.ngeom):
  if not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
  mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);b=a+int(m.mesh_vertnum[mid]);R=np.empty(9);mujoco.mju_quat2Mat(R,m.geom_quat[g]);R=R.reshape(3,3)
  for v in m.mesh_vert[a:b]:verts.append([sum(float(R[i,j])*float(v[j]) for j in range(3))+float(m.geom_pos[g,i]) for i in range(3)])
 points=np.asarray(verts);expected=np.array([points.min(0),points.max(0)])
 assert row['candidate']['status']=='PASS',row
 row['candidate']['max_scalar_bounds_error_m']=float(np.max(np.abs(np.asarray(row['candidate']['bounds'])-expected)))
 assert row['candidate']['max_scalar_bounds_error_m']<1e-15
 im=json.loads((path.parent.parent/'object_input.json').read_text());assert row['candidate']['geometry_content_hash']==im['geometry_content_hash']
 rows.append(row)
assert sha(source)==before
r={'status':'STATIC_OBJECT_LOADER_COMPARISON_ONLY','source':{'path':str(source),'sha256':before},'candidate':{'path':str(candidate),'sha256':sha(candidate)},'rows':rows,'physics_steps':0,'model_calls':0,'shared_source_changed':False}
(P/'comparison_v2.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps([{'case_id':r['case_id'],'current':r['current']['status'],'candidate':r['candidate']['status'],'candidate_scalar_error_m':r['candidate']['max_scalar_bounds_error_m']} for r in rows],indent=2))
