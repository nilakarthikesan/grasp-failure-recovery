"""Supplemental numerical safety-clamp FK audit; no scientific acceptance test."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
counts={'selftest':0,'actual':0,'testing':False}
def no_step(*a,**k):counts['selftest' if counts['testing'] else 'actual']+=1;raise RuntimeError('Static audit forbids integration')
for key in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,key,no_step)
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_link_positions,_body_id,TASK_LINK_NAMES
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root,RetargetedHandPose
path=OUT/'hug_reference_comparison.json';before=sha(path);r=json.loads(path.read_text());v=ShadowPoseValidator(default_dex_urdf_root());counts['testing']=True
for key in ('mj_step','mj_step1','mj_step2'):
 try:getattr(mujoco,key)()
 except RuntimeError:pass
counts['testing']=False
records=[]
for c in r['cases']:
 for name,m in c['methods'].items():
  if m['status']!='computed':continue
  qs=[np.asarray(m[x],float) for x in ('qpos_filtered20_rad','qpos_rad')];points=[]
  for q in qs:
   v.set_pose(RetargetedHandPose(tuple(m['joint_names']),q,np.zeros((21,3)),np.zeros((10,3)),np.eye(4)))
   assert v.data.time==0;points.append(_link_positions(v.model,v.data,v.source_urdf)-v.data.xpos[_body_id(v.model,'palm')])
  delta=np.linalg.norm(points[0]-points[1],axis=1)
  records.append({'native_sha256':c['native_sha256'],'method':name,'max_joint_clamp_rad':m['limits']['max_physical_clamp_delta_rad'],'task_link_position_change_m':delta.tolist(),'maximum_task_link_position_change_m':float(delta.max()),'maximum_fingertip_position_change_m':float(delta[:5].max()),'static_time_s':float(v.data.time)})
summary={}
for name in r['summary']:
 x=[y for y in records if y['method']==name];summary[name]={'case_count':len(x),'maximum_task_link_position_change_m':max(y['maximum_task_link_position_change_m'] for y in x),'maximum_fingertip_position_change_m':max(y['maximum_fingertip_position_change_m'] for y in x),'maximum_joint_clamp_rad':max(y['max_joint_clamp_rad'] for y in x)}
assert sha(path)==before and counts['actual']==0
out={'schema':'supplemental-frozen-physical-limit-clamp-fk','scope':'Forward-kinematic effect of documented <=1.01e-3rad numerical safety clamp. This is no tuning or new grasp sampling; no additional free alignment.','source_report_sha256':before,'script_sha256':sha(__file__),'urdf_sha256':sha(v.source_urdf),'guards':counts,'summary':summary,'cases':records,'new_eval_data':False,'physics_steps':0}
(OUT/'clipping_check.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
