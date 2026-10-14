"""Static contact/FK read of saved final state. Integration APIs disabled."""
from pathlib import Path
import json
import numpy as np
import mujoco

def forbid_integration(*args, **kwargs):
    raise RuntimeError('Static saved-state audit forbids MuJoCo time integration')

for guarded_api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, guarded_api, forbid_integration)

root=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/runs/hug_grasp_eval_2026-10-13_fresh_cube')
native=root/'output/proposals/hug_cube/sample_0000/native'
rundir=root/'output/runs/hug_cube_seed0_nominal'
m=mujoco.MjModel.from_binary_path(str(native/'scene.mjb'))
d=mujoco.MjData(m)
s=np.load(rundir/'final_state.npz')
trace=np.load(rundir/'trajectory.npz')
d.qpos[:]=s['qpos']
d.qvel[:]=0
mujoco.mj_forward(m,d)

def body(bid):
    return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(bid))

def geom(gid):
    return mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(gid)) or f'geom:{gid}'

palm=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'palm')
obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
contacts=[]
for i in range(d.ncon):
    c=d.contact[i]
    b=[body(m.geom_bodyid[x]) for x in [c.geom1,c.geom2]]
    contacts.append({'bodies':b,'geoms':[geom(c.geom1),geom(c.geom2)],'distance_m':float(c.dist),'world_position_m':c.pos.tolist(),'touches_object':bool(obj in [m.geom_bodyid[x] for x in [c.geom1,c.geom2]])})
report={
 'audit_type':'Static contact/FK reconstruction from saved final qpos with saved source scene; no integration/actuation/stepping',
 'guarded_integration_apis':['mj_step','mj_step1','mj_step2'],
 'fk_time_after_all_kinematic_calls_s':float(d.time),
 'saved_contact_count':int(trace['contact_count'][-1]),
 'static_reconstructed_contact_count':int(d.ncon),
 'palm_vs_final_trace_position_error_m':float(np.linalg.norm(d.xpos[palm]-trace['palm_position_m'][-1])),
 'object_vs_final_trace_position_error_m':float(np.linalg.norm(d.xpos[obj]-trace['object_position_m'][-1])),
 'object_contact_bodies':sorted({b for c in contacts if c['touches_object'] for b in c['bodies'] if b!='object'}),
 'contacts':contacts,
 'note':'Contact positions/bodies reconstructed statically. Force values are not reconstructed because final_state omits ctrl/mocap/act state; no native MANO collision rollout has been compared.'
}
output=Path('/private/tmp/grasp-mapping-audit-2026-10-14/hug_saved_final_static_contacts.json')
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
