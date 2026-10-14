"""Static math and FK audit of existing saved HUG artifacts. No mj_step calls."""
from pathlib import Path
import hashlib
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def forbid_integration(*args, **kwargs):
    raise RuntimeError('Static fidelity audit forbids MuJoCo time integration')

for guarded_api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, guarded_api, forbid_integration)

from grasp_failure_prediction.integrations.hug import load_hug_prediction
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.evaluation.retargeting import camera_landmarks_to_wrist, RetargetedHandPose, default_dex_urdf_root
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator, _body_id, _link_positions
from grasp_failure_prediction.evaluation.runner import force_close_targets

PROJECT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT = Path('/private/tmp/grasp-mapping-audit-2026-10-14')
RUN = PROJECT / 'runs/hug_grasp_eval_2026-10-13_fresh_cube'
PROPOSAL = RUN / 'output/proposals/hug_cube/sample_0000/proposal.json'
NATIVE = PROPOSAL.parent / 'native'
OLD = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/fresh_object_scene_eval_2026-10-11/inputs/proposals/scene_000_seed_0/proposal.pkl')
g = load_hug_prediction(NATIVE / 'proposal.pkl')
p = json.loads(PROPOSAL.read_text())
c = json.loads((NATIVE / 'scene_contract.json').read_text())
T_wc = np.load(NATIVE / 'T_world_camera.npy')
T_wm = T_wc @ g.T_camera_wrist
A = MANO_TO_OPERATOR_RIGHT
R_wo = T_wm[:3,:3] @ A.T
operator_landmarks = camera_landmarks_to_wrist(g.landmarks_3d, g.T_camera_wrist) @ A.T
tips = np.array([4,8,12,16,20])
middles = np.array([2,6,10,14,18])
human_indices = np.r_[tips, middles]
vectors = operator_landmarks[human_indices] - operator_landmarks[0]
v = ShadowPoseValidator(default_dex_urdf_root())
q = np.array([p['joint_targets_rad'][name] for name in v.joint_names])
pose = RetargetedHandPose(v.joint_names, q, operator_landmarks, vectors, g.T_camera_wrist)
validation = v.validate(pose)
palm_local = v.data.xpos[_body_id(v.model,'palm')].copy()
R_bp = v.data.xmat[_body_id(v.model,'palm')].reshape(3,3).copy()
achieved = _link_positions(v.model,v.data,v.source_urdf) - palm_local
native_world = g.landmarks_3d @ T_wc[:3,:3].T + T_wc[:3,3]
native_mesh_world = g.mesh_vertices @ T_wc[:3,:3].T + T_wc[:3,3]
shadow_world = achieved @ R_wo.T + T_wm[:3,3]
R_obj = Rotation.from_quat(c['object_orientation_xyzw']).as_matrix()
t_obj = np.array(c['object_position_m'])
native_object = (native_world-t_obj) @ R_obj
shadow_object = (shadow_world-t_obj) @ R_obj
native_mesh_object = (native_mesh_world-t_obj) @ R_obj
residual_world = shadow_world-native_world[human_indices]
residual_object = shadow_object-native_object[human_indices]
world_errors = np.linalg.norm(residual_world,axis=1)
object_errors = np.linalg.norm(residual_object,axis=1)
R_opalm = Rotation.from_quat(p['object_to_palm']['quaternion_xyzw']).as_matrix()
R_wp = R_obj @ R_opalm
t_wp = R_obj @ np.array(p['object_to_palm']['translation_m']) + t_obj
R_wo_proposal = R_wp @ R_bp.T
shadow_from_proposal = achieved @ R_wo_proposal.T+t_wp
source_model = mujoco.MjModel.from_binary_path(str(NATIVE/'scene.mjb'))
box_id = mujoco.mj_name2id(source_model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom')
assert source_model.geom_type[box_id] == mujoco.mjtGeom.mjGEOM_BOX
half_extents = source_model.geom_size[box_id].copy()

def box_signed_distance(points):
    d = np.abs(points)-half_extents
    return np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)

native_sd = box_signed_distance(native_object[tips])
shadow_sd = box_signed_distance(shadow_object[:5])
mesh_sd = box_signed_distance(native_mesh_object)
limits=v.model.jnt_range
bound_names=[name for name,value,lim in zip(v.joint_names,q,limits) if min(value-lim[0],lim[1]-value) <= 1e-7]
q_force=force_close_targets(v.joint_names,q,limits,.2)
force_pose=RetargetedHandPose(v.joint_names,q_force,operator_landmarks,vectors,g.T_camera_wrist)
v.set_pose(force_pose)
force_vectors=_link_positions(v.model,v.data,v.source_urdf)-v.data.xpos[_body_id(v.model,'palm')]
force_world=force_vectors@R_wo.T+T_wm[:3,3]
force_object=(force_world-t_obj)@R_obj
force_error=np.linalg.norm(force_world[:5]-native_world[tips],axis=1)
force_movement=np.linalg.norm(force_world[:5]-shadow_world[:5],axis=1)
names=['thumb','index','middle','ring','little']
report={
 'audit_type':'Static algebra and forward kinematics of saved files; no inference, rollout, or fresh evaluation data',
 'guarded_integration_apis':['mj_step','mj_step1','mj_step2'],
 'fk_time_after_all_kinematic_calls_s':float(v.data.time),
 'native_proposal':str(NATIVE/'proposal.pkl'),
 'native_sha256':hashlib.sha256((NATIVE/'proposal.pkl').read_bytes()).hexdigest(),
 'native_equals_previous_scene000_seed0_bytes':(NATIVE/'proposal.pkl').read_bytes()==OLD.read_bytes(),
 'native_camera_landmark_wrist_vs_transform_translation_m':float(np.linalg.norm(g.landmarks_3d[0]-g.T_camera_wrist[:3,3])),
 'axis_conversion_determinant':float(np.linalg.det(A)),
 'axis_conversion_orthogonality_max_error':float(np.max(np.abs(A.T@A-np.eye(3)))),
 'world_to_camera_rotation_determinant':float(np.linalg.det(T_wc[:3,:3])),
 'world_to_camera_rotation_orthogonality_max_error':float(np.max(np.abs(T_wc[:3,:3].T@T_wc[:3,:3]-np.eye(3)))),
 'coordinate_change_roundtrip_max_m':float(np.max(np.linalg.norm(operator_landmarks@R_wo.T+T_wm[:3,3]-native_world,axis=1))),
 'proposal_vs_composed_palm_translation_error_m':float(np.linalg.norm(t_wp-T_wm[:3,3])),
 'proposal_vs_composed_palm_rotation_max_abs_error':float(np.max(np.abs(R_wp-R_wo@R_bp))),
 'proposal_vs_composed_world_shadow_landmark_max_error_m':float(np.max(np.linalg.norm(shadow_from_proposal-shadow_world,axis=1))),
 'palm_relative_vs_world_error_max_delta_m':float(np.max(np.abs(validation.errors_m-world_errors))),
 'palm_relative_vs_object_error_max_delta_m':float(np.max(np.abs(validation.errors_m-object_errors))),
 'mean_tip_error_m':float(np.mean(world_errors[:5])),
 'max_tip_error_m':float(np.max(world_errors[:5])),
 'joint_names_at_limits':bound_names,
 'cube_half_extents_m':half_extents.tolist(),
 'human_wrist_and_shadow_palm_world_m':T_wm[:3,3].tolist(),
 'human_wrist_and_shadow_palm_object_m':native_object[0].tolist(),
 'native_mesh_vertices_inside_cube_count':int((mesh_sd<0).sum()),
 'native_mesh_vertex_box_max_penetration_m':float(max(0,-mesh_sd.min())),
 'native_mesh_vertices_within_2mm_of_cube_surface_count':int((np.abs(mesh_sd)<=.002).sum()),
 'native_mesh_vertices_within_5mm_of_cube_surface_count':int((np.abs(mesh_sd)<=.005).sum()),
 'static_force_close_mean_tip_error_m':float(force_error.mean()),
 'static_force_close_max_tip_error_m':float(force_error.max()),
 'static_force_close_tip_movement_mean_m':float(force_movement.mean()),
 'force_close_joint_changes':[{'joint':name,'saved_target_rad':float(before),'force_close_target_rad':float(after),'actual_added_delta_rad':float(after-before),'limit_min_rad':float(lim[0]),'limit_max_rad':float(lim[1])} for name,before,after,lim in zip(v.joint_names,q,q_force,limits) if abs(after-before)>1e-12],
 'force_close_wrist_joint_max_change_rad':float(max(abs(q_force[i]-q[i]) for i,name in enumerate(v.joint_names) if name.startswith('WR'))),
 'force_close_note':'Static declared +0.2rad joint command, not actual achieved actuator state or a new executed grasp',
 'fingers':[]
}
for i,name in enumerate(names):
    report['fingers'].append({'finger':name,'native_tip_world_m':native_world[tips[i]].tolist(),'shadow_tip_world_m':shadow_world[i].tolist(),'native_tip_object_m':native_object[tips[i]].tolist(),'shadow_tip_object_m':shadow_object[i].tolist(),'world_object_residual_m':float(world_errors[i]),'residual_object_xyz_m':residual_object[i].tolist(),'native_landmark_box_signed_distance_m':float(native_sd[i]),'shadow_landmark_box_signed_distance_m':float(shadow_sd[i]),'static_force_close_tip_movement_m':float(force_movement[i]),'static_force_close_native_tip_error_m':float(force_error[i])})
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'hug_static_fidelity.json').write_text(json.dumps(report,indent=2)+'\n')

colors=['#dc493a','#2084c5','#30a46c','#9569bc','#dc9110']
fig,axes=plt.subplots(1,3,figsize=(17,6.8),layout='constrained')
projections=[(0,2,'Object X','Object Z'),(1,2,'Object Y','Object Z'),(0,1,'Object X','Object Y')]
mm_native=native_object*1000
mm_shadow=shadow_object*1000
mm_box=half_extents*1000
for ax,(a,b,la,lb) in zip(axes,projections):
    ax.add_patch(Rectangle((-mm_box[a],-mm_box[b]),2*mm_box[a],2*mm_box[b],facecolor='#f0f2f5',edgecolor='#333333',lw=1.4,zorder=0))
    for i,(name,color) in enumerate(zip(names,colors)):
        inds=[0,*range(1+4*i,5+4*i)]
        ax.plot(mm_native[inds,a],mm_native[inds,b],'-',color=color,lw=1.5,alpha=.65)
        ax.scatter(mm_native[tips[i],a],mm_native[tips[i],b],marker='o',s=65,facecolors='white',edgecolors=color,linewidths=1.8,zorder=4)
        robot_inds=np.array([i+5,i])
        path=np.vstack((mm_native[0],mm_shadow[robot_inds]))
        ax.plot(path[:,a],path[:,b],'--',color=color,lw=1.5,alpha=.7)
        ax.scatter(mm_shadow[i,a],mm_shadow[i,b],marker='x',s=75,color=color,linewidths=2,zorder=5)
        ax.plot([mm_native[tips[i],a],mm_shadow[i,a]],[mm_native[tips[i],b],mm_shadow[i,b]],color=color,lw=2.3,alpha=.85)
    ax.scatter(mm_native[0,a],mm_native[0,b],marker='s',s=65,color='#222222',zorder=5)
    ax.annotate('MANO wrist / Shadow palm',xy=(mm_native[0,a],mm_native[0,b]),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.scatter(0,0,marker='+',s=75,color='#333333')
    ax.set_xlabel(la+' (mm)');ax.set_ylabel(lb+' (mm)')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    ax.margins(.14)
    ax.set_title(['X–Z view','Y–Z view','X–Y view'][list(axes).index(ax)],fontsize=11)
handles=[Line2D([0],[0],color=co,lw=2,label=f'{name.capitalize()}: {world_errors[i]*1000:.2f} mm') for i,(name,co) in enumerate(zip(names,colors))]
handles += [Line2D([0],[0],color='#555555',marker='o',markerfacecolor='white',label='Native MANO tip / solid skeleton'),Line2D([0],[0],color='#555555',marker='x',ls='--',label='Shadow FK target tip / dashed task links')]
fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False,fontsize=10)
fig.suptitle('Saved HUG cube, seed 0: native MANO vs translated Shadow target\nShared object frame in millimeters; actual camera/world/palm transforms, fixed 50 mm cube\nMean fingertip separation 6.70 mm; largest 14.13 mm. Pose inspection before controller force-close.',fontsize=13)
fig.savefig(OUT/'hug_native_vs_shadow_object_frame.png',dpi=190)
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(17,6.8),layout='constrained')
mm_force=force_object*1000
for ax,(a,b,la,lb) in zip(axes,projections):
    ax.add_patch(Rectangle((-mm_box[a],-mm_box[b]),2*mm_box[a],2*mm_box[b],facecolor='#f0f2f5',edgecolor='#333333',lw=1.4,zorder=0))
    for i,(name,color) in enumerate(zip(names,colors)):
        robot_inds=np.array([i+5,i])
        path=np.vstack((mm_native[0],mm_shadow[robot_inds]))
        force_path=np.vstack((mm_native[0],mm_force[robot_inds]))
        ax.plot(path[:,a],path[:,b],'-',color=color,lw=1.5,alpha=.7)
        ax.plot(force_path[:,a],force_path[:,b],'--',color=color,lw=1.5,alpha=.7)
        ax.scatter(mm_shadow[i,a],mm_shadow[i,b],marker='o',s=65,facecolors='white',edgecolors=color,linewidths=1.8,zorder=4)
        ax.scatter(mm_force[i,a],mm_force[i,b],marker='x',s=75,color=color,linewidths=2,zorder=5)
        ax.annotate('',xy=(mm_force[i,a],mm_force[i,b]),xytext=(mm_shadow[i,a],mm_shadow[i,b]),arrowprops=dict(arrowstyle='->',color=color,lw=1.7))
    ax.scatter(mm_native[0,a],mm_native[0,b],marker='s',s=65,color='#222222',zorder=5)
    ax.scatter(0,0,marker='+',s=75,color='#333333')
    ax.set_xlabel(la+' (mm)');ax.set_ylabel(lb+' (mm)')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.margins(.14)
    ax.set_title(['X–Z view','Y–Z view','X–Y view'][list(axes).index(ax)],fontsize=11)
handles=[Line2D([0],[0],color=co,lw=2,label=f'{name.capitalize()}: {force_movement[i]*1000:.2f} mm displacement') for i,(name,co) in enumerate(zip(names,colors))]
handles += [Line2D([0],[0],color='#555555',marker='o',markerfacecolor='white',label='Saved proposal Shadow target'),Line2D([0],[0],color='#555555',marker='x',ls='--',label='Declared force-close command FK')]
fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=10)
fig.suptitle('Saved HUG cube, seed 0: Shadow target altered by v2 force-close\nShared fixed object frame in millimeters; +0.2 rad flexion before joint-limit clipping\nStatic commanded poses only. Actuated rollout tracks commands subject to object contacts and force limits.',fontsize=13)
fig.savefig(OUT/'hug_shadow_force_close_static_displacement.png',dpi=190)
plt.close(fig)
print(json.dumps(report,indent=2))
