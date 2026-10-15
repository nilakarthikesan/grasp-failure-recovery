"""Current registered scene construction and static named-command placement.

No execute/advance/physics call. A static qpos realization is not servo tracking.
"""
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from grasp_failure_prediction.evaluation.schema import EvaluationCase
from grasp_failure_prediction.evaluation.registry import resolve_case
from grasp_failure_prediction.evaluation.case_runner import _runner_class,_resolve_object_geometry
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco

def build_current_runner(case_payload,project_root,urdf_root):
 case=EvaluationCase.model_validate(case_payload);resolved=resolve_case(case)
 geometry=_resolve_object_geometry(case,Path(project_root));cls=_runner_class(resolved)
 runner=cls(resolved.execution_protocol,urdf_root,physics_timestep_s=resolved.environment.simulator.physics_timestep_s,control_timestep_s=resolved.environment.simulator.control_timestep_s,object_geometry=geometry)
 return runner,resolved,geometry

def reset_captured_scene(runner,case_payload,contract):
 q=np.asarray(contract['object_orientation_xyzw'],float);q=q/np.linalg.norm(q)
 runner.reset(seed=case_payload['seed'],object_mass_kg=float(contract['mass_kg']),object_position_m=np.asarray(contract['object_position_m'],float),object_orientation_wxyz=q[[3,0,1,2]])
 return np.asarray(contract['object_position_m'],float),q[[3,0,1,2]]

def stage_named_root(runner,names,values,root_position,root_quaternion):
 target=reorder_for_mujoco(tuple(names),np.asarray(values,float),runner._hand_joint_names)
 runner._set_kinematic_targets(np.asarray(root_position,float),np.asarray(root_quaternion,float),target)
 if runner.model.nmocap:
  arm=mujoco.mj_name2id(runner.model,mujoco.mjtObj.mjOBJ_BODY,'arm_target');mi=int(runner.model.body_mocapid[arm]);assert mi>=0
  runner.data.mocap_pos[mi]=root_position;runner.data.mocap_quat[mi]=root_quaternion
 if hasattr(runner,'_actuator_ids'):runner.data.ctrl[runner._actuator_ids]=target
 mujoco.mj_forward(runner.model,runner.data)
 assert runner.data.time==0
 return target

def stage_named_palm(runner,names,values,palm_world):
 target=reorder_for_mujoco(tuple(names),np.asarray(values,float),runner._hand_joint_names)
 q=Rotation.from_matrix(np.asarray(palm_world)[:3,:3]).as_quat()[[3,0,1,2]]
 root_position,root_quaternion=runner._root_pose_for_palm(np.asarray(palm_world)[:3,3],q,target)
 stage_named_root(runner,names,values,root_position,root_quaternion)
 return root_position,root_quaternion,target
