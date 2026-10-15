"""Opt-in, outcome-independent grasp-derived approaches for recorded Shadow runs.

This is a named controller condition, not a repaired native HUG proposal. The
existing v1/v2 registries and runners stay unchanged. Static samples are finite
collision diagnostics, not a continuous swept-volume or tracking certificate.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Literal

import mujoco
import numpy as np

from .recorded_runner import RecordedActuatedShadowRunner
from .registry import ExecutionProtocolSpec
from .retargeting import RetargetingError,reorder_for_mujoco
from .runner import OPENING_JOINTS
from .schema import canonical_sha256
from .table_clearance import (TableClearanceConfiguration,TableClearancePlan,
                              _descendant,_lowest_world_z,_rotation,plan_table_clearance)


PROTOCOL_ID='hug_grasp_approach_v1'


@dataclass(frozen=True)
class ApproachConfiguration:
    path_samples: int=33
    allow_vertical_fallback: bool=False
    objectward_epsilon_m: float=1e-12

    def __post_init__(self):
        if type(self.path_samples) is not int or not 3<=self.path_samples<=257:
            raise ValueError('path_samples must be an integer in [3,257]')
        if type(self.allow_vertical_fallback) is not bool:
            raise ValueError('allow_vertical_fallback must be boolean')
        if not np.isfinite(self.objectward_epsilon_m) or self.objectward_epsilon_m<0:
            raise ValueError('objectward epsilon must be finite and nonnegative')


class GraspApproachProtocol(ExecutionProtocolSpec):
    execution_protocol_id: Literal['hug_grasp_approach_v1']
    controller: Literal['palm_preserving_wrist_grasp_approach_v1']
    base_protocol_id: Literal['fixed_grasp_lift_v2']
    base_protocol_config_hash: str
    clearance_configuration: dict[str,float|int|bool]
    approach_configuration: dict[str,float|int|bool]


def grasp_approach_protocol(base,*,clearance=None,approach=None):
    if base.execution_protocol_id!='fixed_grasp_lift_v2':
        raise ValueError('grasp-derived approach requires fixed_grasp_lift_v2 input')
    clearance=clearance or TableClearanceConfiguration()
    approach=approach or ApproachConfiguration()
    if clearance.check_closing_targets:
        raise ValueError('this protocol plans the near-open approach only; check_closing_targets must be false')
    return GraspApproachProtocol.model_validate(dict(base.model_dump(mode='json'),
                execution_protocol_id=PROTOCOL_ID,controller='palm_preserving_wrist_grasp_approach_v1',
                base_protocol_id='fixed_grasp_lift_v2',base_protocol_config_hash=canonical_sha256(base),
                clearance_configuration=asdict(clearance),approach_configuration=asdict(approach)))


@dataclass(frozen=True)
class GraspApproachPlan:
    feasible: bool
    clearance_plan: TableClearancePlan
    candidates: tuple[dict,...]
    selected_candidate_id: str|None
    direction_world: np.ndarray|None
    rejection_reason: str|None
    configuration: ApproachConfiguration

    @property
    def adjusted_pose(self):return self.clearance_plan.adjusted_pose

    def as_record(self):
        return dict(schema_version='hug_grasp_approach_plan_v1',execution_protocol_id=PROTOCOL_ID,
                    status='planned' if self.feasible else 'refused_unreachable_pregrasp',outcome=None,
                    feasible=self.feasible,clearance_plan=self.clearance_plan.as_record(),
                    candidates=list(self.candidates),candidate_count=len(self.candidates),
                    selected_candidate_id=self.selected_candidate_id,
                    selected_direction_world=None if self.direction_world is None else self.direction_world.tolist(),
                    rejection_reason=self.rejection_reason,configuration=asdict(self.configuration),
                    selection_rule='first feasible in fixed +localY,-localY,+localZ,-localZ order; optional vertical fallback last; never use task outcomes',
                    static_scope='near-open hand at uniform path samples against table/proximal object geometry',
                    continuous_swept_collision_certificate=False,dynamic_tracking_qualified=False,
                    changes_native_proposal=False,controller_wrist_condition_changed=True)


class ApproachPlanningRejected(ValueError):
    """Unexecuted planner refusal, never a completed physical failure label."""
    mapping_status='unreachable_pregrasp'
    code='approach_plan_refused'
    outcome=None
    def __init__(self,plan):
        self.plan=plan
        super().__init__(plan.rejection_reason)


def approach_candidates(palm_quaternion_wxyz,palm_position_m,object_position_m,*,configuration=None):
    """Keep both declared signs in the denominator; reject non-objectward ones."""
    config=configuration or ApproachConfiguration()
    quaternion=np.asarray(palm_quaternion_wxyz,float);palm=np.asarray(palm_position_m,float);obj=np.asarray(object_position_m,float)
    if (quaternion.shape!=(4,) or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion)-1)>1e-6 or palm.shape!=(3,) or obj.shape!=(3,)
            or not np.isfinite(palm).all() or not np.isfinite(obj).all()):
        raise ValueError('finite palm/object placement and unit palm quaternion required')
    rotation=_rotation(quaternion);towards=obj-palm
    specs=[('palm_plus_y',rotation[:,1]),('palm_minus_y',-rotation[:,1]),
           ('finger_plus_z',rotation[:,2]),('finger_minus_z',-rotation[:,2])]
    if config.allow_vertical_fallback:specs.append(('vertical_negative_z',np.array([0.,0.,-1.])))
    records=[];seen=[]
    for name,direction in specs:
        direction=direction/np.linalg.norm(direction)
        projection=float(direction@towards)
        reason=None
        if projection<=config.objectward_epsilon_m:reason='not_objectward'
        elif any(np.allclose(direction,old,rtol=0,atol=1e-12) for old in seen):reason='duplicate_direction'
        if reason is None:seen.append(direction)
        records.append(dict(candidate_id=name,direction_world=direction.tolist(),objectward_projection_m=projection,
                            orientation_eligible=reason is None,rejection_reason=reason))
    return records


def sampled_approach_clearance(runner,pose,root_position_m,root_quaternion_wxyz,direction_world,*,samples):
    """Measure a declared straight near-open path using scratch state only."""
    if type(samples) is not int or samples<3:raise ValueError('at least three path samples required')
    model=runner.model;names=tuple(runner._hand_joint_names)
    target=reorder_for_mujoco(pose.joint_names,pose.qpos,names)
    joint_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names])
    if np.any(joint_ids<0):raise RetargetingError('missing named hand joint')
    limits=model.jnt_range[joint_ids]
    if np.any(target<limits[:,0]) or np.any(target>limits[:,1]):raise RetargetingError('planned targets must satisfy literal current limits')
    near_open=target.copy()
    for i,name in enumerate(names):
        if name in OPENING_JOINTS:near_open[i]-=.2
    near_open=np.clip(near_open,limits[:,0],limits[:,1])
    direction=np.asarray(direction_world,float);position=np.asarray(root_position_m,float);quaternion=np.asarray(root_quaternion_wxyz,float)
    if (direction.shape!=(3,) or position.shape!=(3,) or quaternion.shape!=(4,)
            or not all(np.isfinite(x).all() for x in (direction,position,quaternion))
            or abs(np.linalg.norm(direction)-1)>1e-10
            or abs(np.linalg.norm(quaternion)-1)>1e-6):raise ValueError('finite unit approach direction and root pose required')
    scratch=mujoco.MjData(model);scratch.qpos[:]=runner.data.qpos
    root=int(runner._root_qpos_address);palm=int(runner._palm_body_id);table=int(runner._table_geom_id)
    active=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in runner._hand_body_ids and (model.geom_contype[g] or model.geom_conaffinity[g])]
    proximal=[g for g in active if not _descendant(model,int(model.geom_bodyid[g]),palm)]
    objects=sorted(runner.object_collision_geom_ids)
    if not active or not proximal or not objects:raise RetargetingError('active hand/proximal/object geometry required')
    rows=[]
    for alpha in np.linspace(0.,1.,samples):
        scratch.qpos[model.jnt_qposadr[joint_ids]]=near_open
        scratch.qpos[root:root+7]=np.r_[position-(1-alpha)*runner.protocol.parameters.pregrasp_distance_m*direction,quaternion]
        mujoco.mj_forward(model,scratch)
        top=float(scratch.geom_xpos[table,2]+model.geom_size[table,2])
        clearance=min(_lowest_world_z(model,scratch,g)-top for g in active)
        distance=min(float(mujoco.mj_geomDistance(model,scratch,g,obj,10.,None)) for g in proximal for obj in objects)
        if not np.isfinite(clearance) or not np.isfinite(distance):raise ValueError('nonfinite static path measurement')
        rows.append(dict(alpha=float(alpha),minimum_table_clearance_m=float(clearance),minimum_proximal_object_distance_m=float(distance)))
    return dict(samples=rows,minimum_table_clearance_m=min(r['minimum_table_clearance_m'] for r in rows),
                minimum_proximal_object_distance_m=min(r['minimum_proximal_object_distance_m'] for r in rows),
                near_open_targets_rad=near_open.tolist(),maximum_sample_spacing_m=float(runner.protocol.parameters.pregrasp_distance_m/(samples-1)),
                continuous_swept_certificate=False)


def plan_grasp_approach(runner,pose,palm_position_m,palm_quaternion_wxyz,*,clearance=None,approach=None):
    config=approach or ApproachConfiguration();clearance=clearance or TableClearanceConfiguration()
    if clearance.check_closing_targets:raise ValueError('this approach protocol admits near-open planning only')
    wrist=plan_table_clearance(runner,pose,palm_position_m,palm_quaternion_wxyz,configuration=clearance)
    obj=runner.data.xpos[runner._object_body_id].copy()
    candidates=approach_candidates(palm_quaternion_wxyz,palm_position_m,obj,configuration=config)
    selected=None
    for row in candidates:
        row['status']='NOT_RUN'
        if not wrist.feasible:row.update(status='REFUSED',rejection_reason='no_feasible_palm_preserving_wrist');continue
        if not row['orientation_eligible']:row['status']='REFUSED';continue
        # Evaluate every declared eligible candidate even after selection, so
        # rejection denominators and comparisons are preserved independently.
        measured=sampled_approach_clearance(runner,wrist.adjusted_pose,wrist.root_position_m,wrist.root_quaternion_wxyz,
                                           row['direction_world'],samples=config.path_samples)
        feasible=measured['minimum_table_clearance_m']>=clearance.clearance_m and measured['minimum_proximal_object_distance_m']>=clearance.proximal_object_clearance_m
        row.update(status='FEASIBLE' if feasible else 'REFUSED',path=measured,
                   rejection_reason=None if feasible else 'sampled_table_or_proximal_clearance')
        if feasible and selected is None:selected=row
    return GraspApproachPlan(selected is not None,wrist,tuple(candidates),
                None if selected is None else selected['candidate_id'],
                None if selected is None else np.asarray(selected['direction_world'],float),
                None if selected is not None else 'no declared wrist/path candidate satisfied the static approach policy',config)


class GraspApproachRecordedRunner(RecordedActuatedShadowRunner):
    """Use the selected static plan with the existing recorded actuated motion."""
    def __init__(self,protocol,*args,**kwargs):
        if not isinstance(protocol,GraspApproachProtocol):raise TypeError('use grasp_approach_protocol to version this controller condition')
        self.clearance_configuration=TableClearanceConfiguration(**protocol.clearance_configuration)
        self.approach_configuration=ApproachConfiguration(**protocol.approach_configuration)
        self.approach_plan=None
        super().__init__(protocol,*args,**kwargs)
    def reset(self,**kwargs):
        super().reset(**kwargs);self.approach_plan=None
    def execute(self,pose,*,grasp_palm_position_m,grasp_palm_quaternion_wxyz,**kwargs):
        if 'approach_direction_world' in kwargs:raise ValueError('approach direction is selected by the declared planner; override forbidden')
        self.approach_plan=plan_grasp_approach(self,pose,grasp_palm_position_m,grasp_palm_quaternion_wxyz,
                                              clearance=self.clearance_configuration,approach=self.approach_configuration)
        if not self.approach_plan.feasible:raise ApproachPlanningRejected(self.approach_plan)
        return super().execute(self.approach_plan.adjusted_pose,
                  grasp_palm_position_m=grasp_palm_position_m,grasp_palm_quaternion_wxyz=grasp_palm_quaternion_wxyz,
                  approach_direction_world=self.approach_plan.direction_world.copy(),**kwargs)
