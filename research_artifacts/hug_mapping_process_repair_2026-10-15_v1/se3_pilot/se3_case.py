"""Independent root rotation added to the frozen27D saved-witness Case.

The new search state is [24 joints,3 world palm translations,3 world rotation
vector coordinates]. Scientific surface predicates and original tip caps are
unchanged. Root orientation is a declared new search degree, not a new tolerance.
"""
from pathlib import Path
import sys
import numpy as np

INHERITED=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
sys.path.insert(0,str(INHERITED))
import constraint_engine as base
from se3_math import (decode_state,left_compose_root_quaternion,rotation_bounds_scaled,
    rotation_point_jacobian,ROTATION_SCALE_RAD,ROTATION_COMPONENT_LIMIT_RAD)

base.numba.set_num_threads(2)
assert base.numba.get_num_threads()==2

class Case(base.Case):
    def __init__(self,index):
        super().__init__(index)
        self.original_bounds_27=self.bounds.copy()
        self.bounds=np.concatenate((self.original_bounds_27,rotation_bounds_scaled()))
        self.domain=dict(self.domain)
        self.domain.update(rotation_frame='world_left_composition_about_requested_palm_anchor',
            rotation_vector_scale_rad=ROTATION_SCALE_RAD,
            rotation_component_bounds_rad=[[-ROTATION_COMPONENT_LIMIT_RAD,ROTATION_COMPONENT_LIMIT_RAD]]*3,
            maximum_rotation_vector_norm_rad=float(np.sqrt(3.)*ROTATION_COMPONENT_LIMIT_RAD),
            rotation_bound_is_search_domain_not_acceptance=True,
            original_translation_domain_unchanged=True)
        self.current_root_quaternion=self.rootquat.copy()

    def stage(self,x):
        x=np.asarray(x,dtype=float);q,shift,omega=decode_state(x,self.q0);self.staging_calls+=1
        quat=left_compose_root_quaternion(self.rootquat,omega)
        if np.array_equal(omega,np.zeros(3)):
            corrected=base.stage_translated_palm(self.model,self.data,self.nominal,self.adr,self.rootadr,self.palm,
                self.rootpos,self.rootquat,self.anchor,self.q0,x[:27])
        else:
            self.data.qpos[:]=self.nominal;self.data.qpos[self.adr]=q
            self.data.qpos[self.rootadr:self.rootadr+3]=self.rootpos
            self.data.qpos[self.rootadr+3:self.rootadr+7]=quat
            base.mujoco.mj_forward(self.model,self.data)
            corrected=self.rootpos+(self.anchor+shift-self.data.xpos[self.palm])
            self.data.qpos[self.rootadr:self.rootadr+3]=corrected
            if self.model.nmocap:self.data.mocap_pos[0]=corrected;self.data.mocap_quat[0]=quat
            base.mujoco.mj_forward(self.model,self.data)
        self.current_root_quaternion=quat.copy()
        assert self.data.time==0 and np.array_equal(self.data.qpos[self.objadr:self.objadr+7],self.initial_object)
        assert np.array_equal(self.data.xpos[self.object_body],self.opos) and np.array_equal(self.data.xmat[self.object_body].reshape(3,3),self.oR)
        assert np.array_equal(self.data.qpos[self.rootadr+3:self.rootadr+7],quat)
        assert np.linalg.norm(self.data.xpos[self.palm]-(self.anchor+shift))<1e-12
        return corrected

    def tip_jacobian(self,x):
        self.tip_jacobian_calls+=1;self.stage(x);_,_,omega=decode_state(x,self.q0)
        points=base._link_positions(self.model,self.data,base.a.URDF)[:5]
        old=base.virtual_tip_jacobian(self.model,self.data,self.palm,points,self.tip_bodies,self.dofs)
        result=np.zeros((5,3,30));result[:,:,:27]=old
        for i,point in enumerate(points):
            result[i,:,27:]=rotation_point_jacobian(point-self.data.xpos[self.palm],omega)
        return result

    def exact(self,x):
        # Literal inherited surface calculation; only state decode/domain and
        # explicit optimized-root provenance differ from the old27D wrapper.
        self.exact_calls+=1;raw=self.raw(x)
        hand=base.refresh_colliders(self.original_hand,self.model,self.data,self.opos,self.oR)
        if len(self.oq):
            fields=base.direction.mapped_direction_fields(self.oq,self.on,hand,self.visual_tri,self.orientation_protocol,expected_part=self.op)
            ori=base.orientation_constraints(fields['score'],fields['distance_m'],self.osign,self.oband,
                self.orientation_protocol['facing_dot_numerical_deadzone'],self.orientation_protocol['query_boundary_tolerance_m'],fields['valid'])
        else:
            fields={'valid':np.empty(0,bool),'score':np.empty(0),'distance_m':np.empty(0),'reason_masks':{}}
            ori=base.orientation_constraints(np.empty(0),np.empty(0),self.osign,self.oband,
                self.orientation_protocol['facing_dot_numerical_deadzone'],self.orientation_protocol['query_boundary_tolerance_m'],fields['valid'])
        q,shift,omega=decode_state(x,self.q0)
        aabb=np.asarray(self.domain['world_displacement_AABB_m'])
        in_domain=bool(np.all(x>=self.bounds[:,0]) and np.all(x<=self.bounds[:,1]) and np.all(q>=self.limits[:,0]) and np.all(q<=self.limits[:,1]) and np.all(q>=self.q0-.35) and np.all(q<=self.q0+.35)
            and np.all(shift>=aabb[:,0]) and np.all(shift<=aabb[:,1]) and np.all(np.abs(omega)<=ROTATION_COMPONENT_LIMIT_RAD))
        passed=np.concatenate((raw['proximity']['passed'],ori['passed']))
        exact_margin=np.concatenate((raw['proximity']['margin_m']/.001,ori['sign_margin'],
            ori['band_margin_m']/.001,ori['query_boundary_margin_m']/.001))
        raw.update(exact_orientation_fields=fields,exact_orientation=ori,in_domain=in_domain,
            all_witness_constraints_pass=bool(passed.all()),literal_surface_pass=passed,
            orientation_witness_coverage='PRESENT_CAPPED_EXAMPLES' if len(self.ori_rows) else 'UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS',
            exact_surface_margin_without_optimizer_buffers=exact_margin,
            rank=base.diagnostic_rank(fields['valid'],exact_margin,passed,x),
            root_target_position_m=self.data.qpos[self.rootadr:self.rootadr+3].copy(),
            target_palm_world_position_m=self.data.xpos[self.palm].copy(),target_palm_world_rotation=self.data.xmat[self.palm].reshape(3,3).copy(),
            wrist_world_position_m=self.data.xpos[self.wrist].copy(),wrist_world_rotation=self.data.xmat[self.wrist].reshape(3,3).copy(),
            root_target_quaternion_wxyz=self.current_root_quaternion.copy(),world_root_delta_rotation_vector_rad=omega.copy())
        return raw
