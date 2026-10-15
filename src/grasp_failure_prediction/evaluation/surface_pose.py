"""Current-model 30-coordinate static Shadow placement; no solver or dynamics.

x[:24] are original-relative named joint changes at .025 rad/unit; x[24:27]
are world palm translations at .005 m/unit; x[27:30] are world-left rotation
vector changes at .025 rad/unit. Rotation left-composes the ORIGINAL free-root
orientation. Root translation compensates the joint-dependent palm offset, so
rotation is about the requested palm anchor rather than about the root origin.
Wrist joints may still change palm orientation; all 24 joint coordinates remain.

The caller supplies its immutable original native world-tip targets, literal
historical caps and original nominal q. Caps are never recomputed or rebased.
The default .35-rad original-relative joint box is a search domain, not an
accuracy allowance. Palm bounds must be supplied from the original declared
reach-domain AABB; rotation defaults to the inherited +/-15 degrees/component
search box. These are explicit search domains, never fitted accuracy margins. Source URDF physical limits are literal.

Only a supplied CURRENT MjModel and its original-nominal MjData are consumed.
This adapter owns separate data restored from the same mjSTATE_INTEGRATION on
every placement, leaves caller data/model untouched, and calls mj_forward only.
No historical Case/overlay, inference, optimization, task labels or fidelity
PASS capability is imported. Static realization is not physical tracking.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .pose_validation import TASK_LINK_NAMES, _fixed_tip_offsets

JOINT_SCALE_RAD = .025
PALM_SCALE_M = .005
ROTATION_SCALE_RAD = .025
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
POSE_SCHEMA = "shadow_current_static_named_root_pose_v1"
MODEL_ARRAYS = ("names", "body_parentid", "body_pos", "body_quat", "body_mocapid", "jnt_type", "jnt_bodyid",
    "jnt_pos", "jnt_axis", "jnt_qposadr", "jnt_dofadr", "jnt_range", "geom_type", "geom_bodyid", "geom_pos",
    "geom_quat", "geom_size", "geom_dataid", "geom_contype", "geom_conaffinity", "geom_margin", "geom_gap",
    "mesh_vert", "mesh_face", "mesh_vertadr", "mesh_faceadr", "mesh_vertnum", "mesh_facenum",
    "actuator_trntype", "actuator_trnid", "actuator_gear", "actuator_ctrlrange")


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"finite {shape} {name} required")
    return result


def _readonly(value):
    result = np.array(value, copy=True); result.flags.writeable = False
    return result


def _hash_array(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return "sha256:" + h.hexdigest()


def _model_hash(model):
    h = hashlib.sha256()
    for name in MODEL_ARRAYS:
        value = np.ascontiguousarray(getattr(model, name))
        h.update(name.encode()); h.update(value.dtype.str.encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    h.update(np.asarray(model.opt.gravity).tobytes())
    return "sha256:" + h.hexdigest()


def _quat(value):
    q = _array(value, (4,), "wxyz root quaternion")
    if abs(float(np.linalg.norm(q)) - 1.) > 1e-12:
        raise ValueError("unit root quaternion required; no silent normalization of saved pose")
    return q


def skew(value):
    x,y,z = _array(value, (3,), "rotation vector")
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def left_jacobian_so3(value):
    w = _array(value, (3,), "rotation vector"); k=skew(w); t2=float(w@w)
    if t2 < 1e-8:
        a=.5-t2/24.+t2*t2/720.; b=1./6.-t2/120.+t2*t2/5040.
    else:
        t=np.sqrt(t2); a=(1.-np.cos(t))/t2; b=(t-np.sin(t))/(t2*t)
    return np.eye(3)+a*k+b*(k@k)


def _multiply_quat(a,b):
    return np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]


def left_compose_root(original, omega):
    q = _quat(original); w=_array(omega,(3,),"rotation vector")
    if np.array_equal(w,np.zeros(3)): return q.copy()
    t=float(np.linalg.norm(w)); delta=np.r_[np.cos(t/2.), .5*np.sinc(t/(2.*np.pi))*w]
    result=_multiply_quat(delta,q)
    return result/np.linalg.norm(result)


def _left_rotation_log(target, original):
    q=_quat(target); p=_quat(original); delta=_multiply_quat(q,np.r_[p[0],-p[1:]])
    delta=delta/np.linalg.norm(delta)
    if delta[0]<0: delta=-delta
    sine=float(np.linalg.norm(delta[1:]))
    if sine==0: return np.zeros(3)
    return delta[1:]*(2.*np.arctan2(sine,delta[0])/sine)


def _source_limits(source, names):
    root=ET.parse(source).getroot(); joints={j.get("name"):j for j in root.findall("joint") if j.get("type")=="revolute"}
    if len(names)!=24 or len(set(names))!=24 or set(names)!=set(joints):
        raise ValueError("exactly the 24 named source Shadow revolute joints required")
    result=[]
    for name in names:
        limit=joints[name].find("limit")
        if limit is None: raise ValueError("source joint has no literal limits: "+name)
        lo,hi=float(limit.get("lower")),float(limit.get("upper"))
        if not np.isfinite([lo,hi]).all() or lo>hi: raise ValueError("invalid source joint limits")
        result.append((lo,hi))
    return np.asarray(result)


def _id(model, kind, name):
    value=mujoco.mj_name2id(model,kind,name)
    if value<0: raise ValueError("current compiled model is missing "+name)
    return int(value)


@dataclass(frozen=True)
class SurfacePoseState:
    joint_names: tuple[str,...]
    joint_qpos_rad: np.ndarray
    root_position_m: np.ndarray
    root_quaternion_wxyz: np.ndarray
    palm_position_m: np.ndarray
    palm_rotation: np.ndarray
    tip_positions_world_m: np.ndarray
    tip_errors_world_m: np.ndarray
    tip_error_norms_m: np.ndarray
    original_tip_caps_m: np.ndarray
    literal_tip_caps_pass: np.ndarray
    mean_tip_guard_pass: bool
    physical_joint_limits_pass: bool
    engineering_domain_pass: bool
    source_urdf_sha256: str
    model_geometry_sha256: str

    def serialize_pose(self):
        return dict(schema_version=POSE_SCHEMA,joint_targets_rad=dict(zip(self.joint_names,self.joint_qpos_rad.tolist(),strict=True)),
            root_position_m=self.root_position_m.tolist(),root_quaternion_wxyz=self.root_quaternion_wxyz.tolist(),
            source_urdf_sha256=self.source_urdf_sha256,model_geometry_sha256=self.model_geometry_sha256)


class SurfacePoseAdapter:
    """Independent scratch-data evaluator anchored to one original nominal pose."""
    def __init__(self, model, baseline_data, *, joint_names, source_urdf, original_qpos_rad,
                 native_tip_targets_world_m, original_tip_caps_m, palm_delta_bounds_m, joint_delta_limit_rad=.35,
                 rotation_delta_bounds_rad=None):
        self.model=model; self.source_urdf=Path(source_urdf).resolve(); self.joint_names=tuple(joint_names)
        self.source_urdf_sha256="sha256:"+hashlib.sha256(self.source_urdf.read_bytes()).hexdigest()
        self.physical_bounds=_readonly(_source_limits(self.source_urdf,self.joint_names))
        self.q0=_readonly(_array(original_qpos_rad,(24,),"original nominal joints"))
        self.native_tip_targets_world_m=_readonly(_array(native_tip_targets_world_m,(5,3),"immutable native tip targets"))
        caps=_array(original_tip_caps_m,(5,),"literal original tip caps")
        if np.any(caps<0): raise ValueError("literal tip caps cannot be negative")
        self.original_tip_caps_m=_readonly(caps)
        self.joint_delta_limit_rad=float(joint_delta_limit_rad)
        if not np.isfinite(self.joint_delta_limit_rad) or self.joint_delta_limit_rad<=0:
            raise ValueError("positive finite declared joint delta domain required")
        if palm_delta_bounds_m is None:raise ValueError("original declared palm translation AABB required; no unbounded default")
        self.palm_delta_bounds_m=self._optional_bounds(palm_delta_bounds_m,"palm translation bounds")
        if rotation_delta_bounds_rad is None:rotation_delta_bounds_rad=np.tile([-np.deg2rad(15.),np.deg2rad(15.)],(3,1))
        self.rotation_delta_bounds_rad=self._optional_bounds(rotation_delta_bounds_rad,"rotation-vector bounds")
        self.root_joint_id=_id(model,mujoco.mjtObj.mjOBJ_JOINT,"world_joint")
        if model.jnt_type[self.root_joint_id]!=mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("world_joint must be the actual free root")
        self.root_address=int(model.jnt_qposadr[self.root_joint_id])
        self.root_dof=int(model.jnt_dofadr[self.root_joint_id])
        self.palm_body_id=_id(model,mujoco.mjtObj.mjOBJ_BODY,"palm")
        joints=np.array([_id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.joint_names])
        if np.any(model.jnt_type[joints]!=mujoco.mjtJoint.mjJNT_HINGE): raise ValueError("all named hand joints must be scalar hinges")
        self.joint_addresses=np.asarray(model.jnt_qposadr[joints],int)
        self.joint_dofs=np.asarray(model.jnt_dofadr[joints],int)
        if not np.array_equal(baseline_data.qpos[self.joint_addresses],self.q0):
            raise ValueError("baseline must be the explicit ORIGINAL nominal q; never center on an experimental pose")
        if baseline_data.time!=0: raise ValueError("static original baseline must have zero integration time")
        if not np.all((self.q0>=self.physical_bounds[:,0])&(self.q0<=self.physical_bounds[:,1])):
            raise ValueError("original nominal joints violate literal source limits")
        self.root_position0=_readonly(_array(baseline_data.qpos[self.root_address:self.root_address+3],(3,),"original root position"))
        self.root_quaternion0=_readonly(_quat(baseline_data.qpos[self.root_address+3:self.root_address+7]))
        limits=np.column_stack((np.maximum(self.physical_bounds[:,0],self.q0-self.joint_delta_limit_rad),
                                np.minimum(self.physical_bounds[:,1],self.q0+self.joint_delta_limit_rad)))
        self.joint_bounds=_readonly(limits)
        self.tip_frames=[]
        offsets=_fixed_tip_offsets(self.source_urdf)
        for name in TASK_LINK_NAMES[:5]:
            parent,offset=offsets[name]
            self.tip_frames.append((_id(model,mujoco.mjtObj.mjOBJ_BODY,parent),_readonly(offset)))
        self.actuator_ids=[]
        if model.nu:
            for name,jid in zip(self.joint_names,joints,strict=True):
                aid=_id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,"servo_"+name)
                if model.actuator_trntype[aid]!=mujoco.mjtTrn.mjTRN_JOINT or int(model.actuator_trnid[aid,0])!=jid:
                    raise ValueError("current named servo does not actuate its named joint")
                self.actuator_ids.append(aid)
        self.actuator_ids=np.asarray(self.actuator_ids,int)
        self.mocap_id=None
        if model.nmocap:
            body=_id(model,mujoco.mjtObj.mjOBJ_BODY,"arm_target");self.mocap_id=int(model.body_mocapid[body])
            if self.mocap_id<0: raise ValueError("arm_target must be the current root mocap carrier")
        self._state=np.empty(mujoco.mj_stateSize(model,STATE));mujoco.mj_getState(model,baseline_data,self._state,STATE)
        if not np.isfinite(self._state).all(): raise ValueError("finite original integration state required")
        self._state=_readonly(self._state); self.baseline_state_sha256=_hash_array(self._state)
        self.model_geometry_sha256=_model_hash(model);self.data=mujoco.MjData(model);self.forward_calls=0
        self._restore();self._forward()
        self.palm_position0=_readonly(self.data.xpos[self.palm_body_id])
        self._last_state=None

    @staticmethod
    def _optional_bounds(value,name):
        if value is None:return None
        bounds=_array(value,(3,2),name)
        if np.any(bounds[:,0]>bounds[:,1]) or np.any(bounds[:,0]>0) or np.any(bounds[:,1]<0):
            raise ValueError("declared placement bounds must contain zero")
        return _readonly(bounds)

    def _check_sources(self):
        if "sha256:"+hashlib.sha256(self.source_urdf.read_bytes()).hexdigest()!=self.source_urdf_sha256 or _model_hash(self.model)!=self.model_geometry_sha256:
            raise ValueError("bound source URDF or current model geometry changed")

    def _restore(self):
        self._check_sources();mujoco.mj_resetData(self.model,self.data)
        mujoco.mj_setState(self.model,self.data,self._state.copy(),STATE)

    def _forward(self):
        self.forward_calls+=1;mujoco.mj_forward(self.model,self.data)
        if (self.data.time!=0 or not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.xpos).all() or not np.isfinite(self.data.xmat).all()):
            raise ValueError("nonfinite geometry or time change in static placement")

    def decode(self,x):
        x=_array(x,(30,),"scaled pose state")
        return self.q0+JOINT_SCALE_RAD*x[:24],PALM_SCALE_M*x[24:27],ROTATION_SCALE_RAD*x[27:]

    def _domains(self,q,shift,omega):
        physical=bool(np.all(q>=self.physical_bounds[:,0]) and np.all(q<=self.physical_bounds[:,1]))
        engineering=bool(np.all(q>=self.joint_bounds[:,0]) and np.all(q<=self.joint_bounds[:,1]))
        for values,bounds in ((shift,self.palm_delta_bounds_m),(omega,self.rotation_delta_bounds_rad)):
            if bounds is not None:engineering=engineering and bool(np.all(values>=bounds[:,0]) and np.all(values<=bounds[:,1]))
        return physical,engineering

    def _targets(self,root_position,root_quaternion,q):
        self.data.qpos[self.root_address:self.root_address+3]=root_position
        self.data.qpos[self.root_address+3:self.root_address+7]=root_quaternion
        self.data.qpos[self.joint_addresses]=q
        if len(self.actuator_ids):self.data.ctrl[self.actuator_ids]=q
        if self.mocap_id is not None:
            self.data.mocap_pos[self.mocap_id]=root_position;self.data.mocap_quat[self.mocap_id]=root_quaternion

    def _tips(self):
        return np.array([self.data.xpos[body]+self.data.xmat[body].reshape(3,3)@offset for body,offset in self.tip_frames])

    def _readback(self,q,physical,engineering):
        points=self._tips();errors=points-self.native_tip_targets_world_m
        norms=np.sqrt(np.einsum("ij,ij->i",errors,errors))
        state=SurfacePoseState(self.joint_names,_readonly(q),_readonly(self.data.qpos[self.root_address:self.root_address+3]),
            _readonly(self.data.qpos[self.root_address+3:self.root_address+7]),_readonly(self.data.xpos[self.palm_body_id]),
            _readonly(self.data.xmat[self.palm_body_id].reshape(3,3)),_readonly(points),_readonly(errors),_readonly(norms),
            self.original_tip_caps_m,_readonly(norms<=self.original_tip_caps_m),bool(norms.mean()<=.010),physical,engineering,
            self.source_urdf_sha256,self.model_geometry_sha256)
        self._last_state=state;return state

    def stage(self,x):
        q,shift,omega=self.decode(x);physical,engineering=self._domains(q,shift,omega)
        if not physical or not engineering:raise ValueError("pose outside literal source/original-relative declared domain")
        root_quaternion=left_compose_root(self.root_quaternion0,omega)
        self._restore();self._targets(self.root_position0,root_quaternion,q);self._forward()
        desired=self.palm_position0+shift
        root_position=self.root_position0+(desired-self.data.xpos[self.palm_body_id])
        self._targets(root_position,root_quaternion,q);self._forward()
        if np.max(np.abs(self.data.xpos[self.palm_body_id]-desired))>1e-12:
            raise ValueError("requested palm anchor was not realized")
        return self._readback(q,physical,engineering)

    def tip_jacobian(self,x):
        state=self.stage(x);_,_,omega=self.decode(x)
        palm_jac=np.zeros((3,self.model.nv));mujoco.mj_jac(self.model,self.data,palm_jac,None,state.palm_position_m,self.palm_body_id)
        result=np.zeros((5,3,30))
        for i,(point,(body,_)) in enumerate(zip(state.tip_positions_world_m,self.tip_frames,strict=True)):
            jac=np.zeros((3,self.model.nv));mujoco.mj_jac(self.model,self.data,jac,None,point,body)
            result[i,:,:24]=(jac[:,self.joint_dofs]-palm_jac[:,self.joint_dofs])*JOINT_SCALE_RAD
            result[i,:,24:27]=np.eye(3)*PALM_SCALE_M
            result[i,:,27:]=-skew(point-state.palm_position_m)@left_jacobian_so3(omega)*ROTATION_SCALE_RAD
        return result

    def _named_q(self,values):
        if isinstance(values,Mapping):
            if set(values)!=set(self.joint_names):raise ValueError("exact named joint set required for literal replay")
            values=[values[name] for name in self.joint_names]
        return _array(values,(24,),"named joint positions")

    def encode_pose(self,joint_positions_rad,root_position_m,root_quaternion_wxyz):
        q=self._named_q(joint_positions_rad);position=_array(root_position_m,(3,),"root position");quaternion=_quat(root_quaternion_wxyz)
        self._restore();self._targets(position,quaternion,q);self._forward()
        shift=self.data.xpos[self.palm_body_id]-self.palm_position0
        omega=_left_rotation_log(quaternion,self.root_quaternion0)
        physical,engineering=self._domains(q,shift,omega)
        if not physical or not engineering:raise ValueError("saved pose outside literal original-relative domain")
        x=np.r_[(q-self.q0)/JOINT_SCALE_RAD,shift/PALM_SCALE_M,omega/ROTATION_SCALE_RAD]
        roundtrip=self.stage(x)
        actual_rotation=np.empty(9);requested_rotation=np.empty(9)
        mujoco.mju_quat2Mat(actual_rotation,roundtrip.root_quaternion_wxyz);mujoco.mju_quat2Mat(requested_rotation,quaternion)
        errors=dict(maximum_joint_error_rad=float(np.max(np.abs(roundtrip.joint_qpos_rad-q))),
                    maximum_root_position_error_m=float(np.max(np.abs(roundtrip.root_position_m-position))),
                    maximum_root_rotation_matrix_error=float(np.max(np.abs(actual_rotation-requested_rotation))))
        self.last_encoding_receipt=dict(**errors,comparison_tolerance=1e-12,within_roundtrip_tolerance=all(v<=1e-12 for v in errors.values()),
            exact_literal_command_preservation=False,literal_replay_method="stage_serialized")
        if not self.last_encoding_receipt["within_roundtrip_tolerance"]:raise ValueError("scaled pose encode/decode roundtrip exceeds declared numerical comparison")
        return x

    def stage_serialized(self,payload):
        required={"schema_version","joint_targets_rad","root_position_m","root_quaternion_wxyz","source_urdf_sha256","model_geometry_sha256"}
        if (not isinstance(payload,Mapping) or set(payload)!=required or payload["schema_version"]!=POSE_SCHEMA
                or payload["source_urdf_sha256"]!=self.source_urdf_sha256 or payload["model_geometry_sha256"]!=self.model_geometry_sha256):
            raise ValueError("literal serialized pose source/model/schema mismatch")
        q=self._named_q(payload["joint_targets_rad"]);position=_array(payload["root_position_m"],(3,),"root position");quaternion=_quat(payload["root_quaternion_wxyz"])
        self._restore();self._targets(position,quaternion,q);self._forward()
        shift=self.data.xpos[self.palm_body_id]-self.palm_position0;omega=_left_rotation_log(quaternion,self.root_quaternion0)
        physical,engineering=self._domains(q,shift,omega)
        if not physical or not engineering:raise ValueError("serialized pose outside literal declared domain")
        return self._readback(q,physical,engineering)

    def scaled_bounds(self):
        """Inward representable endpoints; never expand literal q constraints."""
        result=np.tile([-np.inf,np.inf],(30,1))
        result[:24]=(self.joint_bounds-self.q0[:,None])/JOINT_SCALE_RAD
        for i in range(24):
            while self.q0[i]+JOINT_SCALE_RAD*result[i,0]<self.joint_bounds[i,0]:result[i,0]=np.nextafter(result[i,0],np.inf)
            while self.q0[i]+JOINT_SCALE_RAD*result[i,1]>self.joint_bounds[i,1]:result[i,1]=np.nextafter(result[i,1],-np.inf)
        for start,bounds,scale in ((24,self.palm_delta_bounds_m,PALM_SCALE_M),(27,self.rotation_delta_bounds_rad,ROTATION_SCALE_RAD)):
            result[start:start+3]=bounds/scale
            for i in range(3):
                while scale*result[start+i,0]<bounds[i,0]:result[start+i,0]=np.nextafter(result[start+i,0],np.inf)
                while scale*result[start+i,1]>bounds[i,1]:result[start+i,1]=np.nextafter(result[start+i,1],-np.inf)
        return result
