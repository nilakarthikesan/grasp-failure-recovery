"""Reusable hand-prefix command contract for standalone and free-object scenes.

Array/API wrapper only: no MuJoCo import, model constructor, FK or stepping.
The collector must append its one object freejoint after the unchanged native
hand. Unsupported layouts refuse rather than remap or overwrite object state.
"""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
API_PATH=ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py'


def mapping_api():
    spec=importlib.util.spec_from_file_location('new_physics_native_mapping',API_PATH);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


class _HandModelView:
    nq=67;nv=51;nu=51;njnt=19
    def __init__(self,model):self.model=model
    def __getattr__(self,name):return getattr(self.model,name)


@dataclass(frozen=True)
class HandSceneContract:
    hand:object
    global_dimensions:tuple
    object_joint_id:int|None
    object_body_id:int|None
    def metadata(self):
        return dict(schema_version='manosim_native_hand_prefix_scene_contract_v1',global_dimensions=list(self.global_dimensions),hand_qpos_slice=[0,67],hand_dof_slice=[0,51],object_qpos_slice=[67,74] if self.object_joint_id is not None else None,object_dof_slice=[51,57] if self.object_joint_id is not None else None,object_joint_id=self.object_joint_id,object_body_id=self.object_body_id,compiled_hand=self.hand.metadata(),scene_layout='native hand first; optional exactly one appended object freejoint',object_state_mutated_by_controller=False,new_collection_or_training_allowed=False)


def validate_hand_scene(model,resolve_name,assets,rest_quaternions,*,object_joint_name=None,api=None):
    api=mapping_api() if api is None else api;dims=(model.nq,model.nv,model.nu,model.njnt)
    jid=bid=None
    if object_joint_name is None:
        if dims!=(67,51,51,19):raise ValueError('Standalone native hand layout required')
    else:
        if not isinstance(object_joint_name,str) or not object_joint_name or dims!=(74,57,51,20):raise ValueError('Native hand plus exactly one appended object freejoint required')
        jid=int(resolve_name('joint',object_joint_name))
        if jid!=19 or int(model.jnt_type[jid])!=0 or int(model.jnt_qposadr[jid])!=67 or int(model.jnt_dofadr[jid])!=51:raise ValueError('Object freejoint must follow all native hand coordinates')
        bid=int(model.jnt_bodyid[jid]);hand_bodies={int(resolve_name('body',n)) for n in api.BODY_NAMES}
        if bid<=0 or bid in hand_bodies or int(model.body_parentid[bid])!=0 or float(model.body_gravcomp[bid])!=0:raise ValueError('Independent uncompensated object root required')
    hand=api.validate_compiled_contract(_HandModelView(model),resolve_name,assets,rest_quaternions)
    return HandSceneContract(hand,tuple(map(int,dims)),jid,bid)


def verify_hand_scene(model,contract,*,api=None):
    if tuple(map(int,(model.nq,model.nv,model.nu,model.njnt)))!=contract.global_dimensions:raise ValueError('Global scene dimensions changed')
    (mapping_api() if api is None else api).verify_compiled_contract(model,contract.hand)
    return True


def read_hand_state(data,contract):
    q=np.asarray(data.qpos);v=np.asarray(data.qvel)
    if q.shape!=(contract.global_dimensions[0],) or v.shape!=(contract.global_dimensions[1],) or not np.isfinite(q).all() or not np.isfinite(v).all():raise ValueError('Finite actual global state required')
    return q[:67].copy(),v[:51].copy()


def command_from_hand_qpos(qpos67,contract,*,api=None):
    return (mapping_api() if api is None else api).command_from_qpos(qpos67,contract.hand)


def write_hand_controls(data,command,contract):
    """Write only51 controls. No pose/velocity/force/object-state assignment."""
    if getattr(command,'model_signature',None)!=contract.hand.model_signature:raise ValueError('Command/compiled scene identity differs')
    control=np.asarray(command.ctrl)
    if control.shape!=(51,) or np.asarray(data.ctrl).shape!=(51,) or not np.isfinite(control).all():raise ValueError('Finite named51 command required')
    data.ctrl[:]=control
