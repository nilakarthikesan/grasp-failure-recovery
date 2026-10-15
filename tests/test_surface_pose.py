"""Current compiled Shadow geometry; manufactured commands, no optimization."""
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from grasp_failure_prediction.evaluation import surface_pose as pose
from grasp_failure_prediction.evaluation.mapping_validation import source_fk
from grasp_failure_prediction.evaluation.pose_validation import TASK_LINK_NAMES
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root
from grasp_failure_prediction.evaluation.runner import ActuatedShadowRunner


@pytest.fixture(autouse=True)
def no_dynamics_or_solving(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail("no physics or optimization in static current-scene controls")
    for name in ("mj_step","mj_step1","mj_step2"):monkeypatch.setattr(mujoco,name,forbidden)
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
    monkeypatch.setattr(ShadowHandRetargeter,"retarget",forbidden)


def integration(model,data):
    result=np.empty(mujoco.mj_stateSize(model,pose.STATE));mujoco.mj_getState(model,data,result,pose.STATE)
    return result


def rotation(q):
    out=np.empty(9);mujoco.mju_quat2Mat(out,q);return out.reshape(3,3)


@pytest.fixture
def current():
    root=default_dex_urdf_root();urdf=root/"shadow_hand/shadow_hand_right.urdf"
    runner=ActuatedShadowRunner(load_protocol_registry().resolve("fixed_grasp_lift_v2"),root)
    runner.reset(seed=0,object_mass_kg=.15,object_position_m=np.array([.4,0,.025]),object_orientation_wxyz=np.array([1.,0,0,0]))
    names=runner._hand_joint_names;limits=pose._source_limits(urdf,names);q=limits.mean(axis=1)
    p=np.array([.1,-.2,1.2]);quat=np.array([.93,.12,-.19,.22]);quat/=np.linalg.norm(quat)
    runner._set_kinematic_targets(p,quat,q);runner.data.ctrl[runner._actuator_ids]=q
    runner.data.mocap_pos[0]=p;runner.data.mocap_quat[0]=quat;mujoco.mj_forward(runner.model,runner.data)
    fk,_=source_fk(urdf,names,q);carrier=np.eye(4);carrier[:3,:3]=rotation(quat);carrier[:3,3]=p
    root_inverse=np.linalg.inv(fk["forearm"])
    independent_tips=np.array([(carrier@root_inverse@fk[name])[:3,3] for name in TASK_LINK_NAMES[:5]])
    native=independent_tips+np.array([[.001,-.002,.003]]*5)
    caps=np.array([.001,.002,.003,.004,.005])
    arguments=dict(joint_names=names,source_urdf=urdf,original_qpos_rad=q,native_tip_targets_world_m=native,
        original_tip_caps_m=caps,palm_delta_bounds_m=np.tile([-.05,.05],(3,1)))
    return SimpleNamespace(runner=runner,arguments=arguments,independent_tips=independent_tips)


def adapter(current,**overrides):
    return pose.SurfacePoseAdapter(current.runner.model,current.runner.data,**{**current.arguments,**overrides})


def test_current_importer_geometry_and_independent_source_fk(current):
    source=integration(current.runner.model,current.runner.data).copy();geometry=pose._model_hash(current.runner.model)
    a=adapter(current);state=a.stage(np.zeros(30))
    assert len(a.joint_names)==24 and a.data is not current.runner.data
    assert current.runner.model.ngeom>=45 and current.runner.model.nu==24
    np.testing.assert_allclose(state.tip_positions_world_m,current.independent_tips,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(state.joint_qpos_rad,current.arguments["original_qpos_rad"])
    np.testing.assert_array_equal(state.root_position_m,a.root_position0)
    np.testing.assert_array_equal(state.root_quaternion_wxyz,a.root_quaternion0)
    np.testing.assert_array_equal(integration(current.runner.model,current.runner.data),source)
    assert pose._model_hash(current.runner.model)==geometry


def test_original_caps_and_native_targets_copied_without_any_rebase(current):
    a=adapter(current);expected_caps=current.arguments["original_tip_caps_m"].copy();targets=current.arguments["native_tip_targets_world_m"].copy()
    current.arguments["original_tip_caps_m"][:]=99;current.arguments["native_tip_targets_world_m"][:]=99
    state=a.stage(np.zeros(30))
    np.testing.assert_array_equal(state.original_tip_caps_m,expected_caps)
    np.testing.assert_array_equal(a.native_tip_targets_world_m,targets)
    np.testing.assert_array_equal(state.literal_tip_caps_pass,[False,False,False,True,True])
    assert state.mean_tip_guard_pass
    with pytest.raises(ValueError):a.original_tip_caps_m[0]=1


def test_palm_anchor_compensation_for_wrist_and_left_rotation(current):
    a=adapter(current);x=np.zeros(30);x[:2]=[1.2,-.8];x[24:27]=[.4,-.6,.8];x[27:]=[1.,-.7,.5]
    state=a.stage(x)
    np.testing.assert_allclose(state.palm_position_m,a.palm_position0+.005*x[24:27],rtol=0,atol=1e-12)
    np.testing.assert_allclose(rotation(state.root_quaternion_wxyz),rotation(pose.left_compose_root(a.root_quaternion0,.025*x[27:])),rtol=0,atol=1e-14)
    assert np.linalg.norm(state.root_position_m-(a.root_position0+.005*x[24:27]))>1e-5
    np.testing.assert_array_equal(a.data.ctrl[a.actuator_ids],state.joint_qpos_rad)
    np.testing.assert_array_equal(a.data.mocap_pos[a.mocap_id],state.root_position_m)
    np.testing.assert_array_equal(a.data.mocap_quat[a.mocap_id],state.root_quaternion_wxyz)


def test_A_B_A_restores_complete_state_and_object_and_caller(current):
    a=adapter(current);caller=integration(current.runner.model,current.runner.data).copy();object_address=current.runner._object_qpos_address
    x=np.zeros(30);x[:24]=np.linspace(-.2,.2,24);x[24:27]=[.2,.3,-.1];x[27:]=[.5,-.4,.2]
    first=a.stage(x);state=integration(a.model,a.data).copy();other=np.zeros(30);other[24]=2;other[27]=1.1
    second=a.stage(other)
    preserved=np.ones(a.model.nq,dtype=bool);preserved[a.joint_addresses]=False;preserved[a.root_address:a.root_address+7]=False
    np.testing.assert_array_equal(a.data.qpos[preserved],current.runner.data.qpos[preserved])
    assert a.data.time==0
    a.data.ctrl[:]=.123;a.data.mocap_pos[:]=3;a.data.qacc_warmstart[:]=7
    again=a.stage(x)
    np.testing.assert_array_equal(integration(a.model,a.data),state)
    np.testing.assert_array_equal(first.tip_positions_world_m,again.tip_positions_world_m)
    np.testing.assert_array_equal(integration(current.runner.model,current.runner.data),caller)
    np.testing.assert_array_equal(a.data.qpos[object_address:object_address+7],current.runner.data.qpos[object_address:object_address+7])
    assert a.data.time==0


@pytest.mark.parametrize("nonzero_rotation",[False,True])
def test_analytical_scaled_tip_jacobian_matches_central_differences(current,nonzero_rotation):
    a=adapter(current);x=np.zeros(30);x[:24]=np.linspace(-3.,3.,24);x[24:27]=[.2,-.3,.1]
    if nonzero_rotation:x[27:]=[6.,-5.,4.]
    analytical=a.tip_jacobian(x);finite=np.empty_like(analytical);h=1e-4
    for i in range(30):
        d=np.zeros(30);d[i]=h
        finite[:,:,i]=(a.stage(x+d).tip_positions_world_m-a.stage(x-d).tip_positions_world_m)/(2*h)
    np.testing.assert_allclose(analytical,finite,rtol=0,atol=2e-9)
    np.testing.assert_allclose(analytical[:,:,24:27],np.broadcast_to(np.eye(3)*.005,(5,3,3)),rtol=0,atol=0)


def test_literal_serialization_and_experimental_encoding_keep_original_center(current):
    a=adapter(current);x=np.zeros(30);x[:24]=np.linspace(-.4,.4,24);x[24:27]=[1.,-.5,.25];x[27:]=[.8,-.4,.3]
    result=a.stage(x);literal=json.loads(json.dumps(result.serialize_pose()))
    # Reordered JSON dictionaries cannot reorder physical named commands.
    literal["joint_targets_rad"]=dict(reversed(list(literal["joint_targets_rad"].items())))
    replay=a.stage_serialized(literal)
    np.testing.assert_array_equal(replay.joint_qpos_rad,result.joint_qpos_rad)
    np.testing.assert_array_equal(replay.root_position_m,result.root_position_m)
    np.testing.assert_array_equal(replay.root_quaternion_wxyz,result.root_quaternion_wxyz)
    encoded=a.encode_pose(literal["joint_targets_rad"],literal["root_position_m"],literal["root_quaternion_wxyz"])
    np.testing.assert_allclose(encoded,x,rtol=0,atol=1e-12)
    assert a.last_encoding_receipt["within_roundtrip_tolerance"]
    assert a.last_encoding_receipt["exact_literal_command_preservation"] is False
    np.testing.assert_array_equal(a.q0,current.arguments["original_qpos_rad"])
    np.testing.assert_array_equal(a.original_tip_caps_m,current.arguments["original_tip_caps_m"])


@pytest.mark.parametrize("change",["duplicate_names","experimental_center","no_palm_domain","negative_caps","nonzero_time","bad_quaternion"])
def test_invalid_baseline_or_original_contract_refuses(current,change):
    overrides={}
    if change=="duplicate_names":overrides["joint_names"]=[current.arguments["joint_names"][0]]*24
    elif change=="experimental_center":overrides["original_qpos_rad"]=current.arguments["original_qpos_rad"]+.001
    elif change=="no_palm_domain":overrides["palm_delta_bounds_m"]=None
    elif change=="negative_caps":overrides["original_tip_caps_m"]=[-.001]*5
    elif change=="nonzero_time":current.runner.data.time=1
    else:current.runner.data.qpos[current.runner._root_qpos_address+3:current.runner._root_qpos_address+7]=[2,0,0,0]
    with pytest.raises(ValueError):adapter(current,**overrides)


def test_literal_domains_no_clamping_or_tolerance_and_inward_scaled_endpoints(current):
    a=adapter(current);bounds=a.scaled_bounds()
    assert bounds.shape==(30,2) and np.isfinite(bounds).all()
    for i in range(30):
        for side in (0,1):
            x=np.zeros(30);x[i]=bounds[i,side];state=a.stage(x)
            assert state.physical_joint_limits_pass and state.engineering_domain_pass
    for i in (0,24,27):
        x=np.zeros(30);x[i]=bounds[i,1]+1e-5
        with pytest.raises(ValueError,match="domain"):a.stage(x)
    x=np.zeros(30);x[0]=np.nan
    with pytest.raises(ValueError):a.stage(x)


def test_changed_geometry_or_literal_serialized_source_refuses(current):
    a=adapter(current);saved=a.stage(np.zeros(30)).serialize_pose();saved["source_urdf_sha256"]="sha256:"+"0"*64
    with pytest.raises(ValueError,match="mismatch"):a.stage_serialized(saved)
    a.model.geom_size[0,0]+=.001
    with pytest.raises(ValueError,match="geometry changed"):a.stage(np.zeros(30))


def test_nonzero_current_body_transforms_match_independent_literal_source_fk(current):
    a=adapter(current);x=np.zeros(30);x[:24]=np.linspace(-4.,4.,24);x[24:27]=[1.2,-.8,.5];x[27:]=[6.,-5.,4.]
    state=a.stage(x)
    fk,_=source_fk(a.source_urdf,a.joint_names,state.joint_qpos_rad)
    carrier=np.eye(4);carrier[:3,:3]=rotation(state.root_quaternion_wxyz);carrier[:3,3]=state.root_position_m
    base=carrier@np.linalg.inv(fk["forearm"]);checked=0
    for body in range(a.model.nbody):
        name=mujoco.mj_id2name(a.model,mujoco.mjtObj.mjOBJ_BODY,body)
        if body==0 or name not in fk:continue
        expected=base@fk[name]
        np.testing.assert_allclose(a.data.xpos[body],expected[:3,3],rtol=0,atol=1e-10)
        np.testing.assert_allclose(a.data.xmat[body].reshape(3,3),expected[:3,:3],rtol=0,atol=1e-10)
        checked+=1
    assert checked>=25
    tips=np.array([(base@fk[name])[:3,3] for name in TASK_LINK_NAMES[:5]])
    np.testing.assert_allclose(state.tip_positions_world_m,tips,rtol=0,atol=1e-10)
    assert a.data.time==0


def test_source_joint_limits_are_literal_and_override_search_box(current):
    original=current.arguments["original_qpos_rad"].copy();names=current.arguments["joint_names"]
    root=ET.parse(current.arguments["source_urdf"]).getroot()
    literal={j.get("name"):(float(j.find("limit").get("lower")),float(j.find("limit").get("upper")))
             for j in root.findall("joint") if j.get("type")=="revolute"}
    original[0]=literal[names[0]][0]+.01
    runner=current.runner;position=runner.data.qpos[runner._root_qpos_address:runner._root_qpos_address+3].copy()
    quaternion=runner.data.qpos[runner._root_qpos_address+3:runner._root_qpos_address+7].copy()
    runner._set_kinematic_targets(position,quaternion,original);mujoco.mj_forward(runner.model,runner.data)
    a=adapter(current,original_qpos_rad=original)
    np.testing.assert_array_equal(a.physical_bounds,np.array([literal[name] for name in names]))
    assert a.joint_bounds[0,0]==literal[names[0]][0]
    x=np.zeros(30);x[0]=a.scaled_bounds()[0,0];placed=a.stage(x)
    assert placed.joint_qpos_rad[0]>=literal[names[0]][0]
    x[0]-=1e-5
    with pytest.raises(ValueError,match="literal"):a.stage(x)
