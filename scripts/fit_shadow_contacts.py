"""Fit an engineered surface-contact grasp, then test it with actuators.

This changes a robot grasp target and is explicitly not unchanged HUG output.
It does not resize collision surfaces, alter the object, or weld hand to object.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

from actuated_shadow_pilot import ActuatedShadowRunner, setup, run_candidate
from grasp_failure_prediction.evaluation.registry import load_protocol_registry
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--urdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=["x", "y"], default="x")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pose, position, quaternion = setup(args.observation, args.urdf_root)
    protocol = load_protocol_registry().resolve("fixed_grasp_lift_v1")
    protocol = protocol.model_copy(update={"parameters": protocol.parameters.model_copy(update={"grip_command": 1.})})
    runner = ActuatedShadowRunner(protocol, args.urdf_root)
    pose = replace(pose, joint_names=runner._hand_joint_names,
                   qpos=reorder_for_mujoco(pose.joint_names, pose.qpos, runner._hand_joint_names))
    snapshot = np.load(args.observation / "scene_state.npz")
    center = snapshot["qpos"][runner._object_qpos_address:runner._object_qpos_address+3]
    # Opposing cube faces, 0.5mm inside the desired surfaces. These markers are
    # non-colliding query geometry and never appear in the rollout model.
    targets = ({"thdistal": center + [.022, -.008, 0],
                "mfdistal": center + [-.022, .008, 0],
                "ffdistal": center + [-.022, -.008, 0]}
               if args.axis=="x" else
               {"thdistal": center + [0, -.022, 0],
                "mfdistal": center + [.008, .022, 0],
                "ffdistal": center + [-.008, .022, 0]})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"query.xml"
        mujoco.mj_saveLastXML(str(path), runner.model)
        tree = ET.parse(path)
        world = tree.getroot().find("worldbody")
        for name, target in targets.items():
            ET.SubElement(world,"geom",{"name":f"query_{name}","type":"sphere","size":".0001",
                "pos":" ".join(map(str,target)),"contype":"0","conaffinity":"0","rgba":"0 0 0 0"})
        tree.write(path)
        model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    active = [i for i,n in enumerate(pose.joint_names) if n.startswith(("TH", "MF", "FF"))]
    joints = pose.qpos.copy()
    bounds = runner.model.jnt_range[1:25]
    joints = np.clip(joints, bounds[:,0], bounds[:,1])
    initial = np.r_[joints[active], [0.,0.,0.]]
    lower = np.r_[bounds[active,0], [-.025,-.025,-.025]]
    upper = np.r_[bounds[active,1], [.025,.025,.025]]
    geom_groups = {}
    for name in targets:
        bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        geom_groups[name] = [i for i in range(model.ngeom) if model.geom_bodyid[i]==bid]
    marker_ids = {name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f"query_{name}") for name in targets}
    table_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"table")
    contact_residuals = []

    def residual(x):
        q = joints.copy()
        q[active] = x[:-3]
        # Compute a consistent root-to-palm transform with wrist angles fixed.
        root, quat = runner._root_pose_for_palm(position+x[-3:],quaternion,q)
        data.qpos[:] = snapshot["qpos"]
        data.qpos[:3] = root
        data.qpos[3:7] = quat
        data.qpos[runner._hand_qpos_addresses] = q
        mujoco.mj_forward(model,data)
        result = []
        for name, gids in geom_groups.items():
            d = min(mujoco.mj_geomDistance(model,data,g,marker_ids[name],1.,np.zeros(6)) for g in gids)
            result.append(d + .0015)
        # Do not obtain contacts by pushing finger geometry through the table.
        for gids in geom_groups.values():
            d = min(mujoco.mj_geomDistance(model,data,g,table_id,1.,np.zeros(6)) for g in gids)
            result.append(3*min(d,0))
        result.extend(.001*(x[:-3]-initial[:-3]))
        result.extend(.05*x[-3:])
        return np.asarray(result)

    before = residual(initial)[:3]
    solved = least_squares(residual,initial,bounds=(lower,upper),diff_step=1e-4,
                           max_nfev=300,xtol=1e-8,ftol=1e-8,gtol=1e-8)
    after = residual(solved.x)[:3]
    joints[active] = solved.x[:-3]
    engineered = replace(pose,qpos=joints)
    report, trace = run_candidate(runner,engineered,position+solved.x[-3:],quaternion,snapshot)
    report.update({"raw_hug_proposal":False,"engineering_method":"opposing surface contact fit",
                   "contact_fit_before_m":before.tolist(),"contact_fit_after_m":after.tolist(),
                   "solver_converged":bool(solved.success),"solver_evaluations":solved.nfev,
                   "palm_offset_m":solved.x[-3:].tolist(),"collision_shapes_changed":False})
    report['opposing_face_axis']=args.axis
    np.savez(args.output/"trace.npz",qpos=np.stack(runner.saved_qpos),time_s=[s.time_s for s in trace.steps])
    np.savez(args.output/"engineered_target.npz",joints=joints,palm_position_m=position+solved.x[-3:],quaternion_wxyz=quaternion)
    mujoco.mj_saveLastXML(str(args.output/"scene.xml"),runner.model)
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
