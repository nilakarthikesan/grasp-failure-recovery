"""Repeat a saved engineered Shadow grasp and compare an open-thumb control."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import mujoco
import numpy as np

from actuated_shadow_pilot import ActuatedShadowRunner, setup, run_candidate
from grasp_failure_prediction.evaluation.registry import load_protocol_registry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation",type=Path,required=True)
    parser.add_argument("--urdf-root",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    pose,_,_=setup(args.observation,args.urdf_root)
    protocol=load_protocol_registry().resolve("fixed_grasp_lift_v1")
    protocol=protocol.model_copy(update={"parameters":protocol.parameters.model_copy(update={"grip_command":1.})})
    target=np.load(args.target)
    # Saved target explicitly uses MuJoCo names; do not assume Dex array order.
    probe=ActuatedShadowRunner(protocol,args.urdf_root)
    pose=replace(pose,joint_names=probe._hand_joint_names,qpos=target['joints'])
    snapshot=np.load(args.observation/'scene_state.npz')
    reports=[]
    traces=[]
    for name in ['repeat_1','repeat_2','repeat_3','open_thumb']:
        runner=ActuatedShadowRunner(protocol,args.urdf_root)
        candidate=pose
        if name=='open_thumb':
            q=pose.qpos.copy()
            for i,joint in enumerate(pose.joint_names):
                if joint.startswith('TH'):
                    q[i]=0
            candidate=replace(pose,qpos=q)
        report,trace=run_candidate(runner,candidate,target['palm_position_m'],target['quaternion_wxyz'],snapshot)
        report.update({'trial':name,'raw_hug_proposal':False,'collision_shapes_changed':False})
        array=np.stack(runner.saved_qpos)
        traces.append(array)
        np.savez(args.output/f'{name}_trace.npz',qpos=array,time_s=[s.time_s for s in trace.steps])
        reports.append(report)
        print(json.dumps(report),flush=True)
    exact_repeat=all(np.array_equal(traces[0],t) for t in traces[1:3])
    valid=(all(r['score']['success'] for r in reports[:3]) and not reports[3]['score']['success']
           and exact_repeat and all(r['peak_force_fraction_of_configured_limit']<=1.000001 for r in reports))
    summary={'baseline_validation_passed':valid,'exact_repeat_qpos':exact_repeat,'trials':reports,
             'scope':'One engineered cube grasp, ideal arm carrier, simulated 24-servo hand; not unchanged HUG or hardware validation'}
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    mujoco.mj_saveLastXML(str(args.output/'scene.xml'),probe.model)
    if not valid:
        raise SystemExit('Baseline validation did not pass; inspect reports')


if __name__=='__main__':main()
