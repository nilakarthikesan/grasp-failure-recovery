#!/usr/bin/env python3
"""Finite, frozen, static fresh-solve qualification on four saved HUG inputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import mujoco


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def prepare(args):
    root=Path(args.project).resolve();out=Path(args.output).resolve()
    if (out/'protocol.json').exists():
        raise FileExistsError('choose a new versioned output directory')
    prior=root/'research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json'
    old=json.loads(prior.read_text())
    sys.path.insert(0,str(root/'src'));sys.path.extend(old['dependency_dirs'])
    from dex_retargeting.constants import HandType,RetargetingType,RobotName,get_default_config_path
    from grasp_failure_prediction.evaluation.retargeting import LEGACY_RETARGETING_PROFILE,STATIC_RETARGETING_PROFILE
    import dex_retargeting
    import nlopt
    config=Path(get_default_config_path(RobotName.shadow,RetargetingType.vector,HandType.right))
    inventory=Path(old['inventory']);cases=json.loads(inventory.read_text())['modern_cases']
    chosen=[0,8,16,24]
    paths=[inventory,prior,config]
    sources=[Path(__file__).resolve(),root/'src/grasp_failure_prediction/evaluation/fresh_mapping_qualification.py',
             root/'tests/test_fresh_mapping_qualification.py',root/'src/grasp_failure_prediction/evaluation/retargeting.py',
             root/'src/grasp_failure_prediction/evaluation/pose_validation.py',root/'src/grasp_failure_prediction/evaluation/case_runner.py',
             root/'src/grasp_failure_prediction/integrations/hug.py',root/'src/grasp_failure_prediction/integrations/hug_frames.py',
             root/'src/grasp_failure_prediction/proposals/translation.py',root/'src/grasp_failure_prediction/proposals/configs/translations.yaml',
             root/'research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/current_execution_bridge/xml_reference.py',
             root/'research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/current_scene_bridge_v2/native_reference.py']
    paths.extend(sources)
    paths.extend(Path(dex_retargeting.__file__).parent.rglob('*.py'))
    paths.extend(p for p in Path(nlopt.__file__).parent.iterdir() if p.suffix in ('.py','.so'))
    urdf=root/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
    paths.append(urdf);paths.extend(urdf.parent.rglob('*.obj'))
    frozen=[]
    for i in chosen:
        case=cases[i]
        for field in ('native','stored_shadow_command'):
            if sha(case[field]['path']) != case[field]['sha256']:
                raise ValueError(f'inventory input changed: {i}/{field}')
            paths.append(Path(case[field]['path']))
        native=case['native']
        paths.append(Path(native['inference_report_path']))
        if sha(native['inference_report_path']) != native['inference_report_sha256']:
            raise ValueError('inference provenance changed')
        for entry in case['frame_files'].values():
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('frame inventory changed')
            paths.append(Path(entry['path']))
        frozen.append(dict(index=i,object_id=case['object_id'],native_sha256=native['sha256'],
                           checkpoint_sha256=native['checkpoint_sha256'],hug_code_commit=native['hug_code_commit'],
                           input_tensor_hash=native['input_tensor_hash'],inference_report_sha256=native['inference_report_sha256']))
    out.mkdir(parents=True,exist_ok=True);(out/'source').mkdir()
    for path in sources:
        shutil.copyfile(path,out/'source'/path.name)
    settings={profile:dict(profile_id=profile,config_sha256=sha(config),vector_scaling=1.,low_pass_filter_enabled=False,
                           temporal_norm_delta=.004 if profile==LEGACY_RETARGETING_PROFILE else 0.,
                           reset_before_proposal=profile==STATIC_RETARGETING_PROFILE,refinement_calls=20)
              for profile in (LEGACY_RETARGETING_PROFILE,STATIC_RETARGETING_PROFILE)}
    p=dict(schema='hug_fresh_mapping_static_qualification_v1',project=str(root),dependency_dirs=old['dependency_dirs'],
           inventory=str(inventory),source_urdf=str(urdf),cases=frozen,profiles=settings,
           bindings={str(path.resolve()):sha(path) for path in paths},
           reference_xml=str(sources[-2]),reference_native_frame=str(sources[-1]),
           numerical_gates=dict(repeat_command_rad=1e-10,compiled_fk_position_m=1e-10,compiled_rotation_matrix=1e-10,
                                scalar_frame_formula_m=1e-10,native_roundtrip_m=1e-6,literal_source_limit_tolerance_rad=0.,mean_fingertip_guard_m=.010),
           selection='fixed indices0,8,16,24: first original saved proposal per object; no outcome/error filtering; development only',
           schedule=dict(fresh_reference=chosen,fresh_repeat=chosen,reused_forward=chosen,reused_reverse=chosen[::-1],reused_A_B_A=[0,8,0]),
           maximum_project_retarget_calls=38,maximum_inner_refinement_calls=760,
           call_order_scope='fresh-instance reproducibility is separate from reused-instance order independence',
           historical_scope='fresh-vs-saved target/root differences descriptive only; no forced historical equality',
           acceptance_scope='per-case actual Shadow command input/FK/engineering guard with observed provenance; no native surface, physical tracking or new-input generalization claim',
           provider_scope='translation continues default legacy fresh-per-proposal; audit records extra raw/status evidence outside canonical provider; no promotion/default change',
           physics_steps_allowed=0,new_inference_allowed=False,new_outcomes_allowed=False)
    write(out/'protocol.json',p)
    print(json.dumps({'protocol':str(out/'protocol.json'),'sha256':sha(out/'protocol.json')}))


def qualify_pose(pose,record,grasp,case,validator,floating,fdata,xml,native_ref,pv,world_palm_helper,p):
    from grasp_failure_prediction.evaluation.fresh_mapping_qualification import independent_local_operator_landmarks
    source=xml.read_urdf(p['source_urdf']);names=pose.joint_names;q=pose.qpos
    limits=xml.source_limit_rows(source,names,q)
    fixed=xml.forward_xml(source,names,q)
    validation=validator.validate(pose)
    compiled=pv._link_positions(validator.model,validator.data,Path(p['source_urdf']))
    expected=np.array([fixed[name][:3,3] for name in pv.TASK_LINK_NAMES])
    local=independent_local_operator_landmarks(grasp)
    frame_error=float(np.max(np.linalg.norm(local-pose.landmarks_wrist_m,axis=1)))
    # Removing/reapplying a float-valued source rotation is not exact if it has
    # tiny orthogonality defects. Report that precision separately from fitting.
    source_R=grasp.T_camera_wrist[:3,:3]
    local_mano=(grasp.landmarks_3d-grasp.T_camera_wrist[:3,3])@source_R
    roundtrip=float(np.linalg.norm(local_mano@source_R.T+grasp.T_camera_wrist[:3,3]-grasp.landmarks_3d,axis=1).max())
    camera=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
    reference=native_ref.expected_native_palm(camera,grasp.T_camera_wrist,fixed['palm'])
    position,quaternion=world_palm_helper(grasp,pose,validator,Path(case['native']['path']).parent)
    actual_palm=xml.transform(position,xml.quaternion_matrix_wxyz(quaternion))
    # Independent source-XML root inversion; no production runner or controller.
    relative=xml.forward_xml(source,names,q,np.eye(4))['palm']
    carrier=actual_palm@np.linalg.inv(relative)
    carrier_q=native_ref.quaternion_from_matrix(carrier[:3,:3])
    fdata.qpos[:]=floating.qpos0;fdata.qvel[:]=0
    for name,value in zip(names,q,strict=True):
        j=mujoco.mj_name2id(floating,mujoco.mjtObj.mjOBJ_JOINT,name)
        fdata.qpos[int(floating.jnt_qposadr[j])]=value
    j=mujoco.mj_name2id(floating,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');a=int(floating.jnt_qposadr[j])
    fdata.qpos[a:a+7]=np.r_[carrier[:3,3],carrier_q]
    mujoco.mj_forward(floating,fdata)
    realized=pv._link_positions(floating,fdata,Path(p['source_urdf']))
    expected_world=xml.forward_xml(source,names,q,xml.transform(carrier[:3,3],xml.quaternion_matrix_wxyz(carrier_q)))
    world=np.array([expected_world[name][:3,3] for name in pv.TASK_LINK_NAMES])
    palm_id=mujoco.mj_name2id(floating,mujoco.mjtObj.mjOBJ_BODY,'palm')
    tol=p['numerical_gates']
    measurements=dict(fixed_fk_max_error_m=float(np.linalg.norm(compiled-expected,axis=1).max()),
                      world_fk_max_error_m=float(np.linalg.norm(realized-world,axis=1).max()),
                      native_palm_position_error_m=float(np.linalg.norm(position-reference['palm_world'][:3,3])),
                      native_palm_rotation_error=float(np.max(abs(actual_palm[:3,:3]-reference['palm_world'][:3,:3]))),
                      compiled_palm_position_error_m=float(np.linalg.norm(fdata.xpos[palm_id]-actual_palm[:3,3])),
                      compiled_palm_rotation_error=float(np.max(abs(fdata.xmat[palm_id].reshape(3,3)-actual_palm[:3,:3]))),
                      independent_local_frame_formula_error_m=frame_error,native_frame_roundtrip_error_m=roundtrip,
                      source_rotation_orthogonality_error=float(np.max(abs(source_R.T@source_R-np.eye(3)))),
                      mean_fingertip_error_m=validation.mean_fingertip_error_m,maximum_fingertip_error_m=validation.maximum_fingertip_error_m,
                      per_fingertip_errors_m=validation.fingertip_errors_m.tolist(),literal_source_limits=limits,
                      world_palm_position_m=position.tolist(),world_palm_quaternion_wxyz=quaternion.tolist(),
                      carrier_position_m=carrier[:3,3].tolist(),carrier_quaternion_wxyz=carrier_q.tolist())
    checks=dict(literal_source_limits=all(row['literal_pass'] for row in limits),
                fixed_fk=measurements['fixed_fk_max_error_m']<=tol['compiled_fk_position_m'],
                world_fk=measurements['world_fk_max_error_m']<=tol['compiled_fk_position_m'],
                frame_formula=frame_error<=tol['scalar_frame_formula_m'],native_roundtrip=roundtrip<=tol['native_roundtrip_m'],
                native_palm_position=measurements['native_palm_position_error_m']<=tol['compiled_fk_position_m'],
                native_palm_rotation=measurements['native_palm_rotation_error']<=tol['compiled_rotation_matrix'],
                compiled_palm_position=measurements['compiled_palm_position_error_m']<=tol['compiled_fk_position_m'],
                compiled_palm_rotation=measurements['compiled_palm_rotation_error']<=tol['compiled_rotation_matrix'],
                engineering_mean_tip_guard=validation.mean_fingertip_error_m<=tol['mean_fingertip_guard_m'])
    with np.load(case['stored_shadow_command']['path'],allow_pickle=False) as z:
        row=case['stored_shadow_command']['row_index'];old=dict(zip(z['joint_names'].tolist(),z['nominal_target_rad'][row],strict=True))
        saved=np.array([old[name] for name in names])
        old_root=z['root_target_position_m'][row]
    measurements['fresh_vs_saved_max_joint_difference_rad']=float(abs(q-saved).max())
    measurements['fresh_vs_saved_carrier_position_difference_m']=float(np.linalg.norm(carrier[:3,3]-old_root))
    measurements['exact_named_command_readback']=bool(all(fdata.qpos[int(floating.jnt_qposadr[mujoco.mj_name2id(floating,mujoco.mjtObj.mjOBJ_JOINT,name)])]==value for name,value in zip(names,q,strict=True)))
    checks['named_commands_exact']=measurements['exact_named_command_readback']
    assert fdata.time==validator.data.time==0
    return dict(checks=checks,passed=all(checks.values()),measurements=measurements)


def run(args):
    out=Path(args.output).resolve();pp=out/'protocol.json';p=json.loads(pp.read_text());ph=sha(pp)
    if (out/'calls').exists() or (out/'results.json').exists():
        raise FileExistsError('never overwrite qualification output')
    def verify():
        if sha(pp)!=ph:raise ValueError('protocol changed')
        for path,digest in p['bindings'].items():
            if sha(path)!=digest:raise ValueError(f'bound source/input changed: {path}')
    verify();sys.path.insert(0,str(Path(p['project'])/'src'));sys.path.extend(p['dependency_dirs'])
    counts=dict(physics_steps=0,static_forwards=0,project_retarget_calls=0,inner_refinement_calls=0)
    def forbidden(*a,**kw):
        counts['physics_steps']+=1
        raise RuntimeError('fresh mapping qualification forbids physics integration')
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
    original=mujoco.mj_forward
    def forward(*a,**kw):
        counts['static_forwards']+=1
        return original(*a,**kw)
    mujoco.mj_forward=forward
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from grasp_failure_prediction.evaluation.retargeting import ShadowHandRetargeter
    from grasp_failure_prediction.evaluation.fresh_mapping_qualification import capture_solve,compare_commands,settings_gate
    from grasp_failure_prediction.evaluation import pose_validation as pv
    from grasp_failure_prediction.evaluation.case_runner import _hug_world_palm_pose
    xml=module('xml_reference',p['reference_xml']);native_ref=module('qualified_native_reference',p['reference_native_frame'])
    root=Path(p['source_urdf']).parent.parent
    validator=pv.ShadowPoseValidator(root);floating=pv.load_shadow_model(root,floating_root=True);fdata=mujoco.MjData(floating)
    inventory=json.loads(Path(p['inventory']).read_text())['modern_cases']
    grasps={c['index']:load_hug_prediction(inventory[c['index']]['native']['path']) for c in p['cases']}
    (out/'calls').mkdir();records=[];profile_reports=[];started=time.perf_counter()
    for profile,settings in p['profiles'].items():
        baseline={};qualified={};sequences={}
        for schedule,indices in p['schedule'].items():
            reused=None if schedule.startswith('fresh_') else ShadowHandRetargeter(root,profile=profile)
            current=[]
            for index in indices:
                retargeter=ShadowHandRetargeter(root,profile=profile) if reused is None else reused
                counts['project_retarget_calls']+=1
                pose,record=capture_solve(retargeter,grasps[index]);counts['inner_refinement_calls']+=len(record['raw_refinement_qpos_rad'])
                record.update(call_index=len(records),inventory_index=index,profile_id=profile,schedule=schedule,
                              settings_status_gate=settings_gate(record,settings),native_sha256=inventory[index]['native']['sha256'])
                if schedule=='fresh_reference':
                    baseline[index]=record
                    if pose is not None:
                        qualified[index]=qualify_pose(pose,record,grasps[index],inventory[index],validator,floating,fdata,xml,native_ref,pv,_hug_world_palm_pose,p)
                    else:qualified[index]=dict(passed=False,checks={'solver_returned':False})
                    record['static_qualification']=qualified[index]
                else:
                    record['reference_comparison']=compare_commands(baseline[index],record,p['numerical_gates']['repeat_command_rad'])
                target=out/'calls'/f"call_{record['call_index']:02}.json";write(target,record)
                records.append(record);current.append(record)
                print(json.dumps({key:record[key] for key in ('call_index','inventory_index','profile_id','schedule','status','settings_status_gate')}),flush=True)
            sequences[schedule]=current
        cases=[]
        for index in grasps:
            repeated=next(row for row in sequences['fresh_repeat'] if row['inventory_index']==index)
            reused_rows=[row for label,seq in sequences.items() if label.startswith('reused_') for row in seq if row['inventory_index']==index]
            gate=baseline[index]['settings_status_gate'] and repeated['settings_status_gate'] and repeated['reference_comparison']['passed'] and qualified[index]['passed']
            cases.append(dict(index=index,object_id=inventory[index]['object_id'],fresh_instance_gate_pass=bool(gate),
                              fresh_repeat=repeated['reference_comparison'],static_qualification=qualified[index],
                              reused_instance_order_pass=bool(all(row['settings_status_gate'] and row['reference_comparison']['passed'] for row in reused_rows)),
                              reused_max_difference_rad=max((row['reference_comparison']['max_difference_rad'] or 0) for row in reused_rows)))
        profile_reports.append(dict(profile_id=profile,cases=cases,all_fresh_instance_gates_pass=all(row['fresh_instance_gate_pass'] for row in cases),
                                    all_reused_instance_order_gates_pass=all(row['reused_instance_order_pass'] for row in cases)))
    verify()
    if counts['project_retarget_calls']>p['maximum_project_retarget_calls'] or counts['inner_refinement_calls']>p['maximum_inner_refinement_calls']:
        raise AssertionError('finite solve budget exceeded')
    result=dict(schema=p['schema'],protocol_sha256=ph,profiles=profile_reports,counts=counts,runtime_seconds=time.perf_counter()-started,
                runtime=dict(numpy=np.__version__,mujoco=mujoco.__version__,python=sys.version),
                calls=[dict(call_index=row['call_index'],sha256=sha(out/'calls'/f"call_{row['call_index']:02}.json")) for row in records],
                all_bound_sources_inputs_unchanged=True,default_or_profile_promoted=False,physical_execution_qualified=False,
                new_native_inference=False,new_outcomes_or_training=False,saved_replay_acceptance_changed=False)
    write(out/'results.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--project');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        if not args.project:parser.error('prepare requires --project')
        prepare(args)
    else:run(args)


if __name__=='__main__':main()
