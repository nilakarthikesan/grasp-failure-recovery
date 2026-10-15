#!/usr/bin/env python3
"""Validate fresh HUG mapping records or saved commands, with zero physics.

Examples: ``saved --indices 0 8 --hug-root ... --output /new/report.json``;
``proposal --native ... --mapping-record ... --observation ... --output ...``.
Historical evidence is hash-checked and labelled separately from recomputation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Base runtime must precede the preserved external dependency bundle.
import numpy as np
import mujoco


def _exit_code(records, require_fidelity=False):
    """Keep numerical diagnostic exits; explicit strict admission adds code 3."""
    if any(c['runtime'] and c['runtime']['static_input_gate']=='FAIL' for c in records):return 1
    if any(c['runtime'] and c['runtime']['static_input_gate']=='INCOMPLETE' for c in records):return 2
    if require_fidelity and (not records or any(not c.get('fidelity',{}).get('collection_admitted',False) for c in records)):return 3
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--hug-root',type=Path,help='existing licensed MANO/HUG source; omission leaves reconstruction NOT_RUN')
    parser.add_argument('--output',type=Path,required=True,help='new JSON report; existing files are never overwritten')
    parser.add_argument('--prepare-protocol',type=Path,help='write a source/input-bound static run declaration and stop before geometry checks')
    parser.add_argument('--protocol',type=Path,help='verify this predeclared static run before measurement')
    parser.add_argument('--require-fidelity',action='store_true',help='also require real native-surface fidelity admission; no supported positive certifier exists today')
    parser.add_argument('--fidelity-evidence',type=Path,help='optional immutable fidelity evidence JSON; numerical PASS alone is insufficient')
    parser.add_argument('--fidelity-geometry-bindings',type=Path,help='JSON native_hand/mapped_hand/object file-to-hash closures for evidence identity checks')
    sub=parser.add_subparsers(dest='mode',required=True)
    saved=sub.add_parser('saved');saved.add_argument('--inventory',type=Path);saved.add_argument('--indices',type=int,nargs='+')
    saved.add_argument('--evidence-only',action='store_true',help='verify old evidence without recomputing geometry')
    fresh=sub.add_parser('proposal');fresh.add_argument('--native',type=Path,required=True)
    fresh.add_argument('--mapping-record',type=Path,required=True);fresh.add_argument('--observation',type=Path,required=True)
    args=parser.parse_args(argv);root=args.project.resolve();output=args.output.resolve()
    if output.exists() or output.with_suffix('.md').exists():raise FileExistsError('mapping reports are immutable; select fresh output files')
    bootstrap=root/'research_artifacts/hug_native_mano_comparison_2026-10-15_v1/measurement/protocol.json'
    if bootstrap.exists():
        for directory in json.loads(bootstrap.read_text())['dependency_dirs']:
            if directory not in sys.path:sys.path.append(directory)
    sys.path.insert(0,str(root/'src'))
    from grasp_failure_prediction.evaluation import mapping_validation as mv
    from grasp_failure_prediction.evaluation import mapping_fidelity as mf
    from grasp_failure_prediction.evaluation import pose_validation as pv
    from grasp_failure_prediction.evaluation.retargeting import RetargetedHandPose,camera_landmarks_to_wrist,reference_vectors
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
    urdf=root/'third_party/dex-urdf/robots/hands'
    inventory=args.inventory if args.mode=='saved' and args.inventory else root/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
    entries=json.loads(inventory.read_text())['modern_cases']
    selected=args.indices if args.mode=='saved' and args.indices is not None else list(range(len(entries))) if args.mode=='saved' else []
    declaration={'schema_version':'hug_mapping_static_run_protocol_v1','mode':args.mode,'indices':selected,
                 'require_fidelity':args.require_fidelity,
                 'fidelity_evidence':str(args.fidelity_evidence.resolve()) if args.fidelity_evidence else None,
                 'fidelity_geometry_bindings':str(args.fidelity_geometry_bindings.resolve()) if args.fidelity_geometry_bindings else None,
                 'native':str(args.native.resolve()) if args.mode=='proposal' else None,
                 'mapping_record':str(args.mapping_record.resolve()) if args.mode=='proposal' else None,
                 'output':str(output),'evidence_only':args.evidence_only if args.mode=='saved' else False,
                 'hug_root':str(args.hug_root.resolve()) if args.hug_root else None,
                 'tolerances':dict(mv.TOLERANCES),'physics_steps_allowed':0,'retarget_solves_allowed':0,
                 'new_inference_allowed':False,'contact_and_tracking_acceptance':'MEASURED_ONLY',
                 'runtime':{'python':sys.version,'numpy':np.__version__,'mujoco':mujoco.__version__}}
    if args.prepare_protocol:
        from xml.etree import ElementTree as ET
        paths=[Path(__file__).resolve(),Path(mv.__file__),Path(mf.__file__),Path(pv.__file__),inventory,
               mf.DEFAULT_ACCEPTANCE_RULE,root/'tests/test_mapping_fidelity.py',
               root/'src/grasp_failure_prediction/evaluation/retargeting.py',
               root/'tests/test_mapping_validation.py',
               root/'src/grasp_failure_prediction/integrations/hug.py',root/'src/grasp_failure_prediction/integrations/hug_projection.py',
               root/'src/grasp_failure_prediction/integrations/hug_frames.py',
               urdf/'shadow_hand/shadow_hand_right.urdf']
        source_urdf=paths[-1]
        paths.extend((source_urdf.parent/mesh.get('filename')).resolve() for mesh in ET.parse(source_urdf).getroot().findall('.//mesh'))
        if args.hug_root:
            paths.extend([args.hug_root/'src/models/mano.py',args.hug_root/'assets/mano_models/models/MANO_RIGHT.pkl'])
        if args.mode=='saved':
            if len(set(selected))!=len(selected) or any(i<0 or i>=len(entries) for i in selected):raise ValueError('invalid inventory selection')
            for i in selected:
                case=entries[i];paths.extend(Path(case[key]['path']) for key in ('native','stored_shadow_command','scene_mjb'))
                paths.extend(Path(value['path']) for value in case['frame_files'].values())
                paths.extend(path for path in Path(case['native']['path']).parent.iterdir() if path.is_file())
                history=mv.historical_evidence(root,i,case['native']['sha256'])
                paths.extend(Path(ref[key]) for ref in history['references'] for key in ('path','manifest'))
        else:
            paths.extend([args.native,args.mapping_record]);paths.extend(path for path in args.observation.iterdir() if path.is_file())
            value=json.loads(args.mapping_record.read_text());config=value.get('provenance',{}).get('config_path')
            if config:paths.append(Path(config))
        if args.fidelity_evidence:
            paths.append(args.fidelity_evidence)
            value=json.loads(args.fidelity_evidence.read_text())
            paths.extend(Path(name) for name in value.get('artifact_bindings',{}))
        if args.fidelity_geometry_bindings:
            paths.append(args.fidelity_geometry_bindings)
            for bindings in json.loads(args.fidelity_geometry_bindings.read_text()).values():
                paths.extend(Path(name) for name in bindings)
        declaration['bindings']={str(path.resolve()):mv.sha(path) for path in paths}
        path=args.prepare_protocol;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as handle:json.dump(declaration,handle,indent=2,allow_nan=False);handle.write('\n')
        print(json.dumps({'predeclared_protocol':str(path.resolve()),'sha256':mv.sha(path),'physics_steps':0}));return 0
    frozen=None
    if args.protocol:
        frozen=json.loads(args.protocol.read_text())
        if any(frozen.get(key)!=value for key,value in declaration.items()):raise ValueError('run differs from predeclared static protocol')
        for path,digest in frozen['bindings'].items():
            if mv.sha(path)!=digest:raise ValueError(f'bound source/input changed: {path}')
    records=[]
    with mv.static_only() as counts:
        reconstruct=mv.make_mano_reconstructor(args.hug_root) if args.hug_root else None
        def validate(native,observation,names,q,mapping_record=None,world_transform=None,saved_root=None):
            grasp=load_hug_prediction(native)
            local=camera_landmarks_to_wrist(grasp.landmarks_3d,grasp.T_camera_wrist)@MANO_TO_OPERATOR_RIGHT.T
            pose=RetargetedHandPose(tuple(names),np.asarray(q,float),local,reference_vectors(local,mv.HUMAN_TASK_INDICES),grasp.T_camera_wrist)
            camera=np.load(observation/'T_world_camera.npy',allow_pickle=False)
            intrinsic=np.load(observation/'intrinsics.npy',allow_pickle=False)
            from PIL import Image
            with Image.open(observation/'rgb.png') as rgb:size=rgb.size
            cm=cd=None
            if (observation/'scene.mjb').exists() and (observation/'scene_state.npz').exists():
                cm=mujoco.MjModel.from_binary_path(str(observation/'scene.mjb'));cd=mujoco.MjData(cm)
                with np.load(observation/'scene_state.npz',allow_pickle=False) as state:
                    cd.qpos[:]=state['qpos'];cd.qvel[:]=state['qvel']
                mujoco.mj_forward(cm,cd)
            world_palm=None
            if saved_root is not None:
                model=pv.load_shadow_model(urdf,floating_root=True);data=mujoco.MjData(model)
                for name,value in zip(names,q,strict=True):
                    jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);data.qpos[int(model.jnt_qposadr[jid])]=value
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint');a=int(model.jnt_qposadr[jid])
                data.qpos[a:a+7]=saved_root;mujoco.mj_forward(model,data)
                palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'palm');quat=np.empty(4)
                mujoco.mju_mat2Quat(quat,data.xmat[palm]);world_palm=(data.xpos[palm].copy(),quat)
            elif world_transform is not None:
                transform=mv._rigid(world_transform);quat=np.empty(4);mujoco.mju_mat2Quat(quat,transform[:3,:3].reshape(9))
                world_palm=(transform[:3,3],quat)
            return mv.validate_hug_mapping(grasp,pose,urdf_root=urdf,T_world_camera=camera,camera_intrinsics=intrinsic,
                                          image_size=size,world_palm=world_palm,camera_model=cm,camera_data=cd,
                                          native_reconstructor=reconstruct,mapping_record=mapping_record,
                                          mapping_kind='fresh' if mapping_record is not None else 'saved_command')
        if args.mode=='saved':
            indices=selected
            if len(set(indices))!=len(indices) or any(i<0 or i>=len(entries) for i in indices):raise ValueError('indices must be unique existing inventory rows')
            for index in indices:
                case=entries[index];native=Path(case['native']['path']);digest=mv.sha(native)
                if digest!=case['native']['sha256'].removeprefix('sha256:'):raise ValueError('native inventory hash mismatch')
                history=mv.historical_evidence(root,index,digest)
                runtime=None
                if not args.evidence_only:
                    command=case['stored_shadow_command'];path=Path(command['path'])
                    if mv.sha(path)!=command['sha256'].removeprefix('sha256:'):raise ValueError('saved command inventory hash mismatch')
                    with np.load(path,allow_pickle=False) as source:
                        row=command['row_index'];names=source['joint_names'].tolist();q=source['nominal_target_rad'][row]
                        carrier=np.r_[source['root_target_position_m'][row],source['root_target_quaternion_wxyz'][row]]
                    runtime=validate(native,native.parent,names,q,saved_root=carrier)
                records.append(dict(index=index,native=str(native),native_sha256=digest,runtime=runtime,historical=history))
        else:
            record=json.loads(args.mapping_record.read_text());digest=mv.sha(args.native)
            if record.get('native_proposal_sha256','').removeprefix('sha256:')!=digest:raise ValueError('mapping record does not bind native proposal')
            if record.get('mapping_status')!='ok' or record.get('status')!='returned':
                runtime={'schema_version':mv.SCHEMA,'static_input_gate':'FAIL','checks':{'mapping_refusal':mv.check(False,metrics={'mapping_status':record.get('mapping_status'),'reason':record.get('refusal_reason',record.get('message'))},scope='preserved mapping refusal; not a physical failure')},'new_outcome':None}
            else:
                runtime=validate(args.native,args.observation,record['joint_names'],record['qpos_rad'],mapping_record=record,
                                 world_transform=record.get('world_palm_transform'))
            match=next((i for i,c in enumerate(entries) if c['native']['sha256'].removeprefix('sha256:')==digest),None)
            records.append(dict(native=str(args.native.resolve()),native_sha256=digest,mapping_record_sha256=mv.sha(args.mapping_record),
                                runtime=runtime,historical=mv.historical_evidence(root,match,digest) if match is not None else {'checks':{},'scope':'new native identity; historical qualification does not transfer'}))
    geometry=json.loads(args.fidelity_geometry_bindings.read_text()) if args.fidelity_geometry_bindings else None
    for case in records:
        mapped=args.mapping_record if args.mode=='proposal' else Path(entries[case['index']]['stored_shadow_command']['path'])
        case['fidelity']=mf.assess_fidelity_evidence(args.fidelity_evidence,native_path=case['native'],mapped_path=mapped,
            geometry_bindings=geometry,numerical_report=case['runtime'])
    report=dict(schema_version=mv.SCHEMA,mode=args.mode,cases=records,counts=counts,
                implementation_sha256={str(Path(mv.__file__).resolve()):mv.sha(mv.__file__),str(Path(__file__).resolve()):mv.sha(__file__)},
                historical_evidence_does_not_qualify_current_mapping=True,physical_execution_or_large_collection_qualified=False)
    gates=[case['runtime']['static_input_gate'] if case['runtime'] else 'NOT_RUN' for case in records]
    report['final_verdict']='FAIL' if 'FAIL' in gates else 'PASS_WITH_CAVEATS' if gates and all(g=='PASS' for g in gates) else 'INCOMPLETE'
    report['numerical_final_verdict']=report['final_verdict']
    fidelity=[case['fidelity']['fidelity_status'] for case in records]
    report['native_fidelity_status']='FAIL' if 'FAIL' in fidelity else 'PASS' if fidelity and all(v=='PASS' for v in fidelity) else 'UNRESOLVED'
    report['fidelity_collection_admitted']=bool(records) and all(c['fidelity']['collection_admitted'] for c in records)
    report['fidelity_required_for_execution']=True
    report['supported_continuous_certifiers']=list(mf.SUPPORTED_CONTINUOUS_CERTIFIERS)
    report['protocol_sha256']=mv.sha(args.protocol) if args.protocol else None
    report['predeclared_run']=frozen is not None
    if frozen:
        for path,digest in frozen['bindings'].items():
            if mv.sha(path)!=digest:raise ValueError(f'bound source/input changed during run: {path}')
        report['bindings']=frozen['bindings']
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')
    lines=['# HUG mapping validation',f"Numerical verdict: **{report['final_verdict']}**.",
           f"Native-surface fidelity: **{report['native_fidelity_status']}**; collection admitted: **{report['fidelity_collection_admitted']}**.",
           'Scope: static simulator inputs. Contact/tracking remain measurements; native fidelity and large collection are unqualified.','']
    for case in records:
        label=case.get('index',case['native_sha256'][:12]);runtime=case['runtime']
        lines.append(f"- Case {label}: numerical {runtime['static_input_gate'] if runtime else 'NOT_RUN'}; fidelity {case['fidelity']['fidelity_status']}.")
        if runtime:
            for name,row in runtime['checks'].items():lines.append(f"  - {name}: {row['status']} ({row['origin']}).")
        for name,row in case['historical'].get('checks',{}).items():lines.append(f"  - Historical {name}: {row['status']} (hash-bound; not recomputed).")
    with output.with_suffix('.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(output),'cases':len(records),'counts':counts,'runtime_gates':[c['runtime']['static_input_gate'] if c['runtime'] else 'NOT_RUN' for c in records]}))
    return _exit_code(records,args.require_fidelity)


if __name__=='__main__':raise SystemExit(main())
