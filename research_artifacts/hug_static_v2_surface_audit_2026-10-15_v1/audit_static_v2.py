"""Finite production static-v2 translation audit; no native-fidelity PASS.

One fresh public translation per existing native input, followed by static
current-scene full-field checks. No exchange, cap restoration, inference,
physics, task labels or predictor fitting. Preparation requires completed v3
evidence and its independent saved-array readback.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import mujoco

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V3=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v3'
SCHEMA='hug_production_static_v2_full_field_audit_v1'
INDICES=(20,28)
STRATEGY='mano_right_to_shadow_hand_right_static_v2'
PROFILE='independent_static_shadow_vector_v2'
BUDGET=dict(cases=2,maximum_seconds_per_case=600,maximum_static_forwards_per_case=250,
    maximum_raw_callbacks_per_case=150,maximum_full_field_checkpoints_per_case=3,
    maximum_production_translations_per_case=1,maximum_dex_refinement_calls_per_case=20,
    maximum_dex_refinement_calls_total=40,physics_steps=0,native_inference_calls=0,
    surface_optimization_calls=0,predictor_fits=0,task_outcomes=0)
FLAGS=dict(diagnostic_outside_original_search_domain=True,original_caps_targets_and_grid_rebased=False,
    native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False,
    full_orientation_continuous_coverage='NOT_PROVED',preprocessing_scalar_comparison_limit_m=1e-10)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


util=load('static_v2_bound_v3_utilities',V3/'run_repair.py')
read,write,sha,plain=util.read,util.write,util.sha,util.plain


def completed_v3(report_path,readback_path,protocol_path):
    report=read(report_path);independent=read(readback_path)
    if (report.get('status')!='COMPLETED_BOUNDED_PILOT' or report.get('native_fidelity_accepted') is not False
            or report.get('new_collection_or_training_allowed') is not False
            or independent.get('status')!='VERIFIED_SAVED_ARRAYS_AND_REPORT_INTERPRETATION'
            or independent.get('native_fidelity_accepted') is not False
            or independent.get('report_sha256')!=sha(report_path)
            or report.get('protocol_sha256')!=sha(protocol_path) or independent.get('protocol_sha256')!=sha(protocol_path)
            or len(report.get('freeze_commit',''))!=40 or independent.get('freeze_commit')!=report.get('freeze_commit')
            or [r.get('index') for r in report.get('cases',[])]!=list(INDICES)
            or [r.get('index') for r in independent.get('cases',[])]!=list(INDICES)):
        raise ValueError('Complete frozen v3 evidence and matching independent readback required')
    paths=[Path(report_path),Path(readback_path)]
    for row in report['cases']:
        if row.get('status')!='COMPLETED':raise ValueError('Every v3 case must have completed')
        for key in ('result','log'):
            path=ROOT/row[key]
            if sha(path)!=row[key+'_sha256']:raise ValueError('V3 result/log changed')
            paths.append(path)
        result=read(ROOT/row['result'])
        if (result.get('native_fidelity_accepted') is not False or result.get('protocol_sha256')!=sha(protocol_path)
                or result.get('freeze_commit')!=report['freeze_commit']):raise ValueError('V3 case protocol/freeze/fidelity identity differs')
        for item in (result.get('checkpoint_artifacts') or [])+(result.get('auxiliary_artifacts') or []):
            path=Path(item['path'])
            if sha(path)!=item['sha256']:raise ValueError('V3 bound output changed')
            paths.append(path)
    for row in independent['cases']:
        official=next(r for r in report['cases'] if r['index']==row['index'])
        if row.get('result_sha256')!=official['result_sha256']:raise ValueError('V3 independent case identity changed')
        for item in row.get('checkpoints',[]):
            path=ROOT/item['path']
            if sha(path)!=item['sha256']:raise ValueError('Independent v3 checkpoint changed')
            paths.append(path)
    return paths


def production_configuration():
    sys.path.insert(0,str(ROOT/'src'))
    import dex_retargeting
    from grasp_failure_prediction.proposals.translation import load_translation_registry
    spec=load_translation_registry().resolve(STRATEGY)
    evidence=next(x for x in spec.evidence if x.role=='retargeter_config')
    path=Path(dex_retargeting.__file__).parent/'configs/teleop/shadow_hand_right.yml'
    if evidence.sha256!='sha256:'+sha(path):raise ValueError('Production retargeter config evidence differs')
    return dict(strategy_id=STRATEGY,profile_id=PROFILE,refinement_calls=20,config_path=str(path.resolve()),
        config_sha256=sha(path),vector_scaling=1.,low_pass_filter_enabled=False,temporal_norm_delta=0.,
        reset_before_proposal=True,supported_optimizer_status_codes=[1,2,3,4,5,6])


def prepare(protocol,v3_protocol,v3_report,v3_readback):
    if Path(protocol).exists():raise FileExistsError('Audit protocol is immutable')
    prior=read(v3_protocol);util.verify(prior['bindings'])
    prior_paths=completed_v3(v3_report,v3_readback,v3_protocol)
    runtime=util.runtime();sys.path.insert(0,str(ROOT/'src'))
    configuration=production_configuration();inputs=dict(prior['bindings'])
    if Path(v3_protocol).resolve()!=(V3/'protocol.json').resolve():raise ValueError('Only the active frozen v3 protocol is supported')
    paths=prior_paths+[Path(v3_protocol),Path(__file__),HERE/'test_audit_static_v2.py',HERE/'README.md']+source_git_paths()
    paths+=list(V3.glob('*.py'))+[ROOT/'src/grasp_failure_prediction/evaluation/surface_pose.py',
        ROOT/'src/grasp_failure_prediction/evaluation/fresh_mapping_qualification.py',
        ROOT/'src/grasp_failure_prediction/evaluation/retargeting.py',
        ROOT/'src/grasp_failure_prediction/proposals/translation.py',
        ROOT/'src/grasp_failure_prediction/proposals/configs/translations.yaml']
    import dex_retargeting
    package=Path(dex_retargeting.__file__).parent
    paths+=list(package.rglob('*.py'))+[Path(configuration['config_path'])]
    for path in paths:inputs[str(path.resolve())]=sha(path)
    git_paths=[str(x.resolve()) for x in source_git_paths()]
    declaration=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),git_source_paths=git_paths,
        runtime=runtime,bindings=inputs,budgets=dict(BUDGET),production_configuration=configuration,
        source_inventory_indices=list(INDICES),v3_protocol=str(Path(v3_protocol).resolve()),
        v3_report=str(Path(v3_report).resolve()),v3_readback=str(Path(v3_readback).resolve()),
        algorithm='literal legacy baseline, one fresh production translation, new native-anchored palm, full-field, literal fresh readback',
        **FLAGS)
    write(protocol,declaration);return declaration


def source_git_paths():
    paths=list(HERE.glob('*.py'))+[HERE/'README.md']+list(V3.glob('*.py'))
    paths += [ROOT/name for name in ('src/grasp_failure_prediction/evaluation/surface_pose.py',
        'src/grasp_failure_prediction/evaluation/retargeting.py','src/grasp_failure_prediction/evaluation/fresh_mapping_qualification.py',
        'src/grasp_failure_prediction/proposals/translation.py','src/grasp_failure_prediction/proposals/configs/translations.yaml',
        'src/grasp_failure_prediction/evaluation/case_runner.py','src/grasp_failure_prediction/evaluation/runner.py',
        'src/grasp_failure_prediction/evaluation/pose_validation.py','src/grasp_failure_prediction/integrations/hug.py',
        'src/grasp_failure_prediction/integrations/hug_frames.py')]
    return sorted(set(paths))


def validate(protocol,freeze_commit):
    p=read(protocol)
    if (p.get('schema_version')!=SCHEMA or p.get('budgets')!=BUDGET
            or p.get('source_inventory_indices')!=list(INDICES) or p.get('runtime')!=util.runtime()
            or p.get('production_configuration')!=production_configuration()
            or any(p.get(key)!=value for key,value in FLAGS.items())
            or Path(p['v3_protocol']).resolve()!=(V3/'protocol.json').resolve()
            or p.get('git_source_paths')!=[str(x.resolve()) for x in source_git_paths()]):raise ValueError('Audit declaration/runtime/configuration differs')
    if any(str(Path(p[key]).resolve()) not in p['bindings'] for key in ('v3_protocol','v3_report','v3_readback')):
        raise ValueError('V3 declaration/output references must be explicitly bound')
    util.verify(p['bindings']);completed_v3(p['v3_report'],p['v3_readback'],p['v3_protocol'])
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('HEAD must remain exact audit freeze commit')
    paths=[Path(protocol).resolve()]+source_git_paths()
    if any(str(path.resolve()) not in p['bindings'] for path in source_git_paths()):raise ValueError('Missing critical source binding')
    names=[str(x.relative_to(ROOT)) for x in paths]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*names],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*names],check=True)
    return p,commit


def validate_mapping_record(pose,record,configuration,actual_calls):
    if (record.get('schema_version')!='hug_mapping_record_v1' or record.get('translation_strategy_id')!=STRATEGY
            or 'outcome' not in record or record.get('outcome') is not None or record.get('fresh_instance_per_proposal') is not True
            or type(actual_calls) is not int or not 0<=actual_calls<=20):raise ValueError('Production mapping identity/denominator differs')
    chunks=record.get('raw_refinement_qpos_rad',[])
    statuses=record.get('optimizer_status_codes',[])
    if len(statuses)>20 or len(chunks)>20 or len(chunks)>actual_calls or len(statuses)>len(chunks):
        raise ValueError('Malformed raw/status mapping trace')
    if pose is None:
        if record.get('mapping_status')!='refused_invalid_retarget':raise ValueError('Null production output must retain explicit refusal')
        return dict(status='TRANSLATION_REFUSED',output_available=False,physical_outcome=None,
                    raw_return_count=len(chunks),actual_refinement_attempts=actual_calls,
                    raw_trace_finite=bool(all(np.isfinite(np.asarray(x,float)).all() for x in chunks)),
                    raw_return_shapes=[list(np.asarray(x).shape) for x in chunks])
    raw=np.asarray(chunks,float)
    names=tuple(pose.joint_names);q=np.asarray(pose.qpos,float);limits=np.asarray(record.get('joint_limits_rad'),float)
    provenance=record.get('provenance') or {}
    if (record.get('status')!='returned' or record.get('mapping_status')!='ok'
            or record.get('optimizer_status_gate_passed') is not True or actual_calls!=20
            or raw.shape!=(20,24) or not np.isfinite(raw).all() or len(statuses)!=20
            or any(type(s) is not int or s not in configuration['supported_optimizer_status_codes'] for s in statuses)
            or len(names)!=24 or len(set(names))!=24 or q.shape!=(24,) or not np.isfinite(q).all()
            or record.get('joint_names')!=list(names) or not np.array_equal(q,record.get('qpos_rad'))
            or limits.shape!=(24,2) or not np.isfinite(limits).all() or np.any(limits[:,0]>limits[:,1])):
        raise ValueError('Returned production command/raw/status record mismatch')
    for key in ('profile_id','config_path','config_sha256','vector_scaling','low_pass_filter_enabled',
                'temporal_norm_delta','reset_before_proposal','refinement_calls'):
        if provenance.get(key)!=configuration[key]:raise ValueError('Production provenance mismatch '+key)
    if list(provenance.get('optimizer_status_codes',[]))!=statuses:raise ValueError('Provenance status trace mismatch')
    clipping=float(np.max(np.abs(raw[-1]-q)))
    if (record.get('max_final_clipping_rad')!=clipping or not np.array_equal(q,np.clip(raw[-1],limits[:,0],limits[:,1]))):
        raise ValueError('Raw/returned clipping distinction changed')
    return dict(status='PRODUCTION_OUTPUT_RECORDED',output_available=True,physical_outcome=None,
        actual_refinement_attempts=actual_calls,raw_return_count=20,max_final_clipping_rad=clipping,
        optimizer_convergence_not_implied=True)


def save_production_evidence(output,record,preprocessing,pose):
    """Preserve refused nonfinite raw returns losslessly before validation."""
    output=Path(output);trace=output/'production_solver_trace.npz';path=output/'production_mapping_record.json'
    arrays={};inventory={}
    for key,values in dict(preprocessing,raw_refinement_qpos_rad=record.get('raw_refinement_qpos_rad',[])).items():
        inventory[key]=[]
        for number,value in enumerate(values):
            name=f'{key}_{number:02d}';array=np.asarray(value)
            if array.dtype.hasobject:raise ValueError('Numeric preprocessing/raw chunk required')
            arrays[name]=array;inventory[key].append(dict(key=name,shape=list(array.shape),dtype=str(array.dtype)))
    if pose is not None:
        arrays.update(returned_qpos_rad=np.asarray(pose.qpos),returned_landmarks_wrist_m=np.asarray(pose.landmarks_wrist_m),
                      returned_reference_vectors_m=np.asarray(pose.reference_vectors_m))
    with trace.open('xb') as f:np.savez_compressed(f,**arrays)
    def finite(value):
        value=plain(value)
        if isinstance(value,float) and not np.isfinite(value):return {'nonfinite_float':repr(value)}
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        return value
    binding=dict(path=str(trace.resolve()),sha256=sha(trace),bytes=trace.stat().st_size)
    write(path,dict(schema_version='hug_production_mapping_observation_v1',mapping_record=finite(record),
        raw_numeric_and_preprocessing_trace=binding,array_shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
        nonfinite_json_cells_explicitly_encoded=True,physical_outcome=None))
    return dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,trace_npz=binding,array_inventory=inventory)


def preprocessing_receipt(grasp,pose,trace,actual_calls):
    counts={key:len(value) for key,value in trace.items()}
    complete=(counts.get('operator_landmarks_m')==1 and counts.get('reference_vectors_m')==1
              and counts.get('target_link_human_indices')==1 and counts.get('actual_ref_value')==actual_calls)
    if not complete:
        if pose is not None:raise ValueError('Returned pose lacks complete observed preprocessing trace')
        return dict(status='PARTIAL_REFUSED_PREPROCESSING',observed_counts=counts,passed=False)
    from grasp_failure_prediction.evaluation.fresh_mapping_qualification import independent_local_operator_landmarks
    landmarks=np.asarray(trace['operator_landmarks_m'][0]);indices=np.asarray(trace['target_link_human_indices'][0])
    vectors=np.asarray(trace['reference_vectors_m'][0]);inputs=np.asarray(trace['actual_ref_value'])
    if landmarks.shape!=(21,3) or indices.ndim!=2 or indices.shape[0]!=2 or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Actual preprocessing shape/index dtype changed')
    if np.any(indices<0) or np.any(indices>=21):raise ValueError('Actual preprocessing landmark index changed')
    scalar=independent_local_operator_landmarks(grasp);error=float(np.max(np.abs(landmarks-scalar)))
    if (not np.isfinite(landmarks).all() or not np.isfinite(vectors).all() or error>1e-10
            or not np.array_equal(vectors,landmarks[indices[1]]-landmarks[indices[0]])
            or inputs.dtype!=np.float32 or inputs.shape!=(actual_calls,*vectors.shape)
            or any(not np.array_equal(x,vectors.astype(np.float32)) for x in inputs)):
        raise ValueError('Actual production preprocessing/reference inputs differ')
    if pose is not None and (not np.array_equal(pose.landmarks_wrist_m,landmarks) or not np.array_equal(pose.reference_vectors_m,vectors)):
        raise ValueError('Returned pose preprocessing differs from actual solver inputs')
    return dict(status='OBSERVED_INPUTS_CHECKED',observed_counts=counts,passed=True,
        scalar_operator_landmark_max_difference_m=error,comparison_limit_m=1e-10,
        actual_solver_input_dtype=str(inputs.dtype),actual_refinement_input_count=actual_calls,
        actual_target_link_human_indices=indices,physical_outcome=None)


def diagnostic_literal(adapter,payload,pose_module):
    """V4-only measurement; bypass domain REFUSAL, never its flags or bounds."""
    required={'schema_version','joint_targets_rad','root_position_m','root_quaternion_wxyz','source_urdf_sha256','model_geometry_sha256'}
    if (set(payload)!=required or payload['schema_version']!=pose_module.POSE_SCHEMA
            or payload['source_urdf_sha256']!=adapter.source_urdf_sha256
            or payload['model_geometry_sha256']!=adapter.model_geometry_sha256):raise ValueError('Diagnostic literal source/schema mismatch')
    q=adapter._named_q(payload['joint_targets_rad'])
    pos=pose_module._array(payload['root_position_m'],(3,),'root position');quat=pose_module._quat(payload['root_quaternion_wxyz'])
    adapter._restore();adapter._targets(pos,quat,q);adapter._forward()
    shift=adapter.data.xpos[adapter.palm_body_id]-adapter.palm_position0
    omega=pose_module._left_rotation_log(quat,adapter.root_quaternion0)
    physical,engineering=adapter._domains(q,shift,omega)
    state=adapter._readback(q,physical,engineering)
    for key,value in (('joint_qpos_rad',q),('root_position_m',pos),('root_quaternion_wxyz',quat)):
        if not np.array_equal(getattr(state,key),value):raise ValueError('Diagnostic literal command changed '+key)
    return state


def compare_readback(first,second):
    for key in first['fields']:
        if not np.array_equal(first['fields'][key],second['fields'][key]):raise ValueError('Fresh literal readback changed full field '+key)
    for key in ('distance_m','margin_m'):
        if not np.array_equal(first[key],second[key]):raise ValueError('Fresh literal readback changed '+key)
    for key in ('valid','passed','score','distance_m'):
        if not np.array_equal(first['orientation'][key],second['orientation'][key]):raise ValueError('Fresh literal readback changed orientation '+key)
    for key in ('joint_qpos_rad','root_position_m','root_quaternion_wxyz','tip_error_norms_m'):
        if not np.array_equal(getattr(first['state'],key),getattr(second['state'],key)):raise ValueError('Fresh literal readback changed pose '+key)
    return True


def compact(full):
    if sha(full['array_path'])!=full['array_sha256']:raise ValueError('Full-field bytes changed')
    value={k:v for k,v in full.items() if k not in ('fields','margin_m','distance_m')}
    value['field_boolean_counts']={k:int(v.sum()) for k,v in full['fields'].items()}
    return value


def production_evidence_index(binding,output):
    output=Path(output).resolve();path=Path(binding['path']).resolve();trace=Path(binding['trace_npz']['path']).resolve()
    if path!=output/'production_mapping_record.json' or trace!=output/'production_solver_trace.npz':
        raise ValueError('Exact production observation paths required')
    rows=[]
    for file,reference,kind in ((path,binding,'production_mapping_record'),(trace,binding['trace_npz'],'production_solver_trace')):
        if file.is_symlink() or sha(file)!=reference['sha256'] or file.stat().st_size!=reference['bytes']:
            raise ValueError('Production observation changed')
        rows.append(dict(path=str(file),sha256=reference['sha256'],bytes=reference['bytes'],kind=kind))
    saved=read(path)
    if saved['raw_numeric_and_preprocessing_trace']!=binding['trace_npz']:raise ValueError('Solver trace reference changed')
    with np.load(trace,allow_pickle=False) as z:
        if set(z.files)!=set(saved['array_shapes']):raise ValueError('Solver/preprocessing array inventory changed')
        for key,meta in saved['array_shapes'].items():
            if list(z[key].shape)!=meta['shape'] or str(z[key].dtype)!=meta['dtype']:raise ValueError('Observed array shape/dtype changed')
    return rows


def checkpoint_index(output,receipts,visual_binding,record_binding,index):
    """V4 three-scan cap with only the two specifically bound auxiliary NPZs."""
    output=Path(output).resolve()
    auxiliary=util.visual_artifact_index(output,visual_binding,index)+production_evidence_index(record_binding,output)
    allowed={r['path'] for r in auxiliary if r['path'].endswith('.npz')}
    files=sorted(f for f in output.rglob('*.npz') if str(f.resolve()) not in allowed)
    if len(files)>3:raise ValueError('Three full-field checkpoint budget exceeded')
    if not isinstance(receipts,list):raise ValueError('Write-time checkpoint receipts required')
    byname={}
    for row in receipts:
        if set(row)!={'path','sha256','bytes','label'}:raise ValueError('Malformed checkpoint receipt')
        path=Path(row['path']).resolve()
        if path.parent!=output or path.name in byname or not path.name.startswith('checkpoint_') or path.suffix!='.npz':
            raise ValueError('Checkpoint path/inventory changed')
        byname[path.name]=row
    if {x.name for x in files}!=set(byname):raise ValueError('Missing or unindexed full-field checkpoint')
    rows=[]
    for number,file in enumerate(files,1):
        row=byname[file.name]
        if (file.is_symlink() or not file.name.startswith(f'checkpoint_{number:02d}_')
                or sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']):raise ValueError('Write-time full-field checkpoint changed')
        rows.append(dict(path=str(file),sha256=row['sha256'],bytes=row['bytes'],label=row['label']))
    return rows,auxiliary


def measure(case,pose,record,configuration,actual_calls,pose_module):
    gate=validate_mapping_record(pose,record,configuration,actual_calls)
    if pose is None:return dict(status='TRANSLATION_REFUSED',translation_gate=gate,static_v2=None,physical_outcome=None,
                               accepted_mapping=False,native_fidelity_accepted=False)
    from grasp_failure_prediction.evaluation import case_runner,pose_validation
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    grasp=load_hug_prediction(Path(case.source['native']['path']))
    validator=pose_validation.ShadowPoseValidator(Path(case.previous['source_urdf']).parent.parent)
    pos,quat=case_runner._hug_world_palm_pose(grasp,pose,validator,Path(case.source['native']['path']).parent)
    fixed=case.xml.forward_xml(case.xml.read_urdf(case.previous['source_urdf']),pose.joint_names,pose.qpos)
    camera=np.load(case.source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
    native_ref=load('static_v2_independent_native_frame',Path(case.p['v1_protocol']).parent/'source_snapshots/native_reference.py')
    expected=native_ref.expected_native_palm(camera,grasp.T_camera_wrist,fixed['palm'])['palm_world']
    actual=case.xml.transform(pos,case.xml.quaternion_matrix_wxyz(quat))
    residual=float(np.max(np.abs(expected-actual)))
    if residual>1e-12:raise ValueError('Fresh-q independent native/world/palm check failed')
    rootpos,rootquat,_=case.scene_runtime.stage_named_palm(case.runner,pose.joint_names,pose.qpos,actual)
    payload=dict(schema_version=pose_module.POSE_SCHEMA,joint_targets_rad=dict(zip(pose.joint_names,pose.qpos.tolist(),strict=True)),
        root_position_m=rootpos.tolist(),root_quaternion_wxyz=rootquat.tolist(),
        source_urdf_sha256=case.adapter.source_urdf_sha256,model_geometry_sha256=case.adapter.model_geometry_sha256)
    if len(case.checkpoint_artifacts)>=3:raise ValueError('Three-scan budget exhausted before measurement')
    case._raw_budget();state=diagnostic_literal(case.adapter,payload,pose_module)
    output=case._full_current(state,'production_static_v2')
    if len(case.checkpoint_artifacts)>=3:raise ValueError('Three-scan budget exhausted before readback')
    case._raw_budget();replay=diagnostic_literal(case.adapter,payload,pose_module)
    fresh=case._full_current(replay,'fresh_literal_static_v2_readback');compare_readback(output,fresh)
    core=load('static_v2_protection_math',V3/'exchange_core.py');protection=core.full_protection(case.audit_baseline,output)
    q=np.array([payload['joint_targets_rad'][name] for name in case.names])
    return dict(status='MEASURED_PRODUCTION_STATIC_V2',translation_gate=gate,static_v2=compact(output),
        fresh_readback=compact(fresh),fresh_literal_readback_pass=True,serialized_pose=payload,
        independent_new_q_palm_residual= residual,original_relative_joint_delta_rad=q-case.q0,
        maximum_original_relative_joint_delta_rad=float(np.max(abs(q-case.q0))),
        baseline_protection=protection,newly_preserved_relationships=int((output['fields']['preserved']&~case.audit_baseline['fields']['preserved']).sum()),
        literal_caps_mean_source_limits_and_original_domain_pass=core.tips_pass(state),
        original_search_domain_pass=state.engineering_domain_pass,source_joint_limits_pass=state.physical_joint_limits_pass,
        measured_outside_search_domain=not state.engineering_domain_pass,physical_outcome=None,
        accepted_mapping=False,native_fidelity_accepted=False,full_orientation_continuous_coverage='NOT_PROVED')


def check_budgets(counts,case):
    if (counts['static_forwards']>250 or counts['production_translations']>1 or counts['dex_refinement_attempts']>20
            or counts['physics_step_attempts'] or counts['inference_attempts'] or counts['unauthorized_retarget_attempts']
            or counts['native_pose_solve_attempts']>1 or case.counts['raw_callbacks']>150
            or len(case.checkpoint_artifacts)>3):raise ValueError('Audit numerical/call budget violated')


def install_case_budgets(case):
    original_full=case._full_current
    def raw_budget():
        if case.counts['raw_callbacks']>=150:raise RuntimeError('150 raw-callback budget exhausted')
        case.counts['raw_callbacks']+=1
    def full(*a,**k):
        if len(case.checkpoint_artifacts)>=3:raise RuntimeError('Three full-field scans exhausted before query')
        return original_full(*a,**k)
    case._raw_budget=raw_budget;case._full_current=full


def worker(protocol,freeze_commit,output,index):
    p,commit=validate(protocol,freeze_commit);prior=read(p['v3_protocol'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    counts=dict(static_forwards=0,production_translations=0,native_pose_solve_attempts=0,dex_refinement_attempts=0,
        physics_step_attempts=0,inference_attempts=0,unauthorized_retarget_attempts=0)
    def forbidden(kind):
        def fail(*a,**k):counts[kind]+=1;raise RuntimeError('Production mapping audit forbids '+kind)
        return fail
    def block():
        for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden('physics_step_attempts'))
    block();original=mujoco.mj_forward
    def forward(*a,**k):
        if counts['static_forwards']>=250:raise RuntimeError('250 static-forward budget exhausted')
        counts['static_forwards']+=1;return original(*a,**k)
    mujoco.mj_forward=forward
    sys.path.insert(0,str(ROOT/'src'));sys.path.extend(prior['dependency_dirs']);sys.path.insert(0,str(V3/'numerical_snapshots'))
    from grasp_failure_prediction.evaluation import case_runner,retargeting,runner,surface_pose
    from grasp_failure_prediction.proposals.translation import translate_mano_with_mapping_record,STATIC_MANO_STRATEGY
    from grasp_failure_prediction.integrations.hug import load_hug_prediction
    from dex_retargeting.seq_retarget import SeqRetargeting
    if STATIC_MANO_STRATEGY!=STRATEGY:raise ValueError('Production strategy constant changed')
    case_runner._run_case_inference=forbidden('inference_attempts');case_runner.run_case=forbidden('inference_attempts')
    for cls in (runner.AdroitShadowRunner,runner.ActuatedShadowRunner):
        cls.execute=forbidden('physics_step_attempts');cls._advance=forbidden('physics_step_attempts')
    real_retarget=retargeting.ShadowHandRetargeter.retarget;real_refinement=SeqRetargeting.retarget
    phase={'production':False}
    def native_retarget(self,*a,**k):
        if not phase['production'] or self.profile!=PROFILE or counts['production_translations']!=1 or counts['native_pose_solve_attempts']>=1:
            return forbidden('unauthorized_retarget_attempts')()
        counts['native_pose_solve_attempts']+=1
        return real_retarget(self,*a,**k)
    preprocessing=dict(mano_local_landmarks_m=[],operator_landmarks_m=[],reference_vectors_m=[],target_link_human_indices=[],actual_ref_value=[])
    original_camera=retargeting.camera_landmarks_to_wrist;original_vectors=retargeting.reference_vectors
    def camera_inputs(*a,**k):
        value=original_camera(*a,**k)
        if phase['production']:preprocessing['mano_local_landmarks_m'].append(np.array(value,copy=True))
        return value
    def vector_inputs(landmarks,indices):
        value=original_vectors(landmarks,indices)
        if phase['production']:
            preprocessing['operator_landmarks_m'].append(np.array(landmarks,copy=True))
            preprocessing['target_link_human_indices'].append(np.array(indices,copy=True))
            preprocessing['reference_vectors_m'].append(np.array(value,copy=True))
        return value
    retargeting.camera_landmarks_to_wrist=camera_inputs;retargeting.reference_vectors=vector_inputs
    def refinement(self,*a,**k):
        if not phase['production'] or counts['dex_refinement_attempts']>=20:
            return forbidden('unauthorized_retarget_attempts')()
        preprocessing['actual_ref_value'].append(np.array(a[0] if a else k['ref_value'],copy=True))
        counts['dex_refinement_attempts']+=1;return real_refinement(self,*a,**k)
    retargeting.ShadowHandRetargeter.retarget=native_retarget;SeqRetargeting.retarget=refinement
    bridge=load('static_v2_frozen_v3_bridge',V3/'repair_case.py')
    case=None;result=None;failure=None;checkpoints=[];auxiliary=[];record_binding=None
    started=time.perf_counter()
    try:
        case=bridge.RepairCase(index,prior,output,guard_reinstall=block);block()
        install_case_budgets(case)
        native=Path(case.source['native']['path']);native_before=sha(native)
        baseline=case.original_full();case.audit_baseline=baseline
        counts['production_translations']+=1;phase['production']=True
        grasp=load_hug_prediction(native)
        try:pose,record=translate_mano_with_mapping_record(grasp,STATIC_MANO_STRATEGY)
        finally:phase['production']=False
        record_binding=save_production_evidence(output,record,preprocessing,pose)
        observed_preprocessing=preprocessing_receipt(grasp,pose,preprocessing,counts['dex_refinement_attempts'])
        result=measure(case,pose,record,p['production_configuration'],counts['dex_refinement_attempts'],surface_pose)
        result['preprocessing_receipt']=observed_preprocessing
        result['legacy_baseline']=compact(baseline)
        if sha(native)!=native_before or native_before!=case.native_hash:raise ValueError('Native source changed during mapping audit')
        result['native_before_after_sha256']=native_before
        check_budgets(counts,case)
        visual=case.source_receipt['visual_transform_check']
        checkpoints,auxiliary=checkpoint_index(output,case.checkpoint_artifacts,visual,record_binding,index)
        if len(checkpoints)!=(1 if pose is None else 3):raise ValueError('Audit checkpoint denominator changed')
        util.no_mapping_acceptance(result);validate(protocol,freeze_commit)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
    report=dict(schema_version=SCHEMA,source_inventory_index=index,freeze_commit=commit,protocol_sha256=sha(protocol),
        status='COMPLETED_BOUNDED_STATIC_V2_AUDIT' if failure is None else 'HARD_STOP',failure=failure,
        counts=counts,case_counts=None if case is None else case.counts,elapsed_s=time.perf_counter()-started,
        audit=result if failure is None else None,production_mapping_record=record_binding,
        checkpoint_artifacts=checkpoints if failure is None else None,auxiliary_artifacts=auxiliary if failure is None else None,
        source_receipt=None if case is None else case.source_receipt,
        quarantined_partial_artifacts=util.failure_checkpoint_inventory(output) if failure else [],
        accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,physical_outcome=None,
        new_collection_or_training_allowed=False)
    util.no_mapping_acceptance(report);write(output/'result.json',report)
    return 0 if failure is None else 1


def run(protocol,freeze_commit,output):
    p,commit=validate(protocol,freeze_commit);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(freeze_commit=commit,protocol_sha256=sha(protocol),budgets=BUDGET))
    rows=[]
    for index in INDICES:
        validate(protocol,freeze_commit);log=output/f'case_{index}.log';started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--protocol',str(Path(protocol).resolve()),
            '--freeze-commit',commit,'--output',str(output/str(index)),'--index',str(index)]
        with log.open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
        path=output/str(index)/'result.json'
        good=code==0 and path.exists() and read(path)['status']=='COMPLETED_BOUNDED_STATIC_V2_AUDIT'
        rows.append(dict(index=index,status='COMPLETED' if good else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR',
            returncode=code,elapsed_s=time.perf_counter()-started,log=str(log),log_sha256=sha(log),
            result=str(path) if path.exists() else None,result_sha256=sha(path) if path.exists() else None))
        if not good:break
    done={x['index'] for x in rows};rows += [dict(index=i,status='NOT_RUN',reason='earlier_case_hard_stop') for i in INDICES if i not in done]
    validate(protocol,freeze_commit)
    attempts=attempt_denominator(rows)
    report=dict(schema_version=SCHEMA,freeze_commit=commit,protocol_sha256=sha(protocol),cases=rows,
        status='COMPLETED_BOUNDED_STATIC_V2_PILOT' if all(x['status']=='COMPLETED' for x in rows) else 'HARD_STOP',
        **attempts,accepted_mapping=False,native_fidelity_accepted=False,
        production_promotion=False,new_collection_or_training_allowed=False,physical_outcome=None)
    write(output/'report.json',report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='COMPLETED_BOUNDED_STATIC_V2_PILOT' else 1


def attempt_denominator(rows):
    subtotal=0;unknown=0;started=0
    for row in rows:
        if row.get('status')=='NOT_RUN':continue
        started+=1;path=row.get('result');value=None
        if path and Path(path).is_file() and sha(path)==row.get('result_sha256'):
            value=read(path).get('counts',{}).get('dex_refinement_attempts')
        if type(value) is int and 0<=value<=20:subtotal+=value
        else:unknown+=1
    if subtotal>40 or started>2:raise ValueError('Total forty-refinement/two-case budget exceeded')
    return dict(actual_dex_refinement_attempts=None if unknown else subtotal,
        reported_dex_refinement_attempts_subtotal=subtotal,workers_missing_valid_count_receipt=unknown,
        conservative_dex_refinement_attempts_upper_bound=subtotal+20*unknown,
        started_workers=started,attempt_total_known=unknown==0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,required=True)
    q.add_argument('--v3-protocol',type=Path,default=V3/'protocol.json')
    q.add_argument('--v3-report',type=Path,default=V3/'run_v1/report.json')
    q.add_argument('--v3-readback',type=Path,default=V3/'run_v1/independent_readback.json')
    for name in ('run','worker'):
        q=sub.add_parser(name);q.add_argument('--protocol',type=Path,required=True);q.add_argument('--freeze-commit',required=True)
        q.add_argument('--output',type=Path,required=True)
        if name=='worker':q.add_argument('--index',type=int,choices=INDICES,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args.protocol,args.v3_protocol,args.v3_report,args.v3_readback);return 0
    return run(args.protocol,args.freeze_commit,args.output) if args.command=='run' else worker(args.protocol,args.freeze_commit,args.output,args.index)


if __name__=='__main__':raise SystemExit(main())
