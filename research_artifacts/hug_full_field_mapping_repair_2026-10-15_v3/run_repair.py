"""Freeze/run one finite current-scene full-field mapping repair pilot.

No task collection, native inference or predictor training. Preparation only
reads/copies mathematical source and hashes immutable inputs. A returned local
fit is diagnostic; this program has no native-fidelity PASS capability.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import mujoco

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
V1=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v1'
V2=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v2'
DIAG=ROOT/'research_artifacts/hug_mapping_matmul_diagnosis_2026-10-15_v1'
SCHEMA='hug_current_full_field_exchange_pilot_v3'
INDICES=(20,28)
BUDGET=dict(cases=2,maximum_seconds_per_case=600,maximum_static_forwards_per_case=1500,
            maximum_raw_callbacks_per_case=1500,maximum_full_field_checkpoints_per_case=6,
            maximum_exchange_rounds_per_case=3,maximum_local_model_rows=512,physics_steps=0,native_inference_calls=0,
            official_dex_retarget_calls=0,task_outcomes=0,predictor_fits=0)
LOCAL_CONFIG=dict(tip_inward_target_m=1e-9,local_regularization_weight=1e-6,
                  local_maximum_iterations=200,local_optimizer_ftol=1e-10,epigraph_slack_upper=1000.)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if is_dataclass(value):return {f.name:plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(plain(value),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


SAME_PART_ADMISSION = dict(schema_version='hug_full_field_same_part_admission_v3',minimum_newly_preserved_relationships=1,
    baseline_preserved_retention='all',current_preserved_retention='all',new_definite_violations_allowed=0,accepted_mapping=False,
    domain='complete_fixed_cell_by_same_anatomical_part_by_band_inventory',full_orientation_continuous_coverage='NOT_PROVED')
RESTORATION = dict(maximum_iterations=8,maximum_callbacks=400,activation_margin_m=1e-7)


def validate_core_contract(core):
    expected=dict(MAX_ROUNDS=3,MAX_FULL_CHECKPOINTS=6,RESTORE_MAX_ITERATIONS=8,
        RESTORE_MAX_CALLBACKS=400,RESTORE_ACTIVATION_MARGIN_M=1e-7,RESTORE_INWARD_M=1e-9,MAX_LOCAL_MODEL_ROWS=512)
    if any(getattr(core,key,None)!=value for key,value in expected.items()) or core.SAME_PART_ADMISSION!=SAME_PART_ADMISSION:
        raise ValueError('Core constants/admission differ from the finite v3 declaration')


def visual_artifact_index(output,binding,case_index):
    if not isinstance(binding,dict) or case_index not in INDICES:
        raise ValueError('Exact source-receipt visual binding/case identity required')
    output=Path(output).resolve();directory=output/'visual_transform_check'
    expected_json=directory/f'case_{case_index}_visual_transform.json'
    expected_npz=directory/f'case_{case_index}_visual_transform.npz'
    if (Path(binding['path']).resolve()!=expected_json or Path(binding['operands_and_outputs']['path']).resolve()!=expected_npz
            or expected_json.is_symlink() or expected_npz.is_symlink()):
        raise ValueError('Visual artifact must have its exact per-case output path')
    helper=load('v3_completion_visual_readback',HERE/'visual_transform_check.py')
    receipt=helper.verify_receipt(binding)
    if receipt['case_index']!=case_index:raise ValueError('Visual receipt case identity changed')
    return [dict(path=str(file),sha256=sha(file),bytes=file.stat().st_size,kind=kind)
            for file,kind in ((expected_json,'visual_transform_receipt'),(expected_npz,'visual_transform_operands'))]


def checkpoint_index(output,receipts,visual_binding=None,case_index=None):
    """Verify every checkpoint against its append-only write-time digest."""
    output=Path(output).resolve()
    auxiliary=visual_artifact_index(output,visual_binding,case_index) if visual_binding is not None else []
    allowed={r['path'] for r in auxiliary if r['kind']=='visual_transform_operands'}
    files=sorted(p for p in output.rglob('*.npz') if str(p.resolve()) not in allowed)
    if len(files)>BUDGET['maximum_full_field_checkpoints_per_case']:
        raise ValueError('Checkpoint cap exceeded')
    if not isinstance(receipts,list):raise ValueError('Write-time checkpoint receipts required')
    expected={}
    for row in receipts:
        if not isinstance(row,dict) or set(row)!={'path','sha256','bytes','label'}:
            raise ValueError('Malformed write-time checkpoint receipt')
        path=Path(row['path']).resolve()
        if path.parent!=output or path.name in expected or not path.name.startswith('checkpoint_') or path.suffix!='.npz':
            raise ValueError('Duplicate/outside/unrecognized checkpoint receipt')
        if isinstance(row['bytes'],bool) or not isinstance(row['bytes'],int) or row['bytes']<=0:
            raise ValueError('Positive checkpoint byte count required')
        expected[path.name]=row
    if {f.name for f in files}!=set(expected):raise ValueError('Missing or unindexed checkpoint arrays')
    index=[]
    for number,file in enumerate(files,1):
        row=expected[file.name]
        if file.is_symlink() or not file.is_file() or not file.name.startswith(f'checkpoint_{number:02d}_'):
            raise ValueError('Checkpoint sequence/regular file required')
        if sha(file)!=row['sha256'] or file.stat().st_size!=row['bytes']:
            raise ValueError('Write-time checkpoint arrays changed: '+file.name)
        index.append(dict(path=str(file),sha256=row['sha256'],bytes=row['bytes'],label=row['label']))
    return index


def no_mapping_acceptance(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('accepted_mapping','full_strict_mapping_accepted','native_fidelity_accepted','production_promotion') and item is not False:
                raise ValueError('Static diagnostic cannot issue mapping acceptance: '+key)
            no_mapping_acceptance(item)
    elif isinstance(value,(tuple,list)):
        for item in value:no_mapping_acceptance(item)


def same_part_receipt(result):
    """Independent receipt on every indexed cell/part/band; no aggregate tradeoff."""
    baseline=result['baseline'];returned=result['returned'];a=baseline['fields'];b=returned['fields']
    for key in ('preserved','required','definite_lost','definite_added'):
        aa=np.asarray(a[key]);bb=np.asarray(b[key])
        if aa.dtype!=bool or bb.dtype!=bool or aa.shape!=bb.shape or aa.ndim!=3 or aa.shape[1:]!=(6,3):
            raise ValueError('Full same-part six-part/three-band boolean fields required')
    if not np.array_equal(a['required'],b['required']):raise ValueError('Native-definite domain changed')
    lost=int((a['preserved']&~b['preserved']).sum())
    new=int(((b['definite_lost']|b['definite_added'])&~(a['definite_lost']|a['definite_added'])).sum())
    repaired=int((b['preserved']&~a['preserved']).sum())
    ori_a=baseline['orientation'];ori_b=returned['orientation']
    ori=True
    for key in ('valid','passed'):
        aa=np.asarray(ori_a[key]);bb=np.asarray(ori_b[key])
        if aa.dtype!=bool or bb.dtype!=bool or aa.shape!=bb.shape:raise ValueError('Orientation witness masks changed')
        ori=ori and bool(np.all(~aa|bb))
    state=returned['state']
    def field(key):return state[key] if isinstance(state,dict) else getattr(state,key)
    errors=np.asarray(field('tip_errors_world_m'),float);caps=np.asarray(field('original_tip_caps_m'),float)
    if errors.shape!=(5,3) or caps.shape!=(5,) or not np.isfinite(errors).all() or not np.isfinite(caps).all():
        raise ValueError('Finite serialized fingertip errors/original caps required')
    norms=np.sqrt(np.einsum('ij,ij->i',errors,errors,optimize=False))
    literal_tips=bool(np.all(norms<=caps) and norms.mean()<=.010)
    domain=bool(field('physical_joint_limits_pass') and field('engineering_domain_pass'))
    protected=(lost==0 and new==0 and ori)
    refused=str(result.get('status','')).startswith('REFUSED_')
    admitted=bool(repaired>=1 and protected and literal_tips and domain and not refused)
    if (isinstance(result.get('repaired_relationship_count'),bool) or result.get('repaired_relationship_count')!=repaired
            or result.get('accepted_relationship_repair') is not admitted):
        raise ValueError('Repair credit contradicts full same-part admission')
    if not refused and not protected:raise ValueError('Returned pose loses original same-part protection')
    return dict(policy=dict(SAME_PART_ADMISSION),required_relationships=int(a['required'].sum()),
        minimum_baseline_preserved_relationships=int(a['preserved'].sum()),returned_preserved_relationships=int(b['preserved'].sum()),
        lost_baseline_preserved=lost,new_definite_violations=new,newly_preserved_relationships=repaired,
        retained_orientation_protected=ori,literal_original_tips_pass=literal_tips,physical_and_engineering_domain_pass=domain,
        local_relationship_repair_admitted=admitted,accepted_mapping=False,continuous_orientation_coverage='NOT_PROVED')


def compact_fit(result,output,receipts,visual_binding=None,case_index=None):
    """Bind all checkpoint arrays, including refused/unselected intermediates."""
    index=checkpoint_index(output,receipts,visual_binding,case_index)
    if result is None:return None
    no_mapping_acceptance(result)
    copied=dict(result);indexed={r['path']:r for r in index}
    for key in ('baseline','returned'):
        full=copied.get(key)
        if not isinstance(full,dict):raise ValueError('Complete baseline/returned field receipts required')
        path=str(Path(full['array_path']).resolve())
        if path not in indexed or indexed[path]['sha256']!=full['array_sha256']:
            raise ValueError('Baseline/returned array is outside the checkpoint index')
        with np.load(path,allow_pickle=False) as saved:
            for name,value in full['fields'].items():
                if name not in saved or not np.array_equal(saved[name],value):raise ValueError('Saved field does not match compacted state')
            for name in ('margin_m','distance_m'):
                if name not in saved or not np.array_equal(saved[name],full[name]):raise ValueError('Saved numerical field differs')
        row={k:v for k,v in full.items() if k not in ('fields','margin_m','distance_m')}
        row['full_arrays_in_bound_npz']=True
        row['field_boolean_counts']={k:int(v.sum()) for k,v in full['fields'].items()}
        copied[key]=row
    copied['checkpoint_artifacts']=index
    copied['auxiliary_artifacts']=visual_artifact_index(output,visual_binding,case_index) if visual_binding is not None else []
    copied['independent_same_part_admission_receipt']=same_part_receipt(result)
    checkpoint_index(output,index,visual_binding,case_index)
    return copied


def validate_completion_budget(counts,case_counts,receipts):
    limits={'static_forwards':'maximum_static_forwards_per_case'}
    for key,limit in limits.items():
        if counts.get(key,-1)<0 or counts[key]>BUDGET[limit]:raise ValueError('Static budget differs at completion')
    for key in ('physics_step_attempts','inference_attempts','official_dex_retarget_attempts'):
        if counts.get(key)!=0:raise ValueError('Forbidden API attempted')
    if (case_counts.get('raw_callbacks',-1)<0 or case_counts['raw_callbacks']>BUDGET['maximum_raw_callbacks_per_case']
            or case_counts.get('full_field_checkpoints')!=len(receipts)
            or len(receipts)>BUDGET['maximum_full_field_checkpoints_per_case']):
        raise ValueError('Callback/checkpoint budget or write-time inventory differs')


def failure_checkpoint_inventory(output):
    """Observed hashes of quarantined partial files; not verified completion."""
    return [dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size,verified_write_time=False)
            for p in sorted(Path(output).rglob('*.npz')) if p.is_file() and not p.is_symlink()]


def runtime():
    import scipy
    # Preserve the base numerical imports above before using the declared
    # existing dependency bundle. Importing runtime metadata builds no scene.
    for directory in read(V1/'protocol.json')['dependency_dirs']:
        if directory not in sys.path:sys.path.append(directory)
    import numba
    return dict(python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__,
                scipy=scipy.__version__,numba=numba.__version__,executable=sys.executable)


def verify(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Bound bytes changed: '+path)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def numerical_clearance(report):
    """Scoped arithmetic clearance, never a distance/fidelity certificate."""
    a=report.get('actual',{})
    if (report.get('status')!='COMPLETED_BOUNDED_OPERAND_DIAGNOSTIC' or report.get('failure') is not None
            or report.get('poses_completed')!=[[20,'original_20'],[20,'experimental_20'],[28,'original_28'],[28,'experimental_28']]
            or report.get('poses_not_run_or_partial') or a.get('inside_mask_disagreements')!=0
            or a.get('hull_predicate_disagreements')!=0 or a.get('einsum_warnings')!=[]):
        raise ValueError('Actual checked halfspace arithmetic is not cleared')
    for key in ('max_matmul_einsum_difference_m','max_einsum_scalar_difference_m','max_matmul_scalar_difference_m'):
        value=a.get(key)
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not np.isfinite(value) or value<0 or value>1e-14:
            raise ValueError('Actual halfspace reference discrepancy exceeds declared arithmetic comparison')


def prior_evidence_clearance(report,independent):
    if (report.get('status')!='COMPLETED_BOUNDED_PILOT' or report.get('native_fidelity_accepted') is not False
            or report.get('new_collection_or_training_allowed') is not False
            or independent.get('status')!='VERIFIED_SAVED_ARRAYS_AND_REPORT_INTERPRETATION'
            or independent.get('native_fidelity_accepted') is not False
            or independent.get('report_sha256')!=sha(V2/'run_v1/report.json')
            or [r.get('index') for r in report.get('cases',[])]!=list(INDICES)
            or [r.get('index') for r in independent.get('cases',[])]!=list(INDICES)):
        raise ValueError('Completed immutable v2 evidence and independent readback required')
    for row in report['cases']:
        if row.get('status')!='COMPLETED':raise ValueError('V2 evidence case is incomplete')
        for key in ('result','log'):
            if sha(ROOT/row[key])!=row[key+'_sha256']:raise ValueError('V2 evidence output changed')
    for row in independent['cases']:
        official=next(r for r in report['cases'] if r['index']==row['index'])
        if row['result_sha256']!=official['result_sha256']:raise ValueError('V2 readback/result identity differs')
        for checkpoint in row['checkpoints']:
            if sha(ROOT/checkpoint['path'])!=checkpoint['sha256']:
                raise ValueError('Bound v2 intermediate checkpoint changed')


def prepare(protocol):
    if Path(protocol).exists():raise FileExistsError('Protocol is immutable')
    v1=read(V1/'protocol.json');v2=read(V2/'protocol.json');verify(v2['bindings'])
    evidence=read(V2/'run_v1/report.json');independent=read(V2/'run_v1/independent_readback.json')
    prior_evidence_clearance(evidence,independent)
    arithmetic=DIAG/'run_v1/report.json';numerical_clearance(read(arithmetic))
    inputs=dict(v2['bindings'])
    for row in evidence['cases']:
        for key in ('result','log'):
            file=(ROOT/row[key]).resolve();inputs[str(file)]=sha(file)
    for row in independent['cases']:
        for checkpoint in row['checkpoints']:
            file=(ROOT/checkpoint['path']).resolve();inputs[str(file)]=checkpoint['sha256']
    for path in (V1/'protocol.json',V1/'run_v1/report.json',V2/'protocol.json',V2/'run_v1/report.json',V2/'run_v1/independent_readback.json',arithmetic,DIAG/'protocol.json',DIAG/'postrun_readback.json'):
        inputs[str(path.resolve())]=sha(path)
    # Sources in temporary historical studies are numerical reference inputs.
    # Freeze local copies; never import any historical Case or call old solve().
    refs={name:Path(path) for name,path in v2['numerical_snapshots'].items()}
    snapshots={}
    for name,source in refs.items():
        target=HERE/'numerical_snapshots'/name;target.parent.mkdir(parents=True,exist_ok=True)
        data=source.read_bytes()
        if target.exists() and target.read_bytes()!=data:raise ValueError('Numerical snapshot differs')
        if not target.exists():target.write_bytes(data)
        inputs[str(source.resolve())]=sha(source);inputs[str(target.resolve())]=sha(target)
        snapshots[name]=str(target.resolve())
    witness=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
    domain=ROOT/'research_artifacts/hug_free_palm_fit_2026-10-15_v1/candidate_cohort_12_20_28.json.gz'
    orientation=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json'
    required=[Path(__file__),HERE/'exchange_core.py',HERE/'repair_case.py',HERE/'geometry_kernel.py',HERE/'geometry_patch.json',
              HERE/'README.md',HERE/'controls.json',HERE/'visual_transform_check.py',HERE/'VISUAL_TRANSFORM_CHECK.md',
              ROOT/'tests/test_hug_full_field_exchange_v3.py',
              ROOT/'src/grasp_failure_prediction/evaluation/surface_pose.py',ROOT/'tests/test_surface_pose.py',
              ROOT/'tests/test_hug_full_field_exchange.py',
              witness,domain,orientation,
              ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/check_orientation.py',
              ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py']
    tests=sorted(HERE.glob('test_*.py'))
    if not tests:raise ValueError('Manufactured exchange/kernel/bridge controls required before preparation')
    required.extend(tests)
    inventory=read(v1['inventory'])
    for index in INDICES:
        source=inventory['modern_cases'][index]
        required.extend([Path(source['scene_mjb']['path']),Path(source['frame_files']['scene_state.npz']['path'])])
        required.append(V1/f'run_v1/{index}/original_{index}_density2mm.npz')
    for path in required:inputs[str(path.resolve())]=sha(path)
    declaration=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),
        runtime=runtime(),bindings=inputs,budgets=dict(BUDGET),local_configuration=dict(LOCAL_CONFIG),
        restoration=dict(RESTORATION),same_part_admission=dict(SAME_PART_ADMISSION),
        v2_protocol=str((V2/'protocol.json').resolve()),v2_report=str((V2/'run_v1/report.json').resolve()),
        v2_independent_readback=str((V2/'run_v1/independent_readback.json').resolve()),
        source_inventory_indices=list(INDICES),v1_protocol=str((V1/'protocol.json').resolve()),
        v1_report=str((V1/'run_v1/report.json').resolve()),runtime_project_root=v1['runtime_project_root'],
        dependency_dirs=v1['dependency_dirs'],geometry_kernel=str((HERE/'geometry_kernel.py').resolve()),
        exact_direction_source=snapshots['exact_direction.py'],constraint_core_source=snapshots['constraint_core.py'],
        local_minimax_source=snapshots['feasible_eligibility_solver.py'],numerical_snapshots=snapshots,
        orientation_saved_witnesses=str(witness),orientation_protocol=str(orientation),domain_source=str(domain),
        scientific_acceptance_sha256=v1['scientific_rule_sha256'],arithmetic_report=str(arithmetic.resolve()),
        scope='Two repeatedly inspected original development grasps; current Shadow geometry; static diagnostic repair only',
        historical_binary_scope='One static forward per case solely for frozen object visual orientation reference; never historical target-hand queries',
        algorithm=dict(original_relative_dof=30,scales=[.025,.005,.025],joint_delta_limit_rad=.35,
            root_rotation_component_bound_degrees=15,exchange_rounds=3,restoration_maximum_iterations=8,
            restoration_maximum_callbacks=400,restoration_activation_margin_m=1e-7,maximum_local_model_rows=512,
            restoration_inward_search_target_m=1e-9,finite_difference_step_scaled=.0001,initial_trust_scaled=.25,
            dyadic_exponents=list(range(11)),full_field_query_chunk_size=256,numba_threads=2,
            active_selection='all retained full-field guard counterexamples plus deterministic worst native-definite unpreserved row per part/band/near-far; complete union retained',
            unresolved_guard_boundary='preservation margin + 2*(cell radius + frozen numerical allowance); near inclusive, far strict with 1 nm inward search target',
            protection='all original and accumulated current passing full-field rows, no new definite failures, exact witness eligibility/passing',
            progress='failure-to-ambiguity or margin-only motion is not a repaired relationship',
            original_baseline_readback='v1 masks/query/native arrays exact; mapped distances <=1e-12 m numerical comparison only',
            final='fresh full-field arrays/masks/orientation plus literal named-root serialization checked independently'),
        full_orientation_continuous_coverage='NOT_PROVED',production_promotion=False,
        thresholds_changed=False,historical_caps_rebased=False,new_task_collection_allowed=False)
    write(protocol,declaration);return declaration


def validate(protocol,freeze_commit):
    p=read(protocol)
    if (p.get('schema_version')!=SCHEMA or p.get('budgets')!=BUDGET or p.get('local_configuration')!=LOCAL_CONFIG
            or p.get('source_inventory_indices')!=list(INDICES) or p.get('runtime')!=runtime()
            or p.get('restoration')!=RESTORATION or p.get('same_part_admission')!=SAME_PART_ADMISSION):
        raise ValueError('Pilot declaration/runtime differs')
    verify(p['bindings']);numerical_clearance(read(p['arithmetic_report']))
    prior_evidence_clearance(read(p['v2_report']),read(p['v2_independent_readback']))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=head:raise ValueError('HEAD must remain the exact freeze commit')
    selected=[Path(protocol).resolve(),Path(__file__).resolve(),HERE/'exchange_core.py',HERE/'repair_case.py',HERE/'geometry_kernel.py']
    selected += [Path(x) for x in p['numerical_snapshots'].values()]
    selected += sorted(HERE.glob('test_*.py'))
    selected += [HERE/'README.md',HERE/'controls.json',HERE/'visual_transform_check.py',HERE/'VISUAL_TRANSFORM_CHECK.md',
                 ROOT/'tests/test_hug_full_field_exchange_v3.py',ROOT/'tests/test_hug_full_field_exchange.py',ROOT/'tests/test_surface_pose.py',
                 ROOT/'src/grasp_failure_prediction/evaluation/surface_pose.py']
    names=[str(x.relative_to(ROOT)) for x in selected]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*names],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*names],check=True)
    return p,commit


def worker(protocol,freeze_commit,output,index):
    p,commit=validate(protocol,freeze_commit)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    counts=dict(static_forwards=0,physics_step_attempts=0,inference_attempts=0,official_dex_retarget_attempts=0)
    def forbidden(kind):
        def fail(*a,**kw):counts[kind]+=1;raise RuntimeError('Static repair forbids '+kind)
        return fail
    def block():
        for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden('physics_step_attempts'))
    block();original=mujoco.mj_forward
    def forward(*a,**kw):
        if counts['static_forwards']>=BUDGET['maximum_static_forwards_per_case']:raise RuntimeError('Static forward budget exhausted')
        counts['static_forwards']+=1;return original(*a,**kw)
    mujoco.mj_forward=forward
    sys.path.insert(0,str(ROOT/'src'));sys.path.extend(p['dependency_dirs'])
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'numerical_snapshots'))
    from grasp_failure_prediction.evaluation import case_runner,retargeting,runner
    retargeting.ShadowHandRetargeter.retarget=forbidden('official_dex_retarget_attempts')
    case_runner._run_case_inference=forbidden('inference_attempts');case_runner.run_case=forbidden('inference_attempts')
    for cls in (runner.AdroitShadowRunner,runner.ActuatedShadowRunner):
        cls.execute=forbidden('physics_step_attempts');cls._advance=forbidden('physics_step_attempts')
    import exchange_core,repair_case
    validate_core_contract(exchange_core)
    math=load('bound_local_minimax',p['local_minimax_source'])
    log=output/'events.jsonl';started=time.perf_counter()
    def emit(event,**value):
        with log.open('a') as f:f.write(json.dumps(plain(dict(event=event,**value)),sort_keys=True,allow_nan=False)+'\n')
    result=None;case=None;failure=None;final=None;compacted=None;checkpoints=None;auxiliary=None
    try:
        case=repair_case.RepairCase(index,p,output,guard_reinstall=block);block()
        result=exchange_core.solve(case,math.local_minimax_step,p['local_configuration'],emit)
        final=case.finalize(result)
        import numba
        if numba.get_num_threads()!=2:raise ValueError('Effective thread count changed')
        if any(counts[k] for k in ('physics_step_attempts','inference_attempts','official_dex_retarget_attempts')):
            raise RuntimeError('Forbidden API attempted')
        validate(protocol,freeze_commit)
        validate_completion_budget(counts,case.counts,case.checkpoint_artifacts)
        visual_binding=case.source_receipt['visual_transform_check']
        compacted=compact_fit(result,output,case.checkpoint_artifacts,visual_binding,index)
        checkpoints=checkpoint_index(output,case.checkpoint_artifacts,visual_binding,index)
        auxiliary=visual_artifact_index(output,visual_binding,index)
        validate(protocol,freeze_commit)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
    report=dict(schema_version=SCHEMA,source_inventory_index=index,protocol_sha256=sha(protocol),freeze_commit=commit,
        status='COMPLETED_BOUNDED_STATIC_REPAIR' if failure is None else 'HARD_STOP',failure=failure,
        counts=counts,case_counts=None if case is None else case.counts,elapsed_s=time.perf_counter()-started,
        fit=compacted if failure is None else None,final_readback=final if failure is None else None,
        checkpoint_artifacts=checkpoints if failure is None else None,auxiliary_artifacts=auxiliary if failure is None else None,
        quarantined_partial_checkpoint_artifacts=failure_checkpoint_inventory(output) if failure else [],
        native_fidelity_accepted=False,production_promotion=False,
        native_fidelity_scope='Full required proximity field plus capped exact orientation witnesses; full continuous orientation not proved')
    if failure is None:no_mapping_acceptance(report)
    write(output/'result.json',report)
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
            try:code=child.wait(timeout=BUDGET['maximum_seconds_per_case'])
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
        result=output/str(index)/'result.json'
        ok=code==0 and result.exists() and read(result)['status']=='COMPLETED_BOUNDED_STATIC_REPAIR'
        rows.append(dict(index=index,status='COMPLETED' if ok else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR',
            returncode=code,elapsed_s=time.perf_counter()-started,log=str(log),log_sha256=sha(log),
            result=str(result) if result.exists() else None,result_sha256=sha(result) if result.exists() else None))
        if not ok:break
    done={x['index'] for x in rows}
    rows += [dict(index=i,status='NOT_RUN',reason='earlier_case_hard_stop') for i in INDICES if i not in done]
    validate(protocol,freeze_commit)
    report=dict(schema_version=SCHEMA,freeze_commit=commit,protocol_sha256=sha(protocol),cases=rows,
        status='COMPLETED_BOUNDED_PILOT' if all(x['status']=='COMPLETED' for x in rows) else 'HARD_STOP',
        native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'report.json',report);print(json.dumps(report),flush=True)
    return 0 if report['status']=='COMPLETED_BOUNDED_PILOT' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,required=True)
    for command in ('run','worker'):
        q=sub.add_parser(command);q.add_argument('--protocol',type=Path,required=True);q.add_argument('--freeze-commit',required=True)
        q.add_argument('--output',type=Path,required=True)
        if command=='worker':q.add_argument('--index',type=int,choices=INDICES,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args.protocol);return 0
    return run(args.protocol,args.freeze_commit,args.output) if args.command=='run' else worker(args.protocol,args.freeze_commit,args.output,args.index)


if __name__=='__main__':raise SystemExit(main())
