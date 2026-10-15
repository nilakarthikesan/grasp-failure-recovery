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
DIAG=ROOT/'research_artifacts/hug_mapping_matmul_diagnosis_2026-10-15_v1'
SCHEMA='hug_current_full_field_exchange_pilot_v2'
INDICES=(20,28)
BUDGET=dict(cases=2,maximum_seconds_per_case=600,maximum_static_forwards_per_case=1500,
            maximum_raw_callbacks_per_case=1500,maximum_full_field_checkpoints_per_case=5,
            maximum_exchange_rounds_per_case=2,physics_steps=0,native_inference_calls=0,
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


def compact_fit(result):
    """Keep full fields in their bound NPZ receipts, not duplicated JSON."""
    if result is None:return None
    copied=dict(result)
    for key in ('baseline','returned'):
        full=copied.get(key)
        if not isinstance(full,dict):continue
        if sha(full['array_path'])!=full['array_sha256']:raise ValueError('Saved full-field arrays changed')
        row={k:v for k,v in full.items() if k not in ('fields','margin_m','distance_m')}
        row['full_arrays_in_bound_npz']=True
        row['field_boolean_counts']={k:int(v.sum()) for k,v in full['fields'].items()}
        copied[key]=row
    return copied


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


def prepare(protocol):
    if Path(protocol).exists():raise FileExistsError('Protocol is immutable')
    v1=read(V1/'protocol.json');verify(v1['bindings'])
    arithmetic=DIAG/'run_v1/report.json';numerical_clearance(read(arithmetic))
    inputs=dict(v1['bindings'])
    for path in (V1/'protocol.json',V1/'run_v1/report.json',arithmetic,DIAG/'protocol.json',DIAG/'postrun_readback.json'):
        inputs[str(path.resolve())]=sha(path)
    # Sources in temporary historical studies are numerical reference inputs.
    # Freeze local copies; never import any historical Case or call old solve().
    refs={
        'exact_direction.py':ROOT/'research_artifacts/hug_surface_constraints_2026-10-15_v2/exact_direction.py',
        'constraint_core.py':ROOT/'research_artifacts/hug_surface_constraints_2026-10-15_v2/constraint_core.py',
        'feasible_eligibility_solver.py':ROOT/'research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/eligibility_guidance/feasible_eligibility_solver.py',
        'extended_policy.py':ROOT/'research_artifacts/hug_mapping_acceptance_repair_2026-10-15_v1/eligibility_guidance/extended_policy.py',
        'protected_surface_policy.py':ROOT/'research_artifacts/hug_mapping_process_repair_2026-10-15_v1/protected_fit/protected_surface_policy.py',
    }
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
            root_rotation_component_bound_degrees=15,exchange_rounds=2,restoration_maximum_iterations=25,
            restoration_inward_search_target_m=1e-9,finite_difference_step_scaled=.0001,initial_trust_scaled=.25,
            dyadic_exponents=list(range(11)),full_field_query_chunk_size=256,numba_threads=2,
            active_selection='deterministic worst native-definite unpreserved row per part/band/near-far; union retained',
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
            or p.get('source_inventory_indices')!=list(INDICES) or p.get('runtime')!=runtime()):
        raise ValueError('Pilot declaration/runtime differs')
    verify(p['bindings']);numerical_clearance(read(p['arithmetic_report']))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=head:raise ValueError('HEAD must remain the exact freeze commit')
    selected=[Path(protocol).resolve(),Path(__file__).resolve(),HERE/'exchange_core.py',HERE/'repair_case.py',HERE/'geometry_kernel.py']
    selected += [Path(x) for x in p['numerical_snapshots'].values()]
    selected += sorted(HERE.glob('test_*.py'))
    selected += [ROOT/'tests/test_hug_full_field_exchange.py',ROOT/'tests/test_surface_pose.py',
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
    if exchange_core.MAX_ROUNDS!=2 or exchange_core.RESTORE_INWARD_M!=1e-9:raise ValueError('Changed exchange parameters')
    math=load('bound_local_minimax',p['local_minimax_source'])
    log=output/'events.jsonl';started=time.perf_counter()
    def emit(event,**value):
        with log.open('a') as f:f.write(json.dumps(plain(dict(event=event,**value)),sort_keys=True,allow_nan=False)+'\n')
    result=None;case=None;failure=None;final=None
    try:
        case=repair_case.RepairCase(index,p,output,guard_reinstall=block);block()
        result=exchange_core.solve(case,math.local_minimax_step,p['local_configuration'],emit)
        final=case.finalize(result)
        import numba
        if numba.get_num_threads()!=2:raise ValueError('Effective thread count changed')
        if any(counts[k] for k in ('physics_step_attempts','inference_attempts','official_dex_retarget_attempts')):
            raise RuntimeError('Forbidden API attempted')
        validate(protocol,freeze_commit)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
    report=dict(schema_version=SCHEMA,source_inventory_index=index,protocol_sha256=sha(protocol),freeze_commit=commit,
        status='COMPLETED_BOUNDED_STATIC_REPAIR' if failure is None else 'HARD_STOP',failure=failure,
        counts=counts,case_counts=None if case is None else case.counts,elapsed_s=time.perf_counter()-started,
        fit=compact_fit(result),final_readback=final,native_fidelity_accepted=False,production_promotion=False,
        native_fidelity_scope='Full required proximity field plus capped exact orientation witnesses; full continuous orientation not proved')
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
