"""Separately freeze the static audit of the completed literal-shape build."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
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
BUILD_PROTOCOL_NAME='build_protocol_v2.json'
SCHEMA='hug_literal_shape_manosim_static_transfer_v2'
UPSTREAM='c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac'
INDICES=(20,28)
BUDGET=dict(cases=2,maximum_seconds_per_case=600,maximum_static_forwards_per_case=200,
    maximum_parametric_mano_forwards_per_case=2,maximum_full_field_variant_scans_per_case=2,
    physics_steps=0,HUG_generator_calls=0,official_retarget_calls=0,predictor_fits=0,task_outcomes=0)
NUMERICAL=dict(joint_anchor_comparison_limit_m=5e-6,rotation_element_comparison_limit=1e-6,
    saved_native_reconstruction_limit_m=5e-6,scientific_surface_acceptance_allowance=False)
REPRESENTATION=dict(wrist='float64 evaluation of the original HUG 6D Gram-Schmidt formula; compare to source float32 and saved wrist',
    fingers='float64 Rodrigues exponential of observed mean-adjusted MANO full_poses; compare to actual source full_rots',
    camera='literal calibration proper SE3; no camera projection or fitted transform',
    adapter='literal proper SO3 inputs; quaternion serialization without fitted correction',
    source_residual_limit=1e-6,silent_SVD_projection=False)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def verify(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Bound source/input changed: '+path)
def runtime():
    util=load('manosim_bound_runtime',V3/'run_repair.py');base=util.runtime()
    import torch
    return dict(base,torch=torch.__version__,parametric_device='cpu')

def upstream_identity(root,paths):
    root=Path(root).resolve()
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=UPSTREAM:raise ValueError('Upstream revision changed')
    relative=sorted({str(Path(x).resolve().relative_to(root)) for x in paths if Path(x).resolve().is_relative_to(root)})
    if not relative:raise ValueError('Measured upstream asset/source closure required')
    subprocess.run(['git','-C',str(root),'ls-files','--error-unmatch','--',*relative],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(root),'diff','--quiet','HEAD','--',*relative],check=True)

def completed_v3():
    protocol=V3/'protocol.json';p=read(protocol)
    report=V3/'run_v1/report.json';r=read(report)
    receipt=V3/'run_v1/independent_readback.json';a=read(receipt)
    if r.get('status')!='COMPLETED_BOUNDED_PILOT' or r.get('protocol_sha256')!=sha(protocol):
        raise ValueError('Completed active v3 evidence required')
    # Its independent receipt is also bound literally; no unused preparation
    # declaration can replace the active protocol's bytes.
    if (a.get('active_protocol_sha256')!=sha(protocol) or a.get('freeze_commit')!=r['freeze_commit']
            or a.get('status')!='VERIFIED_SAVED_ARRAYS_AND_REPORT_INTERPRETATION'
            or a.get('input_source_bindings_verified_before_and_after')!=len(p['bindings'])
            or a.get('HEAD_remained_fixed') is not True or a.get('source_git_blobs_checked') is not True):
        raise ValueError('Independent v3 receipt must identify the actual freeze')
    if [c.get('index') for c in r.get('cases',[])]!=list(INDICES):
        raise ValueError('Both completed v3 development cases required')
    inventory=a.get('artifact_inventory',{})
    def independently_bound(file):
        file=Path(file);entry=inventory.get(str(file.relative_to(ROOT)))
        if not isinstance(entry,dict) or entry.get('sha256')!=sha(file) or entry.get('bytes')!=file.stat().st_size:
            raise ValueError('Prior output bytes are not independently verified')
    independently_bound(report)
    independently_checked={c['index']:c for c in a.get('cases',[])}
    for row in r['cases']:
        result=read(row['result'])
        independently_bound(row['result'])
        if (row.get('status')!='COMPLETED' or sha(row['result'])!=row['result_sha256']
                or result.get('protocol_sha256')!=sha(protocol) or result.get('freeze_commit')!=r['freeze_commit']
                or independently_checked.get(row['index'],{}).get('result_sha256')!=row['result_sha256']):
            raise ValueError('Completed v3 case/protocol identity mismatch')
    return p,[protocol,report,receipt]


def completed_build():
    """Require the completed stage1 and its independent hash readback."""
    protocol=HERE/BUILD_PROTOCOL_NAME;p=read(protocol)
    directory=HERE/'build_run_v1';report_path=directory/'report.json';report=read(report_path)
    worker_path=directory/'worker_result.json';worker=read(worker_path)
    builder_path=directory/'builder_report.json';builder=read(builder_path)
    receipt_path=directory/'independent_readback.json';receipt=read(receipt_path)
    if (report.get('status')!='COMPLETE' or report.get('protocol_sha256')!=sha(protocol)
            or report.get('worker_result_sha256')!=sha(worker_path)
            or worker.get('status')!='COMPLETE' or worker.get('protocol_sha256')!=sha(protocol)
            or worker.get('freeze_commit')!=report.get('freeze_commit')
            or worker.get('builder_report_sha256')!=sha(builder_path)
            or (directory/'raw/build_report.json').read_bytes()!=builder_path.read_bytes()):
        raise ValueError('Completed separately frozen shape build required')
    helper=load('audit_literal_shape_producer',HERE/'shape_build.py')
    if builder.get('build_protocol_sha256')!=sha(protocol) or builder.get('plan_sha256')!=helper.digest(p['plan']):
        raise ValueError('Exact shape build declaration/producer identity differs')
    generated=helper.generated_asset_selection(builder_path)
    if (receipt.get('schema_version')!='manosim_exact_shape_build_independent_saved_readback_v1'
            or receipt.get('status')!='VERIFIED_LITERAL_SHAPE_BUILD_ARTIFACTS_AND_PRODUCER'
            or receipt.get('build_protocol_sha256')!=sha(protocol)
            or receipt.get('freeze_commit')!=report['freeze_commit']
            or receipt.get('transaction_report_sha256')!=sha(report_path)
            or receipt.get('worker_result_sha256')!=sha(worker_path)
            or receipt.get('builder_report_sha256')!=sha(builder_path)
            or receipt.get('all_bindings_verified_before') is not True
            or receipt.get('all_bindings_verified_after') is not True
            or type(receipt.get('expected_bindings')) is not int
            or receipt.get('expected_bindings')!=len(p['bindings'])):
        raise ValueError('Independent completed-build receipt must identify actual stage1')
    inventory=receipt.get('artifact_inventory',{})
    for row in builder['artifacts']:
        observed=inventory.get(row['path'])
        if not isinstance(observed,dict) or observed.get('sha256')!=row['sha256'] or observed.get('bytes')!=row['bytes']:
            raise ValueError('Independent producer artifact coverage incomplete')
    verify(p['bindings']);helper.verify_plan(p['plan'])
    evidence=[protocol,report_path,worker_path,builder_path,receipt_path,directory/'raw/build_report.json']
    return p,generated,evidence


def source_partitions(bindings,generated_root,measurement_sources,inherited_tracked_sources):
    """Git-check only selected measured sources, hash-check every inherited byte.

    Inherited hashes include preserved untracked work and ignored runtime assets;
    their presence in the input closure does not turn them into Git sources.
    """
    raw=Path(generated_root).resolve()
    if not raw.is_relative_to(ROOT):raise ValueError('Declared build raw root must belong to this checkout')
    selected={str(Path(path).resolve()) for path in (*measurement_sources,*inherited_tracked_sources)
        if Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(raw)}
    if any(path not in bindings for path in selected):raise ValueError('Selected Git source missing byte binding')
    return (sorted(selected),
        {path:value for path,value in bindings.items() if Path(path).is_relative_to(raw)})

def prepare(path):
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite a declaration')
    v3,evidence=completed_v3();build,generated,build_evidence=completed_build()
    p=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),
        runtime=runtime(),budgets=dict(BUDGET),numerical_comparisons=dict(NUMERICAL),rotation_representation=dict(REPRESENTATION),
        indices=list(INDICES),v3_protocol=str((V3/'protocol.json').resolve()),
        hug_root='/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug',
        manosim_root='/private/tmp/manosim-c96e2d5-2026-10-15',
        mano_asset_path='/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl',
        adapter_source=str(ROOT/'src/grasp_failure_prediction/evaluation/manosim_pose.py'),
        builder_report=str((HERE/'build_run_v1/builder_report.json').resolve()),generated_assets=generated,
        build_protocol=str((HERE/BUILD_PROTOCOL_NAME).resolve()),build_protocol_sha256=sha(HERE/BUILD_PROTOCOL_NAME),
        build_readback=str((HERE/'build_run_v1/independent_readback.json').resolve()),
        geometry_kernel=str(V3/'geometry_kernel.py'),
        geometry_base=str(Path(v3['v1_protocol']).parent/'source_snapshots/audit_other_surfaces.py'),
        upstream_commit=UPSTREAM,shared_shape_sha256=generated['expected_shape_sha256'],
        scientific_acceptance_sha256=v3['scientific_acceptance_sha256'],
        scope='Two repeatedly inspected development natives, exact saved shape generated by pinned upstream right-only builder; both collision variants, static only',
        kinematic_gate='5 um anchors/reconstruction and 1e-6 rotation elements are numerical comparisons only',
        surface_gate='Unchanged same-part full fields; full orientation NOT_RUN; cannot issue surface PASS',
        production_promotion=False,new_collection_or_training_allowed=False)
    sources=load('manosim_source_inventory',HERE/'audit_case.py').source_inputs(p)
    upstream=Path(p['manosim_root'])
    public=list((upstream/'src').glob('*.py'))
    public += [upstream/'assets/mano_rhand_shape.npy',upstream/'config/myhand.yaml',upstream/'README.md',upstream/'LICENSE',upstream/'pyproject.toml']
    upstream_identity(upstream,public)
    required=[Path(__file__),HERE/'NATIVE_AUDIT.md',HERE/'native_controls.json',ROOT/'.gitignore',Path(p['adapter_source']),
        ROOT/'tests/test_manosim_pose.py',*evidence,*build_evidence,*sources,*public,*HERE.glob('test_*.py')]
    inputs=dict(build['bindings'])
    for file in required:inputs[str(Path(file).resolve())]=sha(file)
    p['bindings']=inputs;p['measurement_sources']=[str(Path(x).resolve()) for x in required]
    p['tracked_sources'],p['raw_generated_bindings']=source_partitions(inputs,generated['source_root'],p['measurement_sources'],build['tracked_sources'])
    write(path,p);return p

def validate(protocol,freeze_commit):
    p=read(protocol)
    if (p.get('schema_version')!=SCHEMA or p.get('budgets')!=BUDGET
            or p.get('numerical_comparisons')!=NUMERICAL or p.get('rotation_representation')!=REPRESENTATION or p.get('runtime')!=runtime()
            or p.get('indices')!=list(INDICES) or p.get('upstream_commit')!=UPSTREAM
            or p.get('production_promotion') is not False or p.get('new_collection_or_training_allowed') is not False):raise ValueError('Native declaration/runtime differs')
    verify(p['bindings'])
    build,generated,evidence=completed_build()
    if (p.get('generated_assets')!=generated or p.get('shared_shape_sha256')!=generated['expected_shape_sha256']
            or p.get('build_protocol')!=str((HERE/BUILD_PROTOCOL_NAME).resolve())
            or p.get('build_protocol_sha256')!=sha(HERE/BUILD_PROTOCOL_NAME)
            or p.get('builder_report')!=str((HERE/'build_run_v1/builder_report.json').resolve())
            or p.get('build_readback')!=str((HERE/'build_run_v1/independent_readback.json').resolve())):
        raise ValueError('Native declaration generated-build provenance differs')
    expected_tracked,expected_raw=source_partitions(p['bindings'],generated['source_root'],p['measurement_sources'],build['tracked_sources'])
    if p.get('raw_generated_bindings')!=expected_raw or p.get('tracked_sources')!=expected_tracked:
        raise ValueError('Raw asset hashes and tracked-source partition differ')
    sources=load('validate_generated_native_source_inventory',HERE/'audit_case.py').source_inputs(p)
    for path in sources:
        if p['bindings'].get(str(path.resolve()))!=sha(path):raise ValueError('Current generated/native source closure missing from declaration')
    upstream_identity(p['manosim_root'],p['measurement_sources'])
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('HEAD must stay at the exact native measurement freeze')
    selected={Path(protocol).resolve(),Path(__file__).resolve()}
    selected.update(Path(x) for x in p['tracked_sources'])
    relative=[str(x.relative_to(ROOT)) for x in sorted(selected)]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*relative],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*relative],check=True)
    return p,commit

def artifact_index(output,rows):
    output=Path(output).resolve();seen=set();checked=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'path','sha256','bytes','label'}:
            raise ValueError('Complete write-time artifact record required')
        literal=Path(row['path']);file=literal.resolve()
        if (not file.is_relative_to(output) or file in seen or literal.is_symlink()
                or any(x.is_symlink() for x in literal.parents if x!=output and x.is_relative_to(output))
                or isinstance(row['bytes'],bool) or not isinstance(row['bytes'],int) or row['bytes']<=0
                or not file.is_file() or row['bytes']!=file.stat().st_size or sha(file)!=row['sha256']):
            raise ValueError('Measurement artifact changed or escaped output')
        seen.add(file);checked.append(dict(row))
    if {x.resolve() for x in output.rglob('*.npz')}!={x for x in seen if x.suffix=='.npz'}:
        raise ValueError('Unindexed or missing native measurement arrays')
    return checked

def check_result(value,counts,case_counts):
    if value.get('kinematic_pose_transfer_status') not in ('PASS','FAIL'):
        raise ValueError('Explicit numerical pose-transfer verdict required')
    if value.get('surface_fidelity_status') not in ('FAIL','INCOMPLETE'):
        raise ValueError('This finite native diagnostic cannot issue surface PASS')
    for key in ('native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'):
        if value.get(key) is not False:raise ValueError('Native diagnostic cannot release production')
    def nested(item):
        if isinstance(item,dict):
            for key,entry in item.items():
                if key in ('native_fidelity_accepted','production_promotion','new_collection_or_training_allowed','accepted_mapping') and entry is not False:
                    raise ValueError('Nested production admission is forbidden')
                if key in ('task_outcome','physical_success','task_success'):raise ValueError('No physical task labels in a static audit')
                nested(entry)
        elif isinstance(item,list):
            for entry in item:nested(entry)
    nested(value)
    if (any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in counts.values())
            or not 0<counts['static_forwards']<=200 or any(counts[k] for k in counts if k!='static_forwards')):
        raise ValueError('Static operation budget violated')
    if (any(isinstance(case_counts[k],bool) or not isinstance(case_counts[k],int) for k in ('parametric_mano_forwards','full_field_variant_scans'))
            or case_counts['parametric_mano_forwards']!=2 or case_counts['full_field_variant_scans']!=2):
        raise ValueError('Fixed native parametric/variant scan budget violated')

def worker(protocol,freeze_commit,output,index):
    p,commit=validate(protocol,freeze_commit);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    counts=dict(static_forwards=0,physics_step_attempts=0,HUG_generator_attempts=0,official_retarget_attempts=0)
    def refuse(kind):
        def fail(*a,**k):counts[kind]+=1;raise RuntimeError('Native static qualification forbids '+kind)
        return fail
    def block():
        for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,refuse('physics_step_attempts'))
    block();original=mujoco.mj_forward
    def forward(*a,**k):
        if counts['static_forwards']>=200:raise RuntimeError('200 static forwards exhausted')
        counts['static_forwards']+=1;return original(*a,**k)
    mujoco.mj_forward=forward
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
    from grasp_failure_prediction.evaluation import case_runner,retargeting,runner
    case_runner._run_case_inference=refuse('HUG_generator_attempts')
    case_runner.run_case=refuse('HUG_generator_attempts')
    retargeting.ShadowHandRetargeter.retarget=refuse('official_retarget_attempts')
    for cls in (runner.AdroitShadowRunner,runner.ActuatedShadowRunner):
        cls.execute=refuse('physics_step_attempts');cls._advance=refuse('physics_step_attempts')
    started=time.perf_counter();case=None;value=None;failure=None;artifacts=[]
    try:
        module=load('frozen_native_audit_case',HERE/'audit_case.py');block()
        case=module.AuditCase(index,p,output,guard_reinstall=block);block();value=case.run()
        check_result(value,counts,case.counts)
        if len(case.checkpoint_artifacts)!=2:raise ValueError('Exactly two write-time variant field receipts required')
        artifacts=artifact_index(output,case.checkpoint_artifacts+case.auxiliary_artifacts)
        validate(protocol,freeze_commit)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
    report=dict(schema_version=SCHEMA,source_inventory_index=index,protocol_sha256=sha(protocol),freeze_commit=commit,
        status='COMPLETED_NATIVE_STATIC_DIAGNOSTIC' if failure is None else 'HARD_STOP',failure=failure,
        elapsed_s=time.perf_counter()-started,counts=counts,case_counts=None if case is None else case.counts,
        measurement=value,artifacts=artifacts,source_receipt=None if case is None else case.source_receipt,
        native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'result.json',report);return int(failure is not None)

def run(protocol,freeze_commit,output):
    p,commit=validate(protocol,freeze_commit);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(protocol_sha256=sha(protocol),freeze_commit=commit,budgets=BUDGET))
    rows=[]
    for index in INDICES:
        validate(protocol,freeze_commit);log=output/f'case_{index}.log';started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--protocol',str(Path(protocol).resolve()),
            '--freeze-commit',commit,'--output',str(output/str(index)),'--index',str(index)]
        with log.open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
        result=output/str(index)/'result.json'
        good=code==0 and result.exists() and read(result).get('status')=='COMPLETED_NATIVE_STATIC_DIAGNOSTIC'
        rows.append(dict(index=index,status='COMPLETED' if good else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR',
            returncode=code,elapsed_s=time.perf_counter()-started,log=str(log),log_sha256=sha(log),
            result=str(result) if result.exists() else None,result_sha256=sha(result) if result.exists() else None))
        if not good:break
    done={r['index'] for r in rows};rows += [dict(index=i,status='NOT_RUN') for i in INDICES if i not in done]
    validate(protocol,freeze_commit)
    report=dict(schema_version=SCHEMA,protocol_sha256=sha(protocol),freeze_commit=commit,cases=rows,
        status='COMPLETED_NATIVE_STATIC_PILOT' if all(r['status']=='COMPLETED' for r in rows) else 'HARD_STOP',
        native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'report.json',report);print(json.dumps(report),flush=True);return int(report['status']=='HARD_STOP')

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,default=HERE/'native_protocol.json')
    for name in ('run','worker'):
        q=sub.add_parser(name);q.add_argument('--protocol',type=Path,required=True);q.add_argument('--freeze-commit',required=True)
        q.add_argument('--output',type=Path,required=True)
        if name=='worker':q.add_argument('--index',type=int,choices=INDICES,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.protocol);return 0
    return run(a.protocol,a.freeze_commit,a.output) if a.command=='run' else worker(a.protocol,a.freeze_commit,a.output,a.index)

if __name__=='__main__':raise SystemExit(main())
