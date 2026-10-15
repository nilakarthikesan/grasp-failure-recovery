"""Freeze one literal-shape upstream build; no native posing or task rollout."""
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
PRIOR=ROOT/'research_artifacts/manosim_native_transfer_2026-10-15_v1'
SCHEMA='manosim_literal_shape_build_transaction_v1'
BUDGET=dict(maximum_seconds=600,maximum_neutral_mano_forwards=1,MuJoCo_models=0,
    MuJoCo_forwards=0,physics_steps=0,HUG_generator_calls=0,retarget_calls=0,predictor_fits=0)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def read(path):return json.loads(Path(path).read_text())
def helper():return load('literal_shape_builder',HERE/'shape_build.py')
def prior():return load('native_v1_transaction',PRIOR/'run_native_transfer.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def runtime():
    value=prior().runtime()
    import importlib.metadata
    return dict(value,trimesh=importlib.metadata.version('trimesh'))

def builder_dependency_bindings():
    """Installed supporting source/binaries actually available to upstream imports."""
    import importlib.metadata
    bindings={};versions={}
    for name in ('tyro','rich','PyYAML','scipy'):
        distribution=importlib.metadata.distribution(name);versions[name]=distribution.version
        for relative in distribution.files or ():
            if str(relative).endswith(('.py','.so','.dylib','/METADATA','/WHEEL')):
                path=Path(distribution.locate_file(relative)).resolve();bindings[str(path)]=sha(path)
    return versions,bindings

def prepare(path):
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite a build declaration')
    util=prior();v3,evidence=util.completed_v3();util.verify(v3['bindings']);r=runtime()
    old=read(PRIOR/'protocol.json');v1=read(v3['v1_protocol']);inventory=read(v1['inventory'])
    inputs=[dict(index=i,**inventory['modern_cases'][i]['native']) for i in (20,28)]
    # The native record may also contain descriptive metadata; bind only literal input identity.
    inputs=[{k:row[k] for k in ('index','path','sha256')} for row in inputs]
    plan=helper().build_plan(inputs,old['manosim_root'],old['mano_asset_path'])
    dependencies=load('prior_native_inventory',PRIOR/'audit_case.py').source_inputs(old)
    required=[Path(__file__),HERE/'shape_build.py',HERE/'test_shape_build.py',HERE/'test_run_shape_build.py',
        HERE/'README.md',HERE/'build_controls_v2.json',ROOT/'.gitignore',PRIOR/'run_native_transfer.py',
        PRIOR/'protocol.json',PRIOR/'run_v1/report.json',PRIOR/'run_v1/20/result.json',
        PRIOR/'run_v1/independent_readback.json',*evidence,*dependencies]
    versions,support=builder_dependency_bindings()
    bindings=dict(v3['bindings']);bindings.update(plan['bindings']);bindings.update(support)
    for file in required:bindings[str(Path(file).resolve())]=helper().sha(file)
    upstream_files=[p for p in bindings if Path(p).is_relative_to(plan['upstream_root'])]
    util.upstream_identity(plan['upstream_root'],upstream_files)
    # Historical inputs include intentionally ignored third-party geometry and
    # preserved work from other providers. Their bytes stay bound, while the
    # actual transaction program and declaration must be committed and clean.
    tracked_sources=sorted({str(Path(x).resolve()) for x in required
        if Path(x).resolve().is_relative_to(ROOT)})
    p=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),runtime=r,budgets=BUDGET,
        plan=plan,bindings=bindings,tracked_sources=tracked_sources,builder_dependency_versions=versions,
        upstream_files=upstream_files,scientific_acceptance_sha256=v3['scientific_acceptance_sha256'],
        native_pose_execution='NOT_RUN; requires a separate generated-asset-bound audit freeze',
        source_selection='Pinned upstream right-only _build_side; unchanged myhand build configuration rules',
        production_promotion=False,new_collection_or_training_allowed=False)
    write(path,p);return p

def validate(protocol,freeze_commit):
    p=read(protocol);util=prior()
    if (p.get('schema_version')!=SCHEMA or p.get('budgets')!=BUDGET or p.get('runtime')!=runtime()
            or p.get('production_promotion') is not False or p.get('new_collection_or_training_allowed') is not False):
        raise ValueError('Build declaration/runtime differs')
    util.verify(p['bindings']);helper().verify_plan(p['plan'])
    util.upstream_identity(p['plan']['upstream_root'],p['upstream_files'])
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('HEAD must remain at the exact build freeze')
    selected=sorted(set(p['tracked_sources'])|{str(Path(protocol).resolve())})
    if any(not Path(x).is_relative_to(ROOT) for x in selected):raise ValueError('Tracked source escaped worktree')
    relative=[str(Path(x).relative_to(ROOT)) for x in selected]
    subprocess.run(['git','-C',str(ROOT),'ls-files','--error-unmatch','--',*relative],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','-C',str(ROOT),'diff','--quiet','HEAD','--',*relative],check=True)
    return p,commit

def install_guards(counts):
    def refuse(kind):
        def fail(*a,**k):counts[kind]+=1;raise RuntimeError('Shape build forbids '+kind)
        return fail
    class NoModel:
        from_xml_path=staticmethod(refuse('MuJoCo_model_attempts'))
        from_xml_string=staticmethod(refuse('MuJoCo_model_attempts'))
        from_binary_path=staticmethod(refuse('MuJoCo_model_attempts'))
        def __new__(cls,*a,**k):return refuse('MuJoCo_model_attempts')()
    mujoco.MjModel=NoModel
    mujoco.mj_forward=refuse('MuJoCo_forward_attempts')
    for key in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,key,refuse('physics_step_attempts'))

def check_completion(report,counts,protocol_sha256,plan_digest):
    if (report.get('status')!='COMPLETE' or report.get('failure') is not None
            or report.get('build_protocol_sha256')!=protocol_sha256 or report.get('plan_sha256')!=plan_digest):
        raise ValueError('Complete producer report with exact declaration required')
    for k in ('production_promotion','new_collection_or_training_allowed','native_fidelity_accepted','licensed_asset_copied'):
        if report.get(k) is not False:raise ValueError('Shape construction cannot admit task data')
    if any(isinstance(v,bool) or not isinstance(v,int) or v!=0 for v in counts.values()):
        raise ValueError('Forbidden operation attempted')
    actual=report['counts']
    if (actual.get('neutral_mano_forwards')!=1 or actual.get('neutral_mano_forward_attempts')!=1
            or actual.get('geometry_builder_returns')!=1):
        raise ValueError('Exactly one neutral producer forward required')
    if any(actual.get(k)!=0 for k in ('MuJoCo_forwards','physics_steps','HUG_generator_calls','retarget_calls')):
        raise ValueError('Producer crossed build-only scope')

def worker(protocol,freeze_commit,output):
    p,commit=validate(protocol,freeze_commit);output=Path(output)
    counts=dict(MuJoCo_model_attempts=0,MuJoCo_forward_attempts=0,physics_step_attempts=0,
        HUG_generator_attempts=0,retarget_attempts=0)
    install_guards(counts)
    sys.path.insert(0,str(ROOT/'src'))
    from grasp_failure_prediction.evaluation import case_runner,retargeting
    def blocked(key):
        def fail(*a,**k):counts[key]+=1;raise RuntimeError('Shape build forbids '+key)
        return fail
    case_runner._run_case_inference=blocked('HUG_generator_attempts');case_runner.run_case=blocked('HUG_generator_attempts')
    retargeting.ShadowHandRetargeter.retarget=blocked('retarget_attempts')
    value=None;failure=None;started=time.perf_counter()
    try:
        value=helper().build_exact_shape(p['plan'],output/'raw',build_protocol_sha256=sha(protocol),
            guard_reinstall=lambda:install_guards(counts))
        raw=output/'raw/build_report.json';copy=output/'builder_report.json'
        with copy.open('xb') as f:f.write(raw.read_bytes())
        check_completion(value,counts,sha(protocol),helper().digest(p['plan']))
        helper().generated_asset_selection(copy);validate(protocol,freeze_commit)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
    report=dict(schema_version=SCHEMA,status='COMPLETE' if failure is None else 'HARD_STOP',failure=failure,
        protocol_sha256=sha(protocol),freeze_commit=commit,elapsed_s=time.perf_counter()-started,
        forbidden_operation_attempts=counts,producer_counts=None if value is None else value['counts'],
        builder_report_sha256=sha(output/'builder_report.json') if (output/'builder_report.json').is_file() else None,
        production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'worker_result.json',report);return int(failure is not None)

def run(protocol,freeze_commit,output):
    p,commit=validate(protocol,freeze_commit);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',dict(protocol_sha256=sha(protocol),freeze_commit=commit,budgets=BUDGET))
    started=time.perf_counter();log=output/'build.log'
    command=[sys.executable,str(Path(__file__).resolve()),'worker','--protocol',str(Path(protocol).resolve()),
        '--freeze-commit',commit,'--output',str(output.resolve())]
    with log.open('x') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=BUDGET['maximum_seconds'])
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
    result=output/'worker_result.json';good=code==0 and result.is_file() and read(result).get('status')=='COMPLETE'
    validate(protocol,freeze_commit)
    report=dict(schema_version=SCHEMA,status='COMPLETE' if good else 'HARD_STOP',returncode=code,
        elapsed_s=time.perf_counter()-started,protocol_sha256=sha(protocol),freeze_commit=commit,
        worker_result_sha256=sha(result) if result.is_file() else None,log_sha256=sha(log),
        counts_known=result.is_file(),production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'report.json',report);print(json.dumps(report),flush=True);return int(not good)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','worker'):
        q=sub.add_parser(name);q.add_argument('--protocol',required=True,type=Path)
        if name!='prepare':q.add_argument('--freeze-commit',required=True);q.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.protocol);return 0
    return run(a.protocol,a.freeze_commit,a.output) if a.command=='run' else worker(a.protocol,a.freeze_commit,a.output)

if __name__=='__main__':raise SystemExit(main())
