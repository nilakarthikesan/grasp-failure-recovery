"""Freeze saved-geometry construction ablations; never construct a native model."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.metadata
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=ROOT/'research_artifacts/manosim_native_transfer_2026-10-15_v2'
NATIVE_PROTOCOL_SHA256='c81cdd617ea98ad0a0edf1b37dddb9e5a78871b59cf2898b9d216fa847cc3207'
NATIVE_READBACK_SHA256='5cff858388ff1edb7ab34d4c17868907d8b701198c024a9bc2e70e12201ab13d'
NATIVE_BINDING_COUNT=1879
SCHEMA='manosim_saved_geometry_construction_ablation_v1'
INDICES=(20,28)
# Variant names and finite counts are finalized with ablation_case before freeze.
VARIANTS=('native_reference','posed_uncapped','neutral_lbs_uncapped','rigid_neutral_uncapped',
    'rigid_dropped_fixedlabels','rigid_dropped_bonelabels','rigid_hulls_fullprecision',
    'rigid_hulls_obj_decimal','serialized_obj_xml_saved_frames')
BUDGET=dict(cases=2,maximum_seconds_per_case=600,maximum_full_field_scans_per_case=9,
    maximum_triangle_point_tests_per_query_cell=18620,query_chunk_points=256,
    MuJoCo_models=0,MuJoCo_FK_calls=0,MuJoCo_forwards=0,physics_steps=0,
    MANO_model_calls=0,HUG_generator_calls=0,retarget_calls=0,predictor_fits=0,task_outcomes=0)
FORBIDDEN_KEYS=('MuJoCo_model_attempts','MuJoCo_data_attempts','MuJoCo_FK_attempts','MuJoCo_forward_attempts',
    'physics_step_attempts','MANO_import_attempts','HUG_generator_attempts','retarget_attempts','training_attempts')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def verify(bindings):
    for path,expected in bindings.items():
        if sha(path)!=expected:raise ValueError('Bound prior/source/runtime bytes changed: '+path)
def exact_inventory_file(path,inventory):
    path=Path(path).resolve();row=inventory.get(str(path))
    if not isinstance(row,dict) or row.get('sha256')!=sha(path) or row.get('bytes')!=path.stat().st_size:
        raise ValueError('Prior output lacks exact independent byte coverage: '+str(path))


def completed_native():
    """Read/hash frozen evidence only; never import its old worker/runtime."""
    protocol=NATIVE/'native_protocol_v2.json';readback=NATIVE/'run_v1/independent_readback_v2.json'
    if sha(protocol)!=NATIVE_PROTOCOL_SHA256 or sha(readback)!=NATIVE_READBACK_SHA256:
        raise ValueError('Exact completed native protocol and final independent readback required')
    p=read(protocol);a=read(readback);report_path=NATIVE/'run_v1/report.json';report=read(report_path)
    if (len(p.get('bindings',{}))!=NATIVE_BINDING_COUNT or p.get('indices')!=list(INDICES)
            or p.get('schema_version')!='hug_literal_shape_manosim_static_transfer_v2'
            or report.get('status')!='COMPLETED_NATIVE_STATIC_PILOT'
            or report.get('protocol_sha256')!=NATIVE_PROTOCOL_SHA256
            or a.get('schema_version')!='manosim_literal_shape_native_static_independent_saved_readback_v2'
            or a.get('status')!='VERIFIED_COMPLETED_NATIVE_STATIC_SAVED_ARRAYS'
            or a.get('active_protocol_sha256')!=NATIVE_PROTOCOL_SHA256
            or a.get('freeze_commit')!=report.get('freeze_commit')
            or a.get('expected_bindings')!=NATIVE_BINDING_COUNT or isinstance(a.get('expected_bindings'),bool)
            or a.get('all_bindings_verified_before') is not True or a.get('all_bindings_verified_after') is not True
            or a.get('actual_execution_status')!='COMPLETED_NATIVE_STATIC_PILOT' or a.get('errors')!=[]):
        raise ValueError('Both independently verified native development cases required')
    inventory=a.get('artifact_inventory',{})
    if not isinstance(inventory,dict):raise ValueError('Independent prior output inventory required')
    for path in inventory:exact_inventory_file(path,inventory)
    exact_inventory_file(report_path,inventory)
    if [row.get('index') for row in report.get('cases',[])]!=list(INDICES):raise ValueError('Native case order changed')
    checked={row['index']:row for row in a.get('cases',[])}
    evidence={protocol.resolve(),readback.resolve(),report_path.resolve(),*(Path(path).resolve() for path in inventory)}
    for row in report['cases']:
        result_path=Path(row['result']);result=read(result_path);independent=checked.get(row['index'],{})
        exact_inventory_file(result_path,inventory)
        if (row.get('status')!='COMPLETED' or row.get('returncode')!=0
                or sha(result_path)!=row['result_sha256'] or result.get('status')!='COMPLETED_NATIVE_STATIC_DIAGNOSTIC'
                or result.get('protocol_sha256')!=NATIVE_PROTOCOL_SHA256 or result.get('freeze_commit')!=report['freeze_commit']
                or independent.get('status')!='VERIFIED_COMPLETED_STATIC_DIAGNOSTIC'
                or independent.get('result_sha256')!=row['result_sha256']):
            raise ValueError('Native completed case/result/freeze identity changed')
        for artifact in result['artifacts']:
            path=Path(artifact['path']);exact_inventory_file(path,inventory)
            if sha(path)!=artifact['sha256'] or path.stat().st_size!=artifact['bytes']:
                raise ValueError('Prior creation-time native artifact changed')
        # A numerical pose pass never changes the separately failed surface gate.
        if independent.get('kinematic_gate')!=result['measurement']['kinematic_pose_transfer_status'] or independent.get('surface_gate')!=result['measurement']['surface_fidelity_status']:
            raise ValueError('Native interpretation differs from independent evidence')
    verify(p['bindings'])
    return p,sorted(evidence)


def bootstrap(p):
    """Load base NumPy first, then append only existing frozen dependency paths."""
    source=read(p['v3_protocol']) if 'v3_protocol' in p else read(p['native_v3_protocol'])
    sys.path.insert(0,str(ROOT/'src'))
    for directory in source['dependency_dirs']:
        if directory not in sys.path:sys.path.append(directory)

def runtime():
    value=dict(python=__import__('platform').python_version(),numpy=np.__version__,executable=sys.executable)
    for name in ('mujoco','scipy','numba'):value[name]=importlib.metadata.version(name)
    return value


def source_partitions(bindings,measurement_sources,inherited_tracked_sources,raw_root):
    raw=Path(raw_root).resolve()
    if not raw.is_relative_to(ROOT/'runs'):raise ValueError('Raw geometry must use the existing ignored runs root')
    selected={str(Path(path).resolve()) for path in inherited_tracked_sources}
    if any(not Path(path).is_relative_to(ROOT) or Path(path).is_relative_to(ROOT/'runs') for path in selected):
        raise ValueError('Inherited selected Git source escaped checkout or entered ignored runs')
    selected.update(str(Path(path).resolve()) for path in measurement_sources
        if Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(ROOT/'runs')
        and '/build_run_v1/raw/' not in str(Path(path))
        and (Path(path).suffix in ('.py','.md','.json') or Path(path).name=='.gitignore'))
    if any(path not in bindings for path in selected):raise ValueError('Selected Git source omitted from byte bindings')
    return sorted(selected)


def permitted_roots(p,output):
    tracked=Path(p['output_root']).resolve();raw=Path(p['raw_root']).resolve();output=Path(output).resolve()
    expected_tracked=(HERE/'run_v1').resolve();expected_raw=(ROOT/'runs/manosim_geometry_ablation_2026-10-15_v1/run_v1/raw').resolve()
    if tracked!=expected_tracked or raw!=expected_raw or output!=tracked:
        raise ValueError('Exactly the two prospectively declared output roots required')
    if tracked.is_relative_to(raw) or raw.is_relative_to(tracked):raise ValueError('Tracked/raw output roots must be disjoint')
    return tracked,raw


def case_limits(prior):
    old=read(read(prior['v3_protocol'])['v1_protocol'])
    result={}
    for index in INDICES:
        plan=next(row for row in old['cases'] if row['source_inventory_index']==index)
        if len(plan['grids'])!=1:raise ValueError('Fixed original full grid required')
        grid=plan['grids'][0];n=grid['query_count']
        if isinstance(n,bool) or not isinstance(n,int) or n<=0:raise ValueError('Finite original query count required')
        result[str(index)]=dict(query_count=n,maximum_triangle_point_tests=n*18620,
            required_relationships=grid['required_relationships'])
    return result


def prepare(path):
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite an ablation declaration')
    if not VARIANTS or not BUDGET:raise ValueError('Variant budget must be finalized before preparation')
    prior,evidence=completed_native();bootstrap(prior)
    p=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),
        runtime=runtime(),indices=list(INDICES),variants=list(VARIANTS),budgets=dict(BUDGET),
        native_protocol=str((NATIVE/'native_protocol_v2.json').resolve()),native_protocol_sha256=NATIVE_PROTOCOL_SHA256,
        native_readback=str((NATIVE/'run_v1/independent_readback_v2.json').resolve()),native_readback_sha256=NATIVE_READBACK_SHA256,
        native_v3_protocol=prior['v3_protocol'],native_binding_count=NATIVE_BINDING_COUNT,
        native_run=str((NATIVE/'run_v1').resolve()),builder_report=prior['builder_report'],
        mano_asset_path=prior['mano_asset_path'],case_limits=case_limits(prior),
        field_rules=str(Path(read(prior['v3_protocol'])['v1_protocol']).parent/'audit_full_field.py'),
        output_root=str((HERE/'run_v1').resolve()),raw_root=str((ROOT/'runs/manosim_geometry_ablation_2026-10-15_v1/run_v1/raw').resolve()),
        geometry_kernel=prior['geometry_kernel'],geometry_base=prior['geometry_base'],
        fixed_inputs='Existing saved native poses, literal shape, object/camera frames, original full query grids and unchanged same-part whole-cell predicates',
        geometry_outputs_are_new=True,prior_data_relabelled=False,accepted_mapping=False,
        native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    module=load('ablation_source_inventory',HERE/'ablation_case.py')
    sources=module.source_inputs(p)
    required=[Path(__file__),HERE/'test_run_ablation.py',HERE/'ablation_case.py',HERE/'README.md',HERE/'controls_v2.json',
        ROOT/'.gitignore',*HERE.glob('test_*.py'),*evidence,*sources]
    bindings=dict(prior['bindings'])
    for file in required:bindings[str(Path(file).resolve())]=sha(file)
    p['bindings']=bindings;p['measurement_sources']=[str(Path(file).resolve()) for file in required]
    p['tracked_sources']=source_partitions(bindings,p['measurement_sources'],prior['tracked_sources'],p['raw_root'])
    write(path,p);return p


def validate(protocol,freeze_commit,output=None):
    p=read(protocol)
    if (p.get('schema_version')!=SCHEMA or p.get('indices')!=list(INDICES) or p.get('variants')!=list(VARIANTS)
            or p.get('budgets')!=BUDGET or p.get('native_protocol_sha256')!=NATIVE_PROTOCOL_SHA256
            or p.get('native_readback_sha256')!=NATIVE_READBACK_SHA256 or p.get('native_binding_count')!=NATIVE_BINDING_COUNT):
        raise ValueError('Ablation declaration identity/variants/quota changed')
    verify(p['bindings']);prior,evidence=completed_native();bootstrap(prior)
    if runtime()!=p['runtime']:raise ValueError('Ablation runtime metadata changed')
    if (p.get('case_limits')!=case_limits(prior) or p.get('native_run')!=str((NATIVE/'run_v1').resolve())
            or p.get('builder_report')!=prior['builder_report'] or p.get('mano_asset_path')!=prior['mano_asset_path']
            or p.get('field_rules')!=str(Path(read(prior['v3_protocol'])['v1_protocol']).parent/'audit_full_field.py')):
        raise ValueError('Saved input/grid/field construction changed')
    for key in ('prior_data_relabelled','accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'):
        if p.get(key) is not False:raise ValueError('Ablation cannot admit production data or rewrite old labels')
    selected=source_partitions(p['bindings'],p['measurement_sources'],prior['tracked_sources'],p['raw_root'])
    if selected!=p.get('tracked_sources'):raise ValueError('Selected source/runtime partition changed')
    for path in evidence:
        if p['bindings'].get(str(path))!=sha(path):raise ValueError('Prior byte evidence omitted from new declaration')
    module=load('ablation_validate_inventory',HERE/'ablation_case.py')
    for path in module.source_inputs(p):
        if p['bindings'].get(str(Path(path).resolve()))!=sha(path):raise ValueError('Current ablation source/input closure omitted')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',freeze_commit+'^{commit}'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=commit:
        raise ValueError('HEAD must remain at the exact ablation freeze')
    for name in sorted(set(selected)|{str(Path(protocol).resolve())}):
        path=Path(name).resolve();relative=str(path.relative_to(ROOT))
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])
        if hashlib.sha256(blob).hexdigest()!=sha(path):raise ValueError('Selected source/protocol Git blob differs')
    permitted_roots(p,p['output_root'] if output is None else output)
    return p,commit


def artifact_index(roots,rows,*,complete):
    roots=tuple(Path(root).resolve() for root in roots);seen=set();checked=[]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'path','sha256','bytes','label'}:raise ValueError('Creation-time artifact schema required')
        literal=Path(row['path']);path=literal.resolve()
        if (not literal.is_absolute() or path in seen or not path.is_file() or literal.is_symlink()
                or not any(path.is_relative_to(root) for root in roots)
                or any(parent.is_symlink() for parent in literal.parents if any(parent.is_relative_to(root) for root in roots))
                or isinstance(row['bytes'],bool) or not isinstance(row['bytes'],int) or row['bytes']<=0
                or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):
            raise ValueError('Output artifact changed, duplicated, escaped or aliased')
        seen.add(path);checked.append(dict(row))
    if complete:
        actual={path.resolve() for root in roots for path in root.rglob('*.npz')}
        if actual!={path for path in seen if path.suffix=='.npz'}:raise ValueError('Unindexed or missing ablation array artifact')
    return checked


def partial_artifact_receipt(roots,case):
    rows=[] if case is None else [dict(row) for row in case.artifact_rows]
    verified=[];errors=[];seen=set()
    for row in rows:
        try:
            path=Path(row['path']).resolve()
            if path in seen:raise ValueError('Duplicated original creation-time artifact path')
            seen.add(path);verified.extend(artifact_index(roots,[row],complete=False))
        except Exception as error:errors.append(dict(path=row.get('path'),type=type(error).__name__,message=str(error)))
    indexed={Path(row['path']).resolve() for row in verified}
    unindexed=sorted(str(path.resolve()) for root in roots for path in Path(root).rglob('*.npz') if path.resolve() not in indexed)
    return dict(recorded_rows=rows,verified_rows=verified,verification_errors=errors,
        unindexed_array_paths=unindexed,complete_array_inventory=False)



class QueryBudget:
    """Called before each scan/chunk; no exceeded operation is executed."""
    def __init__(self,limit):self.limit=limit;self.stages=[];self.triangle_point_tests=0
    def __call__(self,kind,value):
        if kind=='full_field_scan':
            if len(self.stages)>=len(VARIANTS) or value!=VARIANTS[len(self.stages)]:
                raise RuntimeError('Fixed ordered nine-scan quota exhausted or changed before scan')
            self.stages.append(value)
        elif kind=='triangle_point_tests':
            if not self.stages or isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError('Positive integer triangle-point query count after declared scan required')
            if self.triangle_point_tests+value>self.limit:
                raise RuntimeError('Triangle-point test quota exhausted before chunk')
            self.triangle_point_tests+=value
        else:raise ValueError('Unknown budget operation')


class ForbiddenImports(importlib.abc.MetaPathFinder):
    def __init__(self,counts):self.counts=counts
    def find_spec(self,fullname,path=None,target=None):
        groups={'MANO_import_attempts':('torch','manotorch','manopth','smplx'),
            'HUG_generator_attempts':('hug',),'training_attempts':('sklearn','grasp_failure_prediction.prediction')}
        for key,prefixes in groups.items():
            if any(fullname==prefix or fullname.startswith(prefix+'.') for prefix in prefixes):
                self.counts[key]+=1;raise RuntimeError('Saved-geometry ablation forbids '+fullname)
        return None


def install_guards(counts):
    import mujoco
    def refuse(kind):
        def fail(*a,**k):counts[kind]+=1;raise RuntimeError('Saved-geometry ablation forbids '+kind)
        return fail
    class NoModel:
        from_xml_path=staticmethod(refuse('MuJoCo_model_attempts'))
        from_xml_string=staticmethod(refuse('MuJoCo_model_attempts'))
        from_binary_path=staticmethod(refuse('MuJoCo_model_attempts'))
        def __new__(cls,*a,**k):return refuse('MuJoCo_model_attempts')()
    mujoco.MjModel=NoModel;mujoco.MjData=refuse('MuJoCo_data_attempts')
    for api in ('mj_forward','mj_forwardSkip','mj_fwdPosition','mj_fwdVelocity','mj_fwdAcceleration'):
        if hasattr(mujoco,api):setattr(mujoco,api,refuse('MuJoCo_forward_attempts'))
    for api in ('mj_kinematics','mj_comPos','mj_camlight','mj_tendon','mj_transmission'):
        if hasattr(mujoco,api):setattr(mujoco,api,refuse('MuJoCo_FK_attempts'))
    for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,refuse('physics_step_attempts'))
    module=sys.modules.get('grasp_failure_prediction.evaluation.retargeting')
    if module is not None:
        for name in ('ShadowHandRetargeter','DexRetargeter'):
            cls=getattr(module,name,None)
            if cls is not None:
                for method in ('__init__','retarget'):
                    if hasattr(cls,method):setattr(cls,method,refuse('retarget_attempts'))


def check_result(value,counts,case_counts,budget):
    if not isinstance(value,dict) or value.get('schema_version')!='manosim_saved_geometry_construction_ablation_case_v1':
        raise ValueError('Explicit saved-geometry ablation result schema required')
    if [row.get('stage_id') for row in value.get('stages',[])]!=list(VARIANTS):raise ValueError('Exactly nine fixed stages required')
    control=value.get('reference_control',{})
    error=control.get('max_distance_error_m')
    if (control.get('passed') is not True or control.get('exact_masks') is not True or isinstance(error,bool)
            or not isinstance(error,(int,float)) or not np.isfinite(error) or error<0 or error>1e-12):
        raise ValueError('Native reference reproduction is required before causal stages')
    def nested(item):
        if isinstance(item,dict):
            for key,entry in item.items():
                if key in ('accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed') and entry is not False:
                    raise ValueError('Ablation cannot admit mapping, physics or production')
                if key in ('task_outcome','physical_success','task_success'):raise ValueError('Static ablation cannot create task labels')
                if key=='physical_outcome' and entry is not None:raise ValueError('Static ablation physical outcome must be null')
                if key in ('surface_fidelity_status','full_strict_mapping_status') and entry not in ('FAIL','INCOMPLETE'):
                    raise ValueError('Finite fields cannot issue surface fidelity PASS')
                nested(entry)
        elif isinstance(item,list):
            for entry in item:nested(entry)
    nested(value)
    if value.get('counts')!=case_counts:raise ValueError('Measurement/case operation counters differ')
    for key in ('accepted_mapping','native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'):
        if value.get(key) is not False:raise ValueError('Explicit no-admission flags required')
    if set(counts)!=set(FORBIDDEN_KEYS) or any(type(v) is not int or v!=0 for v in counts.values()):
        raise ValueError('Forbidden operation attempted')
    expected={'full_field_scans','triangle_point_tests','model_calls','FK_engine_calls','physics_steps',
        'HUG_generator_calls','retarget_calls','predictor_fits'}
    if set(case_counts)!=expected or any(type(v) is not int or v<0 for v in case_counts.values()):raise ValueError('Complete finite case counters required')
    if (case_counts['full_field_scans']!=9 or budget.stages!=list(VARIANTS)
            or not 0<case_counts['triangle_point_tests']==budget.triangle_point_tests<=budget.limit
            or any(value for key,value in case_counts.items() if key not in ('full_field_scans','triangle_point_tests'))):
        raise ValueError('Case counters disagree with pre-work guards')


def worker(protocol,freeze_commit,output,index):
    p,commit=validate(protocol,freeze_commit,output);tracked,raw=permitted_roots(p,output)
    if index not in INDICES:raise ValueError('Only the two fixed development natives')
    directory=tracked/str(index);raw_case=raw/str(index)
    directory.mkdir(parents=True,exist_ok=False);raw_case.mkdir(parents=True,exist_ok=False)
    counts={key:0 for key in FORBIDDEN_KEYS};install_guards(counts)
    finder=ForbiddenImports(counts);sys.meta_path.insert(0,finder)
    budget=QueryBudget(p['case_limits'][str(index)]['maximum_triangle_point_tests'])
    case=None;measurement=None;failure=None;artifacts=[];partial=None;started=time.perf_counter()
    try:
        module=load('frozen_saved_geometry_ablation',HERE/'ablation_case.py');install_guards(counts)
        case=module.AblationCase(index,p,raw_case,guard_reinstall=lambda:install_guards(counts),budget_guard=budget)
        measurement=case.run()
        if measurement.get('index')!=index:raise ValueError('Measurement belongs to another fixed native')
        check_result(measurement,counts,case.counts,budget)
        artifacts=artifact_index((directory,raw_case),case.artifact_rows,complete=True)
        validate(protocol,freeze_commit,output)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(directory/'exception.txt').write_text(traceback.format_exc())
        partial=partial_artifact_receipt((directory,raw_case),case)
    if finder in sys.meta_path:sys.meta_path.remove(finder)
    result=dict(schema_version=SCHEMA,source_inventory_index=index,protocol_sha256=sha(protocol),freeze_commit=commit,
        status='COMPLETED_SAVED_GEOMETRY_DIAGNOSTIC' if failure is None else 'HARD_STOP',failure=failure,
        elapsed_s=time.perf_counter()-started,counts=counts,case_counts=None if case is None else dict(case.counts),
        guarded_operations=dict(stages=list(budget.stages),triangle_point_tests=budget.triangle_point_tests,limit=budget.limit),
        measurement=measurement,artifacts=artifacts,partial_artifacts=partial,
        source_receipt=None if case is None else getattr(case,'source_receipt',None),
        accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    # Publish the tracked scalar result last; preserve an exact ignored raw copy.
    write(raw_case/'result.json',result)
    with (directory/'result.json').open('xb') as f:f.write((raw_case/'result.json').read_bytes())
    return int(failure is not None)


def run(protocol,freeze_commit,output):
    p,commit=validate(protocol,freeze_commit,output);tracked,raw=permitted_roots(p,output)
    if tracked.exists() or raw.exists():raise FileExistsError('Never overwrite a tracked or ignored ablation run')
    tracked.mkdir(parents=True,exist_ok=False);raw.mkdir(parents=True,exist_ok=False)
    write(tracked/'started.json',dict(protocol_sha256=sha(protocol),freeze_commit=commit,budgets=dict(BUDGET),
        indices=list(INDICES),variants=list(VARIANTS),raw_root=str(raw)))
    rows=[]
    for index in INDICES:
        validate(protocol,freeze_commit,output);log=tracked/f'case_{index}.log';started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--protocol',str(Path(protocol).resolve()),
            '--freeze-commit',commit,'--output',str(tracked),'--index',str(index)]
        with log.open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=BUDGET['maximum_seconds_per_case'])
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();code=None
        result=tracked/str(index)/'result.json';raw_result=raw/str(index)/'result.json'
        good=code==0 and result.is_file() and read(result).get('status')=='COMPLETED_SAVED_GEOMETRY_DIAGNOSTIC'
        if good and (not raw_result.is_file() or raw_result.read_bytes()!=result.read_bytes()):good=False
        rows.append(dict(index=index,status='COMPLETED' if good else 'TIME_BUDGET_EXCEEDED' if code is None else 'ERROR',
            returncode=code,elapsed_s=time.perf_counter()-started,log=str(log),log_sha256=sha(log),
            result=str(result) if result.is_file() else None,result_sha256=sha(result) if result.is_file() else None,
            raw_result=str(raw_result) if raw_result.is_file() else None,raw_result_sha256=sha(raw_result) if raw_result.is_file() else None))
        if not good:break
    done={row['index'] for row in rows};rows.extend(dict(index=index,status='NOT_RUN') for index in INDICES if index not in done)
    validate(protocol,freeze_commit,output)
    completed_artifacts=[]
    for row in rows:
        if row['status']=='COMPLETED':
            result=Path(row['result']);raw_result=Path(row['raw_result'])
            if sha(result)!=row['result_sha256'] or sha(raw_result)!=row['raw_result_sha256'] or result.read_bytes()!=raw_result.read_bytes():
                raise ValueError('Earlier completed scalar result changed before publication')
            completed_artifacts.extend(read(result)['artifacts'])
    artifact_index((tracked,raw),completed_artifacts,complete=all(row['status']=='COMPLETED' for row in rows))
    report=dict(schema_version=SCHEMA,protocol_sha256=sha(protocol),freeze_commit=commit,cases=rows,
        status='COMPLETED_SAVED_GEOMETRY_PILOT' if all(row['status']=='COMPLETED' for row in rows) else 'HARD_STOP',
        raw_root=str(raw),accepted_mapping=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    write(tracked/'report.json',report);print(json.dumps(report),flush=True);return int(report['status']=='HARD_STOP')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    for name in ('run','worker'):
        q=sub.add_parser(name);q.add_argument('--protocol',type=Path,required=True);q.add_argument('--freeze-commit',required=True);q.add_argument('--output',type=Path,required=True)
        if name=='worker':q.add_argument('--index',type=int,choices=INDICES,required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.protocol);return 0
    return run(a.protocol,a.freeze_commit,a.output) if a.command=='run' else worker(a.protocol,a.freeze_commit,a.output,a.index)

if __name__=='__main__':raise SystemExit(main())
