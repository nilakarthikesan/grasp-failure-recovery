"""Prepare-only by default; all real saved-geometry queries require a committed once-run protocol."""
from __future__ import annotations
import argparse
import hashlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'research_artifacts/manosim_full_surface_reference_mj340_2026-10-15_v2'
V3=ROOT/'research_artifacts/hug_full_field_mapping_repair_2026-10-15_v3'
OLD=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
PURE=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py'
SCHEMA='manosim_saved_full_grid_orientation_qualification_v2'
PRIOR_PROTOCOL_SHA='00b3ee6d13c3f2c9ed88fe1f07977905ea0f09de3cdea0ccb8327ff9f7246a76'
PRIOR_REPORT_SHA='6c4f180aa2579c16a14f07a29bd046466c0a4cbdfebdea5042ee667424839d06'
PRIOR_INDEPENDENT_SHA='50661b96420355a59f01a2893c0ec4ebb288d1862a37f6eb0d22e4dec4b8f1ca'
PURE_SHA='b235f8b88b5ecb0fc2bc842c3fdb4a17d4c2eb526f321a449169dcbc3cc27bdd'
RULE_SHA='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4'
FAILED=ROOT/'research_artifacts/manosim_orientation_qualification_2026-10-15_v1'
FAILED_PROTOCOL_SHA='fd72601922c754c3728b2435aada0ccdd372115263d24ca9c521fe7be602e27a'
FAILED_READBACK_SHA='dc0b6f7beab5940f3691adaa4ddd30903caefcbc41e2a02a2725abf3b7fd5b2f'
INDICES=(20,28)
STAGES=('native','compiled')
LIMITS=dict(cases=2,seconds_per_case=600,chunk_points=128,hand_faces=1552,hand_surfaces=2,
    hand_distance_tests_per_query=3104,hand_winding_tests_per_query=3104,
    object_distance_tests_per_query_per_object_face=2,object_winding_tests_per_query_per_object_face=2,
    numerical_distance_replay_limit_m=1e-12,models=0,data=0,forwards=0,physics_steps=0,
    MANO_calls=0,HUG_calls=0,retarget_calls=0,fits=0,task_outcomes=0)
SOURCE_NAMES=('run_orientation.py','orientation_core.py','test_orientation.py','README.md','controls.json')
BLOCKED=('mujoco','torch','manotorch','roma','grasp_failure_prediction','trimesh')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def invalid(x):raise ValueError('Nonfinite JSON constant '+x)
    value=json.loads(Path(path).read_text(),parse_constant=invalid)
    def finite(v):
        if isinstance(v,float) and not np.isfinite(v):raise ValueError('Nonfinite JSON number')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        if isinstance(v,list):
            for x in v:finite(x)
    finite(value);return value

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def binding(path,label=None):
    p=Path(path).resolve();r=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    if label:r['label']=label
    return r

def verify(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Bound input changed: '+path)

def exact(path,inventory):
    p=Path(path).resolve();row=inventory[str(p)]
    if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Inventory changed: '+str(p))

def load_core():
    spec=importlib.util.spec_from_file_location('saved_orientation_core',HERE/'orientation_core.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def package_files():
    paths={Path(sys.executable).resolve()}
    for name in ('numpy','numba','llvmlite'):
        dist=importlib.metadata.distribution(name)
        for item in dist.files or ():
            p=Path(dist.locate_file(item)).resolve()
            if p.is_file() and p.suffix not in ('.pyc','.pyo'):paths.add(p)
    return paths

def runtime():return dict(python=sys.version,numpy=np.__version__,numba=importlib.metadata.version('numba'),llvmlite=importlib.metadata.version('llvmlite'),executable=str(Path(sys.executable).resolve()))

def inherited():
    pp,rp,ip=PRIOR/'protocol.json',PRIOR/'run_v1/report.json',PRIOR/'run_v1/independent_readback.json'
    if (sha(pp),sha(rp),sha(ip))!=(PRIOR_PROTOCOL_SHA,PRIOR_REPORT_SHA,PRIOR_INDEPENDENT_SHA):raise ValueError('Exact independently completed native surface pilot required')
    p,r,i=read(pp),read(rp),read(ip)
    if (r['status']!='COMPLETED_LOCKED_NATIVE_SURFACE_PILOT' or i['status']!='VERIFIED_COMPLETED_LOCKED_NATIVE_SURFACE_FIELDS_FULL_GATE_INCOMPLETE'
        or r['protocol_sha256']!=sha(pp) or i['active_protocol_sha256']!=sha(pp) or i['freeze_commit']!=r['freeze_commit']
        or i['expected_bindings']!=3041 or len(p['bindings'])!=3041 or i['errors'] or not i['all_bindings_verified_before'] or not i['all_bindings_verified_after']
        or i['verification_model_FK_distance_field_physics_calls']!=0 or [x['index'] for x in r['cases']]!=list(INDICES)):
        raise ValueError('Prior completed/independent closure contract changed')
    verify(p['bindings']);inventory=i['artifact_inventory']
    for name in inventory:exact(name,inventory)
    paths={pp,rp,ip,*map(Path,inventory)}
    cases={}
    for row in r['cases']:
        if row['status']!='COMPLETED' or row['returncode']!=0:raise ValueError('Both prior cases completed required')
        result=read(row['result']);exact(row['result'],inventory);exact(row['raw_result'],inventory)
        if sha(row['result'])!=row['result_sha256'] or Path(row['result']).read_bytes()!=Path(row['raw_result']).read_bytes():raise ValueError('Prior scalar copies changed')
        artifacts={Path(x['path']).name:x for x in result['artifacts']}
        needed=('native_reference_geometry.npz','native_reference_field.npz','compiled_locked_native_surface_geometry.npz','compiled_locked_native_surface_field.npz')
        cases[str(row['index'])]={k:artifacts[k] for k in needed}
        for k in needed:exact(artifacts[k]['path'],inventory)
    return p,paths,cases

def failed_prior():
    pp=FAILED/'protocol_v2.json';ip=FAILED/'run_v1/independent_readback.json';rp=FAILED/'run_v1/report.json'
    if sha(pp)!=FAILED_PROTOCOL_SHA or sha(ip)!=FAILED_READBACK_SHA:raise ValueError('Exact independently verified failedv1 required')
    p,i,r=read(pp),read(ip),read(rp)
    if (i['status']!='VERIFIED_HARD_STOP_CACHE_WIDTH_BEFORE_ALL_QUERY_WORK' or i['expected_bindings']!=3946
        or i['active_protocol_sha256']!=FAILED_PROTOCOL_SHA or i['report_sha256']!=sha(rp)
        or r['status']!='HARD_STOP' or i['freeze_commit']!=r['freeze_commit'] or not i['all_bindings_verified_before'] or not i['all_bindings_verified_after']
        or i['verification_model_FK_distance_orientation_physics_calls']!=0):raise ValueError('Prior failed-run lineage changed')
    prior_result=read(r['cases'][0]['result']['path'])
    if ([(x['index'],x['status']) for x in r['cases']]!=[(20,'HARD_STOPPED_PARTIAL'),(28,'NOT_RUN_AFTER_HARD_STOP')]
        or prior_result['artifacts'] or prior_result['unindexed_partial_files']
        or any(prior_result['query_work_counts'].values()) or any(prior_result['forbidden_operations'].values())
        or prior_result['failure']['message']!='Original query/radius/label/face alignment changed'):
        raise ValueError('Preserved failedv1 must stop before all query work')
    verify(p['bindings'])
    for name in i['artifact_inventory']:exact(name,i['artifact_inventory'])
    bindings=dict(p['bindings']);bindings.update({name:row['sha256'] for name,row in i['artifact_inventory'].items()})
    bindings[str(ip.resolve())]=sha(ip)
    return bindings,set(p['tracked_sources'])|{str(pp.resolve()),str(ip.resolve()),str(rp.resolve())}


def query_paths():
    base=Path('/private/tmp/hug-other-surfaces-2026-10-14/visual_reference_surfaces')
    return {20:base/'hug_milk_box_scene_001_seed_0_density2mm_queries.npz',28:base/'hug_mug_scene_003_seed_0_density2mm_queries.npz'}

def prepare(protocol):
    protocol=Path(protocol).resolve()
    if protocol.exists():raise FileExistsError(protocol)
    prior,paths,cases=inherited();failed_bindings,failed_tracked=failed_prior();paths.update(map(lambda n:HERE/n,SOURCE_NAMES));paths.update(package_files())
    archived={x for x in (HERE/'unused_prepare_v1').rglob('*') if x.is_file()};paths.update(archived)
    rule=OLD/'acceptance/acceptance_rule.json'
    if sha(PURE)!=PURE_SHA or sha(rule)!=RULE_SHA:raise ValueError('Original immutable numerical source/acceptance rule changed')
    paths.update((PURE,rule,OLD/'orientation/protocol.json',OLD/'orientation/check_orientation.py'))
    # The object visual operand provenance is independently checked before reusing its triangles.
    v3ip=V3/'run_v1/independent_readback.json';vi=read(v3ip);paths.add(v3ip)
    if vi.get('input_source_bindings_verified_before_and_after') != 379 or vi.get('no_geometry_or_FK_replay') is not True:raise ValueError('Independent visual-source receipt required')
    vp=ROOT/vi['active_protocol_path']
    if sha(vp)!=vi['active_protocol_sha256']:raise ValueError('Visual-reference active protocol changed')
    verify(read(vp)['bindings']);paths.add(vp)
    vinventory={str((ROOT/Path(k)).resolve()) if not Path(k).is_absolute() else k:v for k,v in vi['artifact_inventory'].items()}
    specs={}
    for index in INDICES:
        q=query_paths()[index];visual=V3/f'run_v1/{index}/visual_transform_check/case_{index}_visual_transform.npz'
        vjson=visual.with_suffix('.json');exact(visual,vinventory);exact(vjson,vinventory)
        paths.update((q,visual,vjson));inputs=cases[str(index)]
        for row in inputs.values():paths.add(Path(row['path']))
        # Header/array reads only; no geometric queries at declaration time.
        with np.load(q,allow_pickle=False) as z:count=len(z['query_object_m'])
        with np.load(visual,allow_pickle=False) as z:faces=len(z['original_triangles'])
        expected=(19174,8404) if index==20 else (45484,10000)
        if (count,faces)!=expected:raise ValueError('Original full-grid cardinality changed')
        for prefix in ('native_reference','compiled_locked_native_surface'):
            check_queries(arrays(q),arrays(inputs[prefix+'_field.npz']['path']),arrays(inputs[prefix+'_geometry.npz']['path']))
        specs[str(index)]=dict(index=index,query=binding(q),visual=binding(visual),visual_receipt=binding(vjson),inputs=inputs,
            query_count=count,object_face_count=faces,limits=dict(hand_triangle_tests=2*count*1552,hand_winding_triangle_tests=2*count*1552,
            object_distance_triangle_tests=2*count*faces,object_winding_triangle_tests=2*count*faces))
    # Bind all inherited source/data/runtime bytes, not only newly convenient paths.
    bindings=dict(failed_bindings);bindings.update(prior['bindings']);bindings.update({str(x.resolve()):sha(x) for x in paths})
    tracked=set(prior['tracked_sources'])|failed_tracked|{str(x.resolve()) for x in archived}|{str((HERE/n).resolve()) for n in SOURCE_NAMES}|{str(PURE.resolve()),str(rule.resolve()),str((OLD/'orientation/protocol.json').resolve()),str((OLD/'orientation/check_orientation.py').resolve())}
    p=dict(schema_version=SCHEMA,indices=list(INDICES),stages=list(STAGES),limits=LIMITS,runtime=runtime(),
        original_thresholds=dict(distance_tie_m=1e-8,normal_difference=1e-6,facing_deadzone=1e-6,query_boundary_m=1e-8,visual_boundary_m=1e-7),
        cases=specs,bindings=bindings,tracked_sources=sorted(tracked),pure_source=str(PURE),
        output_root=str(HERE/'run_v1'),raw_root=str(ROOT/'runs'/HERE.name/'run_v1/raw'),
        failed_protocol_sha256=FAILED_PROTOCOL_SHA,failed_independent_sha256=FAILED_READBACK_SHA,
        prior_protocol_sha256=PRIOR_PROTOCOL_SHA,prior_report_sha256=PRIOR_REPORT_SHA,prior_independent_sha256=PRIOR_INDEPENDENT_SHA,
        admission_allowed=False,scope='saved fixed pose full original grids; no articulated/physical claim')
    verify(bindings);write(protocol,p);return p

def git_receipt(protocol,p):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in [*p['tracked_sources'],str(Path(protocol).resolve())]:
        path=Path(name);rel=str(path.resolve().relative_to(ROOT))
        content=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=sha(path):raise ValueError('Source/protocol must be committed unchanged: '+rel)
    return commit

def validate(protocol,committed=True):
    p=read(protocol)
    if p['schema_version']!=SCHEMA or p['indices']!=list(INDICES) or p['stages']!=list(STAGES) or p['limits']!=LIMITS or p['runtime']!=runtime():raise ValueError('Frozen declaration/runtime drift')
    if p.get('original_thresholds')!=dict(distance_tie_m=1e-8,normal_difference=1e-6,facing_deadzone=1e-6,query_boundary_m=1e-8,visual_boundary_m=1e-7):raise ValueError('Original scientific thresholds changed')
    if p.get('admission_allowed') is not False or p.get('pure_source')!=str(PURE):raise ValueError('Source/admission declaration changed')
    for index in INDICES:
        spec=p['cases'][str(index)];count,faces=(19174,8404) if index==20 else (45484,10000)
        expected=dict(hand_triangle_tests=2*count*1552,hand_winding_triangle_tests=2*count*1552,object_distance_triangle_tests=2*count*faces,object_winding_triangle_tests=2*count*faces)
        if spec['index']!=index or spec['query_count']!=count or spec['object_face_count']!=faces or spec['limits']!=expected:raise ValueError('Full-grid budget declaration changed')
        for row in [spec['query'],spec['visual'],spec['visual_receipt'],*spec['inputs'].values()]:
            if p['bindings'].get(row['path'])!=row['sha256'] or Path(row['path']).stat().st_size!=row['bytes']:raise ValueError('Unbound case input')
    verify(p['bindings'])
    return p,git_receipt(protocol,p) if committed else None

class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:raise RuntimeError('Forbidden orientation-stage import: '+fullname)

def install_guard():
    if any(name.split('.')[0] in BLOCKED for name in sys.modules):raise RuntimeError('Forbidden engine/model already loaded')
    sys.meta_path.insert(0,NoEngine())

class Budget:
    def __init__(self,limits):self.limit=limits;self.count={k:0 for k in limits};self.start=time.monotonic()
    def __call__(self,kind,count):
        if time.monotonic()-self.start>LIMITS['seconds_per_case']:raise TimeoutError('Fixed per-case wall clock exhausted')
        if kind=='checkpoint':return
        if kind not in self.limit or count<0 or self.count[kind]+count>self.limit[kind]:raise RuntimeError('Before-work query budget exceeded: '+kind)
        self.count[kind]+=count

def npz_write(path,arrays,label,artifacts):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**arrays)
    row=binding(p,label);row['arrays']={k:dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()}
    artifacts.append(row);return row

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def original_cache_hand_distances(q,count):
    distance=q['native_distance_by_part_m']
    if (distance.shape!=(count,8) or not np.isfinite(distance[:,:6]).all()
        or not np.isposinf(distance[:,6:]).all()):
        raise ValueError('Exact historical8slots: six finite hand columns plus unused+Inf carrier columns required')
    return distance[:,:6]


def check_queries(q,field,geometry):
    if (q['query_object_m'].shape!=(len(field['cell_radius_m']),3) or not np.array_equal(q['query_object_m'],field['query_object_m'])
        or not np.array_equal(original_cache_hand_distances(q,len(field['cell_radius_m'])),field['native_distance_by_part_m'])
        or geometry['triangles_object_m'].shape!=(1552,3,3) or geometry['anatomical_part_index'].shape!=(1552,)
        or not np.array_equal(geometry['native_face_id'],np.arange(1552))
        or not np.isfinite(q['area_mm2']).all() or np.any(q['area_mm2']<=0)
        or not np.isfinite(field['cell_radius_m']).all() or np.any(field['cell_radius_m']<=0)):
        raise ValueError('Original query/radius/label/face alignment changed')

def worker(protocol,index):
    install_guard();p,freeze=validate(protocol);core=load_core();spec=p['cases'][str(index)];raw=Path(p['raw_root'])/str(index)
    raw.mkdir(parents=True,exist_ok=False);artifacts=[];budget=Budget(spec['limits']);stages=[];features={};failure=None
    result=dict(schema_version=SCHEMA,source_inventory_index=index,protocol_sha256=sha(protocol),freeze_commit=freeze,status='HARD_STOP',physical_outcome=None,
        artifacts=artifacts,admission_allowed=False,native_fidelity_accepted=False,new_collection_or_training_allowed=False)
    try:
        q=arrays(spec['query']['path']);visual=arrays(spec['visual']['path'])['original_triangles'];primitive=core.primitives(p['pure_source'])
        saved={}
        for stage,prefix in (('native','native_reference'),('compiled','compiled_locked_native_surface')):
            stages.append(dict(stage=stage,status='STARTED'))
            g=arrays(spec['inputs'][prefix+'_geometry.npz']['path']);f=arrays(spec['inputs'][prefix+'_field.npz']['path']);check_queries(q,f,g);saved[stage]=f
            def chunk(start,stop,values):npz_write(raw/f'{stage}/chunk_{start:06d}_{stop:06d}.npz',values,'orientation_chunk_'+stage,artifacts)
            feat=core.features(q['query_object_m'],f['cell_radius_m'],q['visual_triangle_outward_normal'],g['triangles_object_m'],g['anatomical_part_index'].astype(np.int64),visual,primitive,budget,chunk)
            target=f['native_distance_by_part_m'] if stage=='native' else f['mapped_distance_by_part_m']
            delta=float(np.max(np.abs(feat['distance_by_part']-target)))
            if delta>LIMITS['numerical_distance_replay_limit_m']:raise ValueError('Unchanged literal distance readback mismatch')
            npz_write(raw/f'{stage}_features.npz',feat,'orientation_full_features_'+stage,artifacts);features[stage]=feat
            stages[-1].update(status='COMPLETED',maximum_distance_readback_error_m=delta)
        mapped_near=saved['compiled']['mapped_distance_by_part_m'][:,:,None]+saved['compiled']['cell_radius_m'][:,None,None]+core.E<=core.BANDS
        comparisons=core.compare(features['native'],features['compiled'],saved['native']['cell_radius_m'],saved['native']['native_near'],saved['native']['native_far'],mapped_near)
        npz_write(raw/'comparison.npz',comparisons,'all_query_part_band_orientation',artifacts)
        result.update(status='COMPLETED_SAVED_ORIENTATION_DIAGNOSTIC',measurement=core.summary(comparisons,q['area_mm2']))
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if stages and stages[-1]['status']=='STARTED':stages[-1]['status']='HARD_STOPPED_PARTIAL'
    result.update(failure=failure,stages=stages,query_work_counts=budget.count,elapsed_s=time.monotonic()-budget.start,
        forbidden_operations={k:0 for k in ('models','data','forwards','physics','MANO','HUG','retarget','fits','task_outcomes')})
    try:
        verify(p['bindings'])
        if git_receipt(protocol,p)!=freeze:raise ValueError('HEAD changed during worker')
        verify_artifacts(artifacts)
    except BaseException as e:result.update(status='HARD_STOP',post_validation_failure=dict(type=type(e).__name__,message=str(e)))
    indexed={Path(x['path']).resolve() for x in artifacts};result['unindexed_partial_files']=[binding(x) for x in raw.rglob('*') if x.is_file() and x.resolve() not in indexed]
    write(raw/'result.json',result);return 0 if result['status']=='COMPLETED_SAVED_ORIENTATION_DIAGNOSTIC' else 1

def verify_artifacts(rows):
    for row in rows:
        if binding(row['path'])!={k:row[k] for k in ('path','sha256','bytes')}:raise ValueError('Creation-time artifact changed')
        if 'arrays' in row:
            a=arrays(row['path'])
            if set(a)!=set(row['arrays']):raise ValueError('Array inventory changed')
            for k,v in a.items():
                info=row['arrays'][k]
                if list(v.shape)!=info['shape'] or str(v.dtype)!=info['dtype'] or hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()!=info['sha256']:raise ValueError('Array bytes changed')

def check_result(result,index,p,protocol,freeze):
    if (result.get('schema_version')!=SCHEMA or result.get('source_inventory_index')!=index
        or result.get('protocol_sha256')!=sha(protocol) or result.get('freeze_commit')!=freeze
        or result.get('physical_outcome') is not None or result.get('admission_allowed') is not False
        or result.get('native_fidelity_accepted') is not False or result.get('new_collection_or_training_allowed') is not False
        or result.get('forbidden_operations')!={k:0 for k in ('models','data','forwards','physics','MANO','HUG','retarget','fits','task_outcomes')}):
        raise ValueError('Child identity/null-outcome/zero-engine contract changed')
    counts=result['query_work_counts'];limits=p['cases'][str(index)]['limits']
    if set(counts)!=set(limits) or any(not isinstance(v,int) or v<0 or v>limits[k] for k,v in counts.items()):raise ValueError('Child work accounting outside frozen caps')
    if result['status']=='COMPLETED_SAVED_ORIENTATION_DIAGNOSTIC':
        if (result.get('failure') is not None or [(x['stage'],x['status']) for x in result['stages']]!=[(x,'COMPLETED') for x in STAGES]
            or any(counts[k]!=limits[k] for k in ('hand_triangle_tests','hand_winding_triangle_tests','object_distance_triangle_tests'))
            or result['measurement']['strict_mapping_gate'] not in ('FAIL','INCOMPLETE')
            or result['measurement']['native_fidelity_accepted'] is not False):raise ValueError('Completed child coverage/status contract changed')
    verify_artifacts(result['artifacts'])


def run(protocol):
    p,freeze=validate(protocol);output,raw=Path(p['output_root']),Path(p['raw_root'])
    if output.exists() or raw.exists():raise FileExistsError('Once-only output/STARTED rule')
    output.mkdir(parents=True);raw.mkdir(parents=True);write(output/'STARTED.json',dict(protocol_sha256=sha(protocol),freeze_commit=freeze))
    rows=[];stopped=False;publication_failure=None
    for index in INDICES:
        if stopped:rows.append(dict(index=index,status='NOT_RUN_AFTER_HARD_STOP'));continue
        log=output/f'case_{index}.log';started=time.monotonic();row=dict(index=index,status='STARTED')
        try:
            with log.open('wb') as stream:
                try:
                    child=subprocess.run([sys.executable,str(HERE/'run_orientation.py'),'_worker','--protocol',str(Path(protocol).resolve()),'--index',str(index)],stdout=stream,stderr=subprocess.STDOUT,timeout=610)
                    code=child.returncode
                except subprocess.TimeoutExpired:code=-1
            rp=raw/str(index)/'result.json';row.update(returncode=code,elapsed_s=time.monotonic()-started,log=binding(log))
            if rp.exists():
                result=read(rp);tp=output/str(index)/'result.json';tp.parent.mkdir();tp.write_bytes(rp.read_bytes())
                row.update(status='COMPLETED' if code==0 and result['status']=='COMPLETED_SAVED_ORIENTATION_DIAGNOSTIC' else 'HARD_STOPPED_PARTIAL',result=binding(tp),raw_result=binding(rp))
                check_result(result,index,p,protocol,freeze)
            else:row.update(status='HARD_STOPPED_WITHOUT_RESULT',operation_counts='UNKNOWN_NO_CHILD_RECEIPT',partial_files=[binding(x) for x in (raw/str(index)).rglob('*') if x.is_file()])
        except BaseException as error:
            row.update(status='HARD_STOPPED_PARTIAL',failure=dict(type=type(error).__name__,message=str(error)),partial_files=[binding(x) for x in (raw/str(index)).rglob('*') if x.is_file()])
        rows.append(row);stopped=row['status']!='COMPLETED'
    try:
        # Recheck all earlier child copies/artifacts at final publication.
        for row in rows:
            if 'result' in row:
                result=read(row['result']['path']);check_result(result,row['index'],p,protocol,freeze)
                for name in ('result','raw_result'):
                    if binding(row[name]['path'])!=row[name]:raise ValueError('Child result changed')
                if Path(row['result']['path']).read_bytes()!=Path(row['raw_result']['path']).read_bytes():raise ValueError('Tracked/raw result differs')
        verify(p['bindings'])
        if git_receipt(protocol,p)!=freeze:raise ValueError('Final HEAD changed')
    except BaseException as error:
        stopped=True;publication_failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    report=dict(schema_version=SCHEMA,status='HARD_STOP' if stopped else 'COMPLETED_SAVED_FULL_GRID_ORIENTATION_PILOT',protocol_sha256=sha(protocol),freeze_commit=freeze,cases=rows,publication_failure=publication_failure,
        physical_outcome=None,admission_allowed=False,native_fidelity_accepted=False,new_collection_or_training_allowed=False)
    write(output/'report.json',report);return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','_worker'));parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--index',type=int,choices=INDICES)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.protocol)
    elif args.command=='run':run(args.protocol)
    else:raise SystemExit(worker(args.protocol,args.index))
