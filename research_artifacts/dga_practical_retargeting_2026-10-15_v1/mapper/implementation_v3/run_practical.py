"""Frozen static all-row practical DGA development runner; never inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=Path('/private/tmp/dga_component_constraints_20261015_v1')

def sha(path):return 'sha256:'+hashlib.sha256(Path(path).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
def write(path,value):
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=default);f.write('\n')
def verify(deps):
    for path,digest in deps.items():
        if sha(path)!=digest:raise ValueError('Frozen dependency changed: '+path)

def main(args):
    frozen=json.loads(args.method.read_text())
    if frozen.get('status')!='FROZEN_BEFORE_PRACTICAL_MAPPING_RUN':raise ValueError('Prospective freeze required')
    verify(frozen['dependency_sha256'])
    if sha(args.protocol)!=frozen['protocol']['sha256']:raise ValueError('Practical protocol binding')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('One CPU thread required')
    p=json.loads((OLD/'native_mapping_implementation_frozen_v1.json').read_text())
    ledger=json.loads((OLD/'native_validation/results_v1/native_validation_report.json').read_text())
    if len(ledger['records'])!=16:raise ValueError('All16 ledger required')
    mods={k:module('practical_'+k,p['code_paths'][k]) for k in
          ['context','legacy','audit','geometry','mesh_helper','profile','core']}
    mapper=module('practical_mapper',HERE/'practical_mapper.py')
    uncapped=module('practical_uncapped_context',HERE/'uncapped_context.py')
    from grasp_failure_prediction.proposals.dga_practical import build_practical_reference, validate_practical_pose
    profile=mods['profile'];base=profile.original
    with np.load(p['shared_fitting_queries']['path'],allow_pickle=False) as z:queries=z['queries_object_m']
    args.output.mkdir(exist_ok=False)
    warning_counts=Counter();warning_examples={}
    def observed_warning(message,category,filename,lineno,file=None,line=None):
        key=(category.__name__,str(message),filename,int(lineno))
        warning_counts[key]+=1
        warning_examples.setdefault(key,{'category':category.__name__,'message':str(message),
                                       'filename':filename,'lineno':int(lineno)})
    records=[];saved=[];fields={};began=time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter('always');warnings.showwarning=observed_warning
        for row in ledger['records']:
            g=row['global_sample_index'];record={k:row.get(k) for k in
                ['global_sample_index','seed','batch_index','within_batch_sample_index','native_manifest',
                 'mapping_reference','raw33_row_sha256','source_limit_violations','target_limit_violations']}
            record.update(native_evidence_role='DEVELOPMENT_PREVIOUSLY_INSPECTED',execution_eligible=False)
            if row['source_limit_violations']:
                record['status']='SOURCE_DOMAIN_REJECTED_UNALTERED';records.append(record)
                write(args.output/f'row_{g:02d}.json',record);continue
            if not row.get('native_mapping_input_prerequisites_pass'):
                raise ValueError('Source valid row prerequisite failed: '+str(g))
            manifest,raws,_=profile.load_multidex_raw_ledger(row['native_manifest']['path'],
                expected_object_id='object01',expected_geometry_hash=p['geometry_content_hash'])
            raw=raws[row['within_batch_sample_index']].copy()
            if 'sha256:'+hashlib.sha256(raw.tobytes()).hexdigest()!=row['raw33_row_sha256']:
                raise ValueError('Native row identity changed')
            reference=json.loads(Path(row['mapping_reference']['path']).read_text())
            refnpz=Path(row['mapping_reference']['path']).parent/reference['arrays']['path']
            if sha(refnpz)!=reference['arrays']['sha256']:raise ValueError('Native FP32 reference changed')
            row_start=time.monotonic()
            ctx=mods['context'].MappingContext(raw33=raw,object_from_input=manifest.object_from_input,
                source_urdf=p['source_hand_urdf'],target_urdf=p['target_hand_urdf'],
                target_assets_root=p['target_assets_root'],source_runtime_npz=refnpz,
                source_sample_index=row['within_batch_sample_index'],queries_object_m=queries,
                base=base,audit=mods['audit'],geometry=mods['geometry'],legacy=mods['legacy'],
                mesh_helper=mods['mesh_helper'])
            ctx=uncapped.UncappedContext(ctx,mods['geometry'])
            for value in ctx.backend.dependency_manifest.values():
                if isinstance(value,dict) and 'path' in value and 'sha256' in value:
                    if frozen['dependency_sha256'].get(value['path'])!=value['sha256']:
                        raise ValueError('Unbound actual geometry dependency: '+value['path'])
            for path,digest in ctx.backend.dependency_manifest['target_visual_assets'].items():
                if frozen['dependency_sha256'].get(path)!=digest:raise ValueError('Unbound target visual mesh')
            source=ctx.initialize_source_fields()
            baseline,selected,trace=mapper.map_pose(ctx,source)
            native=base.decode_dga_hand_pose(raw,object_from_input=manifest.object_from_input,
                                            source_hand_urdf=Path(p['source_hand_urdf']))
            portable_reference=build_practical_reference(native,target_hand_urdf=Path(p['target_hand_urdf']))
            if portable_reference['joint_targets_rad']!=baseline['q']:
                raise ValueError('Independent portable reference projection disagrees')
            portable_selected=validate_practical_pose(native,joint_targets_rad=selected['q'],
                object_from_palm=selected['palm'],target_hand_urdf=Path(p['target_hand_urdf']))
            if np.max(np.abs(np.asarray(portable_selected['object_from_target_root'])-selected['root']))>1e-9:
                raise ValueError('Portable structural validation disagrees')
            basepayload=ctx.export_pose_payload(baseline);selectedpayload=ctx.export_pose_payload(selected)
            payload=dict(selectedpayload)
            payload.update({'baseline_'+k:v for k,v in basepayload.items() if k.startswith('target_') or k.startswith('object_from_target') or k=='source_palm_from_target_palm'})
            payload['object_from_input']=np.asarray(manifest.object_from_input,dtype=np.float64)
            saved.append((g,payload))
            fields[f'row{g}_source']=source.copy();fields[f'row{g}_baseline']=baseline['fields'].copy()
            fields[f'row{g}_selected']=selected['fields'].copy()
            record.update(status='IDENTITY_REFERENCE' if trace['identity_pose'] else 'MAPPED_EXPLICIT_RETARGETING',
                elapsed_seconds=time.monotonic()-row_start,joint_names=list(ctx.names),site_names=ctx.sites,
                baseline=mapper.summarize(baseline),selected=mapper.summarize(selected),trace=trace,
                baseline_all_fields=uncapped.mixed_field_metrics(source,baseline['fields']),
                selected_all_fields=uncapped.mixed_field_metrics(source,selected['fields']),
                query_counts=ctx.query_counts,source_identity=ctx.source_identity(),
                backend_dependency_manifest=ctx.backend.dependency_manifest)
            record['portable_reference']=portable_reference;record['portable_selected']=portable_selected
            records.append(record);write(args.output/f'row_{g:02d}.json',record)
            print(json.dumps({'row':g,'status':record['status'],'elapsed_seconds':record['elapsed_seconds'],
                'projection_joints':trace['projection_changed_joint_names'],
                'correction_branches':[x['part'] for x in trace['branches'] if x['adopted_correction']]}),flush=True)
    np.savez(args.output/'poses.npz',global_sample_indices=np.array([g for g,_ in saved],dtype=np.int64),
             **{k:np.stack([payload[k] for _,payload in saved]) for k in saved[0][1]})
    np.savez_compressed(args.output/'fields.npz',queries_object_m=queries,**fields)
    warning_rows=[dict(warning_examples[k],count=v) for k,v in warning_counts.items()]
    report={'schema_version':'dga_practical_mapping_development_v1','method_sha256':sha(args.method),
        'protocol_sha256':sha(args.protocol),'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':time.monotonic()-began,'records':records,
        'accounting':{'native_rows':16,'source_valid_mapped':len(saved),'source_domain_rejected':16-len(saved)},
        'warnings':warning_rows,'zero_execution':{'physics_steps':0,'inference':0,'evaluation_episodes':0,'training':0},
        'execution_eligible':False,'native_fidelity_equivalence':'NOT_CLAIMED',
        'all_rows_previously_inspected_development_only':True,
        'field_scope':'Five finger unsigned visual fields uncapped; palm/wrist unsigned and signed palm diagnostics capped4mm',
        'outputs':{x.name:sha(x) for x in args.output.iterdir() if x.is_file()}}
    if {'mujoco','torch','isaacgym','pybullet'} & set(sys.modules):raise ValueError('Forbidden runtime import')
    verify(frozen['dependency_sha256'])
    write(args.output/'report.json',report)
    print(json.dumps({'status':'COMPLETE','accounting':report['accounting'],'elapsed_seconds':report['elapsed_seconds'],
                      'warning_count':sum(warning_counts.values()),'report_sha256':sha(args.output/'report.json')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
