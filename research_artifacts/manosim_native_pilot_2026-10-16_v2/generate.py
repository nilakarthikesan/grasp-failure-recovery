"""Frozen bounded native HUG pilot proposals. No retarget, physics or training."""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, os, pickle, subprocess, sys, time, traceback, types
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction')
HUG=BASE/'runs/hug_integration_check/hug'
ASSETS=ROOT/'runs/hug_native_pilot_assets_2026-10-15_v1'
CHECKPOINT=BASE/'runs/hug_cpu_audit/checkpoints/hug_full.safetensors'
RUNTIME=ROOT/'runs/manosim_mujoco_3_4_0_runtime_v1'
SEEDS=(42,43)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'),('numpy','dtype'),('_codecs','encode')}
        allowed|={(m,n) for m in ('numpy.core.multiarray','numpy._core.multiarray') for n in ('_reconstruct','scalar')}
        allowed|={(m,'_frombuffer') for m in ('numpy.core.numeric','numpy._core.numeric')}
        if (module,name) not in allowed:raise ValueError('Unsupported observation pickle global '+module+'.'+name)
        return super().find_class(module,name)

def observation(path):
    value=ArrayUnpickler(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(value,dict) or value.get('grasp') is not None:
        raise ValueError('Pinned unlabeled evaluation observation required')
    return value

def prepare():
    import importlib.metadata
    manifest=read(ASSETS/'download_manifest.json')
    if manifest['revision']!='56ea3f100b1a4ec38bb35e986c0dd14d38977283' or manifest['observation_count']!=60:
        raise ValueError('Exact pinned 60-observation cohort required')
    observations=sorted(x['path'] for x in manifest['files'] if x['path'].endswith('.pkl'))
    if len(observations)!=60:raise ValueError('Exact 60-observation cohort required')
    inputs=[ASSETS/'download_manifest.json',ASSETS/'official_test_tree.json',CHECKPOINT]
    inputs += [ASSETS/x['path'] for x in manifest['files']]
    inputs += sorted((HUG/'src').rglob('*.py'))
    inputs += [HUG/'assets/mano_rhand_shape.npy',HUG/'assets/mano_rhand_mesh_faces.npy',HUG/'assets/norm_stats.json']
    inputs += sorted((HUG/'assets/mano_models').rglob('*.pkl'))
    cache=BASE/'runs/hug_integration_check/hf_cache'
    inputs += [p for p in sorted(cache.rglob('*')) if p.is_file() and '/.locks/' not in str(p)]
    inputs += [HERE/'generate.py',ROOT/'src/grasp_failure_prediction/integrations/hug_rng.py',ROOT/'src/grasp_failure_prediction/integrations/hug_preprocessing.py',ROOT/'src/grasp_failure_prediction/evaluation/manosim_capsule.py',ROOT/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1/source_coordinates.py']
    source=ROOT/'runs/manosim_native_source_pipeline_2026-10-15_v1/run_v1/raw/20/literal_source_buffers.npz'
    inputs += [source]
    # Bind the installed numerical runtime plus the source MANO/rotation and
    # inference packages actually resolved below; version strings alone do not
    # identify the source floating-point graph.
    package_roots=[BASE/'.venv/lib/python3.12/site-packages'/n for n in ('numpy','torch','PIL')]
    for depot,names in ((BASE/'runs/hug_integration_check/deps',('manotorch','roma')),
                        (BASE/'runs/hug_integration_check/hug-deps',('manotorch','roma','transformers','safetensors','omegaconf','einops','cv2','timm','huggingface_hub','torchvision'))):
        for name in names:
            candidate=depot/name
            if candidate.is_dir():package_roots.append(candidate)
    for directory in package_roots:
        if not directory.is_dir():raise ValueError('Declared runtime package absent '+str(directory))
        inputs += [p for p in directory.rglob('*') if p.is_file() and p.suffix in {'.py','.so','.dylib','.json','.yaml'}]
    bindings={str(p.resolve()):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(inputs))}
    if bindings[str(CHECKPOINT.resolve())]['sha256']!='515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae':raise ValueError('Pinned checkpoint differs')
    protocol=dict(schema_version='manosim_native_hug_pilot_generation_v2',bindings=bindings,
        observations=observations,seeds=list(SEEDS),point_cloud_seed=42,sampling_steps=50,batch_size=1,
        maximum_primary_proposals=120,maximum_repeat_proposals=1,maximum_source_geometry_calls=120,
        maximum_seconds=1200,HUG_revision='8d1c52d4c24bfae5a369e32e3f134f5601a02630',
        dataset_revision=manifest['revision'],checkpoint_sha256=sha(CHECKPOINT),
        source_buffers=str(source),asset_root=str(ASSETS),HF_HOME=str(cache),
        source_scope='actual HUG outputs; no human oracle labels; finite pilot only',
        selection='all30 official test objects; first and last observations; seeds42,43 before outcomes',
        coordinate_profile='postAA32_exp64_camera64; flat_zero_mean MANO; native rest inverse',
        runtime_package_roots=[str(p.resolve()) for p in package_roots],
        physics_steps=0,retarget_calls=0,training_calls=0,training_allowed=False)
    write(HERE/'generation_protocol.json',protocol)
    print(json.dumps({'protocol_sha256':sha(HERE/'generation_protocol.json'),'bindings':len(bindings),'proposals':120}))

def verify(protocol,freeze):
    if git('rev-parse','HEAD')!=freeze:raise ValueError('Frozen HEAD changed')
    relative=(HERE/'generation_protocol.json').relative_to(ROOT)
    saved=subprocess.check_output(['git','-C',str(ROOT),'show',freeze+':'+str(relative)])
    if hashlib.sha256(saved).hexdigest()!=sha(HERE/'generation_protocol.json'):raise ValueError('Protocol not frozen in Git')
    for path,row in protocol['bindings'].items():
        if Path(path).stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Bound input changed '+path)

def run(output,freeze):
    protocol=read(HERE/'generation_protocol.json');verify(protocol,freeze)
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];errors=[];counts=dict(primary=0,repeat=0,source_geometry=0);current=None
    def before(name,maximum):
        if time.monotonic()-start>protocol['maximum_seconds'] or counts[name]>=maximum:raise RuntimeError('Before-work quota exhausted '+name)
        counts[name]+=1
    try:
        import numpy as np
        import torch, PIL
        sys.path.insert(0,str(ROOT/'src'))
        for p in (BASE/'runs/hug_integration_check/hug-deps',BASE/'runs/hug_integration_check/deps',BASE/'runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages'):
            sys.path.insert(1,str(p))
        os.environ['HF_HOME']=protocol['HF_HOME'];os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        torch.set_num_threads(4)
        if (np.__version__,torch.__version__)!=('2.2.6','2.11.0'):raise ValueError('Frozen CPU runtime differs')
        package=types.ModuleType('hug');package.__path__=[str(HUG/'src')];sys.modules['hug']=package
        from hug.dataloader.grasp_dataset import GraspDataset
        from hug.inference import load_model,load_raw_checkpoint
        from hug.models.mano import mano_params_to_grasp_dict
        from hug.utils import pcl_utils
        from grasp_failure_prediction.integrations.hug_rng import seed_cpu_generation
        from grasp_failure_prediction.integrations.hug_preprocessing import seeded_point_cloud,tensor_content_hash
        from grasp_failure_prediction.evaluation import manosim_capsule as capsule
        import manotorch, roma, huggingface_hub, transformers
        for module in (np,torch,PIL,manotorch,roma,huggingface_hub,transformers):
            origin=Path(module.__file__).resolve()
            if not any(origin.is_relative_to(Path(p)) for p in protocol['runtime_package_roots']):raise ValueError('Runtime resolved outside frozen package closure '+str(origin))
        class Dataset(GraspDataset):
            def _load_grasp_data(self,path):return observation(path)
            def _load_file_list(self,*args,**kwargs):return [ASSETS/p for p in protocol['observations']]
            def _build_pcl(self,depth_m,rgb,K,point_xyz=None):
                xyz,color=seeded_point_cloud(depth_m.numpy(),rgb,K,seed=42,backproject=pcl_utils.backproject_to_pcl,sample=pcl_utils.sample_fixed_n,n_points=self.n_points_input,center=point_xyz,crop_radius=self.pcl_crop_radius)
                return torch.from_numpy(xyz),torch.from_numpy(color)
        model=load_model(CHECKPOINT,use_ema=True,device='cpu').float().eval()
        ck=load_raw_checkpoint(CHECKPOINT,'cpu');state=ck.get('ema')
        if state is None:state=ck['model']
        state={k.removeprefix('module.'):v for k,v in state.items() if k!='n_averaged'}
        missing,unexpected=model.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith(('image_encoder.model.','mano.mano_layer.')) for k in missing):raise ValueError('HUG learned parameters missing/unexpected')
        dataset=Dataset(str(ASSETS),split='eval',pcl_crop_radius=model.pcl_crop_radius)
        with np.load(protocol['source_buffers'],allow_pickle=False) as z:rest=z['rest_quaternions'].copy();shape=z['shape'].copy()
        if not np.array_equal(model.fixed_betas.detach().numpy().reshape(1,10),shape):raise ValueError('Qualified fixed shape differs')
        if model.mano.mano_layer.th_hands_mean.dtype!=torch.float32 or bool(torch.any(model.mano.mano_layer.th_hands_mean!=0)):raise ValueError('Pinned flat zero source hand mean required')
        captured={}
        def hook(module,args,result):captured.update(aa=result.full_poses.detach().numpy()[0].copy(),anchors=result.transforms_abs[0,:,:3,3].detach().numpy().copy(),rotations=result.transforms_abs[0,:,:3,:3].detach().numpy().copy())
        handle=model.mano.mano_layer.register_forward_hook(hook)
        for index,relative in enumerate(protocol['observations']):
            data=observation(ASSETS/relative);batch=dataset[index]
            if data.get('condition_point') is None:raise ValueError('Stored query point required; no stochastic query selection')
            camera=np.asarray(data['T_world_camera'],dtype=np.float64)
            if camera.shape!=(4,4) or not np.isfinite(camera).all():raise ValueError('Literal calibrated camera required')
            tensor_hash=tensor_content_hash({k:v for k,v in batch.items() if hasattr(v,'detach')})
            def sample(seed,name):
                before(name,120 if name=='primary' else 1);seed_cpu_generation(seed,torch_module=torch)
                with torch.inference_mode():return model.sample(batch['point_uv'][None],batch['camera_K'][None],steps=50,rgb=batch['rgb'][None],pcl_xyz=batch['pcl_xyz'][None],pcl_rgb=batch['pcl_rgb'][None])[0]
            for seed in SEEDS:
                current=dict(observation=relative,seed=seed,slot=len(rows))
                prediction=sample(seed,'primary')
                if index==0 and seed==42:
                    repeated=sample(seed,'repeat')
                    np.savez_compressed(output/'first_seed_repeat.npz',original=prediction.detach().numpy(),repeated=repeated.detach().numpy())
                    if not torch.equal(prediction,repeated):raise RuntimeError('First proposal repeat RNG identity failed')
                before('source_geometry',120)
                grasp=mano_params_to_grasp_dict(prediction,model.fixed_betas.squeeze(0),model.mano,batch['camera_K'].numpy(),model.mesh_faces)
                aa=captured['aa'];t=np.asarray(grasp['t']).reshape(3)
                target=capsule.target_from_source(aa,t,camera,rest,shape_float32=shape)
                vertices=np.asarray(grasp['mesh_vertices']);anchors32=captured['anchors']+t
                R=camera[:3,:3]
                world=np.einsum('ij,nj->ni',R,vertices.astype(np.float64),optimize=False)+camera[:3,3]
                anchors=np.einsum('ij,nj->ni',R,anchors32.astype(np.float64),optimize=False)+camera[:3,3]
                rotations=np.einsum('ij,njk->nik',R,captured['rotations'].astype(np.float64),optimize=False)
                key=f'{index:02d}_seed{seed}';path=output/(key+'.npz')
                np.savez_compressed(path,prediction99=prediction.detach().numpy(),postAA32=aa,translation32=t,cameraWorld64=camera,shape=shape,rest_quaternions=rest,native_qpos=target.qpos,vertices_camera32=vertices,vertices_world=world,source_anchors_world=anchors,source_rotations_world=rotations,query_point=batch['point_uv'].numpy(),K=batch['camera_K'].numpy())
                row=dict(id=key,object_path=str(Path(relative).parent),observation=relative,seed=seed,path=str(path),sha256=sha(path),input_sha256=sha(ASSETS/relative),input_tensor_hash=tensor_hash,mapping=target.metadata(),grasp_label_available=False)
                rows.append(row);print(json.dumps({'completed':len(rows),'object':row['object_path'],'seed':seed}),flush=True)
        handle.remove();verify(protocol,freeze)
    except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
    try:verify(protocol,freeze);closure_after=True
    except BaseException as e:closure_after=False;errors.append(dict(type=type(e).__name__,message=str(e),stage='final_input_closure'))
    inventory={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}
    report=dict(schema_version='manosim_native_hug_pilot_proposals_v2',status='COMPLETE' if not errors and len(rows)==120 else 'HARD_STOP',freeze_commit=freeze,protocol_sha256=sha(HERE/'generation_protocol.json'),counts=counts,rows=rows,errors=errors,current_attempt=current,partial_artifact_inventory=inventory,closure_verified_after=closure_after,elapsed_s=time.monotonic()-start,physics_steps=0,retarget_calls=0,training_calls=0,training_allowed=False)
    write(output/'generation_report.json',report)
    print(json.dumps({'status':report['status'],'rows':len(rows),'counts':counts,'errors':errors,'report_sha256':sha(output/'generation_report.json')}))
    return int(bool(errors))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('prepare')
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',required=True)
    a=parser.parse_args();sys.dont_write_bytecode=True
    if a.command=='prepare':prepare()
    else:raise SystemExit(run(a.output,a.freeze))
