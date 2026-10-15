"""One pinned upstream build from literal saved shape; no work at import.

The parent runner must freeze/authorize the build first. A separate audit freeze
then binds the generated asset hashes. No licensed MANO asset is copied here.
"""
from __future__ import annotations
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import traceback
import types
from xml.etree import ElementTree as ET

import numpy as np

UPSTREAM='c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac'
PLAN_SCHEMA='manosim_literal_native_shape_build_plan_v1'
REPORT_SCHEMA='manosim_literal_native_shape_build_report_v1'
SHAPE_REL='assets/mano_rhand_shape.npy'
HAND_REL='assets/literal_native/mano_rhand'
GENERATED_PATHS=(SHAPE_REL,)+tuple(f'{HAND_REL}/{name}' for name in ('mesh_hand.xml','capsule_hand.xml'))+tuple(
    f'{HAND_REL}/meshes/mano_bone_{i}.obj' for i in range(16))
PUBLIC_SOURCES=('src/__init__.py','src/build_mjcf.py','src/mano_build.py','src/paths.py','config/myhand.yaml','LICENSE','README.md','environment.yaml')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def shape_digest(shape):
    value=np.ascontiguousarray(shape,dtype=np.float32).reshape(1,10)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
def upstream_commit(root):return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(plain(value),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def runtime_source_bindings():
    """Read installed package bytes without importing a model or the builder."""
    distribution=importlib.metadata.distribution('trimesh')
    if distribution.version!='5.1.1':raise ValueError('Frozen Trimesh5.1.1 runtime required')
    files={}
    for relative in distribution.files or ():
        text=str(relative)
        if text.endswith(('.py','.so','.dylib','/METADATA','/WHEEL','/RECORD')):
            path=Path(distribution.locate_file(relative)).resolve()
            files[str(path)]=sha(path)
    if not files:raise ValueError('Trimesh source/package inventory unavailable')
    return dict(trimesh_version=distribution.version,numpy_version=np.__version__,files=files)


def bf16_rne_observation(canonical,literal):
    """Scalar IEEE bits observation only; no assertion about checkpoint origin."""
    canonical=np.asarray(canonical)
    if canonical.dtype not in (np.dtype('float32'),np.dtype('float64')) or canonical.size!=10 or not np.isfinite(canonical).all():
        raise ValueError('Ten finite float32/float64 nominal shape coefficients required')
    converted=np.ascontiguousarray(canonical,dtype=np.float32).reshape(1,10)
    if not np.isfinite(converted).all():raise ValueError('Nominal float32 observation conversion overflow')
    bits=converted.view(np.uint32)
    rounded=np.empty((1,10),np.uint32)
    for i,word in enumerate(bits.flat):
        n=int(word);rounded.flat[i]=((n+0x7fff+((n>>16)&1))&0xffff0000)
    values=rounded.view(np.float32)
    return dict(method='nominal_source_dtype_to_float32_then_IEEE_bfloat16_round_to_nearest_ties_to_even_scalar_bits',
        nominal_source_dtype=str(canonical.dtype),observation_intermediate_dtype='float32',
        nominal_to_float32_max_abs_difference=float(np.max(np.abs(canonical.reshape(1,10).astype(np.float64)-converted.astype(np.float64)))),
        observed_exact_match=bool(np.array_equal(values,literal)),
        canonical_to_literal_max_abs_difference=float(np.max(np.abs(canonical.reshape(1,10).astype(np.float64)-literal.astype(np.float64)))),
        rounded_values_float32=values.tolist(),causal_checkpoint_origin_proven=False)


def build_plan(native_inputs,upstream_root,mano_asset_path):
    """Read metadata/source only; require byte-bound cases20/28 literal float32."""
    if [row.get('index') for row in native_inputs]!=[20,28]:raise ValueError('Fixed native20/28 input order required')
    upstream_root=Path(upstream_root).resolve();mano_asset_path=Path(mano_asset_path).resolve()
    if upstream_commit(upstream_root)!=UPSTREAM:raise ValueError('Pinned upstream commit differs')
    if mano_asset_path.name!='MANO_RIGHT.pkl' or mano_asset_path.parent.name!='models':raise ValueError('Explicit licensed right MANO source required')
    bindings={};rows=[];shapes=[]
    for row in native_inputs:
        path=Path(row['path']).resolve()
        if sha(path)!=row['sha256']:raise ValueError('Bound saved native bytes differ')
        value=np.asarray(pickle.loads(path.read_bytes())['shape'])
        if value.dtype!=np.float32 or value.size!=10 or not np.isfinite(value).all():
            raise ValueError('Ten literal finite float32 native shape coefficients required; no conversion/rounding')
        shape=value.copy().reshape(1,10);shapes.append(shape);bindings[str(path)]=row['sha256']
        rows.append(dict(index=row['index'],path=str(path),sha256=row['sha256'],shape_array_sha256=shape_digest(shape)))
    if not np.array_equal(shapes[0],shapes[1]):raise ValueError('Native shapes differ; no pooled or fitted replacement')
    import yaml
    config_path=upstream_root/'config/myhand.yaml';configuration=yaml.safe_load(config_path.read_text())['build']
    for relative in PUBLIC_SOURCES:
        path=upstream_root/relative;bindings[str(path)]=sha(path)
    canonical=upstream_root/'assets/mano_rhand_shape.npy';bindings[str(canonical)]=sha(canonical)
    bindings[str(mano_asset_path)]=sha(mano_asset_path)
    runtime=runtime_source_bindings();bindings.update(runtime['files'])
    return dict(schema_version=PLAN_SCHEMA,upstream_revision=UPSTREAM,upstream_root=str(upstream_root),
        native_inputs=rows,shape_values_float32=shapes[0].tolist(),shape_values_sha256=shape_digest(shapes[0]),
        shape_origin='literal values from both existing native exports, not a causal checkpoint-precision claim',
        canonical_shape_asset_path=str(canonical),canonical_shape_asset_sha256=sha(canonical),
        shape_dtype='float32',shape_array_shape=[1,10],
        bf16_rounding_observation=bf16_rne_observation(np.load(canonical,allow_pickle=False),shapes[0]),
        runtime=runtime,
        mano_asset_path=str(mano_asset_path),licensed_mano_asset_sha256=sha(mano_asset_path),
        build_configuration=configuration,build_configuration_sha256=digest(configuration),
        build_configuration_source=str(config_path),expected_generated_paths=list(GENERATED_PATHS),
        bindings=bindings,maximum_neutral_mano_forwards=1,physics_steps=0,HUG_generator_calls=0,retarget_calls=0,
        licensed_asset_copied=False,production_promotion=False,new_collection_or_training_allowed=False)


def verify_plan(plan):
    if (plan.get('schema_version')!=PLAN_SCHEMA or plan.get('upstream_revision')!=UPSTREAM
            or plan.get('expected_generated_paths')!=list(GENERATED_PATHS)
            or plan.get('maximum_neutral_mano_forwards')!=1
            or plan.get('licensed_asset_copied') is not False
            or plan.get('shape_dtype')!='float32' or plan.get('shape_array_shape')!=[1,10]
            or [r.get('index') for r in plan.get('native_inputs',[])]!=[20,28]
            or any(plan.get(k)!=0 for k in ('physics_steps','HUG_generator_calls','retarget_calls'))
            or plan.get('production_promotion') is not False
            or plan.get('new_collection_or_training_allowed') is not False):raise ValueError('Literal shape build plan differs')
    if upstream_commit(plan['upstream_root'])!=UPSTREAM:raise ValueError('Upstream revision changed')
    for path,value in plan['bindings'].items():
        if sha(path)!=value:raise ValueError('Bound shape/source/license input changed: '+path)
    shape=np.asarray(plan['shape_values_float32'],np.float32).reshape(1,10)
    if shape_digest(shape)!=plan['shape_values_sha256'] or digest(plan['build_configuration'])!=plan['build_configuration_sha256']:
        raise ValueError('Exact shape/configuration plan changed')
    for row in plan['native_inputs']:
        if plan['bindings'].get(row['path'])!=row['sha256']:raise ValueError('Native source missing from byte bindings')
        raw=np.asarray(pickle.loads(Path(row['path']).read_bytes())['shape'])
        if raw.dtype!=np.float32 or raw.size!=10 or not np.array_equal(raw.reshape(1,10),shape) or shape_digest(raw)!=row['shape_array_sha256']:
            raise ValueError('Native shape values changed')
    import yaml
    config=Path(plan['upstream_root'])/'config/myhand.yaml'
    canonical=Path(plan['upstream_root'])/'assets/mano_rhand_shape.npy'
    required=[Path(plan['upstream_root'])/r for r in PUBLIC_SOURCES]+[canonical,Path(plan['mano_asset_path'])]
    if any(str(p) not in plan['bindings'] for p in required):raise ValueError('Critical source closure missing')
    if (str(config)!=plan['build_configuration_source'] or str(canonical)!=plan['canonical_shape_asset_path']
            or digest(yaml.safe_load(config.read_text())['build'])!=plan['build_configuration_sha256']
            or sha(canonical)!=plan['canonical_shape_asset_sha256']
            or sha(plan['mano_asset_path'])!=plan['licensed_mano_asset_sha256']):raise ValueError('Actual configuration/asset declaration differs')
    if bf16_rne_observation(np.load(canonical,allow_pickle=False),shape)!=plan['bf16_rounding_observation']:
        raise ValueError('Shape precision observation differs')
    runtime=runtime_source_bindings()
    if runtime!=plan['runtime'] or any(plan['bindings'].get(k)!=v for k,v in runtime['files'].items()):
        raise ValueError('Frozen Trimesh/NumPy runtime source inventory differs')
    return shape


def _load_upstream(root):
    root=Path(root).resolve();name='_literal_shape_manosim_source'
    if name in sys.modules and list(getattr(sys.modules[name],'__path__',[]))!=[str(root/'src')]:
        raise ValueError('Different upstream builder package already loaded')
    package=types.ModuleType(name);package.__path__=[str(root/'src')];sys.modules[name]=package
    module=importlib.import_module(name+'.build_mjcf')
    if Path(module.__file__).resolve()!=root/'src/build_mjcf.py':raise ValueError('Actual build source differs')
    return module


def _numpy(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def _observed_build(plan,hand_dir,shape,counts,guard_reinstall,observation):
    """Observe source returns/inputs; return the ORIGINAL objects unchanged."""
    module=_load_upstream(plan['upstream_root'])
    if guard_reinstall:guard_reinstall()
    mano_module=importlib.import_module(module.__package__+'.mano_build')
    cls=mano_module.ManoLayer;original_forward=cls.forward;original_geometry=module.forward_mano
    arrays=observation['arrays'];body_meta=observation['body_metadata'];counts['geometry_builder_returns']=0
    def forward(self,*a,**k):
        if counts['neutral_mano_forward_attempts']>=1:raise RuntimeError('One neutral MANO forward exhausted before call')
        counts['neutral_mano_forward_attempts']+=1
        pose=_numpy(a[0]);beta=_numpy(a[1] if len(a)>1 else k['betas'])
        arrays.update(actual_pose_coefficients=pose,actual_shape_coefficients=beta)
        if (pose.shape!=(1,48) or pose.dtype!=np.float32 or beta.dtype!=np.float32 or np.any(pose!=0) or not np.array_equal(beta,shape)
                or self.side!='right' or self.center_idx!=0 or self.use_pca
                or not np.array_equal(_numpy(self.th_hands_mean),np.zeros((1,45)))):
            raise ValueError('Actual neutral MANO build inputs/convention differ')
        counts['neutral_mano_forwards']+=1
        value=original_forward(self,*a,**k)
        arrays.update(actual_pose_coefficients=pose,actual_shape_coefficients=beta,
            neutral_vertices=_numpy(value.verts),neutral_landmarks=_numpy(value.joints),
            neutral_transforms_abs=_numpy(value.transforms_abs),skinning_weights=_numpy(self.th_weights),
            original_faces=_numpy(self.th_faces),actual_hands_mean=_numpy(self.th_hands_mean))
        return value
    def geometry(*a,**k):
        value=original_geometry(*a,**k);counts['geometry_builder_returns']+=1
        bodies,centers,joints,tips=value
        arrays.update(body_centers=np.array(centers,copy=True),body_joints=np.array(joints,copy=True),body_tips=np.array(tips,copy=True))
        for i,body in bodies.items():
            for field in ('v_local','hull_faces','inertia','cm'):arrays[f'bone_{i}_{field}']=np.array(body[field],copy=True)
            mass=float(body['mass']);arrays[f'bone_{i}_mass_kg']=np.asarray(mass)
            body_meta[str(i)]=dict(mass_kg=mass if np.isfinite(mass) else None,object_path=str(body['obj_path']),
                neutral_vertex_count=len(body['v_local']),hull_face_count=len(body['hull_faces']))
        return value
    cls.forward=forward;module.forward_mano=geometry
    try:
        module._build_side('right',Path(hand_dir),shape,Path(plan['mano_asset_path']).parent.parent,plan['build_configuration'])
    finally:cls.forward=original_forward;module.forward_mano=original_geometry
    return arrays,body_meta


def build_exact_shape(plan,output_dir,*,build_protocol_sha256,guard_reinstall=None):
    shape=verify_plan(plan)
    if not isinstance(build_protocol_sha256,str) or re.fullmatch('[0-9a-f]{64}',build_protocol_sha256) is None:
        raise ValueError('Exact frozen build protocol SHA required')
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=False)
    counts=dict(neutral_mano_forward_attempts=0,neutral_mano_forwards=0,geometry_builder_returns=0,MuJoCo_forwards=0,physics_steps=0,HUG_generator_calls=0,retarget_calls=0)
    artifacts=[];failure=None;arrays={};body_meta={};generated={}
    def record(path,label):
        row=dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size,label=label);artifacts.append(row);return row
    shape_path=output/SHAPE_REL;shape_path.parent.mkdir(parents=True)
    with shape_path.open('xb') as f:np.save(f,shape,allow_pickle=False)
    record(shape_path,'literal_native_shape')
    write(output/'build_plan.json',plan);record(output/'build_plan.json','frozen_build_plan')
    for relative in PUBLIC_SOURCES:
        source=Path(plan['upstream_root'])/relative;target=output/'source_snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(source.read_bytes())
        record(target,'unchanged_public_source:'+relative)
    try:
        _observed_build(plan,output/HAND_REL,shape,counts,guard_reinstall,
            dict(arrays=arrays,body_metadata=body_meta))
        if counts['neutral_mano_forwards']!=1 or counts['geometry_builder_returns']!=1:raise ValueError('Exactly one neutral producer forward/geometry return required')
        if not np.array_equal(shape,np.asarray(plan['shape_values_float32'],np.float32)):raise ValueError('Actual build shape input mutated')
        if digest(plan['build_configuration'])!=plan['build_configuration_sha256']:raise ValueError('Source build configuration mutated')
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite neutral producer geometry')
        if set(body_meta)!=set(map(str,range(16))):raise ValueError('Exactly sixteen neutral bone producer records required')
        expected=set(GENERATED_PATHS)
        actual={str(x.relative_to(output)) for x in (output/'assets').rglob('*') if x.is_file()}
        if actual!=expected:raise ValueError('Expected exact right-only shape/2XML/16OBJ generated inventory')
        for relative in GENERATED_PATHS:
            file=output/relative
            if file.is_symlink() or not file.is_file():raise ValueError('Generated artifact must be a regular file')
            generated[relative]=sha(file)
            if relative!=SHAPE_REL:record(file,'generated_asset:'+relative)
        for name in ('mesh_hand.xml','capsule_hand.xml'):
            tree=ET.parse(output/HAND_REL/name).getroot()
            if sorted(node.get('file') for node in tree.findall('./asset/mesh'))!=sorted(f'meshes/mano_bone_{i}.obj' for i in range(16)):
                raise ValueError('Generated XML mesh closure differs')
        verify_plan(plan)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));(output/'exception.txt').write_text(traceback.format_exc())
        record(output/'exception.txt','build_failure_traceback')
        # Preserve partial producer outputs as they stand after the failed call;
        # a partial inventory can never be selected for an audit.
        indexed={row['path'] for row in artifacts}
        for path in sorted((output/'assets').rglob('*')):
            if path.is_file() and not path.is_symlink() and str(path.resolve()) not in indexed:
                record(path,'partial_generated_asset_after_failure')
    if arrays:
        path=output/'neutral_producer_geometry.npz'
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        record(path,'observed_neutral_producer_geometry')
    report=dict(schema_version=REPORT_SCHEMA,status='COMPLETE' if failure is None else 'HARD_STOP',failure=failure,
        generated_root=str(output),shape_asset_path=str(shape_path),shape_asset_sha256=sha(shape_path),
        right_hand_directory=str(output/HAND_REL),generated_files=generated,upstream_revision=UPSTREAM,
        build_protocol_sha256=build_protocol_sha256,plan_sha256=digest(plan),native_inputs=plan['native_inputs'],
        shape_values_float32=plan['shape_values_float32'],shape_values_sha256=plan['shape_values_sha256'],
        shape_dtype=plan['shape_dtype'],shape_array_shape=plan['shape_array_shape'],
        bf16_rounding_observation=plan['bf16_rounding_observation'],runtime=plan['runtime'],
        build_configuration=plan['build_configuration'],build_configuration_sha256=plan['build_configuration_sha256'],
        licensed_mano_asset_path=plan['mano_asset_path'],licensed_mano_asset_sha256=plan['licensed_mano_asset_sha256'],
        public_source_bindings={str(Path(plan['upstream_root'])/r):plan['bindings'][str(Path(plan['upstream_root'])/r)] for r in PUBLIC_SOURCES},
        counts=counts,body_metadata=body_meta,artifacts=artifacts,
        artifact_hash_time='shape/snapshots immediately after write; generated geometry immediately after pinned build return; partial files immediately after failure',
        physical_parameters='same upstream rules/config; per-bone masses and inertias recomputed from exact-shape hull geometry',
        licensed_asset_copied=False,native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False)
    write(output/'build_report.json',report);return report


def generated_asset_selection(report_path):
    """Verify immutable producer output before a SEPARATE audit declaration."""
    report_path=Path(report_path).resolve();report=json.loads(report_path.read_text())
    if (report.get('schema_version')!=REPORT_SCHEMA or report.get('status')!='COMPLETE' or report.get('failure') is not None
            or report.get('upstream_revision')!=UPSTREAM or report.get('counts',{}).get('neutral_mano_forwards')!=1
            or report.get('licensed_asset_copied') is not False):raise ValueError('Completed exact-shape build receipt required')
    literal_root=Path(report['generated_root']);root=literal_root.resolve();generated=report['generated_files']
    if not literal_root.is_absolute() or literal_root!=root or not root.is_dir():raise ValueError('Literal generated root required')
    if set(generated)!=set(GENERATED_PATHS):raise ValueError('Exact19 generated asset files required')
    indexed={}
    for row in report['artifacts']:
        literal=Path(row['path']);path=literal.resolve()
        if (literal!=path or not path.is_relative_to(root) or path in indexed or not path.is_file()
                or any(p.is_symlink() for p in (literal,*literal.parents))):raise ValueError('Invalid generated build artifact path')
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise ValueError('Write-time build artifact changed')
        indexed[path]=row
    for relative,expected in generated.items():
        path=root/relative
        if path not in indexed or sha(path)!=expected or indexed[path]['sha256']!=expected:raise ValueError('Generated asset missing or changed')
    shape=root/SHAPE_REL
    if str(shape)!=report['shape_asset_path'] or sha(shape)!=report['shape_asset_sha256']:raise ValueError('Exact generated shape identity differs')
    values=np.load(shape,allow_pickle=False)
    if values.dtype!=np.float32 or values.shape!=(1,10) or shape_digest(values)!=report['shape_values_sha256']:
        raise ValueError('Literal generated shape bytes/values differ')
    if not np.array_equal(values,np.asarray(report['shape_values_float32'],np.float32)):raise ValueError('Literal generated shape values differ')
    plan=json.loads((root/'build_plan.json').read_text())
    if (digest(plan)!=report['plan_sha256'] or plan['native_inputs']!=report['native_inputs']
            or plan['shape_values_sha256']!=report['shape_values_sha256']
            or plan['build_configuration_sha256']!=report['build_configuration_sha256']
            or digest(report['build_configuration'])!=report['build_configuration_sha256']
            or report.get('production_promotion') is not False
            or report.get('native_fidelity_accepted') is not False
            or report.get('new_collection_or_training_allowed') is not False
            or any(report['counts'].get(k)!=0 for k in ('MuJoCo_forwards','physics_steps','HUG_generator_calls','retarget_calls'))
            or report['counts'].get('neutral_mano_forward_attempts')!=1
            or report['counts'].get('geometry_builder_returns')!=1):raise ValueError('Build plan/report or operation bounds differ')
    return dict(source_root=str(root),shape_asset=str(shape),expected_shape_sha256=report['shape_asset_sha256'],
        right_hand_directory=str(root/HAND_REL),variant_xmls={v:str(root/HAND_REL/f'{v}_hand.xml') for v in ('mesh','capsule')},
        public_source_snapshots=[str(root/'source_snapshots'/r) for r in PUBLIC_SOURCES],
        builder_provenance=dict(path=str(report_path),sha256=sha(report_path)),build_protocol_sha256=report['build_protocol_sha256'],
        native_inputs=report['native_inputs'])
