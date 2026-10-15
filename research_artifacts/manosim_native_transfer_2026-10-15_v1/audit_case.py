"""Finite native MANOSim case bridge; no model or geometry work at import."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np

HERE=Path(__file__).resolve().parent
PARTS=('palm','index','middle','ring','little','thumb')
BONE_PART=(0,1,1,1,2,2,2,4,4,4,3,3,3,5,5,5)
VARIANTS=('mesh','capsule')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    return value


def source_inputs(p):
    """Read-only path inventory; imports no model, HUG, Torch or adapter."""
    root=Path(p['manosim_root']).resolve();hug=Path(p['hug_root']).resolve()
    paths={HERE/'native_decode.py',HERE/'audit_case.py',*HERE.glob('test_*.py'),Path(p['mano_asset_path']),
        hug/'src/models/mano.py',hug/'src/utils/transform_utils.py',hug/'src/utils/data_keys.py',
        root/'assets/mano_rhand_shape.npy',root/'README.md',root/'LICENSE',root/'environment.yaml',
        root/'config/myhand.yaml',root/'src/mano_build.py',root/'src/build_mjcf.py',root/'src/paths.py'}
    for variant in VARIANTS:
        xml=root/'assets/myhand/mano_rhand'/f'{variant}_hand.xml';paths.add(xml)
        tree=ET.parse(xml).getroot()
        for node in tree.iter():
            if node.get('file'):
                if node.tag!='mesh':raise ValueError('Unexpected MANOSim asset dependency')
                path=(xml.parent/node.get('file')).resolve()
                if not path.is_relative_to(root):raise ValueError('Upstream mesh asset escaped source root')
                paths.add(path)
    # Existing installed dependency sources are metadata only, never executed.
    v3=read(p['v3_protocol'])
    for directory in v3['dependency_dirs']:
        base=Path(directory)
        for package in ('manotorch','roma'):
            if (base/package).is_dir():paths.update((base/package).rglob('*.py'))
    return sorted(path.resolve() for path in paths)


def quaternion_matrix_xyzw(value):
    x,y,z,w=np.asarray(value,float)
    if not np.isfinite([x,y,z,w]).all() or abs(np.linalg.norm([x,y,z,w])-1)>1e-12:
        raise ValueError('Literal unit captured object quaternion required')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def compare_pose(state,decoded):
    anchor=np.linalg.norm(np.asarray(state.joint_anchor_world_m)-decoded['independent_anchors_world_m'],axis=1)
    rotation=float(np.max(abs(np.asarray(state.world_body_rotations)-decoded['independent_rotations_world'])))
    return dict(max_joint_anchor_error_m=float(anchor.max()),per_bone_joint_anchor_errors_m=anchor.tolist(),
        max_world_rotation_element_error=rotation,numerical_pose_pass=bool(anchor.max()<=5e-6 and rotation<=1e-6),
        numerical_limits_are_contact_allowances=False,body_centroid_used_as_joint_anchor=False)


def exact_replay(first,second):
    if set(first.named_qpos)!=set(second.named_qpos):raise ValueError('Named native pose inventory changed')
    for key in first.named_qpos:
        if not np.array_equal(first.named_qpos[key],second.named_qpos[key]):raise ValueError('Literal native named qpos changed')
    for key in ('joint_anchor_world_m','world_body_rotations','local_body_rotations','body_origin_world_m','wrist_anchor_world_transform'):
        if not np.array_equal(getattr(first,key),getattr(second,key)):raise ValueError('Literal native readback changed '+key)


def capsule_signed_distance(points_object,position_object,rotation_object,radius,halfheight):
    """Exact compiled capsule boundary distance; no tessellation/normal claim."""
    p=np.asarray(points_object,float);position=np.asarray(position_object,float);rotation=np.asarray(rotation_object,float)
    if (p.ndim!=2 or p.shape[1]!=3 or position.shape!=(3,) or rotation.shape!=(3,3)
            or not all(np.isfinite(x).all() for x in (p,position,rotation))
            or not np.isfinite(radius) or radius<=0 or not np.isfinite(halfheight) or halfheight<0):
        raise ValueError('Finite compiled capsule coordinates and literal radius/halfheight required')
    if (np.max(abs(np.einsum('ji,jk->ik',rotation,rotation,optimize=False)-np.eye(3)))>1e-12
            or abs(np.linalg.det(rotation)-1)>1e-12):raise ValueError('Proper rigid capsule frame required')
    local=np.einsum('ij,jk->ik',p-position,rotation,optimize=False)
    axis=np.zeros_like(local);axis[:,2]=np.clip(local[:,2],-halfheight,halfheight)
    return np.linalg.norm(local-axis,axis=1)-radius


class AuditCase:
    def __init__(self,index,p,output,guard_reinstall=None):
        self.source_receipt=dict(source_inventory_index=index,static_only=True,old_target_hand_geometry_used=False)
        self.counts=dict(parametric_mano_forwards=0,full_field_variant_scans=0,collider_point_queries=0)
        self.checkpoint_artifacts=[];self.auxiliary_artifacts=[]
        self.index=index;self.p=p;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        if index not in (20,28):raise ValueError('Only the two predeclared saved native development cases')
        v3=read(p['v3_protocol']);self.v1path=Path(v3['v1_protocol']);self.v1=load('native_fixed_fields',self.v1path.parent/'audit_full_field.py')
        self.old=read(self.v1path);self.plan=next(c for c in self.old['cases'] if c['source_inventory_index']==index)
        self.source=read(self.old['inventory'])['modern_cases'][index]
        if len(self.plan['grids'])!=1:raise ValueError('Exactly the frozen full original query grid required')
        self.grid=self.plan['grids'][0]
        self.a=load('native_shared_boundary_math',p['geometry_base'])
        if guard_reinstall:guard_reinstall()
        kernel=load('native_diagnosed_halfspace_math',p['geometry_kernel'])
        if guard_reinstall:guard_reinstall()
        kernel.install(self.a)
        self.a.load_case=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No historical target-hand/Shadow constructor'))
        import numba
        numba.set_num_threads(2)
        with np.load(self.grid['original_cache'],allow_pickle=False) as q,np.load(self.grid['native_cache'],allow_pickle=False) as native:
            self.query=q['query_object_m'].copy();self.radius=q['cell_radius_m'].copy()
            self.native_distance=native['native_distance_by_part_m'][:,:6].copy()
            if not np.array_equal(self.query,native['query_object_m']):raise ValueError('Native/full-grid cell order changed')
        if len(self.query)!=self.grid['query_count']:raise ValueError('Frozen full-grid denominator changed')
        self.source_receipt.update(native_sha256=self.source['native']['sha256'],query_count=len(self.query),
            required_relationships=self.grid['required_relationships'],scientific_predicates_rebased=False,
            numerical_math_only=p['geometry_kernel'],native_shape_source='pinned myhand shape values must exactly match saved native beta')

    def _artifact(self,path,label,checkpoint=False):
        path=Path(path);row=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label)
        (self.checkpoint_artifacts if checkpoint else self.auxiliary_artifacts).append(row);return row

    def _write_json(self,name,value,label):
        path=self.output/name
        with path.open('x') as f:json.dump(plain(value),f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
        return self._artifact(path,label)

    def _parametric_budget(self):
        if self.counts['parametric_mano_forwards']>=2:raise RuntimeError('Two CPU parametric MANO forwards exhausted')
        self.counts['parametric_mano_forwards']+=1

    def _object_reference(self):
        import mujoco
        model=mujoco.MjModel.from_binary_path(self.source['scene_mjb']['path']);data=mujoco.MjData(model)
        with np.load(self.source['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:data.qpos[:]=z['qpos']
        data.qvel[:]=0;mujoco.mj_forward(model,data)
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
        if body<0:raise ValueError('Captured object frame missing')
        pos=data.xpos[body].copy();rotation=data.xmat[body].reshape(3,3).copy()
        contract=read(self.source['frame_files']['scene_contract.json']['path'])
        if (np.max(abs(pos-contract['object_position_m']))>1e-12
                or np.max(abs(rotation-quaternion_matrix_xyzw(contract['object_orientation_xyzw'])))>1e-12):
            raise ValueError('Captured object-only reference disagrees with contract')
        if data.time!=0 or not np.isfinite(rotation).all() or not np.isfinite(pos).all():raise ValueError('Invalid static object reference')
        return pos,rotation

    def _full_field(self,model,data,variant,object_pos,object_rotation):
        if self.counts['full_field_variant_scans']>=2:raise RuntimeError('Two full-field variant scans exhausted before query')
        self.counts['full_field_variant_scans']+=1
        import mujoco
        geometries=[];bones=[]
        for gid in range(model.ngeom):
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):continue
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid]))
            if not name.startswith('mano_body_'):raise ValueError('Unexpected actual native collision body')
            bone=int(name.rsplit('_',1)[1]);bones.append(bone)
            if model.geom_margin[gid]!=0 or model.geom_gap[gid]!=0:raise ValueError('Collider margin/gap differs from literal boundary rule')
            if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CAPSULE:
                rep=dict(kind=int(model.geom_type[gid]),size=model.geom_size[gid].copy(),
                    pos=np.einsum('i,ij->j',data.geom_xpos[gid]-object_pos,object_rotation,optimize=False),
                    R=np.einsum('ji,jk->ik',object_rotation,data.geom_xmat[gid].reshape(3,3),optimize=False),
                    representation='exact analytic capsule from actual compiled frame/radius/halfheight')
            else:rep=self.a.geom_rep(model,data,gid,object_rotation,object_pos)
            rep['part_index']=BONE_PART[bone];geometries.append(rep)
        if sorted(bones)!=list(range(16)):raise ValueError('Exactly one actual collider per native MANO bone required')
        distances=np.full((len(self.query),6),np.inf)
        for geom in geometries:
            part=geom['part_index']
            for start in range(0,len(self.query),256):
                points=self.query[start:start+256]
                if geom['kind']==mujoco.mjtGeom.mjGEOM_CAPSULE:
                    signed=capsule_signed_distance(points,geom['pos'],geom['R'],geom['size'][0],geom['size'][1])
                else:signed,_,_=self.a.geom_query(geom,points)
                if not np.isfinite(signed).all():raise ValueError('Nonfinite actual native collider distances')
                distances[start:start+len(points),part]=np.minimum(distances[start:start+len(points),part],np.abs(signed))
                self.counts['collider_point_queries']+=len(points)
        fields,summary=self.v1.field_status(self.native_distance,distances,self.radius)
        if int(fields['required'].sum())!=self.grid['required_relationships']:raise ValueError('Native-definite relationship denominator changed')
        path=self.output/f'{variant}_full_field.npz'
        with path.open('xb') as f:np.savez_compressed(f,query_object_m=self.query,cell_radius_m=self.radius,
            native_distance_by_part_m=self.native_distance,mapped_distance_by_part_m=distances,**fields)
        artifact=self._artifact(path,variant+'_complete_fixed_field',checkpoint=True)
        return dict(variant=variant,summary=summary,array_artifact=artifact,actual_collider_count=16,
            same_anatomical_part_order=list(PARTS),surface_fidelity_status='FAIL' if (fields['definite_lost']|fields['definite_added']).any() else 'INCOMPLETE',
            field_scope='all original native-definite cell/part/band relationships; individual collider surfaces, not exterior union',
            full_orientation_status='NOT_RUN',continuous_surface_equivalence='NOT_PROVED',native_fidelity_accepted=False)

    def run(self):
        import mujoco
        adapter_module=load('bound_upstream_static_pose_adapter',self.p['adapter_source'])
        decode=load('bound_native_parameter_decoder',HERE/'native_decode.py')
        shape=Path(self.p['manosim_root'])/'assets/mano_rhand_shape.npy'
        arrays,receipt=decode.decode_native(self.source['native']['path'],self.source['frame_files']['T_world_camera.npy']['path'],
            shape,self.p['hug_root'],self.p['mano_asset_path'],self._parametric_budget)
        path=self.output/'decoded_native_reference.npz'
        with path.open('xb') as f:np.savez_compressed(f,**arrays)
        decoded_binding=self._artifact(path,'actual_source_MANO_and_independent_FK')
        self._write_json('native_decode_receipt.json',receipt,'native_decode_receipt')
        if receipt['native_sha256']!=self.source['native']['sha256']:raise ValueError('Native input identity changed')
        if not receipt['native_reconstruction_pass'] or not receipt['independent_fk_pass']:
            raise ValueError('Source reconstruction/independent FK failed; no collider qualification')
        object_pos,object_rotation=self._object_reference();variants=[]
        identity=adapter_module.asset_shape_identity(shape)
        for variant in VARIANTS:
            xml=Path(self.p['manosim_root'])/'assets/myhand/mano_rhand'/f'{variant}_hand.xml'
            assets=adapter_module.source_asset_closure(xml,shape,source_root=self.p['manosim_root'],
                extra_source_paths=[Path(self.p['manosim_root'])/'src/mano_build.py',Path(self.p['manosim_root'])/'src/build_mjcf.py'])
            model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
            adapter=adapter_module.ManoSimStaticPoseAdapter(model,data,source_assets=assets,asset_shape_identity=identity)
            state=adapter.stage(arrays['staging_wrist_world_transform'],{f'mano_joint_{i}':arrays['staging_full_rotations'][i] for i in range(1,16)},
                asset_shape_identity=identity)
            pose=state.serialize_pose();self._write_json(variant+'_literal_pose.json',pose,variant+'_literal_pose')
            replay=adapter.stage_serialized(pose);exact_replay(state,replay)
            numerical=compare_pose(replay,arrays)
            kinematic_path=self.output/f'{variant}_kinematic_readback.npz'
            with kinematic_path.open('xb') as f:np.savez_compressed(f,
                compiled_joint_anchors_world_m=replay.joint_anchor_world_m,compiled_world_body_rotations=replay.world_body_rotations,
                compiled_local_body_rotations=replay.local_body_rotations,compiled_body_origins_world_m=replay.body_origin_world_m,
                adapter_independent_joint_anchors_world_m=replay.independent_joint_anchor_world_m,
                adapter_independent_world_body_rotations=replay.independent_world_body_rotations,
                adapter_independent_body_origins_world_m=replay.independent_body_origin_world_m,
                wrist_anchor_world_transform=replay.wrist_anchor_world_transform,root_body_world_transform=replay.root_body_world_transform)
            kinematic_binding=self._artifact(kinematic_path,variant+'_compiled_and_independent_kinematic_readback')
            self._write_json(variant+'_kinematic_receipt.json',dict(kinematic_comparison=numerical,readback_artifact=kinematic_binding,
                initial_matrix_to_quaternion_precision=state.quaternion_precision,literal_replay_quaternion_precision=replay.quaternion_precision,
                adapter_independent_readback=replay.independent_readback,source_model_sha256=replay.model_sha256,
                source_closure_sha256=replay.source_closure_sha256,baseline_state_sha256=replay.baseline_state_sha256),variant+'_kinematic_receipt')
            field=self._full_field(model,adapter.data,variant,object_pos,object_rotation)
            field.update(kinematic_comparison=numerical,literal_pose_readback_pass=True,asset_closure=assets,
                initial_matrix_to_quaternion_precision=state.quaternion_precision,literal_replay_quaternion_precision=replay.quaternion_precision,
                kinematic_readback_artifact=kinematic_binding,adapter_independent_readback=replay.independent_readback,
                adapter_static_forwards=adapter.forward_calls,source_model_sha256=replay.model_sha256)
            variants.append(field)
        self.source_receipt.update(shape_identity=identity,decoded_reference_artifact=decoded_binding,
            native_decoder_receipt=receipt,object_reference_world_position_m=object_pos.tolist(),object_reference_world_rotation=object_rotation.tolist(),
            object_reference_scope='one bound historical object-only frame; historical target-hand geometry never queried')
        return dict(kinematic_pose_transfer_status='PASS' if all(v['kinematic_comparison']['numerical_pose_pass'] for v in variants) else 'FAIL',
            surface_fidelity_status='FAIL' if any(v['surface_fidelity_status']=='FAIL' for v in variants) else 'INCOMPLETE',
            variants=variants,source_reconstruction=receipt,counts=dict(self.counts),
            kinematic_scope='native joint anchors and body rotations only; rigid bone hull/capsule and posed LBS skin remain distinct',
            native_fidelity_accepted=False,production_promotion=False,new_collection_or_training_allowed=False,
            full_orientation_status='NOT_RUN',no_physics_or_HUG_generator=True)
