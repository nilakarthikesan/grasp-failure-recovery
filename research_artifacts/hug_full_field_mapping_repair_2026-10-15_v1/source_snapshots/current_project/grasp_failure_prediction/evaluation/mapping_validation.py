"""Reusable static HUG mapping checks and explicitly historical evidence joins.

No inference, retarget optimization or dynamics are run. PASS is always scoped
to a named check; sampled contact/tracking measurements never become fidelity
acceptance. Use ``make_mano_reconstructor`` to supply licensed reconstruction
for fresh proposals; omitting it deliberately leaves that gate NOT_RUN.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sys
import types
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
from grasp_failure_prediction.integrations.hug_projection import (
    inverse_transform_points, project_camera, rotation_from_saved_six_d, transform_points,
)
from .retargeting import (LEGACY_RETARGETING_PROFILE, STATIC_RETARGETING_PROFILE,
                         camera_landmarks_to_wrist, reference_vectors)
from . import pose_validation as pv


SCHEMA = 'hug_mapping_validation_v1'
HUMAN_TASK_INDICES = np.array([[0]*10,[4,8,12,16,20,2,6,10,14,18]])
TOLERANCES = dict(native_reconstruction_m=1e-6, frame_formula_m=1e-10,
                  compiled_fk_m=1e-10, rotation_matrix=1e-10,
                  stored_2d_px=1e-8, world_roundtrip_m=1e-12,
                  projection_roundtrip_px=1e-8, compiled_camera=1e-10,
                  compiled_intrinsics_px=1e-9, native_rotation=1e-6,
                  native_translation_m=1e-12, mean_tip_guard_m=.010)
_GUARD_STACK=[]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(passed=None, *, origin='recomputed_static', metrics=None, scope='', status=None):
    return dict(status=status or ('NOT_RUN' if passed is None else 'PASS' if passed else 'FAIL'),
                origin=origin, scope=scope, metrics=metrics or {})


def combined_status(checks):
    values=[row['status'] for row in checks]
    return 'FAIL' if 'FAIL' in values else 'PASS' if values and all(v=='PASS' for v in values) else 'INCOMPLETE'


@contextmanager
def static_only():
    """Sequential-process guard; restore APIs even when an input is refused."""
    counts={'physics_step_attempts':0,'static_forwards':0}
    outer=not _GUARD_STACK
    if outer:
        originals={name:getattr(mujoco,name) for name in ('mj_step','mj_step1','mj_step2','mj_forward')}
        def blocked(*a,**kw):
            for active in _GUARD_STACK:active['physics_step_attempts']+=1
            raise RuntimeError('mapping validation forbids physics integration')
        def forward(*a,**kw):
            for active in _GUARD_STACK:active['static_forwards']+=1
            return originals['mj_forward'](*a,**kw)
        for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,blocked)
        mujoco.mj_forward=forward
    _GUARD_STACK.append(counts)
    try:yield counts
    finally:
        assert _GUARD_STACK.pop() is counts
        if outer:
            for name,value in originals.items():setattr(mujoco,name,value)


def _rigid(value):
    value=np.asarray(value,float)
    if (value.shape!=(4,4) or not np.isfinite(value).all()
            or not np.array_equal(value[3],[0,0,0,1])
            or not np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),rtol=0,atol=1e-6)
            or abs(np.linalg.det(value[:3,:3])-1)>1e-6):
        raise ValueError('finite proper camera/wrist transform required')
    return value


def _rotation(axis,angle):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    x,y,z=axis;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)


def source_fk(urdf_path,names,qpos):
    """Independent fixed-root XML FK and literal limits; no project importer."""
    tree=ET.parse(urdf_path).getroot();joints=tree.findall('joint')
    names=tuple(names);qpos=np.asarray(qpos,float)
    expected={j.get('name') for j in joints if j.get('type')=='revolute'}
    if len(names)!=len(set(names)) or set(names)!=expected or qpos.shape!=(len(names),) or not np.isfinite(qpos).all():
        raise ValueError('exactly one finite command per source revolute joint required')
    commands=dict(zip(names,qpos,strict=True));children={j.find('child').get('link') for j in joints}
    roots={n.get('name') for n in tree.findall('link')}-children
    if len(roots)!=1:raise ValueError('source must be one rooted tree')
    frames={roots.pop():np.eye(4)};pending=list(joints);limits=[]
    while pending:
        progressed=False
        for joint in pending[:]:
            parent=joint.find('parent').get('link')
            if parent not in frames:continue
            origin=joint.find('origin');local=np.eye(4)
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                r,p,y=np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
                local[:3,:3]=_rotation([0,0,1],y)@_rotation([0,1,0],p)@_rotation([1,0,0],r)
            kind=joint.get('type')
            if kind=='revolute':
                axis=joint.find('axis');axis=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
                if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)==0:raise ValueError('invalid source axis')
                value=commands[joint.get('name')];motion=np.eye(4);motion[:3,:3]=_rotation(axis,value);local=local@motion
                bound=joint.find('limit');lo,hi=float(bound.get('lower')),float(bound.get('upper'))
                limits.append(dict(joint_name=joint.get('name'),value_rad=float(value),lower_rad=lo,upper_rad=hi,passed=bool(lo<=value<=hi)))
            elif kind!='fixed':raise ValueError('only fixed/revolute source joints supported')
            frames[joint.find('child').get('link')]=frames[parent]@local
            pending.remove(joint);progressed=True
        if not progressed:raise ValueError('disconnected/cyclic source tree')
    return frames,limits


def _quat_matrix(q):
    result=np.empty(9);mujoco.mju_quat2Mat(result,np.asarray(q,float));return result.reshape(3,3)


def _normalized_matrix(rotation):
    # Declared current matrix-to-quaternion convention; no anatomical refitting.
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(rotation,float).reshape(9))
    return _quat_matrix(q/np.linalg.norm(q))


def mapping_record_gate(record,pose):
    """Require actual observed statuses/raw outputs and fresh-instance policy."""
    if record is None:return check(scope='fresh solve record missing')
    try:
        prov=record['provenance'];raw=np.asarray(record['raw_refinement_qpos_rad'],float)
        q=np.asarray(record['qpos_rad'],float);limits=np.asarray(record['joint_limits_rad'],float)
        statuses=record['optimizer_status_codes'];count=prov['refinement_calls']
        expected=np.clip(raw[-1],limits[:,0],limits[:,1])
        profile=prov.get('profile_id');config=Path(prov.get('config_path',''))
        valid=(profile in (LEGACY_RETARGETING_PROFILE,STATIC_RETARGETING_PROFILE)
               and type(count) is int and count==20
               and prov.get('temporal_norm_delta')==(.004 if profile==LEGACY_RETARGETING_PROFILE else 0.)
               and prov.get('reset_before_proposal')==(profile==STATIC_RETARGETING_PROFILE)
               and config.is_file() and sha(config)==prov.get('config_sha256')
               and record.get('schema_version')=='hug_mapping_record_v1' and record.get('mapping_status')=='ok'
               and record.get('status')=='returned' and 'outcome' in record and record['outcome'] is None
               and record.get('fresh_instance_per_proposal') is True
               and record['joint_names']==list(pose.joint_names)
               and q.shape==pose.qpos.shape and np.array_equal(q,pose.qpos)
               and raw.shape==(count,len(q)) and np.isfinite(raw).all()
               and limits.shape==(len(q),2) and np.isfinite(limits).all()
               and np.array_equal(q,expected) and len(statuses)==count
               and all(type(code) is int and code>0 for code in statuses)
               and prov.get('vector_scaling')==1. and prov.get('low_pass_filter_enabled') is False
               and bool(record.get('translation_strategy_id')))
        return check(bool(valid),metrics=dict(profile_id=prov.get('profile_id'),refinement_calls=count,
                    statuses=statuses,max_final_clipping_rad=float(np.max(abs(raw[-1]-q)))),scope='observed fresh solver provenance; no convergence/quality guarantee')
    except (KeyError,TypeError,ValueError,IndexError,OSError):
        return check(False,scope='malformed fresh solver record')


def make_mano_reconstructor(hug_root):
    """Load licensed MANO kinematics only, without checkpoint/generator inference.

    Caller supplies the existing HUG/MANO dependencies. Source and licensed
    model hashes are returned in the reference, without redistributing assets.
    """
    root=Path(hug_root).resolve()
    package=types.ModuleType('hug');package.__path__=[str(root/'src')];sys.modules['hug']=package
    import torch
    from hug.models.mano import MANO
    model=MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
    bindings={str(path):sha(path) for path in (root/'src/models/mano.py',root/'assets/mano_models/models/MANO_RIGHT.pkl')}
    def reconstruct(grasp):
        if grasp.R_6d is None or grasp.pose_6d is None:raise ValueError('MANO reconstruction requires original six-D parameters')
        params=np.concatenate((grasp.t.reshape(3),grasp.R_6d.reshape(6),grasp.pose_6d.reshape(90))).astype(np.float32)
        with torch.no_grad():result=model(torch.from_numpy(params[None]),betas=torch.from_numpy(grasp.shape.astype(np.float32)[None]))
        return dict(mesh_vertices=result['vertices'][0].numpy()+params[:3],
                    landmarks_3d=result['landmarks_3d'][0].numpy()+params[:3],
                    mesh_faces=model.mano_layer.get_mano_closed_faces().detach().numpy(),bindings=bindings)
    return reconstruct


def validate_hug_mapping(grasp,pose,*,urdf_root,T_world_camera,camera_intrinsics,image_size,
                         world_palm=None,camera_model=None,camera_data=None,camera_name='front',
                         native_reconstructor=None,mapping_record=None,mapping_kind='fresh'):
    """Validate each supplied fresh/saved mapping without solving or integration.

    ``world_palm`` is the actual mapped (position, quaternion_wxyz), not the
    expected native pose. Pass the captured compiled camera model/data to test
    calibration. Missing required inputs produce INCOMPLETE, never a pass.
    """
    if mapping_kind not in ('fresh','saved_command'):raise ValueError('unknown mapping kind')
    checks={};metrics={}
    with static_only() as counts:
        try:
            camera=_rigid(T_world_camera);wrist=_rigid(grasp.T_camera_wrist)
            local=camera_landmarks_to_wrist(grasp.landmarks_3d,wrist)@MANO_TO_OPERATOR_RIGHT.T
            vectors=reference_vectors(local,HUMAN_TASK_INDICES)
            frame_error=max(float(abs(pose.landmarks_wrist_m-local).max()),float(abs(pose.reference_vectors_m-vectors).max()))
            checks['native_frame_input']=check(frame_error<=TOLERANCES['frame_formula_m'],metrics={'max_abs_error_m':frame_error},scope='declared MANO-to-operator scalar inputs')
            urdf=Path(urdf_root)/'shadow_hand/shadow_hand_right.urdf'
            frames,limits=source_fk(urdf,pose.joint_names,pose.qpos)
            checks['literal_joint_limits']=check(all(row['passed'] for row in limits),metrics={'joints':limits},scope='literal source XML; zero tolerance')
            validator=pv.ShadowPoseValidator(urdf_root);v=validator.validate(pose)
            actual=pv._link_positions(validator.model,validator.data,urdf)
            expected=np.array([frames[name][:3,3] for name in pv.TASK_LINK_NAMES])
            fk_error=float(np.linalg.norm(actual-expected,axis=1).max())
            checks['compiled_hand_realization']=check(fk_error<=TOLERANCES['compiled_fk_m'],metrics={'max_task_position_error_m':fk_error},scope='independent XML FK versus current compiled fixed hand')
            checks['mean_fingertip_guard']=check(v.mean_fingertip_error_m<=TOLERANCES['mean_tip_guard_m'],metrics={'mean_m':v.mean_fingertip_error_m,'maximum_m':v.maximum_fingertip_error_m,'per_tip_m':v.fingertip_errors_m.tolist(),'guard_m':TOLERANCES['mean_tip_guard_m']},scope='existing mean engineering guard; no surface/contact fidelity certificate')
            checks['static_contacts']=check(status='MEASURED',metrics={'pairs':[list(pair) for pair in v.contact_pairs]},scope='isolated static hand contacts; not grasp success')
            world_wrist=camera@wrist
            expected_rotation=_normalized_matrix(world_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T@frames['palm'][:3,:3])
            metrics['expected_world_palm_position_m']=world_wrist[:3,3].tolist()
            metrics['expected_world_palm_rotation']=expected_rotation.tolist()
            if world_palm is None:checks['mapped_world_palm']=check(scope='actual mapped world palm was not supplied')
            else:
                position,quaternion=world_palm
                position=np.asarray(position,float);quaternion=np.asarray(quaternion,float)
                valid=position.shape==(3,) and quaternion.shape==(4,) and np.isfinite(position).all() and np.isfinite(quaternion).all() and abs(np.linalg.norm(quaternion)-1)<=1e-10
                pe=float(np.linalg.norm(position-world_wrist[:3,3])) if valid else None
                re=float(abs(_quat_matrix(quaternion)-expected_rotation).max()) if valid else None
                checks['mapped_world_palm']=check(valid and pe<=TOLERANCES['compiled_fk_m'] and re<=TOLERANCES['rotation_matrix'],metrics={'position_error_m':pe,'rotation_matrix_error':re},scope='actual mapped placement versus native wrist/operator/source-palm convention')
            k=np.asarray(camera_intrinsics,float);width,height=image_size
            uv,positive=project_camera(grasp.landmarks_3d,k,denominator_epsilon=1e-8,include_nonpositive=True)
            restored=inverse_transform_points(transform_points(grasp.landmarks_3d,camera),camera)
            uv2,_=project_camera(restored,k,denominator_epsilon=1e-8,include_nonpositive=True)
            errors={'world_roundtrip_m':float(abs(restored-grasp.landmarks_3d).max()),
                    'projection_roundtrip_px':float(abs(uv2-uv).max()) if np.isfinite(uv).all() else None,
                    'native_translation_m':float(abs(wrist[:3,3]-grasp.t).max()),
                    'native_rotation':float(abs(wrist[:3,:3]-rotation_from_saved_six_d(grasp.R_6d)).max()) if grasp.R_6d is not None else None,
                    'stored_2d_px':float(abs(uv-grasp.landmarks_2d).max()) if grasp.landmarks_2d is not None and np.isfinite(uv).all() else None}
            checks['camera_export_consistency']=check(all(value is not None and value<=TOLERANCES[key] for key,value in errors.items()),metrics={**errors,'positive_depth_landmarks':int(positive.sum())},scope='export/frame consistency; no observed hand-keypoint ground truth')
            if camera_model is None or camera_data is None:checks['compiled_camera_calibration']=check(scope='captured compiled camera model/data not supplied')
            else:
                cid=mujoco.mj_name2id(camera_model,mujoco.mjtObj.mjOBJ_CAMERA,camera_name)
                if cid<0:raise ValueError('captured compiled camera missing')
                compiled=np.eye(4);compiled[:3,:3]=camera_data.cam_xmat[cid].reshape(3,3)*[1.,-1.,-1.];compiled[:3,3]=camera_data.cam_xpos[cid]
                focal=height/(2*np.tan(np.deg2rad(camera_model.cam_fovy[cid])/2))
                ck=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
                ce=float(abs(compiled-camera).max());ke=float(abs(ck-k).max())
                checks['compiled_camera_calibration']=check(ce<=TOLERANCES['compiled_camera'] and ke<=TOLERANCES['compiled_intrinsics_px'],metrics={'transform_max_abs':ce,'intrinsic_max_abs_px':ke},scope='captured compiled camera/intrinsics')
            if native_reconstructor is None:checks['native_reconstruction']=check(scope='licensed MANO reconstructor not supplied')
            else:
                reference=native_reconstructor(grasp)
                ve=float(np.linalg.norm(np.asarray(reference['mesh_vertices'])-grasp.mesh_vertices,axis=1).max())
                le=float(np.linalg.norm(np.asarray(reference['landmarks_3d'])-grasp.landmarks_3d,axis=1).max())
                faces=grasp.mesh_faces is not None and np.array_equal(reference['mesh_faces'],grasp.mesh_faces)
                checks['native_reconstruction']=check(ve<=TOLERANCES['native_reconstruction_m'] and le<=TOLERANCES['native_reconstruction_m'] and faces,metrics={'max_vertex_error_m':ve,'max_landmark_error_m':le,'faces_exact':bool(faces),'reference_bindings':reference.get('bindings',{})},scope='saved MANO parameters reconstructed using supplied licensed source kinematics')
            checks['fresh_mapping_provenance']=mapping_record_gate(mapping_record,pose) if mapping_kind=='fresh' else check(status='NOT_APPLICABLE',scope='literal saved command replay; no fresh solver invoked')
        except Exception as exc:
            checks['runtime_validation_error']=check(False,metrics={'type':type(exc).__name__,'message':str(exc)},scope='input/runtime refusal; no physical label')
    required=['native_frame_input','literal_joint_limits','compiled_hand_realization','mean_fingertip_guard','mapped_world_palm','camera_export_consistency','compiled_camera_calibration','native_reconstruction']
    if mapping_kind=='fresh':required.append('fresh_mapping_provenance')
    gate=combined_status([checks.get(name,check(scope='earlier validation refused')) for name in required])
    if 'runtime_validation_error' in checks:gate='FAIL'
    return dict(schema_version=SCHEMA,mapping_kind=mapping_kind,static_input_gate=gate,checks=checks,metrics=metrics,
                tolerances=dict(TOLERANCES),counts=dict(counts),native_surface_fidelity_accepted=False,
                physical_tracking_qualified=False,large_collection_qualified=False,new_outcome=None)


class BoundEvidence:
    """Verify archived payloads, including losslessly compressed historical JSON."""
    def __init__(self,directory,manifest='manifest.json'):
        self.root=Path(directory);self.manifest_path=self.root/manifest
        value=json.loads(self.manifest_path.read_text());self.entries=value.get('files',value.get('files_sha256',{}))
        if not isinstance(self.entries,dict):raise ValueError('unsupported evidence manifest')
        self.references=[]
    def read(self,relative):
        expected=self.entries.get(relative)
        if isinstance(expected,dict):expected=expected.get('sha256')
        if not expected:raise ValueError(f'evidence not covered by manifest: {relative}')
        path=self.root/relative
        if path.exists():payload=path.read_bytes()
        elif path.with_suffix(path.suffix+'.gz').exists():
            path=path.with_suffix(path.suffix+'.gz');payload=gzip.decompress(path.read_bytes())
        else:raise FileNotFoundError(path)
        digest=hashlib.sha256(payload).hexdigest()
        if digest!=str(expected).removeprefix('sha256:'):raise ValueError(f'evidence hash mismatch: {path}')
        self.references.append(dict(path=str(path.resolve()),file_sha256=sha(path),payload_sha256=digest,manifest=str(self.manifest_path.resolve()),manifest_sha256=sha(self.manifest_path)))
        return json.loads(payload)


def historical_evidence(project,index,native_sha256):
    """Join only matching native identities; absent evidence is explicitly NOT_RUN."""
    root=Path(project)/'research_artifacts';out={};refs=[];identity=native_sha256.removeprefix('sha256:')
    def load_group(name,directory,reader,manifest='manifest.json'):
        try:
            store=BoundEvidence(root/directory,manifest);out[name]=reader(store);refs.extend(store.references)
        except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:
            out[name]=check(origin='bound_historical',scope=str(exc),status='NOT_RUN')
    def native(store):
        row=store.read(f'measurement/cases/case_{index:02}.json')
        if row['native_sha256']!=identity:raise ValueError('native identity mismatch')
        return check(row['native_reconstruction_pass'],origin='bound_historical',metrics={'reconstruction':row['source_reconstruction']},scope='original native parameter reconstruction only')
    load_group('native_reconstruction','hug_native_mano_comparison_2026-10-15_v1',native)
    def contact(store):
        row=store.read(f'measurement/cases/case_{index:02}.json')
        if row['native_sha256']!=identity:raise ValueError('native identity mismatch')
        return check(status='MEASURED',origin='bound_historical',metrics={'shadow_tip_errors_m':row['shadow_tip_errors_m'],'parts':row['parts']},scope='finite component distances/band changes; no surface/contact acceptance')
    load_group('component_contact_metrics','hug_native_mano_comparison_2026-10-15_v1',contact)
    def exposed(store):
        row=store.read(f'cases/case_{index:02}.json')
        if row['native_sha256']!=identity:raise ValueError('native identity mismatch')
        return check(status='MEASURED',origin='bound_historical',metrics={'sampling':row['sampling'],'skin_topology':row['skin_topology']},scope='sampled native-proxy union; penetration intervals/unknown normals retained; fidelity unaccepted')
    load_group('native_exposed_boundary','hug_exposed_boundary_2026-10-15_v1',exposed)
    def scene(store):
        row=store.read(f'cases/case_{index:02}.json')
        if row['native_sha256']!=identity:raise ValueError('native identity mismatch')
        return check(row['current_command_within_source_limits_and_realized'],origin='bound_historical',metrics={'command_realization_pass':row['current_assembled_static_command_realization_pass'],'literal_source_limits_pass':row['all_literal_source_joint_limits_pass']},scope='existing assembled saved-command scene realization, no new run')
    load_group('assembled_saved_scene','hug_mapping_acceptance_repair_2026-10-15_v1/current_scene_bridge_v2',scene,'bundle_manifest.json')
    def camera(store):
        p=store.read('protocol.json');report=store.read('results.json');row=next(c for c in report['cases'] if c['index']==index)
        if p['bindings'].get(row['native'])!=identity:raise ValueError('native identity mismatch')
        return check(row['numerical_consistency_pass'],origin='bound_historical',metrics={'checks':row['checks'],'errors':row['errors']},scope='existing camera/export consistency; no observed 2D ground truth')
    load_group('saved_camera','hug_camera_projection_2026-10-15_v1',camera)
    def fresh(store):
        protocol=store.read('protocol.json');report=store.read('results.json')
        case=next(c for c in protocol['cases'] if c['index']==index)
        if case['native_sha256']!=identity:raise ValueError('native identity mismatch')
        profiles=[dict(profile_id=p['profile_id'],**next(c for c in p['cases'] if c['index']==index)) for p in report['profiles']]
        return check(all(c['fresh_instance_gate_pass'] for c in profiles),origin='bound_historical',metrics={'profiles':profiles},scope='four-input development fresh-instance qualification; reused legacy failures retained')
    load_group('fresh_solve_development','hug_fresh_mapping_qualification_2026-10-15_v1',fresh)
    def pilot(store):
        protocol=store.read('protocol.json');report=store.read('summary.json')
        if protocol['cases'][str(index)]['native_sha256'].removeprefix('sha256:')!=identity:raise ValueError('native identity mismatch')
        rows=[r for r in report['schedule'] if r['source_inventory_index']==index]
        if not rows:raise ValueError('not included in the bounded pilot')
        return check(all(r['status']=='COMPLETED' for r in rows),origin='bound_historical',metrics={'attempts':rows,'tracking_acceptance':'MEASURED_ONLY; no physical-fidelity threshold','determinism':'PASS' if any(r.get('determinism',{}).get('passed') for r in rows if r.get('determinism')) else 'NOT_RUN'},scope='completed saved-command recorder pilot; physical task outcomes are not mapping/fidelity labels')
    load_group('dynamic_saved_replay_and_tracking','hug_saved_recording_pilot_2026-10-15_v1',pilot)
    return dict(checks=out,references=refs,recomputed_geometry=False,new_physics_steps=0,
                qualifies_current_output=False,
                scope='native-identity-linked historical commands/protocols; never substitutes current output validation')
