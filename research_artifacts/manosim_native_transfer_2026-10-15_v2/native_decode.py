"""Decode existing HUG MANO parameters; never load or sample a grasp model."""
from __future__ import annotations
import hashlib
import importlib
from pathlib import Path
import pickle
import sys
import types

import numpy as np

PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
SNAP_TO_NATIVE=(0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3)
POSITION_LIMIT_M=5e-6
ROTATION_LIMIT=1e-6


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value,shape,name):
    result=np.array(value,dtype=float,copy=True)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('Finite '+str(shape)+' '+name+' required')
    return result


def transform(points,rotation,position):
    return np.einsum('ij,nj->ni',rotation,points,optimize=False)+position


def proper_rotation(value,name,tolerance=ROTATION_LIMIT):
    value=np.asarray(value,float)
    product=np.einsum('...ji,...jk->...ik',value,value,optimize=False)
    if (value.shape[-2:]!=(3,3) or not np.isfinite(value).all()
            or np.max(abs(product-np.eye(3)))>tolerance
            or np.max(abs(np.linalg.det(value)-1))>tolerance):raise ValueError('Proper orthonormal '+name+' required')
    return value


def axis_angle_rotations(full_poses):
    """Explicit float64 Exp of the observed mean-adjusted source coefficients."""
    vectors=array(full_poses,(48,),'actual mean-adjusted axis angles').reshape(16,3)
    result=[]
    for v in vectors:
        x,y,z=v;k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        t2=float(np.dot(v,v))
        if t2<1e-12:a=1.-t2/6.+t2*t2/120.;b=.5-t2/24.+t2*t2/720.
        else:
            t=np.sqrt(t2);a=np.sin(t)/t;b=(1.-np.cos(t))/t2
        result.append(np.eye(3)+a*k+b*np.einsum('ij,jk->ik',k,k,optimize=False))
    return np.array(result)


def source_six_d_float64(value):
    """Same HUG reshape(3,2)/ROMA epsilon0 Gram-Schmidt, explicit float64."""
    columns=array(np.asarray(value).reshape(6),(6,),'saved wrist6D').reshape(3,2)
    x=columns[:,0];nx=float(np.linalg.norm(x))
    if nx==0:raise ValueError('Degenerate source wrist6D first column')
    x=x/nx;y=columns[:,1]-np.dot(x,columns[:,1])*x;ny=float(np.linalg.norm(y))
    if ny==0:raise ValueError('Degenerate source wrist6D colinear columns')
    y=y/ny;return proper_rotation(np.stack((x,y,np.cross(x,y)),axis=-1),'float64 source6D',tolerance=1e-12)


def independent_fk(neutral_anchors,full_rotations,wrist_world):
    """Double-precision parent recursion; no MANO/MuJoCo/model import."""
    rest=array(neutral_anchors,(16,3),'neutral MANO anchors')
    rots=array(full_rotations,(16,3,3),'literal mean-adjusted local rotations')
    world=array(wrist_world,(4,4),'wrist world transform')
    if not np.array_equal(world[3],[0,0,0,1]) or np.max(abs(rest[0]))>1e-12:
        raise ValueError('Wrist-centered neutral anchors and homogeneous world transform required')
    positions=np.empty((16,3));rotations=np.empty((16,3,3));positions[0]=world[:3,3];rotations[0]=world[:3,:3]
    # Root world orientation already contains full_rotations[0].
    for joint in range(1,16):
        parent=PARENTS[joint]
        positions[joint]=positions[parent]+np.einsum('ij,j->i',rotations[parent],rest[joint]-rest[parent],optimize=False)
        rotations[joint]=np.einsum('ij,jk->ik',rotations[parent],rots[joint],optimize=False)
    return positions,rotations


def _numpy(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def observed_parametric_forward(mano,params,beta,torch,count_forward):
    """Observe the ACTUAL post-mean axis-angle rotations and MANO FK outputs."""
    layer=mano.mano_layer
    if (layer.use_pca or layer.center_idx!=0 or layer.rot_mode!='axisang' or layer.side!='right'
            or not np.array_equal(_numpy(layer.th_hands_mean),np.zeros((1,45)))):
        raise ValueError('Actual HUG MANO flat/no-PCA/right/wrist-centered convention differs')
    original_rot=layer.rotation_by_axisang;original_forward=layer.forward;observed=[];outputs=[]
    def observe_rot(*a,**k):
        value=original_rot(*a,**k);observed.append({name:_numpy(value[name]) for name in ('full_rots','full_poses')});return value
    def observe_layer(*a,**k):
        value=original_forward(*a,**k);outputs.append(value);return value
    layer.rotation_by_axisang=observe_rot;layer.forward=observe_layer
    try:
        with torch.no_grad():
            count_forward();posed=mano(torch.from_numpy(params[None]),betas=torch.from_numpy(beta))
    finally:layer.rotation_by_axisang=original_rot;layer.forward=original_forward
    if len(observed)!=1 or len(outputs)!=1:raise ValueError('Exactly one actual HUG MANO pose forward expected')
    with torch.no_grad():
        count_forward();neutral=layer(torch.zeros((1,48),dtype=torch.float32),torch.from_numpy(beta))
    return dict(vertices=_numpy(posed['vertices'])[0],landmarks=_numpy(posed['landmarks_3d'])[0],
        raw_6d_global_rotation=_numpy(posed['R_3x3'])[0],raw_6d_local_rotations=_numpy(posed['pose_3x3'])[0],
        actual_full_rotations=observed[0]['full_rots'][0],actual_full_poses=observed[0]['full_poses'][0],
        posed_transforms_abs=_numpy(outputs[0].transforms_abs)[0],
        neutral_anchors=_numpy(neutral.transforms_abs)[0,:,:3,3],
        faces=_numpy(layer.get_mano_closed_faces()),actual_hands_mean=_numpy(layer.th_hands_mean))


def verify_decoded(raw,decoded,camera,shape):
    """Checks parameter reconstruction and independent kinematics separately."""
    camera=array(camera,(4,4),'camera world transform')
    if not np.array_equal(camera[3],[0,0,0,1]):raise ValueError('Homogeneous camera world transform required')
    proper_rotation(camera[:3,:3],'camera rotation',tolerance=1e-12)
    beta=np.asarray(raw['shape'],np.float32).reshape(1,10)
    if not np.array_equal(beta,np.asarray(shape,np.float32).reshape(1,10)):
        raise ValueError('Saved native shape differs from the pinned MANOSim myhand shape')
    t=np.asarray(raw['t'],np.float32).reshape(3)
    native_vertices=array(raw['mesh_vertices'],(778,3),'saved native vertices')
    native_landmarks=array(raw['landmarks_3d'],(21,3),'saved native landmarks')
    vertices=array(decoded['vertices'],(778,3),'reconstructed vertices')+t
    landmarks=array(decoded['landmarks'],(21,3),'reconstructed landmarks')+t
    full=array(decoded['actual_full_rotations'],(16,3,3),'actual post-mean rotations')
    proper_rotation(full,'actual post-mean rotations')
    staging_rotations=axis_angle_rotations(decoded['actual_full_poses'])
    staging_rotations[0]=source_six_d_float64(np.asarray(raw['R_6d'],np.float32))
    representation_error=float(np.max(abs(staging_rotations-full)))
    neutral=array(decoded['neutral_anchors'],(16,3),'source neutral anchors')
    transforms=array(decoded['posed_transforms_abs'],(16,4,4),'actual MANO transforms')
    wrist=np.eye(4);wrist[:3,:3]=np.einsum('ij,jk->ik',camera[:3,:3],full[0],optimize=False)
    wrist[:3,3]=transform(t[None],camera[:3,:3],camera[:3,3])[0]
    staging_wrist=wrist.copy();staging_wrist[:3,:3]=np.einsum('ij,jk->ik',camera[:3,:3],staging_rotations[0],optimize=False)
    independent_positions,independent_rotations=independent_fk(neutral,full,wrist)
    source_positions=transform(transforms[:,:3,3]+t,camera[:3,:3],camera[:3,3])
    source_rotations=np.einsum('ij,njk->nik',camera[:3,:3],transforms[:,:3,:3],optimize=False)
    vertex_error=np.linalg.norm(vertices-native_vertices,axis=1);landmark_error=np.linalg.norm(landmarks-native_landmarks,axis=1)
    anchor_error=np.linalg.norm(independent_positions-source_positions,axis=1)
    saved_anchor_world=transform(native_landmarks[list(SNAP_TO_NATIVE)],camera[:3,:3],camera[:3,3])
    saved_anchor_error=np.linalg.norm(independent_positions-saved_anchor_world,axis=1)
    rotation_error=float(np.max(np.abs(independent_rotations-source_rotations)))
    faces_equal=bool(np.array_equal(decoded['faces'],np.asarray(raw['mesh_faces'])))
    wrist_source=array(raw['T_camera_wrist'],(4,4),'saved native wrist')
    if (not np.array_equal(wrist_source[3],[0,0,0,1])
            or not np.array_equal(wrist_source[:3,3].astype(np.float32),t)):
        raise ValueError('Saved native wrist translation/homogeneous provenance differs from raw t')
    proper_rotation(wrist_source[:3,:3],'saved native wrist rotation')
    raw_rotation_error=float(np.max(abs(np.asarray(decoded['raw_6d_global_rotation'])-wrist_source[:3,:3])))
    staging_root_saved_error=float(np.max(abs(staging_rotations[0]-wrist_source[:3,:3])))
    staging_root_raw_error=float(np.max(abs(staging_rotations[0]-decoded['raw_6d_global_rotation'])))
    neutral_identity=bool(np.max(abs(neutral[0]))<=1e-12)
    geometry_pass=bool(vertex_error.max()<=POSITION_LIMIT_M and landmark_error.max()<=POSITION_LIMIT_M and faces_equal
                       and raw_rotation_error<=ROTATION_LIMIT and staging_root_saved_error<=ROTATION_LIMIT
                       and staging_root_raw_error<=ROTATION_LIMIT)
    fk_pass=bool(anchor_error.max()<=POSITION_LIMIT_M and saved_anchor_error.max()<=POSITION_LIMIT_M and representation_error<=ROTATION_LIMIT
                 and rotation_error<=ROTATION_LIMIT and neutral_identity)
    arrays=dict(decoded,params_translation_m=t,shape=beta,camera_world=camera,wrist_world_transform=wrist,
        native_saved_T_camera_wrist=wrist_source,mano_parents=np.asarray(PARENTS,int),snap_to_native=np.asarray(SNAP_TO_NATIVE,int),
        original_native_vertices_camera_m=native_vertices,original_native_landmarks_camera_m=native_landmarks,
        original_native_faces=np.asarray(raw['mesh_faces']),
        staging_wrist_world_transform=staging_wrist,staging_full_rotations=staging_rotations,
        independent_anchors_world_m=independent_positions,independent_rotations_world=independent_rotations,
        mano_anchors_world_m=source_positions,mano_rotations_world=source_rotations,
        native_vertices_world_m=transform(native_vertices,camera[:3,:3],camera[:3,3]),
        native_landmarks_world_m=transform(native_landmarks,camera[:3,:3],camera[:3,3]))
    receipt=dict(native_reconstruction_pass=geometry_pass,independent_fk_pass=fk_pass,
        max_vertex_reconstruction_error_m=float(vertex_error.max()),max_landmark_reconstruction_error_m=float(landmark_error.max()),
        face_table_exact=faces_equal,max_independent_anchor_error_m=float(anchor_error.max()),
        max_saved_21_to_16_anchor_error_m=float(saved_anchor_error.max()),
        max_independent_rotation_element_error=rotation_error,raw_6d_saved_wrist_rotation_error=raw_rotation_error,
        staging_root_vs_saved_wrist_rotation_error=staging_root_saved_error,staging_root_vs_raw6d_rotation_error=staging_root_raw_error,
        explicit_float64_axisangle_representation_max_change=representation_error,
        staging_rotation_representation='root:float64 same-source6D Gram-Schmidt; locals:float64 Exp of observed actual mean-adjusted source axis angles; no SVD or fit',
        raw_6d_to_actual_axisangle_rotation_change=float(max(np.max(abs(np.asarray(decoded['raw_6d_global_rotation'])-full[0])),
            np.max(abs(np.asarray(decoded['raw_6d_local_rotations'])-full[1:])))),
        numerical_position_limit_m=POSITION_LIMIT_M,numerical_rotation_limit=ROTATION_LIMIT,
        limits_are_contact_allowances=False,shape_values_exact=True,flat_hand_mean=True,use_pca=False,center_idx=0,
        rotation_convention='parent_to_child_body_rotation_v1',no_shadow_operator_basis=True,
        root_world_rotation='camera rotation times ACTUAL post-axisangle MANO root rotation',
        anchor_order='original sixteen MANO bones; SNAP_TO_NATIVE gives corresponding saved 21 landmarks',
        native_fidelity_accepted=False,physical_outcome=None)
    return arrays,receipt


def decode_native(native_path,camera_path,shape_path,hug_root,mano_asset_path,count_forward):
    """Actual existing-parameter CPU MANO reconstruction, only called after freeze."""
    native_path=Path(native_path);before=sha(native_path);raw=pickle.loads(native_path.read_bytes())
    params=np.concatenate([np.asarray(raw['t']).reshape(3),np.asarray(raw['R_6d']).reshape(6),
                           np.asarray(raw['pose_6d']).reshape(90)]).astype(np.float32)
    if not np.isfinite(params).all():raise ValueError('Finite saved99D MANO parameters required')
    beta=np.asarray(raw['shape'],np.float32).reshape(1,10);shape=np.load(shape_path,allow_pickle=False)
    if not np.array_equal(beta,np.asarray(shape,np.float32).reshape(1,10)):raise ValueError('Native/upstream shape identity differs')
    hug_root=Path(hug_root).resolve();package=types.ModuleType('hug');package.__path__=[str(hug_root/'src')]
    if 'hug' in sys.modules and list(getattr(sys.modules['hug'],'__path__',[]))!=package.__path__:
        raise ValueError('A different HUG package is already loaded')
    sys.modules['hug']=package
    source=importlib.import_module('hug.models.mano');keys=importlib.import_module('hug.utils.data_keys')
    if Path(source.__file__).resolve()!=hug_root/'src/models/mano.py':raise ValueError('HUG MANO decoder source differs')
    expected=(Path(keys.MANO_MODELS_FOLDER)/'models/MANO_RIGHT.pkl').resolve()
    if Path(mano_asset_path).resolve()!=expected:raise ValueError('Licensed MANO source path differs; no substitution')
    import torch
    torch.set_num_threads(2)
    mano=source.MANO(side='right',use_pca=False,flat_hand_mean=True).float().eval()
    decoded=observed_parametric_forward(mano,params,beta,torch,count_forward)
    arrays,receipt=verify_decoded(raw,decoded,np.load(camera_path,allow_pickle=False),shape)
    arrays['saved_params_99d_float32']=params
    receipt.update(native_sha256=before,shape_file_sha256=sha(shape_path),licensed_mano_asset_sha256=sha(mano_asset_path),
        hug_mano_source_sha256=sha(source.__file__),classification='saved_parameter_MANO_FK_not_HUG_grasp_generator_inference',
        parametric_forwards=2,licensed_asset_copied=False)
    if sha(native_path)!=before:raise ValueError('Native file changed during CPU MANO reconstruction')
    return arrays,receipt
