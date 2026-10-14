"""Guarded static nominal-pose coverage audit using existing saved cube state."""
from pathlib import Path
import hashlib
import json
import pickle
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def forbid_integration(*args, **kwargs):
    raise RuntimeError('Static nominal contact audit forbids MuJoCo time integration')

for api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, api, forbid_integration)

BASE=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/runs/hug_grasp_eval_2026-10-13_fresh_cube')
PFILE=BASE/'output/proposals/hug_cube/sample_0000/proposal.json'
NATIVE=PFILE.parent/'native'
OUT=Path('/private/tmp/grasp-mapping-audit-2026-10-14')
proposal=json.loads(PFILE.read_text())
contract=json.loads((NATIVE/'scene_contract.json').read_text())
native=pickle.loads((NATIVE/'proposal.pkl').read_bytes())
snapshot=np.load(NATIVE/'scene_state.npz')
model=mujoco.MjModel.from_binary_path(str(NATIVE/'scene.mjb'))
data=mujoco.MjData(model)
data.qpos[:]=snapshot['qpos']
data.qvel[:]=0
jid=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
bid=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
gid=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
root_adr=int(model.jnt_qposadr[jid('world_joint')])
object_adr=int(model.jnt_qposadr[jid('object_free')])
object_initial_qpos=data.qpos[object_adr:object_adr+7].copy()
hand_names=list(proposal['joint_targets_rad'])
for name,value in proposal['joint_targets_rad'].items():
    data.qpos[int(model.jnt_qposadr[jid(name)])]=value
data.qpos[root_adr:root_adr+3]=0
data.qpos[root_adr+3:root_adr+7]=[1,0,0,0]
mujoco.mj_forward(model,data)
palm_id=bid('palm')
object_id=bid('object')
forearm_id=bid('forearm')
palm_translation_base=data.xpos[palm_id].copy()
palm_rotation_base=data.xmat[palm_id].reshape(3,3).copy()
object_world_translation=data.xpos[object_id].copy()
object_world_rotation=data.xmat[object_id].reshape(3,3).copy()
object_to_palm_rotation=Rotation.from_quat(proposal['object_to_palm']['quaternion_xyzw']).as_matrix()
desired_palm_rotation=object_world_rotation@object_to_palm_rotation
desired_palm_translation=object_world_translation+object_world_rotation@np.array(proposal['object_to_palm']['translation_m'])
root_rotation=desired_palm_rotation@palm_rotation_base.T
root_translation=desired_palm_translation-root_rotation@palm_translation_base
root_quat=Rotation.from_matrix(root_rotation).as_quat()[[3,0,1,2]]
data.qpos[root_adr:root_adr+3]=root_translation
data.qpos[root_adr+3:root_adr+7]=root_quat
if model.nmocap:
    data.mocap_pos[0]=root_translation
    data.mocap_quat[0]=root_quat
for name,value in proposal['joint_targets_rad'].items():
    actuator=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'position_'+name)
    if actuator>=0:
        data.ctrl[actuator]=value
mujoco.mj_forward(model,data)
assert np.array_equal(data.qpos[object_adr:object_adr+7],object_initial_qpos)
box_id=gid('object_geom')
assert model.geom_type[box_id]==mujoco.mjtGeom.mjGEOM_BOX
half_extents=model.geom_size[box_id].copy()
assert np.allclose(model.geom_pos[box_id],0)
assert np.allclose(model.geom_quat[box_id],[1,0,0,0])

def to_object(points):
    return (np.asarray(points)-object_world_translation)@object_world_rotation

def box_signed_distance(points):
    delta=np.abs(points)-half_extents
    return np.linalg.norm(np.maximum(delta,0),axis=1)+np.minimum(np.max(delta,axis=1),0)

FACES=[('+X',0,1),('-X',0,-1),('+Y',1,1),('-Y',1,-1),('+Z',2,1),('-Z',2,-1)]

def nearest_faces(points):
    """Nearest bounded face rectangle; choose first listed face for edge ties."""
    points=np.asarray(points).reshape(-1,3)
    distances=[]
    for label,axis,sign in FACES:
        closest=np.clip(points,-half_extents,half_extents)
        closest[:,axis]=sign*half_extents[axis]
        distances.append(np.linalg.norm(points-closest,axis=1))
    distance=np.stack(distances,axis=1)
    indices=np.argmin(distance,axis=1)
    signed=[]
    for point,index in zip(points,indices):
        _,axis,sign=FACES[index]
        signed.append(float(sign*point[axis]-half_extents[axis]))
    return indices,np.asarray(signed),distance[np.arange(len(points)),indices]

T_wc=np.load(NATIVE/'T_world_camera.npy')
vertices_world=np.asarray(native['mesh_vertices'])@T_wc[:3,:3].T+T_wc[:3,3]
vertices_object=to_object(vertices_world)
signed_distance=box_signed_distance(vertices_object)
face_indices,face_signed_distance,face_rectangle_distance=nearest_faces(vertices_object)
native_counts={}
for i,(label,axis,sign) in enumerate(FACES):
    in_face=face_indices==i
    penetrating=in_face&(signed_distance<0)
    near2=in_face&(np.abs(signed_distance)<=.002)
    near5=in_face&(np.abs(signed_distance)<=.005)
    native_counts[label]={
     'total_vertices_nearest_face':int(in_face.sum()),
     'penetrating_vertices':int(penetrating.sum()),
     'within_2mm_surface_vertices':int(near2.sum()),
     'within_5mm_surface_vertices':int(near5.sum()),
     'deepest_penetration_m':float(max(0,-signed_distance[in_face].min())) if in_face.any() else 0,
     'near_2mm_object_coordinate_min_m':vertices_object[near2].min(axis=0).tolist() if near2.any() else None,
     'near_2mm_object_coordinate_max_m':vertices_object[near2].max(axis=0).tolist() if near2.any() else None,
    }

def descends(body,ancestor):
    while body>0:
        if body==ancestor:return True
        body=int(model.body_parentid[body])
    return False

hand_bodies={i for i in range(model.nbody) if descends(i,forearm_id)}
table_id=gid('table')
contacts=[]
for i in range(data.ncon):
    c=data.contact[i]
    ids=[int(c.geom1),int(c.geom2)]
    bodies=[int(model.geom_bodyid[g]) for g in ids]
    if object_id in bodies and any(b in hand_bodies for b in bodies):
        category='hand_object'
    elif table_id in ids and any(b in hand_bodies for b in bodies):
        category='hand_table'
    elif all(b in hand_bodies for b in bodies):
        category='hand_self'
    elif table_id in ids and object_id in bodies:
        category='object_table'
    else:category='other'
    point_object=to_object(c.pos)
    fi,fs,fd=nearest_faces(point_object)
    contacts.append({
     'category':category,
     'bodies':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in bodies],
     'geoms':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or f'geom:{g}' for g in ids],
     'collision_distance_m':float(c.dist),
     'contact_position_world_m':c.pos.tolist(),
     'contact_position_initial_object_m':point_object.tolist(),
     'nearest_cube_face':FACES[int(fi[0])][0] if category=='hand_object' else None,
     'face_signed_plane_distance_m':float(fs[0]) if category=='hand_object' else None,
     'face_rectangle_distance_m':float(fd[0]) if category=='hand_object' else None,
    })
shadow_counts={face[0]:{'hand_object_contact_count':0,'hand_body_names':[]} for face in FACES}
for contact in contacts:
    if contact['category']=='hand_object':
        record=shadow_counts[contact['nearest_cube_face']]
        record['hand_object_contact_count']+=1
        record['hand_body_names']+= [b for b in contact['bodies'] if b!='object']
for record in shadow_counts.values():record['hand_body_names']=sorted(set(record['hand_body_names']))
native2_faces=[name for name,r in native_counts.items() if r['within_2mm_surface_vertices']]
native5_faces=[name for name,r in native_counts.items() if r['within_5mm_surface_vertices']]
native_inside_faces=[name for name,r in native_counts.items() if r['penetrating_vertices']]
shadow_faces=[name for name,r in shadow_counts.items() if r['hand_object_contact_count']]
report={
 'audit_type':'Static saved nominal Shadow pose at original initial captured object state; before approach, controller force-close, or lift',
 'guarded_integration_apis':['mj_step','mj_step1','mj_step2'],
 'mujoco_time_after_fk_s':float(data.time),
 'object_initial_state_source':str(NATIVE/'scene_state.npz'),
 'source_model':str(NATIVE/'scene.mjb'),
 'nominal_shadow_source':str(PFILE),
 'native_mano_source':str(NATIVE/'proposal.pkl'),
 'native_proposal_sha256':hashlib.sha256((NATIVE/'proposal.pkl').read_bytes()).hexdigest(),
 'initial_object_position_world_m':object_world_translation.tolist(),
 'initial_object_qpos_xyzw_source_stored_wxyz':object_initial_qpos.tolist(),
 'initial_object_qpos_change_max_abs':float(np.max(np.abs(data.qpos[object_adr:object_adr+7]-object_initial_qpos))),
 'initial_object_position_vs_scene_contract_error_m':float(np.linalg.norm(object_world_translation-np.array(contract['object_position_m']))),
 'nominal_palm_position_error_m':float(np.linalg.norm(data.xpos[palm_id]-desired_palm_translation)),
 'nominal_palm_rotation_error_max_abs':float(np.max(np.abs(data.xmat[palm_id].reshape(3,3)-desired_palm_rotation))),
 'cube_half_extents_m':half_extents.tolist(),
 'face_assignment':'Nearest bounded cube-face rectangle by Euclidean distance; +X,-X,+Y,-Y,+Z,-Z list order breaks edge/corner ties. Also report signed distance to its plane.',
 'native_distance_definition':'Analytic signed distance to cube surface for existing saved MANO mesh vertices; abs(distance)<=2/5mm defines proximity, negative means vertex inside cube.',
 'native_region_counts':native_counts,
 'native_total_vertices':len(vertices_object),
 'native_total_penetrating_vertices':int((signed_distance<0).sum()),
 'native_total_within_2mm_surface_vertices':int((np.abs(signed_distance)<=.002).sum()),
 'native_total_within_5mm_surface_vertices':int((np.abs(signed_distance)<=.005).sum()),
 'shadow_nominal_region_contacts':shadow_counts,
 'shadow_nominal_contacts':contacts,
 'shadow_contact_category_counts':{name:sum(c['category']==name for c in contacts) for name in ['hand_object','hand_table','hand_self','object_table','other']},
 'gross_region_comparison':{
   'native_vertex_regions_within_2mm':native2_faces,
   'native_vertex_regions_within_5mm':native5_faces,
   'native_vertex_penetration_regions':native_inside_faces,
   'shadow_nominal_compiled_hand_object_contact_regions':shadow_faces,
   'shadow_regions_shared_with_native_2mm':sorted(set(shadow_faces)&set(native2_faces)),
   'native_2mm_regions_without_nominal_shadow_contact':sorted(set(native2_faces)-set(shadow_faces)),
   'native_5mm_regions_without_nominal_shadow_contact':sorted(set(native5_faces)-set(shadow_faces)),
 },
 'interpretation_limits':'Vertex proximity/penetration and MuJoCo compiled collision contacts are different measurements and hand surfaces. Face coverage is only a coarse nominal-pose diagnostic; not equality of contact patches, fingertip normals, forces, grasp family, or native MANO physical execution. No after-lift state is used.'
}
(OUT/'hug_nominal_initial_contact_regions.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
