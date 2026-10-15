"""Geometry-only read-q skin updater; no engine/model construction at import.

Only model.flex_vert may change. Stock hand and flex contacts and actuation
must be disabled. This is not a stock-flex Jacobian or physical-force repair.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

SCHEMA='manosim_read_q_complete_skin_geometry_bridge_v1'
OWNER='mano_body_0'
FLEX='complete_mano_read_q_geometry_only_v1'


def digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def finite(a,shape,name):
    a=np.asarray(a)
    if a.shape!=shape or a.dtype!=np.dtype('float64') or not np.isfinite(a).all():
        raise ValueError('Finite explicitfloat64 '+name+' '+str(shape)+' required')
    return a


def rotation(r):
    r=finite(r,(3,3),'actual owner rotation')
    if (np.max(np.abs(np.einsum('ji,jk->ik',r,r,optimize=False)-np.eye(3)))>1e-12
            or abs(float(np.linalg.det(r))-1)>1e-12):
        raise ValueError('Actual proper owner rotation required; no projection')
    return r


def oriented_face_normals(vertices,faces):
    v=np.asarray(vertices);f=np.asarray(faces)
    if (v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3
            or f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=len(v))):
        raise ValueError('Finite vertices and literal complete triangle IDs required')
    tri=v[f];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0]
    cross=np.cross(a,b);area2=np.sqrt(np.sum(cross*cross,axis=1))
    if np.any(area2<=1e-15):raise ValueError('Degenerate face refused without deletion')
    normals=cross/area2[:,None]
    # Diagnostic sensitivity scale only, never an acceptance allowance.
    edge=np.maximum(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1))
    return normals,dict(minimum_double_area_m2=float(area2.min()),
        maximum_edge_over_double_area_per_m=float(np.max(edge/area2)))


def model_array_hashes(model):
    result={}
    for name in dir(model):
        if name.startswith('_') or name=='flex_vert':continue
        value=getattr(model,name)
        if isinstance(value,np.ndarray):result[name]=digest(value)
    options={}
    for name in dir(model.opt):
        if name.startswith('_'):continue
        value=getattr(model.opt,name)
        if callable(value):continue
        if isinstance(value,np.ndarray):options[name]=dict(array_sha256=digest(value))
        elif isinstance(value,(str,bool,int,float,np.generic)):
            options[name]=value.item() if isinstance(value,np.generic) else value
        elif hasattr(value,'value'):options[name]=int(value.value)
        else:raise ValueError('Unaccounted model option '+name)
    result['__all_model_options__']=hashlib.sha256(json.dumps(options,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result


def scene_xml(base_path,old_include_path):
    """Literal original skeleton/inertials retained; contact+actuation off."""
    base_path=Path(base_path).resolve();root=ET.parse(base_path).getroot()
    if root.tag!='mujoco' or root.findall('.//include') or root.findall('.//deformable'):
        raise ValueError('Original standalone generated hand XML required')
    meshes=root.findall('./asset/mesh');geoms=root.findall('.//geom')
    bodies=root.findall('.//body')
    if len(meshes)!=16 or len(geoms)!=16 or {b.get('name') for b in bodies}!={f'mano_body_{j}' for j in range(16)}:
        raise ValueError('Exact original16 mesh/body inventory required')
    protected=lambda tree: [ET.tostring(node,encoding='unicode') for node in tree.iter()
        if node.tag in ('joint','inertial','motor','position','velocity','general')]
    original_protected=protected(root)
    for mesh in meshes:
        p=(base_path.parent/mesh.get('file')).resolve()
        if not p.is_file():raise ValueError('Original bound visual mesh missing')
        mesh.set('file',str(p))
    for geom in geoms:geom.set('contype','0');geom.set('conaffinity','0')
    include=ET.parse(old_include_path).getroot()
    oldflex=include.findall('./deformable/flex')
    if include.tag!='mujocoinclude' or len(oldflex)!=1:raise ValueError('One complete frozen skin flex required')
    flex=oldflex[0]
    if flex.get('dim')!='2' or flex.get('body')!=OWNER or float(flex.get('radius'))!=0:
        raise ValueError('Frozen complete single-owner zero-radius reference required')
    vertices=np.fromstring(flex.get('vertex'),sep=' ');faces=np.fromstring(flex.get('element'),sep=' ',dtype=np.int64)
    if vertices.shape!=(778*3,) or faces.shape!=(1552*3,):raise ValueError('All778vertices/1552faces required')
    flex.set('name',FLEX)
    contact=flex.find('contact')
    if contact is None:raise ValueError('Explicit frozen flex contact parameters required')
    contact.set('contype','0');contact.set('conaffinity','0')
    deformable=ET.SubElement(root,'deformable');deformable.append(flex)
    option=root.find('option')
    if option is None:option=ET.SubElement(root,'option')
    flag=option.find('flag')
    if flag is None:flag=ET.SubElement(option,'flag')
    flag.set('contact','disable');flag.set('actuation','disable')
    if original_protected!=protected(root):raise ValueError('Skeleton/inertial/actuator parameter changed')
    return ET.tostring(root,encoding='unicode')+'\n',dict(
        condition='contact_and_actuation_disabled_read_q_skin_geometry_only_v1',
        old_geom_collision_masks_disabled=16,added_flex_collision_masks_disabled=1,
        original_joint_inertial_actuator_elements_sha256=hashlib.sha256('\n'.join(original_protected).encode()).hexdigest(),
        original_joint_inertial_actuator_parameters_unchanged=True,
        stock_flex_contact_forces_permitted=False,physical_dynamics_qualified=False)


class ArticulatedSkinUpdater:
    """Read-only state inputs; only the private model's flex_vert is writable.

    Fixture must finish its allowed mj_forward first. evaluate_skin receives
    a copy of current namedqpos and returns vertices,anchors,absolute rotations.
    The supplied flex_update is the allowed direct mj_flex call, never mj_step.
    """
    def __init__(self,model,data,*,owner_body_id,flex_id,faces,evaluate_skin,flex_update,
                 before_skin,before_flex,contact_disable_bit,actuation_disable_bit):
        self.model=model;self.data=data;self.owner=int(owner_body_id);self.flex=int(flex_id)
        self.faces=np.array(faces,copy=True);self.evaluate=evaluate_skin;self.flex_update=flex_update
        self.before_skin=before_skin;self.before_flex=before_flex
        if (model.nflex!=1 or self.flex!=0 or int(model.flex_dim[0])!=2
            or int(model.flex_interp[0])!=0 or bool(model.flex_centered[0])):
            raise ValueError('One noncentered linear triangle flex required')
        self.start=int(model.flex_vertadr[0]);self.n=int(model.flex_vertnum[0])
        if self.n<3 or self.start!=0 or self.faces.shape!=(int(model.flex_elemnum[0]),3):
            raise ValueError('Complete literal flex inventory required')
        elementstart=int(model.flex_elemdataadr[0])
        if not np.array_equal(model.flex_elem[elementstart:elementstart+len(self.faces)*3].reshape(-1,3),self.faces):
            raise ValueError('Compiled face order differs')
        if (not np.all(model.flex_vertbodyid==self.owner) or np.any(model.geom_contype)
            or np.any(model.geom_conaffinity) or np.any(model.flex_contype) or np.any(model.flex_conaffinity)):
            raise ValueError('All old/new hand collision masks must be zero')
        bits=int(model.opt.disableflags)
        if bits&int(contact_disable_bit)==0 or bits&int(actuation_disable_bit)==0:
            raise ValueError('Distinct contact/actuation disabled condition required')
        self.immutable=model_array_hashes(model)
        self.disableflags=bits

    def update(self):
        m,d=self.model,self.data
        protected=('qpos','qvel','ctrl','qfrc_applied','xfrc_applied','act','mocap_pos','mocap_quat','userdata',
            'qacc','qacc_smooth','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint',
            'qfrc_smooth','qfrc_inverse','qfrc_damper','qfrc_spring','qfrc_gravcomp','actuator_force')
        saved={name:np.array(getattr(d,name),copy=True) for name in protected if hasattr(d,name)}
        stamp=float(d.time)
        q=finite(saved['qpos'],(67,),'actual read-only namedqpos')
        self.before_skin();state=self.evaluate(q.copy())
        world=finite(state['vertices'],(self.n,3),'complete evaluated skin')
        finite(state['joint_anchors_world'],(16,3),'source skin joint anchors')
        finite(state['absolute_rotations'],(16,3,3),'source skin world rotations')
        r=rotation(d.xmat[self.owner].reshape(3,3));p=finite(d.xpos[self.owner],(3,),'actual owner position')
        local=np.einsum('ij,kj->ki',r.T,world-p,optimize=False)
        m.flex_vert[self.start:self.start+self.n]=local
        self.before_flex();self.flex_update(m,d)
        actual=finite(d.flexvert_xpos[self.start:self.start+self.n],(self.n,3),'actual compiled skin readback').copy()
        for name,value in saved.items():
            if not np.array_equal(getattr(d,name),value):raise ValueError('Updater changed protected state '+name)
        if float(d.time)!=stamp:raise ValueError('Updater changed time')
        if model_array_hashes(m)!=self.immutable or int(m.opt.disableflags)!=self.disableflags:
            raise ValueError('Updater changed another model array/condition')
        normal,condition=oriented_face_normals(actual,self.faces)
        return dict(skin_state=state,compiled_vertices_world_m=actual,compiled_face_normals_world=normal,
            actual_owner_rotation=r.copy(),actual_owner_position=p.copy(),local_vertices_m=local.copy(),
            triangle_condition_diagnostic=condition,protected_state_unchanged=True,
            only_model_flex_vert_changed=True)
