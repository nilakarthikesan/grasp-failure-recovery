"""Manufactured arrays/XML only; no engine, native skin or saved state reads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_read_q_bridge',HERE/'skin_engine_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)


def fixture():
    faces=np.array([[0,1,2]],dtype=np.int64)
    model=SimpleNamespace(nflex=1,flex_dim=np.array([2]),flex_interp=np.array([0]),flex_centered=np.array([0]),
        flex_vertadr=np.array([0]),flex_vertnum=np.array([3]),flex_elemnum=np.array([1]),
        flex_elemdataadr=np.array([0]),flex_elem=faces.ravel().copy(),flex_vertbodyid=np.array([1,1,1]),
        flex_vert=np.zeros((3,3)),geom_contype=np.zeros(16,dtype=int),geom_conaffinity=np.zeros(16,dtype=int),
        flex_contype=np.zeros(1,dtype=int),flex_conaffinity=np.zeros(1,dtype=int),body_mass=np.array([0.,.4]),
        opt=SimpleNamespace(disableflags=3,gravity=np.array([0.,0.,-9.81]),timestep=.002,iterations=100))
    angle=.4;c=np.cos(angle);s=np.sin(angle);r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    data=SimpleNamespace(qpos=np.zeros(67),qvel=np.zeros(51),ctrl=np.zeros(51),time=0.,
        xpos=np.array([[0.,0.,0.],[.2,-.1,.3]]),xmat=np.vstack((np.eye(3).ravel(),r.ravel())),
        flexvert_xpos=np.zeros((3,3)),qfrc_applied=np.zeros(51),xfrc_applied=np.zeros((2,6)),
        act=np.zeros(0),mocap_pos=np.zeros((0,3)),mocap_quat=np.zeros((0,4)),userdata=np.zeros(1))
    for name in ('qacc','qacc_smooth','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint',
        'qfrc_smooth','qfrc_inverse','qfrc_damper','qfrc_spring','qfrc_gravcomp','actuator_force'):
        setattr(data,name,np.zeros(51))
    calls=[];inputs=[]
    def evaluate(q):
        inputs.append(q.copy())
        return dict(vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])+q[:3],
            joint_anchors_world=np.zeros((16,3)),absolute_rotations=np.tile(np.eye(3),(16,1,1)))
    def flex(m,d):d.flexvert_xpos[:]=m.flex_vert@d.xmat[1].reshape(3,3).T+d.xpos[1]
    def make(evaluator=evaluate,update=flex):
        return bridge.ArticulatedSkinUpdater(model,data,owner_body_id=1,flex_id=0,faces=faces,
            evaluate_skin=evaluator,flex_update=update,before_skin=lambda:calls.append('skin'),
            before_flex=lambda:calls.append('flex'),contact_disable_bit=1,actuation_disable_bit=2)
    return model,data,make,calls,inputs,evaluate,flex


def test_read_q_refresh_changes_only_private_flex_vertices_preserving_all_inputs():
    model,data,make,calls,inputs,_,_=fixture();updater=make()
    immutable=bridge.model_array_hashes(model);q=data.qpos.copy();ctrl=data.ctrl.copy();qvel=data.qvel.copy()
    first=updater.update()
    np.testing.assert_allclose(first['compiled_vertices_world_m'],np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]]),rtol=0,atol=1e-16)
    assert bridge.model_array_hashes(model)==immutable
    np.testing.assert_array_equal(data.qpos,q);np.testing.assert_array_equal(data.qvel,qvel);np.testing.assert_array_equal(data.ctrl,ctrl)
    data.qpos[:3]=[.04,-.02,.01] # Only test fixture sets another declared state.
    second=updater.update()
    np.testing.assert_allclose(second['compiled_vertices_world_m']-first['compiled_vertices_world_m'],np.tile([.04,-.02,.01],(3,1)),rtol=0,atol=1e-16)
    assert calls==['skin','flex','skin','flex'] and inputs[1][0]==.04
    assert second['protected_state_unchanged'] and second['only_model_flex_vert_changed']
    np.testing.assert_array_equal(data.qfrc_applied,np.zeros(51));np.testing.assert_array_equal(data.xfrc_applied,np.zeros((2,6)))


@pytest.mark.parametrize('field',['qpos','qvel','ctrl','qfrc_applied','xfrc_applied','userdata','time','body_mass','gravity','timestep','iterations',
    'qacc','qacc_smooth','qfrc_bias','qfrc_passive','qfrc_actuator','qfrc_constraint','qfrc_smooth','qfrc_inverse',
    'qfrc_damper','qfrc_spring','qfrc_gravcomp','actuator_force'])
def test_any_state_force_or_other_model_option_mutation_refuses(field):
    model,data,make,_,_,_,flex=fixture()
    def mutation(m,d):
        flex(m,d)
        if field=='time':d.time=.01
        elif field=='body_mass':m.body_mass[1]+=.01
        elif field=='gravity':m.opt.gravity[0]=1.
        elif field in ('timestep','iterations'):setattr(m.opt,field,getattr(m.opt,field)+1)
        else:getattr(d,field).flat[0]=1.
    updater=make(update=mutation)
    with pytest.raises(ValueError):updater.update()


@pytest.mark.parametrize('defect',['oldgeom','oldaffinity','flexmask','flexaffinity','actuation','contact','centered','owner','faces','interp'])
def test_unsafe_stock_contact_or_topology_conditions_are_refused(defect):
    m,d,make,_,_,_,_=fixture()
    if defect=='oldgeom':m.geom_contype[0]=1
    elif defect=='oldaffinity':m.geom_conaffinity[1]=1
    elif defect=='flexmask':m.flex_contype[0]=1
    elif defect=='flexaffinity':m.flex_conaffinity[0]=1
    elif defect=='actuation':m.opt.disableflags=1
    elif defect=='contact':m.opt.disableflags=2
    elif defect=='centered':m.flex_centered[0]=1
    elif defect=='owner':m.flex_vertbodyid[0]=0
    elif defect=='faces':m.flex_elem[0]=2
    else:m.flex_interp[0]=1
    with pytest.raises(ValueError):make()


@pytest.mark.parametrize('defect',['owner_position','owner_rotation','skin_vertex','skin_anchor','skin_rotation'])
def test_nonfinite_geometry_refuses_before_comparison(defect):
    m,d,make,_,_,evaluate,_=fixture()
    if defect=='owner_position':d.xpos[1,0]=np.nan
    elif defect=='owner_rotation':d.xmat[1,0]=np.nan
    def bad(q):
        out=evaluate(q)
        if defect=='skin_vertex':out['vertices'][0,0]=np.nan
        elif defect=='skin_anchor':out['joint_anchors_world'][0,0]=np.inf
        elif defect=='skin_rotation':out['absolute_rotations'][0,0,0]=np.nan
        return out
    updater=make(evaluator=bad)
    with pytest.raises(ValueError):updater.update()


def test_before_work_skin_and_flex_quota_refusals_do_not_execute_call():
    m,d,make,calls,inputs,_,_=fixture();u=make()
    def exhausted():raise RuntimeError('quota')
    u.before_skin=exhausted
    with pytest.raises(RuntimeError,match='quota'):u.update()
    assert inputs==[] and calls==[]
    u.before_skin=lambda:None;u.before_flex=exhausted
    before=d.flexvert_xpos.copy()
    with pytest.raises(RuntimeError,match='quota'):u.update()
    np.testing.assert_array_equal(d.flexvert_xpos,before)


def test_corresponding_face_normal_winding_and_degenerate_refusal():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    a,_=bridge.oriented_face_normals(v,np.array([[0,1,2]]));b,_=bridge.oriented_face_normals(v,np.array([[0,2,1]]))
    np.testing.assert_array_equal(a,np.array([[0.,0.,1.]]));np.testing.assert_array_equal(b,-a)
    with pytest.raises(ValueError):bridge.oriented_face_normals(v,np.array([[0,0,2]]))


def test_scene_copy_preserves_skeleton_inertials_and_disables_every_contact(tmp_path):
    root=ET.Element('mujoco');assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
    for j in range(16):
        mesh=tmp_path/f'{j}.obj';mesh.write_text('manufactured placeholder; never compiled')
        ET.SubElement(assets,'mesh',name=f'm{j}',file=mesh.name)
        body=ET.SubElement(world,'body',name=f'mano_body_{j}',pos=f'{j} 0 0')
        ET.SubElement(body,'joint',name=f'j{j}',type='ball');ET.SubElement(body,'inertial',mass='.025',pos='0 0 0',diaginertia='.001 .002 .003')
        ET.SubElement(body,'geom',type='mesh',mesh=f'm{j}')
        ET.SubElement(act,'motor',joint=f'j{j}',gear='1')
    base=tmp_path/'hand.xml';ET.ElementTree(root).write(base)
    include=ET.Element('mujocoinclude');deform=ET.SubElement(include,'deformable')
    f=ET.SubElement(deform,'flex',name='frozen',dim='2',body=bridge.OWNER,radius='0',vertex=' '.join(['0']*(778*3)),element=' '.join(['0 1 2']*1552))
    ET.SubElement(f,'contact',contype='1',conaffinity='1');inc=tmp_path/'skin.xml';ET.ElementTree(include).write(inc)
    text,receipt=bridge.scene_xml(base,inc);new=ET.fromstring(text)
    for tag in ('joint','inertial','motor'):
        assert [ET.tostring(x) for x in root.findall('.//'+tag)]==[ET.tostring(x) for x in new.findall('.//'+tag)]
    assert [b.attrib for b in root.findall('.//body')]==[b.attrib for b in new.findall('.//body')]
    assert all(g.get('contype')==g.get('conaffinity')=='0' for g in new.findall('.//geom'))
    assert new.find('./deformable/flex/contact').get('contype')=='0'
    assert new.find('./option/flag').attrib=={'contact':'disable','actuation':'disable'}
    assert receipt['original_joint_inertial_actuator_parameters_unchanged']
    assert receipt['stock_flex_contact_forces_permitted'] is False
