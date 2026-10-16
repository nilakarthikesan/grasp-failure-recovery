"""Manufactured source/compiled-array controls; no real model, FK or physics."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest
from grasp_failure_prediction.evaluation import manosim_capsule as mc


def exp(a):
    a=np.asarray(a,float);angle=np.linalg.norm(a)
    return np.r_[np.cos(angle/2),a*(.5 if angle==0 else np.sin(angle/2)/angle)]

def fixture_assets(tmp_path):
    root=tmp_path/'raw';hand=root/'assets/literal_native/mano_rhand';(hand/'meshes').mkdir(parents=True)
    shape=np.arange(10,dtype=np.float32)[None];shape_path=root/'assets/mano_rhand_shape.npy';np.save(shape_path,shape)
    tree=ET.Element('mujoco');ET.SubElement(tree,'compiler',angle='radian');default=ET.SubElement(tree,'default');ET.SubElement(default,'geom',solimp='0.5 0.95 0.001',solref='0.04 1')
    asset=ET.SubElement(tree,'asset');world=ET.SubElement(tree,'worldbody');bodies=[]
    for i in range(16):
        name=f'mano_bone_{i}';ET.SubElement(asset,'mesh',name=name,file=f'meshes/{name}.obj');(hand/'meshes'/f'{name}.obj').write_text('manufactured asset bytes, never compiled')
        b=ET.SubElement(world if i==0 else bodies[mc.PARENTS[i]],'body',name=mc.BODY_NAMES[i],pos='0 0 0' if i==0 else '.01 0 0');bodies.append(b)
        if i==0:
            for k,x in enumerate('xyz'):
                axis=np.eye(3)[k];ET.SubElement(b,'joint',name='wrist_t'+x,type='slide',axis=' '.join(map(str,axis)),pos='0 0 0',limited='false',damping='5.0')
        ET.SubElement(b,'joint',name='wrist_rot' if i==0 else f'mano_joint_{i}',type='ball',pos='0 0 0',limited='false',damping='5.0' if i==0 else '.01',armature='.01' if i==0 else '.001')
        ET.SubElement(b,'inertial',pos='0 0 0',mass='.025',fullinertia='.001 .002 .003 0 0 0')
        if i==0:ET.SubElement(b,'geom',type='mesh',mesh='mano_bone_0',pos='0 0 0',priority='1',solimp='0.005 0.9 0.025 0.5 2')
        else:ET.SubElement(b,'geom',type='capsule',fromto='0 0 0 0 0 .02',size='.005')
    act=ET.SubElement(tree,'actuator')
    for k,n in enumerate(mc.ACTUATOR_NAMES):
        joint=mc.JOINT_NAMES[k] if k<3 else mc.JOINT_NAMES[3+(k-3)//3]
        gear=np.array([1.,0,0]) if k<3 else np.eye(3)[(k-3)%3]
        ET.SubElement(act,'position',name=n,joint=joint,gear=' '.join(map(str,gear)),kp=str(300. if k<3 else 50. if k<6 else 30.),dampratio=str(1.5 if 3<=k<6 else 1.))
    xml=hand/'capsule_hand.xml';xml.write_text(ET.tostring(tree,encoding='unicode'));(hand/'mesh_hand.xml').write_text('<manufactured_unused_variant/>')
    generated={str(p.relative_to(root)):mc._sha(p) for p in root.rglob('*') if p.is_file()}
    rows=[dict(path=str(root/n),sha256=h,bytes=(root/n).stat().st_size,label='manufactured') for n,h in generated.items()]
    report=root/'build_report.json';report.write_text(json.dumps(dict(schema_version='manosim_literal_native_shape_build_report_v1',status='COMPLETE',failure=None,upstream_revision=mc.UPSTREAM_REVISION,generated_root=str(root),shape_asset_sha256=mc._sha(shape_path),generated_files=generated,artifacts=rows)))
    receipt=mc.inspect_authored_assets(xml,shape_path,dict(path=str(report),sha256=mc._sha(report)),expected_shape_sha256=mc._sha(shape_path),native_shape_float32=shape)
    return receipt,shape,xml,shape_path,report


def fake_model(assets):
    m=SimpleNamespace(nq=67,nv=51,nu=51,njnt=19)
    m.body_parentid=np.r_[0,[0 if i==0 else mc.PARENTS[i]+1 for i in range(16)]]
    m.body_pos=np.vstack([np.zeros((2,3)),np.tile([.01,0,0],(15,1))]);m.body_quat=np.tile([1.,0,0,0],(17,1));m.body_mass=np.r_[0,np.full(16,.025)];m.body_inertia=np.vstack([np.zeros(3),np.tile([.001,.002,.003],(16,1))]);m.body_ipos=np.zeros((17,3));m.body_iquat=m.body_quat.copy();m.body_gravcomp=np.zeros(17)
    m.jnt_type=np.r_[np.full(3,2),np.ones(16)].astype(int);m.jnt_bodyid=np.r_[np.ones(4),np.arange(2,17)].astype(int);m.jnt_pos=np.zeros((19,3));m.jnt_axis=np.vstack([np.eye(3),np.tile([0,0,1],(16,1))]);m.jnt_limited=np.zeros(19,int)
    m.jnt_qposadr=np.r_[np.arange(3),3+4*np.arange(16)];m.jnt_dofadr=np.r_[np.arange(3),3+3*np.arange(16)];m.qpos0=np.r_[np.zeros(3),np.tile([1.,0,0,0],16)]
    m.dof_damping=np.r_[np.full(6,5.),np.full(45,.01)];m.dof_armature=np.r_[np.zeros(3),np.full(3,.01),np.full(45,.001)];m.dof_M0=np.linspace(.1,.2,51)
    m.actuator_trntype=np.zeros(51,int);m.actuator_trnid=np.full((51,2),-1,int);m.actuator_gear=np.zeros((51,6));m.actuator_gaintype=np.zeros(51,int);m.actuator_biastype=np.ones(51,int);m.actuator_dyntype=np.zeros(51,int);m.actuator_gainprm=np.zeros((51,10));m.actuator_biasprm=np.zeros((51,10));m.actuator_ctrllimited=np.zeros(51,int);m.actuator_ctrlrange=np.zeros((51,2));m.actuator_forcelimited=np.zeros(51,int);m.actuator_forcerange=np.zeros((51,2));m.actuator_actrange=np.zeros((51,2))
    # Reverse actuator order proves named scattering, not positional flattening.
    for k,a in enumerate(assets['actuators']):
        aid=50-k;jid=mc.JOINT_NAMES.index(a['joint']);m.actuator_trnid[aid,0]=jid;m.actuator_gear[aid]=a['gear'];m.actuator_gainprm[aid,0]=a['kp'];m.actuator_biasprm[aid,1]=-a['kp'];dof=int(m.jnt_dofadr[jid])+(0 if k<3 else (k-3)%3);m.actuator_biasprm[aid,2]=-2*a['dampratio']*np.sqrt(a['kp']*m.dof_M0[dof])
    m.geom_type=np.r_[0,7,np.full(15,3)].astype(int);m.geom_bodyid=np.r_[0,np.arange(1,17)];m.geom_pos=np.vstack([np.zeros((2,3)),np.tile([0,0,.01],(15,1))]);m.geom_quat=np.tile([1.,0,0,0],(17,1));m.geom_size=np.vstack([np.zeros((2,3)),np.tile([.005,.01,0],(15,1))]);m.geom_dataid=np.r_[-1,0,np.full(15,-1)].astype(int);m.mesh_vert=np.zeros((4,3));m.mesh_face=np.array([[0,1,2]],int);m.mesh_pos=np.zeros((1,3));m.mesh_quat=np.array([[1.,0,0,0]]);m.geom_solref=np.tile([.04,1.],(17,1));m.geom_solimp=np.tile([.5,.95,.001,.5,2.],(17,1));m.geom_friction=np.tile([1.,.005,.0001],(17,1));m.geom_priority=np.zeros(17,int);m.geom_contype=np.ones(17,int);m.geom_conaffinity=np.ones(17,int)
    def resolve(kind,name):
        if kind=='body':return mc.BODY_NAMES.index(name)+1
        if kind=='joint':return mc.JOINT_NAMES.index(name)
        if kind=='actuator':return 50-mc.ACTUATOR_NAMES.index(name)
        if kind=='mesh' and name=='mano_bone_0':return 0
        raise KeyError(name)
    return m,resolve

@pytest.fixture
def compiled(tmp_path):
    assets,shape,*_=fixture_assets(tmp_path);m,resolve=fake_model(assets);rest=np.tile([1.,0,0,0],(16,1))
    return m,resolve,assets,rest,shape


def test_exact_authored_asset_inventory_is_durable_and_not_full_archived_claim(tmp_path):
    a,shape,xml,path,report=fixture_assets(tmp_path)
    assert len(a['files'])==19 and a['archived_temporary_dependency_closure_live_verified'] is False
    assert a['default_contact']==dict(solimp='0.5 0.95 0.001',solref='0.04 1')
    assert len(a['actuators'])==51 and a['shape_array_digest']==mc._digest(shape)
    assert all(a[k] is False for k in mc.CLOSED_FLAGS)
    (xml.parent/'meshes/mano_bone_5.obj').write_bytes(b'mutated')
    with pytest.raises(ValueError,match='write-time'):mc.inspect_authored_assets(xml,path,dict(path=str(report),sha256=mc._sha(report)),expected_shape_sha256=mc._sha(path),native_shape_float32=shape)

@pytest.mark.parametrize('change',['dtype','flat','beta','hash'])
def test_literal_shape_contract_rejects_substitution(tmp_path,change):
    a,shape,xml,path,report=fixture_assets(tmp_path);expected=mc._sha(path)
    if change=='dtype':shape=shape.astype(float)
    elif change=='flat':shape=shape.reshape(10)
    elif change=='beta':shape=shape.copy();shape[0,1]+=1
    else:expected='0'*64
    with pytest.raises(ValueError):mc.inspect_authored_assets(xml,path,dict(path=str(report),sha256=mc._sha(report)),expected_shape_sha256=expected,native_shape_float32=shape)


def test_named_contract_accepts_world_plane_and_command_scatter_preserves_source(compiled):
    m,resolve,a,rest,shape=compiled;c=mc.validate_compiled_contract(m,resolve,a,rest)
    aa=np.linspace(.03,.7,48,dtype=np.float32);t=np.array([.125,-.25,.5],np.float32);camera=np.eye(4);source_before=[x.copy() for x in (aa,t,camera,rest,shape)]
    target=mc.target_from_source(aa,t,camera,rest,shape_float32=shape);q=target.qpos.copy();cmd=mc.command_from_target(target,c)
    expected=np.r_[t.astype(float),aa.astype(float)]
    np.testing.assert_allclose(cmd.ctrl[list(c.actuator_ids)],expected,rtol=0,atol=8*np.finfo(float).eps)
    assert c.actuator_ids==tuple(range(50,-1,-1)) and m.geom_type[0]==0
    assert not cmd.ctrl.flags.writeable and not target.qpos.flags.writeable and not cmd.principal_rotvecs.flags.writeable
    for before,after in zip(source_before,(aa,t,camera,rest,shape)):np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(target.qpos,q);assert mc.verify_compiled_contract(m,c)
    assert all(cmd.metadata()[k] is False for k in mc.CLOSED_FLAGS)

@pytest.mark.parametrize('fault',['radius','endpoint','bodypos','anchor','inertia','mass','parent','gear','gain','damping','forceclamp','ctrlclamp','address','jointowner','rest','nonfinite'])
def test_initial_altered_compiled_model_refused_not_merely_fingerprinted(compiled,fault):
    m,res,a,rest,shape=compiled
    if fault=='radius':m.geom_size[3,0]*=1.1
    elif fault=='endpoint':m.geom_pos[3,2]+=.001
    elif fault=='bodypos':m.body_pos[4,0]+=.001
    elif fault=='anchor':m.jnt_pos[4,0]+=.001
    elif fault=='inertia':m.body_inertia[6,0]*=1.1
    elif fault=='mass':m.body_mass[7]*=1.1
    elif fault=='parent':m.body_parentid[5]=0
    elif fault=='gear':m.actuator_gear[0,0]+=.1
    elif fault=='gain':m.actuator_gainprm[0,0]+=1
    elif fault=='damping':m.actuator_biasprm[0,2]*=1.1
    elif fault=='forceclamp':m.actuator_forcelimited[0]=1
    elif fault=='ctrlclamp':m.actuator_ctrllimited[0]=1
    elif fault=='address':m.jnt_qposadr[4]+=1
    elif fault=='jointowner':m.jnt_bodyid[4]=1
    elif fault=='rest':m.body_quat[3]=exp([.01,0,0])
    else:m.body_inertia[5,0]=np.nan
    with pytest.raises(ValueError):mc.validate_compiled_contract(m,res,a,rest)


def test_contract_detects_later_model_and_mesh_mutation(compiled):
    m,res,a,rest,shape=compiled;c=mc.validate_compiled_contract(m,res,a,rest);m.mesh_vert[0,0]=1.
    with pytest.raises(ValueError,match='changed'):mc.verify_compiled_contract(m,c)

@pytest.mark.parametrize('angle',[0.,.0001,1.2,3.7,6.1,2*np.pi])
def test_principal_control_matches_axisangle_wrapping_and_antipodal_quats(angle):
    q=np.tile([1.,0,0,0],(16,1));q[3]=exp([angle,0,0]);before=q.copy();v,margin,wrapped,_=mc.principal_ball_coordinates(q)
    expected=angle if angle<=np.pi else angle-2*np.pi
    np.testing.assert_allclose(v[3],[expected,0,0],rtol=0,atol=8*np.finfo(float).eps)
    other=mc.principal_ball_coordinates(-q)[0];np.testing.assert_allclose(v,other,rtol=0,atol=8*np.finfo(float).eps)
    np.testing.assert_array_equal(q,before)


def test_pi_guard_and_principal_jump_refuse_without_clipping(compiled):
    m,res,a,rest,shape=compiled;c=mc.validate_compiled_contract(m,res,a,rest)
    q=m.qpos0.copy();q[7:11]=exp([np.pi-1e-7,0,0])
    with pytest.raises(ValueError,match='pi'):mc.command_from_qpos(q,c)
    q[7:11]=exp([3.,0,0]);b=q.copy();b[7:11]=exp([3.3,0,0])
    with pytest.raises(ValueError,match='branch jump'):mc.require_continuous_principal_targets(np.array([q,b]),c)
    assert len(mc.require_continuous_principal_targets(np.array([q,q]),c))==2
    q[3]=.5
    with pytest.raises(ValueError,match='Unit'):mc.command_from_qpos(q,c)


def test_wrong_shape_or_rest_cannot_be_used_for_target_command(compiled):
    m,res,a,rest,shape=compiled;c=mc.validate_compiled_contract(m,res,a,rest);aa=np.linspace(.01,.1,48,dtype=np.float32);t=np.zeros(3,np.float32)
    target=mc.target_from_source(aa,t,np.eye(4),rest,shape_float32=shape+np.float32(1))
    with pytest.raises(ValueError,match='shape/rest'):mc.command_from_target(target,c)


def test_compiled_palm_frame_includes_nonidentity_mesh_centering_and_rotation(compiled):
    m,res,a,rest,shape=compiled
    m.mesh_pos[0]=[.003,-.002,.001];m.mesh_quat[0]=exp([.13,-.2,.07])
    m.geom_pos[1]=m.mesh_pos[0];m.geom_quat[1]=m.mesh_quat[0]
    c=mc.validate_compiled_contract(m,res,a,rest)
    assert mc.verify_compiled_contract(m,c)
    m.geom_pos[1,0]+=.001
    with pytest.raises(ValueError,match='palm mesh frame'):mc.validate_compiled_contract(m,res,a,rest)
    m.geom_pos[1]=m.mesh_pos[0];m.geom_quat[1]=exp([.13,-.2,.08])
    with pytest.raises(ValueError,match='palm mesh frame'):mc.validate_compiled_contract(m,res,a,rest)
    m.geom_quat[1]=m.mesh_quat[0];m.mesh_pos[0,0]+=.001
    with pytest.raises(ValueError,match='changed'):mc.verify_compiled_contract(m,c)


def test_compiled_and_caller_rest_cannot_collude_against_authored_identity(compiled):
    m,res,a,rest,shape=compiled
    m.body_quat[3]=exp([.01,0,0]);rest[2]=m.body_quat[3]
    with pytest.raises(ValueError,match='Authored body rest'):mc.validate_compiled_contract(m,res,a,rest)


@pytest.mark.parametrize('field',['geom_solref','geom_solimp','geom_friction','geom_priority','geom_contype','geom_conaffinity','body_gravcomp','actuator_actrange'])
def test_contract_fingerprints_contact_compensation_and_activation_fields(compiled,field):
    m,res,a,rest,shape=compiled;c=mc.validate_compiled_contract(m,res,a,rest)
    value=getattr(m,field);value.flat[0]+=1
    with pytest.raises(ValueError,match='changed'):mc.verify_compiled_contract(m,c)


def test_pinned_jacobi_absolute_and_cosine_stop_residuals_are_not_machine_zero():
    for diagonal,b in [(np.array([3e-6,2.5e-6,1.5e-6]),5e-13),(np.array([3e-4,2.5e-4,1.5e-4]),5e-11)]:
        full=np.r_[diagonal,b,0.,0.]
        receipt=mc.inertia_representation_receipt(full,diagonal,np.array([1.,0,0,0]))
        assert receipt['maximum_reconstructed_tensor_error_kg_m2']>128*np.finfo(float).eps*diagonal.max()
        assert receipt['maximum_offdiagonal_kg_m2']<=receipt['jacobi_offdiagonal_bound_kg_m2']
        if b>mc.JACOBI_EPS:
            tau=(diagonal[1]-diagonal[0])/(2*b);t=-1/(-tau+np.sqrt(1+tau*tau));c=1/np.sqrt(1+t*t)
            assert c>1-mc.JACOBI_EPS # Exactly the pinned compiler's alternative stop.
        assert receipt['actual_compiler_iteration_count'] is None and receipt['source_inertia_changed'] is False


def test_rotated_inertia_uses_its_compiled_eigenframe_without_changing_tensor():
    principal=np.array([3e-4,2.5e-4,1.5e-4]);q=exp([.3,-.2,.15]);R=mc._rotation(q)
    I=np.einsum('ik,k,jk->ij',R,principal,R,optimize=False);full=np.r_[np.diag(I),I[0,1],I[0,2],I[1,2]]
    r=mc.inertia_representation_receipt(full,principal,q)
    assert r['diagonal_error_kg_m2']<=r['diagonal_arithmetic_bound_kg_m2']
    assert r['maximum_reconstructed_tensor_error_kg_m2']<1e-18


def test_inertia_wrong_diagonal_and_beyond_jacobi_envelope_refused():
    principal=np.array([3e-4,2.5e-4,1.5e-4]);q=np.array([1.,0,0,0])
    with pytest.raises(ValueError,match='principal diagonal'):mc.inertia_representation_receipt(np.r_[principal,0.,0.,0.],principal*np.array([1.1,1,1]),q)
    with pytest.raises(ValueError,match='Jacobi stop envelope'):mc.inertia_representation_receipt(np.r_[principal,1e-7,0.,0.],principal,q)
