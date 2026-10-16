"""Manufactured arrays/files only; no producer or engine imports."""
import importlib.util,sys,json
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).with_name('verify_saved.py');spec=importlib.util.spec_from_file_location('independent_capsule_saved_verifier',p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def contract():return dict(gears=np.vstack([np.tile([1.,0,0,0,0,0],(3,1)),np.tile(np.eye(6)[:3],(16,1))]).tolist(),actuator_ids=list(reversed(range(51))))
def state():
 q=np.zeros(67);q[:3]=[.1,-.2,.3];q[3:]=np.tile([1.,0,0,0],16);return q
def test_principal_sign_equivalence_and_named_order():
 q=state();r=.35;q[3:7]=[np.cos(r/2),0,np.sin(r/2),0];expected=np.r_[q[:3],[0,r,0],np.zeros(45)][::-1]
 np.testing.assert_allclose(v.controls(q,contract()),expected,rtol=0,atol=1e-15);q[3:]*=-1;np.testing.assert_allclose(v.controls(q,contract()),expected,rtol=0,atol=1e-15)
def test_wrapped_principal_and_pi_refusal():
 q=state();r=3.5;q[3:7]=[np.cos(r/2),np.sin(r/2),0,0];assert v.principal(q)[0,0]==pytest.approx(r-2*np.pi)
 q[3:7]=[0.,1.,0,0]
 with pytest.raises(ValueError,match='chart'):v.principal(q)
def test_quaternion_norm_refusal():
 q=state();q[3]*=1.01
 with pytest.raises(ValueError,match='norm'):v.principal(q)
def test_all16_angles_and_pi():
 a=np.tile(np.eye(3),(4,16,1,1));b=np.tile(np.eye(3),(16,1,1));np.testing.assert_array_equal(v.angles(a,b),np.zeros((4,16)));a[:,5]=np.diag([-1,-1,1]);np.testing.assert_array_equal(v.angles(a,b)[:,5],np.full(4,np.pi))
def test_point_jacobian_includes_only_ancestral_chain():
 p=np.array([[.2,-.1,.3]]);a=np.zeros((1,16,3));R=np.tile(np.eye(3),(1,16,1,1));J=v.point_jac(p,a,R);np.testing.assert_array_equal(J[0,:,:3],np.eye(3));assert np.count_nonzero(J[:,:,15:])==0
 omega=np.array([.1,-.2,.3]);np.testing.assert_allclose(J[0,:,3:6]@omega,np.cross(omega,p[0]),atol=1e-17)
def test_empty_contact_jacobian():assert v.point_jac(np.empty((0,3)),np.empty((0,16,3)),np.empty((0,16,3,3))).shape==(0,3,51)
def test_ramp_reference_endpoints():
 f=dict(kind='zero_gravity_tracking',dt_s=.002,steps=750);g=state();s=g.copy();s[:3]+=.01;r=v.reference(f,s,g,np.zeros(3));np.testing.assert_array_equal(r[0],s);np.testing.assert_array_equal(r[-1],g)
def test_strict_json_and_creation_hash(tmp_path):
 p=tmp_path/'bad.json';p.write_text('{"a":1,"a":2}')
 with pytest.raises(ValueError,match='Duplicate'):v.read(p)
 p.write_text('{"a":1e999}')
 with pytest.raises(ValueError,match='exponent'):v.read(p)
 p.write_text('{}');row=v.bind(p);v.rowcheck(row);p.write_text('[]')
 with pytest.raises(ValueError,match='hash'):v.rowcheck(row)
def test_admission_and_decision_tamper():
 x={k:False for k in v.FLAGS};x['physical_outcome']=None;v.closed(x);x['accepted_mapping']=True
 with pytest.raises(ValueError,match='Admission'):v.closed(x)
 with pytest.raises(ValueError,match='decision'):v.scalar_equal({'a':True},{'a':False},'test')
def test_nonfinite_array_and_shape_refused():
 with pytest.raises(ValueError):v.near([np.nan],[np.nan],1,'nonfinite')
 with pytest.raises(ValueError):v.near([1],[[1]],1,'shape')
def test_plan_is_exact_bounded_twelve():
 p=v.plan();assert len(p)==12 and sum(x['steps'] for x in p)==10500 and [x['index'] for x in p]==[20]*6+[28]*6
def test_manufactured_full_fixture_and_force_tamper(tmp_path,monkeypatch):
 monkeypatch.setattr(v,'compiled_snapshot_check',lambda *a,**k:dict(all_output_postconditions_passed=True))
 f=v.plan()[0];N=f['steps'];g=state();R=np.tile(np.eye(3),(16,1,1));source=dict(source_joint_anchors_world=np.zeros((1,16,3)),source_body_rotations_world=R[None]);base=tmp_path/'20';d=base/'zero_gravity_tracking_500';d.mkdir(parents=True);(d/'fixture.xml').write_text('<mujoco><option timestep="0.002" integrator="implicitfast"/></mujoco>');c=contract();ctrl=v.controls(g,c);steps=np.broadcast_to(g,(N,67)).copy();zeros=np.zeros((N,51));gain=np.zeros((51,10));gain[:,0]=30;bias=np.zeros_like(gain);bias[:,1]=-30;bias[:,2]=-1
 np.savez(d/'prevalidation_compiled_model.npz',named_body_ids=np.arange(1,17),named_joint_ids=np.arange(19),named_actuator_ids=np.arange(50,-1,-1),jnt_qposadr=np.array(list(range(3))+[3+4*j for j in range(16)]),jnt_dofadr=np.array(list(range(3))+[3+3*j for j in range(16)]));np.savez(d/'static_command.npz',qpos=g,ctrl=ctrl,actuator_length=ctrl,actuator_force=np.zeros(51),anchors=np.zeros((16,3)),rotations=R)
 np.savez(d/'fixture_model.npz',native_target=g,initial_state=g,plane_normal=np.zeros(3),actuator_gainprm=gain,actuator_biasprm=bias,body_mass=np.r_[0,np.full(16,.025)])
 a=dict(pre_qpos=steps,endpoint_qpos=steps.copy(),pre_qvel=zeros,endpoint_qvel=zeros.copy(),pre_time_s=np.arange(N)*.002,endpoint_time_s=(np.arange(N)+1)*.002,requested_ctrl=np.broadcast_to(ctrl,(N,51)).copy(),effective_ctrl=np.broadcast_to(ctrl,(N,51)).copy(),reference_qpos=steps.copy(),warning_counts=np.zeros((N,8),int),phase_code=(np.arange(N)*.002>=.5).astype(int),body_rotations_start=np.broadcast_to(R,(N,16,3,3)).copy(),bone_angle_error_rad=np.zeros((N,16)),start_actuator_length=np.broadcast_to(ctrl,(N,51)).copy(),start_actuator_force=zeros.copy(),start_actuator_velocity=zeros.copy(),start_qfrc_actuator=zeros.copy(),start_qfrc_passive=zeros.copy(),start_qfrc_constraint=zeros.copy(),start_qfrc_bias=zeros.copy(),actuator_power_W=np.zeros(N),passive_power_W=np.zeros(N),contact_power_W=np.zeros(N))
 np.savez(d/'trajectory.npz',**a);checks=dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True);planned=v.controls(v.reference(f,g,g,np.zeros(3)),c);c.update(inertia_representations=[],schema_version='manosim_authored_capsule_compiled_contract_v2',body_ids=list(range(1,17)),joint_ids=list(range(19)),qpos_addresses=list(range(3))+[3+4*j for j in range(16)],dof_addresses=list(range(3))+[3+3*j for j in range(16)]);meta=dict(compiled=c,static_checks=checks,planned_branch_check=dict(samples=N+1,finite_sampled_check_only=True,ctrl_sequence_sha256=v.hashlib.sha256(planned.tobytes()).hexdigest()));(d/'command_contract.json').write_text(json.dumps(meta));flags={k:False for k in v.FLAGS};assessment=dict(status='PASS',static_source_and_transmission_checks=checks,checks=dict(wrist_tracking_or_declared_sag=True,all16_bone_tracking=True),metrics=dict(wrist_tail_max_m=0.,bone_tail_max_rad=0.,expected_wrist_world_m=g[:3].tolist()),physical_outcome=None,**flags);measurement=dict(assessment=assessment,command_clipping=False,final_qpos=g.tolist(),final_rotations=R.tolist(),physical_outcome=None,**flags);paths={p:v.bind(p) for p in d.iterdir()};result=v.fixture_check(f,base,measurement,source,g,paths);assert result['status']=='PASS'
 a['start_actuator_force'][0,0]=1;np.savez(d/'trajectory.npz',**a)
 with pytest.raises(ValueError,match='actuator force'):v.fixture_check(f,base,measurement,source,g,paths)
def test_inertia_jacobi_stops_and_wrong_diagonal():
 eig=np.array([3e-4,2.5e-4,1.5e-4]);q=np.array([1.,0,0,0]);small=v.inertia_arithmetic(np.r_[eig,5e-13,0.,0.],eig,q);assert small['output_eigenframe_postconditions_passed']
 cosine=v.inertia_arithmetic(np.r_[eig,5e-11,0.,0.],eig,q);assert cosine['maximum_offdiagonal_kg_m2']>1e-12 and cosine['output_eigenframe_postconditions_passed']
 bad=v.inertia_arithmetic(np.r_[eig,1e-7,0.,0.],eig,q);assert not bad['output_eigenframe_postconditions_passed']
 wrong=v.inertia_arithmetic(np.r_[eig,0.,0.,0.],eig*np.array([1.001,1,1]),q);assert not wrong['output_eigenframe_postconditions_passed']
def test_inertia_rotated_frame_and_nonfinite_refusal():
 eig=np.array([3e-4,2.5e-4,1.5e-4]);q=np.array([np.cos(.2),np.sin(.2),0,0]);R=v.rotation(q);I=R@np.diag(eig)@R.T;f=np.r_[np.diag(I),I[0,1],I[0,2],I[1,2]];r=v.inertia_arithmetic(f,eig,q);assert r['output_eigenframe_postconditions_passed'] and r['maximum_reconstructed_tensor_error_kg_m2']<1e-18
 with pytest.raises(ValueError):v.inertia_arithmetic(f,eig,np.array([np.nan,0,0,0]))
def test_snapshot_source_names_and_literal_inertia(tmp_path):
 from xml.etree import ElementTree as ET
 root=ET.Element('mujoco');world=ET.SubElement(root,'worldbody');nodes=[];full=np.array([3e-4,2.5e-4,1.5e-4,0.,0.,0.]);names=[f'mano_body_{i}' for i in range(16)]
 for i,name in enumerate(names):
  node=ET.SubElement(world if i==0 else nodes[v.PARENTS[i]],'body',name=name,pos='0 0 0');ET.SubElement(node,'inertial',mass='.025',pos='0 0 0',fullinertia=' '.join(map(str,full)));nodes.append(node)
 xml=tmp_path/'fixture.xml';ET.ElementTree(root).write(xml);a=dict(source_body_names=np.array(names),named_body_ids=np.arange(1,17),source_fullinertia6=np.tile(full,(16,1)),body_mass=np.r_[0,np.full(16,.025)],body_inertia=np.vstack([np.zeros(3),np.tile(full[:3],(16,1))]),body_iquat=np.tile([1.,0,0,0],(17,1)),body_quat=np.tile([1.,0,0,0],(17,1)),body_ipos=np.zeros((17,3)),body_pos=np.zeros((17,3)),body_parentid=np.r_[0,np.array(v.PARENTS)+1]);path=tmp_path/'early.npz';np.savez(path,**a);r=v.compiled_snapshot_check(path,xml);assert r['body_count']==16 and r['all_output_postconditions_passed']
 a['source_fullinertia6'][3,0]*=2;np.savez(path,**a)
 with pytest.raises(ValueError,match='fullinertia'):v.compiled_snapshot_check(path,xml)
def test_measured_plane_rotation_cache_and_tangent_geometry():
 q=np.array([np.cos(.2),0,np.sin(.2),0]);R=v.rotation(q);n=R[:,2];C=np.array([.2,-.1,.3]);Q=np.eye(3);size=np.array([.005,.02,0]);P=C-n*(size[0]+size[1]*abs(n@Q[:,2]));args=(P,R,C,Q,size,P,R,n)
 result=v.plane_frame_math(*args,0);assert abs(result['independent_analytic_gap_m'])<1e-16
 with pytest.raises(ValueError,match='sameframe'):v.plane_frame_math(*args,2)
 with pytest.raises(ValueError,match='rotation'):v.plane_frame_math(P,np.eye(3),C,Q,size,P,R,n,0)
def test_plane_nonfinite_or_geometry_readback_refused():
 R=np.eye(3);P=np.zeros(3);C=np.array([0.,0,.025]);size=np.array([.005,.02,0]);n=R[:,2]
 with pytest.raises(ValueError,match='position'):v.plane_frame_math(P+1e-4,R,C,R,size,P,R,n,0)
 with pytest.raises(ValueError,match='Finite'):v.plane_frame_math(P,R,C*np.nan,R,size,P,R,n,0)
@pytest.mark.parametrize('present',[False,True])
def test_plane_snapshot_zero_or_one_tangent_contact(tmp_path,present):
 q=np.array([np.cos(.2),0,np.sin(.2),0]);R=v.rotation(q);n=R[:,2];C=np.array([.2,-.1,.3]);size=np.array([.005,.02,0]);P=C-n*(size[0]+size[1]*abs(n@np.array([0.,0,1.])));N=int(present);frame=np.stack([n,R[:,0],R[:,1]]).reshape(9);a=dict(snapshot_schema_version=np.asarray('manosim_capsule_actual_contact_operands_v1'),hand_geom_id=np.asarray(4),plane_geom_id=np.asarray(16),actual_geom_ids=np.array([4,16]),actual_geom_bodyid=np.array([4,0]),actual_geom_sameframe=np.array([0,0]),ncon=np.asarray(N),nefc=np.asarray(0),contact_geom=np.array([[16,4]],int) if N else np.empty((0,2),int),contact_frame=frame[None] if N else np.empty((0,9)),contact_dist=np.zeros(N),contact_dim=np.full(N,3),efc_J=np.empty((0,51)),efc_force=np.zeros(0),efc_R=np.zeros(0),efc_aref=np.zeros(0),qfrc_constraint=np.zeros(51),actual_geom_xpos=np.stack([C,P]),actual_geom_xmat=np.stack([np.eye(3),R]),actual_geom_size=np.stack([size,np.array([1.,1.,.01])]),warning_counts=np.zeros(8,int),reason=np.asarray('SECOND_EXISTING_FORWARD_PLANE_PREFLIGHT'),pre_time_s=np.asarray(0.),endpoint_time_s=np.asarray(0.),planned_plane_position=P,planned_plane_quaternion=q,planned_plane_normal=n)
 path=tmp_path/'plane_preflight.npz';np.savez(path,**a);result=v.contact_snapshot_readback(path,True);assert result['all_preflight_checks_passed'] and len(result['checks'])==7;assert result['absent_contact_checks']==('NOT_APPLICABLE' if not present else None)
 a['actual_geom_sameframe'][1]=2;np.savez(path,**a);result=v.contact_snapshot_readback(path,True);assert not result['all_preflight_checks_passed'] and not result['checks']['sameframe_NONE']
def test_saved_matrix_vector_all_coordinate_scalar_replay():
 A=np.array([[.1,.2,.3],[-.4,.5,-.6],[1.,-1.,1e-9]]);b=np.array([.7,-.8,.9]);r=v.safe_mv(A,b,'manufactured');expected=np.array([.18,-1.22,1.5000000009]);np.testing.assert_allclose(r,expected,atol=1e-15,rtol=0);assert v.SAFE_MV_RECEIPTS[-1]['all_output_coordinates_scalar_checked']==3
 with pytest.raises(ValueError,match='Finite'):v.safe_mv(A*np.nan,b,'nonfinite')
