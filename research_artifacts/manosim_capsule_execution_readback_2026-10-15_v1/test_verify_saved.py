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
def test_manufactured_full_fixture_and_force_tamper(tmp_path):
 f=v.plan()[0];N=f['steps'];g=state();R=np.tile(np.eye(3),(16,1,1));source=dict(source_joint_anchors_world=np.zeros((1,16,3)),source_body_rotations_world=R[None]);base=tmp_path/'20';d=base/'zero_gravity_tracking_500';d.mkdir(parents=True);(d/'fixture.xml').write_text('<mujoco><option timestep="0.002" integrator="implicitfast"/></mujoco>');c=contract();ctrl=v.controls(g,c);steps=np.broadcast_to(g,(N,67)).copy();zeros=np.zeros((N,51));gain=np.zeros((51,10));gain[:,0]=30;bias=np.zeros_like(gain);bias[:,1]=-30;bias[:,2]=-1
 np.savez(d/'static_command.npz',qpos=g,ctrl=ctrl,actuator_length=ctrl,actuator_force=np.zeros(51),anchors=np.zeros((16,3)),rotations=R)
 np.savez(d/'fixture_model.npz',native_target=g,initial_state=g,plane_normal=np.zeros(3),actuator_gainprm=gain,actuator_biasprm=bias,body_mass=np.r_[0,np.full(16,.025)])
 a=dict(pre_qpos=steps,endpoint_qpos=steps.copy(),pre_qvel=zeros,endpoint_qvel=zeros.copy(),pre_time_s=np.arange(N)*.002,endpoint_time_s=(np.arange(N)+1)*.002,requested_ctrl=np.broadcast_to(ctrl,(N,51)).copy(),effective_ctrl=np.broadcast_to(ctrl,(N,51)).copy(),reference_qpos=steps.copy(),warning_counts=np.zeros((N,8),int),phase_code=(np.arange(N)*.002>=.5).astype(int),body_rotations_start=np.broadcast_to(R,(N,16,3,3)).copy(),bone_angle_error_rad=np.zeros((N,16)),start_actuator_length=np.broadcast_to(ctrl,(N,51)).copy(),start_actuator_force=zeros.copy(),start_actuator_velocity=zeros.copy(),start_qfrc_actuator=zeros.copy(),start_qfrc_passive=zeros.copy(),start_qfrc_constraint=zeros.copy(),start_qfrc_bias=zeros.copy(),actuator_power_W=np.zeros(N),passive_power_W=np.zeros(N),contact_power_W=np.zeros(N))
 np.savez(d/'trajectory.npz',**a);checks=dict(qpos=True,anchors=True,rotations=True,transmission=True,zero_servo_force=True);planned=v.controls(v.reference(f,g,g,np.zeros(3)),c);meta=dict(compiled=c,static_checks=checks,planned_branch_check=dict(samples=N+1,finite_sampled_check_only=True,ctrl_sequence_sha256=v.hashlib.sha256(planned.tobytes()).hexdigest()));(d/'command_contract.json').write_text(json.dumps(meta));flags={k:False for k in v.FLAGS};assessment=dict(status='PASS',static_source_and_transmission_checks=checks,checks=dict(wrist_tracking_or_declared_sag=True,all16_bone_tracking=True),metrics=dict(wrist_tail_max_m=0.,bone_tail_max_rad=0.,expected_wrist_world_m=g[:3].tolist()),physical_outcome=None,**flags);measurement=dict(assessment=assessment,command_clipping=False,final_qpos=g.tolist(),final_rotations=R.tolist(),physical_outcome=None,**flags);paths={p:v.bind(p) for p in d.iterdir()};result=v.fixture_check(f,base,measurement,source,g,paths);assert result['status']=='PASS'
 a['start_actuator_force'][0,0]=1;np.savez(d/'trajectory.npz',**a)
 with pytest.raises(ValueError,match='actuator force'):v.fixture_check(f,base,measurement,source,g,paths)
