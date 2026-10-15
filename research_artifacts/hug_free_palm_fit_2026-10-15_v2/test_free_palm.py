"""Manufactured static geometry/solver controls; no saved HUG scene input."""
import gzip,json
import numpy as np
import pytest
import mujoco
from free_palm_kinematics import *
from free_palm_solver import FreePalmConfig,solve_free_palm
from grasp_failure_prediction.evaluation.constrained_surface_refinement import ConstrainedEvaluation

def toy():
    branches=[]
    for i in range(5):
        branch=''
        for j in range(4):
            branch+=f'<body name="f{i}_{j}" pos="0.007 0.009 0.021"><joint name="q{2+i*4+j}" type="hinge" axis="{1 if j%3==0 else 0} {1 if j%3==1 else 0} {1 if j%3==2 else 0}"/><geom type="sphere" size=".003" mass=".01"/>'
        branches.append(branch+'</body>'*4)
    xml='<mujoco><compiler angle="radian"/><worldbody><body name="forearm"><freejoint name="root"/><geom type="sphere" size=".01" mass=".1"/><body pos=".02 .03 .04"><joint name="q0" axis="1 0 0"/><geom type="sphere" size=".01" mass=".1"/><body name="palm" pos=".01 .02 .03"><joint name="q1" axis="0 1 0"/><geom type="sphere" size=".01" mass=".1"/>'+''.join(branches)+'<body><joint name="q22"/><geom type="sphere" size=".003" mass=".01"/><body><joint name="q23"/><geom type="sphere" size=".003" mass=".01"/></body></body></body></body></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'q{j}') for j in range(24)]
    adr=np.array([m.jnt_qposadr[j] for j in ids]);dofs=np.array([m.jnt_dofadr[j] for j in ids]);q0=np.sin(np.arange(24))*.1
    rootp=np.array([.14,-.2,.7]);rootq=np.array([.8,.3,-.4,.2]);rootq/=np.linalg.norm(rootq)
    d.qpos[:3]=rootp;d.qpos[3:7]=rootq;d.qpos[adr]=q0;mujoco.mj_forward(m,d)
    palm=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'palm');anchor=d.xpos[palm].copy();nominal=d.qpos.copy()
    bodies=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'f{i}_3') for i in range(5)])
    offset=np.array([.004,-.003,.023])
    def stage(x):stage_translated_palm(m,d,nominal,adr,0,palm,rootp,rootq,anchor,q0,x)
    def points():return np.array([d.xpos[b]+d.xmat[b].reshape(3,3)@offset for b in bodies])
    targets=points()+np.array([.001,.002,-.001])
    def errors(x):stage(x);return points()-targets
    def jac(x):stage(x);return virtual_tip_jacobian(m,d,palm,points(),bodies,dofs)
    return m,d,stage,points,errors,jac,rootq,anchor,nominal,adr,q0,palm,bodies,dofs

def test_actual_virtual_tip_jacobian_and_floating_root_translation():
    m,d,stage,points,errors,jac,rootq,anchor,nominal,adr,q0,palm,bodies,dofs=toy()
    x=np.sin(np.arange(27))*.4;x[24:]=[.7,-.6,.9]
    gate=verify_jacobian(x,np.tile([-2.,2.],(27,1)),errors,jac)
    assert gate['passed'] and gate['maxabs_difference_m_per_scaled_state']<1e-9
    stage(x)
    assert np.array_equal(d.qpos[3:7],rootq)
    assert np.linalg.norm(d.xpos[palm]-(anchor+.005*x[24:]))<1e-15
    assert np.allclose(jac(x)[:,:,24:],np.broadcast_to(np.eye(3)*.005,(5,3,3)),rtol=0,atol=0)
    stage(np.zeros(27));assert np.array_equal(d.qpos,nominal)
    # Omitting virtual tip offset has a detectably different rotational Jacobian.
    wrong=virtual_tip_jacobian(m,d,palm,np.array([d.xpos[b] for b in bodies]),bodies,dofs)
    assert np.max(np.abs(wrong-jac(np.zeros(27))))>1e-5

def test_bounded_one_sided_jacobian_is_recorded():
    x=np.zeros(27);bounds=np.tile([-1.,1.],(27,1));bounds[0,0]=0
    mat=np.ones((5,3,27))*.0001
    report=verify_jacobian(x,bounds,lambda v:np.einsum('ikj,j->ik',mat,v)+.002,lambda v:mat)
    assert report['spans'][0]['minus_delta_scaled']==0


def test_reach_domain_inward_encoded_limits_and_seed_order(tmp_path):
    links=[]
    for i,tip in enumerate(TIP_NAMES):
        links.append(f'<joint name="j{i}" type="revolute"><parent link="palm"/><child link="l{i}"/><origin xyz="0 0 .04"/></joint><joint name="t{i}" type="fixed"><parent link="l{i}"/><child link="{tip}"/><origin xyz="0 .03 0"/></joint>')
    path=tmp_path/'toy.urdf';path.write_text('<robot>'+''.join(links)+'</robot>')
    reaches=urdf_tip_reaches(path);assert all(abs(r['source_reach_upper_m']-.07)<1e-15 for r in reaches)
    rng=np.random.default_rng(17)
    for _ in range(100):
        q0=rng.uniform(-1,1,24);limits=np.column_stack((q0-rng.uniform(.001,1,24),q0+rng.uniform(.001,1,24)))
        b,domain=state_bounds(q0,limits,np.ones((5,3))*.01,np.ones(5)*.001,np.zeros(3),reaches)
        qlo,_=decode_state(b[:,0],q0);qhi,_=decode_state(b[:,1],q0)
        assert np.all(qlo>=limits[:,0]) and np.all(qhi<=limits[:,1])
        assert np.all(qlo>=q0-.35) and np.all(qhi<=q0+.35)
    b=np.tile([-100.,100.],(27,1));seeds,labels=initial_seeds(b)
    assert seeds.shape==(56,27) and len(labels)==56
    assert np.array_equal(seeds[0,24:],[.25,0,0]) and np.array_equal(seeds[1,24:],[-.25,0,0])
    assert np.allclose(np.linalg.norm(seeds[-8:,24:],axis=1),.03125)
    for block, expected in zip(np.split(seeds,4),(.25,.125,.0625,.03125),strict=True):
        assert np.array_equal(block[:,:24],np.zeros((14,24)))
        assert np.allclose(np.linalg.norm(block[:,24:],axis=1),expected,rtol=1e-14,atol=0)

@pytest.mark.parametrize('guard_accept',[True,False])
def test_translation_selection_or_exact_fallback_with_fresh_guard(guard_accept):
    x0=np.zeros(27);seed=x0.copy();seed[24]=.3;calls=[]
    def errors(x):return np.ones((5,3))*.001
    def full(x):calls.append(x.copy());return ConstrainedEvaluation(np.array([1.-x[24]]),errors(x))
    def guard(x,v):return {'accepted':guard_accept or np.array_equal(x,x0),'refusal_reasons':[] if guard_accept or np.array_equal(x,x0) else ['manufactured refusal']}
    result,report=solve_free_palm(x0,np.tile([-1.,1.],(27,1)),full,evaluate_fingertips=errors,
        evaluate_fingertip_jacobian=lambda x:np.zeros((5,3,27)),candidate_guard=guard,
        predeclared_seed_state_scaled=[seed],config=FreePalmConfig(maximum_pattern_sweeps=0,maximum_full_geometry_evaluations=4))
    assert np.array_equal(result,seed if guard_accept else x0)
    assert np.array_equal(calls[-1],result) and report['final_callbacks_fresh_outside_cache']


def test_zero_native_cap_refuses_nonzero_tip_and_analytic_zero_rows():
    x0=np.zeros(27);seed=x0.copy();seed[24]=.1
    def errors(x):return np.tile([x[24]*.005,0,0],(5,1))
    def jac(x):r=np.zeros((5,3,27));r[:,0,24]=.005;return r
    full=lambda x:ConstrainedEvaluation(np.array([1.-x[24]]),errors(x))
    result,report=solve_free_palm(x0,np.tile([-1.,1.],(27,1)),full,evaluate_fingertips=errors,
        evaluate_fingertip_jacobian=jac,predeclared_seed_state_scaled=[seed],
        config=FreePalmConfig(maximum_pattern_sweeps=1,maximum_full_geometry_evaluations=4,maximum_FK_evaluations=50))
    assert np.array_equal(result,x0) and report['final_assessment']['tip_error_norm_m']==[0]*5
    assert any(e['kind']=='analytic_FK_constraint_jacobian' for e in report['events'])
