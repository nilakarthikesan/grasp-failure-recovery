"""Synthetic buffers only: no licensed assets, native poses, engine or queries."""
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('manufactured_complete_MANO_skin', HERE/'skin_kinematics.py')
skin = importlib.util.module_from_spec(spec); spec.loader.exec_module(skin)


def manufactured(dtype=torch.float64):
    generator = torch.Generator().manual_seed(917)
    template = .02*torch.randn((20,3),generator=generator,dtype=dtype)
    shapedirs = .001*torch.randn((20,3,10),generator=generator,dtype=dtype)
    posedirs = .0003*torch.randn((20,3,135),generator=generator,dtype=dtype)
    regressor = torch.zeros((16,20),dtype=dtype); regressor[torch.arange(16),torch.arange(16)] = 1
    weights = torch.zeros((20,16),dtype=dtype)
    for vertex in range(20):
        weights[vertex,vertex%16] = .25; weights[vertex,(vertex+5)%16] += .75
    arrays = dict(template=template,shapedirs=shapedirs,posedirs=posedirs,joint_regressor=regressor,weights=weights)
    beta = torch.linspace(-.25,.25,10,dtype=dtype)[None]
    translation = torch.tensor([.12,-.09,.24],dtype=dtype)
    omega = torch.linspace(-.3,.4,48,dtype=dtype).reshape(16,3)
    quaternions = skin.rotation_increment(omega)
    return arrays,beta,translation,quaternions


def test_neutral_shape_centering_and_world_translation_are_applied_once():
    arrays,beta,t,_ = manufactured()
    rotations = torch.eye(3,dtype=t.dtype).repeat(16,1,1)
    state = skin.skin_state(rotations,t,beta,arrays)
    expected_shape = arrays['template']+torch.einsum('vci,i->vc',arrays['shapedirs'],beta[0])
    expected = expected_shape-(arrays['joint_regressor']@expected_shape)[0]+t
    # Two algebraically equivalent neutral expressions accumulate differently.
    # A conservative256scalar-operation forward-error bound applies only to
    # this manufactured arithmetic, never to a physical/surface gate.
    eps = torch.finfo(t.dtype).eps
    gamma256 = 256*eps/(1-256*eps)
    magnitude = float(expected_shape.abs().max()+5*(arrays['joint_regressor']@expected_shape).abs().max()+t.abs().max())
    torch.testing.assert_close(state['vertices'],expected,rtol=0,atol=gamma256*magnitude)
    torch.testing.assert_close(state['pose_corrective'],torch.zeros_like(expected),rtol=0,atol=0)
    torch.testing.assert_close(state['joint_anchors_world'][0],t,rtol=0,atol=0)
    shifted = skin.skin_vertices(rotations,t+torch.tensor([.5,-.2,.1]),beta,arrays)
    torch.testing.assert_close(shifted-state['vertices'],torch.tensor([.5,-.2,.1]).double().expand(20,3),rtol=0,atol=1e-16)


def test_rigid_root_covariance_and_local_chain_order_without_double_root():
    arrays,beta,t,q = manufactured()
    rotations = skin.quaternion_matrices(q)
    before = skin.skin_state(rotations,t,beta,arrays)
    g = skin.quaternion_matrices(skin.rotation_increment(torch.tensor([[.4,-.2,.1]]*16,dtype=t.dtype)))[0]
    moved_rotations = torch.cat(((g@rotations[0])[None],rotations[1:]))
    new_t = torch.tensor([-.2,.17,.05],dtype=t.dtype)
    after = skin.skin_state(moved_rotations,new_t,beta,arrays)
    torch.testing.assert_close(after['vertices'],(before['vertices']-t)@g.T+new_t,rtol=0,atol=1e-16)
    torch.testing.assert_close(before['absolute_rotations'][3],rotations[0]@rotations[1]@rotations[2]@rotations[3],rtol=0,atol=0)


def test_all51_tangent_columns_match_independent_central_differences():
    arrays,beta,t,q = manufactured()
    jac = skin.skin_jacobian(t,q,beta,arrays)
    assert jac.shape == (20,3,51) and torch.isfinite(jac).all()
    step = 1e-6
    columns = []
    for column in range(51):
        delta = torch.zeros(51,dtype=t.dtype); delta[column] = step
        plus = skin.tangent_vertices(t,q,delta,beta,arrays)
        minus = skin.tangent_vertices(t,q,-delta,beta,arrays)
        columns.append((plus-minus)/(2*step))
    fd = torch.stack(columns,dim=-1)
    torch.testing.assert_close(jac,fd,rtol=1e-7,atol=1e-10)
    torch.testing.assert_close(jac[:,:,:3],torch.eye(3,dtype=t.dtype)[None].expand(20,3,3),rtol=0,atol=0)


def test_pose_corrective_derivative_is_present_even_at_neutral_pose():
    arrays,beta,t,_ = manufactured(); q = torch.zeros((16,4),dtype=t.dtype); q[:,0] = 1
    without = dict(arrays,posedirs=torch.zeros_like(arrays['posedirs']))
    zero = torch.zeros(51,dtype=t.dtype)
    torch.testing.assert_close(skin.tangent_vertices(t,q,zero,beta,arrays),skin.tangent_vertices(t,q,zero,beta,without),rtol=0,atol=0)
    full,omitted = skin.skin_jacobian(t,q,beta,arrays),skin.skin_jacobian(t,q,beta,without)
    assert torch.max(torch.abs(full[:,:,6:]-omitted[:,:,6:])) > 1e-4
    # Pose corrections use all15 local joint matrices; root rotation alone
    # cannot enter the135-component pose-corrective feature vector.
    torch.testing.assert_close(full[:,:,:6],omitted[:,:,:6],rtol=0,atol=0)


def test_virtual_work_vjp_and_triangle_material_jacobian_agree():
    arrays,beta,t,q = manufactured(); jac = skin.skin_jacobian(t,q,beta,arrays)
    generator = torch.Generator().manual_seed(108)
    forces = torch.randn((20,3),generator=generator,dtype=t.dtype)
    velocity = torch.randn(51,generator=generator,dtype=t.dtype)
    tau = skin.skin_force_vjp(t,q,beta,arrays,forces)
    expected = torch.einsum('vcd,vc->d',jac,forces)
    torch.testing.assert_close(tau,expected,rtol=1e-12,atol=1e-14)
    work = (torch.einsum('vcd,d->vc',jac,velocity)*forces).sum()
    torch.testing.assert_close(tau@velocity,work,rtol=1e-12,atol=1e-14)
    vertices = skin.skin_vertices(skin.quaternion_matrices(q),t,beta,arrays)
    ids = torch.tensor([2,9,17]); bary = torch.tensor([.2,.3,.5],dtype=t.dtype)
    point,jpoint = skin.material_point(vertices,jac,ids,bary)
    delta = 1e-6*velocity
    plus = skin.tangent_vertices(t,q,delta,beta,arrays)[ids]
    minus = skin.tangent_vertices(t,q,-delta,beta,arrays)[ids]
    fd = torch.einsum('i,ic->c',bary,(plus-minus)/(2e-6))
    torch.testing.assert_close(jpoint@velocity,fd,rtol=1e-7,atol=1e-10)
    torch.testing.assert_close(point,torch.einsum('i,ic->c',bary,vertices[ids]),rtol=0,atol=0)


def test_quaternion_double_cover_and_finite_nonzero_tangent_do_not_rewrite_inputs():
    arrays,beta,t,q = manufactured(); saved = q.clone()
    delta = torch.linspace(-.02,.02,51,dtype=t.dtype)
    plus = skin.tangent_vertices(t,q,delta,beta,arrays)
    minus_cover = skin.tangent_vertices(t,-q,delta,beta,arrays)
    torch.testing.assert_close(plus,minus_cover,rtol=0,atol=0)
    assert torch.equal(q,saved)


def test_precision_profiles_remain_explicit_no_implicit_source_target_replacement():
    arrays,beta,t,q = manufactured(torch.float32)
    low = skin.tangent_vertices(t,q,torch.zeros(51,dtype=t.dtype),beta,arrays)
    assert low.dtype == torch.float32
    high_arrays = {name:value.double() for name,value in arrays.items()}
    with pytest.raises(ValueError, match='unit quaternion'):
        skin.tangent_vertices(t.double(),q.double(),torch.zeros(51,dtype=torch.float64),beta.double(),high_arrays)
    proper64 = skin.rotation_increment(torch.linspace(-.3,.4,48,dtype=torch.float64).reshape(16,3))
    high = skin.tangent_vertices(t.double(),proper64,torch.zeros(51,dtype=torch.float64),beta.double(),high_arrays)
    assert high.dtype == torch.float64 and low.dtype == torch.float32
    # q.float32→double is not re-normalized; the strictdouble unit contract
    # requires a genuinely double source representation, rather than claiming
    # float32 quaternions became more precise through a dtype cast.


@pytest.mark.parametrize('change',['beta_flat','missing_pose_terms','mixed_precision','nonfinite','nonunit_quat','negative_weight','nonrotation'])
def test_invalid_or_silently_incomplete_skin_inputs_fail_closed(change):
    arrays,beta,t,q = manufactured(); delta = torch.zeros(51,dtype=t.dtype)
    if change == 'beta_flat': beta = beta[0]
    elif change == 'missing_pose_terms': arrays['posedirs'] = arrays['posedirs'][:,:,:45]
    elif change == 'mixed_precision': arrays['weights'] = arrays['weights'].float()
    elif change == 'nonfinite': arrays['template'][0,0] = float('nan')
    elif change == 'nonunit_quat': q[0,0] *= 2
    elif change == 'negative_weight': arrays['weights'][0,0] = -1
    elif change == 'nonrotation':
        rotations = skin.quaternion_matrices(q); rotations[1] *= 2
        with pytest.raises(ValueError): skin.skin_vertices(rotations,t,beta,arrays)
        return
    with pytest.raises(ValueError): skin.tangent_vertices(t,q,delta,beta,arrays)


def test_literal_weights_are_not_renormalized_or_reassigned():
    arrays,beta,t,q = manufactured(); altered = dict(arrays,weights=arrays['weights'].clone())
    altered['weights'][0] *= .9
    state = skin.skin_state(skin.quaternion_matrices(q),t,beta,altered)
    assert float(state['literal_weight_row_sum'][0]) == pytest.approx(.9)
    assert not torch.equal(state['vertices'][0],skin.skin_state(skin.quaternion_matrices(q),t,beta,arrays)['vertices'][0])


def test_material_point_refuses_invalid_barycentric_or_vertex_inventory():
    arrays,beta,t,q = manufactured(); jac = skin.skin_jacobian(t,q,beta,arrays)
    vertices = skin.skin_vertices(skin.quaternion_matrices(q),t,beta,arrays)
    with pytest.raises(ValueError): skin.material_point(vertices,jac,torch.tensor([0,1,20]),torch.tensor([.2,.3,.5],dtype=t.dtype))
    with pytest.raises(ValueError): skin.material_point(vertices,jac,torch.tensor([0,1,2]),torch.tensor([.2,.3,.6],dtype=t.dtype))


def test_predeclared109_pose_grid_preserves_baseline_and_separates_units():
    grid = skin.motion_grid_spec(); rows = grid['poses']
    assert len(rows) == 109 and len({r['pose_id'] for r in rows}) == 109
    assert rows[0]['delta51'] == [0.]*51 and len(grid['tangent_names']) == 51
    assert sum(row['pose_id'].startswith('finger_') for row in rows) == 90
    assert sum(row['pose_id'].startswith('coupled_') for row in rows) == 6
    assert sum(row['pose_id'].startswith('wrist_') for row in rows) == 6
    assert sum(row['pose_id'].startswith('translation_') for row in rows) == 6
    for row in rows:
        if row['pose_id'].startswith('translation_'):
            assert max(abs(x) for x in row['delta51'][:3]) == .002 and row['delta51'][3:] == [0.]*48
    assert grid['accepted_mapping'] is False and grid['collection_or_training_allowed'] is False
