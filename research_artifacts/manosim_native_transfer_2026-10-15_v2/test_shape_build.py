"""Manufactured producer/provenance controls; no MANO or native asset build."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('literal_shape_build_controls',HERE/'shape_build.py')
build=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(build)
UPSTREAM=Path('/private/tmp/manosim-c96e2d5-2026-10-15')


@pytest.fixture(autouse=True)
def prohibit_native(monkeypatch):
    def forbidden(*a,**k):pytest.fail('Manufactured controls prohibit native model/FK/physics')
    for key in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,key,forbidden)
    monkeypatch.setattr(build,'_load_upstream',forbidden)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    upstream=tmp_path/'upstream';upstream.mkdir()
    for relative in build.PUBLIC_SOURCES:
        path=upstream/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# public fixture\n')
    (upstream/'config/myhand.yaml').write_text('build:\n  total_mass: 0.4\n  actuator:\n    kp: 30\n')
    canonical=np.array([-2.37,-1.25,-2.05,-.85,1.66,-1.35,-1.85,-.67,-1.69,-1.21],np.float32)
    path=upstream/'assets/mano_rhand_shape.npy';path.parent.mkdir();np.save(path,canonical)
    literal=np.array(build.bf16_rne_observation(canonical,np.zeros((1,10)))['rounded_values_float32'],np.float32)
    native=[]
    for i in (20,28):
        path=tmp_path/f'native_{i}.pkl';path.write_bytes(pickle.dumps(dict(shape=literal.copy())))
        native.append(dict(index=i,path=str(path),sha256=build.sha(path)))
    licensed=tmp_path/'licensed/models/MANO_RIGHT.pkl';licensed.parent.mkdir(parents=True);licensed.write_bytes(b'not a licensed model; fixture only')
    monkeypatch.setattr(build,'upstream_commit',lambda root:build.UPSTREAM)
    monkeypatch.setattr(build,'runtime_source_bindings',lambda:dict(trimesh_version='5.1.1',numpy_version=np.__version__,files={}))
    plan=build.build_plan(native,upstream,licensed)
    return SimpleNamespace(plan=plan,native=native,upstream=upstream,licensed=licensed,literal=literal)


def fake_build(plan,hand_dir,shape,counts,guard_reinstall,observation):
    if guard_reinstall:guard_reinstall()
    counts.update(neutral_mano_forward_attempts=1,neutral_mano_forwards=1,geometry_builder_returns=1)
    hand_dir=Path(hand_dir);(hand_dir/'meshes').mkdir(parents=True)
    for i in range(16):(hand_dir/f'meshes/mano_bone_{i}.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    meshes=''.join(f'<mesh file="meshes/mano_bone_{i}.obj"/>' for i in range(16))
    for name in ('mesh_hand.xml','capsule_hand.xml'):
        (hand_dir/name).write_text(f'<mujoco><compiler angle="radian"/><asset>{meshes}</asset></mujoco>')
    observation['arrays'].update(actual_shape_coefficients=shape.copy(),neutral_vertices=np.zeros((1,778,3)),
        neutral_landmarks=np.zeros((1,21,3)),neutral_transforms_abs=np.tile(np.eye(4),(1,16,1,1)),
        actual_pose_coefficients=np.zeros((1,48),np.float32))
    observation['body_metadata'].update({str(i):dict(mass_kg=.025) for i in range(16)})


def complete(fixture,tmp_path,monkeypatch):
    monkeypatch.setattr(build,'_observed_build',fake_build)
    output=tmp_path/'raw';calls=[]
    result=build.build_exact_shape(fixture.plan,output,build_protocol_sha256='a'*64,guard_reinstall=lambda:calls.append(True))
    assert result['status']=='COMPLETE' and calls==[True]
    return output,result


def test_plan_is_literal_readonly_source_bound_and_precision_observational(fixture):
    shape=build.verify_plan(fixture.plan)
    np.testing.assert_array_equal(shape,fixture.literal)
    assert fixture.plan['shape_dtype']=='float32'
    assert fixture.plan['bf16_rounding_observation']['observed_exact_match']
    assert fixture.plan['bf16_rounding_observation']['causal_checkpoint_origin_proven'] is False
    assert fixture.plan['maximum_neutral_mano_forwards']==1
    assert fixture.plan['bindings'][str(fixture.licensed)]==build.sha(fixture.licensed)


@pytest.mark.parametrize('kind',['dtype','unequal','bound_hash'])
def test_mismatching_saved_shape_or_input_bytes_refuse_before_builder(fixture,kind):
    rows=copy.deepcopy(fixture.native);path=Path(rows[1]['path'])
    if kind=='bound_hash':path.write_bytes(b'changed bytes')
    else:
        shape=fixture.literal.copy()
        if kind=='dtype':shape=shape.astype(np.float64)
        else:shape[0,0]+=np.float32(.000001)
        path.write_bytes(pickle.dumps(dict(shape=shape)));rows[1]['sha256']=build.sha(path)
    with pytest.raises(ValueError):build.build_plan(rows,fixture.upstream,fixture.licensed)


@pytest.mark.parametrize('kind',['config','missing_binding','budget','native_order','precision','runtime'])
def test_forged_or_changed_plan_refuses(fixture,kind):
    plan=copy.deepcopy(fixture.plan)
    if kind=='config':plan['build_configuration']['total_mass']=1.;plan['build_configuration_sha256']=build.digest(plan['build_configuration'])
    if kind=='missing_binding':del plan['bindings'][str(fixture.licensed)]
    if kind=='budget':plan['maximum_neutral_mano_forwards']=2
    if kind=='native_order':plan['native_inputs'].reverse()
    if kind=='precision':plan['bf16_rounding_observation']['causal_checkpoint_origin_proven']=True
    if kind=='runtime':plan['runtime']['trimesh_version']='4.0'
    with pytest.raises(ValueError):build.verify_plan(plan)


def test_complete_exact19_and_relocated_report_selects_write_time_assets(fixture,tmp_path,monkeypatch):
    output,result=complete(fixture,tmp_path,monkeypatch)
    assert set(result['generated_files'])==set(build.GENERATED_PATHS)
    np.testing.assert_array_equal(np.load(output/build.SHAPE_REL),fixture.literal)
    assert result['counts']['neutral_mano_forwards']==1
    assert all(result['counts'][k]==0 for k in ('MuJoCo_forwards','physics_steps','HUG_generator_calls','retarget_calls'))
    tracked=tmp_path/'tracked_builder_report.json';tracked.write_bytes((output/'build_report.json').read_bytes())
    selected=build.generated_asset_selection(tracked)
    assert selected['source_root']==str(output)
    assert selected['builder_provenance']['sha256']==build.sha(tracked)
    assert selected['expected_shape_sha256']==result['generated_files'][build.SHAPE_REL]
    assert result['native_fidelity_accepted'] is result['production_promotion'] is False
    with pytest.raises(FileExistsError):build.build_exact_shape(fixture.plan,output,build_protocol_sha256='a'*64)


@pytest.mark.parametrize('kind',['obj','shape','artifact_receipt','protocol_budget','unindexed','symlink'])
def test_generated_selection_tamper_and_forged_promotion_refuse(fixture,tmp_path,monkeypatch,kind):
    output,result=complete(fixture,tmp_path,monkeypatch)
    report=output/'build_report.json'
    if kind=='obj':(output/build.GENERATED_PATHS[-1]).write_text('changed')
    if kind=='shape':np.save(output/build.SHAPE_REL,np.zeros((1,10),np.float32))
    if kind=='artifact_receipt':result['artifacts'][-1]['sha256']='0'*64
    if kind=='protocol_budget':result['counts']['physics_steps']=1
    if kind=='unindexed':result['artifacts']=[r for r in result['artifacts'] if r['path']!=str(output/build.GENERATED_PATHS[-1])]
    if kind=='symlink':
        obj=output/build.GENERATED_PATHS[-1];target=tmp_path/'external.obj';target.write_bytes(obj.read_bytes());obj.unlink();obj.symlink_to(target)
    report.write_text(json.dumps(result))
    with pytest.raises(ValueError):build.generated_asset_selection(report)


def test_partial_upstream_failure_keeps_observed_arrays_and_prevents_selection(fixture,tmp_path,monkeypatch):
    def fail(plan,hand_dir,shape,counts,guard_reinstall,observation):
        counts.update(neutral_mano_forward_attempts=1,neutral_mano_forwards=1)
        observation['arrays']['observed_before_exception']=shape.copy()
        hand_dir=Path(hand_dir);hand_dir.mkdir(parents=True);(hand_dir/'partial.obj').write_text('partial native-free fixture')
        raise RuntimeError('manufactured post-forward producer failure')
    monkeypatch.setattr(build,'_observed_build',fail)
    output=tmp_path/'failed';result=build.build_exact_shape(fixture.plan,output,build_protocol_sha256='a'*64)
    assert result['status']=='HARD_STOP' and result['counts']['neutral_mano_forwards']==1
    np.testing.assert_array_equal(np.load(output/'neutral_producer_geometry.npz')['observed_before_exception'],fixture.literal)
    assert any(r['label']=='build_failure_traceback' for r in result['artifacts'])
    assert any(r['label']=='partial_generated_asset_after_failure' and r['sha256']==build.sha(r['path']) for r in result['artifacts'])
    with pytest.raises(ValueError):build.generated_asset_selection(output/'build_report.json')


def test_invalid_protocol_hash_fails_before_output_creation(fixture,tmp_path):
    with pytest.raises(ValueError,match='protocol SHA'):build.build_exact_shape(fixture.plan,tmp_path/'never',build_protocol_sha256='HEAD')
    assert not (tmp_path/'never').exists()


def test_bf16_scalar_ties_to_even_both_signs():
    bits=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0,0x80000000,0x3f800000,0xbf800000,0x3f807fff,0x3f808001],np.uint32)
    expected=np.array([0x3f800000,0x3f820000,0xbf800000,0xbf820000,0,0x80000000,0x3f800000,0xbf800000,0x3f800000,0x3f810000],np.uint32)
    result=build.bf16_rne_observation(bits.view(np.float32),expected.view(np.float32).reshape(1,10))
    assert result['observed_exact_match'] and not result['causal_checkpoint_origin_proven']


def test_float64_nominal_conversion_observation_never_changes_literal_native(fixture):
    nominal=np.array([-2.37,-1.25,-2.05,-.85,1.66,-1.35,-1.85,-.67,-1.69,-1.21],np.float64)
    path=fixture.upstream/'assets/mano_rhand_shape.npy';np.save(path,nominal)
    plan=build.build_plan(fixture.native,fixture.upstream,fixture.licensed)
    result=plan['bf16_rounding_observation']
    assert result['nominal_source_dtype']=='float64' and result['observation_intermediate_dtype']=='float32'
    assert result['nominal_to_float32_max_abs_difference']>0
    assert result['observed_exact_match'] and result['causal_checkpoint_origin_proven'] is False
    assert plan['canonical_shape_asset_sha256']==build.sha(path)
    np.testing.assert_array_equal(build.verify_plan(plan),fixture.literal)
    assert plan['shape_dtype']=='float32'


def test_observer_original_returns_unmodified_one_call_guard_and_method_restoration(fixture,tmp_path,monkeypatch):
    returned=[]
    class Layer:
        side='right';center_idx=0;use_pca=False;th_hands_mean=np.zeros((1,45))
        th_weights=np.zeros((778,16));th_faces=np.zeros((1538,3),int)
        def forward(self,*a,**k):
            result=SimpleNamespace(verts=np.zeros((1,778,3)),joints=np.zeros((1,21,3)),transforms_abs=np.tile(np.eye(4),(1,16,1,1)))
            returned.append(result);return result
    bodies={i:dict(v_local=np.zeros((4,3)),hull_faces=np.zeros((4,3),int),inertia=np.eye(3),cm=np.zeros(3),mass=.025,obj_path=tmp_path/f'{i}.obj') for i in range(16)}
    geometry_return=(bodies,np.zeros((16,3)),np.zeros((16,3)),np.zeros((5,3)))
    module=SimpleNamespace(__package__='fake',forward_mano=lambda *a,**k:geometry_return)
    first=Layer.forward;geometry_original=module.forward_mano;actual=[]
    def side(*a):
        layer=Layer();actual.append(layer.forward(np.zeros((1,48),np.float32),fixture.literal))
        assert module.forward_mano() is geometry_return
        with pytest.raises(RuntimeError,match='exhausted'):layer.forward(np.zeros((1,48),np.float32),fixture.literal)
    module._build_side=side
    monkeypatch.setattr(build,'_load_upstream',lambda root:module)
    monkeypatch.setattr(build.importlib,'import_module',lambda name:SimpleNamespace(ManoLayer=Layer))
    counts=dict(neutral_mano_forwards=0,neutral_mano_forward_attempts=0);observed=dict(arrays={},body_metadata={})
    build._observed_build(fixture.plan,tmp_path/'unused',fixture.literal,counts,None,observed)
    assert actual[0] is returned[0] and Layer.forward is first and module.forward_mano is geometry_original
    assert counts['neutral_mano_forwards']==1 and counts['neutral_mano_forward_attempts']==1
    assert set(observed['body_metadata'])==set(map(str,range(16)))


def test_pinned_upstream_prepare_bodies_trimesh511_cube_and_tetra_api(tmp_path):
    """Execute ONLY public pure prepare_bodies logic on manufactured solids."""
    import trimesh
    from scipy.spatial import ConvexHull
    assert trimesh.__version__=='5.1.1'
    source=ast.parse((UPSTREAM/'src/mano_build.py').read_text())
    names={'orient_outward','save_obj','prepare_bodies'}
    isolated=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    namespace=dict(np=np,trimesh=trimesh,ConvexHull=ConvexHull,Path=Path)
    exec(compile(isolated,'pinned_manufactured_prepare_bodies','exec'),namespace)
    cube=np.array([[x,y,z] for x in (-1.,1.) for y in (-2.,2.) for z in (-3.,3.)])
    tetra=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    bodies,centers=namespace['prepare_bodies']([
        dict(vertices=cube,faces=np.zeros((0,3),int),joint_idx=0),
        dict(vertices=tetra,faces=np.zeros((0,3),int),joint_idx=1)],.4,tmp_path/'manufactured')
    masses=np.array([.4*48/(48+1/6),.4*(1/6)/(48+1/6)])
    assert np.isfinite([b['mass'] for b in bodies.values()]).all()
    np.testing.assert_allclose([bodies[i]['mass'] for i in (0,1)],masses,rtol=1e-14,atol=0)
    np.testing.assert_allclose(centers[:2],[[0,0,0],[.25,.25,.25]],rtol=0,atol=1e-15)
    np.testing.assert_allclose(bodies[0]['cm'],[0,0,0],rtol=0,atol=1e-15)
    np.testing.assert_allclose(bodies[1]['cm'],[0,0,0],rtol=0,atol=1e-15)
    np.testing.assert_allclose(bodies[0]['inertia'],np.diag(masses[0]/12*np.array([52,40,20])),rtol=1e-14,atol=1e-15)
    tetra_inertia=np.full((3,3),masses[1]/80);np.fill_diagonal(tetra_inertia,masses[1]*3/40)
    np.testing.assert_allclose(bodies[1]['inertia'],tetra_inertia,rtol=1e-14,atol=1e-15)
    assert all(np.linalg.eigvalsh(b['inertia']).min()>0 for b in bodies.values())
