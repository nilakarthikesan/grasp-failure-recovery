"""Generated selection and call-wiring controls, all native work manufactured."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
case=module('literal_native_generated_case_controls',HERE/'audit_case.py')


@pytest.fixture(autouse=True)
def prohibit_real_work(monkeypatch):
    def forbidden(*a,**k):pytest.fail('No actual native/model/geometry/physics operation in wiring controls')
    for name in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,forbidden)


def selection(tmp_path):
    root=tmp_path/'raw'
    return dict(source_root=str(root),shape_asset=str(root/'assets/mano_rhand_shape.npy'),
        expected_shape_sha256='a'*64,right_hand_directory=str(root/'assets/literal_native/mano_rhand'),
        variant_xmls={v:str(root/'assets/literal_native/mano_rhand'/f'{v}_hand.xml') for v in ('mesh','capsule')},
        public_source_snapshots=[str(root/'source_snapshots/src/mano_build.py')],
        builder_provenance=dict(path=str(tmp_path/'builder_report.json'),sha256='b'*64),
        build_protocol_sha256='c'*64,native_inputs=[dict(index=i,path=f'native{i}',sha256='d'*64) for i in (20,28)])


@pytest.mark.parametrize('change',[None,'shape','variant','receipt','producer'])
def test_frozen_generated_selection_is_exact_not_implicitly_reconstructed(tmp_path,monkeypatch,change):
    actual=selection(tmp_path);p=dict(builder_report='bound_builder_report',generated_assets=copy.deepcopy(actual));calls=[]
    def select(path):calls.append(path);return actual
    monkeypatch.setattr(case,'load',lambda *a:SimpleNamespace(generated_asset_selection=select))
    if change=='shape':p['generated_assets']['expected_shape_sha256']='0'*64
    if change=='variant':p['generated_assets']['variant_xmls']['mesh']='nominal/mesh_hand.xml'
    if change=='receipt':p['generated_assets']['builder_provenance']['sha256']='0'*64
    if change=='producer':p['generated_assets']['build_protocol_sha256']='0'*64
    if change:
        with pytest.raises(ValueError,match='selection differs'):case.generated_selection(p)
    else:assert case.generated_selection(p)==actual
    assert calls==['bound_builder_report']


def test_source_inputs_bind_generated_and_producer_arrays_without_nominal_colliders(tmp_path,monkeypatch):
    generated=selection(tmp_path);producer_arrays=tmp_path/'raw/producer.npz'
    report=Path(generated['builder_provenance']['path']);report.write_text(json.dumps(dict(artifacts=[dict(path=producer_arrays)]),default=str))
    v3=tmp_path/'v3.json';v3.write_text(json.dumps(dict(dependency_dirs=[])))
    monkeypatch.setattr(case,'generated_selection',lambda p:generated)
    p=dict(manosim_root=str(tmp_path/'upstream'),hug_root=str(tmp_path/'HUG'),mano_asset_path=str(tmp_path/'MANO_RIGHT.pkl'),
        v3_protocol=str(v3),builder_report=str(report))
    paths=case.source_inputs(p)
    assert producer_arrays in paths and Path(generated['shape_asset']) in paths
    assert all(Path(x) in paths for x in generated['public_source_snapshots'])
    assert not any('assets/myhand/mano_rhand' in str(path) for path in paths)
    assert (tmp_path/'upstream/src/build_mjcf.py') in paths


def test_run_uses_generated_shape_xml_receipt_and_explicit_expected_digest(tmp_path,monkeypatch):
    generated=selection(tmp_path);calls=[]
    monkeypatch.setattr(case,'generated_selection',lambda p:copy.deepcopy(generated))
    arrays=dict(independent_anchors_world_m=np.zeros((16,3)),independent_rotations_world=np.tile(np.eye(3),(16,1,1)),
        staging_wrist_world_transform=np.eye(4),staging_full_rotations=np.tile(np.eye(3),(16,1,1)))
    receipt=dict(native_sha256='nativeSHA',native_reconstruction_pass=True,independent_fk_pass=True)
    current=case.AuditCase.__new__(case.AuditCase)
    current.p=dict(adapter_source='adapter',builder_report='builder',generated_assets=generated,hug_root='HUG',mano_asset_path='licensed')
    current.generated_assets=copy.deepcopy(generated);current.output=tmp_path/'audit';current.output.mkdir()
    current.source=dict(native=dict(path='native',sha256='nativeSHA'),frame_files={'T_world_camera.npy':dict(path='camera')})
    current.source_receipt={};current.checkpoint_artifacts=[];current.auxiliary_artifacts=[]
    current.counts=dict(parametric_mano_forwards=0,full_field_variant_scans=0,collider_point_queries=0)
    def decode(native,camera,shape,hug,licensed,callback):
        calls.append(('decode',str(shape)));callback();callback();return arrays,receipt
    def identity(shape,**kwargs):calls.append(('identity',str(shape),kwargs));return dict(shape_asset_sha256='a'*64)
    def closure(xml,shape,**kwargs):calls.append(('closure',str(xml),str(shape),kwargs));return {'closure':'manufactured'}
    state=SimpleNamespace(named_qpos={'wrist_tx':0},joint_anchor_world_m=np.zeros((16,3)),
        world_body_rotations=np.tile(np.eye(3),(16,1,1)),local_body_rotations=np.tile(np.eye(3),(16,1,1)),
        body_origin_world_m=np.ones((16,3)),independent_joint_anchor_world_m=np.zeros((16,3)),
        independent_world_body_rotations=np.tile(np.eye(3),(16,1,1)),independent_body_origin_world_m=np.ones((16,3)),
        wrist_anchor_world_transform=np.eye(4),root_body_world_transform=np.eye(4),quaternion_precision={'precision':'retained'},
        independent_readback={'pose':'manufactured'},model_sha256='model',source_closure_sha256='closure',baseline_state_sha256='baseline',
        serialize_pose=lambda:{'named':'literal'})
    class Adapter:
        forward_calls=2;data='independent manufactured data'
        def __init__(self,model,data,**kwargs):calls.append(('adapter',kwargs))
        def stage(self,wrist,local,**kwargs):
            assert set(local)=={f'mano_joint_{i}' for i in range(1,16)};return state
        def stage_serialized(self,pose):assert pose=={'named':'literal'};return state
    adapter=SimpleNamespace(asset_shape_identity=identity,source_asset_closure=closure,ManoSimStaticPoseAdapter=Adapter)
    monkeypatch.setattr(case,'load',lambda name,path:adapter if path=='adapter' else SimpleNamespace(decode_native=decode))
    monkeypatch.setattr(mujoco,'MjModel',SimpleNamespace(from_xml_path=lambda path:calls.append(('xml',path)) or 'model'))
    monkeypatch.setattr(mujoco,'MjData',lambda model:'data')
    monkeypatch.setattr(current,'_object_reference',lambda:(np.zeros(3),np.eye(3)))
    def full(model,data,variant,*a):
        current.counts['full_field_variant_scans']+=1
        path=current.output/f'{variant}_full_field.npz';np.savez_compressed(path,x=[1.])
        current._artifact(path,variant,checkpoint=True)
        return dict(surface_fidelity_status='INCOMPLETE',native_fidelity_accepted=False)
    monkeypatch.setattr(current,'_full_field',full)
    result=current.run()
    assert current.counts['parametric_mano_forwards']==2 and current.counts['full_field_variant_scans']==2
    assert [x[1] for x in calls if x[0]=='xml']==[generated['variant_xmls'][v] for v in ('mesh','capsule')]
    assert [x for x in calls if x[0]=='decode']==[('decode',generated['shape_asset'])]
    for item in [x for x in calls if x[0]=='closure']:
        assert item[3]['source_root']==generated['source_root']
        assert item[3]['builder_provenance']==generated['builder_provenance']
        assert item[3]['expected_shape_sha256']==generated['expected_shape_sha256']
        assert item[3]['extra_source_paths']==generated['public_source_snapshots']
    for item in [x for x in calls if x[0]=='adapter']:
        assert item[1]['expected_shape_sha256']==generated['expected_shape_sha256']
    assert result['kinematic_pose_transfer_status']=='PASS' and result['surface_fidelity_status']=='INCOMPLETE'
    assert all(result[k] is False for k in ('native_fidelity_accepted','production_promotion','new_collection_or_training_allowed'))
    assert len(current.checkpoint_artifacts)==2
