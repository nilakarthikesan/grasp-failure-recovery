"""Only manufactured MANOSim-layout models; no upstream/native evaluation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grasp_failure_prediction.evaluation import manosim_pose as mp


def _quat(rotvec):return ' '.join(map(str,Rotation.from_rotvec(rotvec).as_quat()[[3,0,1,2]]))

def manufactured_xml():
    root=ET.Element('mujoco',model='manufactured_manosim')
    ET.SubElement(root,'compiler',angle='radian')
    world=ET.SubElement(root,'worldbody')
    other=ET.SubElement(world,'body',name='unrelated',pos='.7 0 .5')
    ET.SubElement(other,'freejoint',name='unrelated_free')
    ET.SubElement(other,'geom',type='sphere',size='.01',mass='1')
    bodies=[]
    for i in range(16):
        parent=world if i==0 else bodies[mp.PARENTS[i]]
        body=ET.SubElement(parent,'body',name=f'mano_body_{i}',pos='.1 .2 .3' if i==0 else '.025 .003 -.002',
                           quat=_quat([.12,-.09,.07] if i==0 else [0,.03*(i%3),-.04]))
        bodies.append(body)
        if i==0:
            # Reordered translations and nonzero joint anchor, no flat qpos assumption.
            for name,axis in [('wrist_tz','0 0 1'),('wrist_tx','1 0 0'),('wrist_ty','0 1 0')]:
                ET.SubElement(body,'joint',name=name,type='slide',axis=axis,limited='false',ref='.013')
            ET.SubElement(body,'joint',name='wrist_rot',type='ball',pos='.01 -.02 .03',limited='false')
        else:ET.SubElement(body,'joint',name=f'mano_joint_{i}',type='ball',pos='.006 -.001 .003',limited='false')
        ET.SubElement(body,'geom',name=f'bone_geom_{i}',type='sphere',pos='.004 .002 -.001',size='.006',mass='.01',contype='0',conaffinity='0')
    actuator=ET.SubElement(root,'actuator')
    for label in 'xyz':ET.SubElement(actuator,'position',name='wrist_motor_t'+label,joint='wrist_t'+label,kp='2')
    for i in range(16):
        for axis,label in enumerate('xyz'):
            gear=['0','0','0'];gear[axis]='1'
            ET.SubElement(actuator,'position',name='wrist_motor_r'+label if i==0 else f'mano_motor_{i}_{label}',
                joint='wrist_rot' if i==0 else f'mano_joint_{i}',gear=' '.join(gear),kp='2')
    return root


@pytest.fixture
def source(tmp_path,monkeypatch):
    shape=tmp_path/'shape.npy';np.save(shape,np.arange(10,dtype=np.float32))
    monkeypatch.setattr(mp,'EXPECTED_SHAPE_SHA256',mp._sha(shape))
    (tmp_path/'mano_rhand').mkdir()
    path=tmp_path/'mano_rhand/mesh_hand.xml';path.write_text(ET.tostring(manufactured_xml(),encoding='unicode'))
    def build(xml=None):
        if xml is not None:path.write_text(ET.tostring(xml,encoding='unicode'))
        assets=mp.source_asset_closure(path,shape,source_root=tmp_path)
        identity=mp.asset_shape_identity(shape)
        model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
        # Full-state restoration includes unrelated scene coordinates and ctrl.
        data.ctrl[:]=np.linspace(-.1,.1,model.nu)
        data.qpos[0]=.83;data.qvel[0]=.27;data.qacc_warmstart[:]=.013
        return model,data,assets,identity
    model,data,assets,identity=build()
    return SimpleNamespace(path=path,shape=shape,root=tmp_path,build=build,model=model,data=data,assets=assets,identity=identity)


def adapter(source):return mp.ManoSimStaticPoseAdapter(source.model,source.data,source_assets=source.assets,asset_shape_identity=source.identity)
def target():
    t=np.eye(4);t[:3,:3]=Rotation.from_rotvec([.37,-.28,.41]).as_matrix();t[:3,3]=[.45,-.12,.61]
    local={name:Rotation.from_rotvec([.01*i,-.016*i,.008*i]).as_matrix() for i,name in enumerate(mp.LOCAL_NAMES,1)}
    return t,local


def test_named_placement_reads_joint_anchors_not_centroids_with_rest_rotations(source):
    a=adapter(source);t,local=target();state=a.stage(t,local,asset_shape_identity=source.identity)
    np.testing.assert_allclose(state.wrist_anchor_world_transform,t,atol=1e-12,rtol=0)
    for i,name in enumerate(mp.LOCAL_NAMES,1):np.testing.assert_allclose(state.local_body_rotations[i],local[name],atol=1e-12,rtol=0)
    # Closed-form root anchor/body distinction with a nonzero pivot.
    c=source.model.jnt_pos[a.balls[0]]
    np.testing.assert_allclose(state.root_body_world_transform[:3,3],t[:3,3]-t[:3,:3]@c,atol=1e-12,rtol=0)
    assert np.linalg.norm(state.joint_anchor_world_m[1]-state.body_origin_world_m[1])>.001
    np.testing.assert_allclose(state.joint_anchor_world_m,state.independent_joint_anchor_world_m,atol=1e-12,rtol=0)
    assert state.independent_readback['anchor_max_error_m']<1e-12
    assert state.independent_readback['integration_steps']==0 and not state.independent_readback['ctrl_written']
    assert state.bone_joint_names==mp.BALL_NAMES and len(state.joint_addresses)==19 and len(state.actuator_addresses)==51
    assert a.qadr['wrist_tx']!=0 and a.qadr['wrist_tx']!=a.qadr['wrist_tz']
    assert all(not r['ctrl_written'] and r['qpos_quaternion_is_not_ctrl'] for r in state.actuator_addresses)
    assert not state.joint_anchor_world_m.flags.writeable


def test_full_frozen_state_repeated_order_and_unrelated_scene_preserved(source):
    a=adapter(source);t,local=target();before=source.data.qpos.copy();ctrl=source.data.ctrl.copy()
    s=a.stage(t,local,asset_shape_identity=source.identity)
    a.data.qpos[:]=0;a.data.ctrl[:]=99;a.data.qvel[:]=4;a.data.time=9
    other=t.copy();other[:3,3]+=.03;a.stage(other,local,asset_shape_identity=source.identity)
    again=a.stage(t,local,asset_shape_identity=source.identity)
    np.testing.assert_array_equal(s.joint_anchor_world_m,again.joint_anchor_world_m)
    np.testing.assert_array_equal(a.data.ctrl,ctrl);assert a.data.qpos[0]==before[0] and a.data.qvel[0]==.27 and a.data.time==0
    np.testing.assert_array_equal(source.data.qpos,before);np.testing.assert_array_equal(source.data.ctrl,ctrl)
    assert a.forward_calls==3


def test_literal_named_qpos_roundtrip_preserves_full_quaternion_precision(source):
    a=adapter(source);t,local=target();s=a.stage(t,local,asset_shape_identity=source.identity)
    payload=json.loads(json.dumps(s.serialize_pose()))
    r=a.stage_serialized(payload)
    np.testing.assert_array_equal(s.joint_anchor_world_m,r.joint_anchor_world_m)
    np.testing.assert_array_equal(s.world_body_rotations,r.world_body_rotations)
    for name in mp.BALL_NAMES:
        np.testing.assert_array_equal(s.named_qpos[name],r.named_qpos[name])
        assert not r.quaternion_precision[name]['normalization_applied']
        assert not s.quaternion_precision[name]['decimal_rounding_applied']
        assert s.quaternion_precision[name]['rotation_max_error']<1e-12
    assert payload['ctrl_convention']=='not_commanded_frozen_baseline_retained'


@pytest.mark.parametrize('change',['left','units','convention','domain','wrong_shape','alias','missing','nonfinite','scale_rotation','reflect'])
def test_invalid_explicit_inputs_refused_without_forward(source,change):
    a=adapter(source);t,local=target();kwargs=dict(asset_shape_identity=source.identity)
    if change=='left':kwargs['handedness']='left'
    elif change=='units':kwargs['units']='mm'
    elif change=='convention':kwargs['rotation_convention']='euler_xyz'
    elif change=='domain':kwargs['domain']='unknown'
    elif change=='wrong_shape':kwargs['asset_shape_identity']={**source.identity,'shape_asset_sha256':'0'*64}
    elif change=='alias':local['mano_joint_01']=local.pop('mano_joint_1')
    elif change=='missing':local.pop('mano_joint_15')
    elif change=='nonfinite':t[0,3]=np.nan
    elif change=='scale_rotation':local['mano_joint_1'][0]*=2
    else:t[:3,:3]=np.diag([-1.,1.,1.])
    with pytest.raises(ValueError):a.stage(t,local,**kwargs)
    assert a.forward_calls==0


@pytest.mark.parametrize('change',['missing_axis','axis_alias','body_alias','joint_alias','actuator_alias','actuator_axis','limited','bad_parent'])
def test_unknown_model_layout_or_missing_axes_refused(source,change):
    xml=manufactured_xml()
    if change=='missing_axis':xml.find(".//position[@name='mano_motor_8_y']").set('name','not_the_upstream_axis')
    elif change=='axis_alias':xml.find(".//joint[@name='wrist_ty']").set('axis','1 0 0')
    elif change=='body_alias':xml.find(".//body[@name='mano_body_7']").set('name','renamed_body')
    elif change=='joint_alias':xml.find(".//joint[@name='mano_joint_7']").set('name','renamed_joint');xml.find(".//position[@name='mano_motor_7_x']").set('joint','renamed_joint');xml.find(".//position[@name='mano_motor_7_y']").set('joint','renamed_joint');xml.find(".//position[@name='mano_motor_7_z']").set('joint','renamed_joint')
    elif change=='actuator_alias':ET.SubElement(xml.find('actuator'),'position',name='extra_alias',joint='mano_joint_1',gear='1 0 0',kp='1')
    elif change=='actuator_axis':xml.find(".//position[@name='mano_motor_4_z']").set('gear','0 1 0')
    elif change=='limited':j=xml.find(".//joint[@name='mano_joint_1']");j.set('limited','true');j.set('range','0 .8')
    else:body=xml.find(".//body[@name='mano_body_2']");xml.find(".//body[@name='mano_body_1']").remove(body);xml.find(".//body[@name='mano_body_0']").append(body)
    m,d,assets,identity=source.build(xml)
    with pytest.raises(ValueError):mp.ManoSimStaticPoseAdapter(m,d,source_assets=assets,asset_shape_identity=identity)


@pytest.mark.parametrize('change',['names','quat','identity','flat','extra','units'])
def test_serialized_pose_is_literal_and_fail_closed(source,change):
    a=adapter(source);t,local=target();payload=a.stage(t,local,asset_shape_identity=source.identity).serialize_pose();calls=a.forward_calls
    if change=='names':payload['named_qpos']['mano_joint_01']=payload['named_qpos'].pop('mano_joint_1')
    elif change=='quat':payload['named_qpos']['mano_joint_1'][0]+=1e-4
    elif change=='identity':payload['model_sha256']='0'*64
    elif change=='flat':payload['named_qpos']=[0.]*67
    elif change=='extra':payload['ctrl']=[0.]*51
    else:payload['units']='cm'
    with pytest.raises(ValueError):a.stage_serialized(payload)
    assert a.forward_calls==calls


def test_source_model_or_frozen_state_mutation_refused(source):
    a=adapter(source);t,local=target()
    source.path.write_text(source.path.read_text()+'\n')
    with pytest.raises(ValueError,match='source asset changed'):a.stage(t,local,asset_shape_identity=source.identity)
    source.path.write_text(source.path.read_text()[:-1]);source.model.jnt_pos[a.balls[1],0]+=.001
    with pytest.raises(ValueError,match='compiled MANOSim model changed'):a.stage(t,local,asset_shape_identity=source.identity)


def test_shape_mismatch_and_nonzero_baseline_time_refused(source):
    with pytest.raises(ValueError,match='shape asset identity'):mp.asset_shape_identity(source.path)
    source.data.time=.1
    with pytest.raises(ValueError,match='time zero'):adapter(source)


def test_source_asset_closure_captures_transitive_references_and_rejects_omission(source):
    include=source.path.parent/'included.xml';include.write_text('<mujocoinclude><asset><mesh name="unused" file="part.obj"/></asset></mujocoinclude>')
    mesh=source.path.parent/'part.obj';mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n')
    xml=manufactured_xml();ET.SubElement(xml,'include',file='included.xml');source.path.write_text(ET.tostring(xml,encoding='unicode'))
    assets=mp.source_asset_closure(source.path,source.shape,source_root=source.root)
    assert {str(source.path),str(source.shape),str(include),str(mesh)}==set(assets['absolute_files'])
    m=mujoco.MjModel.from_xml_path(str(source.path));d=mujoco.MjData(m)
    bad=copy.deepcopy(assets);bad['absolute_files'].pop(str(mesh));bad['files'].pop('mano_rhand/part.obj')
    payload={k:bad[k] for k in ('schema_version','upstream_revision','source_variant','xml_relative_path','files','asset_shape_identity')};bad['closure_sha256']=mp._json_hash(payload)
    with pytest.raises(ValueError,match='incomplete'):mp.ManoSimStaticPoseAdapter(m,d,source_assets=bad,asset_shape_identity=source.identity)


def test_source_unknown_compiler_units_refused(source):
    xml=manufactured_xml();xml.find('compiler').set('angle','degree');source.path.write_text(ET.tostring(xml,encoding='unicode'))
    with pytest.raises(ValueError,match='radian'):mp.source_asset_closure(source.path,source.shape,source_root=source.root)


@pytest.mark.parametrize('field',['xanchor','xpos','xmat'])
def test_nonfinite_actual_kinematic_readback_is_never_a_nan_comparison_pass(source,monkeypatch,field):
    a=adapter(source);t,local=target();real=mp.mujoco.mj_forward
    def corrupt(m,d):
        real(m,d)
        if field=='xanchor':d.xanchor[a.balls[1],0]=np.nan
        elif field=='xpos':d.xpos[a.bodies[1],0]=np.nan
        else:d.xmat[a.bodies[1],0]=np.nan
    monkeypatch.setattr(mp.mujoco,'mj_forward',corrupt)
    with pytest.raises(ValueError,match='Nonfinite independent/actual'):a.stage(t,local,asset_shape_identity=source.identity)


def test_static_warning_and_mutated_frozen_state_refused(source,monkeypatch):
    a=adapter(source);t,local=target();real=mp.mujoco.mj_forward
    def warn(m,d):
        real(m,d);d.warning.number[0]+=1
    monkeypatch.setattr(mp.mujoco,'mj_forward',warn)
    with pytest.raises(ValueError,match='warning during static'):a.stage(t,local,asset_shape_identity=source.identity)
    monkeypatch.setattr(mp.mujoco,'mj_forward',real)
    a._state.flags.writeable=True;a._state[0]=.01
    with pytest.raises(ValueError,match='Frozen baseline'):a.stage(t,local,asset_shape_identity=source.identity)


def test_left_or_aliased_source_directory_is_not_labeled_right(source):
    for name in ('mano_lhand','renamed_hand'):
        directory=source.root/name;directory.mkdir();path=directory/'mesh_hand.xml'
        path.write_text(ET.tostring(manufactured_xml(),encoding='unicode'))
        with pytest.raises(ValueError,match='right-hand mano_rhand'):
            mp.source_asset_closure(path,source.shape,source_root=source.root)


@pytest.fixture
def generated(source):
    """Synthetic producer receipt and nineteen synthetic output assets only."""
    root=source.root/'generated';hand=root/'assets/literal_native/mano_rhand';(hand/'meshes').mkdir(parents=True)
    shape=root/'assets/mano_rhand_shape.npy';np.save(shape,np.arange(10,dtype=np.float32)+.125)
    mesh_text='v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n'
    for i in range(16):(hand/f'meshes/mano_bone_{i}.obj').write_text(mesh_text)
    xml=manufactured_xml();assets=ET.SubElement(xml,'asset')
    for i in range(16):ET.SubElement(assets,'mesh',name=f'mano_bone_{i}',file=f'meshes/mano_bone_{i}.obj')
    for name in ('mesh_hand.xml','capsule_hand.xml'):(hand/name).write_text(ET.tostring(xml,encoding='unicode'))
    report=dict(schema_version='manosim_literal_native_shape_build_report_v1',status='COMPLETE',failure=None,
        upstream_revision=mp.UPSTREAM_REVISION,
        shape_asset_sha256=mp._sha(shape),generated_root=str(root),
        generated_files={name:mp._sha(root/name) for name in sorted(mp._GENERATED_FILES)},
        producer_details='Additional producer source/protocol bindings are owned by the frozen build report.')
    report['artifacts']=[dict(path=str(root/name),sha256=report['generated_files'][name],
        bytes=(root/name).stat().st_size,label='manufactured:'+name) for name in sorted(mp._GENERATED_FILES)]
    report_path=source.root/'build_report.json'
    def write_report():
        report_path.write_text(json.dumps(report,sort_keys=True,allow_nan=False))
        return dict(path=str(report_path),sha256=mp._sha(report_path))
    binding=write_report()
    def closure(**overrides):
        kwargs=dict(source_root=root,expected_shape_sha256=report['shape_asset_sha256'],builder_provenance=binding)
        kwargs.update(overrides)
        return mp.source_asset_closure(hand/'mesh_hand.xml',shape,**kwargs)
    return SimpleNamespace(root=root,hand=hand,shape=shape,report=report,report_path=report_path,
        binding=binding,write_report=write_report,closure=closure,digest=mp._sha(shape))


def test_canonical_default_closure_payload_and_identity_are_unchanged(source):
    explicit=mp.source_asset_closure(source.path,source.shape,source_root=source.root,
        expected_shape_sha256=mp.EXPECTED_SHAPE_SHA256)
    assert explicit==source.assets
    assert mp.asset_shape_identity(source.shape,expected_shape_sha256=mp.EXPECTED_SHAPE_SHA256)==source.identity
    assert explicit['schema_version']=='manosim_static_source_asset_closure_v1'
    assert 'builder_provenance' not in explicit and 'expected_shape_sha256' not in explicit
    a=mp.ManoSimStaticPoseAdapter(source.model,source.data,source_assets=explicit,asset_shape_identity=source.identity,
        expected_shape_sha256=mp.EXPECTED_SHAPE_SHA256)
    assert a.expected_shape_sha256==mp.EXPECTED_SHAPE_SHA256


def test_generated_shape_requires_explicit_frozen_expectation_and_provenance(generated):
    with pytest.raises(ValueError,match='shape asset identity'):mp.asset_shape_identity(generated.shape)
    identity=mp.asset_shape_identity(generated.shape,expected_shape_sha256=generated.digest)
    assert identity['shape_asset_sha256']==generated.digest
    with pytest.raises(ValueError,match='explicit expected shape'):generated.closure(expected_shape_sha256=None)
    with pytest.raises(ValueError,match='requires frozen builder'):generated.closure(builder_provenance=None)
    assets=generated.closure()
    assert assets['schema_version']=='manosim_static_source_asset_closure_v2'
    assert assets['expected_shape_sha256']==generated.digest
    assert assets['builder_provenance']==generated.binding
    assert set(assets['files'])==mp._GENERATED_FILES
    assert assets['identity_is_not_mesh_generation_or_fidelity_proof']


def test_generated_adapter_named_roundtrip_is_shape_specific(generated):
    assets=generated.closure();identity=assets['asset_shape_identity']
    model=mujoco.MjModel.from_xml_path(str(generated.hand/'mesh_hand.xml'));data=mujoco.MjData(model)
    with pytest.raises(ValueError,match='explicit expected shape'):
        mp.ManoSimStaticPoseAdapter(model,data,source_assets=assets,asset_shape_identity=identity)
    a=mp.ManoSimStaticPoseAdapter(model,data,source_assets=assets,asset_shape_identity=identity,
        expected_shape_sha256=generated.digest)
    t,local=target();state=a.stage(t,local,asset_shape_identity=identity)
    replay=a.stage_serialized(json.loads(json.dumps(state.serialize_pose())))
    np.testing.assert_array_equal(state.joint_anchor_world_m,replay.joint_anchor_world_m)
    assert replay.asset_shape_identity==identity
    assert a.forward_calls==2
    with pytest.raises(ValueError,match='Wrong shape'):
        a.stage(t,local,asset_shape_identity={**identity,'shape_asset_sha256':mp.EXPECTED_SHAPE_SHA256})
    payload=state.serialize_pose();payload['asset_shape_identity']['shape_asset_sha256']=mp.EXPECTED_SHAPE_SHA256
    with pytest.raises(ValueError,match='Wrong shape'):a.stage_serialized(payload)
    assert a.forward_calls==2


@pytest.mark.parametrize('value',['','f'*63,'F'*64,'x'*64,123,False])
def test_invalid_explicit_expected_shape_is_not_an_auto_discovery(source,value):
    with pytest.raises(ValueError,match='lowercase SHA-256'):
        mp.asset_shape_identity(source.shape,expected_shape_sha256=value)


@pytest.mark.parametrize('change',['missing','extra','shape','status','schema','root','xml_hash','other_xml_hash','mesh_hash','shape_hash','receipt_hash','binding_extra','escaping_xml','upstream','failure','artifact_missing','artifact_disagree','artifact_size'])
def test_generated_builder_inventory_is_strict_and_complete(generated,change):
    report=generated.report
    if change=='missing':report['generated_files'].pop(next(k for k in report['generated_files'] if k.endswith('mano_bone_15.obj')))
    elif change=='extra':report['generated_files']['extra.txt']='0'*64
    elif change=='shape':report['shape_asset_sha256']='0'*64
    elif change=='status':report['status']='PARTIAL'
    elif change=='schema':report['schema_version']='unverified_builder'
    elif change=='root':report['generated_root']=str(generated.root.parent)
    elif change in ('xml_hash','other_xml_hash','mesh_hash','shape_hash'):
        name={'xml_hash':'assets/literal_native/mano_rhand/mesh_hand.xml',
            'other_xml_hash':'assets/literal_native/mano_rhand/capsule_hand.xml',
            'mesh_hash':'assets/literal_native/mano_rhand/meshes/mano_bone_7.obj',
            'shape_hash':'assets/mano_rhand_shape.npy'}[change]
        (generated.root/name).write_bytes((generated.root/name).read_bytes()+b'\n')
    elif change=='receipt_hash':generated.report_path.write_text(generated.report_path.read_text()+'\n')
    elif change=='binding_extra':generated.binding['source']='unbound'
    elif change=='upstream':report['upstream_revision']='0'*40
    elif change=='failure':report['failure']={'type':'ManufacturedError'}
    elif change=='artifact_missing':report['artifacts'].pop()
    elif change=='artifact_disagree':report['artifacts'][0]['sha256']='0'*64
    elif change=='artifact_size':report['artifacts'][0]['bytes']+=1
    else:
        path=generated.hand/'mesh_hand.xml';xml=ET.parse(path).getroot()
        xml.find(".//mesh[@name='mano_bone_0']").set('file','../../../unlisted.obj')
        unlisted=generated.root/'unlisted.obj';unlisted.write_text('unlisted')
        path.write_text(ET.tostring(xml,encoding='unicode'))
        report['generated_files']['assets/literal_native/mano_rhand/mesh_hand.xml']=mp._sha(path)
    if change not in ('receipt_hash','binding_extra'):
        generated.binding.update(generated.write_report())
    with pytest.raises(ValueError):generated.closure()


def test_generated_binding_and_all_nineteen_assets_rechecked_before_forward(generated):
    assets=generated.closure();identity=assets['asset_shape_identity']
    model=mujoco.MjModel.from_xml_path(str(generated.hand/'mesh_hand.xml'));data=mujoco.MjData(model)
    a=mp.ManoSimStaticPoseAdapter(model,data,source_assets=assets,asset_shape_identity=identity,
        expected_shape_sha256=generated.digest)
    # The unselected variant remains in the closure and cannot change unnoticed.
    other=generated.hand/'capsule_hand.xml';other.write_text(other.read_text()+'\n')
    t,local=target()
    with pytest.raises(ValueError,match='source asset changed'):a.stage(t,local,asset_shape_identity=identity)
    assert a.forward_calls==0
    other.write_text(other.read_text()[:-1])
    generated.report_path.write_text(generated.report_path.read_text()+'\n')
    with pytest.raises(ValueError,match='receipt bytes changed'):a.stage(t,local,asset_shape_identity=identity)
    assert a.forward_calls==0


@pytest.mark.parametrize('text',['{"schema_version":1,"schema_version":2}', '{"unbound":NaN}', '{"unbound":1e999}'])
def test_generated_builder_report_rejects_ambiguous_json(generated,text):
    generated.report_path.write_text(text);generated.binding['sha256']=mp._sha(generated.report_path)
    with pytest.raises(ValueError):generated.closure()
