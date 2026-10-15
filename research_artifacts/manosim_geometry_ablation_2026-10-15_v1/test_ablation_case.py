"""Manufactured topology/transform/guard controls; no native evaluation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('manufactured_saved_ablation',HERE/'ablation_case.py')
ablation=importlib.util.module_from_spec(spec);spec.loader.exec_module(ablation)


@pytest.fixture(autouse=True)
def forbid_engines(monkeypatch):
    def no(*a,**k):pytest.fail('No native model/FK/geometry work in manufactured controls')
    for name in ('mj_forward','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,no)


def test_source_faces_map_by_tuple_not_position_and_caps_stay_distinct():
    closed=np.array([[0,1,2],[2,1,3],[1,4,3],[0,2,4]])
    source=np.array([[4,2,0],[3,1,2],[1,2,0]])
    np.testing.assert_array_equal(ablation.source_face_subset(closed,source),[3,1,0])
    with pytest.raises(ValueError):ablation.source_face_subset(closed,np.array([[0,3,4]]))
    with pytest.raises(ValueError):ablation.source_face_subset(closed,np.array([[0,1,2],[2,1,0]]))


def test_group_sum_semantics_and_mean_weight_face_tie_are_preserved():
    weights=np.zeros((3,16),np.float64)
    weights[:,0]=.4;weights[:,13:16]=.2
    vertex,labels=ablation.anatomical_labels(weights,np.array([[0,1,2]]))
    assert list(vertex)==[5,5,5] and list(labels)==[5]
    assert list(np.argmax(weights,axis=1))==[0,0,0]
    # Every vertex votes for a different group. Mean group weight, not lowest
    # part index or individual-bone ownership, resolves the face majority tie.
    weights=np.zeros((3,16),np.float64);weights[0,0]=1;weights[1,1]=1.1;weights[2,4]=1
    assert ablation.anatomical_labels(weights,np.array([[0,1,2]]))[1][0]==1


def test_dropped_face_orphan_support_is_separate_from_whole_owner_cloud():
    owner=np.array([0,0,0,0,1,1,1])
    source=np.array([[0,1,2],[1,3,4],[4,5,6]])
    keep,ledger=ablation.support_ledger(owner,source)
    np.testing.assert_array_equal(keep,[True,False,True])
    assert ledger[0]==dict(bone=0,assigned_owner_vertices=4,referenced_retained_vertices=3,orphan_owner_vertices=1)
    assert ledger[1]['orphan_owner_vertices']==0
    # The orphan remains owned; a later hull may include it. It is not silently
    # discarded to make the hull comparable to retained triangle support.
    assert (owner==0).sum()==4


def test_lbs_and_rigid_use_saved_transforms_common_world_translation_once():
    neutral=np.array([[1.,0,0],[0.,1.,0]])
    rest=np.zeros((16,3));rotations=np.tile(np.eye(3),(16,1,1));rotations[1]=np.diag([-1.,-1.,1.])
    anchors=np.tile([10.,20.,30.],(16,1));weights=np.zeros((2,16));weights[:,:2]=.5
    lbs,rigid=ablation.articulated_vertices(neutral,rest,rotations,anchors,weights,np.array([0,1]))
    np.testing.assert_array_equal(lbs,anchors[:2])
    np.testing.assert_array_equal(rigid,[[11.,20.,30.],[10.,19.,30.]])
    changed=anchors+[7.,-3.,2.]
    lbs2,rigid2=ablation.articulated_vertices(neutral,rest,rotations,changed,weights,np.array([0,1]))
    np.testing.assert_array_equal(lbs2-lbs,np.tile([7.,-3.,2.],(2,1)))
    np.testing.assert_array_equal(rigid2-rigid,np.tile([7.,-3.,2.],(2,1)))


def test_obj_readback_retains_literal_coordinates_and_faces(tmp_path):
    path=tmp_path/'mesh.obj';path.write_text('v 0.123457 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n')
    vertices,faces=ablation.read_obj(path)
    assert vertices[0,0]==.123457;np.testing.assert_array_equal(faces,[[0,1,2]])
    path.write_text(path.read_text().replace('f 1 2 3','f -1 2 3'))
    with pytest.raises(ValueError):ablation.read_obj(path)


def test_xml_replay_includes_literal_palm_geometry_offset_and_rejects_unhandled_transform(tmp_path):
    assets=''.join(f'<mesh name="bone{i}" file="bone{i}.obj"/>' for i in range(16))
    bodies=''.join(f'<body name="mano_body_{i}"><geom type="mesh" mesh="bone{i}" pos="{.05 if i==0 else 0} 0 0"/></body>' for i in range(16))
    path=tmp_path/'mesh_hand.xml';path.write_text(f'<mujoco><asset>{assets}</asset><worldbody>{bodies}</worldbody></mujoco>')
    placements=ablation.xml_mesh_placements(path)
    np.testing.assert_array_equal(placements[0]['position'],[.05,0,0])
    assert set(placements)==set(range(16))
    path.write_text(path.read_text().replace('type="mesh"','type="mesh" quat="1 0 0 0"',1))
    with pytest.raises(ValueError,match='orientation'):ablation.xml_mesh_placements(path)


class Rules:
    @staticmethod
    def field_status(native,mapped,radius):
        near=native[:,:,None]<=np.array([.001,.002,.005]);mnear=mapped[:,:,None]<=np.array([.001,.002,.005])
        required=np.ones_like(near,bool);zero=np.zeros_like(near,bool)
        return dict(native_near=near,native_far=~near,native_ambiguous=zero,required=required,
            preserved=near==mnear,definite_lost=near&~mnear,definite_added=~near&mnear,unresolved_required=zero),{}


def test_reference_limit_and_every_mask_gate_even_when_distances_close():
    native=np.full((2,6),.01);radius=np.zeros(2);saved=Rules.field_status(native,native,radius)[0]
    good=ablation.reference_control(native.copy(),native,radius,Rules,saved)
    assert good['passed'] and good['exact_masks'] and good['max_distance_error_m']==0
    shifted=native.copy();shifted[0,0]+=2e-12
    assert not ablation.reference_control(shifted,native,radius,Rules,saved)['passed']
    bad_saved={k:v.copy() for k,v in saved.items()};bad_saved['native_near'][0,0,0]=True
    assert not ablation.reference_control(native,native,radius,Rules,bad_saved)['passed']


def manufactured_case(tmp_path,monkeypatch,guard=None):
    case=ablation.AblationCase.__new__(ablation.AblationCase)
    case.raw=tmp_path;case.artifact_rows=[];case.source_receipt={};case.index=20
    case.counts=dict(full_field_scans=0,triangle_point_tests=0);case.max_tests=ablation.TRIANGLE_TEST_FACTOR*2
    case.query=np.array([[0.,0.,1.],[0.,0.,2.]]);case.radius=np.zeros(2);case.native_distance=np.full((2,6),.01)
    case.rules=Rules;case.actual=Rules.field_status(case.native_distance,case.native_distance,case.radius)[0]
    case.grid=dict(required_relationships=36);case.guard=guard or (lambda *a:None)
    calls=[]
    def query(points,tri,labels,nparts):calls.append(len(points));return np.full((len(points),6),.01),None,None
    case.kernel=SimpleNamespace(nearest_parts=query)
    monkeypatch.setitem(ablation.WITNESSES,20,((0,2,5),))
    geometry=dict(triangles_object_m=np.tile(np.array([[[0.,0,0],[1.,0,0],[0.,1,0]]]),(6,1,1)),
        anatomical_part_index=np.arange(6),native_face_id=np.arange(6),bone_owner=np.arange(6),source_vertex_ids=np.zeros((6,3),int))
    return case,geometry,calls


def test_query_budget_precedes_kernel_and_geometry_is_sealed_before_query(tmp_path,monkeypatch):
    calls=[]
    def guard(kind,value):
        calls.append((kind,value))
        if kind=='triangle_point_tests':raise RuntimeError('manufactured hard quota')
    case,geometry,kernel_calls=manufactured_case(tmp_path,monkeypatch,guard)
    with pytest.raises(RuntimeError,match='hard quota'):case._scan(ablation.STAGES[0],geometry)
    assert kernel_calls==[] and case.counts==dict(full_field_scans=1,triangle_point_tests=0)
    assert calls==[('full_field_scan','native_reference'),('triangle_point_tests',12)]
    assert len(case.artifact_rows)==1 and case.artifact_rows[0]['sha256']==ablation.sha(case.artifact_rows[0]['path'])
    assert Path(case.artifact_rows[0]['path']).name=='native_reference_geometry.npz'


def test_fixed_stage_order_and_scan_guard_precede_any_query(tmp_path,monkeypatch):
    def guard(*a):raise RuntimeError('scan forbidden')
    case,geometry,kernel=manufactured_case(tmp_path,monkeypatch,guard)
    with pytest.raises(RuntimeError,match='ordered'):case._scan(ablation.STAGES[1],geometry)
    with pytest.raises(RuntimeError,match='scan forbidden'):case._scan(ablation.STAGES[0],geometry)
    assert not kernel and case.counts['full_field_scans']==0


def test_failed_reference_stops_before_constructing_any_causal_stage(tmp_path,monkeypatch):
    case,geometry,calls=manufactured_case(tmp_path,monkeypatch)
    continued=[]
    def geometries():
        yield ablation.STAGES[0],geometry
        continued.append(True);yield ablation.STAGES[1],geometry
    case.geometries=geometries
    case.native_distance[0,0]+=.0001
    with pytest.raises(ValueError,match='reference replay failed'):case.run()
    assert continued==[] and case.counts['full_field_scans']==1 and len(calls)==1
    assert case.source_receipt['reference_control']['passed'] is False


def test_stage_has_no_admission_even_when_manufactured_fields_match(tmp_path,monkeypatch):
    case,geometry,_=manufactured_case(tmp_path,monkeypatch)
    row,distances=case._scan(ablation.STAGES[0],geometry)
    assert all(row[k] is False for k in ablation.FLAGS)
    assert row['face_inventory']==dict.fromkeys(ablation.PARTS,1)
    assert len(case.artifact_rows)==2 and case.counts['triangle_point_tests']==12
