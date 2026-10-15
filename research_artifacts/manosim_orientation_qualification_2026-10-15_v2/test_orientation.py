"""Manufactured-only controls. No actual saved native query or engine import."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')

def load(name,file):
    s=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
C=load('orientation_manufactured_core','orientation_core.py')
H=load('orientation_manufactured_harness','run_orientation.py')

def fake_feature(score=1.):
    d=np.array([[.002,.05,.06,.07,.08,.09]])
    return dict(distance_by_part=d,global_point_valid=np.array([True]),global_nearest_part=np.array([0]),global_score=np.array([score]),global_distance=np.array([.002]),
        cell_candidate_count=np.ones((1,6),int),cell_competing_normal=np.zeros((1,6),bool),nearest_facet_edge_clearance_m=np.ones((1,6)),query_inside_hand=np.array([False]),
        closest_visual_inside=np.array([False]),closest_visual_distance_m=np.array([.01]),cell_facet_score_min=np.full((1,6),score),cell_facet_score_max=np.full((1,6),score))

def masks():
    near=np.zeros((1,6,3),bool);near[:,0,2]=True;far=~near
    return near,far

def test_matching_never_promotes_pass():
    near,far=masks();a=C.compare(fake_feature(),fake_feature(),np.array([1e-5]),near,far,near)
    r=C.summary(a,np.array([1.]))
    assert a['point_eligible'][0,0,2] and a['cell_stable_matching'][0,0,2]
    assert r['strict_mapping_gate']=='INCOMPLETE' and not r['native_fidelity_accepted']

def test_reliable_sign_reversal_fails():
    near,far=masks();a=C.compare(fake_feature(),fake_feature(-1),np.array([1e-5]),near,far,near)
    assert a['point_sign_reversal'][0,0,2] and a['cell_sign_reversal'][0,0,2]
    assert C.summary(a,np.ones(1))['strict_mapping_gate']=='FAIL'

@pytest.mark.parametrize('reason',['global_point_valid','cell_competing_normal','nearest_facet_edge_clearance_m','closest_visual_distance_m','cell_candidate_count'])
def test_unreliable_or_edge_never_cell_proof(reason):
    f=fake_feature()
    if reason=='global_point_valid':f[reason][:]=False
    elif reason=='cell_competing_normal':f[reason][:]=True
    elif reason=='cell_candidate_count':f[reason][:]=2
    else:f[reason][:]=0
    near,far=masks();a=C.compare(f,fake_feature(),np.array([1e-5]),near,far,near)
    assert a['cell_unresolved'][0,0,2] and not a['cell_stable_matching'].any()

def test_proximity_ambiguous_is_retained_unresolved():
    near,far=masks();near[:]=False;far[0,0,1]=False
    a=C.compare(fake_feature(),fake_feature(),np.array([.001]),near,far,near)
    assert a['orientation_possible_required'][0,0,1] and a['cell_unresolved'][0,0,1]

def test_tangent_is_unresolved():
    near,far=masks();a=C.compare(fake_feature(0.),fake_feature(),np.array([1e-5]),near,far,near)
    assert not a['point_eligible'].any() and a['cell_unresolved'][0,0,2]

def triangle_fixture():
    t=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    tri=np.repeat(t[None],7,axis=0);labels=np.array([0,1,2,3,4,5,0],np.int64)
    normals=np.tile([0.,0.,1.],(7,1));normals[-1]=[0,0,-1]
    d=np.ones((1,7));d[0,6]=1.+C.E*.5
    cp=np.repeat(np.array([[[.2,.2,0.]]]),7,axis=1)
    return tri,labels,normals,d,cp

def test_all_ties_not_first_face():
    tri,labels,norm,d,cp=triangle_fixture()
    out=C.reduce_candidates(np.array([[.2,.2,1.]]),np.array([.001]),tri,labels,norm,np.array([[0.,0.,1.]]),d,cp)
    assert out['point_tie_count'][0,0]==2 and out['point_competing_normal'][0,0]
    assert out['cell_facet_score_min'][0,0]==-1 and out['cell_facet_score_max'][0,0]==1

def test_candidate_bound_has_two_error_terms():
    tri,labels,norm,d,cp=triangle_fixture();r=1e-4;d[0,6]=1+2*r+1.5*C.E
    out=C.reduce_candidates(np.zeros((1,3)),np.array([r]),tri,labels,norm,np.array([[0.,0.,1.]]),d,cp)
    assert out['cell_candidate_count'][0,0]==2

def test_tiny_manufactured_jit_closest_and_full_tie_inventory():
    source=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py'
    p=C.primitives(source);triangle,reduce=C.compile_chunk_helpers(p)
    tri,labels,norm,_,_=triangle_fixture();q=np.array([[.2,.2,1.]])
    d,cp=triangle(q,tri,p.closest_triangle)
    out=dict(zip(C.REDUCED_KEYS,reduce(q,np.array([.001]),tri,labels,norm,np.array([[0.,0.,1.]]),d,cp)))
    assert np.allclose(d,1.) and out['point_tie_count'][0,0]==2
    assert out['point_competing_normal'][0,0]

def test_primitives_reject_unsafe_dependency(tmp_path):
    p=tmp_path/'bad.py';p.write_text('def closest_triangle(a):\n return evil(a)\ndef nearest_parts(a):\n return a\ndef winding_inside(a):\n return a\n')
    with pytest.raises(ValueError,match='dependency'):C.primitives(p)

def test_degenerate_is_not_dropped():
    with pytest.raises(ValueError,match='Degenerate'):C.normal_faces(np.zeros((1,3,3)))

def test_zero_volume_has_no_outward_claim():
    with pytest.raises(ValueError,match='volume'):C.normal_faces(np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]))

def test_before_work_budget_refuses():
    b=H.Budget({'hand_triangle_tests':2});b('hand_triangle_tests',2)
    with pytest.raises(RuntimeError):b('hand_triangle_tests',1)
    assert b.count['hand_triangle_tests']==2

def test_creation_time_array_and_bytes_tamper(tmp_path):
    rows=[];p=tmp_path/'array.npz';H.npz_write(p,{'x':np.arange(3)},'manufactured',rows);H.verify_artifacts(rows)
    np.savez(p,x=np.arange(4))
    with pytest.raises(ValueError):H.verify_artifacts(rows)

def test_json_nonfinite_exponent_refused(tmp_path):
    p=tmp_path/'x.json';p.write_text('{"x":1e999}')
    with pytest.raises(ValueError):H.read(p)

def test_missing_part_refused():
    tri,labels,norm,d,cp=triangle_fixture();labels[:]=0
    with pytest.raises(ValueError,match='Missing anatomical'):C.reduce_candidates(np.zeros((1,3)),np.ones(1),tri,labels,norm,np.ones((1,3)),d,cp)

def test_original_full_grid_alignment_refuses(tmp_path):
    q={'query_object_m':np.zeros((1,3)),'native_distance_by_part_m':np.zeros((1,6)),'area_mm2':np.ones(1)}
    field=dict(cell_radius_m=np.ones(1),query_object_m=np.ones((1,3)),native_distance_by_part_m=np.zeros((1,6)))
    with pytest.raises(ValueError):H.check_queries(q,field,{})


def test_tiny_manufactured_complete_feature_pipeline():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,1.],[1.,1.,1.],[0.,1.,1.]])
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    tri=v[faces];labels=np.repeat(np.arange(6,dtype=np.int64),2)
    p=C.primitives(ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py')
    b=H.Budget(dict(hand_triangle_tests=12,hand_winding_triangle_tests=12,object_distance_triangle_tests=12,object_winding_triangle_tests=12))
    saved=[];out=C.features(np.array([[.2,.2,-.001]]),np.array([1e-5]),np.array([[0.,0.,1.]]),tri,labels,tri+np.array([0.,0.,-2.]),p,b,lambda a,z,x:saved.append((a,z)))
    assert saved==[(0,1)] and out['distance_by_part'].shape==(1,6)
    assert np.isclose(out['global_distance'][0],.001) and not out['query_inside_hand'][0]


def test_partial_npz_is_discoverable_not_claimed_creation_proof(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();p=raw/'partial.npz';p.write_bytes(b'partial')
    assert H.binding(p)['bytes']==7


def test_engine_import_finder_refuses_without_importing_engine():
    with pytest.raises(RuntimeError,match='Forbidden'):H.NoEngine().find_spec('mujoco')


def fake_parent(monkeypatch,tmp_path,final_failure=False):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    p=dict(output_root=str(tmp_path/'tracked'),raw_root=str(tmp_path/'raw'),bindings={},cases={str(i):{'limits':dict(hand_triangle_tests=10,hand_winding_triangle_tests=10,object_distance_triangle_tests=10,object_winding_triangle_tests=10)} for i in H.INDICES})
    monkeypatch.setattr(H,'validate',lambda path:(p,'frozen'))
    monkeypatch.setattr(H,'git_receipt',lambda path,p:'frozen')
    def child(argv,**kwargs):
        index=int(argv[-1]);raw=Path(p['raw_root'])/str(index);raw.mkdir()
        r=dict(schema_version=H.SCHEMA,source_inventory_index=index,protocol_sha256=H.sha(protocol),freeze_commit='frozen',status='HARD_STOP',physical_outcome=None,
            admission_allowed=False,native_fidelity_accepted=False,new_collection_or_training_allowed=False,
            forbidden_operations={k:0 for k in ('models','data','forwards','physics','MANO','HUG','retarget','fits','task_outcomes')},query_work_counts={k:0 for k in p['cases'][str(index)]['limits']},artifacts=[],
            failure={'type':'ManufacturedFailure','message':'after stage start'},stages=[{'stage':'native','status':'HARD_STOPPED_PARTIAL'}])
        H.write(raw/'result.json',r)
        return type('Child',(),{'returncode':1})()
    monkeypatch.setattr(H.subprocess,'run',child)
    if final_failure:
        def broken(bindings):raise ValueError('manufactured final bound mutation')
        monkeypatch.setattr(H,'verify',broken)
    return protocol,p


def test_attempted_failure_and_later_not_run_are_preserved(monkeypatch,tmp_path):
    protocol,p=fake_parent(monkeypatch,tmp_path)
    report=H.run(protocol)
    assert report['status']=='HARD_STOP'
    assert [x['status'] for x in report['cases']]==['HARD_STOPPED_PARTIAL','NOT_RUN_AFTER_HARD_STOP']
    assert H.read(Path(p['output_root'])/'20/result.json')['stages'][0]['status']=='HARD_STOPPED_PARTIAL'
    assert Path(p['output_root'],'STARTED.json').exists()
    with pytest.raises(FileExistsError):H.run(protocol)


def test_final_validation_failure_still_writes_hard_stop(monkeypatch,tmp_path):
    protocol,p=fake_parent(monkeypatch,tmp_path,True)
    report=H.run(protocol)
    assert report['status']=='HARD_STOP' and report['publication_failure']['type']=='ValueError'
    assert H.read(Path(p['output_root'])/'report.json')==report


def test_nearly_equal_tie_normals_straddling_tangent_deadzone_retained():
    tri,labels,norm,d,cp=triangle_fixture()
    norm[0]=[-1.2e-6,0.,np.sqrt(1-(1.2e-6)**2)]
    norm[6]=[-.4e-6,0.,np.sqrt(1-(.4e-6)**2)]
    out=C.reduce_candidates(np.zeros((1,3)),np.array([1e-5]),tri,labels,norm,np.array([[1.,0.,0.]]),d,cp)
    assert not out['point_competing_normal'][0,0]
    assert out['point_tie_score_min'][0,0]<C.DOT_EPS<out['point_tie_score_max'][0,0]
    assert (out['point_tie_score_min'][0,0]<=C.DOT_EPS and out['point_tie_score_max'][0,0]>=-C.DOT_EPS)


def layout_fixture():
    d=np.arange(12,dtype=float).reshape(2,6)/1000
    q=dict(query_object_m=np.zeros((2,3)),native_distance_by_part_m=np.column_stack([d,np.full((2,2),np.inf)]),area_mm2=np.ones(2))
    f=dict(query_object_m=q['query_object_m'].copy(),native_distance_by_part_m=d.copy(),cell_radius_m=np.ones(2)*.001)
    g=dict(triangles_object_m=np.zeros((1552,3,3)),anatomical_part_index=np.zeros(1552),native_face_id=np.arange(1552))
    return q,f,g


def test_exact_eight_slot_historical_layout_projects_only_fixed_hand_parts():
    q,f,g=layout_fixture();H.check_queries(q,f,g)
    np.testing.assert_array_equal(H.original_cache_hand_distances(q,2),f['native_distance_by_part_m'])


@pytest.mark.parametrize('mutation',['carrier_finite','missing_carriers','changed_hand_target','negative_inf_carrier'])
def test_cache_projection_never_hides_real_alignment_change(mutation):
    q,f,g=layout_fixture()
    if mutation=='carrier_finite':q['native_distance_by_part_m'][0,6]=0.
    elif mutation=='missing_carriers':q['native_distance_by_part_m']=q['native_distance_by_part_m'][:,:6]
    elif mutation=='changed_hand_target':f['native_distance_by_part_m'][0,0]+=.001
    else:q['native_distance_by_part_m'][0,7]=-np.inf
    with pytest.raises(ValueError):H.check_queries(q,f,g)
