"""Manufactured receipts/arrays only; no figure rendering or native operations."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('saved_capsule_plotter',HERE/'plot_saved_capsule.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return m.binding(path)
def fixture(i,kind,dt):return dict(index=i,kind=kind,dt_s=dt,duration_s=.004,steps=round(.004/dt))
def manufactured(tmp_path,partial=False,no_worker=False):
    raw=tmp_path/'raw';raw.mkdir();plan=[fixture(i,k,dt) for i in (20,28) for dt in (.002,.001) for k in m.KINDS]
    p=tmp_path/'protocol.json';put(p,dict(schema_version=m.SOURCE_SCHEMA,raw_root=str(raw),method=dict(plan=plan)))
    base=dict(protocol_sha256=m.sha(p),freeze_commit='1'*40,physical_outcome=None,**{k:False for k in m.CLOSED_FLAGS})
    report=dict(schema_version=m.SOURCE_SCHEMA+'_report',status='HARD_STOP' if partial or no_worker else 'COMPLETED',qualification='NOT_QUALIFIED',worker_result=None,raw_result=None,**base)
    rows=[];schedule=[];cases=[]
    for index in (20,28):
        measurements=[]
        for f in [x for x in plan if x['index']==index]:
            key=m._fixture_key(f);directory=raw/key;directory.mkdir(parents=True)
            state='HARD_STOPPED_PARTIAL' if partial else 'COMPLETED_ASSESSED';schedule.append(dict(fixture=f,status=state));n=1 if partial else f['steps'];t=np.arange(n)*f['dt_s'];q=np.zeros((n,67));q[:,3::4]=1.;reference=q.copy();q[:,0]=.001;endpoint=q.copy();endpoint[:,2]=-.01308
            arrays=dict(pre_time_s=t,endpoint_time_s=t+f['dt_s'],pre_qpos=q,endpoint_qpos=endpoint,reference_qpos=reference,bone_angle_error_rad=np.full((n,16),.01),contact_active=np.ones(n,bool),normal_force_N=np.arange(n,dtype=float),engine_gap_m=np.full(n,-.0001))
            for name,a in [('partial_trajectory' if partial else 'trajectory',arrays),('static_command',dict(qpos=q[0])),('fixture_model',dict(native_target=reference[0],body_mass=np.array([.4]),gravity=np.array([0.,0.,-9.81])))]:
                path=directory/(name+'.npz');np.savez(path,**a);rows.append(dict(**m.binding(path),label='manufactured'))
            checks={k:True for k in m.STATIC_CHECKS};path=directory/'command_contract.json';rows.append(dict(**put(path,dict(static_checks=checks)),label='manufactured'))
            if not partial:measurements.append(dict(fixture=f,assessment=dict(status='FAIL',static_source_and_transmission_checks=checks)))
        cases.append(dict(index=index,measurements=measurements))
    if not no_worker:
        worker=dict(schema_version=m.SOURCE_SCHEMA+'_worker',status=report['status'],artifacts=rows,cases=cases,fixture_schedule=schedule,**base);rp=raw/'result.json';wb=put(rp,worker);copy=tmp_path/'worker_result.json';copy.write_bytes(rp.read_bytes());report.update(worker_result=m.binding(copy),raw_result=wb)
    path=tmp_path/'report.json';put(path,report);return path,p


def test_complete_failures_and_clock_aligned_values_retained(tmp_path):
    path,p=manufactured(tmp_path);saved=m.load_saved(path,p,m.sha(path));assert len(saved['fixtures'])==12
    item=saved['fixtures'][0];a=m.series(item)
    np.testing.assert_allclose(a['wrist_reference_error_mm'],1.)
    assert m._status(item)=='FAIL' and item['status']=='COMPLETED_ASSESSED'
    gravity=next(x for x in saved['fixtures'] if x['fixture']['kind']=='authored_gravity_equilibrium');g=m.series(gravity)
    assert g['expected_sag_mm']==pytest.approx(13.08);np.testing.assert_allclose(g['actual_sag_mm'],13.08)


def test_partial_trace_keeps_unknown_assessment_and_measured_denominator(tmp_path):
    path,p=manufactured(tmp_path,partial=True);saved=m.load_saved(path,p,m.sha(path));x=saved['fixtures'][0]
    assert x['partial'] and m._status(x)=='HARD_STOPPED_PARTIAL' and m.series(x)['recorded_steps']==1


def test_timeout_without_worker_does_not_invent_not_run_or_zero_counts(tmp_path):
    path,p=manufactured(tmp_path,no_worker=True);saved=m.load_saved(path,p,m.sha(path))
    assert all(x['status']=='NO_VERIFIED_SCHEDULE' and m.series(x) is None for x in saved['fixtures'])
    assert saved['report'].get('actual_counts') is None


def test_tampered_creation_time_array_refused(tmp_path):
    path,p=manufactured(tmp_path);r=m.read(path);w=m.read(r['worker_result']['path']);a=Path(w['artifacts'][0]['path']);a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):m.load_saved(path,p,m.sha(path))


def test_report_sha_and_worker_freeze_mismatch_refused(tmp_path):
    path,p=manufactured(tmp_path)
    with pytest.raises(ValueError,match='report SHA'):m.load_saved(path,p,'0'*64)
    r=m.read(path);w=m.read(r['worker_result']['path']);w['freeze_commit']='2'*40
    put(Path(r['worker_result']['path']),w);r['worker_result']=m.binding(r['worker_result']['path']);put(path,r)
    with pytest.raises(ValueError,match='identity'):m.load_saved(path,p,m.sha(path))


def test_contact_gap_omits_absent_contact_placeholder(tmp_path):
    path,p=manufactured(tmp_path);saved=m.load_saved(path,p,m.sha(path));x=next(x for x in saved['fixtures'] if x['fixture']['kind']=='isolated_capsule_plane');x['arrays']['trajectory']['contact_active'][0]=False
    a=m.series(x);assert np.isnan(a['active_engine_gap_mm'][0]) and a['active_steps']==1


def test_malformed_clock_and_truncated_completed_trace_refused(tmp_path):
    path,p=manufactured(tmp_path);saved=m.load_saved(path,p,m.sha(path));x=saved['fixtures'][0];x['arrays']['trajectory']['endpoint_time_s'][0]+=.001
    with pytest.raises(ValueError,match='clocks'):m.series(x)
    x['arrays']['trajectory']['endpoint_time_s'][0]-=.001;x['fixture']['steps']+=1
    with pytest.raises(ValueError,match='truncated'):m.series(x)


def test_imports_only_saved_data_dependencies():
    assert not any(k in sys.modules for k in ('mujoco','torch','manotorch','trimesh','scipy'))


@pytest.mark.parametrize('schema',m.SOURCE_SCHEMAS[1:])
def test_separately_identified_new_schema_uses_same_saved_channels(tmp_path,schema):
    path,p=manufactured(tmp_path);r=m.read(path);protocol=m.read(p);protocol['schema_version']=schema;put(p,protocol)
    worker=m.read(r['worker_result']['path']);worker['schema_version']=schema+'_worker';worker['protocol_sha256']=m.sha(p)
    put(Path(r['worker_result']['path']),worker);put(Path(r['raw_result']['path']),worker)
    r.update(schema_version=schema+'_report',protocol_sha256=m.sha(p),worker_result=m.binding(r['worker_result']['path']),raw_result=m.binding(r['raw_result']['path']));put(path,r)
    saved=m.load_saved(path,p,m.sha(path));assert saved['protocol']['schema_version']==schema
    assert len(saved['fixtures'])==12 and m._status(saved['fixtures'][0])=='FAIL'
