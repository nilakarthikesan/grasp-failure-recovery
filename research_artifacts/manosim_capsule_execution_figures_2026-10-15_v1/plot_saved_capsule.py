"""Saved-array figures only. No model, skin, FK, contact query, replay or simulation.

The source experiment's reported assessment remains authoritative. Plotting does
not recalculate qualification or promote the capsule model to full-skin parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA='manosim_capsule_saved_figures_v1'
SOURCE_SCHEMA='manosim_authored_capsule_execution_v1'
SOURCE_SCHEMAS=(SOURCE_SCHEMA,'manosim_authored_capsule_execution_v2','manosim_authored_capsule_execution_v3')
KINDS=('zero_gravity_tracking','authored_gravity_equilibrium','isolated_capsule_plane')
STATIC_CHECKS=('qpos','anchors','rotations','transmission','zero_servo_force')
CLOSED_FLAGS=('accepted_mapping','native_task_contact_qualified','full_skin_parity_claimed','new_collection_or_training_allowed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    def reject(v):raise ValueError('Nonfinite JSON')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=reject)
def binding(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def verify_row(row):
    if not {'path','sha256','bytes'}<=set(row) or binding(row['path'])!={k:row[k] for k in ('path','sha256','bytes')}:raise ValueError('Creation-time artifact changed')
def _fixture_key(f):return f"{f['index']}/{f['kind']}_{round(1/f['dt_s'])}"
def _same_identity(x,p,protocol_sha):
    if x.get('protocol_sha256')!=protocol_sha or not isinstance(x.get('freeze_commit'),str) or len(x['freeze_commit'])!=40:raise ValueError('Saved experiment identity differs')
    if x.get('physical_outcome') is not None or any(x.get(k) is not False for k in CLOSED_FLAGS):raise ValueError('Source scope/admission differs')
def _arrays(row):
    verify_row(row)
    with np.load(row['path'],allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    if any(v.dtype.hasobject or not np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite/object saved array')
    return a

def load_saved(report_path,protocol_path,expected_report_sha256):
    """Read only verified result/creation-time artifacts; missing data is explicit."""
    report_path=Path(report_path).resolve();protocol_path=Path(protocol_path).resolve()
    if sha(report_path)!=expected_report_sha256:raise ValueError('Explicit report SHA differs')
    report=read(report_path);p=read(protocol_path);ph=sha(protocol_path)
    schema=p.get('schema_version')
    if schema not in SOURCE_SCHEMAS or report.get('schema_version')!=schema+'_report' or report.get('status') not in ('COMPLETED','HARD_STOP'):raise ValueError('Unsupported experiment report')
    _same_identity(report,p,ph)
    plan=p['method']['plan']
    if len(plan)!=12 or {f['index'] for f in plan}!={20,28} or {f['kind'] for f in plan}!=set(KINDS) or {f['dt_s'] for f in plan}!={.002,.001} or len({_fixture_key(f) for f in plan})!=12:raise ValueError('Exact declared12-fixture plan required')
    raw=Path(p['raw_root']).resolve();worker=None;artifacts={};measurements={};schedules={}
    if report.get('worker_result'):
        verify_row(report['worker_result']);worker=read(report['worker_result']['path']);_same_identity(worker,p,ph)
        if worker.get('schema_version')!=schema+'_worker' or worker.get('freeze_commit')!=report['freeze_commit'] or worker.get('status') not in ('COMPLETED','HARD_STOP') or (report['status']=='COMPLETED' and worker.get('status')!='COMPLETED'):raise ValueError('Worker/parent identity differs')
        if report.get('raw_result'):
            verify_row(report['raw_result'])
            if report['raw_result']['sha256']!=report['worker_result']['sha256']:raise ValueError('Portable/raw worker bytes differ')
        for a in worker.get('artifacts',[]):
            path=Path(a['path'])
            if path.is_symlink() or not path.resolve().is_relative_to(raw) or str(path.resolve()) in artifacts:raise ValueError('Artifact escape or duplicate')
            verify_row(a);artifacts[str(path.resolve())]=a
        for c in worker.get('cases',[]):
            for m in c.get('measurements',[]):
                key=_fixture_key(m['fixture'])
                if key in measurements:raise ValueError('Repeated assessed fixture')
                measurements[key]=m
        for row in worker.get('fixture_schedule',[]):
            key=_fixture_key(row['fixture'])
            if key in schedules or row['fixture'] not in plan:raise ValueError('Repeated/undeclared fixture schedule')
            schedules[key]=row['status']
    fixtures=[]
    for f in plan:
        key=_fixture_key(f);directory=raw/key;assessment=measurements.get(key)
        status=schedules.get(key,'NO_VERIFIED_SCHEDULE')
        if status=='COMPLETED_ASSESSED' and assessment is None:raise ValueError('Completed fixture assessment absent')
        item=dict(fixture=f,status=status,assessment=assessment,arrays={},static_checks={},partial=False)
        for name in ('static_command','fixture_model','trajectory','partial_trajectory'):
            row=artifacts.get(str(directory/(name+'.npz')))
            if row:item['arrays'][name]=_arrays(row)
        meta=artifacts.get(str(directory/'command_contract.json'))
        if meta:
            verify_row(meta);item['static_checks']=read(meta['path'])['static_checks']
        elif assessment:item['static_checks']=assessment['assessment'].get('static_source_and_transmission_checks',{})
        if status=='COMPLETED_ASSESSED' and not {'trajectory','static_command','fixture_model'}<=item['arrays'].keys():raise ValueError('Completed fixture plot inputs missing')
        if 'partial_trajectory' in item['arrays'] and 'trajectory' not in item['arrays']:item['partial']=True
        fixtures.append(item)
    if report['status']=='COMPLETED' and (len(measurements)!=12 or any(x['status']!='COMPLETED_ASSESSED' for x in fixtures)):raise ValueError('Completed report lacks12 actual assessments')
    return dict(report=report,protocol=p,fixtures=fixtures,artifact_rows=list(artifacts.values()),report_binding=binding(report_path),protocol_binding=binding(protocol_path),worker_binding=report.get('worker_result'))


def series(item):
    """Same-clock display quantities; frames/errors use pre-step timestamps."""
    a=item['arrays'].get('trajectory',item['arrays'].get('partial_trajectory'))
    if a is None:return None
    f=item['fixture'];n=len(a['pre_time_s'])
    required={'pre_time_s':(n,),'endpoint_time_s':(n,),'pre_qpos':(n,67),'endpoint_qpos':(n,67),'reference_qpos':(n,67),'bone_angle_error_rad':(n,16)}
    if n<1 or n>f['steps']:raise ValueError('Recorded trajectory denominator differs')
    for k,shape in required.items():
        if k not in a or a[k].shape!=shape or not np.isfinite(a[k]).all():raise ValueError('Malformed once-per-step plot channel: '+k)
    pre=a['pre_time_s'];end=a['endpoint_time_s'];dt=f['dt_s']
    if np.max(np.abs(pre-np.arange(n)*dt))>1e-10 or np.max(np.abs(end-pre-dt))>1e-10:raise ValueError('Recorded force/state clocks differ')
    if not item['partial'] and n!=f['steps']:raise ValueError('Completed trajectory truncated')
    s=dict(pre_s=pre,endpoint_s=end,wrist_reference_error_mm=1e3*np.linalg.norm(a['pre_qpos'][:,:3]-a['reference_qpos'][:,:3],axis=1),bone_target_max_deg=np.rad2deg(a['bone_angle_error_rad'].max(axis=1)),recorded_steps=n)
    model=item['arrays'].get('fixture_model')
    if f['kind']=='authored_gravity_equilibrium' and model is not None:
        goal=np.asarray(model['native_target']);mass=float(model['body_mass'].sum());gravity=np.asarray(model['gravity']);kp=300.
        # Authored world-slide kp is protocol-bound, not a fitted equilibrium.
        if goal.shape!=(67,) or gravity.shape!=(3,) or not np.isfinite(mass) or mass<=0:raise ValueError('Malformed saved equilibrium inputs')
        sag=mass*abs(float(gravity[2]))/kp;expected=goal[:3].copy();expected[2]-=sag
        s.update(expected_sag_mm=1e3*sag,actual_sag_mm=1e3*(goal[2]-a['endpoint_qpos'][:,2]),gravity_equilibrium_error_mm=1e3*np.linalg.norm(a['endpoint_qpos'][:,:3]-expected,axis=1),expected_equilibrium_world_m=expected.tolist(),total_mass_kg=mass,authored_wrist_kp_N_per_m=kp)
    if f['kind']=='isolated_capsule_plane':
        for k in ('contact_active','normal_force_N','engine_gap_m'):
            if k not in a or a[k].shape!=(n,) or not np.isfinite(a[k]).all():raise ValueError('Malformed saved contact plot channel')
        active=a['contact_active'].astype(bool);gap=np.where(active,1e3*a['engine_gap_m'],np.nan)
        s.update(normal_force_N=a['normal_force_N'],active_engine_gap_mm=gap,contact_active=active,active_steps=int(active.sum()))
    return s


def _style(item):
    dt=item['fixture']['dt_s'];return dict(color='#2166ac' if dt==.002 else '#b2182b',linestyle='--' if item['partial'] else '-',label=f"{1e3*dt:g} ms"+(' PARTIAL' if item['partial'] else ''))
def _status(item):
    if item['assessment']:return item['assessment']['assessment']['status']
    return item['status']
def _panel_note(ax,items):
    missing=[f"{1e3*x['fixture']['dt_s']:g} ms: {_status(x)}" for x in items if series(x) is None]
    if missing:ax.text(.03,.97,'\n'.join(missing),transform=ax.transAxes,va='top',fontsize=8,color='#a50f15')
def render(saved,output):
    """Invoke only after the actual once-only experiment; writes a fresh directory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Fresh figure output required')
    out.mkdir(parents=True);rows=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    report=saved['report'];header=f"Saved capsule fixtures — {report['status']} / {report.get('qualification','NOT_QUALIFIED')}"
    footer='Two saved development inputs; authored capsule model. No full-skin parity, human replay, task collection or training qualification.'
    def publish(fig,name):
        fig.text(.5,.015,footer,ha='center',fontsize=8);fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(out/(name+'.png'));fig.savefig(out/(name+'.pdf'));plt.close(fig)
        rows.extend(dict(**binding(out/(name+ext)),label=name) for ext in ('.png','.pdf'))
    for family,name in [('zero_gravity_tracking','tracking'),('authored_gravity_equilibrium','gravity_sag'),('isolated_capsule_plane','contact')]:
        fig,axes=plt.subplots(2,2,figsize=(11,6.6),squeeze=False);fig.suptitle(header+'\n'+family.replace('_',' '),fontsize=12)
        for i,index in enumerate((20,28)):
            items=[x for x in saved['fixtures'] if x['fixture']['index']==index and x['fixture']['kind']==family]
            for item in items:
                s=series(item)
                if s is None:continue
                style=_style(item);label=style.pop('label')+' — '+_status(item)
                if family=='zero_gravity_tracking':
                    axes[i,0].plot(s['pre_s'],s['wrist_reference_error_mm'],label=label,**style);axes[i,1].plot(s['pre_s'],s['bone_target_max_deg'],label=label,**style)
                elif family=='authored_gravity_equilibrium':
                    if 'actual_sag_mm' not in s:continue
                    axes[i,0].plot(s['endpoint_s'],s['actual_sag_mm'],label=label,**style);axes[i,0].axhline(s['expected_sag_mm'],color='#555',ls=':',lw=1,label='mg / authored kp' if item is items[0] else None)
                    axes[i,1].plot(s['endpoint_s'],s['gravity_equilibrium_error_mm'],label=label,**style)
                else:
                    axes[i,0].plot(s['pre_s'],s['normal_force_N'],label=label,**style);axes[i,1].plot(s['pre_s'],s['active_engine_gap_mm'],label=label,**style)
            ylabel=('Wrist error to command [mm]','Max bone angle to goal [deg]') if family=='zero_gravity_tracking' else (('Downward wrist sag [mm]','Wrist equilibrium error [mm]') if family=='authored_gravity_equilibrium' else ('Normal reaction [N]','Gap at active contacts [mm]'))
            for j in range(2):
                ax=axes[i,j];ax.set(xlabel='Time [s]',ylabel=ylabel[j],title=f'Input {index}');ax.grid(alpha=.22);_panel_note(ax,items)
                if ax.lines:ax.legend(fontsize=7,loc='best')
                if family!='isolated_capsule_plane':
                    stop=items[0]['fixture']['duration_s'];tail=stop-.1;ax.axvspan(tail,stop,color='#999',alpha=.1)
                    if j==1 or family=='zero_gravity_tracking':ax.plot([tail,stop],[2,2],color='#333',ls=':',lw=1)
                elif j==1:ax.axhline(0,color='#777',ls=':',lw=1)
        publish(fig,name)
    fig,ax=plt.subplots(figsize=(11,7));fig.suptitle(header+'\nRecorded static gates and actual fixture outcomes',fontsize=12)
    matrix=[];labels=[]
    for item in saved['fixtures']:
        matrix.append([1 if item['static_checks'].get(k) is True else 0 if item['static_checks'].get(k) is False else 2 for k in STATIC_CHECKS])
        f=item['fixture'];labels.append(f"{f['index']} | {f['kind'].replace('_',' ')} | {1e3*f['dt_s']:g} ms | {_status(item)}")
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=2,cmap=ListedColormap(['#ef8a62','#a6dba0','#dddddd']))
    ax.set_xticks(range(5),['qpos identity','anchor ≤5 µm','R component ≤1e−6','length ≤1e−10','servo force ≤1e−8'])
    ax.set_yticks(range(len(labels)),labels);ax.tick_params(length=0)
    for i,row in enumerate(matrix):
        for j,x in enumerate(row):ax.text(j,i,('FAIL','PASS','NOT RECORDED')[x],ha='center',va='center',fontsize=8)
    publish(fig,'static_scope')
    derived=[dict(fixture=x['fixture'],status=x['status'],assessment_status=_status(x),plotted_steps=series(x)['recorded_steps'] if series(x) else None,partial=x['partial']) for x in saved['fixtures']]
    manifest=dict(schema_version=SCHEMA,source_experiment_schema=saved['protocol']['schema_version'],source_report=saved['report_binding'],source_protocol=saved['protocol_binding'],source_worker=saved['worker_binding'],source_actual_counts=report.get('actual_counts'),source_status=report['status'],source_qualification=report.get('qualification'),fixtures=derived,creation_time_artifacts_verified=len(saved['artifact_rows']),independent_scientific_readback_performed_by_plotter=False,native_or_model_or_FK_or_query_or_physics_calls=0,artifacts=rows,**{k:False for k in CLOSED_FLAGS})
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',required=True,type=Path);parser.add_argument('--report-sha256',required=True);parser.add_argument('--protocol',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
    saved=load_saved(a.report,a.protocol,a.report_sha256);render(saved,a.output)
