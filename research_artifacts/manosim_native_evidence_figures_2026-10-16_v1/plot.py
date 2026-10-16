"""Saved-data scientific figure. No simulator/model/geometry work."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def render(new_profile,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():raise FileExistsError('Fresh figure directory required')
    versions={'initial':'manosim_native_physics_profile_2026-10-16_v1','corrected':new_profile}
    inputs={};reports={};raws={}
    for key,name in versions.items():
        report_path=ROOT/'research_artifacts'/name/'run_v1/report.json'
        independent=report_path.parent/'independent_readback.json'
        report=read(report_path);receipt=read(independent)
        if receipt.get('errors') or report['status']!='COMPLETED' or receipt.get('status')!='VERIFIED_COMPLETED_BOUNDED_NATIVE_PROFILE_SAVED_READBACK':raise ValueError('Completed independently checked run required')
        expected=dict(path=str(report_path),sha256=sha(report_path),bytes=report_path.stat().st_size)
        if receipt.get('report')!=expected or receipt.get('freeze_commit')!=report['freeze_commit']:raise ValueError('Independent receipt must bind this exact report')
        if key=='corrected' and (report['qualification']!='PASS' or receipt.get('reported_bounded_qualification')!='PASS'):raise ValueError('Corrected profile must independently qualify before rendering a success figure')
        reports[key]=report;raws[key]=ROOT/'runs'/name/'run_v1/raw'
        for path in (report_path,independent):inputs[str(path)]=sha(path)
        worker=read(report['worker_result']['path'])
        for a in worker['artifacts']:
            path=Path(a['path'])
            if path.stat().st_size!=a['bytes'] or sha(path)!=a['sha256']:raise ValueError('Write-time figure input changed')
            inputs[str(path)]=a['sha256']
    old_raw=ROOT/'runs/manosim_capsule_execution_2026-10-15_v3/run_v1/raw'
    baseline=old_raw/'20/authored_gravity_equilibrium_500/trajectory.npz'
    baseline_model=baseline.parent/'fixture_model.npz'
    for path in (baseline,baseline_model):inputs[str(path)]=sha(path)
    old=arrays(baseline);oldgoal=arrays(baseline_model)['native_target']
    current=arrays(raws['corrected']/'20/gravity_motion_hold_500/trajectory.npz')
    goal=arrays(raws['corrected']/'20/gravity_motion_hold_500/fixture_model.npz')['native_target']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,ax=plt.subplots(2,2,figsize=(12.8,8.2),layout='constrained')
    colors={'initial':'#9a3434','corrected':'#1b637a'}
    a=ax[0,0];a.plot(old['endpoint_time_s'],1000*(old['endpoint_qpos'][:,2]-oldgoal[2]),color='#6b6b6b',label='Authored gravity hold, uncompensated')
    a.plot(current['endpoint_time_s'],1000*(current['endpoint_qpos'][:,2]-goal[2]),color=colors['corrected'],label='Compensated motion then hold')
    a.axhline(0,color='#111111',lw=.7);a.set(xlim=(.65,1.5),ylim=(-14,.5),xlabel='Time (s)',ylabel='Wrist height relative to target (mm)',title='A  Gravity compensation removes the measured sag');a.legend(fontsize=8,loc='lower right')
    a.text(.04,.91,'Representative source 20, 2 ms; different declared motion/hold conditions',transform=a.transAxes,fontsize=8,color='#555555')
    a=ax[0,1]
    for key in versions:
        x=arrays(raws[key]/'20/two_finger_planes_500/trajectory.npz')
        for j,finger in enumerate(('Index distal','Middle distal')):
            a.plot(x['pre_time_s'],1000*x['normal_force_N'][:,j],color=colors[key],ls='-' if j==0 else '--',label=f'{key.title()}: {finger}')
    a.set(xlabel='Force clock: preceding interval start (s)',ylabel='Normal reaction (mN)',title='B  Simultaneous finger reactions');a.legend(fontsize=8);a.axvspan(.65,.75,color='#999999',alpha=.12);a.text(.03,.94,'Shading: fixed final 0.1 s contact assessment',transform=a.transAxes,fontsize=8,color='#555555')
    a=ax[1,0]
    for key in versions:
        for i in (20,28):
            x=arrays(raws[key]/f'{i}/guarded_sphere_table_approach_500/trajectory.npz')
            a.plot(x['endpoint_time_s'],1000*x['wrist_goal_error_m'],color=colors[key],ls='-' if i==20 else '--',label=f'{key.title()}: source {i}')
    a.axhline(2,color='#111111',lw=.8,ls=':',label='Unchanged 2 mm hold limit');a.axvline(1,color='#777777',lw=.8,ls=':');a.set(xlabel='State endpoint time (s)',ylabel='Distance from final wrist goal (mm)',title='C  Approach motion followed by a declared settling period');a.set_ylim(-.4,10);a.legend(fontsize=8)
    a.text(.98,.65,'Initial and corrected motion overlap through 1 s',ha='right',transform=a.transAxes,fontsize=8,color='#555555')
    a=ax[1,1]
    for i in (20,28):
        for dt,style in ((500,'-'),(1000,'--')):
            x=arrays(raws['corrected']/f'{i}/gravity_motion_hold_{dt}/trajectory.npz')
            mask=(x['pre_time_s']>=.8)&(x['pre_time_s']<=1.5)
            a.plot(x['pre_time_s'][mask],np.rad2deg(x['bone_angle_error_rad'][mask].max(axis=1)),color='#1b637a' if i==20 else '#6b6b6b',ls=style,label=f'Source {i}, {1000/dt:g} ms')
    a.set(xlim=(.8,1.5),xlabel='Body-frame force clock (s)',ylabel='Maximum of 16 bone angle errors (degrees)',title='D  All-bone orientation error during compensated hold');a.legend(fontsize=8)
    fig.suptitle('Native MANOSim validation evidence — component fixtures',fontsize=16)
    fig.supxlabel('Saved trajectories only · Initial and corrected protocols stay separate · No object-task or predictor accuracy claim',fontsize=9)
    output.mkdir(parents=True)
    fig.savefig(output/'native_validation.png',dpi=180);plt.close(fig)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError('Figure source drift')
    manifest=dict(schema_version='native_saved_physics_evidence_figure_v1',profiles=versions,input_hashes=inputs,
        output_hashes={p.name:sha(p) for p in output.iterdir()},representative_force_plot='source20,2ms; all conditions remain in linked independent receipts',
        native_or_model_or_physics_calls=0,source_sha256=sha(__file__))
    with (output/'manifest.json').open('x') as file:json.dump(manifest,file,indent=2,sort_keys=True);file.write('\n')
    print(json.dumps({'figure':str(output/'native_validation.png'),'input_bindings':len(inputs),'manifest_sha256':sha(output/'manifest.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corrected-profile',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.corrected_profile,a.output)
