"""Render sealed saved arrays only. No model, skin, FK or contact query."""
from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=ROOT/'runs/manosim_saved_validation_figures_2026-10-15_v2'
OUT.mkdir(parents=True,exist_ok=False)
INPUTS={}
def load(path):
    path=Path(path);INPUTS[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(fig,name):
    path=OUT/name;fig.savefig(path,dpi=170,bbox_inches='tight');plt.close(fig);return str(path)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12,9));fig.suptitle('Articulated MANO skin: saved native-state qualification',fontsize=16)
parents=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
states=[];derivatives=[]
for col,index in enumerate((20,28)):
    folder=ROOT/f'runs/manosim_articulated_skin_2026-10-15_v1/run_v1/raw/{index}'
    a=load(folder/'all109_skin_states.npz');j=load(folder/'all51_tangent_derivatives.npz');states.append(a);derivatives.append(j)
    ids=a['pose_ids'].astype(str);selected=int(np.flatnonzero(ids=='coupled_+1_5deg')[0])
    base=a['candidate_float64_vertices_world_m'][0];moved=a['candidate_float64_vertices_world_m'][selected];center=base.mean(0)
    b=(base-center)*1000;m=(moved-center)*1000;faces=a['faces']
    ax=fig.add_subplot(2,2,col+1,projection='3d')
    ax.add_collection3d(Poly3DCollection(b[faces],facecolors='#bbbbbb',edgecolors='#888888',linewidths=.08,alpha=.12))
    ax.add_collection3d(Poly3DCollection(m[faces],facecolors='#2674a8',edgecolors='#18496b',linewidths=.08,alpha=.55))
    for state,color in ((0,'#666666'),(selected,'#102e43')):
        points=(a['candidate_float64_joint_anchors_world_m'][state]-center)*1000
        for k in range(1,16):ax.plot(*points[[parents[k],k]].T,color=color,linewidth=1.2,alpha=.85)
    both=np.r_[b,m];half=np.ptp(both,axis=0).max()*.55;mid=(both.max(0)+both.min(0))/2
    ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[1]-half,mid[1]+half);ax.set_zlim(mid[2]-half,mid[2]+half)
    ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-62)
    ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.set_zlabel('z (mm)')
    displacement=np.linalg.norm(moved-base,axis=1).max()*1000
    ax.set_title(f'Case {index}: grey saved pose / blue coupled +5°\nMax vertex displacement {displacement:.2f} mm')
ax=fig.add_subplot(2,2,3)
for index,a,color in zip((20,28),states,('#2674a8','#b75a23')):
    x=np.arange(109);err=np.max(np.abs(a['candidate_float32_vertices_world_m']-a['source_float32_vertices_world_m']),axis=(1,2))*1e9
    ax.plot(x,err,color=color,label=f'Case {index}: candidate32 vs independent source32')
ax.set(xlabel='Predeclared state index (109 per case)',ylabel='Max coordinate residual (nm)',title='Source skinning parity; all states retained')
ax.legend(fontsize=8);ax.grid(alpha=.2);ax.set_ylim(bottom=0)
ax=fig.add_subplot(2,2,4)
for index,j,color in zip((20,28),derivatives,('#2674a8','#b75a23')):
    error=np.max(np.abs(j['jacobian_m_per_named_tangent']-j['central_difference']),axis=(0,1))
    ax.semilogy(np.arange(51),np.maximum(error,1e-18),color=color,label=f'Case {index}')
ax.axvline(2.5,color='#888888',linestyle=':',linewidth=1)
ax.set(xlabel='Named tangent column (first 3 slides; rest angular)',ylabel='Absolute component residual',title='All 51 derivative columns vs central differences')
ax.legend();ax.grid(alpha=.2)
fig.text(.5,.018,'Saved-array rendering only. Skin function and position increments pass; running collision-surface articulation and native contact forces remain unvalidated.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));outputs=[save(fig,'finger_movement_evidence.png')]

fig,axes=plt.subplots(2,3,figsize=(13,7.5));fig.suptitle('Stock rigid contact: measured checks and a retained sliding failure',fontsize=16)
for dt,hz,color in ((.001,1000,'#2674a8'),(.0005,2000,'#b75a23')):
    folder=ROOT/'runs/manosim_contact_dynamics_2026-10-15_v1/run_v1/raw'
    fall=load(folder/f'freefall_{hz}hz/trajectory.npz');support=load(folder/f'support_flex_{hz}hz/trajectory.npz')
    low=load(folder/f'friction_low_{hz}hz/trajectory.npz');high=load(folder/f'friction_high_{hz}hz/trajectory.npz')
    axes[0,0].plot(fall['endpoint_time_s'],fall['positions_world_m'][:,0,2],color=color,label=f'{hz} Hz')
    axes[0,1].plot(support['pre_time_s'], -support['contact_gap_m']*1e6,color=color)
    axes[0,2].plot(low['endpoint_time_s'],low['velocities_world_m_s'][:,0,0]*1000,color=color)
    axes[1,0].plot(high['pre_time_s'],high['start_qacc'][:,0],color=color,alpha=.65,linewidth=.7,label=f'{hz} Hz')
    axes[1,1].plot(high['pre_time_s'],high['contact_force_contact_N'][:,0]/(.2*9.81),color=color,alpha=.65,linewidth=.7)
    axes[1,2].plot(high['endpoint_time_s'],high['velocities_world_m_s'][:,0,0],color=color,label=f'{hz} Hz')
axes[0,1].axhline(98.1,color='#222222',linestyle='--',linewidth=1,label='Analytic equilibrium 98.1 µm')
axes[0,2].axhline(.613125,color='#222222',linestyle='--',linewidth=1,label='Analytic soft-contact creep')
axes[1,0].axhline(2.4525,color='#222222',linestyle='--',linewidth=1,label='Declared reference 2.4525 m/s²')
axes[1,1].axhline(1,color='#222222',linestyle='--',linewidth=1)
labels=[('Freefall: discrete Euler reference passed','Height (m)'),('Support: analytic penetration passed','Penetration (µm)'),('Low force: regularized creep passed','Lateral speed (mm/s)'),('Strong sliding: retained failure','Lateral acceleration (m/s²)'),('Strong sliding: normal-force oscillation','Contact-normal force / weight'),('Strong sliding: timestep discrepancy','Lateral speed (m/s)')]
for ax,(title,y) in zip(axes.flat,labels):ax.set(title=title,xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2)
for ax in (axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,2]):ax.legend(fontsize=7)
fig.text(.5,.018,'13 / 14 conditions pass. Strong sliding fails the 1 ms reference and timestep comparison. This stock benchmark does not qualify the new MANO contact law.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));outputs.append(save(fig,'physical_contact_evidence_v1.png'))
receipt=dict(scope='Saved arrays only; zero model/skin/FK/contact/physics calls',correction='Use positive contact-normal force rather than signed side1 world-z force; preserve full plotted ranges. Earlier v1 retained.',inputs=INPUTS,outputs={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in outputs},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'render_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'outputs':outputs}))
