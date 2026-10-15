"""Scientific figure from sealed traces only; no engine or force-law call."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=ROOT/'runs/manosim_saved_contact_comparison_2026-10-15_v1';OUT.mkdir(parents=True,exist_ok=False)
inputs={}
def load(path):
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
stock=load(ROOT/'runs/manosim_contact_dynamics_2026-10-15_v2/run_v1/raw/friction_high_8000hz/trajectory.npz')
new=load(ROOT/'runs/manosim_compliant_contact_dynamics_2026-10-15_v1/run_v1/raw/friction_high_8000hz/trajectory.npz')
sr=ROOT/'research_artifacts/manosim_contact_dynamics_2026-10-15_v2/run_v1/report.json'
nr=ROOT/'research_artifacts/manosim_compliant_contact_dynamics_2026-10-15_v1/run_v1/report.json'
for p in (sr,nr):inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
s,n=json.loads(sr.read_text()),json.loads(nr.read_text())
sd=next(x for x in s['timestep_comparisons'] if x['kind']=='friction_high')['maximum_velocity_difference_m_s']
nd=next(x for x in n['timestep_comparisons'] if x['kind']=='friction_high')['maximum_velocity_difference_m_s']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(14,4.3));fig.suptitle('Strong sliding: different declared contact models, independently checked',fontsize=15)
axs[0].plot(stock['pre_time_s'],stock['contact_force_contact_N'][:,0]/1.962,color='#666666',linewidth=.7,label='Stock elliptic contact')
axs[0].plot(new['start_time_s'],new['law_normal_force_n']/1.962,color='#197ca5',linewidth=1.8,label='Declared compliant law')
axs[0].set(xlim=(.4,.5),xlabel='Time (s), final 100 ms',ylabel='Contact-normal force / weight',title='125 µs traces: intermittent vs steady contact');axs[0].legend(fontsize=8)
axs[1].plot(stock['pre_time_s'],stock['start_qacc'][:,0],color='#666666',linewidth=.7)
axs[1].plot(new['start_time_s'],new['start_qacc'][:,0],color='#197ca5',linewidth=1.8)
axs[1].axhline(2.4525,color='#b3602c',linestyle='--',linewidth=1,label='Analytic sliding reference')
axs[1].set(xlim=(.4,.5),xlabel='Time (s), final 100 ms',ylabel='Lateral acceleration (m/s²)',title='Analytic references retained');axs[1].legend(fontsize=8)
axs[2].bar(['Stock contact','Compliant law'],[sd,nd],color=['#777777','#197ca5'],width=.55)
axs[2].axhline(.0122625,color='#b3602c',linestyle='--',linewidth=1,label='Unchanged comparison limit')
for i,value in enumerate((sd,nd)):axs[2].text(i,value+.00065,f'{value:.6f}',ha='center',fontsize=9)
axs[2].set(ylabel='Maximum final velocity difference (m/s)',title='250 vs 125 µs timestep comparison',ylim=(0,.032));axs[2].legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2,axis='y')
fig.text(.5,.025,'Stock: NOT QUALIFIED. Compliant law: 8/8 conditions and 4/4 timestep comparisons pass. Synthetic assumptions; native-hand contact and real-world calibration remain unvalidated.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.91));p=OUT/'contact_model_evidence.png';fig.savefig(p,dpi=170,bbox_inches='tight');plt.close(fig)
(OUT/'render_receipt.json').write_text(json.dumps(dict(scope='Saved-array rendering; zero engine/law/FK/physics calls',input_hashes=inputs,output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,sort_keys=True)+'\n')
print(p)
