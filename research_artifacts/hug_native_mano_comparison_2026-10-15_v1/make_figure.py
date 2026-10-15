from pathlib import Path
import sys,json
import numpy as np
sys.path.append('/private/tmp/grasp-partner-simulation-deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;r=json.loads((B/'measurement/results.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
colors=['#a95d35','#187d8c'];w=.34
rows=r['object_summary'];x=np.arange(4)
for i,(key,label) in enumerate([('shadow_tip_mean_m','Shadow'),('mano_tip_mean_m','Native MANO proxy')]):
 axes[0].bar(x+(i-.5)*w,[1000*v[key] for v in rows],w,label=label,color=colors[i])
axes[0].set_xticks(x,['Cube','Apple','Milk box','Mug']);axes[0].set_ylabel('Mean fingertip discrepancy (mm)');axes[0].set_title('Saved HUG fingertips vs rigid hand tip sites')
rows=r['surface_comparison'];x=np.arange(6)
for i,(key,label) in enumerate([('native_triangle_centroid_boundary_error_shadow_mean_m','Shadow'),('native_triangle_centroid_boundary_error_mano_mean_m','Native MANO proxy')]):
 axes[1].bar(x+(i-.5)*w,[1000*v[key] for v in rows],w,label=label,color=colors[i])
axes[1].set_xticks(x,[v['part'].title() for v in rows]);axes[1].set_ylabel('Mean sampled surface discrepancy (mm)');axes[1].set_title('Native skin samples to same-part proxy surfaces')
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Static mapping comparison • 32 existing HUG grasps • no physics rollouts',fontsize=14)
fig.savefig(B/'comparison.png',dpi=170)
print(B/'comparison.png')
