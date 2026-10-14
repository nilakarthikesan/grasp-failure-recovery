from pathlib import Path
import json,os
OUT=Path(__file__).parent
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=json.loads((OUT/'hug_reference_comparison.json').read_text())
methods=['stored_current','dexpilot_shipped','dexpilot_metric_static']
labels=['Current vector\nscale 1.0','Released DexPilot style\nscale 1.2, filtered','DexPilot sensitivity\nscale 1.0, no filter']
colors=['#2166ac','#b35806','#1b7837']
fig,axes=plt.subplots(1,3,figsize=(16,5.3),gridspec_kw={'width_ratios':[1,1.2,1.2]})
for i,(m,c) in enumerate(zip(methods,colors)):
 values=np.array([x['methods'][m]['tip_mean_error_m']*1000 for x in r['cases']]);axes[0].scatter(i+np.linspace(-.13,.13,32),values,s=22,alpha=.55,color=c)
 mean=values.mean();axes[0].plot([i-.2,i+.2],[mean,mean],lw=3,color=c);axes[0].text(i,mean+.5,f'{mean:.2f}',ha='center',color=c,fontweight='bold')
axes[0].set_xticks(range(3),labels,fontsize=8.5);axes[0].set_ylabel('Per-pose mean fingertip position error (mm)');axes[0].set_title('32 existing development poses',fontsize=12);axes[0].set_ylim(0,29)
x=np.arange(5);fingers=['thumb','index','middle','ring','little'];width=.25
for offset,(m,c,label) in enumerate(zip(methods,colors,labels)):
 s=r['summary'][m]['by_finger'];mid=[s[f]['middle_mean_error_m']*1000 for f in fingers];tip=[s[f]['tip_mean_error_m']*1000 for f in fingers]
 axes[1].bar(x+(offset-1)*width,mid,width,color=c,label=label.replace('\n',' '));axes[2].bar(x+(offset-1)*width,tip,width,color=c)
axes[1].set_title('Middle-link proxy errors',fontsize=12);axes[2].set_title('Fingertip errors',fontsize=12)
for ax in axes[1:]:ax.set_xticks(x,fingers,fontsize=9);ax.set_ylabel('Mean position error (mm)')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('HUG to Shadow: changing the retargeting objective changes what is preserved',fontsize=15,y=.99)
fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=9)
fig.text(.5,.005,'Same saved native points, source shape, object frames and wrist-to-palm convention. No inference, physics steps or outcome labels.\nReference is a released five-finger adaptation; lower point error does not certify contact preservation. Existing cases are development diagnostics.',ha='center',fontsize=8.5)
fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(OUT/'hug_reference_tradeoff.png',dpi=180);plt.close(fig)
print(OUT/'hug_reference_tradeoff.png')
