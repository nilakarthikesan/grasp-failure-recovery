"""Static diagnostic figure; no simulation or acceptance-score manufacture."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent

def main():
    source=OUT/'summary.json';report=json.loads(source.read_text());cases=report['cases']
    labels=[('Apple' if c['object_id']=='hug_apple' else 'Milk box' if c['object_id']=='hug_milk_box' else 'Mug')+'\n'+('A' if i%2==0 else 'B') for i,c in enumerate(cases)]
    x=np.arange(len(cases));width=.25
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    palette=['#2563eb','#d97706','#059669'];names=['Original mapper','Raw surface candidate','Guarded candidate']
    for j,(prefix,name,color) in enumerate(zip(['original','raw','guarded'],names,palette,strict=True)):
        axes[0].bar(x+(j-1)*width,[c[f'{prefix}_full_objective'] for c in cases],width,label=name,color=color)
        axes[1].bar(x+(j-1)*width,[c[f'{prefix}_mean_fingertip_error_mm'] for c in cases],width,label=name,color=color)
    axes[0].set_title('Lower geometric objective on fixed query sites',fontsize=12,fontweight='bold')
    axes[0].set_ylabel('Frozen full-pose objective (dimensionless)')
    axes[1].set_title('Fingertip anatomy tradeoff remains',fontsize=12,fontweight='bold')
    axes[1].set_ylabel('Mean native-to-Shadow fingertip error (mm)')
    axes[1].axhline(10,color='#991b1b',lw=1.8,ls='--',label='Existing engineering guard: 10 mm')
    for ax in axes:
        ax.set_xticks(x,labels);ax.set_ylim(0,None);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[1].set_ylim(0,16)
    fig.suptitle('HUG mapping refinement on six existing development grasps',fontsize=15,fontweight='bold')
    axes[0].legend(loc='upper left',fontsize=9,frameon=False)
    axes[1].legend(loc='upper left',fontsize=9,frameon=False)
    fig.text(.5,-.025,'Static FK only. A lower objective or engineering-guard pass is not scientific mapping acceptance.',ha='center',fontsize=10)
    dest=OUT/'refinement_tradeoff.png';fig.savefig(dest,dpi=180,bbox_inches='tight');plt.close(fig)
    metadata={'figure':dest.name,'source_summary_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'plotted_fields':['original_full_objective','raw_full_objective','guarded_full_objective',
            'original_mean_fingertip_error_mm','raw_mean_fingertip_error_mm','guarded_mean_fingertip_error_mm'],
        'existing_engineering_guard_mm':10,'not_a_scientific_acceptance_margin':True,
        'no_statistical_error_bars':'Six development inputs, not independent replicas; deterministic geometric diagnostics.',
        'new_eval_data':False}
    (OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()
