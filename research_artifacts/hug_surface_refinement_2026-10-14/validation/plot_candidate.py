"""Static diagnostic figure; independent parts are not contact area."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
B=Path(__file__).parent;r=json.loads((B/'summary.json').read_text());assert len(r['cases'])==6
c=r['cases'];x=np.arange(6);labels=['Apple A','Apple B','Milk A','Milk B','Mug A','Mug B'];colors=['#2b6cb0','#dd6b20']
io0=[];io1=[];lost0=[];lost1=[];added0=[];added1=[];tip0=[];tip1=[];flips0=[];flips1=[];eligible0=[];eligible1=[]
candidate=json.loads(Path('/private/tmp/hug_surface_refinement_2026-10-14/guarded_candidate_targets.json').read_text());cm={v['native_sha256']:v for v in candidate['cases']}
for case in c:
    b=case['resolutions']['2']['bands']['2'];io0.append(b['baseline_unsigned_whole_IoU_centroid']);io1.append(b['candidate_unsigned_whole_IoU_centroid'])
    o=b['baseline_same_part'];n=b['candidate_same_part'];lost0.append(o['sum_independent_part_lost_area_lower_mm2']);lost1.append(n['sum_independent_part_lost_area_lower_mm2']);added0.append(o['sum_independent_part_added_area_lower_mm2']);added1.append(n['sum_independent_part_added_area_lower_mm2']);tip0.append(cm[case['native_sha256']]['engineering_guard']['baseline']['mean_fingertip_error_m']*1000);tip1.append(case['engineering_guard_audit']['mean_fingertip_error_m']*1000);flips0.append(b['baseline_orientation']['sampled_reversal_count']);flips1.append(b['candidate_orientation']['sampled_reversal_count']);eligible0.append(b['baseline_orientation']['outside_visual_eligible']['query_count']);eligible1.append(b['candidate_orientation']['outside_visual_eligible']['query_count'])
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,'xtick.labelsize':10,'figure.dpi':180})
fig,axs=plt.subplots(2,2,figsize=(13.2,9.2));ax=axs.ravel();w=.35
for z,left,right in ((ax[0],io0,io1),(ax[2],tip0,tip1),(ax[3],flips0,flips1)):
    z.bar(x-w/2,left,w,color=colors[0],label='Original mapping');z.bar(x+w/2,right,w,color=colors[1],label='Guarded candidate');z.set_xticks(x,labels,rotation=35,ha='right');z.spines[['top','right']].set_visible(False);z.grid(axis='y',alpha=.18);z.set_axisbelow(True)
ax[0].set_title('Whole-hand unsigned proximity');ax[0].set_ylabel('2 mm centroid IoU');ax[0].set_ylim(0,max(io0+io1)*1.25)
ax[1].bar(x-w/2,lost0,w,color=colors[0]);ax[1].bar(x+w/2,lost1,w,color=colors[1]);ax[1].bar(x-w/2,added0,w,bottom=lost0,color=colors[0],hatch='///',alpha=.55);ax[1].bar(x+w/2,added1,w,bottom=lost1,color=colors[1],hatch='///',alpha=.55);ax[1].set_xticks(x,labels,rotation=35,ha='right');ax[1].spines[['top','right']].set_visible(False);ax[1].grid(axis='y',alpha=.18);ax[1].set_axisbelow(True);ax[1].set_title('Definite lost / added proximity');ax[1].set_ylabel('Sum of six part-area lower bounds (mm²)')
ax[2].set_title('Native fingertip alignment');ax[2].set_ylabel('Mean error over five fingertips (mm)');ax[2].axhline(10,color='#555',ls='--',lw=1);ax[2].text(.03,.91,'Existing 10 mm engineering guard',transform=ax[2].transAxes,fontsize=9);ax[2].set_ylim(0,11.5)
ax[3].set_title('Object-facing direction reversals');ax[3].set_ylabel('Reliable sampled reversals (queries)');ax[3].set_ylim(0,max(flips0+flips1)+2.5)
for i in range(6):
    if flips0[i]==flips1[i]==0:
        ax[3].text(i,.12,f'N: {eligible0[i]}→{eligible1[i]}',ha='center',va='bottom',fontsize=8,color='#555')
    else:
        ax[3].text(i-w/2,flips0[i]+.12,f'N={eligible0[i]}',ha='center',va='bottom',fontsize=8,color=colors[0]);ax[3].text(i+w/2,flips1[i]+.12,f'N={eligible1[i]}',ha='center',va='bottom',fontsize=8,color=colors[1])
ax[3].text(.02,.94,'N = eligible queries; denominator changes',transform=ax[3].transAxes,fontsize=9,color='#444')
handles,legend=ax[0].get_legend_handles_labels();handles.append(Patch(facecolor='#888',hatch='///',alpha=.55));legend.append('Added proximity (hatched)');fig.legend(handles,legend,ncol=3,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
fig.suptitle('HUG surface refinement on six existing development grasps',fontsize=17,y=.99)
status=f"Frozen strict preservation: {r['strict_fail_resolution_count']}/{r['resolution_count']} resolution checks fail. Engineering guard is a separate check."
fig.text(.5,.057,status,ha='center',fontsize=10);fig.text(.5,.027,'Same original cells (maximum triangle area 4 mm²), proximity band 2 mm. Part-area sums may overlap; no physical contact-area or outcome claim.',ha='center',fontsize=9,color='#444');fig.text(.5,.010,'Reversal counts use the actual eligible domain for each mapping. Counts alone cannot establish better direction preservation.',ha='center',fontsize=9,color='#444')
fig.subplots_adjust(top=.88,bottom=.16,wspace=.27,hspace=.58);path=B/'guarded_candidate_tradeoffs.png';fig.savefig(path,dpi=180);plt.close(fig)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();meta={'schema':'hug_guarded_candidate_tradeoff_figure_v1','script_sha256':sha(__file__),'summary_sha256':sha(B/'summary.json'),'candidate_sha256':sha(Path('/private/tmp/hug_surface_refinement_2026-10-14/guarded_candidate_targets.json')),'figure_sha256':sha(path),'case_order':[x['case_id'] for x in c],'density_mm':2,'band_mm':2,'no_physical_contact_area_claim':True,'scope':'Six saved development cases; nine strict-rule resolutions are summarized separately. Original native/geometry/query definitions unchanged.'};(B/'figure_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(path)
