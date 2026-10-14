"""Frozen-report scientific figure, with conditional interval caveats."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent

def main():
    summary=json.loads((OUT/'summary.json').read_text());records=summary['records'];coarse=[x for x in records if x['density_mm']==2];first=[];seen=set()
    for row in records:
        if row['object_id'] not in seen:first.append(row);seen.add(row['object_id'])
    fig,axes=plt.subplots(1,3,figsize=(18,7.5),gridspec_kw={'width_ratios':[1,1.4,1.4]});fig.subplots_adjust(left=.05,right=.98,bottom=.18,top=.83,wspace=.42)
    ax=axes[0]
    for j,row in enumerate(first):
        obj=row['object_id'];fine=next(x for x in records if x['case_id']==row['case_id'] and x['density_mm']==1)
        for item,delta,color in [(row,-.1,'#808a98'),(fine,.1,'#276fbf')]:
            s=item['cell_radius_mm'];ax.vlines(j+delta,s['median'],s['max'],color=color,lw=2);ax.scatter(j+delta,s['median'],marker='o',color=color,s=35);ax.scatter(j+delta,s['p95'],marker='s',color=color,s=30);ax.scatter(j+delta,s['max'],marker='^',color=color,s=35)
    ax.set_xticks(range(3),['Apple','Milk box','Mug']);ax.set_ylabel('Centroid-to-cell radius (mm)');ax.set_title('Area density leaves large cells\nCircle: median · square: p95 · triangle: max',fontsize=11);ax.grid(axis='y',alpha=.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='#808a98',marker='o',label='2 mm area density'),Line2D([0],[0],color='#276fbf',marker='o',label='1 mm area density')],fontsize=9,loc='upper left')
    y=np.arange(len(records));labels=[]
    for row in records:
        obj=row['object_id'].replace('hug_','').replace('milk_box','Milk box').title();scene=row['case_id'].split('_scene_')[1].split('_seed_')[0];labels.append(f'{obj} {scene} · {row["density_mm"]} mm')
    ax=axes[1]
    for j,row in enumerate(records):
        color='#276fbf' if row['density_mm']==1 else '#808a98';w=row['bands']['2']['whole'];i=w['iou_interval'];estimate=w['centroid_estimates']['iou']
        if i['lower'] is not None:ax.hlines(j,i['lower'],i['upper'],color=color,lw=3);ax.plot([i['lower'],i['upper']],[j,j],linestyle='none',marker='|',color=color,markersize=10)
        ax.scatter(estimate,j,color=color,s=35,zorder=3)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(-.02,1.02);ax.set_xlabel('Whole-hand proximity IoU');ax.set_title('2 mm band: centroid estimates and\nconditional cell-based interval',fontsize=11);ax.grid(axis='x',alpha=.2)
    ax=axes[2];lost=[row['bands']['2']['whole']['lost_native_area']['lower_mm2'] for row in records];new=[row['bands']['2']['whole']['new_mapped_area']['lower_mm2'] for row in records];ax.barh(y-.17,lost,height=.32,label='Definitely lost',color='#ad4e51');ax.barh(y+.17,new,height=.32,label='Definitely added',color='#c59324');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Area lower bound (mm²)');ax.set_title('2 mm band: resolved relationship changes\nUncertain cells are excluded from these bars',fontsize=11);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=9)
    fig.suptitle('Existing HUG → Shadow Hand: sampling uncertainty does not remove measured distortion',fontsize=15,y=.96)
    fig.text(.5,.07,'Six saved development poses; three finer repeats. Uniform numerical allowance 10 nm is an explicit assumption, not a formal arithmetic certificate.\nRadii cover complete visual-triangle cells in real arithmetic. Whole-hand distance includes compatible collider surfaces, including carriers and buried interfaces.\nThese are geometric proximity diagnostics, not physical contact, grasp quality or new evaluation episodes.',ha='center',va='center',fontsize=10)
    path=OUT/'sampling_uncertainty.png';fig.savefig(path,dpi=150);plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();metadata={'figure_sha256':sha(path),'script_sha256':sha(__file__),'summary_sha256':sha(OUT/'summary.json'),'source_report_sha256':summary['source_report_sha256'],'band_mm':2,'numerical_allowance_m':1e-8,'conditional_intervals':True,'p95_area_weighted':False,'first_radius_panel':'First source-order group perobject; actualcellradiusmedian/p95/max, query-countpercentiles notstatisticalCI.','new_eval_data':False};(OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'figure':str(path)}))
if __name__=='__main__':main()
