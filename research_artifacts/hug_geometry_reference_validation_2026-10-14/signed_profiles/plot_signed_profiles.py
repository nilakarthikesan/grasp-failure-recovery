"""Scientific comparison of independent sampled signed-distance profiles."""
from pathlib import Path
import hashlib,json,os
base=Path(__file__).resolve().parent;os.environ.setdefault('MPLCONFIGDIR',str(base/'matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
v=json.loads((base/'signed_hand_visual_profiles.json').read_text());methods=['native','stored_current','dexpilot_shipped','dexpilot_metric_static'];names=['Native MANO','Current mapping','Shipped reference','Metric-static reference'];colors=['#222222','#1888a6','#ca6a2e','#7863a8'];labels=[]
for c in v['cases']:
    name={'hug_apple':'Apple','hug_milk_box':'Milk box','hug_mug':'Mug'}[c['object_id']];group=1 if c['source_inventory_index'] in (8,16,24) else 2;labels.append(f'{name}, group {group}')
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained');y=np.arange(len(labels));metadata=[]
for j,(method,name,color) in enumerate(zip(methods,names,colors,strict=True)):
    percentage=[];maxima=[];finer=[]
    for c in v['cases']:
        r=c['resolutions']['4'];profile=r['native_area_profile']['whole'] if method=='native' else r['methods'][method]['primary_palm_fingers_profile']['whole'];percentage.append(profile['negative_weight_fraction']*100);maxima.append(profile['finite_sampled_max_penetration_mm'])
        fine=c['resolutions'].get('2');fp=None if fine is None else (fine['native_area_profile']['whole'] if method=='native' else fine['methods'][method]['primary_palm_fingers_profile']['whole']);finer.append(fp)
    offset=(j-1.5)*.17
    for ax,values in zip(axes,[percentage,maxima],strict=True):ax.plot(values,y+offset,'o',color=color,label=name,markersize=6)
    for i,fine in enumerate(finer):
        if fine is None:continue
        fv=[fine['negative_weight_fraction']*100,fine['finite_sampled_max_penetration_mm']]
        for ax,value,old in zip(axes,fv,[percentage[i],maxima[i]],strict=True):ax.plot([old,value],[y[i]+offset,y[i]+offset],color=color,alpha=.7);ax.plot(value,y[i]+offset,'o',mfc='white',mec=color,markersize=8,zorder=3)
    metadata.append({'method':method,'negative_area_percent_density4':percentage,'finite_sampled_max_depth_mm_density4':maxima,'fine_density2_available':[f is not None for f in finer]})
for ax in axes:ax.set_yticks(y,labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.25);ax.set_xlim(left=0)
axes[0].set(xlabel='Sampled hand area inside visible solid (%)',title='Independent signed surface profiles');axes[1].set(xlabel='Finite sampled maximum negative depth (mm)',title='Finite samples can miss deeper penetration');axes[1].legend(loc='lower right',fontsize=9)
fig.suptitle('Existing HUG poses: native penetration and retargeting changes',fontsize=16)
fig.supxlabel('Filled markers: 4 mm triangle-area density. Hollow: predeclared 2 mm repeats of first group per object.\nTargets use palm/fingers only; carriers reported separately. Curved primitive area weights approximate.\nIndependent shape profiles, not matched points or contact forces; no success requirement or new evaluation data.',fontsize=10)
fig.savefig(base/'signed_profiles_comparison.png',dpi=150)
(base/'figure_metadata.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((base/'signed_hand_visual_profiles.json').read_bytes()).hexdigest(),'source_case_labels':labels,'profiles':metadata,'interpretation':'Signed geometry inside closed visual reference; finite sampling and distinct hand shapes. No contact-force/outcome/acceptance claim.'},indent=2)+'\n')
