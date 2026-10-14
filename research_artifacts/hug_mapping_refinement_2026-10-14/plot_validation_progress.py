from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE=Path('/private/tmp/grasp-hug-refinement-2026-10-14/hug_surface_refinement.json')
SAVED=Path('/private/tmp/grasp-hug-existing-inventory-2026-10-14/static_saved_hug_fk.json')
OUT=Path('/private/tmp/hug_mapping_validation_progress_2026-10-14.png')
surface=json.loads(SURFACE.read_text()); saved=json.loads(SAVED.read_text())
fig,axes=plt.subplots(1,2,figsize=(12.5,5.1),layout='constrained')
objects=[('object01','Cube'),('hug_apple','Apple'),('hug_milk_box','Milk box'),('hug_mug','Mug')]
colors=['#20649c','#42855b','#98691e','#81549d']
for i,((key,name),color) in enumerate(zip(objects,colors)):
    rows=[r for r in saved['cases'] if r['object_id']==key]
    xs=np.arange(len(rows))+i*9
    means=np.array([r['mean_tip_world_error_m'] for r in rows])*1000
    maxima=np.array([r['maximum_tip_world_error_m'] for r in rows])*1000
    axes[0].vlines(xs,means,maxima,color=color,alpha=.4,lw=1)
    axes[0].scatter(xs,means,color=color,s=30,zorder=3)
    axes[0].scatter(xs,maxima,facecolors='none',edgecolors=color,s=35,marker='D',zorder=3)
axes[0].set_xticks([3.5,12.5,21.5,30.5],[label for _,label in objects])
axes[0].set_ylabel('Native-to-Shadow fingertip separation (mm)')
axes[0].set_xlabel('Existing proposals grouped by object (8 per object)')
axes[0].set_title('Nominal point correspondence across 32 saved poses')
axes[0].set_ylim(0,20);axes[0].grid(axis='y',alpha=.2)
axes[0].legend(handles=[Line2D([],[],marker='o',color='#4a4a4a',ls='none',label='Mean across five fingertips'),Line2D([],[],marker='D',markerfacecolor='none',color='#4a4a4a',ls='none',label='Largest individual fingertip error')],frameon=False,fontsize=9)
for band,color in zip(['1','2','5'],['#596b7a','#20649c','#98691e']):
    xs=np.array([2.,1.,.5])
    ys=[surface['grids'][str(x) if x==.5 else str(int(x))]['bands'][band]['occupied_or_near']['jaccard'] for x in xs]
    axes[1].plot(xs,ys,marker='o',color=color,label=f'{band} mm diagnostic band')
axes[1].set_xlim(2.12,.38);axes[1].set_xticks([2,1,.5]);axes[1].set_ylim(0,1)
axes[1].set_xlabel('Cube-face grid spacing (mm; finer to the right)')
axes[1].set_ylabel('Occupied-or-near footprint intersection / union')
axes[1].set_title('Cube geometry differences persist on finer grids')
axes[1].grid(alpha=.2);axes[1].legend(frameon=False,fontsize=9)
fig.suptitle('HUG mapping validation: static evidence before controller closure',fontsize=14)
fig.text(.5,-.035,'Source: 32 existing Oct 11 HUG poses and Oct 14 saved-cube geometry audit. No new inference or physics.\nPoint residuals and occupied-or-near overlap are diagnostics, not physical contact fidelity or predictor accuracy.',ha='center',fontsize=9)
fig.savefig(OUT,dpi=180,bbox_inches='tight');plt.close(fig)
print(str(OUT))
