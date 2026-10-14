"""Scientific static figure from frozen geometry-only measurements."""
from pathlib import Path
import os,hashlib,json
BASE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(BASE/'matplotlib_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def sections(tris,q,u,n):
    w=np.cross(u,n);offset=(tris-q)@w;segments=[]
    for tri,dist in zip(tris,offset,strict=True):
        if dist.min()>1e-10 or dist.max()<-1e-10:continue
        points=[]
        for i,j in ((0,1),(1,2),(2,0)):
            if abs(dist[i])<1e-10:points.append(tri[i])
            if dist[i]*dist[j]<0:points.append(tri[i]+dist[i]/(dist[i]-dist[j])*(tri[j]-tri[i]))
        if len(points)<2:continue
        points=np.unique(np.round(points,12),axis=0)
        if len(points)>=2:
            delta=points[:2]-q;segments.append(np.column_stack([delta@u,delta@n])*1000)
    return segments

report=json.loads((BASE/'geometry_cause_report.json').read_text())
fig=plt.figure(figsize=(17,11),layout='constrained');metadata=[]
for row,case in enumerate(report['cases']):
    resolution=case['resolutions']['2'];data=np.load(BASE/(case['object_id']+'_plot_scratch.npz'));name={'hug_apple':'Apple','hug_milk_box':'Milk box','hug_mug':'Mug'}[case['object_id']];witness=resolution['worst_distance_witnesses'][0];q=np.asarray(witness['position_object_m']);n=np.asarray(witness['outward_normal']);axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u)
    ax=fig.add_subplot(3,3,row*3+1,projection='3d');tri=data['visual_tri']*1000;faces=Poly3DCollection(tri,facecolor='#999999',alpha=.08,edgecolor='none',rasterized=True);ax.add_collection3d(faces)
    stride=max(1,int(np.ceil(len(data['q'])/12000)));sel=np.arange(0,len(data['q']),stride);sc=ax.scatter(*(data['q'][sel]*1000).T,c=data['dist'][sel]*1000,cmap='magma',vmin=0,vmax=40,s=2,alpha=.85,rasterized=True);ax.scatter(*(q*1000),color='#17b0c6',s=40,marker='x',linewidth=2);bound=tri.reshape(-1,3);center=(bound.min(0)+bound.max(0))/2;extent=np.max(bound.max(0)-bound.min(0))/2*1.08
    ax.set(xlim=(center[0]-extent,center[0]+extent),ylim=(center[1]-extent,center[1]+extent),zlim=(center[2]-extent,center[2]+extent),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=f'{name}: actual collider boundary samples');ax.set_box_aspect((1,1,1));ax.view_init(23,-55);ax.tick_params(labelsize=8)
    for coordinate_axis in (ax.xaxis,ax.yaxis,ax.zaxis):coordinate_axis.set_major_locator(MaxNLocator(4))
    if row==0:fig.colorbar(sc,ax=ax,shrink=.62,pad=.02,label='Distance to visual mesh (mm)')
    ax=fig.add_subplot(3,3,row*3+2);vs=sections(data['visual_tri'],q,u,n);cs=sections(data['collision_tri'],q,u,n);ax.add_collection(LineCollection(vs,colors='#262626',linewidths=1.6,label='Visual reference'));ax.add_collection(LineCollection(cs,colors='#079cb0',linewidths=.8,alpha=.85,label='Individual convex colliders'));ax.scatter([0],[0],c='#b62e5c',s=36,zorder=4,label='Worst-distance sample');vpoints=np.concatenate(vs);cpoints=np.concatenate(cs);limits=np.concatenate([vpoints,cpoints]);ax.set_xlim(limits[:,0].min()-4,limits[:,0].max()+4);ax.set_ylim(limits[:,1].min()-4,limits[:,1].max()+4);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set(xlabel='Slice transverse coordinate (mm)',ylabel='Slice outward-ray coordinate (mm)',title=f'Slice through {witness["distance_to_visual_mm"]:.2f} mm witness')
    gap=witness['first_other_compiled_hull_ray_entry_m'];gaptext='No next collider on this outward ray' if gap is None else f'Next collider along outward ray: {gap*1e6:.3g} µm';ax.text(.02,.02,gaptext,transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    if row==0:ax.legend(loc='upper right',fontsize=7)
    ax=fig.add_subplot(3,3,row*3+3);probes=resolution['outward_probe_sensitivity'];eps=np.asarray(list(map(float,probes.keys())));total=resolution['query_domain']['retained_union_surface_area_mm2'];allarea=np.asarray([r['retained_area_entering_union_mm2'] for r in probes.values()]);inside=np.asarray([r['inside_visual_area_entering_union_mm2'] for r in probes.values()]);ax.plot(eps*1e6,100*allarea/total,'o-',color='#079cb0',label='All retained collider samples');ax.plot(eps*1e6,100*inside/total,'s--',color='#b62e5c',label='Also inside visual object');ax.set_xscale('log');ax.set(xlabel='Outward probe length (µm)',ylabel='Area entering collider union (%)',title='Sensitivity to tiny local gaps');ax.grid(alpha=.2);ax.set_ylim(0,100)
    if row==0:ax.legend(fontsize=7)
    metadata.append({'object_id':case['object_id'],'displayed_query_stride':stride,'displayed_queries':len(sel),'collision_query_count':len(data['q']),'slice_origin_object_m':q.tolist(),'slice_transverse_unit':u.tolist(),'slice_outward_unit':n.tolist(),'witness':witness})
fig.suptitle('HUG source object geometry: internal collider boundaries and visible reference surfaces',fontsize=16)
fig.supxlabel('Static existing assets; no physics, inference or grasp changes. Slices show individual collider facets, including interior facets.\nLocal ray re-entry and probe sensitivity do not establish global accessibility or physical contact.',fontsize=10)
fig.savefig(BASE/'object_geometry_causes.png',dpi=150)
(BASE/'figure_metadata.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((BASE/'geometry_cause_report.json').read_bytes()).hexdigest(),'plot':'object_geometry_causes.png','scope':'Scientific static geometry investigation; metric data fixed in report. Deterministic subsampling for display only. No PNG pixel analysis used for metrics.','cases':metadata},indent=2)+'\n')
