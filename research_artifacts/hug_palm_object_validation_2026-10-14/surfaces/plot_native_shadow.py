"""Optional static scientific figure and read-only collider/visual diagnostic.

Does not alter the frozen main diagnostic or historical data. Uses the first
inventory-order seed-0 group per object, fixed before any optional metrics.
"""
from pathlib import Path
import hashlib,json,pickle,time,warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import audit_other_surfaces as a
OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
COLORS=['#858d96','#2186bd','#38a06b','#9568b1','#db9828','#d54c45','#aaaaaa','#aaaaaa']
prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
inventory=json.loads((prior/'existing_hug_inventory.json').read_text());fk=json.loads((prior/'static_saved_hug_fk.json').read_text());fkmap={c['native_sha256']:c for c in fk['cases']}
protocol=json.loads((OUT/'protocol_v3.json').read_text());cases=[];seen=set()
for r in protocol['cases']:
 c=inventory['modern_cases'][r['source_inventory_index']]
 if c['object_id'] not in seen:cases.append(c);seen.add(c['object_id'])
assert a.sha(a.ASSET)==json.loads((ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/surface/protocol.json').read_text())['mano_asset_sha256']
with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
weights=np.asarray(asset['weights']);gw=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
fig=plt.figure(figsize=(13,14),layout='constrained');metrics=[]
for row,c in enumerate(cases):
 meta,tri,labels,norm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(c,gw,fkmap)
 visible=[g for g in hand if g['part'] not in ['forearm_carrier','wrist_carrier']]
 target=[]
 for g in visible:
  if g['tri_object'] is not None:t=g['tri_object']
  else:
   size=g['size'];theta=np.arange(32)*2*np.pi/32
   if g['kind']==a.mujoco.mjtGeom.mjGEOM_CYLINDER:verts=np.array([[size[0]*np.cos(t),size[0]*np.sin(t),z] for z in [-size[1],size[1]] for t in theta])
   else:
    phi=np.arange(1,16)*np.pi/16;verts=np.array([[size[0]*np.sin(p)*np.cos(t),size[0]*np.sin(p)*np.sin(t),size[0]*np.cos(p)] for p in phi for t in theta]+[[0,0,size[0]],[0,0,-size[0]]])
   lt,_,_=a.hull(verts);t=lt@g['R'].T+g['pos']
  target.append((t,COLORS[a.PARTS.index(g['part'])]))
 bounds=np.concatenate([tri.reshape(-1,3),*[t.reshape(-1,3) for t,_ in target],*[o['tri_object'].reshape(-1,3) for o in objects]])*1000
 lo=bounds.min(0);hi=bounds.max(0);center=(lo+hi)/2;span=max(hi-lo)*1.08;limits=np.column_stack([center-span/2,center+span/2])
 title={'hug_apple':'Apple','hug_milk_box':'Milk box','hug_mug':'Mug'}[c['object_id']]
 for col in [0,1]:
  ax=fig.add_subplot(3,2,row*2+col+1,projection='3d');ax.set_proj_type('ortho');ax.view_init(elev=25,azim=135)
  ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_zlim(*limits[2]);ax.set_box_aspect((1,1,1));ax.tick_params(labelsize=7,pad=0)
  ax.set_xlabel('Object X (mm)',fontsize=8,labelpad=0);ax.set_ylabel('Object Y (mm)',fontsize=8,labelpad=0);ax.set_zlabel('Object Z (mm)',fontsize=8,labelpad=0)
  for o in objects:ax.add_collection3d(Poly3DCollection(o['tri_object']*1000,facecolors='#89939d',alpha=.12,edgecolors='none'))
  if col==0:ax.add_collection3d(Poly3DCollection(tri*1000,facecolors=np.asarray(COLORS)[labels],alpha=.65,edgecolors='none'))
  else:
   for t,color in target:ax.add_collection3d(Poly3DCollection(t*1000,facecolors=color,alpha=.65,edgecolors='none'))
  ax.set_title(f'{title}: '+('saved native MANO' if col==0 else 'original nominal Shadow'),fontsize=11,pad=0)
legend=[Patch(facecolor=COLORS[j],label=a.PARTS[j].capitalize()) for j in range(6)]+[Patch(facecolor='#89939d',alpha=.3,label='Active object collision hulls')]
fig.legend(handles=legend,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
fig.suptitle('Existing HUG grasps: native and original nominal target\nNative mesh and original translated target around the same captured object collision geometry',fontsize=14)
fig.savefig(OUT/'hug_other_objects_native_shadow.png',dpi=190);plt.close(fig)
a.write(OUT/'figure_metadata.json',{'script_sha256':a.sha(__file__),'figure_sha256':a.sha(OUT/'hug_other_objects_native_shadow.png'),'collision_protocol_sha256':a.sha(OUT/'protocol_v3.json'),'native_sha256':[c['native']['sha256'] for c in cases],'selection':'First inventory-order observation group, seed 0, each object. Fixed before metrics.','camera':'Orthographic, elevation25 azimuth135; same object-frame limits and metric scale per native/target row.','visible_geometry':'Actual native saved MANO triangles and per-geom target collision shapes. Gray actual active object hulls; rendering shows individual convex hulls, including overlap, not an exact clipped union mesh. Forearm/wrist carrier hidden for clarity; included in metrics. Curved target primitive tessellation is display only.','physics_steps':0})
