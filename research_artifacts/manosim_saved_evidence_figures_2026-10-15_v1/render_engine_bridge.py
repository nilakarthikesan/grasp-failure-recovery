"""Saved actual engine surfaces; zero model, skin, FK or geometry queries."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/manosim_engine_bridge_evidence_2026-10-15_v1'
OUT.mkdir(parents=True,exist_ok=False)
fig=plt.figure(figsize=(11,8));fig.suptitle('Actual MANOSim surface updates: geometry passes, strict source normals fail',fontsize=14)
inputs={};saved=[]
for k,index in enumerate((20,28)):
    path=ROOT/f'runs/manosim_skin_engine_bridge_2026-10-15_v2/run_v1/raw/{index}/all109_engine_surface_readbacks.npz'
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
    saved.append(a);vertices=a['compiled_vertices_world_m'];faces=a['faces'];ids=a['pose_ids'].astype(str)
    selected=int(np.flatnonzero(ids=='coupled_+1_5deg')[0]);center=vertices[0].mean(0)
    base=(vertices[0]-center)*1000;moved=(vertices[selected]-center)*1000
    ax=fig.add_subplot(2,2,k+1,projection='3d')
    ax.add_collection3d(Poly3DCollection(base[faces],facecolors='#aaaaaa',edgecolors='#777777',linewidths=.06,alpha=.2))
    ax.add_collection3d(Poly3DCollection(moved[faces],facecolors='#2674a8',edgecolors='#18496b',linewidths=.06,alpha=.6))
    allv=np.r_[base,moved];half=np.ptp(allv,axis=0).max()*.55;mid=(allv.min(0)+allv.max(0))/2
    ax.set_xlim(mid[0]-half,mid[0]+half);ax.set_ylim(mid[1]-half,mid[1]+half);ax.set_zlim(mid[2]-half,mid[2]+half)
    ax.set_box_aspect((1,1,1));ax.view_init(23,-62)
    ax.set(xlabel='x (mm)',ylabel='y (mm)',zlabel='z (mm)',title=f'Case {index}: grey baseline / blue coupled +5°\nTwo saved static forwards; no actuator motion claim')
ax=fig.add_subplot(2,2,3)
for index,a,color in zip((20,28),saved,('#2674a8','#b75a23')):
    ax.semilogy(a['continuous64_vertex_errors_m'],color=color,label=f'Case {index}')
ax.set(title='Engine vs continuous MANO skin: PASS',xlabel='Declared state index (109 per case)',ylabel='Max coordinate residual (m)');ax.legend();ax.grid(alpha=.2)
ax=fig.add_subplot(2,2,4)
for index,a,color in zip((20,28),saved,('#2674a8','#b75a23')):
    ax.semilogy(a['source32_normal_component_errors'],color=color,label=f'Case {index}')
ax.axhline(1e-6,color='#a52828',linestyle='--',label='Unchanged strict normal limit')
ax.set(title='Engine vs original source32 normals: FAIL',xlabel='Declared state index (109 per case)',ylabel='Max normal-component residual');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.text(.5,.018,'Geometry-only condition: contact and actuation disabled. Original baseline normal checks also fail. Full mapping, native contact, collection and training remain unqualified.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.94));p=OUT/'engine_fingers_and_normal_failure.png';fig.savefig(p,dpi=170,bbox_inches='tight');plt.close(fig)
(OUT/'render_receipt.json').write_text(json.dumps(dict(scope='Saved-array rendering only; no native evaluations or engine calls',inputs=inputs,output_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,sort_keys=True)+'\n')
print(p)
