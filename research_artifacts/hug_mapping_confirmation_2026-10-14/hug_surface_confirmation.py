"""HUG-only static surface confirmation. Reads saved native/nominal artifacts."""
from pathlib import Path
import hashlib,json,pickle,warnings,sys
import scipy
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import chumpy

def forbidden(*args,**kwargs):
    raise RuntimeError('This static confirmation forbids MuJoCo integration')
for api in ['mj_step','mj_step1','mj_step2']:setattr(mujoco,api,forbidden)

OUT=Path('/private/tmp/grasp-hug-confirmation-2026-10-14')
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PFILE=ROOT/'runs/hug_grasp_eval_2026-10-13_fresh_cube/output/proposals/hug_cube/sample_0000/proposal.json'
NATIVE=PFILE.parent/'native'
ASSET=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl')
proposal=json.loads(PFILE.read_text());source=pickle.loads((NATIVE/'proposal.pkl').read_bytes())
state=np.load(NATIVE/'scene_state.npz')
model=mujoco.MjModel.from_binary_path(str(NATIVE/'scene.mjb'));data=mujoco.MjData(model)
data.qpos[:]=state['qpos'];data.qvel[:]=0
joint_id=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
body_id=lambda name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
root_adr=int(model.jnt_qposadr[joint_id('world_joint')]);object_adr=int(model.jnt_qposadr[joint_id('object_free')])
object_qpos=data.qpos[object_adr:object_adr+7].copy()
for name,value in proposal['joint_targets_rad'].items():data.qpos[int(model.jnt_qposadr[joint_id(name)])]=value
data.qpos[root_adr:root_adr+3]=0;data.qpos[root_adr+3:root_adr+7]=[1,0,0,0]
mujoco.mj_forward(model,data)
palm_id=body_id('palm');obj_id=body_id('object');forearm_id=body_id('forearm')
R_obj=data.xmat[obj_id].reshape(3,3).copy();t_obj=data.xpos[obj_id].copy()
R_palm=R_obj@Rotation.from_quat(proposal['object_to_palm']['quaternion_xyzw']).as_matrix()
t_palm=t_obj+R_obj@np.array(proposal['object_to_palm']['translation_m'])
R_root=R_palm@data.xmat[palm_id].reshape(3,3).T
t_root=t_palm-R_root@data.xpos[palm_id]
quat=Rotation.from_matrix(R_root).as_quat()[[3,0,1,2]]
data.qpos[root_adr:root_adr+3]=t_root;data.qpos[root_adr+3:root_adr+7]=quat
data.mocap_pos[0]=t_root;data.mocap_quat[0]=quat
mujoco.mj_forward(model,data)
assert np.array_equal(object_qpos,data.qpos[object_adr:object_adr+7])
to_obj=lambda points:(np.asarray(points)-t_obj)@R_obj
T_wc=np.load(NATIVE/'T_world_camera.npy')
native_vertices=to_obj(np.asarray(source['mesh_vertices'])@T_wc[:3,:3].T+T_wc[:3,3])
native_faces=np.asarray(source['mesh_faces'],dtype=int)
native_triangles=native_vertices[native_faces]
with warnings.catch_warnings():
    warnings.simplefilter('ignore');asset=pickle.loads(ASSET.read_bytes(),encoding='latin1')
weights=np.asarray(asset['weights'])
GROUPS={'palm':[0],'index':[1,2,3],'middle':[4,5,6],'ring':[10,11,12],'little':[7,8,9],'thumb':[13,14,15]}
PARTS=list(GROUPS)
group_weights=np.column_stack([weights[:,ids].sum(axis=1) for ids in GROUPS.values()])
native_face_parts=np.argmax(group_weights[native_faces].mean(axis=1),axis=1)
COLORS=['#939aa4','#2084c5','#30a46c','#9569bc','#dc9110','#dc493a']

def triangle_data(tri):
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(n,axis=1)
    return n/norm[:,None],norm/2

native_normals,native_areas=triangle_data(native_triangles)
volume=float(np.einsum('ij,ij->i',native_triangles[:,0],np.cross(native_triangles[:,1],native_triangles[:,2])).sum()/6)
if volume<0:native_normals*=-1

def nearest_triangles(points,triangles,batch=32):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=np.einsum('ij,ij->i',n,n)
    d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
    den=d00*d11-d01*d01
    result=[];indices=[];distances=[]
    for start in range(0,len(points),batch):
        p=points[start:start+batch];ap=p[:,None]-a
        projection=p[:,None]-n[None]*(np.einsum('nmi,mi->nm',ap,n)/nn)[...,None]
        projected_ap=projection-a
        d20=np.einsum('nmi,mi->nm',projected_ap,ab);d21=np.einsum('nmi,mi->nm',projected_ap,ac)
        s=(d11*d20-d01*d21)/den;t=(d00*d21-d01*d20)/den
        within=(s>=0)&(t>=0)&(s+t<=1)
        best=projection.copy();best_d=np.where(within,np.sum((p[:,None]-projection)**2,axis=2),np.inf)
        for left,right in [(a,b),(b,c),(c,a)]:
            edge=right-left
            coefficient=np.clip(np.einsum('nmi,mi->nm',p[:,None]-left,edge)/np.einsum('mi,mi->m',edge,edge),0,1)
            q=left+coefficient[...,None]*edge
            sd=np.sum((p[:,None]-q)**2,axis=2);better=sd<best_d
            best_d[better]=sd[better];best[better]=q[better]
        ii=np.argmin(best_d,axis=1);rr=np.arange(len(p))
        result.append(best[rr,ii]);indices.append(ii);distances.append(np.sqrt(best_d[rr,ii]))
    return np.concatenate(result),np.concatenate(indices),np.concatenate(distances)

def inside_mesh(points,triangles,batch=32):
    result=[]
    for start in range(0,len(points),batch):
        p=points[start:start+batch]
        a=triangles[None,:,0]-p[:,None];b=triangles[None,:,1]-p[:,None];c=triangles[None,:,2]-p[:,None]
        na=np.linalg.norm(a,axis=2);nb=np.linalg.norm(b,axis=2);nc=np.linalg.norm(c,axis=2)
        det=np.einsum('nmi,nmi->nm',a,np.cross(b,c))
        den=na*nb*nc+np.einsum('nmi,nmi->nm',a,b)*nc+np.einsum('nmi,nmi->nm',b,c)*na+np.einsum('nmi,nmi->nm',c,a)*nb
        winding=np.sum(2*np.arctan2(det,den),axis=1)/(4*np.pi)
        result.append(np.abs(winding)>.5)
    return np.concatenate(result)

def finger_part(body):
    if body.startswith('ff'):return 'index'
    if body.startswith('mf'):return 'middle'
    if body.startswith('rf'):return 'ring'
    if body.startswith('lf'):return 'little'
    if body.startswith('th'):return 'thumb'
    return 'palm'

def descendants(bid):
    while bid>0:
        if bid==forearm_id:return True
        bid=int(model.body_parentid[bid])
    return False

def hull_triangles(vertices):
    hull=ConvexHull(vertices);tri=vertices[hull.simplices].copy()
    computed=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);flip=np.einsum('ij,ij->i',computed,hull.equations[:,:3])<0
    tri[flip]=tri[flip][:,[0,2,1]]
    return tri,hull.equations

geometries=[]
for geom in range(model.ngeom):
    body=int(model.geom_bodyid[geom])
    if not descendants(body) or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
    kind=int(model.geom_type[geom]);size=model.geom_size[geom].copy()
    name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body)
    R=data.geom_xmat[geom].reshape(3,3).copy();pos=data.geom_xpos[geom].copy()
    eq=None
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh=int(model.geom_dataid[geom]);adr=int(model.mesh_vertadr[mesh]);nv=int(model.mesh_vertnum[mesh])
        vertices=model.mesh_vert[adr:adr+nv].copy();tri,eq=hull_triangles(vertices)
        representation='exact per-geom convex hull of compiled mesh vertices'
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        vertices=np.array([[x,y,z] for x in [-size[0],size[0]] for y in [-size[1],size[1]] for z in [-size[2],size[2]]])
        tri,eq=hull_triangles(vertices);representation='analytic box; exact polygon render'
    elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        theta=np.arange(32)*2*np.pi/32;phi=np.arange(1,16)*np.pi/16
        vertices=np.array([[size[0]*np.sin(p)*np.cos(t),size[0]*np.sin(p)*np.sin(t),size[0]*np.cos(p)] for p in phi for t in theta]+[[0,0,size[0]],[0,0,-size[0]]])
        tri,_=hull_triangles(vertices);representation='analytic sphere; 32x16 render tessellation'
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        theta=np.arange(32)*2*np.pi/32
        vertices=np.array([[size[0]*np.cos(t),size[0]*np.sin(t),z] for z in [-size[1],size[1]] for t in theta])
        tri,_=hull_triangles(vertices);representation='analytic cylinder; 32-segment render tessellation'
    else:raise ValueError(f'Unhandled true collider type {kind}')
    triangle_obj=to_obj((tri.reshape(-1,3)@R.T+pos)).reshape(-1,3,3)
    geometries.append({'geom':geom,'body':name,'part':finger_part(name),'kind':kind,'size':size,'R':R,'pos':pos,'tri_local':tri,'tri_object':triangle_obj,'equations':eq,'representation':representation})

half=np.array([.025,.025,.025]);spacing=.002
uv=(np.arange(25)+.5)*spacing-.025
FACES=[('+X',0,1),('-X',0,-1),('+Y',1,1),('-Y',1,-1),('+Z',2,1),('-Z',2,-1)]
grids=[];face_normals=[]
for label,axis,sign in FACES:
    other=[i for i in range(3) if i!=axis];grid=np.zeros((625,3));grid[:,axis]=sign*.025
    u,vv=np.meshgrid(uv,uv,indexing='xy');grid[:,other[0]]=u.ravel();grid[:,other[1]]=vv.ravel();grids.append(grid)
    n=np.zeros(3);n[axis]=sign;face_normals.append(n)
query=np.concatenate(grids)
native_closest,native_index,native_distance=nearest_triangles(query,native_triangles)
native_inside=inside_mesh(query,native_triangles)
native_signed=np.where(native_inside,-native_distance,native_distance)
native_normal=native_normals[native_index]
tri=native_triangles[native_index];ab=tri[:,1]-tri[:,0];ac=tri[:,2]-tri[:,0];ap=native_closest-tri[:,0]
d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac)
d20=np.einsum('ij,ij->i',ap,ab);d21=np.einsum('ij,ij->i',ap,ac);den=d00*d11-d01*d01
b1=(d11*d20-d01*d21)/den;b2=(d00*d21-d01*d20)/den
bary=np.column_stack([1-b1-b2,b1,b2]);gw=np.einsum('ni,nij->nj',bary,group_weights[native_faces[native_index]])
native_part=np.argmax(gw,axis=1)
shadow_signed=np.full(len(query),np.inf);shadow_normal=np.zeros_like(query);shadow_part=np.full(len(query),-1,dtype=int)

def geom_query(g):
    world=query@R_obj.T+t_obj;p=(world-g['pos'])@g['R'];kind=g['kind'];size=g['size']
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        length=np.linalg.norm(p,axis=1);sd=length-size[0];normal=p/np.maximum(length[:,None],1e-15)
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        delta=np.abs(p)-size;outside=np.maximum(delta,0);od=np.linalg.norm(outside,axis=1)
        sd=od+np.minimum(delta.max(axis=1),0);normal=outside*np.sign(p)/np.maximum(od[:,None],1e-15)
        inside=od==0;axis=np.argmax(delta[inside],axis=1);normal[inside]=0
        row=np.flatnonzero(inside);normal[row,axis]=np.where(p[row,axis]>=0,1.,-1.)
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(p[:,:2],axis=1);rdir=p[:,:2]/np.maximum(radial[:,None],1e-15)
        delta=np.column_stack([radial-size[0],np.abs(p[:,2])-size[1]])
        outside=np.maximum(delta,0);od=np.linalg.norm(outside,axis=1);sd=od+np.minimum(delta.max(axis=1),0)
        normal=np.column_stack([rdir*outside[:,0:1],np.sign(p[:,2])*outside[:,1]])/np.maximum(od[:,None],1e-15)
        inside=od==0;side=delta[:,0]>=delta[:,1];normal[inside]=0
        normal[inside&side,:2]=rdir[inside&side];normal[inside&~side,2]=np.where(p[inside&~side,2]>=0,1.,-1.)
    else:
        q,index,distance=nearest_triangles(p,g['tri_local']);eq=g['equations'];in_geom=np.max(p@eq[:,:3].T+eq[:,3],axis=1)<=1e-10
        sd=np.where(in_geom,-distance,distance);normal=triangle_data(g['tri_local'])[0][index]
    return sd,normal@g['R'].T@R_obj

for g in geometries:
    # Far mesh hulls cannot affect <=5mm cube-face coverage; avoid heavy queries.
    lo=g['tri_object'].reshape(-1,3).min(axis=0);hi=g['tri_object'].reshape(-1,3).max(axis=0)
    separation=np.maximum(np.maximum(lo-half,-half-hi),0)
    if g['kind']==mujoco.mjtGeom.mjGEOM_MESH and np.linalg.norm(separation)>.02:continue
    sd,normal=geom_query(g);update=sd<shadow_signed
    shadow_signed[update]=sd[update];shadow_normal[update]=normal[update];shadow_part[update]=PARTS.index(g['part'])

contacts=[]
for i in range(data.ncon):
    c=data.contact[i];b1=int(model.geom_bodyid[c.geom1]);b2=int(model.geom_bodyid[c.geom2])
    if obj_id not in [b1,b2] or not any(descendants(b) for b in [b1,b2]):continue
    normal=c.frame[:3].copy()*(1 if b1==obj_id else -1)
    contacts.append({'bodies':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in [b1,b2]],'object_position_m':to_obj(c.pos).tolist(),'cube_to_hand_normal_object':(normal@R_obj).tolist(),'distance_m':float(c.dist)})

face_results={};cell_area_mm2=spacing**2*1e6
for i,(name,axis,sign) in enumerate(FACES):
    sel=slice(i*625,(i+1)*625);n=native_signed[sel];s=shadow_signed[sel]
    result={'native_inside_grid_area_mm2':float((n<0).sum()*cell_area_mm2),'shadow_inside_any_collider_grid_area_mm2':float((s<0).sum()*cell_area_mm2)}
    for tol in [.002,.005]:
        nk=n<=tol;sk=s<=tol;inter=nk&sk;union=nk|sk;tag=f'{int(tol*1000)}mm'
        result[tag]={'native_occupied_or_near_area_mm2':float(nk.sum()*cell_area_mm2),'shadow_occupied_or_near_area_mm2':float(sk.sum()*cell_area_mm2),'intersection_area_mm2':float(inter.sum()*cell_area_mm2),'union_area_mm2':float(union.sum()*cell_area_mm2),'native_only_area_mm2':float((nk&~sk).sum()*cell_area_mm2),'shadow_only_area_mm2':float((sk&~nk).sum()*cell_area_mm2),'jaccard':float(inter.sum()/union.sum()) if union.any() else None,'native_skinning_parts':{part:float(((native_part[sel]==j)&nk).sum()*cell_area_mm2) for j,part in enumerate(PARTS)},'shadow_collider_parts':{part:float(((shadow_part[sel]==j)&sk).sum()*cell_area_mm2) for j,part in enumerate(PARTS)}}
    normal_cube=face_normals[i];nvalid=(n>1e-8)&(n<=.005);svalid=(s>1e-8)&(s<=.005);both=nvalid&svalid
    result['geometric_normals']={'native_separated_near_5mm_count':int(nvalid.sum()),'shadow_separated_near_5mm_count':int(svalid.sum()),'native_mean_dot_cube_outward':float(np.mean(native_normal[sel][nvalid]@normal_cube)) if nvalid.any() else None,'shadow_mean_dot_cube_outward':float(np.mean(shadow_normal[sel][svalid]@normal_cube)) if svalid.any() else None,'paired_separated_near_5mm_count':int(both.sum()),'paired_normal_angle_median_deg':float(np.median(np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',native_normal[sel][both],shadow_normal[sel][both]),-1,1))))) if both.any() else None}
    face_results[name]=result
global_results={}
for tol in [.002,.005]:
    n=native_signed<=tol;s=shadow_signed<=tol;inter=n&s;union=n|s
    global_results[f'{int(tol*1000)}mm']={'native_occupied_or_near_area_mm2':float(n.sum()*cell_area_mm2),'shadow_occupied_or_near_area_mm2':float(s.sum()*cell_area_mm2),'intersection_area_mm2':float(inter.sum()*cell_area_mm2),'union_area_mm2':float(union.sum()*cell_area_mm2),'symmetric_difference_area_mm2':float((n^s).sum()*cell_area_mm2),'jaccard':float(inter.sum()/union.sum()) if union.any() else None}

report={
 'scope':'HUG only: saved native MANO mesh versus nominal translated Shadow collision shapes at identical INITIAL cube pose; no force-close/lift/stepping/inference',
 'native_sha256':hashlib.sha256((NATIVE/'proposal.pkl').read_bytes()).hexdigest(),
 'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'runtime_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__,'matplotlib':matplotlib.__version__,'chumpy':chumpy.__version__},
 'licensed_asset_path_only':str(ASSET),'raw_licensed_asset_copied':False,
 'guarded_integration_apis':['mj_step','mj_step1','mj_step2'],'mujoco_time_s':float(data.time),
 'object_qpos_max_change':float(np.max(np.abs(data.qpos[object_adr:object_adr+7]-object_qpos))),
 'native_surface':'Exact saved778vertices and1552triangles, including wrist cap far from cube. Face-normal directions follow saved winding; positive volume; no whole-hand hull.',
 'native_signed_volume_m3':volume,
 'native_skinning_labels':'Argmax of barycentrically interpolated weights summed over anatomical bone groups; approximate anatomical analysis labels, not contact ground truth.',
 'native_groups':GROUPS,
 'shadow_surface':'Union membership/positive outside proximity to actual saved compiled collision geoms. Per-mesh-geom convex hull exactly follows MuJoCo3.3.7 collision geometry. No whole-hand convex hull. Sphere/cylinder distances/normals analytic; curved render is tessellated.',
 'shadow_geom_count':len(geometries),
 'shadow_geoms':[{'geom_id':g['geom'],'body':g['body'],'part':g['part'],'representation':g['representation']} for g in geometries],
 'surface_queries':'3750 deterministic cube-face cell centers at2mm spacing;625perface;4mm²percell; no stochastic samples or RNG seed. Native nearest triangle exact plus winding occupancy. Robot per-geom analytical/convex distance and any-collider occupancy.',
 'coverage_definition':'Occupied-or-near footprint: cube-face query is inside rendered native hand / any Shadow collider OR is separated by at most2/5mm. Negative Shadow minimum-per-geom distance is only occupancy/penetration proxy, not exact union interior depth. These tolerances are diagnostics, not acceptance gates.',
 'normal_definition':'Native closest saved triangle normal; Shadow analytic primitive or individual hull-facet normal. Only externally separated positive<=5mm points used for normal comparison. These geometric normals are not paired native physical contact normals.',
 'camera_and_scale':{'projection':'orthographic','elev_deg':25,'azim_deg':135,'object_axis_limits_mm':[[-65,135],[-95,65],[-30,165]],'crop':'Forearm/wrist carrier excluded from visible mesh panels; palm and fingers visible. All45actualhandcollisiongeoms considered for metrics except mesh bounds>20mm fromcube, which cannot affect<=5mm coverage.'},
 'cube_face_results':face_results,'global_coverage':global_results,'shadow_nominal_contact_normals':contacts,
 'limits':'Static rendered/sampled MANO surface is not native physical collision rollout. No surface equality, contact-force equality, grasp-family preservation, or generalization claim is established. Grid areas are resolution-dependent approximations; part labels are skinning-derived.'
}
report['source_grasp_quality_vs_mapping']='Native penetration or missing contacts belong to the original proposal and are not by themselves mapping failure. Fidelity assessment must preserve poor native proposals too; this comparison measures changes introduced by target embodiment/retargeting rather than repairing or rejecting the source.'
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'hug_surface_confirmation.json').write_text(json.dumps(report,indent=2)+'\n')

def box_distance(points):
    d=np.abs(points)-half
    return np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)

cube_vertices=np.array([[x,y,z] for x in [-.025,.025] for y in [-.025,.025] for z in [-.025,.025]])
cube_tri,_=hull_triangles(cube_vertices)
fig=plt.figure(figsize=(14,8.5),layout='constrained')
axes=[fig.add_subplot(1,2,i+1,projection='3d') for i in range(2)]
for ax in axes:
    ax.set_proj_type('ortho');ax.view_init(elev=25,azim=135)
    ax.set_xlim(-65,135);ax.set_ylim(-95,65);ax.set_zlim(-30,165);ax.set_box_aspect((200,160,195))
    ax.add_collection3d(Poly3DCollection(cube_tri*1000,facecolors='#b6c0cc',alpha=.12,edgecolors='#313945',linewidths=.6))
    ax.set_xlabel('Object X (mm)');ax.set_ylabel('Object Y (mm)');ax.set_zlabel('Object Z (mm)')
    ax.tick_params(labelsize=8)
native_colors=np.array([COLORS[i] for i in native_face_parts])
axes[0].add_collection3d(Poly3DCollection(native_triangles*1000,facecolors=native_colors,alpha=.60,edgecolors='none',axlim_clip=True))
native_near=box_distance(native_triangles.mean(axis=1))<=.002
axes[0].add_collection3d(Poly3DCollection(native_triangles[native_near]*1000,facecolors='#ffb000',alpha=.95,edgecolors='none'))
for g in geometries:
    if g['body'] in ['forearm','wrist']:continue
    tri=g['tri_object'];axes[1].add_collection3d(Poly3DCollection(tri*1000,facecolors=COLORS[PARTS.index(g['part'])],alpha=.58,edgecolors='none',axlim_clip=True))
    near=box_distance(tri.mean(axis=1))<=.002
    if near.any():axes[1].add_collection3d(Poly3DCollection(tri[near]*1000,facecolors='#ffb000',alpha=.95,edgecolors='none'))
if contacts:
    cp=np.array([c['object_position_m'] for c in contacts])*1000
    axes[1].scatter(*cp.T,c='#111111',s=25,depthshade=False)
axes[0].set_title('Saved native MANO surface\nSkinning-derived anatomy; rendered mesh',fontsize=13)
axes[1].set_title('Translated Shadow nominal collision surfaces\nActual per-geom shapes; black: compiled contacts',fontsize=13)
legend=[Patch(facecolor=COLORS[i],label=part.capitalize()) for i,part in enumerate(PARTS)]
legend+=[Patch(facecolor='#ffb000',label='Triangle center within 2 mm of cube or inside')]
fig.legend(handles=legend,loc='outside lower center',ncol=4,frameon=False,fontsize=9)
fig.suptitle('HUG cube, seed 0: native grasp → Shadow before executor closure\nIdentical initial 50 mm cube, object frame, orthographic camera and scale. Forearm/wrist carrier omitted from visible panels.',fontsize=14)
fig.savefig(OUT/'hug_native_shadow_surfaces.png',dpi=200);plt.close(fig)

fig,axs=plt.subplots(3,6,figsize=(16,9),layout='constrained')
categories=ListedColormap(['#f1f3f5','#f4db8b','#e8a12c','#b45030'])
disagreement=ListedColormap(['#f1f3f5','#9ab8d3','#dc9992','#607987'])
for i,(name,axis,sign) in enumerate(FACES):
    sel=slice(i*625,(i+1)*625);n=native_signed[sel];s=shadow_signed[sel]
    for row,d in enumerate([n,s]):
        category=np.select([d<0,d<=.002,d<=.005],[3,2,1],default=0)
        axs[row,i].imshow(category.reshape(25,25),origin='lower',extent=[-25,25,-25,25],cmap=categories,vmin=0,vmax=3,interpolation='none')
    nk=n<=.002;sk=s<=.002;diff=np.select([nk&sk,nk&~sk,sk&~nk],[3,1,2],default=0)
    axs[2,i].imshow(diff.reshape(25,25),origin='lower',extent=[-25,25,-25,25],cmap=disagreement,vmin=0,vmax=3,interpolation='none')
    axs[0,i].set_title(name,fontsize=12)
    coordinate_axes=[j for j in range(3) if j!=axis]
    for row in range(3):
        axs[row,i].set_aspect('equal');axs[row,i].tick_params(labelsize=8)
        axs[row,i].set_xlabel(f"Object {'XYZ'[coordinate_axes[0]]} (mm)",fontsize=8)
        axs[row,i].set_ylabel(f"Object {'XYZ'[coordinate_axes[1]]} (mm)",fontsize=8)
axs[0,0].set_ylabel('Native MANO\nObject Z (mm)');axs[1,0].set_ylabel('Shadow collision union\nObject Z (mm)');axs[2,0].set_ylabel('2 mm footprint difference\nObject Z (mm)')
fig.legend(handles=[Patch(facecolor=categories(3),label='Cube point inside hand / collider'),Patch(facecolor=categories(2),label='Outside, within 2 mm'),Patch(facecolor=categories(1),label='Outside, 2–5 mm'),Patch(facecolor=disagreement(1),label='2 mm native only'),Patch(facecolor=disagreement(2),label='2 mm Shadow only'),Patch(facecolor=disagreement(3),label='2 mm shared')],loc='outside lower center',ncol=3,frameon=False,fontsize=9)
fig.suptitle('HUG-only native/Shadow cube-face surface footprints at nominal initial pose\n25 × 25 deterministic cell centers per face (2 mm spacing). Occupancy/proximity is geometric evidence; no native collision rollout.',fontsize=14)
fig.savefig(OUT/'hug_cube_face_surface_footprints.png',dpi=200);plt.close(fig)
print(json.dumps({'global_coverage':global_results,'face_results':face_results,'shadow_nominal_contacts':contacts,'mujoco_time_s':float(data.time)},indent=2))
