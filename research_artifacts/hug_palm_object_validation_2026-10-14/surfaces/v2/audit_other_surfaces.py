"""Frozen, HUG-only static surface diagnostics on six existing noncube cases.

No generator, optimizer, physics integration or new evaluation episode. Run with
preserved runtime PYTHONPATH from docs/HUG_SINGLE_EVAL_2026-10-13.md. New writes
stay in --out (default this script's directory); licensed assets are read only.
"""
from pathlib import Path
import argparse, ast, hashlib, json, pickle, sys, time, warnings
import numpy as np
import scipy
import mujoco

def forbidden(*args, **kwargs):
    raise RuntimeError('Static HUG audit forbids physics integration')
for api in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, api, forbidden)
# Installed before importing project modules or loading any simulation model.
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import numba
from numba import njit, prange
numba.set_num_threads(4)
import chumpy
from grasp_failure_prediction.evaluation.pose_validation import _link_positions
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root

PARTS = ['palm', 'index', 'middle', 'ring', 'little', 'thumb', 'forearm_carrier', 'wrist_carrier']
GROUPS = {'palm':[0], 'index':[1,2,3], 'middle':[4,5,6], 'ring':[10,11,12], 'little':[7,8,9], 'thumb':[13,14,15]}
ASSET = Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug/assets/mano_models/models/MANO_RIGHT.pkl')
URDF = default_dex_urdf_root() / 'shadow_hand/shadow_hand_right.urdf'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def ident(m, kind, name):
    v = int(mujoco.mj_name2id(m,kind,name)); assert v>=0, name; return v

def hull(vertices):
    h=ConvexHull(vertices); tri=vertices[h.simplices].copy()
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); flip=np.einsum('ij,ij->i',n,h.equations[:,:3])<0
    tri[flip]=tri[flip][:,[0,2,1]]
    assert np.max(vertices@h.equations[:,:3].T+h.equations[:,3])<2e-8
    return tri,h.equations,h

def tri_info(tri):
    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); length=np.linalg.norm(normal,axis=1)
    assert np.all(length>1e-15), 'Degenerate surface triangles'
    return normal/length[:,None],length/2

@njit(inline='always')
def closest_triangle(px,py,pz,a,b,c):
    ab=b-a; ac=c-a; nx=ab[1]*ac[2]-ab[2]*ac[1]; ny=ab[2]*ac[0]-ab[0]*ac[2]; nz=ab[0]*ac[1]-ab[1]*ac[0]
    nn=nx*nx+ny*ny+nz*nz
    apx=px-a[0]; apy=py-a[1]; apz=pz-a[2]
    off=(apx*nx+apy*ny+apz*nz)/nn
    qx=px-off*nx; qy=py-off*ny; qz=pz-off*nz
    d00=np.dot(ab,ab); d01=np.dot(ab,ac); d11=np.dot(ac,ac)
    d20=(qx-a[0])*ab[0]+(qy-a[1])*ab[1]+(qz-a[2])*ab[2]
    d21=(qx-a[0])*ac[0]+(qy-a[1])*ac[1]+(qz-a[2])*ac[2]
    den=d00*d11-d01*d01; s=(d11*d20-d01*d21)/den; t=(d00*d21-d01*d20)/den
    if s>=0 and t>=0 and s+t<=1:
        return qx,qy,qz,(px-qx)**2+(py-qy)**2+(pz-qz)**2
    best=1e100; bx=0.; by=0.; bz=0.
    for edge in range(3):
        if edge==0: l=a; r=b
        elif edge==1: l=b; r=c
        else: l=c; r=a
        ex=r[0]-l[0]; ey=r[1]-l[1]; ez=r[2]-l[2]
        u=((px-l[0])*ex+(py-l[1])*ey+(pz-l[2])*ez)/(ex*ex+ey*ey+ez*ez)
        u=min(max(u,0.),1.); xx=l[0]+u*ex; yy=l[1]+u*ey; zz=l[2]+u*ez
        ds=(px-xx)**2+(py-yy)**2+(pz-zz)**2
        if ds<best: best=ds; bx=xx; by=yy; bz=zz
    return bx,by,bz,best

@njit(parallel=True)
def nearest_parts(points,triangles,labels,nparts):
    distances=np.full((len(points),nparts),np.inf); closest=np.zeros((len(points),3)); indices=np.full(len(points),-1,np.int64)
    for i in prange(len(points)):
        px=points[i,0]; py=points[i,1]; pz=points[i,2]; best=np.inf
        for j in range(len(triangles)):
            qx,qy,qz,d=closest_triangle(px,py,pz,triangles[j,0],triangles[j,1],triangles[j,2])
            if d<distances[i,labels[j]]: distances[i,labels[j]]=d
            if d<best:
                best=d; closest[i,0]=qx; closest[i,1]=qy; closest[i,2]=qz; indices[i]=j
    return np.sqrt(distances),closest,indices

@njit(parallel=True)
def winding_inside(points,triangles):
    inside=np.zeros(len(points),np.bool_); winding=np.zeros(len(points))
    for i in prange(len(points)):
        omega=0.
        for j in range(len(triangles)):
            a=triangles[j,0]-points[i]; b=triangles[j,1]-points[i]; c=triangles[j,2]-points[i]
            na=np.sqrt(np.dot(a,a)); nb=np.sqrt(np.dot(b,b)); nc=np.sqrt(np.dot(c,c))
            det=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0])
            den=na*nb*nc+np.dot(a,b)*nc+np.dot(b,c)*na+np.dot(c,a)*nb
            omega+=2*np.arctan2(det,den)
        winding[i]=omega/(4*np.pi); inside[i]=abs(winding[i])>.5
    return inside,winding

def geom_rep(m,d,g, Robj,tobj):
    kind=int(m.geom_type[g]); size=m.geom_size[g].copy(); mid=int(m.geom_dataid[g]); eq=None; meshmeta=None
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        va=int(m.mesh_vertadr[mid]); nv=int(m.mesh_vertnum[mid]); vertices=m.mesh_vert[va:va+nv].astype(float)
        tri,eq,h=hull(vertices)
        meshmeta={'mesh_id':mid,'compiled_vertices':nv,'compiled_visual_face_count':int(m.mesh_facenum[mid]),'scipy_convex_hull_triangle_count':len(tri), 'hull_volume_m3':float(h.volume), 'hull_vertex_halfspace_max_residual_m':float(np.max(vertices@eq[:,:3].T+eq[:,3])), 'compiled_mesh_graph_address':int(m.mesh_graphadr[mid]), 'compiled_vertices_sha256':hashlib.sha256(vertices.tobytes()).hexdigest()}
        rep='per-geom convex hull of actual compiled mesh vertices; geometric collision envelope, not visual triangles'
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        vertices=np.array([[x,y,z] for x in [-size[0],size[0]] for y in [-size[1],size[1]] for z in [-size[2],size[2]]]); tri,eq,_=hull(vertices); rep='exact analytic box'
    elif kind in [mujoco.mjtGeom.mjGEOM_SPHERE,mujoco.mjtGeom.mjGEOM_CYLINDER]:
        tri=None; rep='exact analytic sphere' if kind==mujoco.mjtGeom.mjGEOM_SPHERE else 'exact analytic cylinder'
    else: raise ValueError(f'Unsupported actual collider {kind}')
    R=Robj.T@d.geom_xmat[g].reshape(3,3); pos=(d.geom_xpos[g]-tobj)@Robj
    triobj=None if tri is None else tri@R.T+pos
    return {'id':g,'body_id':int(m.geom_bodyid[g]),'body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])),'kind':kind,'size':size,'pos':pos,'R':R,'tri':tri,'tri_object':triobj,'equations':eq,'representation':rep,'mesh':meshmeta,'contype':int(m.geom_contype[g]),'conaffinity':int(m.geom_conaffinity[g])}

def geom_query(g,points):
    p=(points-g['pos'])@g['R']; kind=g['kind']; size=g['size']
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        length=np.linalg.norm(p,axis=1); sd=length-size[0]; norm=p/np.maximum(length[:,None],1e-15);norm[length<1e-15]=[1,0,0]; q=p-sd[:,None]*norm
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        delta=np.abs(p)-size; outside=np.maximum(delta,0); od=np.linalg.norm(outside,axis=1); sd=od+np.minimum(delta.max(axis=1),0)
        norm=outside*np.sign(p)/np.maximum(od[:,None],1e-15); inside=od==0; rows=np.flatnonzero(inside); axis=np.argmax(delta[inside],axis=1); norm[inside]=0; norm[rows,axis]=np.where(p[rows,axis]>=0,1.,-1.);q=p-sd[:,None]*norm
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(p[:,:2],axis=1); rdir=p[:,:2]/np.maximum(radial[:,None],1e-15);rdir[radial<1e-15]=[1,0]
        delta=np.column_stack([radial-size[0],np.abs(p[:,2])-size[1]]); outside=np.maximum(delta,0); od=np.linalg.norm(outside,axis=1); sd=od+np.minimum(delta.max(axis=1),0)
        norm=np.column_stack([rdir*outside[:,0:1],np.sign(p[:,2])*outside[:,1]])/np.maximum(od[:,None],1e-15); inside=od==0; side=delta[:,0]>=delta[:,1]; norm[inside]=0; norm[inside&side,:2]=rdir[inside&side];norm[inside&~side,2]=np.where(p[inside&~side,2]>=0,1.,-1.);q=p-sd[:,None]*norm
    else:
        ds,q,idx=nearest_parts(p,g['tri'],np.zeros(len(g['tri']),np.int64),1); distance=ds[:,0]
        inn=np.max(p@g['equations'][:,:3].T+g['equations'][:,3],axis=1)<=1e-10
        sd=np.where(inn,-distance,distance);norm=tri_info(g['tri'])[0][idx]
    return sd,q@g['R'].T+g['pos'],norm@g['R'].T

def object_membership(points,objs,tol=-1e-10):
    result=np.zeros(len(points),bool)
    for g in objs:
        p=(points-g['pos'])@g['R'];result|=np.max(p@g['equations'][:,:3].T+g['equations'][:,3],axis=1)<tol
    return result

def subdivide(tri,density):
    normals,areas=tri_info(tri); pts=[];weights=[];norms=[];depths=[]
    for face,n,area in zip(tri,normals,areas,strict=True):
        edges=np.linalg.norm(face-np.roll(face,1,axis=0),axis=1);level=max(0,int(np.ceil(np.log(area/(density*density))/np.log(4))))
        children=face[None].copy()
        for _ in range(level):
            a=children[:,0];b=children[:,1];c=children[:,2];ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            children=np.concatenate([np.stack([a,ab,ca],1),np.stack([ab,b,bc],1),np.stack([ca,bc,c],1),np.stack([ab,bc,ca],1)])
        assert area/len(children)<=density*density*(1+1e-12)
        pts.append(children.mean(axis=1));weights.append(np.full(len(children),area/len(children)*1e6));norms.append(np.tile(n,(len(children),1)));depths.append(level)
    return np.concatenate(pts),np.concatenate(weights),np.concatenate(norms),depths

def object_queries(objs,edge_mm):
    allq=[];alla=[];alln=[];allids=[]; meta=[]; discarded=0.;coincident=0.
    for g in objs:
        q,a,n,levels=subdivide(g['tri_object'],edge_mm/1000); keep=np.ones(len(q),bool);dupe=np.zeros(len(q),bool)
        for other in objs:
            if other['id']==g['id']:continue
            p=(q-other['pos'])@other['R'];values=p@other['equations'][:,:3].T+other['equations'][:,3];res=values.max(axis=1)
            keep&=res>=-1e-10
            if other['id']<g['id']:
                facet=np.argmax(values,axis=1);on=np.abs(res)<=1e-10
                othernorm=other['equations'][facet,:3]@other['R'].T
                dupe|=on&(np.einsum('ij,ij->i',n,othernorm)>1-1e-10)
        discarded+=float(a[~keep].sum());coincident+=float(a[keep&dupe].sum());keep&=~dupe
        meta.append({'geom_id':g['id'],'source_hull_area_mm2':float(a.sum()),'query_count_before_overlap_removal':len(q),'query_count_after_overlap_removal':int(keep.sum()),'retained_area_mm2':float(a[keep].sum()),'source_triangle_subdivision_depth_min':min(levels),'source_triangle_subdivision_depth_max':max(levels), 'query_weight_mm2':stats(a), 'retained_query_weight_mm2':stats(a[keep]), 'child_max_edge_mm':stats(np.repeat(np.linalg.norm(g['tri_object']-np.roll(g['tri_object'],1,axis=1),axis=2).max(axis=1)*1000/(2.**np.asarray(levels)),4**np.asarray(levels)))})
        allq.append(q[keep]);alla.append(a[keep]);alln.append(n[keep]);allids.append(np.full(keep.sum(),g['id'],int))
    return np.concatenate(allq),np.concatenate(alla),np.concatenate(alln),np.concatenate(allids),{'geoms':meta,'interior_discarded_area_mm2':discarded,'coincident_lower_id_assigned_area_mm2':coincident,'retained_union_surface_area_mm2':float(sum(x.sum() for x in alla)), 'query_weight_mm2':stats(np.concatenate(alla)), 'area_density_mm':edge_mm}

def stats(x,weights=None):
    if not len(x):return None
    r={'min':float(x.min()),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(x.max())}
    if weights is not None:r['area_weighted_mean']=float(np.average(x,weights=weights));order=np.argsort(x);r['area_weighted_median']=float(x[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)])
    return r

def coverage(n,s,a):
    inter=n&s; union=n|s; na=a[n].sum(); sa=a[s].sum(); ia=a[inter].sum(); ua=a[union].sum()
    return {'native_count':int(n.sum()),'shadow_count':int(s.sum()),'intersection_count':int(inter.sum()),'union_count':int(union.sum()),'native_area_mm2':float(na),'shadow_area_mm2':float(sa),'intersection_area_mm2':float(ia),'union_area_mm2':float(ua),'lost_native_area_mm2':float(a[n&~s].sum()),'new_shadow_area_mm2':float(a[s&~n].sum()),'jaccard':float(ia/ua) if ua else None,'native_retention':float(ia/na) if na else None,'shadow_overlap':float(ia/sa) if sa else None}

def compare(query,area,native_tri,native_labels,native_norm,shadow,objects,geomid=None):
    nd,nq,ni=nearest_parts(query,native_tri,native_labels,len(PARTS)); npart=native_labels[ni]; rows=np.arange(len(query));ndist=nd[rows,npart]; nn=native_norm[ni]
    ninside,wind=winding_inside(query,native_tri)
    sd=np.full_like(nd,np.inf);sq=np.full((len(query),len(PARTS),3),np.nan);sn=np.full_like(sq,np.nan);sinside=np.zeros(len(query),bool)
    evaluated=[]
    for g in shadow:
        if g['tri_object'] is not None:
            lo=g['tri_object'].reshape(-1,3).min(0);hi=g['tri_object'].reshape(-1,3).max(0)
            gap=np.maximum(np.maximum(lo-query,query-hi),0);sel=np.linalg.norm(gap,axis=1)<=.005+1e-12
        else:sel=np.ones(len(query),bool)
        evaluated.append({'geom_id':g['id'],'queries_evaluated':int(sel.sum())})
        if not sel.any():continue
        sg,q,norm=geom_query(g,query[sel]);sinside[sel]|=sg<0;part=PARTS.index(g['part']);idx=np.flatnonzero(sel);dist=np.abs(sg);update=dist<sd[idx,part];idx=idx[update];sd[idx,part]=dist[update];sq[idx,part]=q[update];sn[idx,part]=norm[update]
    spart=np.argmin(sd,axis=1); sdist=sd[rows,spart];sc=sq[rows,spart];ssn=sn[rows,spart]
    # Infinity outside the fixed 5 mm broad phase cannot affect band diagnostics.
    valid=sdist<=.005; assert np.all(np.isfinite(sc[valid]))
    ext=(~ninside)&(~sinside)&(ndist>1e-8)&(sdist>1e-8)
    nobj=object_membership(nq,objects);sobj=np.zeros(len(query),bool);sobj[valid]=object_membership(sc[valid],objects)
    angles=np.full(len(query),np.nan);angles[valid]=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',nn[valid],ssn[valid]),-1,1)))
    separ=np.linalg.norm(nq-sc,axis=1)*1000
    def paired(sel):
        return {'query_count':int(sel.sum()),'area_mm2':float(area[sel].sum()),'normal_angle_deg':stats(angles[sel],area[sel]),'closest_point_separation_mm':stats(separ[sel],area[sel]),'both_closest_inside_object_count':int((sel&nobj&sobj).sum()),'one_closest_inside_object_count':int((sel&(nobj^sobj)).sum()),'neither_closest_inside_object_count':int((sel&~nobj&~sobj).sum()),'both_closest_inside_object_area_mm2':float(area[sel&nobj&sobj].sum()),'neither_closest_inside_object_area_mm2':float(area[sel&~nobj&~sobj].sum()),'neither_inside_object_normal_angle_deg':stats(angles[sel&~nobj&~sobj],area[sel&~nobj&~sobj])}
    bands={}
    for band in [1,2,5]:
        b=band/1000;npnear=ndist<=b;spnear=sdist<=b;ng=ninside|npnear;sg=sinside|spnear
        ne=(~ninside)&(ndist>1e-8)&npnear;se=(~sinside)&(sdist>1e-8)&spnear;common=ext&npnear&spnear;same=common&(npart==spart);different=common&(npart!=spart)
        bands[str(band)]={'whole_hand_unsigned_near':coverage(npnear,spnear,area),'occupied_or_near':coverage(ng,sg,area),'independent_per_part_unsigned':{part:coverage(nd[:,j]<=b,sd[:,j]<=b,area) for j,part in enumerate(PARTS)},'exterior_nearest_labels':{'native_exterior_near_count':int(ne.sum()),'shadow_exterior_near_count':int(se.sum()),'common_both_exterior_near_count':int(common.sum()),'common_same_label':paired(same),'common_different_label':paired(different),'native_near_shadow_not_exterior_near_count':int((ne&~se).sum()),'shadow_near_native_not_exterior_near_count':int((se&~ne).sum()),'native_near_shadow_not_exterior_near_area_mm2':float(area[ne&~se].sum()),'shadow_near_native_not_exterior_near_area_mm2':float(area[se&~ne].sum()),'common_label_confusion_counts':{part:{other:int((common&(npart==j)&(spart==k)).sum()) for k,other in enumerate(PARTS)} for j,part in enumerate(PARTS)},'common_same_label_by_part':{part:paired(same&(npart==j)) for j,part in enumerate(PARTS)}}}
        if geomid is not None:bands[str(band)]['per_object_geom_unsigned']={str(g['id']):{'whole_hand':coverage(npnear[geomid==g['id']],spnear[geomid==g['id']],area[geomid==g['id']]),'by_part':{part:coverage(nd[geomid==g['id'],j]<=b,sd[geomid==g['id'],j]<=b,area[geomid==g['id']]) for j,part in enumerate(PARTS)}} for g in objects}
    return {'query_count':len(query),'native_winding_occupancy':{'count':int(ninside.sum()),'area_mm2':float(area[ninside].sum()),'winding_value_range':[float(wind.min()),float(wind.max())]},'shadow_any_collider_occupancy':{'count':int(sinside.sum()),'area_mm2':float(area[sinside].sum())},'shadow_geometry_query_counts':evaluated,'bands':bands}, {'query_object_m':query,'area_mm2':area,'native_distance_by_part_m':nd,'shadow_distance_by_part_m':sd,'native_nearest_part':npart,'shadow_nearest_part':spart,'native_inside':ninside,'shadow_inside':sinside}

def load_case(case, groupweights,fk_byhash):
    files=[case['native'],case['scene_mjb'],case['stored_shadow_command'],*case['frame_files'].values()]
    files+= [{'path':case['native']['inference_report_path'],'sha256':case['native']['inference_report_sha256']},{'path':case['collection_metadata_path'],'sha256':case['collection_metadata_sha256']}]
    for r in files: assert sha(r['path'])==r['sha256'],r['path']
    source=pickle.loads(Path(case['native']['path']).read_bytes()); c=case['stored_shadow_command']
    with np.load(c['path'],allow_pickle=False) as actions:
        row=c['row_index'];names=actions['joint_names'].tolist();target=actions['nominal_target_rad'][row].copy();rootpos=actions['root_target_position_m'][row].copy();rootquat=actions['root_target_quaternion_wxyz'][row].copy()
    assert len(names)==24 and names==c['joint_names']; assert target.shape==(24,);assert c['not_final_qpos_or_post_force_close_target'] and c['zero_applied_noise_at_selected_row']
    m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
    with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as state:d.qpos[:]=state['qpos']
    d.qvel[:]=0; mujoco.mj_forward(m,d)
    objid=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');forearm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm'); rootadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);objadr=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')])
    initial=d.qpos[objadr:objadr+7].copy(); initialpos=d.xpos[objid].copy();initialR=d.xmat[objid].reshape(3,3).copy()
    d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
    for name,v in zip(names,target,strict=True):
        j=ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);assert m.jnt_range[j,0]-1e-9<=v<=m.jnt_range[j,1]+1e-9;d.qpos[int(m.jnt_qposadr[j])]=v
    if m.nmocap:d.mocap_pos[0]=rootpos;d.mocap_quat[0]=rootquat
    mujoco.mj_forward(m,d);assert d.time==0;assert np.array_equal(d.qpos[objadr:objadr+7],initial);assert np.array_equal(d.xpos[objid],initialpos);assert np.array_equal(d.xmat[objid].reshape(3,3),initialR)
    T=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False); verts=(np.asarray(source['mesh_vertices'])@T[:3,:3].T+T[:3,3]-initialpos)@initialR;faces=np.asarray(source['mesh_faces'],int);tri=verts[faces];norm,_=tri_info(tri)
    volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6);assert abs(volume)>1e-9
    if volume<0:norm*=-1
    vlabel=np.argmax(groupweights,axis=1);fv=vlabel[faces];counts=np.column_stack([(fv==j).sum(1) for j in range(6)]);major=counts.max(1);means=groupweights[faces].mean(1);labels=np.argmax(np.where(counts==major[:,None],means,-np.inf),axis=1).astype(np.int64)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);_,cnt=np.unique(edges,axis=0,return_counts=True);assert np.all(cnt==2),'Native whole mesh must be closed before winding'
    objects=[];visual=[];allhand=[]
    def descendant(b,root):
        while b>0:
            if b==root:return True
            b=int(m.body_parentid[b])
        return False
    exclusions=[sorted([int(s)>>16,int(s)&65535]) for s in m.exclude_signature]
    pairs=[{'geom1':int(m.pair_geom1[j]),'geom2':int(m.pair_geom2[j])} for j in range(m.npair)]
    assert not pairs,'Explicit pairs require a separately frozen protocol'
    for g in range(m.ngeom):
        b=int(m.geom_bodyid[g]);active=bool(m.geom_contype[g] or m.geom_conaffinity[g])
        if descendant(b,objid):
            if active:objects.append(geom_rep(m,d,g,initialR,initialpos))
            else:visual.append({'id':g,'type':int(m.geom_type[g]),'mesh_id':int(m.geom_dataid[g]),'collision_masks':[int(m.geom_contype[g]),int(m.geom_conaffinity[g])],'disabled_visual_mesh':True,'compiled_visual_face_count':int(m.mesh_facenum[int(m.geom_dataid[g])]) if int(m.geom_type[g])==mujoco.mjtGeom.mjGEOM_MESH else None,'object_frame_vertex_bounds_m': [((m.mesh_vert[int(m.mesh_vertadr[int(m.geom_dataid[g])]):int(m.mesh_vertadr[int(m.geom_dataid[g])])+int(m.mesh_vertnum[int(m.geom_dataid[g])])].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-initialpos)@initialR).min(axis=0).tolist(),((m.mesh_vert[int(m.mesh_vertadr[int(m.geom_dataid[g])]):int(m.mesh_vertadr[int(m.geom_dataid[g])])+int(m.mesh_vertnum[int(m.geom_dataid[g])])].astype(float)@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-initialpos)@initialR).max(axis=0).tolist()] if int(m.geom_type[g])==mujoco.mjtGeom.mjGEOM_MESH else None})
        elif descendant(b,forearm) and active:
            gg=geom_rep(m,d,g,initialR,initialpos);body=gg['body'];gg['part']='forearm_carrier' if body=='forearm' else 'wrist_carrier' if body=='wrist' else 'index' if body.startswith('ff') else 'middle' if body.startswith('mf') else 'ring' if body.startswith('rf') else 'little' if body.startswith('lf') else 'thumb' if body.startswith('th') else 'palm'
            gg['object_pair_compatibility']=[{'object_geom_id':o,'mask_compatible':bool((int(m.geom_contype[g])&int(m.geom_conaffinity[o])) or (int(m.geom_contype[o])&int(m.geom_conaffinity[g]))),'body_excluded':sorted([b,int(m.geom_bodyid[o])]) in exclusions} for o in range(m.ngeom) if descendant(int(m.geom_bodyid[o]),objid) and (m.geom_contype[o] or m.geom_conaffinity[o])]
            allhand.append(gg)
    assert objects;assert all(o['kind'] in [mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH] for o in objects)
    hand=[g for g in allhand if any(p['mask_compatible'] and not p['body_excluded'] for p in g['object_pair_compatibility'])]
    assert len(hand)==len(allhand), 'Mixed object compatibility requires per-object-query handling'
    assert all(all(p['mask_compatible'] and not p['body_excluded'] for p in g['object_pair_compatibility']) for g in hand)
    links=_link_positions(m,d,URDF);prev=fk_byhash[case['native']['sha256']];expected=np.array([r['shadow_world_m'] for r in prev['links']]);delta=float(np.max(np.linalg.norm(links-expected,axis=1)));assert delta<1e-12
    contract=json.loads(Path(case['frame_files']['scene_contract.json']['path']).read_text());assert contract['geometry_content_hash']==case['geometry_content_hash'] and contract['object_id']==case['object_id'];assert np.linalg.norm(initialpos-np.asarray(contract['object_position_m']))<1e-12;assert np.max(np.abs(initialR-Rotation.from_quat(contract['object_orientation_xyzw']).as_matrix()))<1e-12
    meta={'object_id':case['object_id'],'observation_group_id':case['observation_group_id'],'generation_seed':case['native']['seed'],'native_sha256':case['native']['sha256'],'geometry_content_hash':case['geometry_content_hash'],'geometry_binding_matches_scene_contract':True,'observation_file_sha256':contract['observation_files_sha256'],'inference_input_tensor_hash':case['native']['input_tensor_hash'],'input_sha256':{r['path']:r['sha256'] for r in files},'source_hug_commit':case['native']['hug_code_commit'],'source_hug_checkpoint_sha256':case['native']['checkpoint_sha256'],'stored_command':c,'recovered_original_targets':dict(zip(names,map(float,target),strict=True)),'root_target_world_position_m':rootpos.tolist(),'root_target_world_quaternion_wxyz':rootquat.tolist(),'native_mesh':{'vertices':len(verts),'faces':len(faces),'closed_edge_incidence_two':True,'signed_volume_m3':volume,'mixed_triangle_vertex_labels':int(((counts>0).sum(1)>1).sum()),'vertex_majority_tied_triangles':int(((counts==major[:,None]).sum(1)>1).sum()),'triangle_labels':{part:int((labels==j).sum()) for j,part in enumerate(PARTS[:6])}},'object_collider_semantics':'Exposed union of per-geom compiled convex collision hulls; disabled visual mesh excluded. Does not reconstruct original visual concavity.','object_geoms':[{k:g[k] for k in ['id','body','kind','representation','mesh','contype','conaffinity']}|{'object_frame_vertex_bounds_m':[g['tri_object'].reshape(-1,3).min(axis=0).tolist(),g['tri_object'].reshape(-1,3).max(axis=0).tolist()],'object_frame_position_m':g['pos'].tolist(),'object_frame_rotation':g['R'].tolist()} for g in objects],'disabled_visual_geoms':visual,'shadow_geoms':[{k:g[k] for k in ['id','body','part','kind','representation','mesh','contype','conaffinity','object_pair_compatibility']} for g in allhand],'collision_filter':{'npair':m.npair,'nexclude':m.nexclude,'explicit_pairs':pairs,'excluded_body_pairs':exclusions},'static_invariants':{'physics_steps':0,'time_s':float(d.time),'object_qpos_unchanged':True,'object_world_pose_unchanged':True,'prior_saved_fk_max_link_difference_m':delta,'joint_count':len(names)}}
    return meta,tri,labels,norm,hand,objects,m,d,initial,objadr,initialpos,initialR

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction'));ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();out=args.out;root=args.repo;started=time.perf_counter()
    protocol=out/'protocol_v2.json';ph=sha(protocol);p=json.loads(protocol.read_text());prior=root/'research_artifacts/hug_mapping_refinement_2026-10-14';ip=prior/'saved_proposals/existing_hug_inventory.json';fp=prior/'saved_proposals/static_saved_hug_fk.json';geometry=root/'research_artifacts/hug_mapping_confirmation_2026-10-14/hug_surface_confirmation.py'
    assert sha(ip)==p['source_inventory_sha256'];assert sha(fp)==p['source_fk_report_sha256'];assert sha(geometry)==p['prior_geometry_script_sha256'];assert sha(prior/'surface/refine_hug_surfaces.py')==p['prior_refinement_script_sha256']
    oldmanifest=root/'research_artifacts/hug_mapping_confirmation_2026-10-14/manifest.json';manifest=json.loads(oldmanifest.read_text())
    for name,h in manifest['files_sha256'].items():assert sha(oldmanifest.parent/name)==h
    assetexpected=json.loads((prior/'surface/protocol.json').read_text())['mano_asset_sha256'];assert sha(ASSET)==assetexpected
    with warnings.catch_warnings():warnings.simplefilter('ignore'); asset=pickle.loads(ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);gw=np.column_stack([weights[:,ids].sum(1) for ids in GROUPS.values()])
    inventory=json.loads(ip.read_text());fk=json.loads(fp.read_text());fkmap={c['native_sha256']:c for c in fk['cases']};cases=[inventory['modern_cases'][c['source_inventory_index']] for c in p['cases']]
    for c,planned in zip(cases,p['cases'],strict=True):assert c['native']['sha256']==planned['native_sha256']
    # Only pure prior function definitions reused, hash checked above. No prior
    # model loading or writer executed. Exact acceleration is checked against it.
    tree=ast.parse(geometry.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nearest_triangles'];ns={'np':np};exec(compile(ast.Module(body=defs,type_ignores=[]),str(geometry)+':nearest-function-only','exec'),ns)
    points=np.array([[.2,.3,.4],[.5,.5,.2],[-.1,-.1,0],[0,0,1],[1,0,0]]);tri=np.array([[[0.,0,0],[1,0,0],[0,1,0]]]);ds,q,ix=nearest_parts(points,tri,np.array([0]),1);expected=np.array([[.2,.3,0],[.5,.5,0],[0,0,0],[0,0,0],[1,0,0]]);assert np.allclose(q,expected,atol=1e-14);assert np.allclose(ds[:,0],np.linalg.norm(points-expected,axis=1),atol=1e-14)
    records=[];first=set();sanity={'known_triangle_projection_and_edge_cases':5,'accelerated_vs_previous_numpy_max_distance_difference_m':0.,'accelerated_vs_previous_numpy_max_closest_point_difference_m':0.}
    for case in cases:
        print(json.dumps({'progress':'case_start','native':case['native']['path'],'object':case['object_id']}),flush=True)
        meta,tri,labels,norm,hand,objects,m,d,initial,objadr,initialpos,initialR=load_case(case,gw,fkmap)
        levels=[2,1] if case['object_id'] not in first else [2];first.add(case['object_id']);meta['resolutions']={}; nominal_qpos=d.qpos.copy()
        for edge in levels:
            tick=time.perf_counter();query,area,qn,gid,qmeta=object_queries(objects,edge);assert np.allclose(np.linalg.norm(qn,axis=1),1,atol=1e-12)
            print(json.dumps({'progress':'query_surface','object':case['object_id'],'area_density_mm':edge,'queries':len(query),'area_mm2':float(area.sum())}),flush=True)
            sel=np.linspace(0,len(query)-1,32,dtype=int);oq,oi,od=ns['nearest_triangles'](query[sel],tri,batch=32);dd,qq,ii=nearest_parts(query[sel],tri,np.zeros(len(tri),np.int64),1);ed=float(np.max(np.abs(dd[:,0]-od)));eq=float(np.max(np.linalg.norm(qq-oq,axis=1)));assert ed<1e-12 and eq<1e-12;sanity['accelerated_vs_previous_numpy_max_distance_difference_m']=max(sanity['accelerated_vs_previous_numpy_max_distance_difference_m'],ed);sanity['accelerated_vs_previous_numpy_max_closest_point_difference_m']=max(sanity['accelerated_vs_previous_numpy_max_closest_point_difference_m'],eq)
            result,arrays=compare(query,area,tri,labels,norm,hand,objects,gid);result['object_query_surface']=qmeta;result['area_density_mm']=edge;result['runtime_seconds']=time.perf_counter()-tick;meta['resolutions'][str(edge)]=result
            caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name;np.savez_compressed(out/f'{caseid}_density{edge}mm_queries.npz',**arrays,object_geom_id=gid,object_outward_normal=qn)
            assert np.array_equal(d.qpos,nominal_qpos);meta['static_invariants']['all_static_qpos_unchanged']=True
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial) and np.array_equal(d.xpos[ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')],initialpos) and np.array_equal(d.xmat[ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')].reshape(3,3),initialR)
            print(json.dumps({'progress':'resolution_complete','object':case['object_id'],'area_density_mm':edge,'seconds':result['runtime_seconds'],'occupied_near_2mm_iou':result['bands']['2']['occupied_or_near']['jaccard']}),flush=True)
        for path,h in meta['input_sha256'].items():assert sha(path)==h
        records.append(meta);write(out/f'{case["object_id"]}_{Path(case["native"]["path"]).parent.name}.json',meta)
    # Exact legacy cube grid makes regression meaningful across surface domains.
    cube=next(c for c in inventory['modern_cases'] if c['object_id']=='object01' and c['native']['seed']==0);meta,tri,labels,norm,hand,objects,*_=load_case(cube,gw,fkmap);assert len(objects)==1 and objects[0]['kind']==mujoco.mjtGeom.mjGEOM_BOX
    uv=(np.arange(25)+.5)*.002-.025; grids=[]
    for ax,sign in [(0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)]:
        qq=np.zeros((625,3));qq[:,ax]=sign*.025;other=[j for j in range(3) if j!=ax];u,v=np.meshgrid(uv,uv,indexing='xy');qq[:,other[0]]=u.ravel();qq[:,other[1]]=v.ravel();grids.append(qq)
    cr,_=compare(np.concatenate(grids),np.full(3750,4.),tri,labels,norm,hand,objects);old=json.loads((oldmanifest.parent/'hug_surface_confirmation.json').read_text());reg={}
    for band in [2,5]:
        now=cr['bands'][str(band)]['occupied_or_near'];before=old['global_coverage'][f'{band}mm'];mapping={'native_area_mm2':'native_occupied_or_near_area_mm2','shadow_area_mm2':'shadow_occupied_or_near_area_mm2','intersection_area_mm2':'intersection_area_mm2','union_area_mm2':'union_area_mm2','jaccard':'jaccard'};delta={k:now[k]-before[v] for k,v in mapping.items()};assert max(abs(v) for v in delta.values())<1e-8;reg[str(band)]={'passed':True,'new_minus_prior':delta}
    write(out/'cube_baseline_regression.json',{'native_sha256':cube['native']['sha256'],'grid':'exact prior 3750 face centers, 2 mm spacing','checks':reg,'result':cr})
    assert sha(protocol)==ph and sha(ip)==p['source_inventory_sha256'] and sha(fp)==p['source_fk_report_sha256'] and sha(ASSET)==assetexpected
    for name,h in manifest['files_sha256'].items():assert sha(oldmanifest.parent/name)==h
    report={'schema':p['schema'],'protocol_sha256':ph,'script_sha256':sha(__file__),'runtime_seconds':time.perf_counter()-started,'runtime_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__,'numba':numba.__version__},'input_provenance':{'inventory_sha256':sha(ip),'prior_fk_sha256':sha(fp),'prior_geometry_script_sha256':sha(geometry),'licensed_mano_asset_path':str(ASSET),'licensed_mano_asset_sha256':assetexpected,'licensed_asset_copied':False,'task_urdf_sha256':sha(URDF)},'scope':p['scope'],'guards_installed_before_model_load':True,'guarded_integration_apis':p['guards'],'physics_steps':0,'new_inference':False,'new_evaluation_cases':False,'retarget_optimizer_used':False,'calibration_used':False,'original_case_order_preserved':True,'historical_manifest_unchanged':True,'sanity_checks':sanity,'cube_regression':reg,'cases':records,'limitations':p['limitations'],'interpretation':'Fixed unsigned proximity bands characterize anatomical surface changes at the original mapped nominal pose, including poor native proposals. Occupancy is separate. Same-label exterior geometric normals do not certify physical contact preservation or equivalence; no acceptance threshold chosen.'}
    write(out/'hug_other_object_surfaces.json',report);print(json.dumps({'completed':True,'seconds':report['runtime_seconds'],'report':str(out/'hug_other_object_surfaces.json')}),flush=True)
if __name__=='__main__':main()
