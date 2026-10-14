"""Frozen static signed hand/object profiles. No optimization or physics."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, pickle, sys, time, warnings
import numpy as np
import mujoco
import xml.etree.ElementTree as ET
CALLS=[]
def forbidden(*args,**kwargs):
    CALLS.append('integration_attempt')
    raise RuntimeError('Static signed profile forbids physics')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
from scipy.spatial import ConvexHull

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def weighted_quantile(values,weights,quantiles):
    order=np.argsort(values);cum=np.cumsum(weights[order]);return [float(values[order[min(np.searchsorted(cum,q*cum[-1]),len(order)-1)]]) for q in quantiles]
def profile(sd,weights,p):
    if len(sd)==0:return {'sample_count':0,'total_weight':0.,'empty':True}
    assert np.isfinite(sd).all();neg=sd<0;pos=sd>0;total=weights.sum();edges=np.array([-np.inf,*p['distance_bins_mm'],np.inf]);histcount=np.histogram(sd*1000,edges)[0];histweight=np.histogram(sd*1000,edges,weights=weights)[0];assert histcount.sum()==len(sd)
    return {'sample_count':len(sd),'total_weight':float(total),'minimum_signed_distance_mm':float(sd.min()*1000),'maximum_signed_distance_mm':float(sd.max()*1000),'finite_sampled_max_penetration_mm':float(max(0.,-sd.min())*1000),'negative_count':int(neg.sum()),'positive_count':int(pos.sum()),'boundary_count':int((sd==0).sum()),'negative_weight':float(weights[neg].sum()),'negative_weight_fraction':float(weights[neg].sum()/total),'positive_weight_fraction':float(weights[pos].sum()/total),'boundary_weight_fraction':float(weights[sd==0].sum()/total),'mean_signed_distance_mm':float(np.average(sd*1000,weights=weights)),'signed_distance_weighted_quantiles_mm':dict(zip(['p01','p05','p25','p50','p75','p95','p99'],weighted_quantile(sd*1000,weights,[.01,.05,.25,.5,.75,.95,.99]),strict=True)),'conditional_negative_depth_mean_mm':float(np.average(-sd[neg]*1000,weights=weights[neg])) if neg.any() else None,'signed_histogram':{'edges_mm':['-infinity',*p['distance_bins_mm'],'+infinity'],'sample_counts':histcount.tolist(),'weights':histweight.tolist()},'miss_bands':{str(b):{'all_samples_unsigned_farther_than_band':bool(np.min(np.abs(sd))>b/1000),'all_samples_positive_farther_than_band':bool(sd.min()>b/1000),'unsigned_near_count':int((np.abs(sd)<=b/1000).sum()),'negative_count':int(neg.sum())} for b in p['miss_bands_mm']}}
def profile_parts(sd,w,label,parts,p,units):
    whole=profile(sd,w,p);per={part:profile(sd[label==j],w[label==j],p) for j,part in enumerate(parts)};fingers=['thumb','index','middle','ring','little'];return {'weight_units':units,'whole':whole,'per_part':per,'per_finger_miss_count':{str(b):{'unsigned_proximity_miss_count':sum(per[f]['miss_bands'][str(b)]['all_samples_unsigned_farther_than_band'] for f in fingers if not per[f].get('empty')),'exterior_gap_miss_count':sum(per[f]['miss_bands'][str(b)]['all_samples_positive_farther_than_band'] for f in fingers if not per[f].get('empty')),'empty_finger_sample_count':sum(per[f].get('empty',False) for f in fingers)} for b in p['miss_bands_mm']}}
def strict_inside(points,g,tol):
    local=(points-g['pos'])@g['R'];kind=g['kind'];size=g['size']
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:return np.linalg.norm(local,axis=1)<size[0]-tol
    if kind==mujoco.mjtGeom.mjGEOM_CYLINDER:return (np.linalg.norm(local[:,:2],axis=1)<size[0]-tol)&(np.abs(local[:,2])<size[1]-tol)
    if kind==mujoco.mjtGeom.mjGEOM_BOX:return np.all(np.abs(local)<size-tol,axis=1)
    return np.max(local@g['equations'][:,:3].T+g['equations'][:,3],axis=1)<-tol
def on_boundary_with_same_normal(points,normals,g,tol):
    local=(points-g['pos'])@g['R'];localn=normals@g['R'];kind=g['kind'];size=g['size']
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        radial=np.linalg.norm(local,axis=1);sd=radial-size[0];normal=local/np.maximum(radial[:,None],1e-15)
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(local[:,:2],axis=1);delta=np.column_stack([radial-size[0],np.abs(local[:,2])-size[1]]);sd=delta.max(1);normal=np.zeros_like(local);side=delta[:,0]>=delta[:,1];normal[side,:2]=local[side,:2]/np.maximum(radial[side,None],1e-15);normal[~side,2]=np.sign(local[~side,2])
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        delta=np.abs(local)-size;axis=np.argmax(delta,axis=1);sd=delta.max(1);normal=np.zeros_like(local);normal[np.arange(len(local)),axis]=np.sign(local[np.arange(len(local)),axis])
    else:
        values=local@g['equations'][:,:3].T+g['equations'][:,3];idx=np.argmax(values,axis=1);sd=values[np.arange(len(local)),idx];normal=g['equations'][idx,:3]
    return (np.abs(sd)<=tol)&(np.einsum('ij,ij->i',normal,localn)>1-1e-6)
def primitive_tris(g,lib,chord):
    kind=g['kind'];size=g['size'];meta={'query_centroids_projected_to_analytic_surface':False,'area_weight_exact_for_actual_geometry':True}
    if g['tri'] is not None:return g['tri'],meta
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        vertices=np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]])*size[0];tri,_,_=lib.hull(vertices);level=0
        def chord_bound(triangles):
            normal,_=lib.tri_info(triangles);return float(np.max(size[0]-np.abs(np.einsum('ij,ij->i',normal,triangles[:,0]))))
        while chord_bound(tri)>chord:
            a,b,c=tri[:,0],tri[:,1],tri[:,2];ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
            for vertex in (ab,bc,ca):vertex[:]=vertex/np.linalg.norm(vertex,axis=1)[:,None]*size[0]
            tri=np.concatenate([np.stack([a,ab,ca],1),np.stack([ab,b,bc],1),np.stack([ca,bc,c],1),np.stack([ab,bc,ca],1)]);level+=1
        meta.update({'primitive':'sphere','octahedral_subdivision_level':level,'conservative_maximum_facet_chord_deficit_m':chord_bound(tri),'maximum_centroid_radial_deficit_m':float(np.max(size[0]-np.linalg.norm(tri.mean(1),axis=1))),'query_centroids_projected_to_analytic_surface':True,'area_weight_exact_for_actual_geometry':False})
    else:
        r,h=size[0],size[1];segments=max(12,int(np.ceil(np.pi/np.arccos(np.clip(1-chord/r,-1,1)))));angles=np.arange(segments)*2*np.pi/segments;vertices=np.concatenate([np.column_stack([r*np.cos(angles),r*np.sin(angles),np.full(segments,z)]) for z in (-h,h)]);tri,_,_=lib.hull(vertices);meta.update({'primitive':'cylinder','angular_segments':segments,'maximum_side_chord_deficit_m':float(r*(1-np.cos(np.pi/segments))),'query_centroids_projected_to_analytic_surface':True,'area_weight_exact_for_actual_geometry':False})
    return tri,meta
def hand_queries(hand,density,lib,p):
    points=[];weights=[];normals=[];labels=[];metadata=[];gids=[]
    for g in hand:
        gdensity=p['carrier_area_density_mm'] if g['part'] in ('forearm_carrier','wrist_carrier') else density
        tri,meta=primitive_tris(g,lib,p['primitive_max_chord_error_m']);q,a,n,_=lib.subdivide(tri,gdensity/1000);kind=g['kind'];size=g['size']
        if kind==mujoco.mjtGeom.mjGEOM_SPHERE:q=q/np.linalg.norm(q,axis=1)[:,None]*size[0];n=q/size[0]
        elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
            side=np.abs(n[:,2])<.5;q[side,:2]=q[side,:2]/np.linalg.norm(q[side,:2],axis=1)[:,None]*size[0];n[side]=0;n[side,:2]=q[side,:2]/size[0]
        q=q@g['R'].T+g['pos'];n=n@g['R'].T;keep=np.ones(len(q),bool);inside=np.zeros(len(q),bool);probe=np.zeros(len(q),bool);dupe=np.zeros(len(q),bool)
        for other in hand:
            if other['id']==g['id']:continue
            inside|=strict_inside(q,other,p['target_union_interior_tolerance_m']);probe|=strict_inside(q+p['target_union_outward_probe_m']*n,other,p['target_union_interior_tolerance_m'])
            if other['id']<g['id']:dupe|=on_boundary_with_same_normal(q,n,other,p['target_union_interior_tolerance_m'])
        keep&=~inside&~probe&~dupe;metadata.append({'geom_id':g['id'],'body':g['body'],'part':g['part'],'area_density_mm':gdensity,'query_count_before_filter':len(q),'source_facet_area_mm2':float(a.sum()),'retained_count':int(keep.sum()),'retained_area_mm2':float(a[keep].sum()),'buried_inside_other_area_mm2':float(a[inside].sum()),'outward_probe_occluded_area_mm2':float(a[~inside&probe].sum()),'coincident_duplicate_area_mm2':float(a[~inside&~probe&dupe].sum()),'primitive_approximation':meta})
        points.append(q[keep]);weights.append(a[keep]);normals.append(n[keep]);labels.append(np.full(keep.sum(),lib.PARTS.index(g['part']),int));gids.append(np.full(keep.sum(),g['id'],int))
    return np.concatenate(points),np.concatenate(weights),np.concatenate(labels),metadata
def signed_to_visual(points,visualtri,lib,p):
    distance,_,_=lib.nearest_parts(points,visualtri,np.zeros(len(visualtri),np.int64),1);distance=distance[:,0];bounds=visualtri.reshape(-1,3);lo=bounds.min(0);hi=bounds.max(0);candidate=np.all((points>=lo)&(points<=hi),axis=1);inside=np.zeros(len(points),bool);winding=np.zeros(len(points));inside[candidate],winding[candidate]=lib.winding_inside(points[candidate],visualtri);sd=np.where(inside,-distance,distance);sd[distance<=p['signed_boundary_tolerance_m']]=0
    return sd,{'winding_min':float(winding.min()),'winding_max':float(winding.max()),'outside_visual_AABB_winding_skipped_count':int((~candidate).sum()),'absolute_winding_near_half_count':int((np.abs(np.abs(winding)-.5)<1e-3).sum()),'inside_count_before_boundary_tolerance':int(inside.sum())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent);args=ap.parse_args();out=args.out;proto=out/'protocol.json';ph=sha(proto);p=json.loads(proto.read_text());start=time.perf_counter();inputs={p['source_paths'][k]:h for k,h in p['source_sha256'].items()}
    for path,h in inputs.items():assert sha(path)==h,path
    spec=importlib.util.spec_from_file_location('geometry_library',p['source_paths']['query_library']);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    for name in p['guards']:setattr(mujoco,name,forbidden)
    guardtests=[]
    for name in p['guards']:
        try:getattr(mujoco,name)()
        except RuntimeError:guardtests.append(name)
    CALLS.clear()
    inventory=json.loads(Path(p['source_paths']['inventory']).read_text())['modern_cases'];reference=json.loads(Path(p['source_paths']['reference_report']).read_text());refmap={c['native_sha256']:c for c in reference['cases']};geo=json.loads(Path(p['source_paths']['geometry_cause_report']).read_text());geomap={c['object_id']:c for c in geo['cases']};prior=Path(p['source_paths']['inventory']).parent/'static_saved_hug_fk.json';inputs[str(prior)]=sha(prior);fkmap={c['native_sha256']:c for c in json.loads(prior.read_text())['cases']};urdf=Path(reference['urdf_path']);assert sha(urdf)==reference['urdf_sha256'];inputs[str(urdf)]=sha(urdf);source_limits={joint.get('name'):[float(joint.find('limit').get('lower')),float(joint.find('limit').get('upper'))] for joint in ET.parse(urdf).getroot().findall('joint') if joint.find('limit') is not None and joint.find('limit').get('lower') is not None}
    inputs[str(lib.ASSET)]=sha(lib.ASSET)
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(lib.ASSET.read_bytes(),encoding='latin1')
    mano_weights=np.asarray(asset['weights']);gw=np.column_stack([mano_weights[:,ids].sum(1) for ids in lib.GROUPS.values()]);vertexlabels=np.argmax(gw,axis=1);records=[];seen=set()
    for index in p['inventory_indices']:
        case=inventory[index];print(json.dumps({'start':case['object_id'],'inventory_index':index}),flush=True);meta,tri,nativelabels,norm,hand,objects,m,d,initial,objadr,tobj,Robj=lib.load_case(case,gw,fkmap);inputs.update(meta['input_sha256']);initialobjR=Robj.copy();initialobjpos=tobj.copy();nominalq=d.qpos.copy();source=pickle.loads(Path(case['native']['path']).read_bytes());T=np.load(case['frame_files']['T_world_camera.npy']['path']);nativevertices=(np.asarray(source['mesh_vertices'])@T[:3,:3].T+T[:3,3]-tobj)@Robj;visualid=meta['disabled_visual_geoms'][0]['id'];mid=int(m.geom_dataid[visualid]);va=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);cf=m.mesh_face[int(m.mesh_faceadr[mid]):int(m.mesh_faceadr[mid])+int(m.mesh_facenum[mid])].astype(int);cv=(m.mesh_vert[va:va+nv].astype(float)@d.geom_xmat[visualid].reshape(3,3).T+d.geom_xpos[visualid]-tobj)@Robj;visualtri=cv[cf]
        assert geomap[case['object_id']]['visual_topology']['closed_consistently_oriented_for_winding'];assert geomap[case['object_id']]['source_mjb_sha256']==case['scene_mjb']['sha256'];levels=[4,2] if case['object_id'] not in seen else [4];seen.add(case['object_id']);ref=refmap[case['native']['sha256']];vertexsd,vertexwind=signed_to_visual(nativevertices,visualtri,lib,p);record={'object_id':case['object_id'],'native_sha256':case['native']['sha256'],'observation_group_id':case['observation_group_id'],'source_inventory_index':index,'native_unweighted_vertex_profile':profile_parts(vertexsd,np.ones(len(nativevertices)),vertexlabels,lib.PARTS,p,'vertex count'),'native_vertex_winding':vertexwind,'resolutions':{}};rootadr=int(m.jnt_qposadr[lib.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        for density in levels:
            nq,na,_,_=lib.subdivide(tri,density/1000);nl=[]
            for triangle,label in zip(tri,nativelabels,strict=True):small,_,_,_=lib.subdivide(triangle[None],density/1000);nl.extend([label]*len(small))
            nl=np.asarray(nl,int);assert len(nl)==len(nq);nsd,nw=signed_to_visual(nq,visualtri,lib,p);rr={'native_area_profile':profile_parts(nsd,na,nl,lib.PARTS,p,'mm2 native triangle area'),'native_winding':nw,'methods':{}}
            for method in p['methods']:
                fixed=ref['methods'][method];assert fixed['status']=='computed';q=np.asarray(fixed['qpos_rad']);lower=np.asarray([source_limits[n][0] for n in fixed['joint_names']]);upper=np.asarray([source_limits[n][1] for n in fixed['joint_names']]);assert np.all(q>=lower-1e-12) and np.all(q<=upper+1e-12)
                d.qpos[:]=nominalq;d.qpos[rootadr:rootadr+3]=fixed['root_target_position_m'];d.qpos[rootadr+3:rootadr+7]=fixed['root_target_quaternion_wxyz'];d.qvel[:]=0;scene_excess={}
                for name,val in zip(fixed['joint_names'],q,strict=True):
                    j=lib.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);d.qpos[int(m.jnt_qposadr[j])]=val;extra=max(m.jnt_range[j,0]-val,val-m.jnt_range[j,1],0.);scene_excess[name]=float(extra)
                if m.nmocap:d.mocap_pos[0]=fixed['root_target_position_m'];d.mocap_quat[0]=fixed['root_target_quaternion_wxyz']
                mujoco.mj_forward(m,d);targetq=d.qpos.copy();assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial);assert np.array_equal(d.xpos[lib.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')],initialobjpos);assert np.array_equal(d.xmat[lib.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')].reshape(3,3),initialobjR)
                newhand=[]
                for g in hand:
                    ng=lib.geom_rep(m,d,g['id'],Robj,tobj);ng['part']=g['part'];newhand.append(ng)
                hq,ha,hl,hmeta=hand_queries(newhand,density,lib,p);anatomic=hl<6;print(json.dumps({'queries':case['object_id'],'density':density,'method':method,'native':len(nq),'target_retained':len(hq),'target_palm_fingers':int(anatomic.sum()),'target_carriers':int((~anatomic).sum())}),flush=True);hsd,hw=signed_to_visual(hq,visualtri,lib,p);rr['methods'][method]={'profile':profile_parts(hsd,ha,hl,lib.PARTS,p,'mm2 sampled heterogeneous full target facet area; curved primitive weights approximate'),'primary_palm_fingers_profile':profile_parts(hsd[anatomic],ha[anatomic],hl[anatomic],lib.PARTS,p,'mm2 sampled palm/finger facet area; curved primitive weights approximate'),'carriers_profile':profile(hsd[~anatomic],ha[~anatomic],p),'winding':hw,'hand_geom_sampling':hmeta,'source_physical_urdf_bounds_pass':True,'source_physical_lower_rad':lower.tolist(),'source_physical_upper_rad':upper.tolist(),'captured_scene_decimal_bound_excess_rad':scene_excess,'captured_scene_max_decimal_bound_excess_rad':max(scene_excess.values()),'original_target_qpos_and_root_unchanged':True,'joint_names':fixed['joint_names'],'qpos_rad':fixed['qpos_rad'],'root_target_position_m':fixed['root_target_position_m'],'root_target_quaternion_wxyz':fixed['root_target_quaternion_wxyz'],'time_s':float(d.time)};assert np.array_equal(d.qpos,targetq)
            record['resolutions'][str(density)]=rr
        records.append(record);write(out/f'case_{index:02d}.json',record)
    for path,h in inputs.items():assert sha(path)==h,path
    assert sha(proto)==ph and not CALLS
    report={'schema':p['schema'],'protocol_sha256':ph,'script_sha256':sha(__file__),'input_sha256_before_and_after':inputs,'runtime_seconds':time.perf_counter()-start,'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'mujoco':mujoco.__version__},'guard_selftests':guardtests,'integration_attempts':len(CALLS),'physics_steps':0,'new_inference':False,'new_eval_data':False,'optimizer_used':False,'production_changes':False,'cases':records,'limitations':p['limitations']};write(out/'signed_hand_visual_profiles.json',report);print(json.dumps({'completed':True,'seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
