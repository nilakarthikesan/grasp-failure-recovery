"""Static object geometry cause diagnostic; protocol frozen before metrics.

Run with the existing Python/MuJoCo/Numba runtime. Writes only to --out.
Never invokes inference, retargeting, physics integration or label collection.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ATTEMPTS = []
def forbidden(*args, **kwargs):
    ATTEMPTS.append('attempt')
    raise RuntimeError('Static diagnostic forbids integration')
for name in ('mj_step', 'mj_step1', 'mj_step2'):
    setattr(mujoco, name, forbidden)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from grasp_failure_prediction.evaluation.object_assets import load_object_geometry

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def read_obj(p):
    vertices=[]; faces=[]
    for line in Path(p).read_text().splitlines():
        tokens=line.split()
        if not tokens: continue
        if tokens[0]=='v': vertices.append(list(map(float,tokens[1:4])))
        elif tokens[0]=='f':
            indices=[int(x.split('/')[0]) for x in tokens[1:]]
            indices=[x-1 if x>0 else len(vertices)+x for x in indices]
            for j in range(1,len(indices)-1): faces.append([indices[0],indices[j],indices[j+1]])
    return np.asarray(vertices),np.asarray(faces,dtype=int)
def xml_rotation(element):
    if element.get('quat'):
        w,x,y,z=map(float,element.get('quat').split()); return Rotation.from_quat([x,y,z,w]).as_matrix()
    if element.get('euler'): return Rotation.from_euler('xyz',list(map(float,element.get('euler').split()))).as_matrix()
    assert not any(element.get(x) for x in ('axisangle','xyaxes','zaxis'))
    return np.eye(3)
def topology(v,f):
    unique, inverse=np.unique(v,axis=0,return_inverse=True); wf=inverse[f]
    directed=np.concatenate([wf[:,[0,1]],wf[:,[1,2]],wf[:,[2,0]]])
    undirected=np.sort(directed,axis=1); edges,idx,count=np.unique(undirected,axis=0,return_inverse=True,return_counts=True)
    signed=np.where(directed[:,0]<directed[:,1],1,-1); balance=np.bincount(idx,weights=signed,minlength=len(edges))
    tri=v[f]; twice_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    closed=bool(np.all(count==2) and np.all(balance==0) and np.all(twice_area>1e-15) and abs(volume)>1e-12)
    return {'original_vertices':len(v),'exact_welded_vertices':len(unique),'triangles':len(f),'edge_count':len(edges),'boundary_edges_count':int((count==1).sum()),'non_two_incident_edges_count':int((count!=2).sum()),'inconsistent_orientation_edges_count':int((balance!=0).sum()),'degenerate_triangles_count':int((twice_area<=1e-15).sum()),'signed_volume_m3':float(volume),'closed_consistently_oriented_for_winding':closed,'self_intersections_exhaustively_checked':False}
def first_ray_entry(g,q,n,tol=1e-10):
    lp=(q-g['pos'])@g['R']; ln=n@g['R']; eq=g['equations']; a=eq[:,:3]@ln; b=eq[:,:3]@lp+eq[:,3]
    enter=0.; leave=np.inf
    for aa,bb in zip(a,b,strict=True):
        if abs(aa)<1e-12:
            if bb>tol: return None
        elif aa<0: enter=max(enter,-bb/aa)
        else: leave=min(leave,-bb/aa)
    return float(max(enter,0.)) if leave>=enter and leave>=0 else None
def nearest_ray_entry(objs,q,n,source_id):
    hits=[]
    for g in objs:
        if g['id']==source_id: continue
        t=first_ray_entry(g,q,n)
        if t is not None: hits.append((t,g['id']))
    return min(hits) if hits else (None,None)
def summary(values,weights):
    order=np.argsort(values); cum=np.cumsum(weights[order]); total=weights.sum()
    return {'area_weighted_mean_mm':float(np.average(values*1000,weights=weights)),'area_weighted_p50_mm':float(values[order[np.searchsorted(cum,total*.5)]]*1000),'area_weighted_p95_mm':float(values[order[np.searchsorted(cum,total*.95)]]*1000),'max_mm':float(values.max()*1000),'area_gt_1mm_mm2':float(weights[values>.001].sum()),'area_gt_2mm_mm2':float(weights[values>.002].sum()),'area_gt_5mm_mm2':float(weights[values>.005].sum())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent);args=ap.parse_args();out=args.out;out.mkdir(exist_ok=True);started=time.perf_counter()
    protocol=out/'protocol.json';ph=sha(protocol);p=json.loads(protocol.read_text());library=Path(p['prior_query_library_path']);assert sha(library)==p['prior_query_library_sha256']
    spec=importlib.util.spec_from_file_location('geometry_library',library);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
    guard_test=[]
    for name in p['guards']:
        try: getattr(mujoco,name)(None,None)
        except RuntimeError:guard_test.append(name)
        else:raise AssertionError('guard failed')
    ATTEMPTS.clear()
    ip=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json';assert sha(ip)==p['inventory_sha256'];inventory=json.loads(ip.read_text())['modern_cases'];inputs={str(ip):sha(ip),str(library):sha(library),str(ROOT/'src/grasp_failure_prediction/evaluation/object_assets.py'):sha(ROOT/'src/grasp_failure_prediction/evaluation/object_assets.py')};records=[]
    for index,pairindex in zip(p['inventory_indices'],p['paired_group_indices'],strict=True):
        case=inventory[index];pair=inventory[pairindex];print(json.dumps({'start':case['object_id']}),flush=True)
        assert case['object_id']==pair['object_id'] and case['geometry_content_hash']==pair['geometry_content_hash'];assert case['scene_mjb']['sha256']==pair['scene_mjb']['sha256']
        for c in (case,pair):
            for r in (c['scene_mjb'],c['frame_files']['scene_state.npz'],c['frame_files']['scene_contract.json']): assert sha(r['path'])==r['sha256'];inputs[r['path']]=r['sha256']
            for file in ('collection_metadata_path',):inputs[c[file]]=sha(c[file])
        xmlpath=Path(case['scene_mjb']['path']).with_suffix('.xml');inputs[str(xmlpath)]=sha(xmlpath);xml=ET.fromstring(xmlpath.read_bytes());gxml=xml.find("worldbody/body[@name='object']");assert gxml is not None
        meshxml={m.get('name'):m for m in xml.find('asset')};sourcepath=Path(meshxml[gxml.find('geom').get('mesh')].get('file')).parent/'object.xml';properties=sourcepath.parent/'properties.json';inputs[str(sourcepath)]=sha(sourcepath);inputs[str(properties)]=sha(properties);props=json.loads(properties.read_text());source=ET.fromstring(sourcepath.read_bytes());sourcegeometry=load_object_geometry(sourcepath,expected_content_hash=case['geometry_content_hash'])
        for rel,h in sourcegeometry.files_sha256.items():inputs[str(sourcepath.parent/rel)]=h.removeprefix('sha256:')
        m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as state:d.qpos[:]=state['qpos']
        d.qvel[:]=0;mujoco.mj_forward(m,d);initialq=d.qpos.copy();objid=lib.ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');Robj=d.xmat[objid].reshape(3,3).copy();tobj=d.xpos[objid].copy();objs=[];visuals=[];correspondences=[];rawobjs=[];visualtri=None;visualv=None;visualf=None
        source_assets=list(source.find('asset'));source_geoms=list(source.find('worldbody/body').findall('geom'));scene_geoms=list(gxml.findall('geom'));assert len(source_geoms)==len(scene_geoms)
        for sg,og in zip(scene_geoms,source_geoms,strict=True):
            gid=lib.ident(m,mujoco.mjtObj.mjOBJ_GEOM,sg.get('name'));mid=int(m.geom_dataid[gid]);mx=meshxml[sg.get('mesh')];rawpath=Path(mx.get('file'));inputs[str(rawpath)]=sha(rawpath);rawv,rawf=read_obj(rawpath);scale=np.asarray(list(map(float,mx.get('scale','1 1 1').split())));assert not mx.get('refpos') and not mx.get('refquat');rawv=rawv*scale
            rawR=xml_rotation(og);rawpos=np.asarray(list(map(float,og.get('pos','0 0 0').split())));rawv=rawv@rawR.T+rawpos
            va=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);compiledv=(m.mesh_vert[va:va+nv].astype(float)@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]-tobj)@Robj
            rd=cKDTree(rawv).query(compiledv)[0];cd=cKDTree(compiledv).query(rawv)[0];ca=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid]);cf=m.mesh_face[ca:ca+nf].astype(int);active=bool(m.geom_contype[gid] or m.geom_conaffinity[gid])
            corr={'geom_id':gid,'geom_name':sg.get('name'),'raw_path':str(rawpath),'raw_sha256':sha(rawpath),'raw_vertex_count':len(rawv),'compiled_vertex_count':len(compiledv),'raw_face_count':len(rawf),'compiled_face_count':nf,'compiled_mesh_vertex_dtype':str(m.mesh_vert.dtype),'declared_mesh_scale':scale.tolist(),'raw_geom_local_position_m':rawpos.tolist(),'raw_geom_local_rotation':rawR.tolist(),'compiled_geom_object_position_m':((d.geom_xpos[gid]-tobj)@Robj).tolist(),'compiled_mesh_pos_m':m.mesh_pos[mid].tolist(),'compiled_mesh_quat_wxyz':m.mesh_quat[mid].tolist(),'raw_to_compiled_nearest_vertex_max_m':float(cd.max()),'compiled_to_raw_nearest_vertex_max_m':float(rd.max()),'raw_bounds_m':[rawv.min(0).tolist(),rawv.max(0).tolist()],'compiled_bounds_m':[compiledv.min(0).tolist(),compiledv.max(0).tolist()],'active':active}
            assert rd.max()<2e-8 and cd.max()<2e-8,'Source/compiled placement discrepancy';correspondences.append(corr)
            if active:
                objs.append(lib.geom_rep(m,d,gid,Robj,tobj));tri,eq,_=lib.hull(rawv);rawobjs.append({'id':gid,'pos':np.zeros(3),'R':np.eye(3),'tri':tri,'tri_object':tri,'equations':eq})
            else:visuals.append(gid);visualv=compiledv;visualf=cf;visualtri=compiledv[cf]
        assert len(visuals)==1;top=topology(visualv,visualf);assert top['closed_consistently_oriented_for_winding'],'Cannot infer visual inside/outside safely'
        contract=json.loads(Path(case['frame_files']['scene_contract.json']['path']).read_text());assert np.max(np.abs(tobj-np.asarray(contract['object_position_m'])))<1e-12;assert np.max(np.abs(Robj-Rotation.from_quat(contract['object_orientation_xyzw']).as_matrix()))<1e-12
        paired=[]
        pm=mujoco.MjModel.from_binary_path(pair['scene_mjb']['path']);pd=mujoco.MjData(pm)
        with np.load(pair['frame_files']['scene_state.npz']['path'],allow_pickle=False) as state:pd.qpos[:]=state['qpos']
        pd.qvel[:]=0;mujoco.mj_forward(pm,pd);pobj=lib.ident(pm,mujoco.mjtObj.mjOBJ_BODY,'object');pR=pd.xmat[pobj].reshape(3,3);pt=pd.xpos[pobj]
        for corr in correspondences:
            gid=lib.ident(pm,mujoco.mjtObj.mjOBJ_GEOM,corr['geom_name']);mid=int(pm.geom_dataid[gid]);va=int(pm.mesh_vertadr[mid]);nv=int(pm.mesh_vertnum[mid]);cv=(pm.mesh_vert[va:va+nv].astype(float)@pd.geom_xmat[gid].reshape(3,3).T+pd.geom_xpos[gid]-pt)@pR
            gid0=corr['geom_id'];mid0=int(m.geom_dataid[gid0]);v0=(m.mesh_vert[int(m.mesh_vertadr[mid0]):int(m.mesh_vertadr[mid0])+int(m.mesh_vertnum[mid0])].astype(float)@d.geom_xmat[gid0].reshape(3,3).T+d.geom_xpos[gid0]-tobj)@Robj
            delta=float(np.max(np.linalg.norm(cv-v0,axis=1)));assert delta<1e-12;paired.append({'geom_name':corr['geom_name'],'compiled_vertices_object_frame_max_difference_m':delta})
        levels=[2,1];res={};plotarrays={}
        for density in levels:
            q,a,n,gids,qmeta=lib.object_queries(objs,density);dist,cp,ix=lib.nearest_parts(q,visualtri,np.zeros(len(visualtri),np.int64),1);dist=dist[:,0];vin,wind=lib.winding_inside(q,visualtri);ambig=(np.abs(np.abs(wind)-.5)<1e-3)
            probes={}
            for eps in p['outward_probe_lengths_m']:
                entered=lib.object_membership(q+eps*n,objs,tol=-p['inside_halfspace_tolerance_m']);probes[str(eps)]={'retained_count_entering_union':int(entered.sum()),'retained_area_entering_union_mm2':float(a[entered].sum()),'inside_visual_area_entering_union_mm2':float(a[entered&vin].sum())}
            vq,va,vn,_=lib.subdivide(visualtri,density/1000);vinside=lib.object_membership(vq,objs);vd=np.full(len(vq),np.inf)
            for g in objs:sd,_,_=lib.geom_query(g,vq);vd=np.minimum(vd,np.abs(sd))
            # Reverse distance is proximity to per-collider surfaces, not exact exterior union distance for interior queries.
            rawq=np.array(q);rawinside=lib.object_membership(rawq,rawobjs);insideweights={'inside_visual_area_mm2':float(a[vin].sum()),'outside_visual_area_mm2':float(a[~vin].sum()),'ambiguous_winding_count':int(ambig.sum()),'winding_range':[float(wind.min()),float(wind.max())],'collision_query_inside_raw_source_union_count':int(rawinside.sum()),'collision_query_inside_raw_source_union_area_mm2':float(a[rawinside].sum())}
            witnesses=[]
            for k in np.argsort(-dist,kind='stable')[:p['witness_count_per_object']]:
                entry,eid=nearest_ray_entry(objs,q[k],n[k],int(gids[k]));rentry,reid=nearest_ray_entry(rawobjs,q[k],n[k],int(gids[k]));w={'query_index':int(k),'source_geom_id':int(gids[k]),'position_object_m':q[k].tolist(),'outward_normal':n[k].tolist(),'query_area_mm2':float(a[k]),'distance_to_visual_mm':float(dist[k]*1000),'nearest_visual_position_object_m':cp[k].tolist(),'nearest_visual_triangle_index':int(ix[k]),'inside_visual_winding':bool(vin[k]),'visual_winding_value':float(wind[k]),'first_other_compiled_hull_ray_entry_m':entry,'first_other_compiled_hull_id':eid,'first_other_raw_hull_ray_entry_m':rentry,'first_other_raw_hull_id':reid};witnesses.append(w)
            res[str(density)]={'query_domain':qmeta,'collision_to_visual_distance':summary(dist,a),'collision_query_winding_characterization':insideweights,'outward_probe_sensitivity':probes,'visual_to_per_collider_surface_distance':summary(vd,va),'visual_query_inside_collision_union_area_mm2':float(va[vinside].sum()),'visual_query_area_mm2':float(va.sum()),'visual_query_count':len(vq),'worst_distance_witnesses':witnesses}
            if density==2:plotarrays={'q':q,'a':a,'dist':dist,'inside':vin,'n':n,'gids':gids,'visual_tri':visualtri,'collision_tri':np.concatenate([g['tri_object'] for g in objs])}
            print(json.dumps({'object':case['object_id'],'density_mm':density,'collision_mean_mm':res[str(density)]['collision_to_visual_distance']['area_weighted_mean_mm'],'inside_visual_area_mm2':insideweights['inside_visual_area_mm2']}),flush=True)
        assert d.time==0 and np.array_equal(d.qpos,initialq);assert np.array_equal(d.xpos[objid],tobj) and np.array_equal(d.xmat[objid].reshape(3,3),Robj);assert pd.time==0
        # Arrays are local scratch for plots, not evaluation episodes or publication inputs.
        np.savez_compressed(out/(case['object_id']+'_plot_scratch.npz'),**plotarrays)
        records.append({'object_id':case['object_id'],'source_inventory_index':index,'paired_group_index':pairindex,'geometry_content_hash':case['geometry_content_hash'],'source_object_xml_path':str(sourcepath),'source_build_properties_path':str(properties),'declared_build_params':props['build_params'],'source_mjb_path':case['scene_mjb']['path'],'source_mjb_sha256':case['scene_mjb']['sha256'],'source_object_xml_sha256':sha(sourcepath),'compiled_vs_original':correspondences,'visual_topology':top,'both_observed_groups_same_compiled_geometry_in_object_frame':paired,'resolutions':res,'static_invariants':{'time_s':float(d.time),'paired_group_time_s':float(pd.time),'all_qpos_unchanged':True,'object_pose_unchanged':True,'physics_steps':0}})
    for path,h in inputs.items():assert sha(path)==h,path
    assert sha(protocol)==ph and sha(library)==p['prior_query_library_sha256'];assert not ATTEMPTS
    report={'schema':p['schema'],'protocol_sha256':ph,'script_sha256':sha(__file__),'query_library_sha256':sha(library),'runtime_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'mujoco':mujoco.__version__},'runtime_seconds':time.perf_counter()-started,'input_sha256_before_and_after':inputs,'physics_steps':0,'integration_guard_self_tests':guard_test,'integration_attempts_in_investigation':len(ATTEMPTS),'new_inference':False,'new_eval_data':False,'production_changes':False,'cases':records,'limitations':p['limitations']};write(out/'geometry_cause_report.json',report);print(json.dumps({'completed':True,'runtime_s':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
