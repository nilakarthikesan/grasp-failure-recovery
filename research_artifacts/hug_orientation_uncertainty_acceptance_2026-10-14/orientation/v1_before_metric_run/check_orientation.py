"""Frozen existing-HUG object-facing orientation diagnostic; static only."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, pickle, time, warnings
import numpy as np
import mujoco
from numba import njit, prange

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PREV=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces'
COUNTS={'actual':0,'selftest':0}; TESTING=False

def forbidden(*args,**kwargs):
    COUNTS['selftest' if TESTING else 'actual']+=1
    raise RuntimeError('Static HUG orientation audit forbids integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
spec=importlib.util.spec_from_file_location('frozen_surface_library',PREV/'audit_other_surfaces.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)


def read(p):return json.loads(Path(p).read_text())
def weighted_stats(x,w):
    x=np.asarray(x);w=np.asarray(w)
    if not len(x):return None
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.all(w>0)
    order=np.argsort(x);xx=x[order];ww=w[order];c=np.cumsum(ww)
    return {'count':len(x),'area_mm2':float(w.sum()),'min':float(x.min()),'max':float(x.max()),'area_weighted_mean':float(np.average(x,weights=w)),'area_weighted_median':float(xx[np.searchsorted(c,c[-1]*.5)]),'area_weighted_p95':float(xx[np.searchsorted(c,c[-1]*.95)])}
def amount(mask,area):return {'query_count':int(mask.sum()),'area_mm2':float(area[mask].sum())}

@njit(parallel=True)
def triangle_candidates(points,tri,labels,normals,mind,chosenq,chosenn,tol,normtol):
    counts=np.zeros(len(points),np.int64);labelbits=np.zeros(len(points),np.int64);normalamb=np.zeros(len(points),np.bool_);pointamb=np.zeros(len(points),np.bool_)
    for i in prange(len(points)):
        for j in range(len(tri)):
            qx,qy,qz,d2=a.closest_triangle(points[i,0],points[i,1],points[i,2],tri[j,0],tri[j,1],tri[j,2])
            if np.sqrt(d2)<=mind[i]+tol:
                counts[i]+=1;labelbits[i]|=1<<labels[j]
                dn=normals[j]-chosenn[i]
                if np.sqrt(np.dot(dn,dn))>normtol:normalamb[i]=True
                ds=(qx-chosenq[i,0])**2+(qy-chosenq[i,1])**2+(qz-chosenq[i,2])**2
                if np.sqrt(ds)>tol:pointamb[i]=True
    return counts,labelbits,normalamb,pointamb


def primitive_ambiguous(g,points,closest,tol):
    p=(points-g['pos'])@g['R'];q=(closest-g['pos'])@g['R'];size=g['size'];kind=g['kind'];amb=np.zeros(len(p),bool)
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:
        amb=np.linalg.norm(p,axis=1)<=tol
    elif kind==mujoco.mjtGeom.mjGEOM_BOX:
        # Multiple incident faces at the chosen edge/corner; normal cone rather
        # than a unique geometric facet normal. SDF gradient is still recorded.
        on=np.abs(np.abs(q)-size)<=tol;amb=on.sum(1)>1
        delta=np.abs(p)-size;interior=np.all(delta<=0,axis=1)
        ranks=np.sort(delta,axis=1);amb|=interior&(ranks[:,-1]-ranks[:,-2]<=tol)
    elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial=np.linalg.norm(q[:,:2],axis=1);side=np.abs(radial-size[0])<=tol;cap=np.abs(np.abs(q[:,2])-size[1])<=tol
        amb=side&cap
        delta=np.column_stack([np.linalg.norm(p[:,:2],axis=1)-size[0],np.abs(p[:,2])-size[1]])
        interior=np.all(delta<=0,axis=1);amb|=interior&(np.abs(delta[:,0]-delta[:,1])<=tol)
        amb|=side&(np.linalg.norm(p[:,:2],axis=1)<=tol)
    return amb


def visual_mesh(meta,m,d,pos,R):
    vs=meta['disabled_visual_geoms'];assert len(vs)==1
    g=vs[0]['id'];mid=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);fa=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid]);v=m.mesh_vert[va:va+nv].astype(float);f=m.mesh_face[fa:fa+nf].astype(int)
    vo=(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-pos)@R;tri=vo[f];norm,areas=a.tri_info(tri)
    volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6);assert abs(volume)>1e-9
    if volume<0:norm=-norm
    return tri,norm,{'geom_id':g,'triangles':len(tri),'signed_volume_m3':volume,'outward_orientation_global_flip':volume<0,'excluded_degenerate_triangle_count':0,'surface_area_mm2':float(areas.sum()*1e6),'self_intersections_exhaustively_checked':False}


def visual_membership(points,tri,p):
    if not len(points):return np.zeros(0,np.int8),{}
    distance,_,_=a.nearest_parts(points,tri,np.zeros(len(tri),np.int64),1);distance=distance[:,0]
    vertices=tri.reshape(-1,3);within=np.all((points>=vertices.min(0))&(points<=vertices.max(0)),axis=1);inside=np.zeros(len(points),bool);w=np.zeros(len(points));inside[within],w[within]=a.winding_inside(points[within],tri)
    kind=np.where(inside,-1,1).astype(np.int8);kind[distance<=p['object_boundary_tolerance_m']]=0
    return kind,{'point_count':len(points),'inside_count':int((kind<0).sum()),'outside_count':int((kind>0).sum()),'boundary_count':int((kind==0).sum()),'minimum_winding':float(w.min()),'maximum_winding':float(w.max()),'half_winding_warning_count':int((np.abs(np.abs(w)-.5)<1e-3).sum()),'AABB_exact_outside_reject_count':int((~within).sum())}


def inspect_orientation(q,area,on,nd,ninside,npart,tri,labels,nnorm,hand,visualtri,full,p):
    sd=full['current_distance_by_part_m'];sinside=full['shadow_inside'];rows=np.arange(len(q));spart=sd.argmin(1);ndist=nd[rows,npart];sdist=sd[rows,spart]
    sites=(ndist<=.005)|(sdist<=.005);idx=np.flatnonzero(sites);qq=q[idx];nq=np.full_like(q,np.nan);nn=np.full_like(q,np.nan);sq=np.full_like(q,np.nan);sn=np.full_like(q,np.nan)
    dparts,localnq,nix=a.nearest_parts(qq,tri,labels,len(a.PARTS));assert np.max(np.abs(dparts[:,:6]-nd[idx,:6]))<1e-12;assert np.array_equal(labels[nix],npart[idx]);nq[idx]=localnq;nn[idx]=nnorm[nix]
    nc,nb,na,npa=triangle_candidates(qq,tri,labels,nnorm,ndist[idx],localnq,nn[idx],p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance'])
    nlabelamb=(nb&(nb-1))!=0
    nvalid=np.zeros(len(q),bool);nvalid[idx]=~(na|npa|nlabelamb)
    ncounts=np.zeros(len(q),int);ncounts[idx]=nc
    nnormalamb=np.zeros(len(q),bool);nnormalamb[idx]=na
    npointamb=np.zeros(len(q),bool);npointamb[idx]=npa
    nlabelamball=np.zeros(len(q),bool);nlabelamball[idx]=nlabelamb
    chosengeom=full['nearest_geom_by_part'][rows,spart];sq[idx]=full['current_closest_by_part_m'][idx,spart[idx]];sn[idx]=full['current_normal_by_part'][idx,spart[idx]]
    scount=np.zeros(len(q),int);snormalamb=np.zeros(len(q),bool);spointamb=np.zeros(len(q),bool);slabelbits=np.zeros(len(q),np.int64);sprimitive=np.zeros(len(q),bool);smeshfacet=np.zeros(len(q),bool)
    # Evaluate only sites and only bounding-box-near geometric candidates. The
    # shared sampling phase already supplies exact full finite distances.
    for g in hand:
        candidate=sites.copy()
        if g['tri_object'] is not None:
            vertices=g['tri_object'].reshape(-1,3);gap=np.maximum(np.maximum(vertices.min(0)-q,q-vertices.max(0)),0);candidate&=np.linalg.norm(gap,axis=1)<=.005+p['tie_distance_tolerance_m']
        gi=np.flatnonzero(candidate)
        if not len(gi):continue
        signed,c,n=a.geom_query(g,q[gi]);near=np.abs(signed)<=sdist[gi]+p['tie_distance_tolerance_m'];tieidx=gi[near]
        if not len(tieidx):continue
        scount[tieidx]+=1;slabelbits[tieidx]|=1<<a.PARTS.index(g['part']);snormalamb[tieidx]|=np.linalg.norm(n[near]-sn[tieidx],axis=1)>p['normal_vector_difference_tolerance'];spointamb[tieidx]|=np.linalg.norm(c[near]-sq[tieidx],axis=1)>p['tie_distance_tolerance_m']
        if g['tri_object'] is not None and g['kind']==mujoco.mjtGeom.mjGEOM_MESH:
            tn=a.tri_info(g['tri_object'])[0];cc,bb,amb,pa=triangle_candidates(q[tieidx],g['tri_object'],np.zeros(len(g['tri_object']),np.int64),tn,np.abs(signed[near]),c[near],n[near],p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance']);smeshfacet[tieidx]|=amb|pa
        else:sprimitive[tieidx]|=primitive_ambiguous(g,q[tieidx],c[near],p['tie_distance_tolerance_m'])
    slabelamb=(slabelbits&(slabelbits-1))!=0;svalid=sites&~(snormalamb|spointamb|slabelamb|sprimitive|smeshfacet)
    assert np.all(scount[sites]>0) and np.all(ncounts[sites]>0)
    # Buried target closest points are retained as a separate unavailable class.
    sburied=np.zeros(len(q),bool)
    for g in hand:
        gi=idx[chosengeom[idx]!=g['id']]
        if len(gi):ss,_,_=a.geom_query(g,sq[gi]);sburied[gi]|=ss< -p['tie_distance_tolerance_m']
    svalid&=~sburied
    nclass=np.zeros(len(q),np.int8);sclass=np.zeros(len(q),np.int8);nclass[idx],nmem=visual_membership(nq[idx],visualtri,p);sclass[idx],smem=visual_membership(sq[idx],visualtri,p)
    ncomp=np.full(len(q),np.nan);scomp=np.full(len(q),np.nan);angle=np.full(len(q),np.nan);ncomp[idx]=-np.einsum('ij,ij->i',nn[idx],on[idx]);scomp[idx]=-np.einsum('ij,ij->i',sn[idx],on[idx]);angle[idx]=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',nn[idx],sn[idx]),-1,1)))
    def sign_witnesses(eligible):
        eps=p['facing_dot_numerical_deadzone'];positive_native=eligible&(ncomp>eps);negative_native=eligible&(ncomp< -eps);positive_mapped=eligible&(scomp>eps);negative_mapped=eligible&(scomp< -eps)
        flips={'native_facing_mapped_away':positive_native&negative_mapped,'native_away_mapped_facing':negative_native&positive_mapped}
        witnesses={}
        for name,mask in flips.items():
            ii=np.flatnonzero(mask);margin=np.minimum(np.abs(ncomp[ii]),np.abs(scomp[ii]));ii=ii[np.argsort(margin)[::-1]][:p['maximum_witnesses_per_stratum']]
            witnesses[name]=[{'query_index':int(j),'query_object_m':q[j].tolist(),'area_weight_mm2':float(area[j]),'object_outward_normal':on[j].tolist(),'native_part':a.PARTS[npart[j]],'mapped_part':a.PARTS[spart[j]],'native_unsigned_distance_m':float(ndist[j]),'mapped_unsigned_distance_m':float(sdist[j]),'native_closest_point_object_m':nq[j].tolist(),'mapped_closest_point_object_m':sq[j].tolist(),'native_outward_normal':nn[j].tolist(),'mapped_outward_normal':sn[j].tolist(),'native_facing_compatibility':float(ncomp[j]),'mapped_facing_compatibility':float(scomp[j]),'native_and_mapped_query_exterior':True,'both_closest_outside_visual':True,'excluded_tie_buried_degenerate_flags_all_false':True} for j in ii]
        return {'eligible_outside_visual_denominator':amount(eligible,area),'native_or_mapped_dot_in_numerical_deadzone':amount(eligible&((np.abs(ncomp)<=eps)|(np.abs(scomp)<=eps)),area),'flip_denominators':{name:amount(mask,area) for name,mask in flips.items()},'witnesses':witnesses,'interpretation':'Observed pointwise witnesses on fixed encoded surfaces. Area weights are centroid quadrature estimates; no continuous orientation-area certificate.'}
    def orient(sel):
        valid=sel&nvalid&svalid;delta=scomp-ncomp
        strata={'both_closest_outside_visual':(nclass>0)&(sclass>0),'both_closest_inside_visual':(nclass<0)&(sclass<0),'one_inside_one_outside_visual':((nclass<0)&(sclass>0))|((nclass>0)&(sclass<0)),'one_or_both_on_visual_boundary':(nclass==0)|(sclass==0)}
        def measurements(ss):return {'denominator':amount(ss,area),'native_facing_compatibility':weighted_stats(ncomp[ss],area[ss]),'mapped_facing_compatibility':weighted_stats(scomp[ss],area[ss]),'mapped_minus_native_compatibility':weighted_stats(delta[ss],area[ss]),'hand_normal_angle_deg':weighted_stats(angle[ss],area[ss]),'native_nonfacing':amount(ss&(ncomp<=0),area),'mapped_nonfacing':amount(ss&(scomp<=0),area)}
        return {'candidate_denominator':amount(sel,area),'valid_denominator':amount(valid,area),'excluded_union_denominator':amount(sel&~(nvalid&svalid),area),'exclusions_overlap':{'native_competing_normal':amount(sel&nnormalamb,area),'native_multiple_closest_points':amount(sel&npointamb,area),'native_competing_part':amount(sel&nlabelamball,area),'mapped_competing_geom_normal':amount(sel&snormalamb,area),'mapped_multiple_closest_points':amount(sel&spointamb,area),'mapped_competing_part':amount(sel&slabelamb,area),'mapped_mesh_competing_facet':amount(sel&smeshfacet,area),'mapped_analytic_edge_corner_or_axis':amount(sel&sprimitive,area),'mapped_buried_closest_point':amount(sel&sburied,area)},'valid_all':measurements(valid),'valid_visual_closest_point_strata':{name:measurements(valid&ss) for name,ss in strata.items()},'pointwise_sign_check':sign_witnesses(valid&(nclass>0)&(sclass>0))}
    bands={}
    for band in (1,2,5):
        nnear=ndist<=band/1000;snear=sdist<=band/1000;ne=(~ninside)&(ndist>p['query_boundary_tolerance_m'])&nnear;se=(~sinside)&(sdist>p['query_boundary_tolerance_m'])&snear;common=ne&se;same=common&(npart==spart);different=common&(npart!=spart)
        def unilateral(sel,valid,values,part):return {'candidate_denominator':amount(sel,area),'valid_denominator':amount(sel&valid,area),'excluded_denominator':amount(sel&~valid,area),'facing_compatibility':weighted_stats(values[sel&valid],area[sel&valid]),'nonfacing':amount(sel&valid&(values<=0),area),'by_part':{label:{'candidate_denominator':amount(sel&(part==j),area),'valid_denominator':amount(sel&valid&(part==j),area),'facing_compatibility':weighted_stats(values[sel&valid&(part==j)],area[sel&valid&(part==j)])} for j,label in enumerate(a.PARTS)}}
        bands[str(band)]={'whole_unsigned_proximity':a.coverage(nnear,snear,area),'independent_per_part_unsigned_proximity':{part:a.coverage(nd[:,j]<=band/1000,sd[:,j]<=band/1000,area) for j,part in enumerate(a.PARTS)},'native_exterior':unilateral(ne,nvalid,ncomp,npart),'mapped_exterior':unilateral(se,svalid,scomp,spart),'common_same_label':orient(same),'common_different_label':orient(different),'common_same_label_by_part':{part:orient(same&(npart==j)) for j,part in enumerate(a.PARTS)},'native_exterior_lost':amount(ne&~se,area),'mapped_exterior_new':amount(se&~ne,area),'query_inside_native_hand':amount(nnear&ninside,area),'query_inside_mapped_hand_union':amount(snear&sinside,area),'exterior_label_confusion':{part:{other:amount(common&(npart==j)&(spart==k),area) for k,other in enumerate(a.PARTS)} for j,part in enumerate(a.PARTS)}}
    return {'query_count':len(q),'query_area_mm2':float(area.sum()),'normal_evaluated_union_5mm':amount(sites,area),'native_point_tie_sites':amount(sites&(ncounts>1),area),'mapped_multi_geom_tie_sites':amount(sites&(scount>1),area),'native_visual_membership_diagnostic':nmem,'mapped_visual_membership_diagnostic':smem,'bands':bands,'limitations':p['metric_limitations']}


def sanity(p):
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('xyz',[.27,-.43,.61]).as_matrix();t=np.array([.023,-.017,.031]);box=np.array([[x,y,z] for x in (-.01,.01) for y in (-.02,.02) for z in (-.03,.03)]);tri,_,_=a.hull(box);norm=a.tri_info(tri)[0];tri=tri@R.T+t;norm=norm@R.T
    # Points above a rotated top facet, with expected object outward normal +z.
    on=np.array([0,0,1.])@R.T;point=np.array([[.001,.003,.04]])@R.T+t;dist,q,ix=a.nearest_parts(point,tri,np.zeros(len(tri),np.int64),1);assert abs(dist[0,0]-.01)<1e-14;assert np.linalg.norm(norm[ix[0]]-on)<1e-14
    facing=float(-np.dot(-on,on));parallel=float(-np.dot(on,on));orthogonal=float(-np.dot(np.array([1.,0,0])@R.T,on));assert abs(facing-1)<1e-14 and abs(parallel+1)<1e-14 and abs(orthogonal)<1e-14
    diagpoint=np.array([[0,0,.04]])@R.T+t;dd,qq,jj=a.nearest_parts(diagpoint,tri,np.zeros(len(tri),np.int64),1);cc,bb,amb,pa=triangle_candidates(diagpoint,tri,np.zeros(len(tri),np.int64),norm,dd[:,0],qq,norm[jj],p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance']);assert cc[0]>=2 and not amb[0] and not pa[0]
    cornerpoint=np.array([[.02,.03,.04]])@R.T+t;dd,qq,jj=a.nearest_parts(cornerpoint,tri,np.zeros(len(tri),np.int64),1);cc,bb,amb,pa=triangle_candidates(cornerpoint,tri,np.zeros(len(tri),np.int64),norm,dd[:,0],qq,norm[jj],p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance']);assert amb[0]
    classes,_=visual_membership(np.array([[0,0,0],[0,0,.04],[.01,0,0]])@R.T+t,tri,p);assert np.array_equal(classes,[-1,1,0])
    return {'rotated_translated_box_distance_error_m':float(abs(dist[0,0]-.01)),'antiparallel_facing':facing,'parallel_nonfacing':parallel,'orthogonal':orthogonal,'coplanar_triangle_tie_kept':True,'sharp_corner_competing_normals_excluded':True,'visual_inside_outside_boundary_classes':classes.tolist(),'manufactured_static_only':True}


def main():
    global TESTING
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent);ap.add_argument('--sampling-ephemeral',type=Path,default=Path('/private/tmp/hug_sampling_checks_2026-10-14/ephemeral'));args=ap.parse_args();out=args.out;started=time.perf_counter();pp=out/'protocol.json';ph=a.sha(pp);p=read(pp)
    for path,h in p['source_file_sha256'].items():assert a.sha(path)==h,path
    protected=read(ROOT/'research_artifacts/hug_geometry_reference_validation_2026-10-14/protected_input_sha256.json')
    for path,h in protected.items():assert a.sha(path)==h,path
    TESTING=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Integration guard failed')
    TESTING=False;a.write(out/'sanity_report.json',sanity(p))
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';inventory=read(prior/'existing_hug_inventory.json');fk=read(prior/'static_saved_hug_fk.json');fkmap={c['native_sha256']:c for c in fk['cases']};geom=read(ROOT/'research_artifacts/hug_geometry_reference_validation_2026-10-14/geometry/geometry_cause_report.json');gm={c['object_id']:c for c in geom['cases']}
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);groupweights=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()]);records=[]
    for planned in p['cases']:
        case=inventory['modern_cases'][planned['source_inventory_index']];assert case['native']['sha256']==planned['native_sha256'];meta,tri,labels,norm,hand,objs,m,d,initial,objadr,pos,R=a.load_case(case,groupweights,fkmap);nominal=d.qpos.copy();vt,vn,vm=visual_mesh(meta,m,d,pos,R);assert gm[case['object_id']]['visual_topology']['closed_consistently_oriented_for_winding'];caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        record={'source_inventory_index':planned['source_inventory_index'],'object_id':case['object_id'],'source_case':Path(case['native']['path']).parent.name,'native_sha256':case['native']['sha256'],'input_sha256':meta['input_sha256'],'object_geometry':vm,'resolutions':{}}
        for density in planned['densities_mm']:
            cache=Path(p['original_query_cache_root'])/f'{caseid}_density{density}mm_queries.npz';assert a.sha(cache)==p['query_cache_sha256'][str(cache)]
            with np.load(cache,allow_pickle=False) as arr:data={k:arr[k].copy() for k in arr.files}
            q,area,on,depths=a.subdivide(vt,density/1000);assert np.array_equal(q,data['query_object_m']);assert np.array_equal(area,data['area_mm2']);on=np.repeat(vn,4**np.asarray(depths),axis=0);assert np.allclose(on,data['visual_triangle_outward_normal'],rtol=0,atol=1e-14)
            shared=args.sampling_ephemeral/f'{caseid}_density{density}mm_full_current.npz';binding=shared.with_suffix('.binding.json')
            if not shared.exists():raise FileNotFoundError(f'Sampling sibling exact fields not ready: {shared}')
            b=read(binding);assert b['original_cache_sha256']==a.sha(cache);assert b['ephemeral_npz_sha256']==a.sha(shared)
            with np.load(shared,allow_pickle=False) as arr:full={k:arr[k].copy() for k in arr.files}
            assert np.array_equal(full['query_object_m'],q) and np.array_equal(full['area_mm2'],area)
            # Regress unchanged prior ≤5mm mapped distances and occupancy.
            oldsd=data['shadow_distance_by_part_m'];near=oldsd<=.005;assert np.max(np.abs(full['current_distance_by_part_m'][near]-oldsd[near]))<1e-12;assert np.array_equal(full['shadow_inside'],data['shadow_inside'])
            tick=time.perf_counter();print(json.dumps({'starting':caseid,'density_mm':density,'query_count':len(q)}),flush=True)
            metric=inspect_orientation(q,area,on,data['native_distance_by_part_m'],data['native_inside'],data['native_nearest_part'],tri,labels,norm,hand,vt,full,p);metric['area_density_mm']=density;metric['runtime_seconds']=time.perf_counter()-tick;metric['original_cache_sha256']=a.sha(cache);metric['sampling_shared_binding']=b;record['resolutions'][str(density)]=metric;a.write(out/f'{caseid}_orientation.json',record)
            assert d.time==0 and np.array_equal(d.qpos,nominal) and np.array_equal(d.qpos[objadr:objadr+7],initial)
            print(json.dumps({'complete':caseid,'density_mm':density,'seconds':metric['runtime_seconds'],'same_label_2mm':metric['bands']['2']['common_same_label']['valid_denominator']}),flush=True)
        for path,h in meta['input_sha256'].items():assert a.sha(path)==h
        record['time_s']=float(d.time);record['qpos_and_object_pose_unchanged']=True;records.append(record)
    for path,h in p['source_file_sha256'].items():assert a.sha(path)==h
    for path,h in protected.items():assert a.sha(path)==h
    assert a.sha(pp)==ph and COUNTS['actual']==0
    report={'schema':p['schema'],'protocol_sha256':ph,'script_sha256':a.sha(__file__),'physics_steps':0,'guards':COUNTS,'new_inference':False,'new_eval_data':False,'retarget_optimizer':False,'production_changes':False,'protected_hashes_unchanged':True,'runtime_seconds':time.perf_counter()-started,'runtime_versions':{'python':a.sys.version.split()[0],'numpy':np.__version__,'mujoco':a.mujoco.__version__,'numba':a.numba.__version__},'cases':records,'limitations':p['metric_limitations']};a.write(out/'orientation_report.json',report);print(json.dumps({'completed':True,'seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
