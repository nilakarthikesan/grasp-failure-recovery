"""Native-relative actual surface-direction features on fixed object sites.

Uses the previous frozen geometry/normal eligibility definitions. No fake
normals from fingertip vectors, no positive-contact target, no changing query
stratum after fitting. Scores at unavailable candidate sites are diagnostic;
the solver must use the prescribed nonzero invalid-site residual.
"""
from pathlib import Path
import importlib.util
import numpy as np
import mujoco
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SOURCE=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/check_orientation.py'
sp=importlib.util.spec_from_file_location('frozen_surface_orientation',SOURCE);o=importlib.util.module_from_spec(sp);sp.loader.exec_module(o);a=o.a

def _amount(mask,areas):
    return {'query_count':int(mask.sum()),'area_mm2':float(areas[mask].sum())}

def _normal_invalid(n):
    length=np.linalg.norm(n,axis=1)
    return ~np.isfinite(n).all(1)|(length<=1e-15)|(np.abs(length-1)>1e-6)

def native_direction_fields(points,object_normals,triangles,labels,normal,visual_triangles,p):
    """Exact nearest native facet, whole-hand exterior and old tie safeguards."""
    q=np.asarray(points,float);on=np.asarray(object_normals,float);assert q.shape==on.shape and q.ndim==2 and q.shape[1]==3
    bypart,cp,ix=a.nearest_parts(q,triangles,labels,len(a.PARTS));part=labels[ix];dist=bypart[np.arange(len(q)),part];n=normal[ix]
    count,bits,amb,pa=o.triangle_candidates(q,triangles,labels,normal,dist,cp,n,p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance']);inside,winding=a.winding_inside(q,triangles);membership,mem=o.visual_membership(cp,visual_triangles,p)
    reasons={'query_inside_hand':inside,'query_on_hand_boundary':dist<=p['query_boundary_tolerance_m'],'outside_5mm_band':dist>.005,'competing_normal':amb,'multiple_closest_points':pa,'competing_part':(bits&(bits-1))!=0,'closest_inside_or_on_visual':membership<=0,'invalid_hand_or_object_normal':_normal_invalid(n)|_normal_invalid(on)}
    valid=~np.logical_or.reduce(list(reasons.values()));score=-np.einsum('ij,ij->i',n,on)
    assert np.isfinite(score).all() and np.isfinite(dist).all() and np.isfinite(cp).all()
    return {'distance_m':dist,'distance_by_part_m':bypart,'part_index':part,'closest_object_m':cp,'outward_normal':n,'score':score,'valid':valid,'reason_masks':reasons,'nearest_triangle_index':ix,'tie_count':count,'visual_closest_membership':membership,'visual_membership_diagnostic':mem,'whole_hand_winding':winding}


BURIAL_AUDIT = {'mesh_checks':0, 'mesh_points':0, 'mesh_exterior_skipped':0,
                'mesh_interior_exact_distance_points':0, 'primitive_checks':0,
                'primitive_points':0}

def reset_burial_audit():
    for key in BURIAL_AUDIT: BURIAL_AUDIT[key]=0

def exact_buried_membership(g, points, tolerance):
    """Literal old negative-depth predicate, with exterior mesh work omitted.

    Compute the COMPLETE old local-coordinate/halfspace matrix before selecting
    points. For old inside=False, signed distance is nonnegative and cannot be
    below -tolerance. For old inside=True, retain the same exact triangle kernel,
    coordinate values and triangle order; -distance < -tol iff distance > tol.
    No AABB rejection or halfspace-distance approximation is performed.
    Analytic primitive checks remain the original geom_query verbatim.
    """
    if g['kind']!=mujoco.mjtGeom.mjGEOM_MESH:
        BURIAL_AUDIT['primitive_checks']+=1;BURIAL_AUDIT['primitive_points']+=len(points)
        signed,_,_=a.geom_query(g,points)
        return signed < -tolerance
    local=(points-g['pos'])@g['R']
    inside=np.max(local@g['equations'][:,:3].T+g['equations'][:,3],axis=1)<=1e-10
    result=np.zeros(len(points),bool)
    BURIAL_AUDIT['mesh_checks']+=1;BURIAL_AUDIT['mesh_points']+=len(points)
    BURIAL_AUDIT['mesh_exterior_skipped']+=int((~inside).sum())
    BURIAL_AUDIT['mesh_interior_exact_distance_points']+=int(inside.sum())
    if inside.any():
        distance,_,_=a.nearest_parts(local[inside],g['tri'],np.zeros(len(g['tri']),np.int64),1)
        result[inside]=distance[:,0]>tolerance
    return result

def mapped_direction_fields(points,object_normals,hand_geoms,visual_triangles,p,expected_part=None):
    """Actual nearest compiled collider normal plus full prior eligibility.

    Geometry dictionaries may cache immutable LOCAL triangles, with dynamic
    pos/R supplied after FK. World tri_object is deliberately never read, so a
    stale optimizer-only world cache cannot affect normals/ties or membership.
    Duplicate coordinates are computed once then expanded, preserving every
    fixed-bin residual and its original object normal/expected part.
    """
    qq=np.asarray(points,float);on=np.asarray(object_normals,float);assert qq.shape==on.shape and qq.ndim==2 and qq.shape[1]==3
    q,expand=np.unique(qq,axis=0,return_inverse=True);N=len(q);P=len(a.PARTS);sd=np.full((N,P),np.inf);closest=np.full((N,P,3),np.nan);normal=np.full_like(closest,np.nan);geom=np.full((N,P),-1,np.int64);inside=np.zeros(N,bool);computed=[]
    for g in hand_geoms:
        signed,cp,n=a.geom_query(g,q);computed.append((g,signed,cp,n));inside|=signed<0;pi=a.PARTS.index(g['part']);dist=np.abs(signed);upd=dist<sd[:,pi];sd[upd,pi]=dist[upd];closest[upd,pi]=cp[upd];normal[upd,pi]=n[upd];geom[upd,pi]=g['id']
    if not hand_geoms:raise ValueError('No mapped geometry is available: refuse evaluation rather than invent a normal')
    rows=np.arange(N);part=sd.argmin(1);dist=sd[rows,part];cp=closest[rows,part];n=normal[rows,part];chosen=geom[rows,part];count=np.zeros(N,np.int64);bits=np.zeros(N,np.int64);amb=np.zeros(N,bool);pa=np.zeros(N,bool);facet=np.zeros(N,bool);primitive=np.zeros(N,bool);buried=np.zeros(N,bool)
    assert np.isfinite(dist).all() and np.isfinite(cp).all() and np.isfinite(n).all()
    for g,signed,gcp,gn in computed:
        near=np.abs(signed)<=dist+p['tie_distance_tolerance_m'];ii=np.flatnonzero(near)
        if len(ii):
            count[ii]+=1;bits[ii]|=1<<a.PARTS.index(g['part']);amb[ii]|=np.linalg.norm(gn[ii]-n[ii],axis=1)>p['normal_vector_difference_tolerance'];pa[ii]|=np.linalg.norm(gcp[ii]-cp[ii],axis=1)>p['tie_distance_tolerance_m']
            if g['kind']==mujoco.mjtGeom.mjGEOM_MESH:
                # R is a proper collider pose, so local Euclidean tie distances
                # and normal differences equal their world definitions.
                localq=(q[ii]-g['pos'])@g['R'];localcp=(gcp[ii]-g['pos'])@g['R'];localn=gn[ii]@g['R'];tn=a.tri_info(g['tri'])[0]
                cc,bb,fa,pm=o.triangle_candidates(localq,g['tri'],np.zeros(len(g['tri']),np.int64),tn,np.abs(signed[ii]),localcp,localn,p['tie_distance_tolerance_m'],p['normal_vector_difference_tolerance']);facet[ii]|=fa|pm
            else:primitive[ii]|=o.primitive_ambiguous(g,q[ii],gcp[ii],p['tie_distance_tolerance_m'])
        j=np.flatnonzero(chosen!=g['id'])
        if len(j):buried[j]|=exact_buried_membership(g,cp[j],p['tie_distance_tolerance_m'])
    assert np.all(count>0)
    membership,mem=o.visual_membership(cp,visual_triangles,p)
    unexpanded={'query_inside_hand':inside,'query_on_hand_boundary':dist<=p['query_boundary_tolerance_m'],'outside_5mm_band':dist>.005,'competing_normal':amb,'multiple_closest_points':pa,'competing_part':(bits&(bits-1))!=0,'mapped_mesh_competing_facet':facet,'mapped_analytic_edge_corner_axis':primitive,'buried_closest_point':buried,'closest_inside_or_on_visual':membership<=0}
    reasons={key:value[expand] for key,value in unexpanded.items()};reasons['invalid_hand_or_object_normal']=_normal_invalid(n[expand])|_normal_invalid(on)
    if expected_part is not None:
        ref=np.broadcast_to(np.asarray(expected_part,np.int64),(len(qq),));reasons['different_nearest_part']=part[expand]!=ref
        if np.any(ref<0) or np.any(ref>=P):raise ValueError('Expected native anatomical part index is outside declared proxy parts')
        represented=np.isfinite(sd).any(0);reasons['expected_part_geometry_unavailable']=~represented[ref]
    valid=~np.logical_or.reduce(list(reasons.values()));score=-np.einsum('ij,ij->i',n[expand],on);assert np.isfinite(score).all()
    return {'distance_m':dist[expand],'distance_by_part_m':sd[expand],'part_index':part[expand],'closest_object_m':cp[expand],'outward_normal':n[expand],'score':score,'valid':valid,'reason_masks':reasons,'nearest_geom_id':chosen[expand],'tie_count':count[expand],'visual_closest_membership':membership[expand],'visual_membership_diagnostic':mem,'unique_coordinate_query_count':N,'bin_query_count':len(qq)}

def freeze_direction_sites(query,area,object_normals,native_triangles,native_labels,native_normals,baseline_hand_geoms,visual_triangles,p,native_distance_cache,native_inside_cache,native_nearest_part_cache,baseline_full_fields,bins_per_part=64):
    """Freeze exact original shared-valid stratum and its positive quadrature.

    Cached original distances/membership only prefilter eligibility candidates;
    actual native and baseline features are freshly recomputed and regressed.
    The 64 equal-area quantile bins per available part carry mass1/6. Empty
    parts have no fit bins and are explicitly unavailable, without redistributing
    their missing mass into the others. This does not certify full orientation.
    """
    q=np.asarray(query,float);area=np.asarray(area,float);on=np.asarray(object_normals,float);nd=np.asarray(native_distance_cache,float);sd=np.asarray(baseline_full_fields['current_distance_by_part_m'],float);np_=np.asarray(native_nearest_part_cache,np.int64);sp=sd.argmin(1);rows=np.arange(len(q));nmin=nd[rows,np_];smin=sd[rows,sp]
    pre=(nmin<=.005)&(smin<=.005)&(~native_inside_cache)&(~baseline_full_fields['shadow_inside'])&(nmin>p['query_boundary_tolerance_m'])&(smin>p['query_boundary_tolerance_m'])&(np_==sp);idx=np.flatnonzero(pre)
    native=native_direction_fields(q[idx],on[idx],native_triangles,native_labels,native_normals,visual_triangles,p);baseline=mapped_direction_fields(q[idx],on[idx],baseline_hand_geoms,visual_triangles,p,expected_part=native['part_index']);assert np.max(np.abs(native['distance_m']-nmin[idx]),initial=0)<1e-12;assert np.max(np.abs(baseline['distance_m']-smin[idx]),initial=0)<1e-12;assert np.array_equal(native['part_index'],np_[idx]);valid=native['valid']&baseline['valid'];out={}
    for pi,part in enumerate(a.PARTS[:6]):
        selected=idx[valid&(native['part_index']==pi)];total=float(area[selected].sum());source_rows=np.flatnonzero(valid&(native['part_index']==pi))
        if not len(selected):
            out[part]={'available':False,'original_eligible_query_count':0,'original_eligible_area_mm2':0.,'query_indices':np.empty(0,np.int64),'weights':np.empty(0),'native_score':np.empty(0),'baseline_score':np.empty(0),'reason':'Original strict paired direction stratum has no eligible sites; no fitting mass or preservation PASS.'};continue
        quantiles=(np.arange(bins_per_part)+.5)*total/bins_per_part;positions=np.searchsorted(np.cumsum(area[selected]),quantiles);bins=selected[np.minimum(positions,len(selected)-1)];brows=source_rows[np.minimum(positions,len(selected)-1)];weights=np.full(bins_per_part,1/(6*bins_per_part))
        out[part]={'available':True,'original_eligible_query_count':len(selected),'original_eligible_area_mm2':total,'query_indices':bins,'query_area_mm2':area[bins],'weights':weights,'native_score':native['score'][brows],'baseline_score':baseline['score'][brows],'native_outward_normal':native['outward_normal'][brows],'baseline_outward_normal':baseline['outward_normal'][brows],'native_closest_object_m':native['closest_object_m'][brows],'baseline_closest_object_m':baseline['closest_object_m'][brows],'source_candidate_indices':selected,'unique_selected_query_count':len(np.unique(bins)),'native_away_bin_count':int((native['score'][brows]< -p['facing_dot_numerical_deadzone']).sum()),'native_facing_bin_count':int((native['score'][brows]>p['facing_dot_numerical_deadzone']).sum()),'native_or_baseline_deadzone_bin_count':int(((np.abs(native['score'][brows])<=p['facing_dot_numerical_deadzone'])|(np.abs(baseline['score'][brows])<=p['facing_dot_numerical_deadzone'])).sum()),'native_valid':native['valid'][brows],'baseline_valid':baseline['valid'][brows]}
        assert np.all(out[part]['native_valid']) and np.all(out[part]['baseline_valid'])
    diagnostic={'full_query_count':len(q),'simple_original_shared_exterior_prefilter':_amount(pre,area),'native_candidate_invalid_reason_denominators':{key:_amount(mask,area[idx]) for key,mask in native['reason_masks'].items()},'baseline_candidate_invalid_reason_denominators':{key:_amount(mask,area[idx]) for key,mask in baseline['reason_masks'].items()},'exact_original_shared_valid':_amount(valid,area[idx]),'native_source_scores_are_targets':True,'native_away_states_preserved':True,'baseline_cached_distance_regression':True,'bins_per_available_part':bins_per_part,'mass_per_available_part':1/6,'missing_part_mass_not_redistributed':True,'normal_cell_or_outcome_proof':False}
    return out,diagnostic

def direction_residual(native_score,candidate_fields,weights,normal_weight=1.,invalid_penalty=2.):
    """Fixed residual length/mass; unavailable candidate evidence stays nonzero."""
    ref=np.asarray(native_score,float);w=np.asarray(weights,float);score=np.asarray(candidate_fields['score'],float);valid=np.asarray(candidate_fields['valid'],bool)
    assert ref.shape==w.shape==score.shape==valid.shape and np.isfinite(ref).all() and np.isfinite(w).all() and np.all(w>0)
    delta=np.where(valid,score-ref,invalid_penalty);r=normal_weight*np.sqrt(w)*delta;assert np.all(r[~valid]!=0);return r
