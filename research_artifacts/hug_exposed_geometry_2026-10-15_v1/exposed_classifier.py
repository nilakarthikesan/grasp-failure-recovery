"""Conservative point diagnostics for a union of closed convex colliders.

No CSG surface is constructed. Exact-distance statements are conditional on the
provided component signed-distance/closest-point oracle and floating arithmetic;
these are not interval-certified numerical proofs. Ambiguous coincident surfaces
remain unresolved. The same-part subset includes only boundary points also on the
whole union boundary. Neither positive unsigned distance nor nonnegative component
SDF is by itself a free-space/exposure certificate.
"""
import numpy as np


def classify(geoms, points, expected_parts, query, tolerance_m=1e-8):
    points=np.asarray(points,dtype=float)
    parts=list(expected_parts)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('finite Nx3 points required')
    if len(parts)!=len(points) or not geoms:raise ValueError('matching parts and nonempty solids required')
    if not np.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('positive finite tolerance required')
    ids=[g['id'] for g in geoms]
    if len(set(ids))!=len(ids):raise ValueError('unique geom IDs required')
    N=len(points);G=len(geoms);eps=float(tolerance_m)
    if not N:return {'rows':[],'component_point_queries':0,'candidate_point_queries':0}
    values=[];cps=[];normals=[]
    def checked(g,p):
        s,c,n=query(g,p);s=np.asarray(s,float);c=np.asarray(c,float);n=np.asarray(n,float)
        if s.shape!=(len(p),) or c.shape!=p.shape or n.shape!=p.shape:raise ValueError('oracle shape')
        if not (np.isfinite(s).all() and np.isfinite(c).all() and np.isfinite(n).all()):raise ValueError('nonfinite oracle result')
        return s,c,n
    for g in geoms:
        s,c,n=checked(g,points);values.append(s);cps.append(c);normals.append(n)
    s=np.asarray(values);cp=np.asarray(cps);norm=np.asarray(normals);d=np.abs(s)
    distance_residual=np.abs(np.linalg.norm(cp-points[None,:,:],axis=2)-d)
    nearest_global=np.argmin(d,axis=0)
    nearest_part=[]
    for j,p in enumerate(parts):
        ix=[i for i,g in enumerate(geoms) if g['part']==p]
        if not ix:raise ValueError('expected part unavailable: '+str(p))
        nearest_part.append(ix[int(np.argmin(d[ix,j]))])
    nearest_part=np.asarray(nearest_part)
    owner=np.concatenate((nearest_global,nearest_part));site=np.tile(np.arange(N),2)
    candidates=cp[owner,site]
    at_candidates=np.asarray([checked(g,candidates)[0] for g in geoms])
    rows=[]
    for j in range(N):
        inside=np.flatnonzero(s[:,j]<-eps);outside=bool(np.all(s[:,j]>eps))
        occupancy='STRICT_COMPONENT_INTERIOR' if len(inside) else ('STRICT_UNION_EXTERIOR' if outside else 'BOUNDARY_OR_COINCIDENT_UNRESOLVED')
        def candidate(k):
            i=int(owner[k]);other=np.ones(G,bool);other[i]=False
            minimum=float(np.min(at_candidates[other,k])) if np.any(other) else None
            ownres=float(abs(at_candidates[i,k]));buried=np.flatnonzero(other & (at_candidates[:,k]<-eps))
            uncertain=np.flatnonzero(other & (np.abs(at_candidates[:,k])<=eps))
            if distance_residual[i,j]>eps:status='ORACLE_DISTANCE_INCONSISTENT'
            elif ownres>eps:status='OWN_SURFACE_RESIDUAL_UNRESOLVED'
            elif len(buried):status='STRICTLY_BURIED_IN_OTHER_COMPONENT'
            elif minimum is None or minimum>eps:status='STRICTLY_SEPARATED_EXPOSED_POINT'
            else:status='COINCIDENT_OR_NEAR_BOUNDARY_UNRESOLVED'
            return {'geom_id':ids[i],'part':geoms[i]['part'],'component_index':i,
                'point_object_m':candidates[k].tolist(),'outward_normal':norm[i,j].tolist(),
                'distance_from_query_m':float(np.linalg.norm(candidates[k]-points[j])),
                'own_surface_residual_m':ownres,'sdf_closest_distance_residual_m':float(distance_residual[i,j]),'minimum_other_signed_distance_m':minimum,
                'strict_burying_geom_ids':[ids[z] for z in buried],
                'strict_burying_parts':sorted(set(geoms[z]['part'] for z in buried)),
                'coincident_or_near_geom_ids':[ids[z] for z in uncertain], 'exposure_status':status,
                'normal_uniqueness_verified':False}
        gc=candidate(j);pc=candidate(j+N)
        globalmin=float(d[nearest_global[j],j]);partmin=float(d[nearest_part[j],j])
        contradiction=bool(outside and gc['exposure_status']=='STRICTLY_BURIED_IN_OTHER_COMPONENT')
        oracle_consistent=bool(not contradiction and np.all(distance_residual[:,j]<=eps) and gc['own_surface_residual_m']<=eps and pc['own_surface_residual_m']<=eps)
        exact_global=bool(outside and not contradiction and oracle_consistent)
        whole_lower=globalmin if exact_global else (float(np.max(np.maximum(-s[:,j],0))) if len(inside) else 0.)
        exposed=[c for c in (gc,pc) if c['exposure_status']=='STRICTLY_SEPARATED_EXPOSED_POINT']
        whole_upper=globalmin if exact_global else (min(c['distance_from_query_m'] for c in exposed) if exposed else None)
        part_exposed=[c for c in exposed if c['part']==parts[j]]
        part_upper=min(c['distance_from_query_m'] for c in part_exposed) if part_exposed else None
        part_exact=pc['exposure_status']=='STRICTLY_SEPARATED_EXPOSED_POINT'
        part_exact=bool(part_exact and oracle_consistent)
        if part_exact:part_upper=partmin
        if not oracle_consistent:whole_lower=None;whole_upper=None;part_upper=None
        # Bounds are conditional geometric bounds, not machine-certified interval arithmetic.
        # Retain raw values, do not silently repair a numerical ordering discrepancy.
        row={'query_object_m':points[j].tolist(),'expected_part':parts[j],
            'occupancy_status':occupancy,'strict_containing_geom_ids':[ids[z] for z in inside],
            'strict_containing_parts':sorted(set(geoms[z]['part'] for z in inside)),
            'signed_component_distances_m':s[:,j].tolist(),'component_geom_ids':ids,
            'global_individual_surface_distance_m':globalmin,'part_individual_surface_distance_m':partmin,
            'global_nearest_component_candidate':gc,'part_nearest_component_candidate':pc,
            'whole_union_boundary_lower_bound_m':whole_lower,'whole_union_boundary_upper_bound_m':whole_upper,
            'whole_union_distance_exact_conditional':exact_global,
            'oracle_consistent':oracle_consistent,'maximum_sdf_closest_distance_residual_m':float(np.max(distance_residual[:,j])),
            'part_exposed_subset_lower_bound_m':partmin if oracle_consistent else None,'part_exposed_subset_upper_bound_m':part_upper,
            'part_exposed_subset_distance_exact_conditional':part_exact,
            'exterior_burial_contradiction':contradiction,
            'global_near_tie_geom_ids':[ids[z] for z in np.flatnonzero(d[:,j]<=globalmin+eps)],
            'bound_order_consistent':bool((whole_upper is None or (whole_lower is not None and whole_lower<=whole_upper+eps)) and (part_upper is None or partmin<=part_upper+eps)),
            'normal_uniqueness_verified':False,'interval_arithmetic_certificate':False}
        rows.append(row)
    return {'rows':rows,'component_point_queries':G*N,'candidate_point_queries':G*2*N,
            'numerical_tolerance_m':eps,'ray_queries':0,'new_sampling':False}
