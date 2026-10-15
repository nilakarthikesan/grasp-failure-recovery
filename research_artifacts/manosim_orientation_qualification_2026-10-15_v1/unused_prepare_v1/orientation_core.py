"""All-grid saved-surface orientation diagnostics; no engine or top-level queries.

Point eligibility retains the old outward-facing definition and tie safeguards.
Cell feature candidates use the 1-Lipschitz DISTANCE bound only: d_face(q) <=
d_part(q)+2*r+2*e. No angular Lipschitz assumption is made. Cell sign evidence
requires a single stable interior facet plus exterior/label clearance; all
other required cells remain unresolved. No full acceptance can be issued here.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

PARTS=('palm','index','middle','ring','little','thumb')
BANDS=np.array([.001,.002,.005])
E=1e-8
NORMAL_TOL=1e-6
DOT_EPS=1e-6
OBJECT_BOUNDARY=1e-7
HALF_WINDING_WARNING=1e-3
CHUNK=128


def primitives(path):
    """Compile only three named pure functions from the immutable old source.

    AST extraction never executes that file's imports, model loader, hull code,
    globals or main. The literal closest-triangle and winding algorithms are
    unchanged. The caller must freeze/hash the entire supplied source first.
    """
    import numba
    wanted=('closest_triangle','nearest_parts','winding_inside')
    tree=ast.parse(Path(path).read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    if [node.name for node in nodes]!=list(wanted):raise ValueError('Exact three original pure geometry functions required')
    allowed={'np','njit','prange','range','len','min','max','abs','closest_triangle'}
    for node in nodes:
        locals_={x.arg for x in node.args.args}|{x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
        unknown={x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}-locals_-allowed
        if unknown or any(isinstance(x,(ast.Import,ast.ImportFrom)) for x in ast.walk(node)):
            raise ValueError('Unexpected pure-function dependency: '+str(unknown))
    namespace=dict(np=np,njit=numba.njit,prange=numba.prange)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    numba.set_num_threads(2)
    return SimpleNamespace(**{name:namespace[name] for name in wanted})


def normal_faces(triangles):
    tri=np.asarray(triangles,float)
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or not np.isfinite(tri).all():raise ValueError('Finite triangles required')
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
    if np.any(length<=1e-15):raise ValueError('Degenerate facet: refuse without dropping it')
    volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]),optimize=False).sum()/6)
    if abs(volume)<=1e-9:raise ValueError('Outward global volume orientation unavailable')
    return cross/length[:,None]*(1 if volume>0 else -1),volume


def edge_clearance(point,tri):
    result=np.inf
    for i in range(3):
        a=tri[i];b=tri[(i+1)%3];edge=b-a
        t=min(max(np.dot(point-a,edge)/np.dot(edge,edge),0.),1.)
        result=min(result,float(np.linalg.norm(point-(a+t*edge))))
    return result


def reduce_tuple(points,radius,tri,labels,normals,object_normals,distances,closest):
    """Manufacturable reference for candidate inventory; actual worker JITs it.

    distances/closest are chunk-by-face arrays computed once. All candidates
    are reduced, never a capped witness list. Deterministic first facet is used
    only to diagnose tie differences, never to override an ambiguous result.
    """
    n=len(points);f=len(tri)
    minimum=np.full((n,6),np.inf);chosen=np.full((n,6),-1,np.int64)
    for i in range(n):
        for j in range(f):
            part=labels[j]
            if distances[i,j]<minimum[i,part]:
                minimum[i,part]=distances[i,j];chosen[i,part]=j
    if not np.isfinite(minimum).all():raise ValueError('Missing anatomical part; no denominator redistribution')
    point_count=np.zeros((n,6),np.int64);cell_count=np.zeros_like(point_count)
    point_normal_amb=np.zeros((n,6),np.bool_);point_position_amb=np.zeros_like(point_normal_amb)
    cell_normal_amb=np.zeros_like(point_normal_amb)
    lo=np.full((n,6),np.inf);hi=np.full((n,6),-np.inf)
    point=np.zeros((n,6,3));normal=np.zeros_like(point);edge=np.zeros((n,6))
    for i in range(n):
        for part in range(6):
            j=chosen[i,part];point[i,part]=closest[i,j];normal[i,part]=normals[j]
            edge[i,part]=edge_clearance(point[i,part],tri[j])
        for j in range(f):
            part=labels[j];d=distances[i,j];reference=chosen[i,part]
            if d<=minimum[i,part]+E:
                point_count[i,part]+=1
                point_normal_amb[i,part]|=np.linalg.norm(normals[j]-normals[reference])>NORMAL_TOL
                point_position_amb[i,part]|=np.linalg.norm(closest[i,j]-closest[i,reference])>E
            if d<=minimum[i,part]+2*radius[i]+2*E:
                cell_count[i,part]+=1
                score=-np.dot(normals[j],object_normals[i]);lo[i,part]=min(lo[i,part],score);hi[i,part]=max(hi[i,part],score)
                cell_normal_amb[i,part]|=np.linalg.norm(normals[j]-normals[reference])>NORMAL_TOL
    return (minimum,chosen,point,normal,point_count,point_normal_amb,point_position_amb,
        cell_count,cell_normal_amb,lo,hi,edge)


REDUCED_KEYS=('distance_by_part','nearest_face','closest_by_part','normal_by_part',
    'point_tie_count','point_competing_normal','point_competing_position','cell_candidate_count',
    'cell_competing_normal','cell_facet_score_min','cell_facet_score_max','nearest_facet_edge_clearance_m')


def reduce_candidates(*args):
    return dict(zip(REDUCED_KEYS,reduce_tuple(*args)))


def triangle_chunk(points,tri,closest_triangle):
    """Called only after a before-work triangle quota charge; never at prepare."""
    distance=np.empty((len(points),len(tri)));closest=np.empty((len(points),len(tri),3))
    for i in range(len(points)):
        for j in range(len(tri)):
            x,y,z,d2=closest_triangle(points[i,0],points[i,1],points[i,2],tri[j,0],tri[j,1],tri[j,2])
            if not np.isfinite(d2) or d2<0:raise ValueError('Invalid literal point-triangle result')
            distance[i,j]=np.sqrt(d2);closest[i,j,0]=x;closest[i,j,1]=y;closest[i,j,2]=z
    return distance,closest


def compile_chunk_helpers(primitive):
    """Compile pure bounded arrays; no actual query is executed by this function."""
    import numba
    edge=numba.njit(edge_clearance)
    # A separate namespace avoids changing module globals used by manufactured controls.
    tree=ast.parse(Path(__file__).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reduce_tuple')
    ns=dict(np=np,E=E,NORMAL_TOL=NORMAL_TOL,edge_clearance=edge)
    exec(compile(ast.Module(body=[node],type_ignores=[]),__file__,'exec'),ns)
    reduce=numba.njit(ns['reduce_tuple'])
    triangle=numba.njit(triangle_chunk)
    return triangle,reduce


def features(query,radius,object_normals,tri,labels,visual_tri,primitive,budget,save_chunk):
    normal,volume=normal_faces(tri);visual_normal,visual_volume=normal_faces(visual_tri)
    if labels.shape!=(len(tri),) or set(np.unique(labels))!=set(range(6)):
        raise ValueError('All six original parts required')
    triangle,reduce=compile_chunk_helpers(primitive);pieces=[]
    for start in range(0,len(query),CHUNK):
        stop=min(start+CHUNK,len(query));q=query[start:stop];r=radius[start:stop];on=object_normals[start:stop]
        budget('hand_triangle_tests',len(q)*len(tri))
        d,cp=triangle(q,tri,primitive.closest_triangle)
        values=dict(zip(REDUCED_KEYS,reduce(q,r,tri,labels,normal,on,d,cp)))
        part=values['distance_by_part'].argmin(axis=1);rows=np.arange(len(q));mind=values['distance_by_part'][rows,part]
        nearest=values['nearest_face'][rows,part];point=values['closest_by_part'][rows,part]
        # Competing labels are checked over the complete global nearest tie set.
        competing=np.sum(values['distance_by_part']<=mind[:,None]+E,axis=1)>1
        budget('hand_winding_triangle_tests',len(q)*len(tri))
        inside,winding=primitive.winding_inside(q,tri)
        budget('object_distance_triangle_tests',len(q)*len(visual_tri))
        vd,_,_=primitive.nearest_parts(point,visual_tri,np.zeros(len(visual_tri),np.int64),1)
        within=np.all((point>=visual_tri.reshape(-1,3).min(axis=0))&(point<=visual_tri.reshape(-1,3).max(axis=0)),axis=1)
        vi=np.zeros(len(q),bool);vw=np.zeros(len(q));idx=np.flatnonzero(within)
        budget('object_winding_triangle_tests',len(idx)*len(visual_tri))
        if len(idx):vi[idx],vw[idx]=primitive.winding_inside(point[idx],visual_tri)
        score=-np.einsum('ij,ij->i',normal[nearest],on,optimize=False)
        invalidnormal=(np.abs(np.linalg.norm(on,axis=1)-1)>NORMAL_TOL)|(np.abs(np.linalg.norm(normal[nearest],axis=1)-1)>NORMAL_TOL)
        reasons=dict(query_inside_hand=inside,query_on_hand_boundary=mind<=E,outside_5mm_band=mind>.005,
            competing_normal=values['point_competing_normal'][rows,part],multiple_closest_points=values['point_competing_position'][rows,part],
            competing_part=competing,closest_inside_or_on_visual=vi|(vd[:,0]<=OBJECT_BOUNDARY),invalid_hand_or_object_normal=invalidnormal,
            half_winding_unreliable=(np.abs(np.abs(winding)-.5)<HALF_WINDING_WARNING)|(np.abs(np.abs(vw)-.5)<HALF_WINDING_WARNING))
        values.update(global_nearest_part=part,global_nearest_face=nearest,global_distance=mind,global_score=score,
            query_inside_hand=inside,query_winding=winding,closest_visual_inside=vi,closest_visual_winding=vw,
            closest_visual_distance_m=vd[:,0],global_point_valid=~np.logical_or.reduce(list(reasons.values())))
        values.update({'reason_'+key:value for key,value in reasons.items()})
        if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite orientation arrays')
        save_chunk(start,stop,values);pieces.append(values);budget('checkpoint',0)
    return {key:np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}


def compare(native,mapped,radius,native_near,native_far,mapped_near):
    n=len(radius);shape=(n,6,3)
    if native_near.shape!=shape or native_far.shape!=shape or mapped_near.shape!=shape:raise ValueError('Original allpart/band masks required')
    possible=~native_far
    part=np.arange(6)[None,:];rows=np.arange(n)
    pointvalid=native['global_point_valid']&mapped['global_point_valid']&(native['global_nearest_part']==mapped['global_nearest_part'])
    pointvalid &= (np.abs(native['global_score'])>DOT_EPS)&(np.abs(mapped['global_score'])>DOT_EPS)
    pointband=(native['global_distance'][:,None]<=BANDS)&(mapped['global_distance'][:,None]<=BANDS)
    pointmask=pointvalid[:,None,None]&(part[:,:,None]==native['global_nearest_part'][:,None,None])&pointband[:,None,:]
    flips=pointmask&((native['global_score']*mapped['global_score'])<0)[:,None,None]
    reasons={};cellvalid=np.ones((n,6),bool)
    for name,feature in (('native',native),('mapped',mapped)):
        selected=feature['global_nearest_part'][:,None]==part
        d=feature['distance_by_part'];other=np.sort(d,axis=1)[:,1]-np.min(d,axis=1)
        reasons[name+'_global_part_unstable']=~selected|(other[:,None]<=2*radius[:,None]+2*E)
        reasons[name+'_facet_unstable']=(feature['cell_candidate_count']!=1)|feature['cell_competing_normal']
        reasons[name+'_facet_edge_possible']=feature['nearest_facet_edge_clearance_m']<=radius[:,None]+E
        reasons[name+'_query_exterior_unproved']=feature['query_inside_hand'][:,None]|(feature['global_distance'][:,None]-radius[:,None]<=E)
        reasons[name+'_closest_exterior_unproved']=feature['closest_visual_inside'][:,None]|(feature['closest_visual_distance_m'][:,None]-radius[:,None]-2*E<=OBJECT_BOUNDARY)
        reasons[name+'_point_eligibility_unavailable']=np.broadcast_to(~feature['global_point_valid'][:,None],(n,6))
        # A facet is constant inside a stable feature region; no angular distance bound.
        dotdead=(feature['cell_facet_score_min']<=DOT_EPS)&(feature['cell_facet_score_max']>=-DOT_EPS)
        reasons[name+'_facing_sign_ambiguous']=dotdead
    for value in reasons.values():cellvalid &= ~value
    stable=possible&native_near&mapped_near&cellvalid[:,:,None]
    different=(native['global_score']*mapped['global_score'])<0
    cellflip=stable&different[:,None,None]
    status=np.zeros(shape,np.int8);status[possible]=3;status[stable]=1;status[cellflip]=2
    return dict(orientation_possible_required=possible,point_eligible=pointmask,point_sign_reversal=flips,
        cell_sign_status=status,cell_stable_matching=stable&~cellflip,cell_sign_reversal=cellflip,
        cell_unresolved=possible&~stable,**{'cell_reason_'+key:value for key,value in reasons.items()})


def summary(arrays,areas):
    result={}
    for pi,part in enumerate(PARTS):
        result[part]={}
        for bi,band in enumerate((1,2,5)):
            row={}
            for key in ('orientation_possible_required','point_eligible','point_sign_reversal','cell_stable_matching','cell_sign_reversal','cell_unresolved'):
                mask=arrays[key][:,pi,bi];row[key]=dict(query_count=int(mask.sum()),centroid_area_mm2=float(areas[mask].sum()))
            result[part][str(band)]=row
    flips=int(arrays['point_sign_reversal'].sum()+arrays['cell_sign_reversal'].sum())
    unresolved=int(arrays['cell_unresolved'].sum())
    return dict(by_part_and_band=result,reliable_point_sign_reversal_observations=int(arrays['point_sign_reversal'].sum()),
        reliable_cell_sign_reversal_observations=int(arrays['cell_sign_reversal'].sum()),unresolved_required_cell_count=unresolved,
        fixed_pose_orientation_status='FAIL' if flips else 'INCOMPLETE',
        strict_mapping_gate='FAIL' if flips else 'INCOMPLETE',full_orientation_continuous_coverage='NOT_PROVED',
        object_facet_boundary_multinormal_coverage='NOT_PROVED',outward_normals_conditional_on_closed_nonselfintersecting_source=True,
        area_weights_are_centroid_quadrature=True,no_finger_motion_or_physics_claim=True,
        native_fidelity_accepted=False,new_collection_or_training_allowed=False)
