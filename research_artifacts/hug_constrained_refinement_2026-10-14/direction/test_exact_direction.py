"""Manufactured exact feature checks; no source proposal or dynamics."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import exact_direction as e
ROOT=e.ROOT;p=json.loads((ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json').read_text())
def hull_box(size):
    v=np.array([[x,y,z] for x in (-size[0],size[0]) for y in (-size[1],size[1]) for z in (-size[2],size[2])]);return e.a.hull(v)[0]
def geom(kind,size,pos=None,R=None,part=None,gid=1):
    pos=np.zeros(3) if pos is None else np.asarray(pos,float);R=np.eye(3) if R is None else np.asarray(R,float);tri=hull_box(np.asarray(size)) if kind in (mujoco.mjtGeom.mjGEOM_BOX,mujoco.mjtGeom.mjGEOM_MESH) else None
    eq=None if tri is None else e.a.hull(tri.reshape(-1,3))[1]
    return {'id':gid,'kind':int(kind),'size':np.asarray(size,float),'pos':pos,'R':R,'tri':tri,'tri_object':None if tri is None else tri@R.T+pos,'equations':eq,'part':e.a.PARTS[0] if part is None else part}
far=hull_box(np.array([.002,.002,.002]))+np.array([10.,10.,10.]);on=np.array([[0.,0.,1.]])
box=geom(mujoco.mjtGeom.mjGEOM_BOX,[.005,.005,.005]);q=np.array([[0.,0.,.006]])
f=e.mapped_direction_fields(q,on,[box],far,p,expected_part=0);assert f['valid'][0] and abs(f['score'][0]+1)<1e-14
ref=np.array([-1.]);r=e.direction_residual(ref,f,np.array([1/6]));assert r[0]==0
corner=e.mapped_direction_fields(np.array([[.006,.006,0.]]),on,[box],far,p,0);assert not corner['valid'][0] and corner['reason_masks']['mapped_analytic_edge_corner_axis'][0];assert e.direction_residual(ref,corner,np.array([1/6]))[0]>0
inside=e.mapped_direction_fields(np.array([[0.,0.,0.]]),on,[box],far,p,0);assert not inside['valid'][0] and inside['reason_masks']['query_inside_hand'][0]
boundary=e.mapped_direction_fields(np.array([[0.,0.,.005]]),on,[box],far,p,0);assert boundary['reason_masks']['query_on_hand_boundary'][0]
outside=e.mapped_direction_fields(np.array([[0.,0.,.020]]),on,[box],far,p,0);assert outside['reason_masks']['outside_5mm_band'][0]
missing=e.mapped_direction_fields(q,on,[box],far,p,expected_part=1);assert not missing['valid'][0] and missing['reason_masks']['expected_part_geometry_unavailable'][0] and missing['reason_masks']['different_nearest_part'][0]
duplicate=box.copy();duplicate['id']=2;dup=e.mapped_direction_fields(q,on,[box,duplicate],far,p,0);assert dup['valid'][0] and dup['tie_count'][0]==2
other=geom(mujoco.mjtGeom.mjGEOM_BOX,[.005,.005,.005],part=e.a.PARTS[1],gid=2);parttie=e.mapped_direction_fields(q,on,[box,other],far,p,0);assert parttie['reason_masks']['competing_part'][0] and not parttie['valid'][0]
left=geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.003,0,0],[-.004,0,0],gid=2);right=geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.003,0,0],[.004,0,0],gid=3);pointtie=e.mapped_direction_fields(np.array([[0.,0.,0.]]),on,[left,right],far,p,0);assert pointtie['reason_masks']['competing_normal'][0] and pointtie['reason_masks']['multiple_closest_points'][0]
sphere=geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.003,0,0]);axis=e.mapped_direction_fields(np.array([[0.,0.,0.]]),on,[sphere],far,p,0);assert axis['reason_masks']['mapped_analytic_edge_corner_axis'][0]
cyl=geom(mujoco.mjtGeom.mjGEOM_CYLINDER,[.003,.004,0]);rim=e.mapped_direction_fields(np.array([[.004,0.,.005]]),on,[cyl],far,p,0);assert rim['reason_masks']['mapped_analytic_edge_corner_axis'][0]
mesh=geom(mujoco.mjtGeom.mjGEOM_MESH,[.005,.005,.005]);coplanar=e.mapped_direction_fields(q,on,[mesh],far,p,0);assert coplanar['valid'][0]
sharp=e.mapped_direction_fields(np.array([[.006,.006,.006]]),on,[mesh],far,p,0);assert sharp['reason_masks']['mapped_mesh_competing_facet'][0]
# Closest point outside own collider but buried in another; query also inside
# union, both invalid reasons retained and not treated as separate samples.
inner=geom(mujoco.mjtGeom.mjGEOM_BOX,[.002,.002,.002],gid=4);outer=geom(mujoco.mjtGeom.mjGEOM_BOX,[.006,.006,.006],gid=5);buried=e.mapped_direction_fields(np.array([[0.,0.,.003]]),on,[inner,outer],far,p,0);assert buried['reason_masks']['query_inside_hand'][0] and buried['reason_masks']['buried_closest_point'][0]
visual=hull_box(np.array([.005,.005,.005]));cp_inside=e.mapped_direction_fields(q,on,[inner],visual,p,0);assert cp_inside['reason_masks']['closest_inside_or_on_visual'][0]
badnormal=e.mapped_direction_fields(q,np.zeros_like(on),[box],far,p,0);assert badnormal['reason_masks']['invalid_hand_or_object_normal'][0]
# Nonidentity transform + intentionally stale world triangles: exact helper
# only consumes local triangles and current R/pos, matching fresh geometry.
A=Rotation.from_euler('xyz',[.31,-.43,.27]).as_matrix();t=np.array([.017,-.03,.021]);rot=mesh.copy();rot['R']=A;rot['pos']=t;fresh=rot.copy();fresh['tri_object']=rot['tri']@A.T+t;qrot=q@A.T+t;onrot=on@A.T
stale=e.mapped_direction_fields(qrot,onrot,[rot],far@A.T+t,p,0);freshfields=e.mapped_direction_fields(qrot,onrot,[fresh],far@A.T+t,p,0);assert np.max(np.abs(stale['score']-freshfields['score']))==0;assert stale['valid'][0] and np.allclose(stale['closest_object_m'],coplanar['closest_object_m']@A.T+t,rtol=0,atol=1e-15)
# A true native away feature outside a nearby visual object: tilted top facet
# projects beyond the visual object's side although q is on its top face.
V=hull_box(np.array([.002,.002,.002]));Q=np.array([[.0019,0.,.002]]);R=Rotation.from_euler('y',-np.pi/4).as_matrix();hn=R@np.array([0.,0.,1.]);cp=Q[0]-.001*hn;center=cp-.005*hn;G=geom(mujoco.mjtGeom.mjGEOM_MESH,[.005,.005,.005],center,R);pieces=[G['tri_object']]+[hull_box(np.array([.005,.005,.005]))+np.array([.1*j,.1,.1]) for j in range(1,6)];nt=np.concatenate(pieces);nn=e.a.tri_info(nt)[0];labels=np.concatenate([np.full(len(x),j,np.int64) for j,x in enumerate(pieces)]);native=e.native_direction_fields(Q,on,nt,labels,nn,V,p);baseline=e.mapped_direction_fields(Q,on,[G],V,p,0);assert native['valid'][0] and baseline['valid'][0];assert native['score'][0]<0 and abs(native['score'][0]+np.sqrt(.5))<1e-12;assert np.allclose(native['score'],baseline['score'],atol=1e-12)
# Original inspect_orientation eligibility on this manufactured true away site
# exactly agrees; matching negative source score does not become +1 target.
nd=native['distance_by_part_m'];full={'current_distance_by_part_m':np.full_like(nd,np.inf),'shadow_inside':np.zeros(1,bool),'current_closest_by_part_m':np.full(nd.shape+(3,),np.nan),'current_normal_by_part':np.full(nd.shape+(3,),np.nan),'nearest_geom_by_part':np.full(nd.shape,-1,np.int64)};full['current_distance_by_part_m'][:,0]=baseline['distance_m'];full['current_closest_by_part_m'][:,0]=baseline['closest_object_m'];full['current_normal_by_part'][:,0]=baseline['outward_normal'];full['nearest_geom_by_part'][:,0]=G['id'];orig=e.o.inspect_orientation(Q,np.array([1.]),on,nd,np.zeros(1,bool),np.zeros(1,np.int64),nt,labels,nn,[G],V,full,p);assert orig['bands']['5']['common_same_label']['pointwise_sign_check']['eligible_outside_visual_denominator']['query_count']==1
bins,diag=e.freeze_direction_sites(Q,np.array([1.]),on,nt,labels,nn,[G],V,p,nd,np.zeros(1,bool),np.zeros(1,np.int64),full,64);assert bins[e.a.PARTS[0]]['available'] and len(bins[e.a.PARTS[0]]['weights'])==64;assert np.isclose(bins[e.a.PARTS[0]]['weights'].sum(),1/6);assert bins[e.a.PARTS[0]]['native_away_bin_count']==64;assert bins[e.a.PARTS[0]]['unique_selected_query_count']==1;assert all(not x['available'] for name,x in bins.items() if name!=e.a.PARTS[0])
expanded=e.mapped_direction_fields(np.repeat(Q,64,axis=0),np.repeat(on,64,axis=0),[G],V,p,0);assert expanded['unique_coordinate_query_count']==1 and expanded['bin_query_count']==64;assert np.max(np.abs(e.direction_residual(bins[e.a.PARTS[0]]['native_score'],expanded,bins[e.a.PARTS[0]]['weights'])))<1e-12
report={'schema':'hug_exact_direction_manufactured_tests_v1','helper_sha256':hashlib.sha256((Path(__file__).parent/'exact_direction.py').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_away_target_score':float(native['score'][0]),'native_away_64_bins_preserved':True,'source_inspect_orientation_eligibility_parity':True,'fixed_missing_part_mass_not_redistributed':True,'duplicate_coordinate_expansion_preserves64bins':True,'nonidentity_stale_vs_fresh_world_cache_parity':True,'invalid_cases':['analytic_edge_corner','hand_inside','hand_boundary','outside_5mm','part_unavailable_mismatch','competing_part','competing_normal_multiple_closest','sphere_axis','cylinder_rim','mesh_sharp_facet_tie','buried_closest','visual_closest_inside','invalid_unit_normal'],'same_normal_point_duplicate_geom_retained':True,'coplanar_mesh_tie_retained':True,'invalid_fixed_sites_nonzero_residual':True,'physics_steps':0,'new_eval_data':False,'normal_unit_sanity_tolerance':1e-6,'normal_unit_sanity_is_API_numerical_check_not_angular_margin':True};(Path(__file__).parent/'manufactured_direction_tests.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
