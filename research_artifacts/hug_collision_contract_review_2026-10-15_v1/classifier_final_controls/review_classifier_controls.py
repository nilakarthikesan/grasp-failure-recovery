"""Independent analytic-oracle and rigid-covariance controls; no saved HUG queries."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
P=Path('/private/tmp/hug_exposed_geometry_2026-10-15_v1/exposed_classifier.py');O=Path(__file__).parent
spec=importlib.util.spec_from_file_location('reviewed_exposure_classifier',P);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
def check(name,condition):
 if not bool(condition):raise AssertionError(name)
 rows.append({'name':name,'passed':True})
def sphere(i,c,r,part):return {'id':i,'center':np.asarray(c,float),'radius':r,'part':part}
def oracle(g,x):
 z=np.asarray(x)-g['center'];rr=np.sqrt(np.einsum('ij,ij->i',z,z));normal=np.zeros_like(z);nz=rr>0;normal[nz]=z[nz]/rr[nz,None];normal[~nz]=[0,0,1]
 return rr-g['radius'],g['center']+g['radius']*normal,normal
def classify(gs,q,part,fn=oracle):return mod.classify(gs,np.asarray(q,float),part,fn)
gs=[sphere(7,[-.5,0,0],1,'palm'),sphere(9,[.5,0,0],1,'index')]
a=classify(gs,[[0,0,0]],['palm'])['rows'][0]
check('analytic_overlapping_union_lower_bound_not_exact',a['whole_union_boundary_lower_bound_m']==.5 and not a['whole_union_distance_exact_conditional'])
# Independent oracle chooses the radial +Z at centers, and ±X at the overlap query.
check('buried_nearest_does_not_supply_fake_upper',a['whole_union_boundary_upper_bound_m'] is None and a['part_exposed_subset_upper_bound_m'] is None)
q=np.asarray([[-3.,.2,.1],[3.,-.3,.4]]);parts=['palm','index'];a=classify(gs,q,parts)['rows']
ang=.73;R=np.asarray([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]]);shift=np.asarray([4.,-2.,.3]);rot=[sphere(g['id'],R@g['center']+shift,g['radius'],g['part']) for g in gs];b=classify(rot,q@R.T+shift,parts)['rows']
check('rigid_transform_preserves_distances',all(abs(u['whole_union_boundary_lower_bound_m']-v['whole_union_boundary_lower_bound_m'])<1e-12 and abs(u['part_exposed_subset_lower_bound_m']-v['part_exposed_subset_lower_bound_m'])<1e-12 for u,v in zip(a,b)))
check('rigid_transform_preserves_exposure_states',all(u['occupancy_status']==v['occupancy_status'] and u['part_nearest_component_candidate']['exposure_status']==v['part_nearest_component_candidate']['exposure_status'] for u,v in zip(a,b)))
check('rigid_transform_preserves_points',all(np.linalg.norm(R@np.asarray(u['part_nearest_component_candidate']['point_object_m'])+shift-v['part_nearest_component_candidate']['point_object_m'])<1e-12 for u,v in zip(a,b)))
b=classify(gs[::-1],q,parts)['rows'];check('component_order_invariance_unique_nearest',all(u['whole_union_boundary_lower_bound_m']==v['whole_union_boundary_lower_bound_m'] and u['part_nearest_component_candidate']['geom_id']==v['part_nearest_component_candidate']['geom_id'] for u,v in zip(a,b)))
one=[sphere(1,[0,0,0],1,'palm')];near=classify(one,[[1+5e-9,0,0],[1+15e-9,0,0]],['palm','palm'])['rows']
check('numerical_boundary_not_strict_exterior',near[0]['occupancy_status']=='BOUNDARY_OR_COINCIDENT_UNRESOLVED' and near[1]['occupancy_status']=='STRICT_UNION_EXTERIOR')
# Deliberately corrupt the reported distance away from the boundary while
# retaining a genuine closest boundary point. This must not become exact.
def wrong_distance(g,x):
 s,c,n=oracle(g,x);return np.where(np.abs(s)>.01,s*.5,s),c,n
bad=classify(one,[[2,0,0]],['palm'],wrong_distance)['rows'][0]
check('inconsistent_oracle_cannot_claim_exact_global_or_part',not bad['whole_union_distance_exact_conditional'] and not bad['part_exposed_subset_distance_exact_conditional'])
check('inconsistent_oracle_cannot_invent_upper',bad['whole_union_boundary_upper_bound_m'] is None and bad['part_exposed_subset_upper_bound_m'] is None)
# A malformed normal is rejected before any geometric claims.
def malformed(g,x):
 s,c,n=oracle(g,x);n[:]=np.nan;return s,c,n
try:classify(one,[[2,0,0]],['palm'],malformed)
except ValueError:rejected=True
else:rejected=False
check('nonfinite_oracle_fails_closed',rejected)
out={'schema':'independent_exposure_classifier_controls_v1','passed':True,'control_count':len(rows),'controls':rows,'source_bindings':{str(P):hashlib.sha256(P.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'new_saved_grasp_queries':0,'new_eval_data':False,'physics_steps':0,'scope':'Manufactured analytic spheres, independent rigid transform/order checks and deliberately inconsistent-oracle refusal. No sampling of actual grasps.'}
p=O/'classifier_controls_review.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'checks':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
