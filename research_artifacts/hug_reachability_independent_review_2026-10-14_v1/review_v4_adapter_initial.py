from pathlib import Path
from fractions import Fraction
from decimal import Decimal,localcontext
from itertools import product
import hashlib,json,math,xml.etree.ElementTree as ET
import numpy as np
B=Path('/private/tmp/hug_reachability_bound_2026-10-14_v1/full_wrist_v4'); P=B/'protocol.json';R=B/'report.json'
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
p=json.loads(P.read_text());r=json.loads(R.read_text());checks=0
def check(v,msg):
 global checks
 checks+=1
 if not v:raise AssertionError(msg)
check(sha(P)=='df062ec85fb1a7cca99517fb63b36e2245d06520e01da1eeef8a23884141a2ae','protocol SHA')
check(sha(B/'bound_palm_reachability.py')==p['implementation_sha256']=='bf4dabb37bb87d6f7339614da931f63bd9ace43adb49375d9a692163446b960e','script SHA')
check(r['protocol_sha256']==sha(P) and r['implementation_sha256']==p['implementation_sha256'],'report provenance')
for path,h in p['input_sha256'].items():check(sha(path)==h,'input '+path)
source=json.loads(Path(p['witness_file']).read_text())
check(source['acceptance_rule_sha256']=='87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4','frozen acceptance')
for k in ('effective_numeric_allowance_m','extra_transform_arithmetic_allowance_m','rho_rounding_allowance_m'):check(p[k]==1e-7,'positive declared allowance')
check(p['maximum_joint_delta_rad']==.35,'local domain')
inv=json.loads(Path(p['inventory']).read_text())['modern_cases']
oldpath=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/hug_other_object_surfaces.json')
old=json.loads(oldpath.read_text())
phys={j.get('name'):[float(j.find('limit').get(x)) for x in ('lower','upper')] for j in ET.parse(p['urdf']).getroot().findall('joint') if j.find('limit') is not None}
rows=[]
for c in r['cases']:
 idx=c['source_inventory_index'];case=inv[idx];qnames=case['stored_shadow_command']['joint_names'];row=case['stored_shadow_command']['row_index']
 with np.load(case['stored_shadow_command']['path'],allow_pickle=False) as z:q0=z['nominal_target_rad'][row].copy();check(z['joint_names'].tolist()==qnames,'qnames')
 check(len(qnames)==len(set(qnames))==24 and q0.shape==(24,),'24 named joints')
 check(c['native_sha256']==case['native']['sha256'],'native SHA')
 check(sorted(c['palm_ancestor_joints'])==sorted(p['wrist_joint_names']),'wrist-only moving ancestors')
 wi=[qnames.index(n) for n in p['wrist_joint_names']]
 dom=np.asarray(c['wrist_domain_rad']);expected=np.array([phys[qnames[j]] for j in wi])
 check(np.array_equal(dom,expected) and np.all(dom[:,1]>dom[:,0]),'exact full source-URDF wrist domain')
 check(np.array_equal(q0[wi],np.asarray(c['wrist_q0_rad'])),'wrist q0')
 olds=[x for x in old['cases'] if x['native_sha256']==c['native_sha256']];check(len(olds)==1,'old anatomical snapshot')
 oldrec=olds[0];oldpalms=[g for g in oldrec['shadow_geoms'] if g['part']=='palm']
 check(c['palm_geom_ids']==[g['id'] for g in oldpalms],'all and only old frozen palm geom IDs')
 check(oldrec['input_sha256'][case['scene_mjb']['path']]==case['scene_mjb']['sha256'],'same compiled scene as historical membership')
 geoms=c['radius_geometry_records'];check(len(geoms)==len(oldpalms)==8,'8 actual boxes')
 corner_error=0.;rho=c['palm_radius_upper_m'];maxraw=0.
 for g,prior in zip(geoms,oldpalms,strict=True):
  check(g['geom_id']==prior['id'] and g['body']==prior['body']=='palm' and g['kind']==prior['kind']==6,'exact palm/body/box membership')
  check(g['contype']==prior['contype'] and g['conaffinity']==prior['conaffinity'] and bool(g['contype'] or g['conaffinity']),'active masks')
  size=np.asarray(g['size']);loc=np.asarray(g['geom_local_position_m']);quat=np.asarray(g['geom_local_quaternion_wxyz'])
  check(np.all(size>0) and abs(float(quat@quat)-1)<1e-14,'positive box and proper local quaternion')
  w,x,y,z=quat;rot=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
  corners=np.asarray(list(product((-1.,1.),repeat=3)))*size;corners=corners@rot.T+loc
  v=np.asarray(g['palm_frame_unique_vertices_m']);check(v.shape==(8,3),'all8 corners retained')
  distance=np.linalg.norm(v[:,None]-corners[None,:],axis=2);err=float(max(distance.min(0).max(),distance.min(1).max()));corner_error=max(corner_error,err)
  check(err<1e-12,'recorded vertices agree exact analytic box corners')
  norms=np.sqrt(np.einsum('ij,ij->i',v,v));raw=float(norms.max());maxraw=max(maxraw,raw)
  check(abs(raw-g['raw_maximum_vertex_radius_m'])<1e-16,'per-geom raw radius')
  check(abs(np.linalg.norm(g['maximum_vertex_palm_frame_m'])-raw)<1e-16,'maximum radius vertex')
  check(float(np.linalg.norm(corners,axis=1).max())<rho,'padded rho exceeds all exact local box corner norms')
 check(abs(maxraw-c['palm_radius_raw_maximum_m'])<1e-16,'global raw radius')
 check(c['palm_radius_padding_m']==p['rho_rounding_allowance_m'] and c['palm_radius_upper_m']==c['palm_radius_raw_maximum_m']+p['rho_rounding_allowance_m'],'positive rho padding')
 witnesses=c['existing_witnesses'];expectedw=[x for x in source['witnesses'] if x['native_sha256']==c['native_sha256'] and x['anatomical_part']=='palm'];check(witnesses==expectedw,'exact previously published witness list')
 for x in witnesses:
  check(x['relationship_change']=='lost_native' and x['cell_area_mm2']>0 and x['centroid_radius_m']>0,'positive-area existing lost cell')
  check(x['cell_library_sha256']==p['input_sha256'][str(Path(p['witness_file']).parent/'cell_intervals.py')],'cell-bound library SHA')
  check(x['acceptance_rule_sha256']==source['acceptance_rule_sha256'],'unchanged scientific rule')
  eps=max(x['numerical_allowance_m'],p['effective_numeric_allowance_m'])
  check(x['native_centroid_distance_m']+x['centroid_radius_m']+eps<=x['band_mm']/1000,'native definitely near at100nm')
  cp=Path(p['query_cache_root'])/(x['case_id']+'_density'+str(x['density_mm'])+'mm_queries.npz');check(sha(cp)==x['original_query_cache_sha256'],'query cache SHA')
  with np.load(cp,allow_pickle=False) as z:
   i=x['child_cell_index'];check(np.array_equal(z['query_object_m'][i],np.asarray(x['centroid_object_m'])),'centroid exact identity');check(float(z['area_mm2'][i])==x['cell_area_mm2'],'area exact identity');check(float(z['native_distance_by_part_m'][i,0])==x['native_centroid_distance_m'],'native distance exact identity')
 allleaves=c['covered_leaves']+c['unresolved_leaves'];paths=[tuple(x['path']) for x in allleaves];known=set(paths);check(len(known)==len(paths),'unique paths')
 check(sum((Fraction(1,2**len(q)) for q in paths),Fraction())==1,'exact binary mass1')
 internal=set()
 for leaf,path in zip(allleaves,paths,strict=True):
  check(len(path)==leaf['depth']<=p['maximum_subdivision_depth'],'depth')
  check(not any(path[:j] in known for j in range(len(path))),'prefix-free')
  box=dom.copy()
  for j,bit in enumerate(path):
   check(type(bit) is int and bit in (0,1),'binary path bit');internal.add(path[:j]);axis=int(np.argmax(box[:,1]-box[:,0]));mid=box.mean(axis=1)[axis];box[axis,1-bit]=mid
  check(np.array_equal(box,np.asarray(leaf['wrist_box_rad'])),'exact path rectangle')
 for path in internal:check(path+(0,) in internal or path+(0,) in known,'left child complete');check(path+(1,) in internal or path+(1,) in known,'right child complete')
 uneval=sum(bool(x.get('not_evaluated_due_to_budget')) for x in c['unresolved_leaves'])
 check(c['nodes']==len(internal)+len(allleaves)-uneval<=p['maximum_nodes'],'strict node count')
 check(c['geometry_pose_calls']==c['nodes']+1+2*len(p['control_fractions']),'geometry pose count')
 decimal_min=None;max_motion_roundoff=0.
 with localcontext() as ctx:
  ctx.prec=70
  D=Decimal.from_float
  for leaf in c['covered_leaves']:
   check(leaf['witness_index'] in range(len(witnesses)),'existing witness index');x=witnesses[leaf['witness_index']];box=np.asarray(leaf['wrist_box_rad']);half=(box[:,1]-box[:,0])/2;motion=rho*float(half.sum());eps=max(x['numerical_allowance_m'],p['effective_numeric_allowance_m'])
   low=leaf['centroid_distance_at_midpoint_m']-motion-x['centroid_radius_m']-eps-p['extra_transform_arithmetic_allowance_m'];margin=low-x['band_mm']/1000
   check(motion==leaf['palm_surface_motion_bound_m'],'motion arithmetic');check(low==leaf['mapped_all_q_all_cell_distance_lower_m'],'lower-bound arithmetic');check(margin==leaf['strict_margin_m'] and margin>0,'strict positive margin');check(leaf['band_m']==x['band_mm']/1000,'unchanged band')
   real_motion=D(rho)*sum(((D(float(b[1]))-D(float(b[0])))/2 for b in box),Decimal())
   decmargin=D(leaf['centroid_distance_at_midpoint_m'])-real_motion-D(x['centroid_radius_m'])-D(eps)-D(p['extra_transform_arithmetic_allowance_m'])-D(x['band_mm']/1000)
   check(decmargin>0,'70-digit margin remains positive');decimal_min=decmargin if decimal_min is None else min(decimal_min,decmargin);max_motion_roundoff=max(max_motion_roundoff,abs(float(real_motion)-motion))
 check(c['minimum_leaf_strict_margin_m']==min(x['strict_margin_m'] for x in c['covered_leaves']),'minimum margin')
 for leaf in c['unresolved_leaves']:
  if not leaf.get('not_evaluated_due_to_budget'):check(leaf['best_existing_witness_margin_m']<=0,'evaluated unresolved nonpositive')
 full=not c['unresolved_leaves'];check((c['status']=='CURRENT_DOMAIN_STRICT_PALM_PRESERVATION_IMPOSSIBLE_CONDITIONAL')==full,'complete status only')
 check(c['scope_does_not_cover_all_URDF_robot_poses'] and c['finger_constraints_ignored_enlarges_domain_conservatively'],'scoped claim')
 check(c==json.loads((B/(c['case_id']+'_bound.json')).read_text()),'per-case report exact')
 for ctrl in c['controls']:
  check(ctrl['finger_independence_maxabs_m']<1e-14,'reported independence control');check(ctrl['max_corresponded_palm_vertex_movement_m']<=ctrl['analytic_upper_m']+p['control_numeric_allowance_m'],'reported motion control')
 rows.append({'index':idx,'status':c['status'],'nodes':c['nodes'],'certified_leaves':len(c['covered_leaves']),'unresolved_leaves':len(c['unresolved_leaves']),'exact_prefix_partition':True,'minimum_70digit_strict_margin_m':str(decimal_min),'maximum_motion_double_vs70digit_difference_m':max_motion_roundoff,'all8box_corners_radius_checked':True,'maximum_box_local_vertex_difference_m':corner_error})
check(r['physics_steps']==0 and not r['new_eval_data'] and not r['new_inference'] and not r['training'],'static scope')
print(json.dumps({'independent_read_only_review':'PASS','checks':checks,'report_sha256':sha(R),'protocol_sha256':sha(P),'all_input_hashes_verified':len(p['input_sha256']),'historical_membership_report_sha256':sha(oldpath),'cases':rows,'geometry_calls_performed_by_reviewer':0,'new_eval_data':False,'scope':'Conditional full source-URDF wrist domain at fixed root orientation and fixed palm position only; stored closest-surface distances and native cell-radius certificate inherited, no independent geometry recomputation or formally directed interval arithmetic.'},indent=2))
