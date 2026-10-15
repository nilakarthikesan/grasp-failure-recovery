"""Independent archive-only recount of adaptive complete-cube supremum bounds.

Never loads mesh-query/mapper/simulator modules. Reconstructs cell partitions,
coordinates, residual intervals, all pruning decisions and final selected q.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, hashlib, json, math
import numpy as np

FRACTIONS=(Fraction(9,23),Fraction(7,19))
ERROR=2e-9
PARTS={'index':'FF','middle':'MF','ring':'RF','little':'LF','thumb':'TH'}
NAMES=('WRJ2','WRJ1','FFJ4','FFJ3','FFJ2','FFJ1','MFJ4','MFJ3','MFJ2','MFJ1','RFJ4','RFJ3','RFJ2','RFJ1','LFJ5','LFJ4','LFJ3','LFJ2','LFJ1','THJ5','THJ4','THJ3','THJ2','THJ1')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def archive(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main(args):
 report=json.loads(args.report.read_text());checks=[];bindings={str(args.report):sha(args.report)}
 def check(label,condition,detail=None):
  checks.append({'check':label,'pass':bool(condition),'detail':detail})
  if not condition:raise ValueError(label+': '+str(detail))
 def close(a,b):return np.allclose(a,b,rtol=0,atol=1e-16)
 inputdir=Path(report['cohort_input']);proposals=archive(inputdir/'poses.npz'); finals=archive(args.report.parent/'final_static_poses.npz')
 bindings[str(inputdir/'poses.npz')]=sha(inputdir/'poses.npz');bindings[str(args.report.parent/'final_static_poses.npz')]=sha(args.report.parent/'final_static_poses.npz')
 details=[]
 for record in report['records']:
  g=record['global_sample_index']
  if record.get('source_domain_violations'):
   check('source_invalid_not_in_final',g not in finals['global_sample_indices'],g);continue
  i=proposals['global_sample_indices'].tolist().index(g);j=finals['global_sample_indices'].tolist().index(g)
  final_expected=proposals['baseline_target_joint_values'][i].copy()
  for part, branch in record['branches'].items():
   ids=[k for k,n in enumerate(NAMES) if n.startswith(PARTS[part])]
   identity=np.array_equal(proposals['target_joint_values'][i,ids],proposals['baseline_target_joint_values'][i,ids])
   check('branch_identity_matches_q',identity==branch['same_target_geometry_as_baseline'],[g,part])
   if identity:
    check('identity_status',branch['decision']=='PASS_BY_TARGET_GEOMETRY_IDENTITY',[g,part]);continue
   lower=np.zeros(2);old_leaves=[];expected_cells=np.array([(f,u,v) for f in range(6) for u in range(50) for v in range(50)],dtype=np.int64);inherited=np.full((len(expected_cells),2),np.inf)
   for level, description in enumerate(branch['levels']):
    path=Path(description['query_archive']);bindings[str(path)]=sha(path);z=archive(path);cells=z['cells'];h=Fraction(1,1000*2**level)
    check('complete_refinement_cells',np.array_equal(cells,expected_cells),[g,part,level])
    check('no_duplicate_query_cells',len(set(map(tuple,cells)))==len(cells),[g,part,level])
    check('inherited_parent_upper',np.array_equal(z['inherited_upper_m'],inherited),[g,part,level])
    query=np.empty((len(cells),3))
    for k,(face,u,v) in enumerate(cells):
     axis=int(face)//2;others=[d for d in range(3) if d!=axis]
     query[k,axis]=float(Fraction(-1 if face%2==0 else 1,40))
     query[k,others[0]]=float(Fraction(-1,40)+(int(u)+FRACTIONS[0])*h)
     query[k,others[1]]=float(Fraction(-1,40)+(int(v)+FRACTIONS[1])*h)
    check('rational_cell_query_locations',close(query,z['queries_object_m']),[g,part,level])
    s,b,c=(z[k] for k in ['source_distance_m','baseline_distance_m','candidate_distance_m'])
    check('finite_nonnegative_distances',all(a.shape==(len(cells),) and np.isfinite(a).all() and (a>=0).all() for a in (s,b,c)),[g,part,level])
    rho=float(h)*math.sqrt(float((1-FRACTIONS[0])**2+(1-FRACTIONS[1])**2))
    residual=np.column_stack((np.abs(s-b),np.abs(s-c)))
    lower=np.maximum(lower,np.maximum(0,residual.max(axis=0)-ERROR))
    upper=np.minimum(inherited,residual+2*rho+ERROR)
    check('per_cell_upper_recount',close(upper,z['cell_upper_m']),[g,part,level])
    check('cover_radius_recount',abs(rho-description['covering_radius_m'])<1e-16,[g,part,level])
    check('global_lower_recount',close(lower,description['sampled_global_lower_m']),[g,part,level])
    global_upper=np.vstack([entry[2] for entry in old_leaves]+[upper]).max(axis=0)
    terminal=global_upper[1]<=lower[0] or lower[1]>global_upper[0] or level==2
    if level==len(branch['levels'])-1:
     if terminal:old_leaves.append((level,cells,upper))
     else:
      refine=(upper[:,0]>lower[0])|(upper[:,1]>lower[1])
      check('early_stop_only_no_refinement_left',not refine.any(),[g,part,level]);old_leaves.append((level,cells,upper))
    else:
     check('no_refinement_after_decisive_bound',not terminal,[g,part,level])
     refine=(upper[:,0]>lower[0])|(upper[:,1]>lower[1])
     if (~refine).any():old_leaves.append((level,cells[~refine],upper[~refine]))
     chosen=cells[refine];bu=upper[refine]
     expected_cells=np.array([(int(f),2*int(u)+du,2*int(v)+dv) for du,dv in [(0,0),(0,1),(1,0),(1,1)] for f,u,v in chosen],dtype=np.int64)
     inherited=np.vstack([bu]*4)
   coverpath=Path(branch['leaf_cover_archive']);bindings[str(coverpath)]=sha(coverpath);cover=archive(coverpath)
   expected={tuple([level,*cell]):bound for level,cells,bounds in old_leaves for cell,bound in zip(cells,bounds,strict=True)}
   check('leaf_keys_unique',len(set(map(tuple,cover['leaf_cells'])))==len(cover['leaf_cells']),[g,part])
   check('leaf_partition_recount',set(map(tuple,cover['leaf_cells']))==set(expected),[g,part])
   coverage=np.zeros((6,200,200),np.int64)
   for cell,bound in zip(cover['leaf_cells'],cover['leaf_upper_m'],strict=True):
    level,face,u,v=map(int,cell);size=2**(2-level);check('leaf_bound_recount',close(bound,expected[tuple(cell)]),[g,part,*cell.tolist()])
    coverage[face,u*size:(u+1)*size,v*size:(v+1)*size]+=1
   check('full_cube_exact_once_coverage',np.all(coverage==1) and np.array_equal(coverage,cover['coverage_counts']),[g,part])
   global_upper=np.array(list(expected.values())).max(axis=0)
   check('cover_global_bounds',close(global_upper,cover['global_upper_m']) and close(lower,cover['global_lower_m']),[g,part])
   check('report_global_bounds',close([branch['baseline']['continuous_max_lower_m'],branch['candidate']['continuous_max_lower_m']],lower) and close([branch['baseline']['continuous_max_upper_m'],branch['candidate']['continuous_max_upper_m']],global_upper),[g,part])
   decision='PASS_NONREGRESSION' if global_upper[1]<=lower[0] else 'FAIL_REGRESSION' if lower[1]>global_upper[0] else 'UNRESOLVED'
   check('independent_geometry_decision',decision==branch['decision'],[g,part])
   strict=(global_upper[1]<lower[0] or branch['independent_max_landmark_change_m']<0 or branch['independent_landmark_sse_change_m2']<0)
   adopt=(decision=='PASS_NONREGRESSION' and branch['independent_max_landmark_change_m']<=0 and branch['independent_landmark_sse_change_m2']<=0 and strict)
   check('independent_adoption_decision',adopt==branch['adopt_candidate'],[g,part])
   if adopt:final_expected[ids]=proposals['target_joint_values'][i,ids]
   details.append({'row':g,'part':part,'decision':decision,'adopted':adopt,'baseline_interval_m':[float(lower[0]),float(global_upper[0])],'candidate_interval_m':[float(lower[1]),float(global_upper[1])],'leaf_cells':len(expected),'queried_cells':sum(x['queried_cell_count'] for x in branch['levels'])})
  check('final_q_exact_branch_selection',np.array_equal(final_expected,finals['target_joint_values'][j]),g)
  check('final_root_unchanged',np.array_equal(finals['object_from_target_root_world'][j],proposals['object_from_target_root_world'][i]),g)
  check('final_raw_unchanged',np.array_equal(finals['source_hand_qpos'][j],proposals['source_hand_qpos'][i]),g)
 for relative,digest in report['output_hashes'].items():check('output_hash',sha(args.report.parent/relative)==digest,relative)
 result={'schema_version':'independent_adaptive_saved_archive_recount_v1','generated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','source_bindings':bindings,'checks':checks,'check_count':len(checks),'nonidentity_branch_results':details,'new_triangle_queries':0,'physics_steps':0,'native_inference_calls':0,'training_calls':0,'claim':'Independent recount of saved conditional bounds, cell coverage, pruning, numerical decisions and final branch selection. No assertion that the assumed 1 nm per-distance arithmetic error is universally proven.'}
 with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'PASS','checks':len(checks),'branches':len(details),'output':str(args.output),'sha256':sha(args.output)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);p.add_argument('--output',required=True,type=Path);main(p.parse_args())
