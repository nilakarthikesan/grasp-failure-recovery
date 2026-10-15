#!/usr/bin/env python3
"""Independent manufactured controls; no saved-case FK or geometry measurement."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
OUT=Path(__file__).parent
PRIMARY=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1')
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
CORE=PRIMARY/'constraint_core.py'
CELL=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/sampling/cell_intervals.py'
ACCEPT=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
c=load('review_core',CORE);old=load('review_cell',CELL)
bindings={str(p):sha(p) for p in [CORE,CELL,ACCEPT]}
checks=[]
def check(name,value):
 assert bool(value),name
 checks.append(name)
def prox(d,near,r=2**-12,eps=2**-24,b=2**-9):
 return c.proximity_constraints(np.array([d]),np.array([r]),np.array([eps]),np.array([b]),np.array([near]))
b=2**-9;r=2**-12;eps=2**-24;err=r+eps
near_d=b-err;far_d=b+err
p=prox(near_d,True)
check('near equality passes literal upper <= band',p['passed'][0])
check('near equality stricter optimizer target is negative',p['solver_margin'][0]<0)
check('one ULP beyond near equality may round onto equality',prox(np.nextafter(near_d,np.inf),True)['passed'][0])
check('two ULPs beyond near equality fails',not prox(np.nextafter(np.nextafter(near_d,np.inf),np.inf),True)['passed'][0])
check('one ULP inside near equality passes',prox(np.nextafter(near_d,-np.inf),True)['passed'][0])
p=prox(far_d,False)
check('far equality fails strict lower > band',not p['passed'][0] and p['strict_far_boundary'][0])
check('one ULP beyond far equality passes',prox(np.nextafter(far_d,np.inf),False)['passed'][0])
check('one ULP below far equality fails',not prox(np.nextafter(far_d,-np.inf),False)['passed'][0])
check('ambiguous mapped interval is not restored native near',not prox(b,True)['passed'][0])
check('ambiguous mapped interval is not restored native far',not prox(b,False)['passed'][0])
# Independent literal operation-order comparison over endpoints and neighbors.
ds=np.array([0.,b,near_d,far_d,*[np.nextafter(v,z) for v in (near_d,far_d) for z in (-np.inf,np.inf)]])
for near in (True,False):
 p=c.proximity_constraints(ds,np.full(ds.shape,r),np.full(ds.shape,eps),np.full(ds.shape,b),np.full(ds.shape,near))
 lo=np.maximum(0.,ds-(r+eps));hi=ds+(r+eps)
 check('literal bounds byte equality for '+str(near),np.array_equal(p['lower_m'],lo) and np.array_equal(p['upper_m'],hi))
 check('literal predicate equality for '+str(near),np.array_equal(p['passed'],hi<=b if near else lo>b))

def ori(score,sign=1.,distance=.0005,band=.001,eligible=True):
 return c.orientation_constraints(np.array([score]),np.array([distance]),np.array([sign]),np.array([band]),1e-6,1e-8,None if eligible is None else np.array([eligible]))
check('native facing preserved',ori(.5)['passed'][0])
check('native away preserved without flipping desired sign',ori(-.5,sign=-1.)['passed'][0])
check('native away becoming facing fails',not ori(.5,sign=-1.)['passed'][0])
check('positive sign deadzone equality fails',not ori(1e-6)['passed'][0])
check('negative sign deadzone equality fails',not ori(-1e-6,sign=-1.)['passed'][0])
check('outside positive deadzone passes',ori(np.nextafter(1e-6,np.inf))['passed'][0])
check('source band equality passes',ori(.5,distance=.001)['passed'][0])
check('retains 1mm source band despite generic 5mm validity',not ori(.5,distance=.002)['passed'][0])
check('query boundary equality fails',not ori(.5,distance=1e-8)['passed'][0])
check('invalidated eligibility never clears a row',not ori(.5,eligible=False)['passed'][0])
check('raw-only orientation has no final feasibility result',ori(.5,eligible=None)['passed'] is None)
# Duplicate constraints leave maximum violation unchanged, not averaged away.
p=c.proximity_constraints(np.array([b,b+.01]),np.array([r,r]),np.array([eps,eps]),np.array([b,b]),np.array([True,False]))
base=float(np.max(np.maximum(0.,-p['solver_margin'])))
rep=np.repeat(p['solver_margin'],[8,2])
check('duplicating constraints preserves epigraph maximum',float(np.max(np.maximum(0.,-rep)))==base)
check('large success cannot cancel separate failure',base>0 and np.min(p['solver_margin'])<0)
# Exact legacy tip arithmetic and mean inequality.
e=np.zeros((5,3));e[:,0]=.010;caps=np.full(5,.010)
t=c.exact_tip_status(e,caps)
check('five original caps and mean exactly 10mm pass',t['passed'] and t['mean_pass'] and np.all(t['individual_pass']))
e_bad=e.copy();e_bad[0,0]=np.nextafter(.010,np.inf)
check('one ULP over individual cap fails without allowance',not c.exact_tip_status(e_bad,caps)['passed'])
e_big=np.zeros((5,3));e_big[:,0]=.010001
check('mean above 10mm fails even when all caps permit',not c.exact_tip_status(e_big,np.full(5,.02))['passed'])
# Final rank honors literal feasibility, independent from inward search buffer.
x=np.zeros(27)
rank=c.diagnostic_rank
feasible=rank(np.array([True]),np.array([0.,1e-15]),np.array([True,True]),x)
infeasible=rank(np.array([True]),np.array([0.]),np.array([True,False]),x)
check('literal feasible outranks zero-margin strict failure',feasible<infeasible)
check('invalidation cannot improve diagnostic rank',rank([True],[-1.],[False],x)<rank([False],[100.],[False],x))
check('worst normalized deficit precedes number of failed rows',rank([True],[-.1,-.1],[False,False],x)<rank([True],[-.2,1.],[False,True],x))
check('pose norm only breaks equal scientific rank',rank([True],[1.],[True],x)<rank([True],[1.],[True],np.ones(27)))
check('zero orientation rows explicit numeric handling',rank([],np.array([1.]),np.array([True]),x)[0]==0)
check('inputs unchanged after controls',all(sha(Path(p))==v for p,v in bindings.items()))
result={'schema':'hug-independent-manufactured-constraints-review-v1','passed':True,'check_count':len(checks),'checks':checks,'source_bindings':bindings,'scope':'Manufactured scalar/vector boundary, duplicate epigraph and ranking controls only. No saved-case FK, new geometry, fit, inference, physics, eval episodes, or labels. Does not prove attainable Shadow surface constraints or full mapping validity.','acceptance_sha256':sha(ACCEPT)}
(OUT/'manufactured_constraint_controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'report_sha256':sha(OUT/'manufactured_constraint_controls.json')}))
