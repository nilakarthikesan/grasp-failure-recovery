"""Independent V1/V2 equivalence and non-flat-guidance manufactured controls."""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
OUT=Path(__file__).parent;V1=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1');V2=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
a=load('v1core',V1/'constraint_core.py');b=load('v2core',V2/'constraint_core.py');checks=[]
def check(label,v):
 assert bool(v),label;checks.append(label)
def call(core,ds,near,band=.005,radius=.002,eps=1e-8):
 ds=np.asarray(ds,float);return core.proximity_constraints(ds,np.full(ds.shape,radius),np.full(ds.shape,eps),np.full(ds.shape,band),np.full(ds.shape,near))
for band in (.001,.002,.005):
 error=.002+1e-8;cross=band+error
 ds=np.array([0.,error/2,np.nextafter(error,-np.inf),error,np.nextafter(error,np.inf),np.nextafter(cross,-np.inf),cross,np.nextafter(cross,np.inf),.03])
 for near in (False,True):
  x=call(a,ds,near,band);y=call(b,ds,near,band)
  check(f'all final literal fields identical band={band} near={near}',all(np.array_equal(x[k],y[k]) for k in x if k!='solver_margin'))
  check(f'optimizer feasible signs identical band={band} near={near}',np.array_equal(x['solver_margin']>=0,y['solver_margin']>=0))
  if near:check(f'near guidance byte-identical band={band}',np.array_equal(x['solver_margin'],y['solver_margin']))
  else:
   check(f'positive-band strict far equivalence band={band}',np.array_equal(np.maximum(0.,ds-error)>band,ds-error>band))
   check(f'new far guidance is exact unclamped expression band={band}',np.array_equal(y['solver_margin'],((ds-error)-band-1e-9)/.001))
h=1e-6;ds=np.array([.001-h,.001,.001+h]);old=call(a,ds,False)['solver_margin'];new=call(b,ds,False)['solver_margin']
check('old far guidance plateau has zero distance gradient',old[2]-old[0]==0)
check('new far guidance distance gradient 1000 per meter',abs((new[2]-new[0])/(2*h)-1000)<1e-7)
check('new guidance does not make plateau state feasible',np.all(new<0) and np.all(old<0))
for band in (0.,-.001):
 rejected=False
 try:call(b,np.array([.001]),False,band)
 except ValueError:rejected=True
 check('nonpositive sourceband rejected '+str(band),rejected)
for band in (.001,.002,.005):
 error=.002+1e-8;x=call(b,np.array([band+error]),False,band)
 check('strict far equality remains refused '+str(band),not x['passed'][0])
trees=[ast.parse((v/'constraint_core.py').read_text()) for v in (V1,V2)]
funcs=[{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)} for tree in trees]
for name in ('orientation_constraints','exact_tip_status','diagnostic_rank'):
 check('AST unchanged '+name,ast.dump(funcs[0][name])==ast.dump(funcs[1][name]))
for name in ('error','lower','upper','margin_m','passed'):
 def assignment(f):return next(n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 check('literal assignment AST unchanged '+name,ast.dump(assignment(funcs[0]['proximity_constraints']))==ast.dump(assignment(funcs[1]['proximity_constraints'])))
delta=json.loads((V2/'code_delta.json').read_bytes())
for f in delta['files']:
 check('code delta hashes '+f['file'],sha(V1/f['file'])==f['v1_sha256'] and sha(V2/f['file'])==f['v2_sha256'])
 check('only core file differs '+f['file'],f['identical']==(f['file']!='constraint_core.py') and ((V1/f['file']).read_bytes()==(V2/f['file']).read_bytes())==f['identical'])
report={'passed':True,'check_count':len(checks),'checks':checks,'source_bindings':{str(p):sha(p) for p in (V1/'constraint_core.py',V2/'constraint_core.py',V2/'code_delta.json',Path(__file__))},'theorem':'For b>0, max(0,y)>b iff y>b. For y<=0 both fail; for y>0 the expressions are identical. The same reasoning applies to the stricter inward solver target b+1nm. Thus unclamped far guidance removes the zero-gradient clamp without enlarging either final or inward-target feasibility. Original final literal arithmetic remains exactly unchanged.','scope':'Manufactured scalar/AST/hash checks only. This does not predict solver success or certify any source grasp mapping.','new_FK':0,'new_geometry_queries':0,'new_fit':False,'new_eval_data':False}
p=OUT/'far_guidance_controls.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'check_count':len(checks),'report_sha256':sha(p)}))
