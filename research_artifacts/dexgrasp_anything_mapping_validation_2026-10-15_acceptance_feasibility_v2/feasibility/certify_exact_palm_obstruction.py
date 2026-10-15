"""Exact declared-math palm obstruction with rational outward source-pose intervals.
No floating trig, optimization, surface query, model call or physics is used in proof arithmetic.
Saved float values enter as their exact binary rational values; XML decimals enter exactly.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import isqrt,factorial
from pathlib import Path
import hashlib,json,argparse
from datetime import datetime,timezone
import xml.etree.ElementTree as ET

PRECISION=80
SCALE=10**PRECISION
NORMALIZATION_BOUNDS=[]

def lower(x):return F((x.numerator*SCALE)//x.denominator,SCALE)
def upper(x):return -lower(-x)
def exact(x):
 if isinstance(x,F):return x
 if isinstance(x,float):return F.from_float(x)
 return F(x)
class I:
 def __init__(self,a,b=None):
  self.a=lower(exact(a));self.b=upper(exact(a if b is None else b));assert self.a<=self.b
 def __add__(self,other):
  o=ii(other);return I(self.a+o.a,self.b+o.b)
 __radd__=__add__
 def __neg__(self):return I(-self.b,-self.a)
 def __sub__(self,other):return self+-ii(other)
 def __rsub__(self,other):return ii(other)+-self
 def __mul__(self,other):
  o=ii(other);v=[self.a*o.a,self.a*o.b,self.b*o.a,self.b*o.b];return I(min(v),max(v))
 __rmul__=__mul__
 def __truediv__(self,other):
  o=ii(other);assert not o.a<=0<=o.b,(o.a,o.b);return self*I(1/o.b,1/o.a)
 def __rtruediv__(self,other):return ii(other)/self
 def __pow__(self,n):
  assert isinstance(n,int) and n>=0
  z=I(1)
  for _ in range(n):z=z*self
  return z
 def sqrt(self):
  assert self.a>=0
  a=isqrt((self.a.numerator*SCALE*SCALE)//self.a.denominator)
  b=isqrt((self.b.numerator*SCALE*SCALE)//self.b.denominator)
  if F(b,SCALE)**2<self.b:b+=1
  out=I(F(a,SCALE),F(b,SCALE));assert out.a**2<=self.a and out.b**2>=self.b;return out
 def mag(self):return max(abs(self.a),abs(self.b))
 def contains(self,x):return self.a<=exact(x)<=self.b

def ii(x):return x if isinstance(x,I) else I(x)
def sincos(x,terms=50):
 x=ii(x);assert x.mag()<4
 s=sum(((-1)**k*x**(2*k+1)/factorial(2*k+1) for k in range(terms)),I(0))
 c=sum(((-1)**k*x**(2*k)/factorial(2*k) for k in range(terms)),I(0))
 # Taylor's theorem, all real sin/cos derivatives have magnitude <=1.
 sr=x.mag()**(2*terms)/factorial(2*terms);cr=x.mag()**(2*terms-1)/factorial(2*terms-1)
 return s+I(-sr,sr),c+I(-cr,cr)
def dot(a,b):return sum((x*y for x,y in zip(a,b)),I(0))
def vadd(a,b):return [x+y for x,y in zip(a,b)]
def vsub(a,b):return [x-y for x,y in zip(a,b)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def unit(v,role):
 n=dot(v,v).sqrt();assert n.a>F(1,10**8)
 NORMALIZATION_BOUNDS.append({'role':role,'norm_interval':n});return [x/n for x in v]
def matmul(A,B):return [[dot(row,col) for col in zip(*B)] for row in A]
def matvec(A,x):return [dot(row,x) for row in A]
def eye(n):return [[I(int(i==j)) for j in range(n)] for i in range(n)]
def rotation(axis,angle):
 a=unit([ii(x) for x in axis],'urdf_axis');s,c=sincos(angle);K=[[I(0),-a[2],a[1]],[a[2],I(0),-a[0]],[-a[1],a[0],I(0)]]
 return [[c*int(i==j)+(1-c)*a[i]*a[j]+s*K[i][j] for j in range(3)] for i in range(3)]
def xmlvec(node,key,default='0 0 0'):return [F(x) for x in (default if node is None else node.get(key,default)).split()]
def urdf_origin(node):
 xyz=xmlvec(node,'xyz');rpy=xmlvec(node,'rpy');r=eye(3)
 for axis,angle in zip([(0,0,1),(0,1,0),(1,0,0)],reversed(rpy)):r=matmul(r,rotation(axis,angle))
 out=eye(4)
 for i in range(3):
  out[i][:3]=r[i];out[i][3]=I(xyz[i])
 return out

def source_palm_exact(raw,joint_names,source_xml,object_from_input):
 assert object_from_input==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
 assert len(raw)==33
 a,b=unit(list(map(I,raw[3:6])),'rotation6d_first'),unit(list(map(I,raw[6:9])),'rotation6d_second')
 u,v=unit(vadd(a,b),'rotation6d_sum'),unit(vsub(a,b),'rotation6d_difference')
 x,y=unit(vadd(u,v),'rotation6d_final_x'),unit(vsub(u,v),'rotation6d_final_y');z=unit(cross(x,y),'rotation6d_final_z')
 S=eye(4)
 for i in range(3):S[i][:3]=[x[i],y[i],z[i]];S[i][3]=I(raw[i])
 q=dict(zip(joint_names,raw[9:]));nodes={j.find('child').get('link'):j for j in source_xml.findall('joint')};chain=[];current='palm'
 while current!='forearm':
  j=nodes[current];chain.append(j);current=j.find('parent').get('link')
 for j in reversed(chain):
  M=urdf_origin(j.find('origin'))
  if j.get('type')=='revolute':
   R=rotation(xmlvec(j.find('axis'),'xyz','1 0 0'),I(q[j.get('name')]))
   full=eye(4)
   for i in range(3):full[i][:3]=R[i]
   M=matmul(M,full)
  else:assert j.get('type')=='fixed'
  S=matmul(S,M)
 # Interval sanity checks accompany the analytic orthonormality argument.
 R=[row[:3] for row in S[:3]];rt=list(map(list,zip(*R)));rtr=matmul(rt,R)
 assert all(rtr[i][j].contains(int(i==j)) for i in range(3) for j in range(3))
 return S

ANCHORS=('palm','ffknuckle','ffproximal','mfknuckle','mfproximal','rfknuckle','rfproximal','lfmetacarpal','thbase','thproximal')
def anchor_coordinates(xml,sites):
 assert len(sites)==28 and len(set(sites))==28 and set(ANCHORS)<=set(sites)
 joints={j.find('child').get('link'):j for j in xml.findall('joint')};out={}
 for site in ANCHORS:
  if site=='palm':out[site]=(F(0),)*3;continue
  chain=[];cur=site
  while cur!='palm':
   j=joints[cur];assert j.get('type') in ['fixed','revolute'];chain.append(j);cur=j.find('parent').get('link')
  chain.reverse();assert all(xmlvec(j.find('origin'),'xyz')==[0,0,0] for j in chain[1:])
  out[site]=tuple(xmlvec(chain[0].find('origin'),'xyz'))
 return out

def solve(A,b):
 # Independent exact Gauss-Jordan elimination, no float linear algebra.
 m=[list(row)+[rhs] for row,rhs in zip(A,b)];n=len(b)
 for j in range(n):
  k=next(i for i in range(j,n) if m[i][j]);m[j],m[k]=m[k],m[j];pivot=m[j][j]
  m[j]=[x/pivot for x in m[j]]
  for i in range(n):
   if i==j:continue
   pivot=m[i][j];m[i]=[u-pivot*v for u,v in zip(m[i],m[j])]
 return [r[-1] for r in m]
def groups_from(s,t):
 pairs={}
 for name in ANCHORS:pairs.setdefault((s[name],t[name]),[]).append(name)
 return [{'source':s,'target':t,'multiplicity':len(names),'sites':names} for (s,t),names in pairs.items()]
def weights(p,groups):
 rows=[(F(1),)+g['target'] for g in groups];G=[[sum((g['multiplicity']*r[i]*r[j] for g,r in zip(groups,rows)),F(0)) for j in range(4)] for i in range(4)]
 lam=solve(G,[F(1)]+list(p));w=[g['multiplicity']*sum((a*b for a,b in zip(row,lam)),F(0)) for g,row in zip(groups,rows)]
 assert sum(w)==1 and all(sum((v*g['target'][j] for v,g in zip(w,groups)),F(0))==p[j] for j in range(3))
 return w

def sqrt_upper(v):return I(v).sqrt().b
def norm_upper(v):return sqrt_upper(sum((x*x for x in v),F(0)))
def boxes(xml):
 root=xml.find("link[@name='palm']");out=[]
 for i,c in enumerate(root.findall('collision')):
  b=c.find('geometry/box');assert b is not None;out.append({'index':i,'center':tuple(xmlvec(c.find('origin'),'xyz')),'half':tuple(x/2 for x in xmlvec(b,'size')),'axis_aligned':xmlvec(c.find('origin'),'rpy')==[0,0,0]})
 return out

def certificate(x,p,face,S,groups,source_box,target_boxes,M,B,guard_anchor_allowance=F(0)):
 assert face in (0,1,2) and abs(x[face])==F(1,40)
 assert all(abs(v)<=F(1,40) for v in x)
 assert source_box['axis_aligned'] and all(h>0 for h in source_box['half'])
 assert M>=0 and B>=0 and guard_anchor_allowance>=0
 assert all(isinstance(g['multiplicity'],int) and g['multiplicity']>0 and len(g['sites'])==g['multiplicity'] for g in groups)
 assert sum(g['multiplicity'] for g in groups)==10
 assert sorted(n for g in groups for n in g['sites'])==sorted(ANCHORS)
 w=weights(p,groups);K=sum((a*a/g['multiplicity'] for a,g in zip(w,groups)),F(0));l1=sum(abs(a) for a in w)
 offset=tuple(sum((a*g['source'][j] for a,g in zip(w,groups)),F(0))-p[j] for j in range(3));offset_upper=norm_upper(offset)
 maximum_bound=(M+guard_anchor_allowance)*l1+offset_upper
 sse_bound=(sqrt_upper(B)+sqrt_upper(F(10))*guard_anchor_allowance)*sqrt_upper(K)+offset_upper
 motion=min(maximum_bound,sse_bound)
 mapped=matvec(S,[I(v) for v in p]+[I(1)])[:3]
 bridge=norm_upper([((v-I(xx)).mag()) for v,xx in zip(mapped,x)])
 source_clear=max((abs(a-c)-h for a,c,h in zip(p,source_box['center'],source_box['half'])),default=F(0))-bridge
 edge_clear=min(F(1,40)-abs(x[i]) for i in range(3) if i!=face)
 records=[]
 for box in target_boxes:
  if not box['axis_aligned']:continue
  inside=min(h-abs(a-c) for a,c,h in zip(p,box['center'],box['half']))-motion-bridge
  square_side=max(F(0),min(inside,source_clear,edge_clear)/4)
  records.append({'target_box_index':box['index'],'inside_clearance_lower_m':inside,'source_outside_clearance_lower_m':source_clear,'square_side_lower_m':square_side,'square_area_lower_m2':square_side**2})
 best=max(records,key=lambda r:r['square_area_lower_m2'])
 return {'status':'CERTIFIED_EXACT_DECLARED_MATH_OBSTRUCTION' if best['square_area_lower_m2']>0 else 'UNRESOLVED','object_point_exact':x,'reference_local_point_exact':p,'face_axis':face,'maximum_guard_exact_m':M,'SSE_guard_exact_m2':B,'guard_anchor_allowance_m':guard_anchor_allowance,'affine_weights_exact':w,'weighted_cauchy_schwarz_K_exact':K,'weights_l1_exact':l1,'shape_offset_norm_upper_m':offset_upper,'motion_upper_m':motion,'maximum_guard_motion_upper_m':maximum_bound,'SSE_guard_motion_upper_m':sse_bound,'exact_source_point_bridge_upper_m':bridge,'cube_edge_clearance_m':edge_clear,'best':best}

def digest(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(v):return F(int(v['numerator']),int(v['denominator']))
def serial(x):
 if isinstance(x,F):return {'numerator':str(x.numerator),'denominator':str(x.denominator),'approximate_float':float(x)}
 if isinstance(x,I):return {'lower':serial(x.a),'upper':serial(x.b),'width':serial(x.b-x.a)}
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 return x

def main():
 import numpy as np
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
 base=Path('/private/tmp/dga_component_constraints_20261015_v1');old=json.loads((base/'feasibility/palm_anchor_obstruction_policy_frozen_v1.json').read_text());source=Path(old['paths']['source_urdf']);target=Path(old['paths']['target_urdf']);sx,tx=ET.parse(source).getroot(),ET.parse(target).getroot();sites=old['site_names'];groups=groups_from(anchor_coordinates(sx,sites),anchor_coordinates(tx,sites));sb,tb=boxes(sx),boxes(tx);assert len(sb)==1 and sb[0]['axis_aligned'] and len(tb)==8
 oldproofpath=base/'feasibility/results_v1/palm_anchor_obstruction_report.json';oldproof=json.loads(oldproofpath.read_text());oldfitpath=Path(old['paths']['original_guard_report']);oldfit=json.loads(oldfitpath.read_text());newfitpath=base/'native_mapping_results_v1/native_component_fit_report.json';newfit=json.loads(newfitpath.read_text());newpalmpath=base/'native_palm_holdout_results_v1/palm_collider_union_report.json';newpalm=json.loads(newpalmpath.read_text())
 newbaselinepath=base/'native_mapping_results_v1/row_0_baseline.json';newbaseline=json.loads(newbaselinepath.read_text())
 assert newfit['site_names']==sites
 assert digest(source)=='sha256:932fca7c411ef9ae4e6a8f17a892b3629840b5b5e1254610e0616e184d1f95c7'
 assert digest(target)=='sha256:29a50063cced2fba81dfe3d208ba54499fc12635c9189d0822f7caa2771c36de'
 assert digest(oldproofpath)=='sha256:8f4adc6f00abae30693a04fd97b7a903a95684273ab591ed894e5f67f461321f'
 assert digest(newfitpath)=='sha256:3b70dae81992aea7288a22b74cddbdda7d08d4bacfc8d1bebb5f32092920ce83'
 assert digest(newpalmpath)=='sha256:30bc34f60449fe1cf59b5149432f607e021ad56a27e741b0e0404d650ca32e0f'
 assert digest(oldfitpath)==old['dependency_sha256'][str(oldfitpath)]
 deps={str(p):digest(p) for p in [Path(__file__).resolve(),source,target,oldproofpath,oldfitpath,newfitpath,newpalmpath,newbaselinepath,base/'feasibility/palm_anchor_obstruction_policy_frozen_v1.json']}
 cases=[]
 for case_id in ['old_development_row1','new_native_row0_original_guards','new_native_row0_component_caps']:
  isold=case_id.startswith('old');idx=1 if isold else 0;fit=next(r for r in (oldfit if isold else newfit)['records'] if r['global_sample_index']==idx)
  mp=Path(fit['native_manifest']['path']);manifest=json.loads(mp.read_text());rawpath=mp.parent/manifest['output']['path'];deps[str(mp)]=digest(mp);deps[str(rawpath)]=digest(rawpath);assert digest(mp)==fit['native_manifest']['sha256'];assert digest(rawpath)==fit['raw_output_sha256']
  with np.load(rawpath,allow_pickle=False) as z:raw=np.asarray(z['hand_qpos'])[fit['within_batch_sample_index']]
  assert 'sha256:'+hashlib.sha256(raw.tobytes(order='C')).hexdigest()==fit['raw33_row_sha256']
  NORMALIZATION_BOUNDS.clear();S=source_palm_exact([F.from_float(float(v)) for v in raw],manifest['joint_names'],sx,manifest['object_from_input'])
  intervalnorms=list(NORMALIZATION_BOUNDS)
  assert not fit['original_source_limit_violations']
  guards=fit['fixed_palm_engineering_reference'] if isold else (newbaseline['fixed_state']['landmark_metrics'] if case_id.endswith('original_guards') else fit['landmark_limits']);M=F.from_float(guards['maximum_site_error_m']);B=F.from_float(guards['all_site_SSE_m2'])
  if isold:
   oldcase=next(c for c in oldproof['cases'] if c['global_sample_index']==idx);best=oldcase['summaries']['ORIGINAL_FIXED_MAX_SSE']['EXACT_DECLARED_GUARDS']['best_witness'];points=[(tuple(map(rat,best['object_query_m'])),tuple(map(rat,best['palm_reference_query_m'])),best['face_axis'])]
  else:
   pc=next(c for c in newpalm['cases'] if c['global_sample_index']==idx);qr=pc['resolutions']['0.25']['query_arrays'];qp=newpalmpath.parent/qr['path'];deps[str(qp)]=digest(qp);assert digest(qp)==qr['sha256']
   # Float midpoint/inverse only selects a rational candidate; every certificate recomputes the real-pose bridge with exact intervals.
   Sm=np.array([[float((v.a+v.b)/2) for v in row] for row in S]);inv=np.linalg.inv(Sm)
   with np.load(qp,allow_pickle=False) as z:
    ids=np.flatnonzero(z['source__outside_union']&z['target__inside_union']);ids=ids if len(ids)<=256 else ids[np.linspace(0,len(ids)-1,256,dtype=int)];xp=z['cube_queries_m'][ids];fp=z['cube_face_ids'][ids]
   points=[]
   for rawx,f in zip(xp,fp):
    axis=int(f)//2;sign=1 if int(f)%2 else -1;xx=list(map(lambda v:F.from_float(float(v)),rawx));xx[axis]=F(sign,40);pv=inv[:3,:3]@np.asarray(list(map(float,xx)))+inv[:3,3];points.append((tuple(xx),tuple(F.from_float(float(v)) for v in pv),axis))
  certs=[certificate(x,p,f,S,groups,sb[0],tb,M,B) for x,p,f in points];best=max(certs,key=lambda c:c['best']['square_area_lower_m2'])
  # Sensitivity retains the historical allowance while no longer assuming source geometry error.
  robust=certificate(tuple(best['object_point_exact']),tuple(best['reference_local_point_exact']),best['face_axis'],S,groups,sb[0],tb,M,B,F(1,10**9))
  cases.append({'case_id':case_id,'source_raw':{'path':str(rawpath),'sha256':digest(rawpath),'row':fit['within_batch_sample_index']},'guard_scope':'original pose-specific fixed-palm max/SSE guards' if isold or case_id.endswith('original_guards') else 'existing component-corrector tighter max/SSE caps, specific to new row0','status':best['status'],'candidate_count':len(certs),'positive_witness_count':sum(c['status'].startswith('CERTIFIED') for c in certs),'source_transform_exact_interval':S,'normalization_non_degeneracy_bounds':intervalnorms,'best_certificate':best,'one_nm_guard_allowance_sensitivity_certificate':robust,'all_certificates':certs})
 report={'schema_version':'dga_exact_declared_math_palm_obstruction_interval_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'rational_interval_decimal_grid_digits':PRECISION,'transcendental_bounds':'50-term exact rational sin/cos Taylor intervals with Lagrange remainder, outward rounding after every interval operation; square roots bounded using integer isqrt inequalities','source_rotation_semantics':'Exact real symmetric robust6D reconstruction of fixed saved binary33 values, then exact declared source URDF wrist FK; identity object_from_input. Unit normalization is nondegenerate and analytically yields a proper source rotation.','target_quantifier':'All proper target palm placements and all finger angles satisfying the indicated exact mathematical 28-site max/SSE guards. Ten joint-independent anchor origins preserve multiplicity.','proof':'Affine reproduction of target-local p; triangle inequality and weighted Cauchy-Schwarz bound target displacement relative to exact source pose; explicit interval bound links selected rational p to exact cube point; positive target interior/source exterior margins extend to a nonzero cube-face square.','source_pose_1nm_assumption_removed':True,'floating_guard_implementation_global_roundoff_certified':False,'guard_semantics_limitation':'Caps are the original recorded binary64 constants interpreted exactly. Distances and SSE in the theorem are real arithmetic. This does not prove that every arbitrary floating implementation of those guard predicates is sound, nor physical or collision-engine success. The1nm guard-allowance sensitivity is quantified separately, not assumed certified.','groups':groups,'source_box':sb[0],'target_boxes':tb,'cases':cases,'dependency_sha256':deps,'criteria_or_target_changed':False,'full_generator_mapping_accepted':False,'zero_execution':{'physics_steps':0,'inference_calls':0,'surface_queries':0,'fit_calls':0,'evaluation_episodes':0,'training_calls':0}}
 args.output.write_text(json.dumps(serial(report),indent=2)+'\n');print(json.dumps({'path':str(args.output),'sha256':digest(args.output),'cases':[{'case_id':c['case_id'],'status':c['status'],'points':c['candidate_count'],'positive':c['positive_witness_count'],'source_bridge_m':float(c['best_certificate']['exact_source_point_bridge_upper_m']),'inside_clearance_m':float(c['best_certificate']['best']['inside_clearance_lower_m']),'area_lower_m2':float(c['best_certificate']['best']['square_area_lower_m2'])} for c in cases]},indent=2))
if __name__=='__main__':main()
