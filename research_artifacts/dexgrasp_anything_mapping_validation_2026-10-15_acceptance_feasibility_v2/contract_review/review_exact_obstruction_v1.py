"""Independent saved-proof audit; no model, fitting, surface query or dynamics."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import mpmath as mp

ROOT=Path('/private/tmp/dga_mapping_acceptance_20261015_v2')
PATH=ROOT/'feasibility/exact_palm_obstruction_v2.json'
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(v):return F(int(v['numerator']),int(v['denominator']))
def fvec(v):return [rat(x) for x in v]
def mpf(v):
 q=v if isinstance(v,F) else F.from_float(float(v));return mp.mpf(q.numerator)/q.denominator
def xvec(n,key,default='0 0 0'):return [F(x) for x in (n.get(key,default) if n is not None else default).split()]
def cross(a,b):return mp.matrix([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def unit(x):return x/mp.sqrt(sum(v*v for v in x))
def rotation(axis,q):
 a=unit(mp.matrix(list(map(mpf,axis))));x,y,z=a;c,s=mp.cos(q),mp.sin(q)
 K=mp.matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
 return c*mp.eye(3)+(1-c)*(a*a.T)+s*K
def transform(xyz,rpy):
 R=rotation([0,0,1],mpf(rpy[2]))*rotation([0,1,0],mpf(rpy[1]))*rotation([1,0,0],mpf(rpy[0]));T=mp.eye(4)
 for i in range(3):
  for j in range(3):T[i,j]=R[i,j]
  T[i,3]=mpf(xyz[i])
 return T
def source_frame(raw,names,xml):
 a=unit(mp.matrix(list(map(mpf,raw[3:6]))));b=unit(mp.matrix(list(map(mpf,raw[6:9]))))
 u=unit(a+b);v=unit(a-b);x=unit(u+v);y=unit(u-v);z=unit(cross(x,y));T=mp.eye(4)
 for i in range(3):
  T[i,0],T[i,1],T[i,2],T[i,3]=x[i],y[i],z[i],mpf(raw[i])
 nodes={n.find('child').get('link'):n for n in xml.findall('joint')};chain=[];p='palm';q=dict(zip(names,raw[9:]))
 while p!='forearm':j=nodes[p];chain.append(j);p=j.find('parent').get('link')
 for j in reversed(chain):
  o=j.find('origin');A=transform(xvec(o,'xyz'),xvec(o,'rpy'))
  if j.get('type')=='revolute':
   R=rotation(xvec(j.find('axis'),'xyz','1 0 0'),mpf(q[j.get('name')]));H=mp.eye(4)
   for k in range(3):
    for l in range(3):H[k,l]=R[k,l]
   A=A*H
  T=T*A
 return T
def anchors(xml,names):
 nodes={n.find('child').get('link'):n for n in xml.findall('joint')};out={}
 for name in names:
  chain=[];p=name
  while p!='palm':j=nodes[p];chain.append(j);p=j.find('parent').get('link')
  chain.reverse()
  assert all(xvec(j.find('origin'),'xyz')==[0,0,0] for j in chain[1:])
  out[name]=tuple(xvec(chain[0].find('origin'),'xyz')) if chain else (F(0),)*3
 return out
def main():
 mp.mp.dps=120;r=json.loads(PATH.read_text());verified=[]
 for p,h in r['dependency_sha256'].items():assert sha(p)==h;verified.append(p)
 surdf=next(Path(p) for p in verified if p.endswith('/shadowhand.urdf'));turdf=next(Path(p) for p in verified if p.endswith('/shadow_hand_right.urdf'))
 sx,tx=ET.parse(surdf).getroot(),ET.parse(turdf).getroot();groups=r['groups'];names=[n for g in groups for n in g['sites']]
 sa,ta=anchors(sx,names),anchors(tx,names)
 assert len(names)==len(set(names))==10
 for g in groups:
  assert g['multiplicity']==len(g['sites'])
  for n in g['sites']:assert sa[n]==tuple(fvec(g['source'])) and ta[n]==tuple(fvec(g['target']))
 counts=0;cases=[]
 for c in r['cases']:
  rawpath=Path(c['source_raw']['path']);manifest=json.loads((rawpath.parent/'native_manifest.json').read_text());assert sha(rawpath)==c['source_raw']['sha256']
  assert manifest['object_from_input']==np.eye(4).tolist();assert len(manifest['joint_names'])==24 and len(set(manifest['joint_names']))==24
  with np.load(rawpath,allow_pickle=False) as z:raw=z['hand_qpos'][c['source_raw']['row']]
  T=source_frame(raw,manifest['joint_names'],sx);intervals=c['source_transform_exact_interval'];maxwidth=F(0)
  for i in range(4):
   for j in range(4):
    a,b=rat(intervals[i][j]['lower']),rat(intervals[i][j]['upper']);assert mpf(a)<=T[i,j]<=mpf(b);maxwidth=max(maxwidth,b-a)
  exact_recount=[]
  for cert in c['all_certificates']+[c['one_nm_guard_allowance_sensitivity_certificate']]:
   p,x=fvec(cert['reference_local_point_exact']),fvec(cert['object_point_exact']);f=cert['face_axis'];assert x[f] in (-F(1,40),F(1,40));assert all(abs(z)<=F(1,40) for z in x)
   w=fvec(cert['affine_weights_exact']);assert sum(w)==1
   assert all(sum(wi*rat(g['target'][j]) for wi,g in zip(w,groups))==p[j] for j in range(3))
   K=sum(wi*wi/g['multiplicity'] for wi,g in zip(w,groups));assert K==rat(cert['weighted_cauchy_schwarz_K_exact'])
   l1=sum(abs(wi) for wi in w);assert l1==rat(cert['weights_l1_exact'])
   off=[sum(wi*rat(g['source'][j]) for wi,g in zip(w,groups))-p[j] for j in range(3)];offup=rat(cert['shape_offset_norm_upper_m']);assert offup>=0 and offup**2>=sum(z*z for z in off)
   M,B=rat(cert['maximum_guard_exact_m']),rat(cert['SSE_guard_exact_m2']);eps=rat(cert['guard_anchor_allowance_m']);assert M>=0 and B>=0 and eps>=0
   mmax=rat(cert['maximum_guard_motion_upper_m']);msse=rat(cert['SSE_guard_motion_upper_m']);motion=rat(cert['motion_upper_m']);assert mmax==(M+eps)*l1+offup
   if eps==0:assert msse>=offup and (msse-offup)**2>=B*K
   else:assert mpf(msse-offup)>=(mp.sqrt(mpf(B))+mp.sqrt(10)*mpf(eps))*mp.sqrt(mpf(K))
   assert motion==min(mmax,msse)
   bridge=rat(cert['exact_source_point_bridge_upper_m']);bridgebox=[]
   for i in range(3):
    lo=rat(intervals[i][3]['lower']);hi=rat(intervals[i][3]['upper'])
    for j in range(3):
     a,b=rat(intervals[i][j]['lower']),rat(intervals[i][j]['upper']);lo+=min(a*p[j],b*p[j]);hi+=max(a*p[j],b*p[j])
    bridgebox.append(max(abs(lo-x[i]),abs(hi-x[i])))
   # The producer rounds intermediate interval arithmetic outwards, so its
   # norm upper bound must cover this tighter unrounded rational interval.
   assert bridge>=0 and bridge**2>=sum(z*z for z in bridgebox)
   best=cert['best'];box=r['target_boxes'][best['target_box_index']];assert box['axis_aligned']
   bc,bh=fvec(box['center']),fvec(box['half']);source=r['source_box'];sc,sh=fvec(source['center']),fvec(source['half']);assert source['axis_aligned']
   inside=min(h-abs(a-b) for a,b,h in zip(p,bc,bh))-motion-bridge;outside=max(abs(a-b)-h for a,b,h in zip(p,sc,sh))-bridge;edge=min(F(1,40)-abs(x[j]) for j in range(3) if j!=f)
   side=max(F(0),min(inside,outside,edge)/4)
   assert inside==rat(best['inside_clearance_lower_m']) and outside==rat(best['source_outside_clearance_lower_m']);assert side==rat(best['square_side_lower_m']) and side*side==rat(best['square_area_lower_m2'])
   assert (side>0)==(cert['status']=='CERTIFIED_EXACT_DECLARED_MATH_OBSTRUCTION');counts+=1;exact_recount.append(float(side*side))
  cases.append({'case_id':c['case_id'],'guard_scope':c['guard_scope'],'high_precision_source_frame_enclosed':True,'maximum_source_interval_width':str(maxwidth),'certificates_recounted':len(exact_recount),'best_area_m2':float(rat(c['best_certificate']['best']['square_area_lower_m2']))})
 out={'schema_version':'independent_exact_palm_obstruction_review_v1','status':'PASS_SCOPED_EXACT_MATH_CERTIFICATE','proof':{'path':str(PATH),'sha256':sha(PATH)},'review_code_sha256':sha(__file__),'verified_dependency_count':len(verified),'exact_certificates_recounted':counts,'cases':cases,'reviewed_mathematics':['Ten anchor origins are joint-independent because descendant origins are zero; group multiplicities preserve the 28-site SSE upper bound.','Affine reproduction and weighted Cauchy-Schwarz correctly bound any proper target palm placement under exact recorded max/SSE guards.','Source robust6D matches pinned function with no normalization floor active. Exact interval Taylor order, outward rounding, reciprocal and isqrt bounds are conservative.','Target interior/source exterior margins transfer through the explicit source-pose bridge; a centered cube-face square of side margin/4 remains strictly mismatched.'],'scope':['Old row1 and new native row0: obstruction under each original pose-specific max/SSE guards; new row0 tighter component caps are separately certified.','This is an exact mathematical guard theorem; it does not certify every arbitrary floating-point implementation of guard predicates.','High precision mpmath frame comparison is an independent numerical cross-check; soundness is from the separately reviewed rational interval formulas.','No statement of global generator impossibility for other poses or for altered targets/criteria.'],'nonblocking_hardening':['certificate() could assert exact cube-face membership, nonnegative caps and axis-aligned source box internally; all actual saved cases independently satisfy these.'],'zero_execution':{'physics_steps':0,'inference_calls':0,'surface_queries':0,'fit_calls':0,'evaluation_episodes':0,'training_calls':0}}
 dst=ROOT/'contract_review/independent_exact_obstruction_review_v1.json';dst.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
