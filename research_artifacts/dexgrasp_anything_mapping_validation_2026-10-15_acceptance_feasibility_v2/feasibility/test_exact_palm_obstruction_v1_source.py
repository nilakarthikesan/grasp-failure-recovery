"""Controls and independent high-precision numerical cross-checks; no physical execution."""
import importlib.util,json,random
from pathlib import Path
from fractions import Fraction as F
import mpmath as mp
P=Path(__file__).with_name('certify_exact_palm_obstruction.py');sp=importlib.util.spec_from_file_location('proof',P);a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
count=0

def check(ok):
 global count
 assert ok;count+=1

def m(v):return mp.mpf(v.numerator)/v.denominator
mp.mp.dps=150
rng=random.Random(20261015)
for _ in range(500):
 x=F(rng.randrange(-1000000,1000000),rng.randrange(1,100000));y=F(rng.randrange(-1000000,1000000),rng.randrange(1,100000));X,Y=a.I(x),a.I(y)
 check((X+Y).contains(x+y));check((X-Y).contains(x-y));check((X*Y).contains(x*y))
 if y:check((X/Y).contains(x/y))
for x in [F(0),F(1),F(2),F(1,10**100),F(123456789,12345)]:
 v=a.I(x).sqrt();check(v.a*v.a<=x<=v.b*v.b)
for x in [F(0),F(1,10**12),F(-37,100),F(1),F(3),F(-3)]:
 s,c=a.sincos(x);check(m(s.a)<=mp.sin(m(x))<=m(s.b));check(m(c.a)<=mp.cos(m(x))<=m(c.b))
for lo,hi in [(F(-1),F(1)),(F(0),F(0))]:
 try:a.I(1)/a.I(lo,hi);raise AssertionError('zero-containing division accepted')
 except AssertionError:count+=1
try:a.I(-1).sqrt();raise RuntimeError('negative sqrt accepted')
except AssertionError:count+=1
G=[{'source':v,'target':v,'multiplicity':mult,'sites':[]} for v,mult in [((F(0),F(0),F(0)),1),((F(1),F(0),F(0)),3),((F(0),F(1),F(0)),3),((F(0),F(0),F(1)),3)]]
x=(F(-1,40),F(0),F(0));src={'center':(F(10),F(0),F(0)),'half':(F(1),)*3,'index':0,'axis_aligned':True};tar={'center':(F(0),)*3,'half':(F(1),)*3,'index':0,'axis_aligned':True}
c=a.certificate(x,x,0,a.eye(4),G,src,[tar],F(0),F(0));check(c['status'].startswith('CERTIFIED'));check(c['best']['square_area_lower_m2']>0)
c=a.certificate(x,x,0,a.eye(4),G,tar,[tar],F(0),F(0));check(c['status']=='UNRESOLVED')
c=a.certificate(x,x,0,a.eye(4),G,src,[tar],F(100),F(100));check(c['status']=='UNRESOLVED')
S=a.eye(4);S[0][3]=a.I(2);c=a.certificate(x,x,0,S,G,src,[tar],F(0),F(0));check(c['status']=='UNRESOLVED')
# Independent high-precision source reconstruction from native saved rows.
import numpy as np,xml.etree.ElementTree as ET
report=json.loads(Path(__file__).with_name('exact_palm_obstruction_v2.json').read_text());sourcename=next(k for k in report['dependency_sha256'] if k.endswith('/shadowhand.urdf'));xml=ET.parse(sourcename).getroot();joints={j.find('child').get('link'):j for j in xml.findall('joint')}
def Fparse(v):return F(int(v['numerator']),int(v['denominator']))
def unit(v):return v/mp.sqrt(sum(x*x for x in v))
def rot(axis,theta):
 u=unit(mp.matrix([m(F(v)) for v in axis]));K=mp.matrix([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
 return mp.eye(3)*mp.cos(theta)+(u*u.T)*(1-mp.cos(theta))+K*mp.sin(theta)
for case in report['cases']:
 rawp=Path(case['source_raw']['path']);man=json.loads(rawp.with_name('native_manifest.json').read_text())
 with np.load(rawp,allow_pickle=False) as z:v=[m(F.from_float(float(x))) for x in z['hand_qpos'][case['source_raw']['row']]]
 u,w=unit(mp.matrix(v[3:6])),unit(mp.matrix(v[6:9]));mid,diff=unit(u+w),unit(u-w);rx,ry=unit(mid+diff),unit(mid-diff);rz=unit(mp.matrix([rx[1]*ry[2]-rx[2]*ry[1],rx[2]*ry[0]-rx[0]*ry[2],rx[0]*ry[1]-rx[1]*ry[0]]))
 S=mp.eye(4)
 for i in range(3):
  S[i,0],S[i,1],S[i,2],S[i,3]=rx[i],ry[i],rz[i],v[i]
 q=dict(zip(man['joint_names'],v[9:]));chain=[];cur='palm'
 while cur!='forearm':j=joints[cur];chain.append(j);cur=j.find('parent').get('link')
 for j in reversed(chain):
  origin=j.find('origin');xyz=[m(F(t)) for t in (origin.get('xyz','0 0 0') if origin is not None else '0 0 0').split()];rpy=[m(F(t)) for t in (origin.get('rpy','0 0 0') if origin is not None else '0 0 0').split()]
  R=rot((0,0,1),rpy[2])*rot((0,1,0),rpy[1])*rot((1,0,0),rpy[0]);ax=j.find('axis');axis=(ax.get('xyz','1 0 0') if ax is not None else '1 0 0').split()
  if j.get('type')=='revolute':R=R*rot(axis,q[j.get('name')])
  M=mp.eye(4)
  for i in range(3):
   M[i,3]=xyz[i]
   for k in range(3):M[i,k]=R[i,k]
  S=S*M
 for i in range(4):
  for k in range(4):
   iv=case['source_transform_exact_interval'][i][k];check(m(Fparse(iv['lower']))<=S[i,k]<=m(Fparse(iv['upper'])))
check(all(case['status'].startswith('CERTIFIED') for case in report['cases']))
result={'schema_version':'dga_interval_proof_controls_v1','status':'PASS','checks':count,'exact_rational_arithmetic_controls':True,'manufactured_obstruction_and_non_obstruction_controls':True,'independent_150_digit_mpmath_source_pose_crosscheck':True,'mpmath_crosscheck_not_used_for_authoritative_bounds':True,'interval_oracle_script_sha256':a.digest(P),'controls_script_sha256':a.digest(Path(__file__)),'report_sha256':a.digest(Path(__file__).with_name('exact_palm_obstruction_v2.json')),'physics_steps':0,'fit_calls':0,'surface_queries':0,'evaluation_episodes':0,'training_calls':0}
Path(__file__).with_name('exact_palm_obstruction_controls_v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
