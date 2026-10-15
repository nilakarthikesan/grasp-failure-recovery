"""Exact structural paired-triangle identity certificate on a declared domain.

Pure rational arithmetic over encoded finite numbers. Deliberately incomplete:
full normal-domain proof currently supports outward tetrahedral surfaces only.
No sampled equality, floating allowance, optimizer or caller validity flags.
"""
from fractions import Fraction as F
import hashlib,json,math

class Unproved(ValueError):pass

def rational(x):
 if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise Unproved('finite numeric coordinates required')
 return F(x)
def vec(x):
 if len(x)!=3:raise Unproved('three coordinates required')
 return tuple(rational(v) for v in x)
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(a,t):return tuple(x*t for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rigid(T):
 if len(T)!=4 or any(len(row)!=4 for row in T):raise Unproved('4x4 transform required')
 t=tuple(tuple(rational(x) for x in row) for row in T)
 if t[3]!=(0,0,0,1):raise Unproved('homogeneous bottom row')
 R=[row[:3] for row in t[:3]]
 if any(dot(R[i],R[j])!=(1 if i==j else 0) for i in range(3) for j in range(3)):raise Unproved('encoded rotation not exactly orthonormal')
 if dot(R[0],cross(R[1],R[2]))!=1:raise Unproved('proper rotation required')
 return t

def mesh(data):
 V=[vec(v) for v in data['vertices']];faces=data['faces'];labels=data['labels'];T=rigid(data['transform'])
 if not V or not faces or len(labels)!=len(faces):raise Unproved('nonempty labelled mesh required')
 out=[]
 for face in faces:
  if len(face)!=3 or any(type(i) is not int or i<0 or i>=len(V) for i in face) or len(set(face))!=3:raise Unproved('triangle connectivity invalid')
  a,b,c=[V[i] for i in face];n=cross(sub(b,a),sub(c,a))
  if dot(n,n)==0:raise Unproved('degenerate surface triangle')
  out.append(tuple(face))
 if any(not isinstance(x,str) or not x for x in labels):raise Unproved('explicit nonempty face labels required')
 return V,out,tuple(labels),T

def permutation(p,n):
 if len(p)!=n or any(type(i) is not int for i in p) or sorted(p)!=list(range(n)):raise Unproved('bijective full correspondence required')
 return p

def structural(source,target,vertex_map,face_map):
 A,AF,AL,AT=mesh(source);B,BF,BL,BT=mesh(target)
 if len(A)!=len(B) or len(AF)!=len(BF):raise Unproved('surface sizes differ')
 permutation(vertex_map,len(A));permutation(face_map,len(AF))
 if AT!=BT:raise Unproved('common encoded rigid transform required')
 if any(A[i]!=B[j] for i,j in enumerate(vertex_map)):raise Unproved('corresponding geometry differs')
 for i,face in enumerate(AF):
  mapped=tuple(vertex_map[j] for j in face);actual=BF[face_map[i]]
  if actual not in (mapped,mapped[1:]+mapped[:1],mapped[2:]+mapped[:2]):raise Unproved('oriented connectivity differs')
  if AL[i]!=BL[face_map[i]]:raise Unproved('part correspondence differs')
 return A,AF,AT

def tetra_outward(V,faces):
 if len(V)!=4 or len(faces)!=4 or {frozenset(f) for f in faces}!={frozenset(set(range(4))-{i}) for i in range(4)}:raise Unproved('normal/exterior proof currently requires a complete tetrahedron')
 for f in faces:
  a,b,c=[V[i] for i in f];n=cross(sub(b,a),sub(c,a));other=next(i for i in range(4) if i not in f)
  if dot(n,sub(V[other],a))>=0:raise Unproved('tetrahedron must be nondegenerate and outward oriented')

def face_info(V,f):
 a,b,c=[V[i] for i in f];n=cross(sub(b,a),sub(c,a));return a,b,c,n,dot(n,n)
def interior(p,a,b,c):
 u=sub(b,a);v=sub(c,a);w=sub(p,a);uu=dot(u,u);uv=dot(u,v);vv=dot(v,v);wu=dot(w,u);wv=dot(w,v);den=uu*vv-uv*uv
 if den<=0:raise Unproved('degenerate carrier')
 s=(vv*wu-uv*wv)/den;t=(uu*wv-uv*wu)/den
 return s>0 and t>0 and s+t<1

def squared_plane_lower(points,a,n,n2):
 h=[dot(n,sub(q,a)) for q in points]
 if min(h)>0:return min(h)**2/n2
 if max(h)<0:return max(h)**2/n2
 return F(0)
def separated_squared(lower,upper,tolerance):
 # sqrt(lower)>sqrt(upper)+tolerance, without irrational arithmetic.
 z=lower-upper-tolerance*tolerance
 return z>0 and z*z>4*upper*tolerance*tolerance

def normal_domain(source_vertices,source_faces,T,domain,target_domain,face_map):
 # Same query/object domain is required; only carrier indices transform under
 # the supplied full source-face correspondence.
 if set(domain)!={'object_surface','query_triangles','object_faces','carrier_faces'} or set(target_domain)!=set(domain):raise Unproved('complete explicit normal-domain data required')
 if domain['object_surface']!=target_domain['object_surface'] or domain['query_triangles']!=target_domain['query_triangles'] or domain['object_faces']!=target_domain['object_faces']:raise Unproved('query/object domain differs')
 carrier=domain['carrier_faces'];target_carrier=target_domain['carrier_faces']
 if any(type(i) is not int or i<0 or i>=len(source_faces) for i in carrier):raise Unproved('invalid carrier face')
 if target_carrier!=[face_map[i] for i in carrier]:raise Unproved('domain carrier correspondence differs')
 OV,OF,OL,OT=mesh(domain['object_surface'])
 if OT!=T:raise Unproved('object/query and paired-hand transforms must coincide')
 tetra_outward(source_vertices,source_faces);tetra_outward(OV,OF)
 queries=domain['query_triangles'];object_faces=domain['object_faces']
 if not queries or len(queries)!=len(carrier) or len(queries)!=len(object_faces):raise Unproved('normal coverage missing')
 tie=F(1,100000000);boundary=F(1,100000000);object_boundary=F(1,10000000);deadzone=F(1,1000000)
 proof=[]
 for k,(raw,j,oj) in enumerate(zip(queries,carrier,object_faces)):
  if len(raw)!=3 or type(oj) is not int or oj<0 or oj>=len(OF):raise Unproved('query triangle/object face invalid')
  Q=[vec(q) for q in raw];qn=cross(sub(Q[1],Q[0]),sub(Q[2],Q[0]))
  if dot(qn,qn)==0:raise Unproved('query domain has zero area')
  a,b,c,n,n2=face_info(source_vertices,source_faces[j]);oa,ob,oc,on,on2=face_info(OV,OF[oj])
  # Each whole triangle lies strictly inside one oriented object facet.
  if any(dot(on,sub(q,oa))!=0 or not interior(q,oa,ob,oc) for q in Q):raise Unproved('query cell not strictly inside declared object facet')
  if dot(qn,on)<=0:raise Unproved('query winding disagrees with outward object normal')
  heights=[dot(n,sub(q,a)) for q in Q]
  if min(heights)<=0 or min(heights)**2/n2<=boundary**2:raise Unproved('query not provably outside source hand with boundary clearance')
  P=[sub(q,scale(n,h/n2)) for q,h in zip(Q,heights)]
  if not all(interior(p,a,b,c) for p in P):raise Unproved('nearest-point projection leaves carrier interior')
  upper=max(heights)**2/n2
  for jj,f in enumerate(source_faces):
   if jj==j:continue
   aa,bb,cc,nn,nn2=face_info(source_vertices,f);lower=squared_plane_lower(Q,aa,nn,nn2)
   if not separated_squared(lower,upper,tie):raise Unproved('competing nearest feature not separated over entire cell')
  # One outward object support plane suffices to certify every projected hand
  # point is exterior to the closed convex object, with the original safeguard.
  object_heights=[dot(on,sub(p,oa)) for p in P]
  if min(object_heights)<=0 or min(object_heights)**2/on2<=object_boundary**2:raise Unproved('closest-point exterior/clearance unproved')
  numerator=-dot(n,on)
  if numerator*numerator<=deadzone*deadzone*n2*on2:raise Unproved('facing sign tangent or in numerical deadzone')
  proof.append({'query_cell':k,'source_carrier_face':j,'target_carrier_face':face_map[j],'object_face':oj,'continuous_facing_sign':1 if numerator>0 else -1,'normal_identity_proof':'same oriented carrier face under identical proper rotation','query_plane_distance_squared_min':str(min(heights)**2/n2),'query_plane_distance_squared_max':str(upper),'strict_feature_and_exterior_coverage':True})
 return proof

def certify(source,target,vertex_map,face_map,bands_m,domain,target_domain):
 """Prove identity on the explicit query cells; fail closed outside capability.

 PASS is scoped to these encoded surfaces/cells. It is not historical C2
 acceptance, MANO/Shadow homology, execution quality, or outcome equivalence.
 """
 result={'schema':'paired_triangle_structural_certificate_v1','status':'UNPROVED','distance_identity':False,'normal_domain_identity':False,'historical_C2_acceptance':False,'real_HUG_Shadow_acceptance':False}
 try:
  bands=[rational(x) for x in bands_m]
  if not bands or any(x<=0 for x in bands) or len(set(bands))!=len(bands):raise Unproved('positive distinct declared bands required')
  V,faces,T=structural(source,target,vertex_map,face_map)
  result.update(distance_identity=True,threshold_predicates_identical_for_all_declared_bands=True)
  if domain is None or target_domain is None:raise Unproved('normal-domain coverage missing')
  proof=normal_domain(V,faces,T,domain,target_domain,face_map)
  result.update(status='PASS_STRUCTURAL_IDENTITY_ON_DECLARED_DOMAIN',normal_domain_identity=True,zero_lost_and_added_area_on_declared_domain=True,continuous_normal_proofs=proof,source_hash=digest(source),target_hash=digest(target),domain_hash=digest(domain),target_domain_hash=digest(target_domain),bands_exact=[str(x) for x in bands],proof_scope='Exactly the supplied oriented query triangles; all arithmetic proofs use exact rational values of encoded inputs. Not a full-object domain claim.',distance_basis='Bijection preserves every labelled oriented triangle and common transform, hence the represented sets and their distance functions are equal. Threshold equality holds even on band-crossing cells.')
 except (Unproved,KeyError,TypeError,IndexError) as exc:result['reason']=str(exc)
 return result
