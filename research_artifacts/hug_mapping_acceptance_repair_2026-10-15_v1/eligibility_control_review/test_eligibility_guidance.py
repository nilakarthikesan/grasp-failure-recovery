"""Manufactured geometry/wiring tests. No simulator or saved HUG geometry imports."""
from pathlib import Path
from types import SimpleNamespace
import ast,sys,unittest
import numpy as np
R=Path('/private/tmp/hug_feasible_eligibility_2026-10-15_v1')
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts')
ns={'np':np,'prange':range}
def loaddefs(path,names):
 nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
loaddefs(W/'hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py',('closest_triangle','nearest_parts','winding_inside','tri_info'))
a=SimpleNamespace(**{n:ns[n] for n in ('closest_triangle','nearest_parts','winding_inside','tri_info')},PARTS=['thumb','palm'])
ns['a']=a
loaddefs(W/'hug_orientation_uncertainty_acceptance_2026-10-14/orientation/check_orientation.py',('triangle_candidates','visual_membership'))
meshkind=7
def geom_query(g,points):
 if 'preset' in g:
  cp=np.tile(g['preset'],(len(points),1));return np.full(len(points),g['distance']),cp,np.tile([0.,0.,1.],(len(points),1))
 ds,cp,idx=a.nearest_parts(points,g['tri'],np.zeros(len(g['tri']),int),1);return ds[:,0],cp,a.tri_info(g['tri'])[0][idx]
a.geom_query=geom_query
base=SimpleNamespace(a=a,refresh_colliders=lambda hand,*args:hand,mujoco=SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_MESH=meshkind)))
class Old:
 def raw(self,x):return {'surface_margin':np.arange(len(self.oq)*5+2,dtype=float)}
ns.update(base=base,PreviousCase=Old)
tree=ast.parse((R/'eligibility_case.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
for n in nodes:
 if isinstance(n,ast.FunctionDef):n.decorator_list=[]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(R/'eligibility_case.py'),'exec'),ns)
Case=ns['Case'];gap=ns['conflicting_mesh_gap'];oldcandidate=ns['triangle_candidates'];tol=1e-8;normtol=1e-6
V=np.array([[0.,0,0],[2,0,0],[0,2,0],[0,0,2]]);faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]]);visual=V[faces]
def plane(z,reverse=False):
 t=np.array([[-1.,-1.,z],[1.,-1.,z],[0.,1.,z]])
 return t[[0,2,1]] if reverse else t
def mesh(*tri,part='thumb'):return {'kind':meshkind,'part':part,'tri':np.array(tri),'pos':np.zeros(3),'R':np.eye(3)}
def preset(cp,part='thumb',distance=.001):return {'kind':0,'part':part,'preset':cp,'distance':distance}
def make(hand):
 c=Case();c.oq=np.zeros((1,3));c.original_hand=hand;c.model=c.data=None;c.opos=np.zeros(3);c.oR=np.eye(3);c.visual_tri=visual;c.orientation_protocol={'tie_distance_tolerance_m':tol,'normal_vector_difference_tolerance':normtol,'object_boundary_tolerance_m':1e-7};return c
def exclusion(hand):
 q=np.zeros((1,3));records=[(g,*geom_query(g,q)) for g in hand];D=min(abs(r[1][0]) for r in records);bad=False
 for g,d,cp,n in records:
  if g['kind']==meshkind and abs(d[0])<=D+tol:
   _,_,fa,pa=oldcandidate(q,g['tri'],np.zeros(len(g['tri']),int),a.tri_info(g['tri'])[0],abs(d),cp,n,tol,normtol);bad|=bool((fa|pa)[0])
 return bad
class Tests(unittest.TestCase):
 def test_no_conflicting_face_positive(self):
  g=mesh(plane(.001));v=make([g]).raw(np.zeros(30));self.assertGreater(v['raw_mesh_conflicting_facet_gap_m'][0],0)
 def test_reversed_coincident_face_conflicts(self):
  hand=[mesh(plane(.001),plane(.001,True))];self.assertTrue(exclusion(hand));self.assertLess(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0)
 def test_inclusive_exact_tie_float_grouping(self):
  d=.05;edge=d+tol;self.assertGreater(edge-d-tol,0.)
  hand=[mesh(plane(d),plane(edge,True))];self.assertTrue(exclusion(hand));self.assertEqual(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0.)
 def test_one_ulp_outside_tie_positive(self):
  hand=[mesh(plane(.001),plane(np.nextafter(.001+tol,np.inf),True))];self.assertFalse(exclusion(hand));self.assertGreater(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0.)
 def test_point_difference_can_conflict_with_same_normal(self):
  z=.001;sep=1e-7
  left=np.array([[-1.,-1.,z],[-sep,1.,z],[-sep,-1.,z]])
  right=np.array([[sep,-1.,z],[sep,1.,z],[1.,-1.,z]])
  hand=[mesh(left,right)];q=np.zeros((1,3));d,cp,n=geom_query(hand[0],q)
  _,_,fa,pa=oldcandidate(q,hand[0]['tri'],np.zeros(2,int),a.tri_info(hand[0]['tri'])[0],d,cp,n,tol,normtol)
  self.assertFalse(fa[0]);self.assertTrue(pa[0]);self.assertLess(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0)
 def test_local_conflict_in_far_mesh_is_not_global_exclusion(self):
  hand=[mesh(plane(.001)),mesh(plane(.002),plane(.002,True),part='palm')];self.assertFalse(exclusion(hand));self.assertGreater(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0.)
 def test_other_mesh_at_global_inclusive_tie_conflicts(self):
  hand=[mesh(plane(.001)),mesh(plane(.001+tol),plane(.001+tol,True),part='palm')];self.assertTrue(exclusion(hand));self.assertEqual(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0.)
 def test_other_mesh_own_normal_not_global_normal(self):
  hand=[mesh(plane(.001)),mesh(plane(.001+tol,True),part='palm')];self.assertFalse(exclusion(hand));self.assertGreater(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0],0.)
 def test_nested_predicate_grid(self):
  for dg in (.001,.001+tol,np.nextafter(.001+tol,np.inf),.002):
   for dz in (0.,tol,np.nextafter(tol,np.inf),1e-5):
    for reverse in (False,True):
     hand=[mesh(plane(.001)),mesh(plane(dg),plane(dg+dz,reverse),part='palm')];self.assertEqual(make(hand).raw(np.zeros(30))['raw_mesh_conflicting_facet_gap_m'][0]<=0,exclusion(hand))
 def test_per_part_tie_selects_declared_part_not_geom_order(self):
  hand=[preset([.2,.2,.2],part='palm'),preset([2.,2.,2.],part='thumb')];v=make(hand).raw(np.zeros(30));self.assertGreater(v['raw_closest_visual_exterior_margin_m'][0],0)
 def test_visual_membership_boundary_and_inside_equivalence(self):
  for point in ([.2,.2,.2],[2.,2.,2.],[.2,.2,0.],[.2,.2,-5e-8],[.2,.2,-1e-7],[.2,.2,-2e-7]):
   c=make([preset(point)]);v=c.raw(np.zeros(30));kind,_=ns['visual_membership'](np.array([point]),visual,c.orientation_protocol);self.assertEqual(v['raw_closest_visual_exterior_margin_m'][0]>0,kind[0]>0)
 def test_original_margin_unchanged_prefix(self):
  v=make([preset([2.,2.,2.])]).raw(np.zeros(30));np.testing.assert_array_equal(v['surface_margin'][:7],v['original_surface_margin']);self.assertEqual(len(v['surface_margin']),9)
 def test_extra_policy_preserves_all_old_masks(self):
  sys.path[:0]=[str(R),'/private/tmp/hug_se3_protected_fit_2026-10-15_v1'];from extended_policy import protected_surface_constraint_policy
  ref={'proximity':{'passed':np.array([True,False])},'exact_orientation':{'passed':np.array([False,True])},'exact_orientation_fields':{'valid':np.array([True,True])},'literal_surface_pass':np.array([True,False,False,True]),'surface_margin':np.ones(16)}
  p=protected_surface_constraint_policy(ref);np.testing.assert_array_equal(p['hard_mask'][-4:],True);np.testing.assert_array_equal(p['slack_coefficients'][-4:],0);np.testing.assert_array_equal(p['protected_literal_surface_rows'],ref['literal_surface_pass']);self.assertEqual(p['hard_count']+p['soft_count'],16)
 def test_extra_policy_rejects_negative_protected_margin(self):
  sys.path[:0]=[str(R),'/private/tmp/hug_se3_protected_fit_2026-10-15_v1'];from extended_policy import protected_surface_constraint_policy
  ref={'proximity':{'passed':np.array([True])},'exact_orientation':{'passed':np.array([True])},'exact_orientation_fields':{'valid':np.array([True])},'literal_surface_pass':np.array([True,True]),'surface_margin':np.ones(8)};ref['surface_margin'][-1]=-1
  with self.assertRaises(ValueError):protected_surface_constraint_policy(ref)
if __name__=='__main__':unittest.main(verbosity=2)
