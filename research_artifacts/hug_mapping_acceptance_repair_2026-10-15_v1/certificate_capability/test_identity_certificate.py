import copy,unittest
import identity_certificate as c
I=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
def fixture(tilt=False):
 hand={'vertices':[[0,0,0],[4,0,0],[0,4,0],[0,0,-4]],'faces':[[0,1,2],[0,3,1],[0,2,3],[1,3,2]],'labels':['palm']*4,'transform':copy.deepcopy(I)}
 obj={'vertices':[[0,0,.0625],[4,0,.3125 if tilt else .0625],[0,4,.0625],[0,0,4.0625]],'faces':[[0,2,1],[0,1,3],[0,3,2],[1,2,3]],'labels':['object']*4,'transform':copy.deepcopy(I)}
 if tilt:q=[[1,1,.125],[1,1.125,.125],[1.125,1,.1328125]]
 else:q=[[1,1,.0625],[1,1.125,.0625],[1.125,1,.0625]]
 domain={'object_surface':obj,'query_triangles':[q],'object_faces':[0],'carrier_faces':[0]}
 return hand,copy.deepcopy(hand),[0,1,2,3],[0,1,2,3],[.001,.002,.005,.0625,.12890625],domain,copy.deepcopy(domain)
class Tests(unittest.TestCase):
 def run_case(self,args):return c.certify(*args)
 def assert_pass(self,v):self.assertEqual(v['status'],'PASS_STRUCTURAL_IDENTITY_ON_DECLARED_DOMAIN');self.assertTrue(v['distance_identity'] and v['normal_domain_identity']);self.assertFalse(v['historical_C2_acceptance']);self.assertFalse(v['real_HUG_Shadow_acceptance'])
 def assert_unproved(self,v):self.assertEqual(v['status'],'UNPROVED');self.assertFalse(v['normal_domain_identity'])
 def test_full_declared_domain_identity_positive(self):self.assert_pass(self.run_case(fixture()))
 def test_identity_on_threshold_equality(self):
  v=self.run_case(fixture());self.assert_pass(v);self.assertIn('1/16',v['bands_exact'])
 def test_identity_with_actual_threshold_crossing(self):
  args=fixture(True);self.assertLess(args[5]['query_triangles'][0][0][2],.12890625);self.assertGreater(args[5]['query_triangles'][0][2][2],.12890625);self.assert_pass(self.run_case(args))
 def test_common_rotation_translation_normal_identity(self):
  args=list(fixture(True));T=[[0,-1,0,.125],[1,0,0,-.25],[0,0,1,.5],[0,0,0,1]]
  for surface in (args[0],args[1],args[5]['object_surface'],args[6]['object_surface']):surface['transform']=copy.deepcopy(T)
  self.assert_pass(self.run_case(args))
 def test_explicit_permuted_correspondence_positive(self):
  args=list(fixture());target=args[1];vm=[2,0,3,1];fm=[3,1,0,2];target['vertices']=[None]*4;target['faces']=[None]*4
  for i,j in enumerate(vm):target['vertices'][j]=args[0]['vertices'][i]
  for i,j in enumerate(fm):target['faces'][j]=[vm[k] for k in args[0]['faces'][i]]
  args[2]=vm;args[3]=fm;args[6]['carrier_faces']=[fm[0]];self.assert_pass(self.run_case(args))
 def test_changed_geometry_rejected(self):
  a=list(fixture());a[1]['vertices'][0][0]=.001;v=self.run_case(a);self.assert_unproved(v);self.assertFalse(v['distance_identity'])
 def test_changed_winding_rejected(self):
  a=list(fixture());a[1]['faces'][0]=[0,2,1];self.assert_unproved(self.run_case(a))
 def test_changed_connectivity_rejected(self):
  a=list(fixture());a[1]['faces'][0]=[0,1,3];self.assert_unproved(self.run_case(a))
 def test_changed_part_label_rejected(self):
  a=list(fixture());a[1]['labels'][0]='thumb';self.assert_unproved(self.run_case(a))
 def test_nonbijective_map_rejected(self):
  a=list(fixture());a[2]=[0,0,2,3];self.assert_unproved(self.run_case(a))
 def test_changed_transform_rejected(self):
  a=list(fixture());a[1]['transform'][0][3]=.125;self.assert_unproved(self.run_case(a))
 def test_improper_transform_rejected(self):
  a=list(fixture());a[0]['transform'][0][0]=-1;a[1]['transform'][0][0]=-1;self.assert_unproved(self.run_case(a))
 def test_unverified_decimal_rotation_rejected(self):
  a=list(fixture());T=[[.6,-.8,0,0],[.8,.6,0,0],[0,0,1,0],[0,0,0,1]]
  a[0]['transform']=copy.deepcopy(T);a[1]['transform']=copy.deepcopy(T);self.assert_unproved(self.run_case(a))
 def test_missing_normal_domain_never_full_pass(self):
  a=list(fixture());a[5]=None;v=self.run_case(a);self.assert_unproved(v);self.assertTrue(v['distance_identity']);self.assertNotIn('zero_lost_and_added_area_on_declared_domain',v)
 def test_changed_query_domain_rejected(self):
  a=list(fixture());a[6]['query_triangles'][0][0][0]+=.125;self.assert_unproved(self.run_case(a))
 def test_normal_coverage_missing_carrier_rejected(self):
  a=list(fixture());del a[5]['carrier_faces'];self.assert_unproved(self.run_case(a))
 def test_query_edge_projection_rejected(self):
  a=list(fixture());a[5]['query_triangles'][0][0][0]=0;a[6]=copy.deepcopy(a[5]);self.assert_unproved(self.run_case(a))
 def test_competing_feature_not_certified_rejected(self):
  a=list(fixture());a[5]['query_triangles'][0]=[[.001,1,.0625],[.001,1.125,.0625],[.002,1,.0625]];a[6]=copy.deepcopy(a[5]);self.assert_unproved(self.run_case(a))
 def test_reversed_common_outward_hand_normal_rejected(self):
  a=list(fixture());a[0]['faces'][0]=[0,2,1];a[1]['faces'][0]=[0,2,1];self.assert_unproved(self.run_case(a))
 def test_degenerate_query_rejected(self):
  a=list(fixture());a[5]['query_triangles'][0][2]=a[5]['query_triangles'][0][1];a[6]=copy.deepcopy(a[5]);self.assert_unproved(self.run_case(a))
 def test_tangent_deadzone_rejected_after_exterior_checks(self):
  a=list(fixture());K=2**21;obj=a[5]['object_surface'];obj['vertices']=[[0,0,.25-K],[4,0,.25+3*K],[0,4,.25-K],[0,0,4.25-K]]
  a[5]['query_triangles']=[[[1,1,.25],[1,1.125,.25],[1+2**-24,1,.375]]];a[6]=copy.deepcopy(a[5]);v=self.run_case(a);self.assert_unproved(v);self.assertIn('deadzone',v['reason'])
 def test_empty_query_coverage_rejected(self):
  a=list(fixture());a[5]['query_triangles']=[];a[5]['carrier_faces']=[];a[5]['object_faces']=[];a[6]=copy.deepcopy(a[5]);self.assert_unproved(self.run_case(a))
 def test_one_bad_cell_prevents_partial_domain_pass(self):
  a=list(fixture());a[5]['query_triangles'].append([[.001,1,.0625],[.001,1.125,.0625],[.002,1,.0625]]);a[5]['carrier_faces'].append(0);a[5]['object_faces'].append(0);a[6]=copy.deepcopy(a[5]);self.assert_unproved(self.run_case(a))
 def test_invalid_band_rejected(self):
  for bands in ([],[0],[-1],[.001,.001]):
   a=list(fixture());a[4]=bands;self.assert_unproved(self.run_case(a))
 def test_nonfinite_geometry_rejected(self):
  a=list(fixture());a[0]['vertices'][0][0]=float('nan');self.assert_unproved(self.run_case(a))
if __name__=='__main__':unittest.main(verbosity=2)
