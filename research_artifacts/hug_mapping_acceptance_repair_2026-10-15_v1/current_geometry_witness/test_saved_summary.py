import unittest
from summarize_saved_results import classify

class ClassificationControls(unittest.TestCase):
 def test_interval_counterexamples_distinct_from_insufficient_proof(self):
  rows=[{'row':{'relationship_change':k,'band_mm':2,'anatomical_part':'palm'}} for k in ['lost_native','added','lost_native','added','lost_native','added']]
  p={'lower_m':[.004,0,.001,.002,0,.003],'upper_m':[.006,.001,.003,.003,.002,.004],'passed':[False,False,False,False,True,True]}
  r=classify(p,{'valid':[],'score':[],'distance_m':[]},{'passed':[]},rows,[],{'facing_dot_numerical_deadzone':1e-8,'query_boundary_tolerance_m':1e-9})
  self.assertEqual([r['proximity'][k] for k in ('preserved','definite_lost','definite_added','unresolved')],[2,1,1,2]);self.assertTrue(r['orientation']['no_orientation_evidence'])
 def test_native_away_opposite_deadzone_and_ineligible_partition(self):
  ori=[{'band_mm':2,'row':{'native_part':'index','native_facing_compatibility':s}} for s in [-.5,.5,.5,.5]]
  r=classify({'lower_m':[],'upper_m':[],'passed':[]},{'valid':[True,True,True,False],'score':[-.5,-.5,0,.5],'distance_m':[.001]*4},{'passed':[True,False,False,False]},[],ori,{'facing_dot_numerical_deadzone':1e-8,'query_boundary_tolerance_m':1e-9})
  self.assertEqual([r['orientation'][k] for k in ('preserved','valid_opposite_within_source_band','valid_sign_unresolved_within_source_band','invalid_eligibility')],[1,1,1,1])
 def test_outside_source_band_is_not_reliable_frozen_stratum_opposite(self):
  o=[{'band_mm':2,'row':{'native_part':'index','native_facing_compatibility':.5}}]
  r=classify({'lower_m':[],'upper_m':[],'passed':[]},{'valid':[True],'score':[-.5],'distance_m':[.003]},{'passed':[False]},[],o,{'facing_dot_numerical_deadzone':1e-8,'query_boundary_tolerance_m':1e-9})
  self.assertEqual(r['orientation']['valid_opposite_within_source_band'],0);self.assertEqual(r['orientation']['eligible_outside_source_band'],1)
if __name__=='__main__':unittest.main(verbosity=2)
