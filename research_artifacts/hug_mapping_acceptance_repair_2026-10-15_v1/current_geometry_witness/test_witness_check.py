import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,'/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
import constraint_core as core
from witness_check import PARTS,anatomical_part,descendant,fixed_rows,evaluate_fixed

class Controls(unittest.TestCase):
 def test_carriers_retained(self):
  self.assertEqual(anatomical_part('forearm'),'forearm_carrier');self.assertEqual(anatomical_part('wrist'),'wrist_carrier')
 def test_distinct_finger_names(self):
  self.assertEqual([anatomical_part(n) for n in ['ffdistal','mfdistal','rfdistal','lfdistal','thdistal']],list(PARTS[1:6]))
 def test_namespaced_owner(self):self.assertEqual(anatomical_part('robot/mfdistal'),'middle')
 def test_unknown_owner_refused(self):
  with self.assertRaises(ValueError):anatomical_part('unregistered_hand_geom')
 def test_descendant_root_included(self):
  p=[0,0,1,2,0];self.assertTrue(descendant(p,1,1));self.assertTrue(descendant(p,3,1));self.assertFalse(descendant(p,4,1))
 def test_literal_near_and_strict_far(self):
  p=core.proximity_constraints([.0005,.0025],[.0005,.0005],[0,0],[.001,.002],[True,False])
  np.testing.assert_array_equal(p['passed'],[True,False])
 def test_away_and_invalid_preserved(self):
  o=core.orientation_constraints([-.5,-.5],[.001,.001],[-1,-1],[.002,.002],1e-8,1e-9,[True,False])
  np.testing.assert_array_equal(o['passed'],[True,False])
 def test_historical_cap_never_rebased(self):
  e=np.zeros((5,3));e[:,0]=.001;caps=np.full(5,.001);e[0,0]=np.nextafter(.001,np.inf)
  self.assertFalse(core.exact_tip_status(e,caps)['passed']);self.assertEqual(caps[0],.001)
 def test_fixed_row_order_duplicates(self):
  p={'source_inventory_index':20,'row':{'native_sha256':'h','anatomical_part':'palm'}};o={'source_inventory_index':20,'native_sha256':'h','row':{'native_part':'index'}}
  saved={'whole_cell_rows':[p,p],'orientation_rows':[o,o]};pi,oi,pr,ore=fixed_rows(saved,20,'h');self.assertEqual(pi,[0,1]);self.assertEqual(oi,[0,1]);self.assertEqual(len(pr)+len(ore),4)
 def test_hash_refusal(self):
  with self.assertRaises(AssertionError):fixed_rows({'whole_cell_rows':[{'source_inventory_index':20,'row':{'native_sha256':'bad'}}],'orientation_rows':[]},20,'h')
 def test_injected_current_geometry_and_frozen_queries(self):
  p=[{'row':{'centroid_object_m':[1.,2.,3.],'anatomical_part':'palm','centroid_radius_m':.0001,'numerical_allowance_m':0,'band_mm':2,'relationship_change':'lost_native'}}]
  ori=[{'band_mm':2,'row':{'query_object_m':[3.,2.,1.],'object_outward_normal':[1.,0,0],'native_part':'index','native_facing_compatibility':-.5}}]
  hand=[{'part':'palm','version':'current'},{'part':'forearm_carrier','version':'current'}];seen=[]
  def qp(g,h,q):
   self.assertEqual(h[0]['version'],'current');np.testing.assert_array_equal(q,[[1,2,3]]);return np.array([.001]),q.copy(),np.zeros_like(q)
  def direction(q,n,h,t,p,expected_part):
   seen.extend(h);np.testing.assert_array_equal(q,[[3,2,1]]);self.assertEqual(expected_part[0],1)
   return {'score':np.array([-.5]),'distance_m':np.array([.001]),'valid':np.array([True]),'reason_masks':{}}
  r=evaluate_fixed(hand,[],p,ori,{'facing_dot_numerical_deadzone':1e-8,'query_boundary_tolerance_m':1e-9},None,qp,SimpleNamespace(mapped_direction_fields=direction),core)
  self.assertTrue(r['literal_surface_pass'].all());self.assertEqual(seen,hand);self.assertFalse(r['summary']['full_grid_mapping_acceptance'])
 def test_empty_orientation_unavailable(self):
  r=evaluate_fixed([],[],[],[],{'facing_dot_numerical_deadzone':1e-8,'query_boundary_tolerance_m':1e-9},None,None,None,core)
  self.assertEqual(r['summary']['orientation_coverage'],'UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS');self.assertFalse(r['summary']['full_grid_mapping_acceptance'])

if __name__=='__main__':unittest.main(verbosity=2)
