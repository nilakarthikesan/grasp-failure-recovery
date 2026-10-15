import json,unittest
from pathlib import Path
import numpy as np
from exposed_classifier import classify

def sphere(i,p,r,part):return dict(id=i,kind='sphere',center=np.array(p,float),radius=r,part=part)
def box(i,p,s,part):return dict(id=i,kind='box',center=np.array(p,float),size=np.array(s,float),part=part)
def oracle(g,points):
    p=np.asarray(points)-g['center']
    if g['kind']=='sphere':
        length=np.linalg.norm(p,axis=1);n=p/np.where(length>0,length,1)[:,None];n[length==0]=[1,0,0]
        return length-g['radius'],g['center']+g['radius']*n,n
    delta=np.abs(p)-g['size'];positive=np.maximum(delta,0);sd=np.linalg.norm(positive,axis=1)+np.minimum(np.max(delta,axis=1),0)
    cp=np.clip(p,-g['size'],g['size']);n=np.zeros_like(p)
    for j in range(len(p)):
        if np.any(delta[j]>0):n[j]=(p[j]-cp[j])/np.linalg.norm(p[j]-cp[j])
        else:
            ax=int(np.argmax(delta[j]));n[j,ax]=1 if p[j,ax]>=0 else -1;cp[j,ax]=n[j,ax]*g['size'][ax]
    return sd,cp+g['center'],n

def row(gs,p,part):return classify(gs,[p],[part],oracle)['rows'][0]
class Controls(unittest.TestCase):
    def test_single_exterior(self):
        r=row([sphere(1,[0,0,0],1,'palm')],[2,0,0],'palm')
        self.assertEqual(r['occupancy_status'],'STRICT_UNION_EXTERIOR');self.assertTrue(r['whole_union_distance_exact_conditional']);self.assertEqual(r['whole_union_boundary_lower_bound_m'],1)
    def test_single_interior(self):
        r=row([sphere(1,[0,0,0],1,'palm')],[.2,0,0],'palm')
        self.assertEqual(r['occupancy_status'],'STRICT_COMPONENT_INTERIOR');self.assertFalse(r['whole_union_distance_exact_conditional']);self.assertAlmostEqual(r['whole_union_boundary_lower_bound_m'],.8);self.assertAlmostEqual(r['whole_union_boundary_upper_bound_m'],.8)
    def test_nested_exterior_hidden_part(self):
        r=row([sphere(1,[0,0,0],1,'index'),sphere(2,[0,0,0],2,'palm')],[3,0,0],'index')
        self.assertEqual(r['whole_union_boundary_lower_bound_m'],1);self.assertEqual(r['part_nearest_component_candidate']['exposure_status'],'STRICTLY_BURIED_IN_OTHER_COMPONENT');self.assertIsNone(r['part_exposed_subset_upper_bound_m']);self.assertEqual(r['part_exposed_subset_lower_bound_m'],2)
    def test_nested_inside_global_surface_not_boundary(self):
        r=row([sphere(1,[0,0,0],1,'index'),sphere(2,[0,0,0],2,'palm')],[.9,0,0],'palm')
        self.assertAlmostEqual(r['global_individual_surface_distance_m'],.1);self.assertAlmostEqual(r['whole_union_boundary_lower_bound_m'],1.1);self.assertAlmostEqual(r['whole_union_boundary_upper_bound_m'],1.1)
    def test_overlap_buried_closest_does_not_invent_distance(self):
        r=row([sphere(1,[-.5,0,0],1,'index'),sphere(2,[.5,0,0],1,'palm')],[0,0,0],'index')
        self.assertAlmostEqual(r['whole_union_boundary_lower_bound_m'],.5);self.assertIsNone(r['whole_union_boundary_upper_bound_m']);self.assertEqual(r['part_nearest_component_candidate']['exposure_status'],'STRICTLY_BURIED_IN_OTHER_COMPONENT')
    def test_touching_boxes_shared_face(self):
        r=row([box(1,[-1,0,0],[1,1,1],'index'),box(2,[1,0,0],[1,1,1],'palm')],[0,0,0],'index')
        self.assertEqual(r['occupancy_status'],'BOUNDARY_OR_COINCIDENT_UNRESOLVED');self.assertFalse(r['whole_union_distance_exact_conditional']);self.assertFalse(r['part_exposed_subset_distance_exact_conditional']);self.assertIsNone(r['whole_union_boundary_upper_bound_m'])
    def test_coincident_spheres_exterior_distance_but_ambiguous_exposure(self):
        r=row([sphere(1,[0,0,0],1,'index'),sphere(2,[0,0,0],1,'palm')],[2,0,0],'index')
        self.assertTrue(r['whole_union_distance_exact_conditional']);self.assertFalse(r['part_exposed_subset_distance_exact_conditional']);self.assertIsNone(r['part_exposed_subset_upper_bound_m']);self.assertEqual(len(r['global_near_tie_geom_ids']),2)
    def test_carrier_is_not_ignored(self):
        r=row([sphere(1,[0,0,0],1,'palm'),sphere(2,[0,0,0],2,'forearm_carrier')],[3,0,0],'palm')
        self.assertEqual(r['part_nearest_component_candidate']['strict_burying_parts'],['forearm_carrier'])
    def test_boundary_tolerance_unknown(self):
        r=row([sphere(1,[0,0,0],1,'palm')],[1+1e-10,0,0],'palm');self.assertEqual(r['occupancy_status'],'BOUNDARY_OR_COINCIDENT_UNRESOLVED');self.assertFalse(r['whole_union_distance_exact_conditional'])
    def test_distinct_exterior_tie(self):
        r=row([sphere(1,[-2,0,0],1,'index'),sphere(2,[2,0,0],1,'palm')],[0,0,0],'index');self.assertTrue(r['whole_union_distance_exact_conditional']);self.assertTrue(r['part_exposed_subset_distance_exact_conditional']);self.assertFalse(r['normal_uniqueness_verified'])
    def test_invalid_inputs(self):
        for p in [[[np.nan,0,0]],[[1,2]]]:
            with self.assertRaises(ValueError):classify([sphere(1,[0,0,0],1,'palm')],p,['palm'],oracle)
        with self.assertRaises(ValueError):classify([sphere(1,[0,0,0],1,'palm')],[[2,0,0]],['missing'],oracle)
        with self.assertRaises(ValueError):classify([sphere(1,[0,0,0],1,'palm')],[[2,0,0]],['palm'],oracle,float('inf'))
    def test_malformed_oracle_distance(self):
        def bad(g,p):
            sd,cp,n=oracle(g,p);return sd,cp+np.array([.1,0,0]),n
        r=classify([sphere(1,[0,0,0],1,'palm')],[[2,0,0]],['palm'],bad)['rows'][0]
        self.assertFalse(r['oracle_consistent']);self.assertFalse(r['whole_union_distance_exact_conditional']);self.assertIsNone(r['whole_union_boundary_upper_bound_m']);self.assertIsNone(r['part_exposed_subset_upper_bound_m'])
    def test_malformed_oracle_own_surface(self):
        def bad(g,p):
            sd,cp,n=oracle(g,p)
            if np.allclose(p,[[2,0,0]]):return sd,np.array([[3.,0,0]]),n
            return sd,cp,n
        r=classify([sphere(1,[0,0,0],1,'palm')],[[2,0,0]],['palm'],bad)['rows'][0]
        self.assertEqual(r['global_nearest_component_candidate']['exposure_status'],'OWN_SURFACE_RESIDUAL_UNRESOLVED');self.assertFalse(r['whole_union_distance_exact_conditional']);self.assertIsNone(r['whole_union_boundary_upper_bound_m'])
    def test_exterior_burial_contradiction_refuses_all_bounds(self):
        def bad(g,p):
            sd,cp,n=oracle(g,p)
            if g['id']==2:
                sd[np.linalg.norm(p-np.array([1.,0,0]),axis=1)<1e-12]=-.1
            return sd,cp,n
        r=classify([sphere(1,[0,0,0],1,'palm'),sphere(2,[0,0,0],.5,'index')],[[3,0,0]],['palm'],bad)['rows'][0]
        self.assertTrue(r['exterior_burial_contradiction']);self.assertFalse(r['oracle_consistent']);self.assertFalse(r['whole_union_distance_exact_conditional']);self.assertIsNone(r['whole_union_boundary_lower_bound_m']);self.assertIsNone(r['whole_union_boundary_upper_bound_m']);self.assertIsNone(r['part_exposed_subset_lower_bound_m'])
    def test_duplicate_rows_preserved(self):
        r=classify([sphere(1,[0,0,0],1,'palm')],[[2,0,0],[2,0,0]],['palm','palm'],oracle);self.assertEqual(r['rows'][0],r['rows'][1]);self.assertEqual(r['component_point_queries'],2);self.assertEqual(r['candidate_point_queries'],4)
if __name__=='__main__':unittest.main(verbosity=2)
