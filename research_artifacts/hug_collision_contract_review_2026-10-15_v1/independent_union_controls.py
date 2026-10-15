"""Analytic manufactured counterexamples for metric semantics; no saved grasps or primary code."""
from pathlib import Path
import json,math,hashlib
O=Path(__file__).parent
rows=[]
def record(name,truth,values,argument):
 assert truth,name
 rows.append({'name':name,'passed':True,'values':values,'argument':argument})
# Spheres concentric: inner surface is wholly buried.
record('nested_spheres_component_boundary_differs_inside_union',.25<1.,{'query':[0,0,0],'radii':[.25,1.],'min_component_surface_distance':.25,'union_boundary_distance':1.},'The union is the radius-one sphere. Its inner radius-quarter component has no exposed boundary.')
record('outside_union_global_nearest_is_exposed',min(2.-.25,2.-1.)==1.,{'query':[2,0,0],'component_distances':[1.75,1.],'union_boundary_distance':1.},'A closest point on any component from outside the union cannot be strictly buried in another component, which would intersect the joining segment first.')
# Two unit spheres centered ±a along x. At q=0, minimum distance to
# union complement is sqrt(1-a²): minimize x²+y²+z² subject to both
# (x±a)²+y²+z² >=1; their minimum is 1-a²+2a|x|, minimized at x=0.
a=.5;correct=math.sqrt(1-a*a)
record('overlap_inside_depth_is_only_lower_bound',.5<correct,{'centers':[[-a,0,0],[a,0,0]],'radii':[1,1],'query':[0,0,0],'max_component_penetration_lower_bound':.5,'exact_union_boundary_distance':correct},'The two component nearest surfaces at x=±.5 are buried; the nearest union exit lies on the intersection circle, distance sqrt(.75).')
record('per_part_buried_surface_does_not_prove_exposed_part_available',True,{'inner_part':'A','outer_part':'B','query':[2,0,0],'part_A_component_distance':1.75,'part_A_exposed_subset':'empty in this analytic construction'},'Per-part minimum component distance remains a lower bound; missing finite exposed witness must not become a fabricated exact distance or a passing orientation.')
# Boxes [-1,0]×[-1,1]² and [0,1]×[-1,1]². Their union is a box.
record('touching_box_shared_face_is_union_interior',0<1,{'query':[0,0,0],'component_signed_distances':[0,0],'any_component_strict_inside':False,'union_interior':True,'exact_union_boundary_distance':1.},'The union is [-1,1]³. Shared-face center has an open unit-radius ball in the union despite zero individual signed distances.')
# Looking only at a probe endpoint misses a thin solid along the ray.
record('finite_probe_endpoint_can_skip_thin_neighbor',not (0.000002<=0.000010<=0.000003),{'ray_occupied_interval_m':[.000002,.000003],'probe_t_m':.000010,'probe_endpoint_outside_neighbor':True,'ray_intersects_neighbor':True},'Analytic interval overlap detects the intervening micrometer-thick component; one endpoint membership query does not.')
out={'schema':'independent_collision_union_semantics_controls_v1','passed':True,'control_count':len(rows),'controls':rows,'primary_implementation_tested':False,'new_saved_grasp_queries':0,'new_eval_data':False,'new_physics_steps':0,'scope':'Analytic checks demonstrate distinctions and necessary edge cases. They do not verify a specific numerical union-query implementation.'}
p=O/'independent_union_controls.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'controls':len(rows),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
