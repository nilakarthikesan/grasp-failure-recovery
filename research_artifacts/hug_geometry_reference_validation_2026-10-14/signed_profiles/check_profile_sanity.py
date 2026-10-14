"""Independent known-shape checks for static signed-profile functions."""
from pathlib import Path
import ast,hashlib,importlib.util,json
import numpy as np
import mujoco
def forbidden(*args,**kwargs):raise RuntimeError('No physics')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
base=Path(__file__).resolve().parent;source=base/'profile_signed_geometry.py';p=json.loads((base/'protocol.json').read_text());spec=importlib.util.spec_from_file_location('geom_lib',p['source_paths']['query_library']);lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
tree=ast.parse(source.read_text());names=('weighted_quantile','profile','strict_inside','on_boundary_with_same_normal','primitive_tris','signed_to_visual');nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];ns={'np':np,'mujoco':mujoco};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source)+':functions-only','exec'),ns)
vertices=np.asarray([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)]);tri,eq,_=lib.hull(vertices);points=np.array([[0.,0,0],[1.1,0,0],[1.,0,0],[0,0,-1.2]]);sd,wind=ns['signed_to_visual'](points,tri,lib,p);expected=np.array([-1.,.1,0,.2]);assert np.max(np.abs(sd-expected))<1e-12
box={'kind':int(mujoco.mjtGeom.mjGEOM_BOX),'size':np.ones(3),'pos':np.zeros(3),'R':np.eye(3),'id':1};inside=ns['strict_inside'](points,box,1e-8);assert inside.tolist()==[True,False,False,False]
q=np.array([[1.,0,0]]);n=np.array([[1.,0,0]]);assert ns['on_boundary_with_same_normal'](q,n,box,1e-8)[0];neighbor=box|{'pos':np.array([2.,0,0]),'id':2};assert not ns['on_boundary_with_same_normal'](q,n,neighbor,1e-8)[0];assert ns['strict_inside'](q+1e-7*n,neighbor,1e-8)[0]
primitives=[]
for kind,size in [(mujoco.mjtGeom.mjGEOM_SPHERE,[.011,0,0]),(mujoco.mjtGeom.mjGEOM_CYLINDER,[.0135,.015,0])]:
    g={'kind':int(kind),'size':np.array(size),'tri':None};t,meta=ns['primitive_tris'](g,lib,p['primitive_max_chord_error_m']);assert meta.get('conservative_maximum_facet_chord_deficit_m',meta.get('maximum_side_chord_deficit_m'))<=p['primitive_max_chord_error_m'];primitives.append(meta)
profile=ns['profile'](np.array([-.003,-.001,0.,.002,.01]),np.array([1.,2,3,4,5]),p);assert profile['finite_sampled_max_penetration_mm']==3;assert profile['negative_count']==2 and abs(profile['negative_weight_fraction']-.2)<1e-12;assert sum(profile['signed_histogram']['weights'])==15
result={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'profile_script_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'known_box_signed_distance_max_error_m':float(np.max(np.abs(sd-expected))),'exact_outside_AABB_skipped_winding_count':wind['outside_visual_AABB_winding_skipped_count'],'strict_inside_known_box':True,'same_normal_duplicate_and_opposite_normal_abutting_probe':True,'primitive_chord_certificates':primitives,'weighted_negative_profile_and_histogram':True,'physics_steps':0};(base/'profile_sanity.json').write_text(json.dumps(result,indent=2)+'\n')
