"""Independent geometry sanity on manufactured analytic primitives, no episodes."""
from pathlib import Path
import json,pickle
import numpy as np
from scipy.spatial.transform import Rotation
import audit_other_surfaces as a
out=Path(__file__).parent
R=Rotation.from_euler('xyz',[.31,-.27,.61]).as_matrix();pos=np.array([.01,-.04,.025])
cases=[('sphere',a.mujoco.mjtGeom.mjGEOM_SPHERE,[1.,0,0],[[2,0,0],[0,0,0],[0,0,.5]],[[1,0,0],[1,0,0],[0,0,1]],[1,-1,-.5]),('box',a.mujoco.mjtGeom.mjGEOM_BOX,[1.,1,1],[[2,2,0],[.2,.3,.4],[0,0,0]],[[1,1,0],[.2,.3,1],[1,0,0]],[2**.5,-.6,-1]),('cylinder',a.mujoco.mjtGeom.mjGEOM_CYLINDER,[1.,2,0],[[2,0,1],[.5,0,3],[2,0,3],[0,0,0]],[[1,0,1],[.5,0,2],[1,0,2],[1,0,0]],[1,1,2**.5,-1])]
records=[]
for name,kind,size,p,q,sd in cases:
 g={'kind':kind,'size':np.array(size),'R':R,'pos':pos};points=np.array(p,float)@R.T+pos;expected=np.array(q,float)@R.T+pos;got,closest,norm=a.geom_query(g,points);err=max(np.abs(got-np.asarray(sd)).max(),np.linalg.norm(closest-expected,axis=1).max());assert err<1e-12;assert np.allclose(np.linalg.norm(norm,axis=1),1,atol=1e-12);records.append({'shape':name,'case_count':len(p),'nonidentity_rigid_transform':True,'max_signed_distance_or_closest_point_error_m':float(err),'unit_normal':True})
verts=np.array([[x,y,z] for x in [-1.,1] for y in [-1.,1] for z in [-1.,1]]);tri,eq,_=a.hull(verts);points=np.array([[2,2,0],[.2,.3,.4],[0,0,0],[1.1,0,.9]])
g={'kind':a.mujoco.mjtGeom.mjGEOM_MESH,'size':np.ones(3),'R':R,'pos':pos,'tri':tri,'equations':eq};sd,q,norm=a.geom_query(g,points@R.T+pos);gb={'kind':a.mujoco.mjtGeom.mjGEOM_BOX,'size':np.ones(3),'R':R,'pos':pos};analytic,qa,_=a.geom_query(gb,points@R.T+pos);assert np.allclose(sd,analytic,atol=1e-12)
inn,wind=a.winding_inside(points,tri);assert np.array_equal(inn,[False,True,True,False]);assert np.allclose(wind,[0,1,1,0],atol=1e-12)
protocol=json.loads((out/'protocol_v3.json').read_text());native_checks=[]
for c in protocol['cases']:
 src=pickle.loads(Path(c['native_path']).read_bytes());faces=np.asarray(src['mesh_faces'],int);oriented=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);sort=np.sort(oriented,axis=1);_,idx=np.unique(sort,axis=0,return_inverse=True);sign=np.where(oriented[:,0]<oriented[:,1],1,-1);balance=np.bincount(idx,weights=sign);assert np.all(balance==0);native_checks.append({'native_sha256':c['native_sha256'],'every_shared_edge_opposite_orientation':True})
a.write(out/'geometry_sanity.json',{'script_sha256':a.sha(__file__),'main_protocol_sha256':a.sha(out/'protocol_v3.json'),'physics_steps':0,'primitive_analytic_checks':records,'convex_hull_vs_analytic_box_signed_distance_max_difference_m':float(np.abs(sd-analytic).max()),'oriented_closed_cube_winding_expected_inside_outside_pass':True,'native_face_orientation_checks':native_checks,'limitations':'Manufactured geometry exercises mathematical query machinery, not new generated proposals or evaluation cases. Closed orientation does not prove native meshes have no self-intersections.'})
print('geometry sanity passed')
