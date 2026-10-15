"""Independent original URDF mesh-identity audit, no project importer helper.

Compare each collision mesh against its own explicit-name single-mesh MJCF.
Distances query every referenced vertex against the full triangle surface and
against independently built convex hulls, never just a point-cloud nearest neighbor.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json
import numpy as np
import mujoco
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
for n in ['mj_step','mj_step1','mj_step2']:
    setattr(mujoco,n,lambda *a,**k: (_ for _ in ()).throw(RuntimeError('No dynamics')))
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');O=Path('/private/tmp/compiled_hand_audit_20261015_v1/mano')
U=W/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def get_geom(model,gid):
    mid=int(model.geom_dataid[gid]);assert model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH
    va,vn=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid]);fa,fn=int(model.mesh_faceadr[mid]),int(model.mesh_facenum[mid])
    v=np.asarray(model.mesh_vert[va:va+vn],dtype=np.float64)@rot(model.geom_quat[gid]).T+model.geom_pos[gid]
    f=np.asarray(model.mesh_face[fa:fa+fn],dtype=np.int64)
    return v,f,mid

def distances(v,f,p):
    t=trimesh.Trimesh(vertices=v,faces=f,process=False)
    ds=[];cs=[];tis=[]
    for i in range(0,len(p),64):
        closest,dist,tid=trimesh.proximity.closest_point_naive(t,p[i:i+64]);ds.extend(dist);cs.extend(closest);tis.extend(tid)
    ds=np.asarray(ds);cs=np.asarray(cs);tis=np.asarray(tis);assert np.isfinite(ds).all() and np.isfinite(cs).all()
    k=int(np.argmax(ds))
    return {'source_query_count':len(p),'maximum_sampled_vertex_to_surface_distance_m':float(ds[k]),'mean_sampled_vertex_to_surface_distance_m':float(ds.mean()),'witness_query_local_m':p[k].tolist(),'witness_target_closest_local_m':cs[k].tolist(),'witness_target_triangle_index':int(tis[k]),'scope':'Each query is a referenced source mesh vertex on its surface; maximum is a sampled Hausdorff lower estimate, not a continuous maximum or a directed-rounding certificate.'}

# Reproduce native URDF importer using original exact declarations, no post-fix
# project helper or name-alias workaround. Original mesh bytes are never changed.
r=ET.parse(U).getroot();origin=ET.fromstring(ET.tostring(r))
ext=ET.Element('mujoco');ET.SubElement(ext,'compiler',{'strippath':'false'});r.insert(0,ext)
next(j for j in r.findall('joint') if j.get('name')=='world_joint').set('type','floating')
for m in r.findall('.//mesh'):m.set('filename',str((U.parent/m.get('filename')).resolve()))
p=O/'original_import_without_name_fix.urdf';assert not p.exists();p.write_bytes(ET.tostring(r,encoding='utf-8',xml_declaration=True))
model=mujoco.MjModel.from_xml_path(str(p));rows=[]
for link in ['ffdistal','mfdistal','rfdistal','lfdistal']:
    node=next(l for l in origin.findall('link') if l.get('name')==link);c=node.find('collision');m=c.find('geometry/mesh');o=c.find('origin');path=(U.parent/m.get('filename')).resolve()
    spec=ET.Element('mujoco',{'model':'isolated_declared_'+link});ET.SubElement(spec,'compiler',{'angle':'radian','strippath':'false'});asset=ET.SubElement(spec,'asset');ET.SubElement(asset,'mesh',{'name':'unique_declared_'+link,'file':str(path),'scale':m.get('scale','1 1 1')});body=ET.SubElement(ET.SubElement(spec,'worldbody'),'body',{'name':link});geom=ET.SubElement(body,'geom',{'name':'declared_collision','type':'mesh','mesh':'unique_declared_'+link,'pos':o.get('xyz','0 0 0'),'euler':o.get('rpy','0 0 0')})
    s=mujoco.MjModel.from_xml_string(ET.tostring(spec,encoding='unicode'))
    bid=model.body(link).id;gids=np.flatnonzero(model.geom_bodyid==bid);gids=[int(g) for g in gids if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH];assert len(gids)==1
    v,f,mi=get_geom(model,gids[0]);sv,sf,smi=get_geom(s,0)
    row={'link':link,'declared_collision_path':str(path),'declared_collision_sha256':sha(path),'mesh_scale':m.get('scale'),'imported_mesh_name':model.mesh(mi).name,'imported_vertex_count':len(v),'imported_face_count':len(f),'standalone_declared_vertex_count':len(sv),'standalone_declared_face_count':len(sf),'triangle_mesh_identity':bool(np.array_equal(v,sv) and np.array_equal(f,sf))}
    if row['triangle_mesh_identity']:
        row['point_to_triangle_status']='IDENTICAL_ARRAYS_NO_DISTANCE_QUERY_NEEDED'
    else:
        row['declared_vertices_to_imported_triangles']=distances(v,f,sv[np.unique(sf)])
        row['imported_vertices_to_declared_triangles']=distances(sv,sf,v[np.unique(f)])
        # MuJoCo general mesh contacts use convex hulls. These hull measures are
        # separately scoped, avoiding a false claim that render triangles are
        # the runtime contact boundary on nonconvex mesh surfaces.
        hv=ConvexHull(v);hs=ConvexHull(sv)
        row['convex_hull_declared_vertices_to_imported_hull']=distances(v,hv.simplices,sv[hs.vertices])
        row['convex_hull_imported_vertices_to_declared_hull']=distances(sv,hs.simplices,v[hv.vertices])
        row['imported_hull_vertices']=len(hv.vertices);row['declared_hull_vertices']=len(hs.vertices)
    rows.append(row)
visual=U.parent/'meshes/visual/f_distal_pst.obj';collision=U.parent/'meshes/collision/f_distal_pst.obj'
report={'schema_version':'independent_original_urdf_mesh_alias_audit_v1','source_urdf_sha256':sha(U),'script_sha256':sha(__file__),'original_import_urdf_sha256':sha(p),'mujoco_version':mujoco.__version__,'numpy_version':np.__version__,'trimesh_version':trimesh.__version__,'input_mesh_hashes':{str(x):sha(x) for x in [visual,collision]},'cases':rows,'conclusion':'Original MuJoCo URDF import merges different paths with the same f_distal_pst basename. First index collision correctly uses its declared visual mesh; middle/ring/little instead reuse it rather than their declared distinct collision mesh. FK identity cannot detect this geometry conversion defect.','physics_steps':0,'new_grasp_outputs':0,'fits':0,'scientific_mapping_acceptance':'NOT_ASSERTED'}
out=O/'collision_mesh_alias_independent_v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'cases':[{'link':r['link'],'declared_faces':r['standalone_declared_face_count'],'actual_faces':r['imported_face_count'],'identity':r['triangle_mesh_identity'],'max_imported_to_declared_surface_mm':r.get('imported_vertices_to_declared_triangles',{}).get('maximum_sampled_vertex_to_surface_distance_m',0)*1000,'max_imported_hull_to_declared_hull_mm':r.get('convex_hull_imported_vertices_to_declared_hull',{}).get('maximum_sampled_vertex_to_surface_distance_m',0)*1000} for r in rows]},indent=2))
