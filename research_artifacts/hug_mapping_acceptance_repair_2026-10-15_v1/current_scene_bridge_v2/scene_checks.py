"""Read static assembled-scene arrays against independent source assets."""
import numpy as np
import mujoco
from xml_reference import transform,quaternion_matrix_wxyz,transform_points,symmetric_vertex_distance
from bridge_model import ident,compiled_mesh_support
from object_reference import triangle_connectivity

def object_check(model,data,reference,object_position,object_quaternion,mass,tol):
 b=ident(model,mujoco.mjtObj.mjOBJ_BODY,'object');T=transform(object_position,quaternion_matrix_wxyz(object_quaternion));actual=transform(data.xpos[b],data.xmat[b].reshape(3,3));pe=float(np.linalg.norm(actual[:3,3]-T[:3,3]));re=float(np.max(np.abs(actual[:3,:3]-T[:3,:3])))
 gids=set(int(i) for i in np.flatnonzero(model.geom_bodyid==b));rows=[];expected=set()
 if reference is None:
  g=ident(model,mujoco.mjtObj.mjOBJ_GEOM,'object_geom');expected.add(g)
  row={'name':'object_geom','kind':'legacy_box','passed':bool(model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX and np.array_equal(model.geom_size[g],[.025,.025,.025]) and np.linalg.norm(data.geom_xpos[g]-object_position)<=tol['current_position_m'] and np.max(np.abs(data.geom_xmat[g].reshape(3,3)-T[:3,:3]))<=tol['current_rotation_matrix'] and model.geom_contype[g]==1 and model.geom_conaffinity[g]==1 and model.geom_margin[g]==0 and model.geom_gap[g]==0 and np.array_equal(model.geom_friction[g],[1,.005,.0001]))};rows.append(row)
  wantedI=np.eye(3)*mass*(.05**2+.05**2)/12;wantedipos=np.zeros(3)
 else:
  for src in reference['rows']:
   g=ident(model,mujoco.mjtObj.mjOBJ_GEOM,src['compiled_name']);expected.add(g);mid=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
   v0=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);f0=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid]);verts=model.mesh_vert[v0:v0+vn].astype(float);faces=model.mesh_face[f0:f0+fn].astype(int);GT=transform(data.geom_xpos[g],data.geom_xmat[g].reshape(3,3));actualpoints=transform_points(GT,verts);wantedT=T@src['local_frame'];wantedpoints=transform_points(wantedT,src['mesh']['vertices']);distance=symmetric_vertex_distance(actualpoints,wantedpoints)
   masks=bool(model.geom_contype[g]==src['contype'] and model.geom_conaffinity[g]==src['conaffinity'] and model.geom_margin[g]==src['margin'] and model.geom_gap[g]==src['gap'] and model.geom_group[g]==src['group'] and np.array_equal(model.geom_friction[g],src['friction']))
   row={'name':src['compiled_name'],'source_mesh':src['mesh']['path'],'collision_active':src['collision_active'],'source_compiled_vertex_set_error_m':distance,'masks_margin_gap_group_friction_match':masks}
   if src['collision_active']:
    _,support=compiled_mesh_support(model,g);row['compiled_support']=support;shape=bool(distance<=tol['mesh_vertex_set_m'] and support['graph_contains_all_full_hull_extreme_vertex_indices'])
   else:
    facescheck=triangle_connectivity(src['mesh']['vertices'],src['mesh']['faces'],actualpoints,faces,wantedT,tol['mesh_vertex_set_m']);row['visual_triangle_check']=facescheck;shape=bool(distance<=tol['mesh_vertex_set_m'] and facescheck['passed'])
   row['passed']=bool(shape and masks);rows.append(row)
  wantedI=reference['inertia_tensor_body']*(mass/reference['native_mass']);wantedipos=reference['inertial_position']
 inertiaR=quaternion_matrix_wxyz(model.body_iquat[b]);actualI=inertiaR@np.diag(model.body_inertia[b])@inertiaR.T;ie=float(np.max(np.abs(actualI-wantedI)));ipe=float(np.linalg.norm(model.body_ipos[b]-wantedipos));me=abs(float(model.body_mass[b])-mass)
 return {'position_error_m':pe,'rotation_matrix_max_error':re,'mass_error_kg':me,'inertia_tensor_max_error_kg_m2':ie,'inertial_position_error_m':ipe,'actual_object_geom_ids':sorted(gids),'expected_object_geom_ids':sorted(expected),'complete_object_geom_inventory':gids==expected,'geometry_rows':rows,'passed':bool(pe<=tol['current_position_m'] and re<=tol['current_rotation_matrix'] and me<=1e-12 and ie<=1e-12 and ipe<=tol['current_position_m'] and gids==expected and all(r['passed'] for r in rows))}

def controller_and_table_check(runner,source,names,q,root_position,root_quaternion,efforts):
 m=runner.model;d=runner.data;named=dict(zip(names,q,strict=True));js={j['name']:j for j in source['joints'] if j['kind']=='revolute'};rows=[]
 for name in runner._hand_joint_names:
  j=ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);a=ident(m,mujoco.mjtObj.mjOBJ_ACTUATOR,'servo_'+name);kp=50. if name.startswith('WR') else 10.;force=abs(efforts[name]) if name.startswith('WR') else min(abs(efforts[name]),1.)
  ok=bool(m.actuator_trnid[a,0]==j and m.actuator_gaintype[a]==mujoco.mjtGain.mjGAIN_FIXED and m.actuator_biastype[a]==mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_dyntype[a]==mujoco.mjtDyn.mjDYN_NONE and np.array_equal(m.actuator_gear[a],[1,0,0,0,0,0]) and m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and m.actuator_ctrllimited[a] and m.actuator_forcelimited[a] and np.max(np.abs(m.actuator_ctrlrange[a]-js[name]['limits']))<=1e-12 and np.array_equal(m.actuator_forcerange[a],[-force,force]) and m.actuator_gainprm[a,0]==kp and np.array_equal(m.actuator_biasprm[a,:3],[0,-kp,-.3]) and d.ctrl[a]==named[name])
  rows.append({'name':name,'actuator_id':a,'named_target_rad':float(named[name]),'readback_ctrl_rad':float(d.ctrl[a]),'passed':ok})
 table=ident(m,mujoco.mjtObj.mjOBJ_GEOM,'table');tableok=bool(m.geom_type[table]==mujoco.mjtGeom.mjGEOM_BOX and np.array_equal(m.geom_size[table],[.4,.4,.025]) and np.array_equal(d.geom_xpos[table],[0,0,-.025]) and np.array_equal(d.geom_xmat[table].reshape(3,3),np.eye(3)) and m.geom_contype[table]==1 and m.geom_conaffinity[table]==1 and m.geom_margin[table]==0 and m.geom_gap[table]==0 and np.array_equal(m.geom_friction[table],[1,.005,.0001]))
 target=ident(m,mujoco.mjtObj.mjOBJ_BODY,'arm_target');forearm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'forearm');eq=ident(m,mujoco.mjtObj.mjOBJ_EQUALITY,'ideal_arm');weldok=bool(m.neq==1 and m.eq_type[eq]==mujoco.mjtEq.mjEQ_WELD and m.eq_obj1id[eq]==target and m.eq_obj2id[eq]==forearm and m.eq_active0[eq] and d.eq_active[eq] and np.array_equal(m.eq_data[eq],[0,0,0,0,0,0,1,0,0,0,1]) and np.array_equal(m.eq_solref[eq],[.01,1]));mocap=int(m.body_mocapid[target]);mocapok=bool(np.array_equal(d.mocap_pos[mocap],root_position) and np.array_equal(d.mocap_quat[mocap],root_quaternion))
 overrideok=bool(m.npair==0 and m.nexclude==0 and m.nplugin==0 and m.nflex==0);inventory=set(range(m.ngeom));known=set(runner.object_geom_ids)|{table}|{g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in runner._hand_body_ids}
 return {'servo_rows':rows,'servo_count':int(m.nu),'ideal_arm_weld':weldok,'mocap_input_identity':mocapok,'table_matches_declared_shape_frame':tableok,'no_extra_pair_exclusion_plugin_flex':overrideok,'complete_scene_geometry_inventory':inventory==known,'passed':bool(len(rows)==24 and m.nu==24 and all(r['passed'] for r in rows) and weldok and mocapok and tableok and overrideok and inventory==known),'scope':'Static command/address/weld/table configuration only; no servo tracking or force-closure outcome validation.'}
