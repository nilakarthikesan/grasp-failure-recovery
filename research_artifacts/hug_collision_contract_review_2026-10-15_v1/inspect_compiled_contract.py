"""Static compiled geometry metadata inspection; no hand/object distance queries or stepping."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
O=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
I=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
B={};checks=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v,s):
 global checks
 checks+=1
 if not bool(v):raise AssertionError(s)
def bind(p,h=None):
 p=Path(p);v=sha(p);check(h is None or v==h,'source hash '+str(p));B[str(p)]=v

def forbidden(*args,**kwargs):raise RuntimeError('No simulation stepping permitted')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)
bind(I);iv=json.loads(I.read_text());cases=[]
for index in [8,12,16,20,24,28]:
 c=iv['modern_cases'][index]
 for f in [c['scene_mjb'],c['stored_shadow_command'],*c['frame_files'].values()]:bind(f['path'],f['sha256'])
 m=mujoco.MjModel.from_binary_path(c['scene_mjb']['path']);d=mujoco.MjData(m)
 with np.load(c['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
 originalq=d.qpos.copy()
 with np.load(c['stored_shadow_command']['path'],allow_pickle=False) as z:
  row=c['stored_shadow_command']['row_index'];names=z['joint_names'].tolist();q=z['nominal_target_rad'][row];pos=z['root_target_position_m'][row];quat=z['root_target_quaternion_wxyz'][row]
  for name,value in zip(names,q):d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)]]=value
  rootadr=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')];d.qpos[rootadr:rootadr+3]=pos;d.qpos[rootadr+3:rootadr+7]=quat
 if m.nmocap:d.mocap_pos[0]=pos;d.mocap_quat[0]=quat
 d.qvel[:]=0;mujoco.mj_forward(m,d);check(d.time==0,'static time')
 forearm=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'forearm');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
 def desc(b,r):
  while b:
   if b==r:return True
   b=int(m.body_parentid[b])
  return False
 exclusions={tuple(sorted([int(s)>>16,int(s)&65535])) for s in m.exclude_signature}
 records=[];hand=[];objects=[];visual=[];meshes={}
 for g in range(m.ngeom):
  b=int(m.geom_bodyid[g]);is_hand=desc(b,forearm);is_obj=desc(b,obj)
  if not (is_hand or is_obj):continue
  ct=int(m.geom_contype[g]);ca=int(m.geom_conaffinity[g]);active=bool(ct or ca);kind=int(m.geom_type[g]);mid=int(m.geom_dataid[g]);Rbody=d.xmat[b].reshape(3,3);Rgeom=np.empty(9);mujoco.mju_quat2Mat(Rgeom,m.geom_quat[g]);Rgeom=Rgeom.reshape(3,3)
  epos=float(np.max(np.abs(d.xpos[b]+Rbody@m.geom_pos[g]-d.geom_xpos[g])));erot=float(np.max(np.abs(Rbody@Rgeom-d.geom_xmat[g].reshape(3,3))));check(epos<1e-12 and erot<1e-12,'compiled local/body geom transform')
  record={'geom_id':g,'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g),'body_id':b,'body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b),'is_hand':is_hand,'is_object':is_obj,'type':kind,'type_name':mujoco.mjtGeom(kind).name,'contype':ct,'conaffinity':ca,'group':int(m.geom_group[g]),'margin_m':float(m.geom_margin[g]),'gap_m':float(m.geom_gap[g]),'priority':int(m.geom_priority[g]),'condim':int(m.geom_condim[g]),'friction':m.geom_friction[g].tolist(),'size':m.geom_size[g].tolist(),'local_pos':m.geom_pos[g].tolist(),'local_quat':m.geom_quat[g].tolist(),'world_pos':d.geom_xpos[g].tolist(),'world_rotation':d.geom_xmat[g].reshape(3,3).tolist(),'compiled_transform_max_error_m':epos,'compiled_rotation_max_error':erot,'mesh_id':mid if kind==mujoco.mjtGeom.mjGEOM_MESH else None}
  if kind==mujoco.mjtGeom.mjGEOM_MESH and mid not in meshes:
   va=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);v=m.mesh_vert[va:va+nv].astype(float);adr=int(m.mesh_graphadr[mid]);h=ConvexHull(v);nr=int(m.mesh_graph[adr]) if adr>=0 else 0;ids=m.mesh_graph[adr+2+nr:adr+2+2*nr].copy() if adr>=0 else np.arange(nv)
   check(np.all(ids>=0)&np.all(ids<nv),'graph vertex indices');gh=ConvexHull(v[ids]);res=float(np.max(v@gh.equations[:,:3].T+gh.equations[:,3]));check(res<1e-9,'compiled support hull contains all original vertices')
   meshes[mid]={'id':mid,'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,mid),'compiled_vertex_count':nv,'compiled_face_count':int(m.mesh_facenum[mid]),'compiled_vertex_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'mesh_pos':m.mesh_pos[mid].tolist(),'mesh_quat':m.mesh_quat[mid].tolist(),'mesh_graph_address':adr,'mesh_graph_hull_vertex_count':nr,'mesh_graph_hull_vertex_indices':ids.tolist(),'full_scipy_hull_vertex_count':len(h.vertices),'full_hull_extrema_in_compiled_graph':bool(np.isin(h.vertices,ids).all()),'all_vertices_max_compiled_graph_halfspace_residual_m':res,'full_vs_graph_volume_abs_difference_m3':abs(float(h.volume-gh.volume))}
  records.append(record)
  if is_hand and active:hand.append(g)
  if is_obj and active:objects.append(g)
  if is_obj and not active:visual.append(g)
 pairs=[]
 for hg in hand:
  for og in objects:
   bh=int(m.geom_bodyid[hg]);bo=int(m.geom_bodyid[og]);mask=bool((int(m.geom_contype[hg])&int(m.geom_conaffinity[og]))or(int(m.geom_contype[og])&int(m.geom_conaffinity[hg])));excluded=tuple(sorted((bh,bo))) in exclusions;weldsame=bool(m.body_weldid[bh]==m.body_weldid[bo]);parent=bool(m.body_weldid[m.body_parentid[bh]]==m.body_weldid[bo] or m.body_weldid[m.body_parentid[bo]]==m.body_weldid[bh]);check(mask and not excluded and not weldsame and not parent,'all selected hand-object pair filters')
   pairs.append({'hand_geom':hg,'object_geom':og,'mask_compatible':mask,'body_excluded':excluded,'same_welded_body':weldsame,'direct_parent_weld_relation':parent,'geom_margin_pair_m':float(max(m.geom_margin[hg],m.geom_margin[og])),'geom_gap_pair_m':float(max(m.geom_gap[hg],m.geom_gap[og]))})
 check(m.npair==0,'historical explicit-pair precondition');check(m.nflex==0,'no flex mesh collision alternative');check(m.nplugin==0,'no collision plugin');check(d.time==0,'no time advancement')
 contract=json.loads(Path(c['frame_files']['scene_contract.json']['path']).read_text())
 cases.append({'source_index':index,'object_id':c['object_id'],'native_sha256':c['native']['sha256'],'geometry_content_hash':c['geometry_content_hash'],'observation_group_id':c['observation_group_id'],'inference_input_tensor_hash':c['native']['input_tensor_hash'],'scene_mjb_sha256':c['scene_mjb']['sha256'],'saved_runtime_version':c['recorded_runtime_versions']['mujoco'],'ngeom':m.ngeom,'npair':m.npair,'nexclude':m.nexclude,'exclusions':list(map(list,sorted(exclusions))),'option_disableflags':int(m.opt.disableflags),'option_enableflags':int(m.opt.enableflags),'option_o_margin':float(m.opt.o_margin),'hand_geom_count':len(hand),'object_collision_geom_count':len(objects),'object_disabled_visual_geom_ids':visual,'geometry_records':records,'mesh_records':list(meshes.values()),'compatible_pairs':pairs,'object_asset_contract':contract,'physics_steps':0})
for p,h in B.items():check(sha(p)==h,'inputs unchanged')
report={'schema':'hug_actual_compiled_collision_contract_v1','passed':True,'checks':checks,'mujoco_version':mujoco.__version__,'mujoco_package_path':mujoco.__file__,'source_bindings':B,'cases':cases,'actual_static_forward_calls':6,'physics_steps':0,'new_hand_object_distance_queries':0,'new_eval_data':False,'scope':'Actual compiled scene, mesh support hull, body/geom transform, filter and margin metadata. No simulation stepping or new proposal/trajectory evaluation.'}
q=O/'compiled_contract.json';assert not q.exists();q.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'checks':checks,'sha256':sha(q),'cases':[{'index':c['source_index'],'hand_geoms':c['hand_geom_count'],'object_geoms':c['object_collision_geom_count'],'npair':c['npair'],'nexclude':c['nexclude'],'margins':sorted(set(g['margin_m'] for g in c['geometry_records'])),'gaps':sorted(set(g['gap_m'] for g in c['geometry_records'])),'types':sorted(set(g['type_name'] for g in c['geometry_records'] if g['contype'] or g['conaffinity']))} for c in cases]}))
