"""Post-hoc SO(3) radial-shell check on seven existing HUG palm cells.
No MuJoCo, geometry query, inference, fitting, integration or new cell sampling.
"""
from pathlib import Path
from datetime import datetime,timezone
from itertools import product
import hashlib,json,pickle
import numpy as np
B=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quat_rotation(q):
 q=np.asarray(q,float);assert q.shape==(4,) and np.isfinite(q).all() and abs(q@q-1)<1e-14
 w,x,y,z=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def box_surface_distance(point,size):
 delta=np.abs(np.asarray(point,float))-np.asarray(size,float)
 return abs(float(np.linalg.norm(np.maximum(delta,0))+min(float(delta.max()),0.)))
def main():
 assert not (B/'report.json').exists(),'No overwrites'
 p=json.loads((B/'protocol.json').read_text());assert sha(__file__)==p['script_sha256']
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 for point,want in (([0,0,0],1.),([2,0,0],1.),([1,0,0],0.)):
  assert box_surface_distance(point,[1.,1.,1.])==want
 assert 1.-.1-.1>0 and 2.-np.sqrt(3)-.05>0
 old=json.loads(Path(p['frozen_anchor_candidate_path']).read_text());oldmap={x['source_inventory_index']:x for x in old['cases']}
 bound=json.loads(Path(p['bound_report_path']).read_text());boundmap={x['source_inventory_index']:x for x in bound['cases']}
 inv=json.loads(Path(p['inventory_path']).read_text())['modern_cases'];out=[]
 for index in p['case_indices']:
  case=inv[index];b=boundmap[index];saved=oldmap[index];assert saved['native_sha256']==b['native_sha256']==case['native']['sha256']
  anchor=np.asarray(saved['target_palm_world_position_m']);contract=json.loads(Path(case['frame_files']['scene_contract.json']['path']).read_text())
  objpos=np.asarray(contract['object_position_m']);xyzw=np.asarray(contract['object_orientation_xyzw']);oR=quat_rotation(np.r_[xyzw[3],xyzw[:3]])
  with Path(case['native']['path']).open('rb') as f:native=pickle.load(f)
  T=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);Tw=T@np.asarray(native['T_camera_wrist'])
  assert np.max(np.abs(anchor-Tw[:3,3]))<1e-12
  anchor_object=(anchor-objpos)@oR
  geometry=[];inner=[];outer=[]
  for g in b['radius_geometry_records']:
   assert g['kind']==6 and g['body']=='palm';s=np.asarray(g['size']);R=quat_rotation(g['geom_local_quaternion_wxyz']);t=np.asarray(g['geom_local_position_m'])
   assert s.shape==(3,) and np.all(s>0)
   a_local=(-t)@R;d=box_surface_distance(a_local,s)
   corners=(np.asarray(list(product((-1.,1.),repeat=3)))*s)@R.T+t;radius=float(np.linalg.norm(corners,axis=1).max())
   inner.append(d);outer.append(radius);geometry.append({'geom_id':g['geom_id'],'anchor_box_local_m':a_local.tolist(),'unsigned_anchor_to_box_surface_m':d,'all_corner_maximum_radius_m':radius})
  rawmin=min(inner);rawmax=max(outer);rmin=max(0.,rawmin-p['shell_radius_arithmetic_padding_m']);rmax=rawmax+p['shell_radius_arithmetic_padding_m']
  witnesses=[]
  for x in b['existing_witnesses']:
   assert x['anatomical_part']=='palm' and x['relationship_change']=='lost_native' and x['cell_area_mm2']>0
   z=np.asarray(x['centroid_object_m']);rho=float(np.linalg.norm(z-anchor_object));r=x['centroid_radius_m'];band=x['band_mm']/1000;nativeeps=max(x['numerical_allowance_m'],p['inherited_distance_allowance_m'])
   near=x['native_centroid_distance_m']+r+nativeeps<=band
   eps=p['inherited_distance_allowance_m']+p['radial_transform_arithmetic_allowance_m']
   inner_margin=rmin-rho-r-band-eps;outer_margin=rho-r-rmax-band-eps
   witnesses.append({'source_witness':x,'native_definite_near_100nm':bool(near),'centroid_anchor_radial_norm_m':rho,
    'whole_cell_radial_interval_m':[max(0.,rho-r),rho+r],'inner_shell_strict_margin_m':inner_margin,
    'outer_shell_strict_margin_m':outer_margin,'all_SO3_orientation_contradiction_conditional':bool(near and max(inner_margin,outer_margin)>0)})
  any_positive=any(x['all_SO3_orientation_contradiction_conditional'] for x in witnesses)
  out.append({'source_inventory_index':index,'native_sha256':case['native']['sha256'],'fixed_palm_anchor_world_m':anchor.tolist(),
   'object_position_world_m':objpos.tolist(),'object_rotation_world_from_object':oR.tolist(),'fixed_anchor_object_m':anchor_object.tolist(),
   'source_anchor_frame_crosscheck_maxabs_m':float(np.max(np.abs(anchor-Tw[:3,3]))),'radius_geometry_records':geometry,
   'raw_inner_radius_m':rawmin,'raw_outer_radius_m':rawmax,'inner_radius_lower_m':rmin,'outer_radius_upper_m':rmax,
   'witnesses':witnesses,'status':'FIXED_ANCHOR_ALL_SO3_PALM_PROXIMITY_REFUTED_CONDITIONAL' if any_positive else 'NO_POSITIVE_RADIAL_WITNESS',
   'no_claim_of_arbitrary_free_root_translation':True})
 for path,h in p['input_sha256'].items():assert sha(path)==h,path
 report={'schema':p['schema'],'protocol_sha256':sha(B/'protocol.json'),'script_sha256':sha(__file__),
  'cases':out,'manufactured_analytic_box_unsigned_distance_controls':3,'manufactured_positive_inner_outer_controls':2,
  'geometry_calls':0,'physics_steps':0,'new_inference':False,'new_eval_data':False,'new_query_sites':False,'training':False,
  'conditional_engineering_arithmetic_not_formally_directed_rounding':True,'limitations':p['limitations']}
 with (B/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'report':str(B/'report.json'),'report_sha256':sha(B/'report.json'),'case_statuses':[{ 'index':x['source_inventory_index'],'status':x['status'],'inner_radius_lower_m':x['inner_radius_lower_m'],'outer_radius_upper_m':x['outer_radius_upper_m'],'max_inner_margin_m':max(w['inner_shell_strict_margin_m'] for w in x['witnesses']),'max_outer_margin_m':max(w['outer_shell_strict_margin_m'] for w in x['witnesses'])} for x in out]}))
if __name__=='__main__':main()
