"""Independent saved-array / source-URDF radial-shell arithmetic review.

No simulator, geometry library, object-to-hand distance query or new witness.
70-digit results are checked with 90-digit Decimal working precision.
"""
from pathlib import Path
from decimal import Decimal, getcontext
from itertools import product
import hashlib,json,pickle,xml.etree.ElementTree as ET
import numpy as np

B=Path(__file__).parent
getcontext().prec=90
Z=Decimal(0);ONE=Decimal(1);TWO=Decimal(2);CHECKS=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def dec(x):return Decimal.from_float(float(x))
def Dvec(a):return [dec(x) for x in a]
def norm(a):return sum(x*x for x in a).sqrt()
def dot(a,b):return sum(x*y for x,y in zip(a,b,strict=True))
def mv(a,b):return [dot(row,b) for row in a]
def transpose(a):return [list(row) for row in zip(*a,strict=True)]
def mm(a,b):return [[dot(row,col) for col in zip(*b,strict=True)] for row in a]
def ck(ok,message='review check failed'):
 global CHECKS
 CHECKS+=1
 if not ok:raise AssertionError(message)
def close(a,b,eps=Decimal('1e-14')):ck(abs(a-dec(b))<eps,(str(a),b))
def sincos(a):
 s=a;t=a;c=ONE;u=ONE
 for i in range(1,100):
  t=-t*a*a/Decimal((2*i)*(2*i+1));u=-u*a*a/Decimal((2*i-1)*(2*i));s+=t;c+=u
  if abs(t)<Decimal('1e-88') and abs(u)<Decimal('1e-88'):break
 else:raise AssertionError('Trig series did not converge')
 return s,c
def rpy_matrix(rpy):
 r,p,y=rpy;sr,cr=sincos(r);sp,cp=sincos(p);sy,cy=sincos(y)
 return mm(mm([[cy,-sy,Z],[sy,cy,Z],[Z,Z,ONE]],[[cp,Z,sp],[Z,ONE,Z],[-sp,Z,cp]]),[[ONE,Z,Z],[Z,cr,-sr],[Z,sr,cr]])
def quaternion_matrix(q):
 q=Dvec(q);length=norm(q);ck(abs(length-ONE)<Decimal('1e-14'));w,x,y,z=[v/length for v in q]
 return [[ONE-TWO*(y*y+z*z),TWO*(x*y-z*w),TWO*(x*z+y*w)],
 [TWO*(x*y+z*w),ONE-TWO*(x*x+z*z),TWO*(y*z-x*w)],
 [TWO*(x*z-y*w),TWO*(y*z+x*w),ONE-TWO*(x*x+y*y)]]
def anchor_box_minimum(local,size):
 outside=[max(abs(v)-s,Z) for v,s in zip(local,size,strict=True)]
 if any(v>Z for v in outside):return norm(outside)
 return min(s-abs(v) for v,s in zip(local,size,strict=True))
def floats(a):return np.asarray([[float(x) for x in r] for r in a])
def main():
 ck(not (B/'report.json').exists(),'No overwrite')
 p=read(B/'protocol.json');ck(sha(__file__)==p['review_script_sha256'])
 for path,h in p['input_sha256'].items():ck(sha(path)==h,path)
 authorp=read(p['author_protocol']);report=read(p['author_report']);ck(report['protocol_sha256']==sha(p['author_protocol']));ck(report['script_sha256']==sha(p['author_script']))
 for path,h in authorp['input_sha256'].items():ck(sha(path)==h,path)
 ck(authorp['case_indices']==[24,28]);ck(len(report['cases'])==2)
 ck(anchor_box_minimum([Z,Z,Z],[ONE,ONE,ONE])==ONE)
 ck(anchor_box_minimum([TWO,Z,Z],[ONE,ONE,ONE])==ONE)
 ck(anchor_box_minimum([ONE,Z,Z],[ONE,ONE,ONE])==Z)
 inv=read(authorp['inventory_path'])['modern_cases'];bound=read(authorp['bound_report_path']);bmap={x['source_inventory_index']:x for x in bound['cases']};saved={x['source_inventory_index']:x for x in read(authorp['frozen_anchor_candidate_path'])['cases']}
 urdf=ET.parse(p['urdf']).getroot();palm=next(x for x in urdf.findall('link') if x.get('name')=='palm');collisions=palm.findall('collision');ck(len(collisions)==8)
 boxes=[]
 for c in collisions:
  box=c.find('geometry').find('box');ck(box is not None)
  size=[Decimal(v)/TWO for v in box.get('size').split()];origin=c.find('origin');t=[Decimal(v) for v in origin.get('xyz').split()];R=rpy_matrix([Decimal(v) for v in origin.get('rpy').split()])
  local=mv(transpose(R),[-v for v in t]);inner=anchor_box_minimum(local,size)
  corners=[[x+y for x,y in zip(mv(R,[s*Decimal(sign) for s,sign in zip(size,signs,strict=True)]),t,strict=True)] for signs in product((-1,1),repeat=3)]
  outer=max(norm(v) for v in corners);boxes.append({'size':size,'position':t,'rotation':R,'anchor_local':local,'inner':inner,'outer':outer})
 rawmin=min(x['inner'] for x in boxes);rawmax=max(x['outer'] for x in boxes);pad=dec(authorp['shell_radius_arithmetic_padding_m']);rmin=max(Z,rawmin-pad);rmax=rawmax+pad
 fresh=read(p['prior_independent_cell_geometry_report']);ck(fresh['result']=='PASS');ck(fresh['source_inventory_index']==28);freshcells={x['child_cell_index']:x for x in fresh['native_witness_cell_verification']}
 result=[];positive=[];max_radius_delta=Z;max_norm_delta=Z;max_margin_delta=Z
 for case_report in report['cases']:
  index=case_report['source_inventory_index'];case=inv[index];br=bmap[index];ck(case_report['native_sha256']==br['native_sha256']==case['native']['sha256']==saved[index]['native_sha256'])
  ck(len(br['radius_geometry_records'])==len(case_report['radius_geometry_records'])==8)
  for i,(g,ag,box) in enumerate(zip(br['radius_geometry_records'],case_report['radius_geometry_records'],boxes,strict=True)):
   ck(g['geom_id']==ag['geom_id']==8+i and g['body']=='palm' and g['kind']==6)
   ck(g['contype']==1 and g['conaffinity']==1)
   for j in range(3):close(box['size'][j],g['size'][j]);close(box['position'][j],g['geom_local_position_m'][j]);close(box['anchor_local'][j],ag['anchor_box_local_m'][j])
   qm=quaternion_matrix(g['geom_local_quaternion_wxyz']);ck(np.max(np.abs(floats(qm)-floats(box['rotation'])))<1e-14)
   close(box['inner'],ag['unsigned_anchor_to_box_surface_m']);close(box['outer'],ag['all_corner_maximum_radius_m']);max_radius_delta=max(max_radius_delta,abs(box['outer']-dec(ag['all_corner_maximum_radius_m'])))
  close(rawmin,case_report['raw_inner_radius_m']);close(rawmax,case_report['raw_outer_radius_m']);close(rmin,case_report['inner_radius_lower_m']);close(rmax,case_report['outer_radius_upper_m'])
  contract=read(case['frame_files']['scene_contract.json']['path']);objpos=Dvec(contract['object_position_m']);xyzw=contract['object_orientation_xyzw'];oR=quaternion_matrix([xyzw[3],*xyzw[:3]])
  with Path(case['native']['path']).open('rb') as f:native=pickle.load(f)
  T=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);N=np.asarray(native['T_camera_wrist']);ck(T.shape==N.shape==(4,4))
  anchor=[sum(dec(T[i,k])*dec(N[k,3]) for k in range(4)) for i in range(3)]
  for j in range(3):close(anchor[j],saved[index]['target_palm_world_position_m'][j]);close(anchor[j],case_report['fixed_palm_anchor_world_m'][j]);close(objpos[j],case_report['object_position_world_m'][j])
  anchor_object=mv(transpose(oR),[a-b for a,b in zip(anchor,objpos,strict=True)])
  for i in range(3):
   close(anchor_object[i],case_report['fixed_anchor_object_m'][i])
   for j in range(3):close(oR[i][j],case_report['object_rotation_world_from_object'][i][j])
  witnesses=[];ck(len(case_report['witnesses'])==len(br['existing_witnesses']))
  for ar,w in zip(case_report['witnesses'],br['existing_witnesses'],strict=True):
   ck(ar['source_witness']==w);ck(w['anatomical_part']=='palm' and w['relationship_change']=='lost_native' and w['cell_area_mm2']>0)
   cache=Path(p['query_cache_root'])/(w['case_id']+'_density'+str(w['density_mm'])+'mm_queries.npz');ck(sha(cache)==w['original_query_cache_sha256'])
   with np.load(cache,allow_pickle=False) as z:
    ci=w['child_cell_index'];ck(np.array_equal(z['query_object_m'][ci],np.asarray(w['centroid_object_m'])));ck(float(z['area_mm2'][ci])==w['cell_area_mm2']);close(dec(z['native_distance_by_part_m'][ci,0]),w['native_centroid_distance_m'],Decimal('1e-12'))
   radius=dec(w['centroid_radius_m']);band=dec(w['band_mm'])/Decimal(1000);distance_eps=max(dec(w['numerical_allowance_m']),dec(authorp['inherited_distance_allowance_m']));extra=dec(authorp['radial_transform_arithmetic_allowance_m']);native_upper=dec(w['native_centroid_distance_m'])+radius+distance_eps
   near=native_upper<=band;ck(near==ar['native_definite_near_100nm']);ck(near)
   rho=norm([a-b for a,b in zip(Dvec(w['centroid_object_m']),anchor_object,strict=True)]);wholelo=max(Z,rho-radius);wholehi=rho+radius
   inner=rmin-wholehi-band-distance_eps-extra;outer=wholelo-rmax-band-distance_eps-extra
   close(rho,ar['centroid_anchor_radial_norm_m']);close(wholelo,ar['whole_cell_radial_interval_m'][0]);close(wholehi,ar['whole_cell_radial_interval_m'][1]);close(inner,ar['inner_shell_strict_margin_m']);close(outer,ar['outer_shell_strict_margin_m'])
   contradiction=bool(near and max(inner,outer)>0);ck(contradiction==ar['all_SO3_orientation_contradiction_conditional']);max_norm_delta=max(max_norm_delta,abs(rho-dec(ar['centroid_anchor_radial_norm_m'])));max_margin_delta=max(max_margin_delta,abs(outer-dec(ar['outer_shell_strict_margin_m'])),abs(inner-dec(ar['inner_shell_strict_margin_m'])))
   priorcell=False
   if index==28:
    fc=freshcells[ci];close(radius,fc['cell_radius_reconstructed_m']);close(dec(w['cell_area_mm2']),fc['cell_area_mm2']);close(dec(w['native_centroid_distance_m']),fc['native_distance_independent_m'],Decimal('1e-12'));priorcell=True
   row={'child_cell_index':ci,'band_mm':w['band_mm'],'cell_area_mm2':w['cell_area_mm2'],'native_upper_90digit_m':str(native_upper),'rho_90digit_m':str(rho),'whole_cell_radial_lower_90digit_m':str(wholelo),'whole_cell_radial_upper_90digit_m':str(wholehi),'inner_margin_90digit_m':str(inner),'outer_margin_90digit_m':str(outer),'contradiction':contradiction,'positive_area_whole_cell_certificate_not_centroid_only':contradiction,'radius_and_native_geometry_independently_reconstructed_in_prior_bound_audit':priorcell}
   witnesses.append(row)
   if contradiction:positive.append({'source_inventory_index':index,**row})
  want='FIXED_ANCHOR_ALL_SO3_PALM_PROXIMITY_REFUTED_CONDITIONAL' if any(x['contradiction'] for x in witnesses) else 'NO_POSITIVE_RADIAL_WITNESS';ck(want==case_report['status']);result.append({'source_inventory_index':index,'status':want,'source_native_anchor_and_object_transform_recomputed':True,'all_eight_source_URDF_box_radii_and_inner_distances_recomputed':True,'witnesses':witnesses})
 for path,h in p['input_sha256'].items():ck(sha(path)==h,path)
 ck(sha(__file__)==p['review_script_sha256']);ck(len(positive)==1 and positive[0]['source_inventory_index']==28)
 output={'schema':p['schema'],'result':'PASS','review_protocol_sha256':sha(B/'protocol.json'),'review_script_sha256':sha(__file__),'reviewed_protocol_sha256':sha(p['author_protocol']),'reviewed_report_sha256':sha(p['author_report']),'check_count':CHECKS,'arithmetic_working_precision_digits':90,'independent_decimal_source_URDF_trig_and_normalized_quaternion_transforms':True,'independent_source_URDF_raw_inner_radius_m':str(rawmin),'independent_source_URDF_raw_outer_radius_m':str(rawmax),'maximum_box_radius_difference_m':str(max_radius_delta),'maximum_radial_norm_difference_m':str(max_norm_delta),'maximum_margin_difference_m':str(max_margin_delta),'cases':result,'positive_whole_cell_witnesses':positive,'new_object_to_hand_geometry_queries':0,'physics_steps':0,'new_inference':False,'new_query_sites':False,'new_eval_data':False,'training':False,'scope':'Conditional all-SO3 palm orientation obstruction for saved Mug B at fixed native wrist-to-palm anchor, exact same8boxes; not arbitrary translations, different geometries, other proposals or failure-label/predictor adequacy. Mug A is not decided by these radial controls.','not_formal_floating_point_interval_proof':True}
 (B/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(json.dumps({'result':'PASS','check_count':CHECKS,'positive_outer_margin_m':positive[0]['outer_margin_90digit_m'],'max_margin_difference_m':str(max_margin_delta),'report_sha256':sha(B/'report.json')},indent=2))
if __name__=='__main__':main()
