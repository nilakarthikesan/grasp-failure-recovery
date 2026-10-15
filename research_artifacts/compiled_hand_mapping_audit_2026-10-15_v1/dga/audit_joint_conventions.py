from pathlib import Path
exec(Path('/private/tmp/compiled_hand_audit_20261015_v1/dga/audit_dga_frames.py').read_text().split('ledgerp=B/')[0])
D=record(OUT/'target_cases_v1.json');S=Tree(D['source_hand']['path']);T=Tree(D['target_hand']['path']);names=D['joint_names']
qs=[('zero_reference',dict.fromkeys(names,0.0))]+[(x['case_id'],x['target_joint_targets_rad']) for x in D['cases'] if x['variant']=='raw_native_palm_compensated_diagnostic']
rows=[]
for case,q in qs:
 sf,tf=S.fk(q),T.fk(q);si,ti=np.linalg.inv(sf['palm']),np.linalg.inv(tf['palm']);details=[]
 for n in names:
  sj,tj=S.j[n],T.j[n]
  a=si@sf[sj['parent']]@sj['origin'];b=ti@tf[tj['parent']]@tj['origin']
  sa=a[:3,:3]@(sj['axis']/np.linalg.norm(sj['axis']));ta=b[:3,:3]@(tj['axis']/np.linalg.norm(tj['axis']))
  delta=b[:3,3]-a[:3,3];cross=np.cross(sa,ta)
  details.append({'joint':n,'palm_frame_axis_difference_rad':float(np.arctan2(np.linalg.norm(cross),sa@ta)),'joint_origin_displacement_m':float(np.linalg.norm(delta)),'joint_axis_perpendicular_origin_displacement_m':float(np.linalg.norm(delta-sa*(delta@sa))),'local_frame_rotation_difference_rad':float(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())})
 rows.append({'case':case,'joints':details})
report={'schema_version':'dga_source_target_joint_coordinate_conventions_v1','case_export':ref(OUT/'target_cases_v1.json'),'independent_script':ref(Path(__file__)),'rows':rows,'scope':'Local frame orientation differences and renamed local rotation axes alone are not mapping bugs. Compare spatial joint axes after palm alignment. This is geometric description, not a calibration or new optimizer.','thumb_max_spatial_axis_difference_rad':max(j['palm_frame_axis_difference_rad'] for r in rows for j in r['joints'] if j['joint'].startswith('TH')),'little_max_spatial_axis_difference_rad':max(j['palm_frame_axis_difference_rad'] for r in rows for j in r['joints'] if j['joint'].startswith('LF')),'zero_pose_thumb_origin_displacement_m':{j['joint']:j['joint_origin_displacement_m'] for j in rows[0]['joints'] if j['joint'].startswith('TH')},'strict_acceptance':False,'physics_steps':0}
(OUT/'joint_conventions_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['rows']}))
