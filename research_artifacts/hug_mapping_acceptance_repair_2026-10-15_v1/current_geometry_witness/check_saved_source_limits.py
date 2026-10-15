"""Post-run pure XML readback of the eight already evaluated q vectors."""
from pathlib import Path
import sys,json,hashlib,numpy as np
sys.path.insert(0,'/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
from xml_reference import read_urdf,source_limit_rows
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((HERE/'protocol.json').read_text());source=read_urdf(p['source_urdf'])
assert sha(p['source_urdf'])==p['bindings'][str(Path(p['source_urdf']).resolve())]
rows=[]
for pose in p['poses']:
 record=json.loads((HERE/'cases'/(pose['pose_id']+'.json')).read_text())
 assert np.array_equal(record['frozen_pose']['qpos_rad'],pose['qpos_rad'])
 limits=source_limit_rows(source,pose['joint_names'],pose['qpos_rad'])
 passed=all(r['literal_pass'] for r in limits)
 rows.append({'pose_id':pose['pose_id'],'role':pose['role'],'joint_count':len(limits),'all_literal_source_limits_pass':passed,'joint_rows':limits})
report={'schema':'hug_current_witness_literal_source_limit_readback_v1','protocol_sha256':sha(HERE/'protocol.json'),
        'source_urdf_sha256':sha(p['source_urdf']),'xml_helper_sha256':sha('/private/tmp/hug_current_execution_bridge_2026-10-15_v1/xml_reference.py'),
        'check_script_sha256':sha(__file__),'post_run_saved_readback':True,'poses':rows,'all_poses_pass':all(r['all_literal_source_limits_pass'] for r in rows),
        'FK_calls':0,'geometry_queries':0,'physics_steps':0,'acceptance_tolerance_rad':0.0,'saved_targets_changed':False}
out=HERE/'source_limit_readback.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for r in rows:print(json.dumps({'pose':r['pose_id'],'passed':r['all_literal_source_limits_pass'],'refused_joints':[x for x in r['joint_rows'] if not x['literal_pass']]}))
print('sha256='+sha(out))
