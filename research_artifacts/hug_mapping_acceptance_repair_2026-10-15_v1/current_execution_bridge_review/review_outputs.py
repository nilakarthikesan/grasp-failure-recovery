"""Independent saved-output arithmetic/XML-FK replay; imports no MuJoCo."""
from pathlib import Path
import hashlib,json
from xml.etree import ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;P=Path('/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
proto=json.loads((P/'protocol.json').read_text());summary=json.loads((P/'results.json').read_text());checks=[]
def check(n,c):
    if not c:raise AssertionError(n)
    checks.append(n)
def close(n,a,b,tol=1e-14):check(n,np.isfinite(a).all() and np.isfinite(b).all() and np.max(np.abs(np.asarray(a)-np.asarray(b)))<=tol)
check('protocol linkage',summary['protocol_sha256']==sha(P/'protocol.json'))
for path,h in proto['bindings'].items():check('closure '+path,sha(path)==h)
xml=ET.parse(proto['source_urdf']).getroot();joints=[]
for node in xml.findall('joint'):
    org=node.find('origin');axis=node.find('axis');lim=node.find('limit')
    xyz=np.fromstring(org.get('xyz','0 0 0'),sep=' ') if org is not None else np.zeros(3)
    rpy=np.fromstring(org.get('rpy','0 0 0'),sep=' ') if org is not None else np.zeros(3)
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();T[:3,3]=xyz
    joints.append({'name':node.get('name'),'parent':node.find('parent').get('link'),'child':node.find('child').get('link'),'type':node.get('type'),'origin':T,'axis':np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0,0]),'limits':None if lim is None else np.array([float(lim.get('lower')),float(lim.get('upper'))])})
source_limits={j['name']:j['limits'] for j in joints if j['type']=='revolute'}
def frames(command,floating):
    q=dict(zip(command['joint_names'],command['qpos_rad'],strict=True));f={'world':np.eye(4)};remaining=list(joints)
    while remaining:
        prior=len(remaining)
        for j in remaining[:]:
            if j['parent'] not in f:continue
            T=f[j['parent']]@j['origin']
            if floating and j['name']=='world_joint':
                T=np.eye(4);qw=np.asarray(command['root_quaternion_wxyz']);T[:3,:3]=Rotation.from_quat(qw[[1,2,3,0]]).as_matrix();T[:3,3]=command['root_position_m']
            elif j['type']=='revolute':
                R=np.eye(4);a=j['axis'];R[:3,:3]=Rotation.from_rotvec(a/np.linalg.norm(a)*q[j['name']]).as_matrix();T=T@R
            f[j['child']]=T;remaining.remove(j)
        check('XML tree progress',len(remaining)<prior)
    return f
aggregate=[];max_fk=0.;violations=[];historical_geom_failures=[]
for brief in summary['cases']:
    path=Path(brief['path']);check('case hash '+str(brief['index']),sha(path)==brief['sha256']);c=json.loads(path.read_text());i=c['source_inventory_index'];frozen=proto['cases'][i]
    check(f'{i}: literal command',c['literal_stored_command']==frozen);check(f'{i}: native',c['native_sha256']==frozen['native_sha256'])
    fw=frames(frozen,True);ff=frames(frozen,False)
    for field,expected in [('current_fixed_body_and_virtual_links',ff),('current_world_body_and_virtual_links',fw)]:
        values=c[field]
        for row in values['compiled_body_rows']:
            T=expected[row['link']];close(f'{i}:{field}:{row["link"]}: XML p',row['xml_world_position_m'],T[:3,3]);close(f'{i}:{field}:{row["link"]}: XML R',row['xml_world_rotation'],T[:3,:3])
            pe=np.linalg.norm(np.asarray(row['compiled_world_position_m'])-T[:3,3]);re=np.max(np.abs(np.asarray(row['compiled_world_rotation'])-T[:3,:3]));max_fk=max(max_fk,float(pe))
            close(f'{i}:{field}: reported p residual',pe,row['position_error_m']);close(f'{i}:{field}: reported R residual',re,row['rotation_matrix_max_error'])
            check(f'{i}:{field}: pass',row['passed']==bool(pe<=1e-10 and re<=1e-10))
        check(f'{i}:{field}: aggregate',values['passed']==all(r['passed'] for r in values['compiled_body_rows']+values['fixed_virtual_link_rows']))
    xml_world=np.array([fw[n][:3,3] for n in c['task_links']]);close(f'{i}: task XML',xml_world,c['xml_world_task_positions_m'])
    err=np.max(np.linalg.norm(np.asarray(c['current_world_task_positions_m'])-xml_world,axis=1));close(f'{i}: task error',err,c['current_vs_xml_max_task_position_error_m'])
    hist=np.max(np.linalg.norm(np.asarray(c['historical_replayed_task_positions_m'])-np.asarray(c['immutable_historical_task_positions_m']),axis=1));close(f'{i}: history error',hist,c['historical_replay_vs_immutable_max_error_m']);check(f'{i}: historical replay gate',c['historical_replay_matches']==bool(hist<=1e-10))
    drift=np.max(np.linalg.norm(np.asarray(c['current_world_task_positions_m'])-np.asarray(c['historical_replayed_task_positions_m']),axis=1));close(f'{i}: current-old error',drift,c['current_vs_historical_max_task_position_change_m'])
    q=dict(zip(frozen['joint_names'],frozen['qpos_rad'],strict=True));limit_pass=[]
    for row in c['literal_source_joint_limit_rows']:
        name=row['joint_name'];lo,hi=source_limits[name];v=q[name];passed=bool(lo<=v<=hi);excess=float(max(lo-v,v-hi,0.));close(f'{i}:{name}: source limits',row['source_limits_rad'],[lo,hi],0.);check(f'{i}:{name}: exact bound',row['literal_pass']==passed);close(f'{i}:{name}: excess',row['violation_rad'],excess,0.);limit_pass.append(passed)
        if not passed:violations.append({'index':i,'joint':name,'violation_rad':excess,'value':v,'source_limits':[lo,hi]})
    check(f'{i}: limit aggregate',c['all_literal_source_limits_pass']==all(limit_pass))
    for key in ['current_joint_metadata','current_fixed_joint_metadata']:
        check(f'{i}:{key}: joint set',{r['name'] for r in c[key]}==set(source_limits))
        for row in c[key]:
            ex=float(np.max(np.abs(np.asarray(row['compiled_limits_rad'])-source_limits[row['name']])));close(f'{i}:{key}: range residual',ex,row['limit_max_difference_rad']);check(f'{i}:{key}: enabled range gate',row['source_compiled_limits_match']==bool(row['limited'] and ex<=1e-12))
    for key in ['current_fixed_collision_geometry','current_world_collision_geometry','historical_geometry_against_current_source_reference']:
        g=c[key];geom_pass=[]
        for row in g['geometry_rows']:
            if not row['shape_type_matches']:passed=False
            elif row['source_kind']=='mesh':passed=bool(row['symmetric_source_compiled_vertex_set_error_m']<=1e-6 and row['mesh']['graph_contains_all_full_hull_extreme_vertex_indices'])
            else:passed=bool(row['position_error_m']<=1e-10 and row['rotation_matrix_max_error']<=1e-10 and row['size_error_m']<=1e-12)
            if row['shape_type_matches']:
                contact=row['contype']==row['conaffinity']==1 and row['margin_m']==row['gap_m']==0.;check(f'{i}:{key}: contact gate',row['default_contact_contract_matches']==contact);passed=passed and contact
            check(f'{i}:{key}: geom gate',row['passed']==passed);geom_pass.append(passed)
            if key=='historical_geometry_against_current_source_reference' and not passed:historical_geom_failures.append({'index':i,'source_link':row['source_link'],'kind':row['source_kind'],'vertex_error_m':row.get('symmetric_source_compiled_vertex_set_error_m'),'contact_match':row.get('default_contact_contract_matches'),'position_error_m':row.get('position_error_m')})
        inventory=set(g['actual_hand_geom_ids'])==set(g['expected_owner_geom_ids']);check(f'{i}:{key}: inventory',g['complete_hand_inventory_matches']==inventory)
        full=inventory and (g['current_hand_model_has_no_extra_pair_plugin_flex_or_exclusion'] is not False) and len(g['geometry_rows'])==g['source_collision_count'] and all(geom_pass) and all(r['passed'] for r in g['count_rows']);check(f'{i}:{key}: aggregate',g['passed']==full)
    t=proto['engineering_tolerances'];passed=bool(c['current_named_joint_command_roundtrip_exact'] and c['current_fixed_body_and_virtual_links']['passed'] and c['current_world_body_and_virtual_links']['passed'] and c['current_fixed_collision_geometry']['passed'] and c['current_world_collision_geometry']['passed'] and all(r['source_compiled_limits_match'] for r in c['current_joint_metadata']+c['current_fixed_joint_metadata']) and c['current_vs_xml_max_task_position_error_m']<=t['current_position_m'] and c['current_fixed_vs_xml_max_task_position_error_m']<=t['current_position_m'] and c['current_carrier_position_error_m']<=t['current_position_m'] and c['current_carrier_rotation_matrix_error']<=t['current_rotation_matrix'])
    check(f'{i}: command gate',c['command_realization_pass']==passed);check(f'{i}: joint overall',c['command_within_source_limits_and_realized']==bool(passed and all(limit_pass)))
    aggregate.append({'index':i,'command_realization_pass':passed,'all_source_limits_pass':all(limit_pass),'historical_replay_pass':c['historical_replay_matches'],'current_vs_historical_max_position_change_m':drift})
check('all 32 output identities',[r['index'] for r in aggregate]==list(range(32)))
check('96 actual forwards',summary['actual_static_forward_calls']==96)
check('zero steps',summary['physics_steps']==0)
check('summary commands',summary['all_current_command_realizations_pass']==all(c['command_realization_pass'] for c in aggregate))
check('summary source limits',summary['all_literal_source_limits_pass']==all(c['all_source_limits_pass'] for c in aggregate))
check('summary history',summary['all_historical_static_replays_match']==all(c['historical_replay_pass'] for c in aggregate))
receipt={'schema':'hug_current_execution_bridge_saved_output_review_v1','passed':True,'review_scope':'Source identity, saved array arithmetic, independent XML/SciPy FK against saved compiled body frames, source limit predicates and result aggregation. No new compiled model or real geometry execution. Collider geometric residuals are replayed as stored scalars; actual vertices were not retained and those queries were not independently repeated.','source_bindings':{str(P/'protocol.json'):sha(P/'protocol.json'),str(P/'results.json'):sha(P/'results.json'),str(Path(__file__)):sha(__file__),**{r['path']:r['sha256'] for r in summary['cases']}},'arithmetic_and_identity_check_count':len(checks),'checks':checks,'cases':aggregate,'max_independent_compiled_body_position_error_m':max_fk,'literal_source_limit_violations':violations,'historical_geometry_failed_rows':historical_geom_failures,'new_real_geometry_queries':0,'new_mj_forward_calls':0,'new_eval_data':False,'does_not_establish_full_scene_dynamics_native_fidelity_new_profile_or_generalization':True}
out=H/'output_review.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'checks':len(checks),'literal_source_violations':len(violations),'max_independent_FK_error_m':max_fk}))
