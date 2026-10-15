"""Read-only pre-execution source/command closure checks; no MuJoCo import."""
from pathlib import Path
import hashlib,json,ast,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=ROOT/'protocol.draft.json'; d=json.loads(p.read_text()); checks=[]
def check(name,condition):
    if not condition: raise AssertionError(name)
    checks.append(name)
check('declared 32 saved cases',d['case_count']==len(d['cases'])==32)
check('exact source indices', [c['source_inventory_index'] for c in d['cases']]==list(range(32)))
check('unique natives',len({c['native_sha256'] for c in d['cases']})==32)
for path,digest in d['bindings'].items():check('hash '+path,sha(path)==digest)
for c in d['cases']:
    i=c['source_inventory_index']
    with np.load(c['command_path'],allow_pickle=False) as z:
        j=c['row_index']
        check(f'case{i}: names',z['joint_names'].tolist()==c['joint_names'])
        for key,saved in [('qpos_rad','nominal_target_rad'),('root_position_m','root_target_position_m'),('root_quaternion_wxyz','root_target_quaternion_wxyz')]:
            check(f'case{i}: {key}',np.array_equal(np.asarray(c[key]),z[saved][j]))
    check(f'case{i}: 24 finite unique names',len(c['joint_names'])==len(set(c['joint_names']))==24 and np.isfinite(c['qpos_rad']).all())
    check(f'case{i}: native closure',c['native_sha256'] in c['input_bindings'].values())
    for path,h in c['input_bindings'].items():check(f'case{i}: closure subset {path}',d['bindings'][path]==h)
check('96 static forwards',d['finite_budget']['maximum_total_forward_calls']==96)
for name in ['physics_steps','retargeting_solves','hand_object_distance_queries']:check('zero '+name,d['finite_budget'][name]==0)
for name in ['run_bridge.py','bridge_model.py','xml_reference.py','prepare_protocol.py','test_bridge_reference.py']:
    ast.parse((ROOT/name).read_text());check('syntax '+name,True)
check('no previous real output',(not (ROOT/'results.json').exists()) and not (ROOT/'cases').exists())
out={'schema':'hug_current_execution_bridge_source_input_review_v1','passed':True,'protocol_draft_sha256':sha(p),'binding_count':len(d['bindings']),'check_count':len(checks),'checks':checks,'source_bindings':{str(p):sha(p),str(Path(__file__)):sha(__file__),**d['bindings']},'real_FK_calls':0,'physics_steps':0,'new_eval_data':False,'scope':'Immutable source and saved command identity checks plus static code parsing; no real geometry or model execution.'}
target=HERE/'source_input_review.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(target),'sha256':sha(target),'checks':len(checks)}))
