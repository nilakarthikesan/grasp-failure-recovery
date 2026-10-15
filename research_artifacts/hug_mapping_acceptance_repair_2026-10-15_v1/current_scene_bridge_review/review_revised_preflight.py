"""Read-only source/command/hash review; no simulator imports or executions."""
from pathlib import Path
import json,hashlib,ast,sys
import numpy as np
H=Path(__file__).resolve().parent
mode=sys.argv[1];P=Path(sys.argv[2]);d=json.loads((P/'protocol.draft.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks=[]
def check(n,v):
 if not v:raise AssertionError(n)
 checks.append(n)
for path,digest in d['bindings'].items():check('bound '+path,sha(path)==digest)
for path in P.glob('*.py'):ast.parse(path.read_text());check('syntax '+path.name,True)
check('no actual output',not (P/'results.json').exists() and not (P/'cases').exists())
if mode=='scene':
 check('32 frozen cases',len(d['cases'])==d['case_count']==32);check('indices',[c['source_inventory_index'] for c in d['cases']]==list(range(32)))
 for c in d['cases']:
  with np.load(c['command_path'],allow_pickle=False) as z:
   i=c['row_index'];check('names',z['joint_names'].tolist()==c['joint_names'])
   for frozen,stored in [('qpos_rad','nominal_target_rad'),('root_position_m','root_target_position_m'),('root_quaternion_wxyz','root_target_quaternion_wxyz')]:check('literal saved '+frozen,np.array_equal(c[frozen],z[stored][i]))
  for path,h in c['input_bindings'].items():check('case closure '+path,d['bindings'][path]==h)
 check('budget128',d['finite_budget']['maximum_total_forward_calls']==128)
 check('16 controls', 'Ran 16 tests' in (P/'controls_final.log').read_text() and (P/'controls_final.log').read_text().strip().endswith('OK'))
else:
 check('8 frozen poses',len(d['poses'])==d['finite_budget']['poses']==8);check('365 rows',sum(c['proximity_count']+c['orientation_count'] for c in d['poses'])==d['finite_budget']['retained_row_evaluations']==365)
 check('12 controls','Ran 12 tests' in (P/'manufactured_controls.log').read_text() and (P/'manufactured_controls.log').read_text().strip().endswith('OK'))
check('bound registry YAML',all(any(path.endswith('/'+n) for path in d['bindings']) for n in ['environments.yaml','execution_protocols.yaml']))
out={'schema':'hug_'+mode+'_preexecution_source_identity_review_v1','passed':True,'protocol_draft_sha256':sha(P/'protocol.draft.json'),'bindings_count':len(d['bindings']),'checks':checks,'check_count':len(checks),'source_bindings':{str(P/'protocol.draft.json'):sha(P/'protocol.draft.json'),str(Path(__file__)):sha(__file__),**d['bindings']},'reviewer_new_real_geometry_queries':0,'reviewer_FK_calls':0,'new_eval_data':False}
target=Path(sys.argv[3]);assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(target),'sha256':sha(target),'checks':len(checks)}))
