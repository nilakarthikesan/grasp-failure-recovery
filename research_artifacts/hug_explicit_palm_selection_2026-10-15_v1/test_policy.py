from pathlib import Path
import importlib.util,json,copy
path=Path(__file__).parent
spec=importlib.util.spec_from_file_location('explicit_selector',path/'select_explicit_proposal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base={'predeclared_translation_seed':True,'actual_shift_nonzero':True,'proposal_shift_nonzero':True,'literal_tip_feasible':True,'complete_full_callback':True,'objective':2.,'method_order':0,'callback_sequence':2,'proposal_sequence':1}
checks=[]
def check(value,label):
    assert value,label
    checks.append(label)
for flag in ('predeclared_translation_seed','actual_shift_nonzero','proposal_shift_nonzero','literal_tip_feasible','complete_full_callback'):
    row={**base,flag:False,'objective':0.}
    check(not m.qualifies(row),flag+' required independent of lower cost')
    check(m.select([row,base]) is base,flag+' excluded before cost rank')
later={**base,'method_order':1,'callback_sequence':1,'proposal_sequence':1}
check(m.select([later,base]) is base,'method tie order unchanged')
later={**base,'callback_sequence':3,'proposal_sequence':0}
check(m.select([later,base]) is base,'callback tie order unchanged')
later={**base,'proposal_sequence':2}
check(m.select([later,base]) is base,'earliest explicit proposal breaks provenance tie')
lower={**base,'objective':1.}
check(m.select([base,lower]) is lower,'objective primary among eligible')
with (path/'synthetic_policy_checks.json').open('x') as f:json.dump({'passed':True,'count':len(checks),'checks':checks,'only_manufactured_inputs':True,'new_eval_data':False},f,indent=2);f.write('\n')
print(len(checks),'synthetic policy checks PASS')
