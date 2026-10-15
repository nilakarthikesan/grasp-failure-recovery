"""Read-only arithmetic and deliberately tampered trace checks; no geometry."""
from pathlib import Path
import copy,hashlib,importlib.util,json,xml.etree.ElementTree as ET

OUT=Path(__file__).parent
TARGET=OUT.parent/'seed_stage/candidate_targets.json'
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('selection_audit_static_tests',OUT/'selection_audit.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    protocol=json.loads((OUT/'protocol.DRAFT.json').read_text())
    candidate=json.loads(TARGET.read_text())
    urdf=ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'
    ranges={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
    checks=[]
    for case in candidate['cases']:
        mod.verify_recorded_selection(case,ranges,protocol)
        checks.append({'name':f"authentic_case_{case['source_inventory_index']}",'expected':'accept recorded trace','passed':True})
    def checked_mutation(name,change):
        case=copy.deepcopy(candidate['cases'][0]);change(case)
        try:mod.verify_recorded_selection(case,ranges,protocol)
        except (AssertionError,ValueError,TypeError,IndexError,KeyError) as exc:
            checks.append({'name':name,'expected':'reject modified trace','passed':True,'exception_type':type(exc).__name__})
        else:raise AssertionError(f'Tampered record accepted: {name}')
    checked_mutation('returned_q_disagrees_with_selected',lambda c:c['qpos_rad'].__setitem__(0,c['qpos_rad'][0]+1e-5))
    checked_mutation('seed_disagrees_with_declared_old_raw',lambda c:c['predeclared_seed_qpos_rad'].__setitem__(0,c['predeclared_seed_qpos_rad'][0]+1e-5))
    checked_mutation('seed_outside_source_and_trust_bounds',lambda c:c['solver_audit']['predeclared_seed_qpos_rad'][0].__setitem__(0,1e6))
    checked_mutation('reported_tip_norm_inconsistent',lambda c:c['solver_audit']['final']['tip_error_norm_m'].__setitem__(0,c['solver_audit']['final']['tip_error_norm_m'][0]+1e-5))
    checked_mutation('tip_eligibility_boolean_inconsistent',lambda c:c['solver_audit']['proposals'][0]['full_assessment']['individual_tip_nonworsening_pass'].__setitem__(0,False))
    checked_mutation('full_budget_counter_exceeded',lambda c:c['solver_audit']['counters'].__setitem__('full',5))
    checked_mutation('accepted_objective_inconsistent',lambda c:c['solver_audit']['accepted_candidates'][0].__setitem__('objective',0.))
    checked_mutation('trace_event_order_inconsistent',lambda c:c['solver_audit']['events'][0].__setitem__('sequence',2))
    def lose_fresh_full(c):
        last=[e for e in c['solver_audit']['events'] if e['kind']=='full_callback'][-1];last['fresh']=False
    checked_mutation('terminal_full_callback_not_fresh',lose_fresh_full)
    def lose_fresh_fk(c):
        last=[e for e in c['solver_audit']['events'] if e['kind']=='FK_callback'][-1];last['terminal']=False
    checked_mutation('terminal_fk_callback_not_terminal',lose_fresh_fk)
    report={'schema':'hug_feasible_seed_selection_arithmetic_tamper_tests_v2','candidate_sha256':sha(TARGET),'script_sha256':sha(__file__),
        'selection_script_sha256':sha(OUT/'selection_audit.py'),'tests':checks,'test_count':len(checks),'all_passed':True,
        'geometry_calls':0,'physics_steps':0,'independent_pose_preflight':False,'new_eval_data':False}
    (OUT/'selection_arithmetic_tests.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'test_count':len(checks),'all_passed':True}))

if __name__=='__main__':main()
