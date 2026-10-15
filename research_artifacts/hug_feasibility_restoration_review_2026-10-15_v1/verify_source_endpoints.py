"""Read-only endpoint provenance and protected-row extraction; no FK."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
SOURCE=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2/candidate_targets.json')
POOL=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection/candidate_targets.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads(SOURCE.read_text());p=json.loads(POOL.read_text())
bindings={str(SOURCE):sha(SOURCE),str(POOL):sha(POOL)};records=[];checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
check('six_source_cases',sorted(c['source_inventory_index'] for c in s['cases'])==[8,12,16,20,24,28])
for case in s['cases']:
    idx=case['source_inventory_index'];base=next(c for c in p['cases'] if c['source_inventory_index']==idx)
    for field in ['native_sha256','joint_names','original_qpos_rad','bounds_scaled','domain',
                  'original_root_target_position_m','original_root_target_quaternion_wxyz',
                  'root_target_quaternion_wxyz','authoritative_original_tip_caps_sqrt_einsum_m','constraint_source_rows']:
        check(f'{idx}:source_compatibility:{field}',case[field]==base[field])
    trace=Path(case['solver_audit']['trace_path']);bindings[str(trace)]=sha(trace)
    check(f'{idx}:trace_hash',bindings[str(trace)]==case['solver_audit']['trace_sha256'])
    terminals=[]
    with gzip.open(trace,'rt') as f:
        for line in f:
            e=json.loads(line)
            if e.get('kind')=='optimizer_terminal':terminals.append(e)
    check(f'{idx}:exact_two_terminals',len(terminals)==2)
    bounds=np.array(case['bounds_scaled']);a=base['baseline_constraint_diagnostics'];b=base['selected_constraint_diagnostics']
    protected_valid=np.asarray(a['exact_orientation_fields']['valid'],bool)|np.asarray(b['exact_orientation_fields']['valid'],bool)
    protected_pass=np.asarray(a['exact_orientation']['passed'],bool)|np.asarray(b['exact_orientation']['passed'],bool)
    check(f'{idx}:passed_rows_have_eligibility',not np.any(protected_pass&~protected_valid))
    for terminal in terminals:
        k=terminal['start_index'];old=case['solver_audit']['optimizer_starts'][k]
        x=np.array(terminal['terminal_state_scaled'],np.float64)
        check(f'{idx}:{k}:complete_terminal_identity',all(terminal[key]==val for key,val in old.items()))
        check(f'{idx}:{k}:finite_27',x.shape==(27,) and np.isfinite(x).all())
        check(f'{idx}:{k}:source_domain',bool(np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1])))
        records.append(dict(source_inventory_index=idx,start_index=k,source_trace=str(trace),source_trace_event=terminal['event'],
                            source_state_scaled=x.tolist(),source_state_bytes_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                            protected_original_or_pooled_valid_rows=np.flatnonzero(protected_valid).tolist(),
                            protected_original_or_pooled_pass_rows=np.flatnonzero(protected_pass).tolist(),
                            reference_exact_rank=base['selected_constraint_diagnostics']['rank']))
check('twelve_complete_endpoint_records',len(records)==12)
bindings[str(Path(__file__))]=sha(Path(__file__))
r=dict(passed=True,scope='Frozen prior output and trace provenance only; no FK, geometry or new candidate computation.',source_bindings=bindings,checks_passed=len(checks),checks=checks,endpoints=records)
target=OUT/'source_endpoint_review.json';target.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),endpoints=len(records),sha256=sha(target))))
