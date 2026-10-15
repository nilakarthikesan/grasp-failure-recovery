from pathlib import Path
import ast,hashlib,json,shutil
OUT=Path(__file__).parent
OLD=Path('/private/tmp/hug_free_palm_surface_audit_2026-10-15_v1')
s=(OLD/'validate_candidate.py').read_text();original=s
start=s.index("        role=target['audited_pose_role']");end=s.index('        meta,tri,labels',start)
s=s[:start]+'''        role=target['audited_pose_role'];assert role in protocol['allowed_audited_pose_roles']
        assert isinstance(target['constraint_selection_accepted'],bool) and isinstance(target['diagnostic_pose_not_promoted'],bool)
        if role=='surface_constraint_selected':assert target['constraint_selection_accepted']
        if role=='tip_feasible_constraint_refused_diagnostic':assert not target['constraint_selection_accepted'] and target['diagnostic_pose_not_promoted']
        audit_role={'audited_pose_role':role,'constraint_selection_accepted':target['constraint_selection_accepted'],'diagnostic_pose_not_promoted':target['diagnostic_pose_not_promoted'],'rejection_reason':target['rejection_reason'],'selection_source':target.get('selection_source')}
''' +s[end:]
s=s.replace("        if role=='tip_feasible_translated_diagnostic_not_selected':assert np.any(shift!=0)\n",'')
s=s.replace("if role=='original_fallback_no_feasible_nonzero':assert np.array_equal(d.qpos,original_q) and np.all(shift==0) and target['guarded_mapper_selected_fallback']", "if role=='original_fallback':assert np.array_equal(d.qpos,original_q) and np.all(shift==0) and not target['constraint_selection_accepted']")
s=s.replace('hug_free_palm_candidate_independent_static_validation_v1','hug_surface_constraints_candidate_independent_static_validation_v1')
ast.parse(s);(OUT/'validate_candidate.py').write_text(s)
for name in ('protected_hashes.json','run_validation.py','freeze_protocol.py','inherited_metric_block.py.txt'):
 shutil.copyfile(OLD/name,OUT/name)
# Metric helper functions and the actual nine-grid loop remain literal frozen source.
a="        for density in selected['densities_mm']:";b="        for file,expected in meta['input_sha256'].items():";block=s[s.index(a):s.index(b)]
parity={'derived_from':str(OLD/'validate_candidate.py'),'derived_from_sha256':hashlib.sha256(original.encode()).hexdigest(),'new_script_sha256':hashlib.sha256(s.encode()).hexdigest(),'full_fields_and_status_byte_identical':s[:s.index('def main():')]==original[:original.index('def main():')],'density_metric_block_byte_identical':block==(OLD/'inherited_metric_block.py.txt').read_text(),'amendment':'Only explicit surface-constraint selection/diagnostic role metadata; source/FK/native targets, exact original tip caps, root composition, full metric/status code unchanged.'}
(OUT/'wrapper_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
p=json.loads((OLD/'protocol.DRAFT.json').read_text());p['schema']='hug_surface_constraints_independent_static_validation_protocol_v1';p['candidate_path']='/private/tmp/hug_surface_constraints_2026-10-15_v1/candidate_targets.json';p['candidate_schema']='PENDING_PRIMARY_SCHEMA';p['candidate_sha256']='PENDING_CANDIDATES';p['candidate_protocol_expected_sha256']='PENDING_PROTOCOL';p['candidate_bound_files_sha256']={};p['script_sha256']=hashlib.sha256(s.encode()).hexdigest();p['fixed_files_sha256']={k:v for k,v in p['fixed_files_sha256'].items() if not k.startswith(str(OLD))};p['allowed_audited_pose_roles']=['surface_constraint_selected','tip_feasible_constraint_refused_diagnostic','original_fallback'];p['scope']='Same six saved native grasps; independent source/FK/exact frozen caps and mean gates before the same six coarse and three fine fresh static grids. Direct fitting on saved witness rows is developmental and cannot replace full-grid acceptance. Every final pose is labeled accepted by fitting constraints, refused diagnostic, or exact original fallback; none is scientific acceptance by label.';p['candidate_fallback_policy']='Invalid/missing/schema/source poses refuse execution. Tip-feasible constraint-refused diagnostics may be measured with explicit refusal role and not promoted. Original fallback must match complete original scene exactly. Never drop a failing case or convert ambiguity to preservation.';p['selection_trace_audit']='Separate independent fit review checks declared saved witness constraints, finite solver budget, exact original target/cap/domain and candidate selection. This surface audit independently measures final poses; it does not replay all historical callbacks.';p['root_review']='Root authorized direct surface-constraint study on same six existing grasps and unchanged final9 scientific checks; final source/selection review and candidate/protocol bindings required before any grid. Old extra direction SSE is diagnostic only; unchanged original frozen proximity and orientation criteria remain authoritative.';p['execution_authorized_by_root_review']=False
p.pop('engineering_minimum_objective_improvement',None);p.pop('minimum_objective_improvement',None)
(OUT/'protocol.DRAFT.json').write_text(json.dumps(p,indent=2)+'\n')
s=(OLD/'summarize_audit.py').read_text().replace('hug_free_palm_fresh_surface_audit_summary_v1','hug_surface_constraints_fresh_surface_audit_summary_v1')
s=s.replace("c['audit_selection_role']['direction_guard_pass']==targets[i]['direction_guard_pass']", "c['audit_selection_role']['constraint_selection_accepted']==targets[i]['constraint_selection_accepted']")
ast.parse(s);(OUT/'summarize_audit.py').write_text(s)
print(json.dumps(parity,indent=2))
