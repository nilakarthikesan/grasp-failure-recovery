from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
OUT=Path(__file__).parent
rows=[]
for sub in (OUT,OUT/'supplemental_explicit_translation'):
    protocol=json.loads((sub/'protocol.json').read_bytes());path=Path(protocol['candidate_path']);data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==protocol['candidate_sha256'];candidate=json.loads(data)
    rows.append({'audit_directory':str(sub),'audit_protocol_sha256':hashlib.sha256((sub/'protocol.json').read_bytes()).hexdigest(),'candidate_path':str(path),'candidate_sha256':protocol['candidate_sha256'],'cases':[{'source_inventory_index':c['source_inventory_index'],'native_sha256':c['native_sha256'],'audited_pose_role':c['audited_pose_role'],'selection_source':c.get('selection_source'),'explicit_translation_proposal_origin':c.get('explicit_translation_proposal_origin'),'guarded_mapper_selected_fallback':c['guarded_mapper_selected_fallback'],'direction_guard_pass':c['direction_guard_pass'],'diagnostic_pose_not_promoted':c['diagnostic_pose_not_promoted']} for c in candidate['cases']]})
report={'schema':'hug_free_palm_audit_optional_source_metadata_addendum_v1','utc':datetime.now(timezone.utc).isoformat(),'reason':'Frozen wrapper optional metadata getter looked for audit_selection_source while bound independent candidate uses selection_source, leaving that optional display field null. Full candidate SHA/native hash/index and all actual pose checks were preserved. This read-only addendum displays exact source metadata from the same immutable candidate bytes. No scientific metric/code/report changed.','audits':rows,'new_geometry_calculations':0,'new_fit':False,'new_eval_data':False}
(OUT/'selection_source_metadata_addendum.json').write_text(json.dumps(report,indent=2)+'\n')
print('Read-only optional metadata addendum written for both immutable candidates.')
