"""Lossless publication copy after all nine frozen static audits finish."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json
ROOT=Path(__file__).parent;OUT=Path('/private/tmp/hug_surface_constraints_archive_2026-10-15_v1/audit')
assert not OUT.exists(), 'Never overwrite an existing archive'
summary=json.loads((ROOT/'summary.json').read_bytes())
assert summary['all_checks_pass'] and summary['fresh_full_grid_measurements']==9 and summary['physics_steps']==0 and not summary['new_eval_data']
OUT.mkdir();bindings=[];lossless_checks=0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selections=[(p,p.relative_to(ROOT)) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
for name,source in (('saved_row_change_supplement',Path('/private/tmp/hug_surface_constraints_local_change_2026-10-15_v1')),('document_review_supplement',Path('/private/tmp/hug_surface_constraints_document_review_2026-10-15_v1'))):
    assert source.is_dir(), 'Required finalized supplemental directory missing: '+str(source)
    selections.extend((p,Path(name)/p.relative_to(source)) for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts)
for path,rel in selections:
    dest=OUT/rel;data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
    if path.suffix=='.json' and (path.name.endswith('_validation.json') or len(data)>131072):
        dest=Path(str(dest)+'.gz');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data;lossless_checks+=1
        bindings.append({'source_absolute_path':str(path),'source_relative_path':str(rel),'source_uncompressed_sha256':h,'source_bytes':len(data),'archive_relative_path':str(dest.relative_to(OUT)),'archive_sha256':sha(dest),'compression':'gzip_mtime0_lossless'})
    else:
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);assert sha(dest)==h;lossless_checks+=1
        bindings.append({'source_absolute_path':str(path),'source_relative_path':str(rel),'source_sha256':h,'source_bytes':len(data),'archive_relative_path':str(rel),'archive_sha256':sha(dest),'compression':None})
(OUT/'archive_source_bindings.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'scope':'One unchanged nine-grid static audit of six existing HUG native grasps; complete reports, source/witness provenance, immutable pooling protocols and failed preparation history retained. Separate saved-row-change and document-review supplements copied without changing primary/reviewer snapshots. Large JSON compressed losslessly, no scientific source/result changes. Primary selected candidate/solver archives supplied separately.','checks_pass':True,'lossless_source_checks':lossless_checks,'bindings':bindings},indent=2)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
manifest={'schema':'hug_surface_constraints_surface_audit_publication_manifest_v1','utc':datetime.now(timezone.utc).isoformat(),'file_count':len(files),'files_sha256':files,'physics_steps':0,'new_eval_data':False,'fresh_grid_reports':9,'six_existing_native_proposals_only':True,'acceptance_rule_sha256':summary['acceptance_rule_sha256'],'final_audit_protocol_sha256':summary['protocol_sha256'],'selected_candidate_sha256':summary['candidate_sha256']}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for k,v in files.items():assert sha(OUT/k)==v
print(json.dumps({'archive':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'file_count':len(files),'lossless_source_checks':lossless_checks,'total_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'largest_bytes':max(p.stat().st_size for p in OUT.rglob('*') if p.is_file())},indent=2))
