"""Prepare a lossless compact publication copy after both frozen audits finish."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,shutil
ROOT=Path(__file__).parent;OUT=ROOT/'publication_bundle'
assert not OUT.exists(), 'Never overwrite archive'
for p in (ROOT/'summary.json',ROOT/'supplemental_explicit_translation/summary.json'):
    value=json.loads(p.read_bytes());assert value['all_checks_pass']
OUT.mkdir();bindings=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path in sorted(ROOT.rglob('*')):
    if not path.is_file() or OUT in path.parents or '__pycache__' in path.parts:continue
    rel=path.relative_to(ROOT);dest=OUT/rel;data=path.read_bytes();h=hashlib.sha256(data).hexdigest()
    if path.name.endswith('_validation.json'):
        dest=Path(str(dest)+'.gz');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
        bindings.append({'source_absolute_path':str(path),'source_relative_path':str(rel),'source_uncompressed_sha256':h,'source_bytes':len(data),'archive_relative_path':str(dest.relative_to(OUT)),'archive_sha256':sha(dest),'compression':'gzip_mtime0_lossless'})
    else:
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);assert sha(dest)==h
        bindings.append({'source_absolute_path':str(path),'source_relative_path':str(rel),'source_sha256':h,'source_bytes':len(data),'archive_relative_path':str(dest.relative_to(OUT)),'archive_sha256':sha(dest),'compression':None})
(OUT/'archive_source_bindings.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'scope':'Initial nine-grid audit plus separately frozen five-grid explicit-proposal supplement; lossless result compression only. No scientific source/result changes.','bindings':bindings},indent=2)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'manifest.json').write_text(json.dumps({'schema':'hug_free_palm_surface_audit_publication_manifest_v1','utc':datetime.now(timezone.utc).isoformat(),'file_count':len(files),'files_sha256':files,'physics_steps':0,'new_eval_data':False,'initial_fresh_grid_reports':9,'supplemental_fresh_grid_reports':5,'six_existing_native_proposals_only':True},indent=2)+'\n')
print(json.dumps({'archive':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'file_count':len(files),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())},indent=2))
