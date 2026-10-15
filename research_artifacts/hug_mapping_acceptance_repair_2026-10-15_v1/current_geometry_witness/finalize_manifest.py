"""Archive manifest only; no models, FK, queries, optimization or inference."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((HERE/'protocol.json').read_text());result=json.loads((HERE/'results.json').read_text())
assert result['protocol_sha256']==sha(HERE/'protocol.json')
for path,h in p['bindings'].items():assert sha(path)==h,path
files={str(f.relative_to(HERE)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(HERE.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.name!='bundle_manifest.json'}
manifest={'schema':'hug_current_geometry_witness_archive_manifest_v1','protocol_sha256':sha(HERE/'protocol.json'),'result_sha256':sha(HERE/'results.json'),
 'files':files,'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files.values()),'source_bindings_rechecked_after_results':True,
 'licensed_or_native_assets_copied':False,'external_scientific_sources_remain_hash_bound_in_protocol':True,'archive_scope':'Primary fixed-current-representation diagnostic, preparations, controls, outputs, pure summaries and final reviewer receipt. Full reviewer bundle separately retained.'}
out=HERE/'bundle_manifest.json';assert not out.exists();out.write_text(json.dumps(manifest,indent=2)+'\n')
for path,record in json.loads(out.read_text())['files'].items():assert sha(HERE/path)==record['sha256']
print(json.dumps({'manifest_sha256':sha(out),'file_count':manifest['file_count'],'bytes':manifest['total_bytes'],'readback_passed':True}))
