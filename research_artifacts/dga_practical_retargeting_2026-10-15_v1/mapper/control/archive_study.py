"""Persist new evidence without rewriting any prior study or dependency pin."""
from pathlib import Path
import shutil,hashlib,json,datetime
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B=W/'research_artifacts/dga_practical_retargeting_2026-10-15_v1'
PREV=W/'research_artifacts/dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
B.mkdir(exist_ok=True)
for name,source in [('mapper','/private/tmp/dga_practical_retargeting_20261015_v1'),('protocol','/private/tmp/dga_practical_protocol_20261015_v1'),('validation','/private/tmp/dga_practical_validation_20261015_v1')]:
 shutil.copytree(source,B/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
prior={}
for rel,meta in json.loads((PREV/'evidence_manifest.json').read_text())['files'].items():
 prior.setdefault(meta['sha256'].removeprefix('sha256:'),PREV/rel)
local={}
for p in B.rglob('*'):
 if p.is_file():local.setdefault(sha(p),p)
bindings={}
for p in B.rglob('*.json'):
 try:d=json.loads(p.read_text())
 except (ValueError,UnicodeDecodeError):continue
 if not isinstance(d,dict):continue
 for field in ['dependency_sha256','code_sha256','source_bindings','input_hashes','geometry_dependency_hashes','output_hashes','input_partition_reports']:
  values=d.get(field,{})
  if isinstance(values,dict):
   for original,digest in values.items():
    if isinstance(original,str) and original.startswith('/') and isinstance(digest,str) and len(digest.removeprefix('sha256:'))==64:
     bindings.setdefault((original,digest.removeprefix('sha256:')),[]).append(str(p.relative_to(B)))
rows=[];missing=[]
for (original,digest),refs in sorted(bindings.items()):
 if digest in local:p=local[digest];kind='new_study'
 elif digest in prior and sha(prior[digest])==digest:p=prior[digest];kind='prior_committed_study'
 elif Path(original).is_file() and sha(original)==digest:
  p=B/'dependency_blobs'/digest;p.parent.mkdir(exist_ok=True);shutil.copyfile(original,p);local[digest]=p;kind='new_blob'
 else:missing.append({'original_path':original,'sha256':digest,'referenced_by':refs});continue
 assert sha(p)==digest
 rows.append({'original_path':original,'sha256':digest,'repository_path':str(p.relative_to(W)),'kind':kind,'referenced_by':refs})
write(B/'dependency_index.json',{'schema_version':'dga_practical_dependency_index_v1','bindings':rows,'missing_versions':missing,'all_bytes_verified':not missing})
(B/'.gitattributes').write_text('dependency_blobs/* -text -diff\n*.npz binary\n*.mjb binary\n')
print(json.dumps({'bundle':str(B),'dependency_bindings':len(rows),'missing':missing,'copied_files':sum(p.is_file() for p in B.rglob('*'))}))
if missing:raise SystemExit('Missing historical dependency versions; do not claim archive complete')
