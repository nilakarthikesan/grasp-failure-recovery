"""Freeze after root and independent source-only method review; no geometry."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,platform,importlib.metadata,ast,shutil
B=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((B/'protocol.draft.json').read_text());assert not p['execution_authorized_by_root_and_independent_review']
assert sha(B/'run_calibration.py')=='3f13e59e1a5aae6d3aa571ba1d43a278613307050126f17a9d69acd4d5075607'
for table in ('source_sha256','implementation_sha256'):
 for path,h in p[table].items():assert sha(path)==h,path
for path in B.glob('*.py'):ast.parse(path.read_text())
assert not (B/'calibration.json').exists() and not (B/'candidate_targets.json').exists() and not (B/'protocol.json').exists()
# Explicit independent/root approval receipt is supplied by orchestrator after review.
review=json.loads((B/'execution_review.json').read_text());assert review['approved'] and review['root_approved'] and review['independent_approved']
assert review['draft_protocol_sha256']==sha(B/'protocol.draft.json') and review['driver_sha256']==sha(B/'run_calibration.py')
hist=B/'history/final_review_draft';hist.mkdir(parents=True);shutil.copyfile(B/'protocol.draft.json',hist/'protocol.draft.json')
p['execution_authorized_by_root_and_independent_review']=True;p['frozen_utc']=datetime.now(timezone.utc).isoformat();p['review_receipt_sha256']=sha(B/'execution_review.json');p['runtime_python_version']=platform.python_version();p['runtime_package_versions']={}
for name in ('numpy','scipy','torch','mujoco','nlopt','dex_retargeting','manotorch','pin'):
 try:p['runtime_package_versions'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:p['runtime_package_versions'][name]='distribution metadata unavailable; local source hash bound where relevant'
with (B/'protocol.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
print(json.dumps({'frozen_protocol_sha256':sha(B/'protocol.json'),'source_count':len(p['source_sha256']),'after_V1_runtime_abort_before_first_complete_calibration_and_case_outputs':True}))
