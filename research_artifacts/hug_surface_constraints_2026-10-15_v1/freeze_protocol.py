from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
out=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
draft=out/'protocol.draft.json';reviewpath=Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v1/preflight_review.json')
review=json.loads(reviewpath.read_text());p=json.loads(draft.read_text())
assert review['passed'] and review['protocol_draft_sha256']==sha(draft)
for section in ['input_sha256','implementation_sha256','preflight_sha256']:
 for path,h in p[section].items():assert sha(path)==h,path
for path,h in review['prerequisites'].items():assert sha(path)==h,path
p['execution_authorized_after_independent_review']=True
p['reviewed_draft_sha256']=sha(draft)
p['independent_preexecution_review']={'path':str(reviewpath),'sha256':sha(reviewpath),'checks':review['hash_and_start_checks']}
p['preflight_sha256'][str(reviewpath)]=sha(reviewpath)
p['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
p['root_execution_authorization']='Root explicitly approved declared method/budget and authorized execution after independent PASS; no further approval required.'
target=out/'protocol.json';assert not target.exists();target.write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'frozen_protocol':str(target),'sha256':sha(target)}))
