"""Bind the independent V2 review and freeze before any fitting call."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
out=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--review',required=True,type=Path);args=ap.parse_args()
draft=out/'protocol.draft.json';reviewpath=args.review
review=json.loads(reviewpath.read_text());p=json.loads(draft.read_text())
assert review['passed'] and review['protocol_draft_sha256']==sha(draft)
for section in ['input_sha256','implementation_sha256','preflight_sha256']:
 for path,h in p[section].items():assert sha(path)==h,path
for path,h in review.get('prerequisites',{}).items():assert sha(path)==h,path
p['execution_authorized_after_independent_review']=True
p['reviewed_draft_sha256']=sha(draft)
p['independent_preexecution_review']={'path':str(reviewpath),'sha256':sha(reviewpath)}
p['preflight_sha256'][str(reviewpath)]=sha(reviewpath)
p['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
p['root_execution_authorization']='Root explicitly approved the one V2 unclamped far-guidance correction with identical starts/budgets after independent PASS; no third version or retuning.'
target=out/'protocol.json';assert not target.exists();target.write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps({'frozen_protocol':str(target),'sha256':sha(target)}))
