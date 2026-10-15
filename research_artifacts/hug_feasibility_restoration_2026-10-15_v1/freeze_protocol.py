from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
OUT=Path(__file__).parent;REVIEW=Path('/private/tmp/hug_feasibility_restoration_review_2026-10-15_v1/preflight_review.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((OUT/'protocol.draft.json').read_text());r=json.loads(REVIEW.read_text())
assert r['passed'] and r['protocol_draft_sha256']==sha(OUT/'protocol.draft.json')
for section in ['input_sha256','implementation_sha256','preflight_sha256']:
 for path,h in p[section].items():assert sha(path)==h,path
for path,h in r['prerequisites_sha256'].items():assert sha(path)==h,path
p['preflight_sha256'].update(r['prerequisites_sha256']);p['preflight_sha256'][str(REVIEW)]=sha(REVIEW)
p['execution_authorized_after_independent_review']=True;p['reviewed_draft_sha256']=sha(OUT/'protocol.draft.json');p['independent_preexecution_review']={'path':str(REVIEW),'sha256':sha(REVIEW)}
p['frozen_at_utc']=datetime.now(timezone.utc).isoformat();p['root_execution_authorization']='Root explicitly authorized all12 frozen static restorations after independent PASS; preserve declared budgets, no extensions, inference, physics or eval data.'
target=OUT/'protocol.json';assert not target.exists();target.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'protocol':str(target),'sha256':sha(target)}))
