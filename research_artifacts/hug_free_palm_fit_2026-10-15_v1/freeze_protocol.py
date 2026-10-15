from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,platform,importlib.metadata,argparse
OUT=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--review',required=True);args=ap.parse_args()
review=Path(args.review);r=json.loads(review.read_text());p=json.loads((OUT/'protocol.draft.json').read_text())
assert r['passed'] is True and r['reviewed_draft_sha256']==sha(OUT/'protocol.draft.json')
assert p['execution_authorized_by_root_review'] is False and p['draft_only_not_executed']
for table in ('source_sha256','implementation_sha256','query_cache_sha256','full_cache_sha256'):
    for path,h in p[table].items():assert sha(path)==h,path
assert sha(OUT/'protected_hashes.json')==p['protected_snapshot_sha256']
for path,h in json.loads((OUT/'protected_hashes.json').read_text()).items():assert sha(path)==h,path
assert not list(OUT.glob('candidate_cohort_*')) and not (OUT/'full_callback_traces').exists()
# Reviewer PASS and exact reviewed draft are checked explicitly by launch caller.
p.update(frozen_utc=datetime.now(timezone.utc).isoformat(),execution_authorized_by_root_review=True,draft_only_not_executed=False,
    final_review_binding={'path':str(review),'sha256':sha(review),'reviewed_draft_sha256':sha(OUT/'protocol.draft.json')},
    root_review='Root authorized launch after independent final review PASS, with original acceptance unchanged and no inference/physics/eval data.',
    python_version=platform.python_version(),runtime_packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','mujoco','numba')})
with (OUT/'protocol.json').open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),'review_sha256':sha(review),'case_metrics_before_freeze':False}))
