from pathlib import Path
import hashlib,json,datetime
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
checks=[]
for name,manifest in [('dexgrasp_anything_mapping_validation_2026-10-15_component_constraints_v1','evidence_manifest_v1.json'),('dexgrasp_anything_mapping_validation_2026-10-15_acceptance_feasibility_v2','evidence_manifest.json')]:
 root=W/'research_artifacts'/name;p=root/manifest;d=json.loads(p.read_text());bad=[]
 for rel,meta in d['files'].items():
  q=root/rel
  if not q.is_file() or 'sha256:'+hashlib.sha256(q.read_bytes()).hexdigest()!=meta['sha256']:bad.append(rel)
 checks.append({'study':name,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files_checked':len(d['files']),'mismatches':bad})
result={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'all_prior_artifacts_unchanged':not any(r['mismatches'] for r in checks)}
out=Path('/private/tmp/dga_practical_retargeting_20261015_v1/control/prior_preservation_v1.json');out.write_text(json.dumps(result,indent=2)+'\n');print(result)
assert result['all_prior_artifacts_unchanged']
