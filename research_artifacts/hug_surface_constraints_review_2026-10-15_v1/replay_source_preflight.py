"""Independent recorded baseline constraint arithmetic, with no geometry/FK."""
from pathlib import Path
import hashlib,json
import numpy as np
from constraint_arithmetic import Replay
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BASE=Path('/private/tmp/hug_surface_constraints_root_2026-10-15_v1/all_six_source_preflight.json')
SAVED=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
OP=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json'
CAP=ROOT/'research_artifacts/hug_canonical_mcp_calibration_2026-10-14_v1/runtime_v3/candidate_targets.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bind={str(p):sha(p) for p in (BASE,SAVED,OP,CAP,Path(__file__),Path(__file__).with_name('constraint_arithmetic.py'))}
saved=json.loads(SAVED.read_bytes());base=json.loads(BASE.read_bytes());op=json.loads(OP.read_bytes())
caps={c['source_inventory_index']:c['old_original_five_tip_caps_sqrt_einsum_m'] for c in json.loads(CAP.read_bytes())['cases']}
records=[];count=0
for row in base['records']:
 i=row['index'];r=Replay(saved['whole_cell_rows'],saved['orientation_rows'],i,op,caps[i]);result=r.verify(row['baseline_exact'],exact=True)
 r.check(result['tip_pass'],'baseline exact oldcaps');r.check(row['jacobian']['passed'],'source analytic Jacobian gate')
 records.append({'source_inventory_index':i,'checks':r.checks,**result});count+=r.checks
assert all(sha(Path(p))==h for p,h in bind.items())
report={'passed':True,'checks':count,'source_bindings':bind,'cases':records,'scope':'Independent arithmetic replay of recorded original-baseline fields, exact historical caps, source band/sign predicates and diagnostic ranking. No new FK, geometry queries, optimization, inference, physics or eval data.'}
out=Path(__file__).with_name('source_preflight_arithmetic_review.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'checks':count,'report_sha256':sha(out)}))
