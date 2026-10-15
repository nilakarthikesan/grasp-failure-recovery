import json,subprocess
from pathlib import Path
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP=Path('/private/tmp/hug_feasible_refinement_2026-10-14')
PYTHON='/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python'
s=json.loads((TMP/'validation/summary.json').read_text())
assert s['strict_status_counts']=={'PASS':0,'FAIL':9,'UNRESOLVED':0}
assert s['pointwise_orientation_status_counts']=={'PASS':0,'FAIL':9,'UNRESOLVED':0}
base=[PYTHON,str(ROOT/'scripts/update_paper_section.py'),'--paper',str(ROOT/'docs/paper/working_draft.tex'),'--section','hug']
before=json.loads(subprocess.check_output(base+['--inspect'],text=True))
old='''are successful engineering fits, not a surface-preservation certificate. The
independent unchanged nine-resolution mapping audit remains pending; production
mapping is unchanged and contact/failure-label preservation is unconfirmed.
'''
new=r'''are feasible improving points. Two cheap projections converge and four reach
their 40-iteration budget; final feasibility is checked independently of the
optimizer's status. This is not a full-convergence or surface-preservation claim.

The unchanged independent mapping audit gives zero PASS and nine FAIL; all
nine pointwise orientation results now FAIL. Definite lost/added proximity
regions remain despite the improved fitting cost. For example, the coarse
2 mm milk-box B summed added-area lower bound changes from 142.31 to
245.00 mm$^2$, while its fitting cost falls by 38.09\%. At the coarse 5 mm
band, mug A supplies seven reliable sampled sign-reversal witnesses, compared
with none observed previously. These are conditional anatomical-part area
sums and pose-specific eligible sample counts, not unique contact areas or
comparable failure rates. They show that the average fitting objective and
tip constraints do not certify the complete native grasp. The production
mapper therefore remains unchanged; continuous physical contact and
failure-label preservation are still unconfirmed.
'''
assert before['content'].count(old)==1
fragment=TMP/'hug_final_fragment.tex'
fragment.write_text(before['content'].replace(old,new))
result=subprocess.check_output(base+['--fragment-file',str(fragment),'--expected-section-sha256',before['section_sha256']],text=True)
(TMP/'paper_final_section_update.json').write_text(result)
print(result)
