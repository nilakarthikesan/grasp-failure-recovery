"""Publish this completed static development phase without modifying history."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,shutil

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP=Path('/private/tmp/hug_feasible_refinement_2026-10-14')
BUNDLE=ROOT/'research_artifacts/hug_feasible_refinement_2026-10-14'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    assert not BUNDLE.exists()
    summary=read(TMP/'validation/summary.json');fit=read(TMP/'seed_stage/fit_summary.json')
    assert summary['resolution_count']==9 and summary['case_count']==6
    assert summary['all_six_independent_source_and_engineering_preflights_pass']
    assert summary['all_six_feasible_improving_poses_selected']
    assert summary['strict_status_counts']=={'PASS':0,'FAIL':9,'UNRESOLVED':0}
    assert summary['physics_steps']==0 and not summary['new_eval_data'] and not summary['training']
    assert '146 passed' in (TMP/'installed_tests.log').read_text()
    for path,h in read(TMP/'seed_stage/protected_hashes.json').items():assert sha(path)==h,path
    assert sha(ROOT/'src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py')==sha(TMP/'solver_design/feasible_surface_refinement.py')
    assert read(TMP/'seed_stage/independent_seed_numerical_review.json')['failed_checks']==[]
    BUNDLE.mkdir()
    copied=[]
    groups=['diagnosis','objective_diagnosis','solver_design','seed_stage','validation','presentation']
    excluded={'report.json','report.partial.json'}
    for group in groups:
        for src in sorted((TMP/group).rglob('*')):
            if not src.is_file():continue
            rel=src.relative_to(TMP)
            if any(part in ('__pycache__','mplconfig') for part in rel.parts):continue
            if src.suffix=='.pyc':continue
            if group=='objective_diagnosis' and src.name in excluded:continue
            if src.name.endswith('.partial.json'):continue
            dst=BUNDLE/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            if src.suffix=='.json' and src.stat().st_size>1_000_000:
                dst=dst.with_suffix('.json.gz');raw=src.read_bytes();dst.write_bytes(gzip.compress(raw,mtime=0))
                copied.append({'source_path':str(src),'source_sha256':sha(src),'bundle_path':str(dst.relative_to(BUNDLE)),
                    'bundle_sha256':sha(dst),'encoding':'gzip_exact_original_bytes','uncompressed_size_bytes':len(raw)})
                assert gzip.decompress(dst.read_bytes())==raw
            else:
                shutil.copyfile(src,dst)
                copied.append({'source_path':str(src),'source_sha256':sha(src),'bundle_path':str(dst.relative_to(BUNDLE)),
                    'bundle_sha256':sha(dst),'encoding':'original_bytes','size_bytes':src.stat().st_size})
    shutil.copyfile(TMP/'installed_tests.log',BUNDLE/'installed_tests.log')
    for name in ('freeze_hug_feasible_seed_stage.py','freeze_hug_feasible_validation.py','summarize_hug_feasible_validation.py'):
        dst=BUNDLE/'orchestration'/name;dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(Path('/private/tmp')/name,dst)
    orientation=summary['pointwise_orientation_status_counts']
    lines=[]
    for row in fit['six_case_rows']:
        lines.append(f"- Case {row['index']} ({row['object_id']}, {row['source_case']}): cost {row['original_objective']:.6f} → {row['returned_objective']:.6f} ({row['objective_reduction_percent']:.2f}% lower); mean tip error {row['original_mean_tip_mm']:.6f} → {row['returned_mean_tip_mm']:.6f} mm; all five caps pass.")
    doc=f'''# HUG feasible refinement: completed static development phase

14 October 2026 (America/New_York). This phase reuses the same six saved HUG
development grasps and previously rejected raw fits. It generates static
mapping diagnostics only, with no new inference, evaluation episodes, outcome
labels or predictor training. Historical protocols/results are unchanged.

## Why every earlier fit was refused

All 649 recorded major fingertip vectors replay exactly. The six original
five-cap Jacobians have rank five and admit nonzero inward steps within the
source bounds. There is no demonstrated frame, cache or constraint-sign bug
in these recorded checks. The earlier SLSQP surface search evaluates infeasible
trial poses and only tests final/major eligibility afterward; its first major
steps reach the 0.35-rad correction limit on 11–17 joints. None of its recorded
improving major/raw poses passes all original-relative fingertip constraints.

Cheap FK controls restore all six raw poses to the earlier engineering gate;
five converge with the earlier SLSQP settings and one reaches its iteration
limit with a 2.27-nm residual within the earlier declared 10-nm allowance.
Analytic-Jacobian projection converges in 4–13 iterations on all six. These
controls show feasible movement is possible; they are not surface fits.

A frozen 60-call objective stencil also measures a nearby Mug B direction
eligibility jump at two negative 1e-5-rad probes. Finer probes converge, and
the sampled original forward probes avoid the jump. Thus discontinuity is
documented but is not established as the sole cause of all six refusals.
The supported diagnosis is an unsuccessful coupled local search without
feasible-step restoration, rather than contradictory tip caps or global
infeasibility.

## Implemented repair and measured outcome

The isolated experimental API checks/restores cheap fingertip FK before any
new surface score. It supports finite-budget pattern search; that path has
manufactured tests only. The actual frozen study is seed-only: exactly one
immutable old raw pose per case, no pattern sweeps, unchanged quadrature,
distance/closest-point/facing/anchor/joint cost, and the same source URDF
bounds intersected with original q ±0.35 rad.

Positive original tip caps are tightened by a 1-nm restoration target; final
selection adds no arithmetic allowance. It uses declared sqrt(einsum) norms,
all five original-relative caps, a strict 10-mm five-tip mean, a lower frozen
cost, FK/full callback agreement and fresh terminal callbacks outside cache.
Original poor relationships, negative facing targets, invalid-site penalties
and missing direction mass remain in the objective. No lift-quality filter
is introduced. Callback errors/refusal return the exact original target.

All six new poses are selected and all 30 fingertip caps pass. The stage uses
18 full surface calls in 49.57 seconds, reusing the expensive old raw seeds;
this is not a from-scratch inference/fit speed claim. Two cheap projections
converge; four reach the 40-iteration budget while returning independently
checked feasible improving points. No full-convergence claim is made.

{chr(10).join(lines)}

Independent original/final source-URDF versus captured-scene checks pass on
all six. Maximum selected link discrepancy is 7.122e-8 m, within the inherited
1.1e-7-m source representation check. Native wrist/palm placement, root
convention, physical bounds, object geometry/state and time-zero guards pass.
All 89 relevant protected numerical dependencies, inputs and the predictor
remain byte-identical; evolving paper and unrelated provider code are outside
this computational closure.

## Mapping acceptance remains failed

The independent wrapper preserves the literal previous surface/status functions
and complete query/cell/proximity/orientation measurement block. All six source
preflights finish before the unchanged six coarse and three fine comparisons.
The original 1/2/5-mm bands, numerical assumptions, anatomical labels and normal
eligibility remain unchanged.

The strict result is **0 PASS, 9 FAIL**. Pointwise orientation has
{orientation['FAIL']} FAIL and {orientation['UNRESOLVED']} UNRESOLVED results.
Definite lost/added proximity regions remain. The nine resolutions measure
six grasps repeatedly; they are not nine physical grasp failures. Lower cost
and nonworsening tip error do not certify complete surface/contact preservation.
The caps retain existing absolute errors; some tips still exceed 10 mm, since
10 mm applies to the five-tip mean. Continuous physical contact, native failure
label preservation and held-out object/source generalization remain unconfirmed
or untested. The production mapper is unchanged because this fidelity gate
still fails.

The objective is not a monotone guarantee of the declared diagnostics: milk
carton B's coarse 2-mm summed added-area lower bound changes from 142.31 to
245.00 mm² while its fitting cost falls 38.09%. Mug A's coarse 5-mm check
finds seven reliable sampled sign reversals, versus none previously observed.
The previous absence was UNRESOLVED. Area sums are across anatomical parts,
not unique union/contact areas; lower bounds are not exact areas. Sample
eligibility changes with pose, so raw reversal counts are not comparable
failure rates. These facts refute certification from a lower average fit cost.

Next work is a separately frozen study of the remaining anatomical surface
witnesses with the repaired feasibility mechanism. Do not relax the current
acceptance rule, select only successful native grasps, or use these static
results as prediction-training labels. Neither an impossible global mapping
nor a complete HUG correction has been proved.

## Verification and evidence

- 146 installed new/prior surface tests pass, including 26 new manufactured
  feasible/refusal/curvature/cache/budget/callback cases.
- The independent seed review passes 214 arithmetic/provenance checks.
- The recorded selection verifier passes six authentic and ten deliberate
  tamper-rejection checks; these do not replace fresh geometry validation.
- Both scientific figures pass visual QA, include all six cases, and disclose
  Mug B's roughly 1-nm tip change and all four inner-budget exhaustions.
- All original scientific bundles are preserved. No licensed hand asset,
  raw proposal/scene/query cache or training recording is copied into this bundle.

[Portable evidence](../research_artifacts/hug_feasible_refinement_2026-10-14/README.md),
[implementation](../src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py),
[tests](../tests/test_feasible_surface_refinement.py), and the existing
[manuscript](paper/working_draft.tex).
'''
    (ROOT/'docs/HUG_FEASIBLE_REFINEMENT.md').write_text(doc)
    (BUNDLE/'README.md').write_text('''# HUG feasible refinement evidence

Completed static development diagnosis, isolated implementation, six feasible
improving poses and nine unchanged-rule audits. The fingertip constraint
handling is repaired on these cases; full mapping acceptance remains failed.
Read [the guide](../../docs/HUG_FEASIBLE_REFINEMENT.md) and
[compact numerical summary](validation/summary.json).

The `manifest.json` binds each file. Large JSON results are exact gzip
encodings, with both raw and compressed hashes in `source_copy_receipt.json`.
Final complete cohort reports are authoritative. Interim accumulation files
remain in the runtime directory; failed/incomplete cases were not dropped.
Draft protocols, old layout and pre-execution reviews are retained as history
and do not replace the final protocols/completed reports.

Replay requires the original licensed/runtime assets and saved inputs at the
recorded paths; no assets, native inference, physical eval data or learner
recordings are included. Numeric dependencies are bound separately from the
concurrently edited manuscript and unrelated generators. The new API is
experimental and has not replaced the production retargeter.
''')
    write(BUNDLE/'source_copy_receipt.json',{'schema':'hug_feasible_static_exact_copy_receipt_v2',
        'published_utc':datetime.now(timezone.utc).isoformat(),'files':copied,'new_eval_data':False,'physics_steps':0})
    metadata={'schema':'hug_feasible_publication_verification_v2','strict_status_counts':summary['strict_status_counts'],
        'all_six_feasible_improving_poses_selected':True,'all30_exact_computed_original_relative_caps_pass':True,
        'installed_test_count':146,'production_mapper_unchanged':True,'scientific_rule_unchanged':True,
        'implementation_sha256':sha(ROOT/'src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py'),
        'installed_tests_sha256':sha(ROOT/'tests/test_feasible_surface_refinement.py'),
        'numeric_closure_count':89,'all_protected_numeric_sources_unchanged':True,
        'paper_revision_follows_numeric_runs_recorded_separately':True,'physics_steps':0,'new_eval_data':False,'training':False}
    write(BUNDLE/'publication_verification.json',metadata)
    write(BUNDLE/'manifest.json',{'schema':'hug_feasible_static_bundle_manifest_v2',
        'files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file() and p.name!='manifest.json'}})
    print(json.dumps({'bundle':str(BUNDLE),'bound_files':len(read(BUNDLE/'manifest.json')['files']),
        'bundle_manifest_sha256':sha(BUNDLE/'manifest.json'),'guide_sha256':sha(ROOT/'docs/HUG_FEASIBLE_REFINEMENT.md')}))
if __name__=='__main__':main()
