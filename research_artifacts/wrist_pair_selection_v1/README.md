# Frozen wrist pair selection

12 October 2026. This package freezes exact A/B wrist angles and compensated
carrier coordinates before any new rollout or predictor evaluation.

## What was selected

The previous finite audit found no passing combination among the eight
actually observed training wrist pairs and eight mug proposals. This search
therefore uses the declared fallback: the geometrically feasible candidate
nearest the original training wrist-command distribution. B remains the
stored zero-noise corrected pose for the same case.

8/8 cases have a distinct
passing A/B pair. The first eligible pair is `mug_scene002_seed0_noise0.00`, chosen by the existing case order. Its A/B comparison is **two out of range wrists sensitivity only**.

Both wrists must pass the 10 mm nominal table gate and 1 mm proximal-object
gate. Carrier compensation preserves the intended world palm and all 22
finger targets and their nominal world geometry. This only certifies the
near-open approach endpoint used by the existing v3 planner. It does not
certify closure, dynamics or grasp success. The later executor must retain
the 1 mm runtime table guard at every 2 ms physics endpoint and stop without
a label if the margin fails.

## Selection rule

A fixed 13 by 13 grid spans both physical wrist limits. All eight observed
training command pairs and B are also evaluated. Three local 5 by 5
refinement rounds use initial grid spacing, then half and quarter, centered
on the currently closest-training passing candidate. B may be a refinement
center but cannot be selected as A. Candidates must be at least
0.02 rad from B in Euclidean wrist distance.
Every evaluated candidate is recorded, including rejected ones. This is a
finite search, not a continuous global optimum or infeasibility proof.

Training distance is the Euclidean norm of the two wrist commands after
scaling with the original training-only mean and scale at feature indices
48 and 49. No model weights or intercept are read. No prediction, new grasp
outcome, old label or failure type is used to rank candidates or choose the
first pair. Old B metadata contains those historical fields, but only its
geometric certificate and execution settings are consumed. The original
training partition contributes X and episode identities only, never y.

All retained candidates are rechecked for intended palm/finger invariance
within 1e-10. Selected A is re-evaluated before export. Every stored B
certificate repeats within 1e-12. Original source, model, data and asset
hashes agree before and after selection. Simulator time stays zero and the
existing integration blocker is explicitly exercised.

## Frozen pairs

* `mug_scene002_seed0_noise0.00`: A [0.113628255, -0.202076656] rad; B [0.099083839, -0.226802156] rad. Training distances 39.198 and 39.265; separation 0.028686 rad. A closer than B: True. two_out_of_range_wrists_sensitivity_only.
* `mug_scene002_seed1_noise0.00`: A [-0.007272208, -0.209803375] rad; B [0.007272208, -0.185077875] rad. Training distances 28.784 and 28.194; separation 0.028686 rad. A closer than B: False. two_out_of_range_wrists_sensitivity_only.
* `mug_scene002_seed2_noise0.00`: A [0.159988583, -0.555960375] rad; B [0.174533000, -0.531234875] rad. Training distances 70.721 and 69.394; separation 0.028686 rad. A closer than B: False. two_out_of_range_wrists_sensitivity_only.
* `mug_scene002_seed3_noise0.00`: A [0.163624687, -0.384427219] rad; B [0.174533000, -0.359701719] rad. Training distances 56.118 and 55.107; separation 0.027025 rad. A closer than B: False. two_out_of_range_wrists_sensitivity_only.
* `mug_scene003_seed0_noise0.00`: A [-0.011817339, -0.124809469] rad; B [-0.026361755, -0.149534969] rad. Training distances 22.719 and 22.794; separation 0.028686 rad. A closer than B: True. two_out_of_range_wrists_sensitivity_only.
* `mug_scene003_seed1_noise0.00`: A [-0.131808776, -0.056932425] rad; B [-0.146353193, -0.081657925] rad. Training distances 7.297 and 7.659; separation 0.028686 rad. A closer than B: True. two_out_of_range_wrists_sensitivity_only.
* `mug_scene003_seed2_noise0.00`: A [0.094538708, -0.257709031] rad; B [0.109083125, -0.232983531] rad. Training distances 40.898 and 40.580; separation 0.028686 rad. A closer than B: False. two_out_of_range_wrists_sensitivity_only.
* `mug_scene003_seed3_noise0.00`: A [0.087266500, -0.130990844] rad; B [0.072722083, -0.155716344] rad. Training distances 33.179 and 32.739; separation 0.028686 rad. A closer than B: False. two_out_of_range_wrists_sensitivity_only.

Whether A lies within each original command range is recorded separately.
Being closer to training is not the same as being in the training range,
and command familiarity does not imply familiar measured joint histories.
The selected distinct A can have greater training distance than B; this
comparison is reported rather than forcing a favorable direction.
If both settings are out of range, this is a wrist-sensitivity comparison,
not a familiar versus unfamiliar test. A difference in eventual predictions
would still need interpretation alongside actual dynamics and outcomes.

## Files and reproduction

* [pose_manifest.json](pose_manifest.json) is the exact, ordered pose contract
  for an explicit executor. The first eligible pair should be run first with
  zero noise and the unchanged original model.
* [selection_report.json](selection_report.json) contains every candidate,
  rejected gate, distance, invariance check, source binding and allowed-call
  counter. It reports no generated predictions, labels or rollouts.
* [manifest.json](manifest.json) hashes all package payloads and input sources.
* [select_wrist_comparison_poses.py](../../scripts/select_wrist_comparison_poses.py)
  reproduces selection, refusing to overwrite an output directory.

```bash
python -B scripts/select_wrist_comparison_poses.py \
  --project-root /path/to/checkout/with/existing/runs \
  --output /tmp/wrist_pair_selection_reproduction
```

The next experiment is one explicit zero-noise pair. These static results do
not improve the model or answer the later matched-budget data-diversity
question. No additional collection or fitting is justified by this geometry
certificate alone.
