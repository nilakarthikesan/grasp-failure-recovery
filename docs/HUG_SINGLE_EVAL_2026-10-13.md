# One fresh HUG evaluation through grasp-eval

Completed 13 October 2026, at source snapshot `e1abfc3` on `project-work`.
The user authorized one HUG evaluation after separately deferring ContactGen
and UniDexGrasp evaluation generation.

## Declared case and completed result

One `hug_mano_shadow_v1` proposal was freshly generated from the existing
verified cube observation, seed 0 and 50 CPU sampling steps. Dex Retargeting
mapped it to the registered Shadow Hand, and `grasp-eval run` executed one case
with `fixed_grasp_lift_v2`, nominal mass/friction and no extra command noise.
The experiment was validated before execution; `grasp-eval status` reports
one expected and completed success, zero failures and zero remaining cases.

- Maximum lift: **0.151335441733 m**.
- Stable hold: **2.0 s**, with all 50 saved hold samples meeting the lift threshold
  and retaining opposing hand/object contact.
- Peak-to-final height loss: **0.000760347291 m**.
- Hand/table contact: **zero in 163 saved trajectory samples**.
- Video: **163 frames at 25 fps**, duration **6.52 s**.

Independent calculations reproduce lift, hold and final-height values from
`trajectory.npz`. The output video opens and its last frame shows the cube held
above the table. `verification.json` binds the source, model and output hashes.
These observations are an execution check, not predictor accuracy.

The freshly generated native proposal is byte-identical to the earlier cube
scene-000/seed-0 proposal, and every trajectory array matches the earlier saved
HUG-derived replay. This demonstrates deterministic inference/execution
reproducibility on one previously observed situation; it does not add an
independent object, scene or proposal family to the research evidence.

## Local artifacts and provenance

Local run directory in the working checkout:
`runs/hug_grasp_eval_2026-10-13_fresh_cube/`.
It contains `plan.json`, `request.json`, `experiment.json`, `verification.json`
and a last-frame preview. The experiment output is under `output/`; the physical
result, trajectory and replay are at `output/runs/hug_cube_seed0_nominal/`.
Native HUG proposal and inference report remain under
`output/proposals/hug_cube/sample_0000/native/`. Runtime outputs remain ignored.

- HUG source: `8d1c52d4c24bfae5a369e32e3f134f5601a02630`, clean tracked checkout.
- HUG checkpoint SHA-256:
  `515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`.
- Native proposal SHA-256:
  `573cb003458d9d7cfe85b5b4a7877e414257b3e67d5abda04a09bd6d687fa0c2`.
- Canonical proposal hash:
  `sha256:5152d5554bea12c3f9df1b64df7ee27295a5f265e62c23a9b31afb82432b96d6`.
- Environment hash:
  `sha256:954eb93c4a4cffbff9f5b4001823d47dba9c92109427eb97c538ca8532744717`.
- Protocol hash:
  `sha256:a2c6d674b0813b2e21d1fb5d13a339cef80fec718e620b39bc5c21cc4516b99c`.

The frozen predictor was not scored. Its SHA-256 remains
`16c9941a51130daea818c6ad227ef1e3a02e4be9ed1c7aa5a8e8f3e023428beb`.
The generic recorder still omits its required closing velocity/command histories.
No model weights, controller, labels, scientific splits or manuscript changed.

## Restored import path, with no installation changes

The earlier missing-dependency report came from omitting preserved HUG
dependency directories from `PYTHONPATH`. Real compiled torch-cluster CPU
operators, MANO files and cached DINO weights were already available. This run
used them offline; no package installation, checkpoint download or fake model
backend was needed.

The Python executable is
`/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python`.
The original environment lacks the newly declared console entry point, so a
temporary `grasp-eval` script invoked the exact package entry point
`grasp_failure_prediction.grasp_eval:main`, with current working-branch source
first on `PYTHONPATH`. The original `.venv` was preserved.

```sh
export PYTHONPATH="/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages"
export HF_HOME="/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=cgl
```

Actual runtime: Python 3.12.13, NumPy 2.5.3, SciPy 1.18.1, Pydantic 2.13.5,
PyTorch 2.11.0, torchvision 0.26.0, Transformers 4.57.1,
huggingface_hub 0.36.2, torch-cluster 1.6.3, Dex Retargeting 0.5.0,
Pinocchio 3.8.0 and MuJoCo 3.3.7. These differ from the temporary integration-test
runtime; no historical result has been rewritten to use them.

The executed command was `grasp-eval run` with this run's `experiment.json`,
`--hug-root /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug`
and `--checkpoint /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_cpu_audit/checkpoints/hug_full.safetensors`.
Use a new declared output directory for a future run rather than replacing
these artifacts. See [the CLI guide](GRASP_EVAL_CLI.md) and
[provider integration status](CONTACTGEN_AND_UNIDEXGRASP.md).
