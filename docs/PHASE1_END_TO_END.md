# Phase 1: image to grasp proposal to recorded outcome

“We now have a working simulation pipeline: start with a known cube and calibrated RGB-D image, ask HUG for three grasp proposals, check their native MANO mapping, translate them to a Wuji hand, and record what happens during approach, closure, lift and hold.”

“The complete run finished for all three proposals. Native mapping checks passed, but all three Wuji attempts failed to lift and hold the cube. This gives us a reproducible way to inspect failures; it does not yet give us a trained success predictor or evidence that the robot grasps reliably.”

1. **Prepare the scene.** [scene_preparation.py](../src/grasp_failure_prediction/integrations/scene_preparation.py), `prepare_scene()` and `reload_saved_scene()`, produce the hand-free 5 cm, 100 g cube/table scene, actual settled state, RGB, metric depth, mask and camera calibration. Tests cover geometry, analytical inertia, gravity settling, ray/depth agreement and complete state restoration. These are synthetic object assumptions, not a reconstruction of a real scene.
2. **Generate genuine HUG proposals.** [phase1_hug.py](../src/grasp_failure_prediction/integrations/phase1_hug.py), `generate()`, uses the existing pretrained model with fixed seeds 0, 1, 2 and 50 sampling steps. `export_proposal()` preserves the official grasp fields and actual float32 MANO source output in `proposal.npz`. Tests check observation integrity, depth validity, the fixed protocol and literal source-array preservation.
3. **Check native mapping.** [phase1_manosim.py](../src/grasp_failure_prediction/integrations/phase1_manosim.py), `validate_native()`, independently reconstructs MANO vertices and landmarks, then uses `ManoSimStaticPoseAdapter.stage()` to compare all sixteen joint anchors and bone orientations in the captured scene. It verifies the actual proposal shape against the external hand assets. Tests reject shape/camera mismatches and detect source geometry errors even when simulator self-FK agrees. `mapping.png` shows the actual native pose. Capsules and rigid fingertip markers approximate the skinned surface; this stage integrates zero physics steps.
4. **Translate to Wuji.** [phase1_wuji.py](../src/grasp_failure_prediction/evaluation/phase1_wuji.py), `build_model()` and `retarget()`, use the pinned official right Wuji Hand 2.1 model. A bounded fit matches the source landmarks using root placement and finger joints, retaining residuals and joint-limit checks. Tests check proper rigid alignment and camera/world conversion without applying the wrist transform twice. Native MANO fidelity does not establish Wuji anatomical fidelity. Target-pose contact penetration is diagnostic (3.43 mm for seed 0); collision/contact qualification remains unfinished.
5. **Execute and label.** The same module's `run_trial()` commands approach, closure, lift and hold with force-limited finger actuators and an ideal moving wrist fixture. Success requires at least 5 cm of cube lift throughout the one-second hold, checked at every physics step. Each result includes video, trajectory and status. Invalid mappings/runtime failures retain null outcomes; they are not training negatives.

The recorded source is `ff50778f9b10db1411972805c6b857e757920cc7`. The complete single-command run is at:

`/Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_pipeline_demo_2026-10-29_v1`

Its `summary.json` records **3 completed trials, 0 successes, 3 lift-or-hold failures and 0 invalid/unexecuted cases**. Every trial contains 2,500 physics steps. No predictor was trained. Three proposals on one cube do not estimate generalization, useful prediction accuracy, or hardware transfer.

Start the meeting with [seed 0's video](/Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_pipeline_demo_2026-10-29_v1/wuji/seed_000/video.mp4), then [its native mapping image](/Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_pipeline_demo_2026-10-29_v1/native/seed_000/mapping.png) and [the complete summary](/Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_pipeline_demo_2026-10-29_v1/summary.json).

To repeat the approved three-proposal run, use [run_phase1_demo.py](../scripts/run_phase1_demo.py) from this checkout. Supply the existing local model and Wuji asset locations; choose a new output directory:

```bash
MUJOCO_GL=cgl PYTHONDONTWRITEBYTECODE=1 \
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
  scripts/run_phase1_demo.py \
  --scene /Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_scene_preparation_2026-10-28_v1 \
  --output /Users/nilakarthikesan/grasp-failure-prediction/runs/phase1_pipeline_demo_new \
  --hug-root /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug \
  --checkpoint /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_cpu_audit/checkpoints/hug_full.safetensors \
  --wuji-assets /Users/nilakarthikesan/grasp-failure-prediction/local_assets/wuji_hand2_1_5c4dfa \
  --dependency-dir /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps \
  --dependency-dir /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps \
  --dependency-dir /Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages
```

The native validator defaults to the existing external literal-shape MANOSim assets; `--native-assets-root` can select an explicit matching asset bundle. The macOS renderer needs access to the graphics connection.

Expected outputs are `protocol.json`, `summary.json`, logs, `hug/seed_000` through `seed_002`, `native/seed_000` through `seed_002`, and `wuji/seed_000` through `seed_002`. Native folders contain `report.json`, `mapping.png` and static replay poses. Wuji folders contain `report.json`, `result.json`, `retarget.npz`, `trajectory.npz`, full-resolution `physics_trace.npz`, `events.json`, `video.mp4`, `initial.png` and `final.png`. Licensed MANO assets remain outside the repository.

The next research step is to diagnose the failed robot executions, qualify a fixed controller across varied scenes and collect both outcome classes before training an image-plus-proposal predictor. Native mapping checks and successful pipeline execution alone cannot supply that evidence.

A small [versioned evidence bundle](../research_artifacts/phase1_pipeline_demo_2026-10-29_v1/) contains the protocol, summary, per-trial reports, robot videos and a digest receipt. Raw proposal arrays, licensed MANO assets and full traces remain in the local run above.
