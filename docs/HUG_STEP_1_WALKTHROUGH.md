# Step 1: Understand HUG's datasets and choose our starting data

We are building a model that warns when a robot grasp is likely to fail. Step 1 is to understand the available data, inspect one small example, and decide what we still need to collect.

## How to explain the project

> We use a pretrained model to suggest a grasp, execute it with a fixed robot
> controller, and deliberately vary the finger commands. We save what the robot
> did and whether it lifted and held the object. Those attempts will later train
> a separate model to predict failure early enough for a response.

The dataset inspection and cube pilot were checked on 9 October 2026; this
walkthrough was prepared on 10 October 2026. The cube illustration uses Enoch's
`codex/hug-proposal-execution-v1` at `843fcfe` and an additional local collector
on `codex/noisy-rollout-pilot`. Those implementations are separate from this
documentation PR and are not all present on main.

## The two models have different jobs

- **Pretrained HUG:** looks at a color image and depth information, then proposes a human hand grasp that we can translate to our robot hand. We already use its released weights.
- **Our future failure predictor:** will look at information available during a robot attempt and estimate whether that attempt will succeed or fail. We have not trained this model yet.

HUG supplies the starting grasp. Our simulation will supply the robot's experience and outcome labels. [Official HUG project](https://grasping.io/), [code and weights instructions](https://github.com/KevinyWu/hug).

## What the available datasets provide

**1M-HUGs** contains human image-and-grasp examples used to train HUG. Its release lists 1,279,142 training samples, plus 600 validation and 300 test inputs with simulation assets. A million samples does not mean a million robot attempts: multiple images can come from one human grasp. The documented schema includes images, depth, masks, cameras and human hand pose information. Its documented input schema does not contain the binary robot failure labels our predictor needs; the full scene archives were not audited for additional telemetry. [Official dataset card](https://huggingface.co/datasets/kevinywu/1m-hugs).

**HUG-Bench** is the evaluation object collection: 90 objects across five geometry categories and three size groups. There are 60 validation objects and 30 test objects, with ten grasp inputs per object. The benchmark evaluates grasps in simulation and on robots. Its separate asset release contains scene archives. [Paper, Section 5.1](https://arxiv.org/pdf/2606.17054), [official assets](https://huggingface.co/datasets/kevinywu/hug-bench/tree/main).

Validation data helps choose settings; test data measures performance after those choices are finished. These are HUG's original splits. We will define the failure predictor's own training and validation split among development objects and reserve the 30 official test objects from tuning. Entire trajectories must stay together, and related attempts must be grouped to avoid testing on near-copies of training examples.

Both official dataset releases identify their license as **CC BY 4.0**. Keep attribution to Kevin Yuanbo Wu and coauthors, *Human Universal Grasping* (2026), alongside the dataset URL and revision when using these assets. HUG code and MANO models have separate terms.

## Our inspected example: an apple

We downloaded one small validation object, `val/medium_3/apple`, rather than the entire dataset. Its 16 files total 950,726 bytes, under 1 MB. The inspected dataset revision is `56ea3f100b1a4ec38bb35e986c0dd14d38977283`.

- **Ten inputs:** color image, depth, object mask, camera information and a selected object point; all ten have their grasp label withheld.
- **Matching 3D assets:** a visible mesh, two collision meshes, simulator descriptions and physical properties.
- **Configured properties and mesh bounds:** mass 230.4 g from the asset configuration; visible dimensions approximately 8.1 × 8.5 × 9.2 cm calculated from mesh vertices. These are not new physical measurements.
- **Checks completed:** passively inspected the input files without executing their pickle contents; decoded one 224 × 224 image/depth/mask example; compiled the standalone object asset in MuJoCo.
- **Current status:** downloaded and inspected. The apple has not been imported into our Shadow-hand runner or used for a grasp rollout.

See the local inspection report (`runs/hug_benchmark_review_2026-10-09/review_report.json`), download manifest with revision and file hashes (`runs/hug_benchmark_review_2026-10-09/download_manifest.json`), and color image (`runs/hug_benchmark_review_2026-10-09/sample_preview/image.jpg`). The public source is the [official apple folder](https://huggingface.co/datasets/kevinywu/1m-hugs/tree/56ea3f100b1a4ec38bb35e986c0dd14d38977283/val/medium_3/apple).

## A worked illustration using our existing cube pilot

The cube pilot already contains 50 completed attempts: 40 scored successes, 10 scored failures and zero validation/runtime errors. These are existing local examples to explain the plan; collecting them is not a requirement for finishing Step 1. First batch summary (`runs/noisy_pilot_2026-10-09/batch/summary.json`), stress batch summary (`runs/noisy_pilot_2026-10-09/stress_batch/summary.json`).

**What does noise mean?** We start with a HUG grasp and use the same fixed approach, close, lift and hold controller. For each attempt, we draw one bounded random offset for each finger joint target and keep that offset through closing, lifting and holding. Joint limits clip the resulting command. Wrist joints are excluded. This is a fixed offset per attempt, not a fresh random change every frame.

### Selected noise type and every tested strength

So far we use **one noise type: a uniform random finger-command offset held
constant for one episode** (`uniform_episode_bias`). We tested five strengths,
with ten trials at each strength:

- **0 rad:** baseline with no added noise; 10 successes, 0 failures.
- **±0.05 rad (±2.9°):** 10 successes, 0 failures.
- **±0.15 rad (±8.6°):** 10 successes, 0 failures.
- **±0.3 rad (±17.2°):** 7 successes, 3 failures.
- **±0.6 rad (±34.4°):** 3 successes, 7 failures.

Each finger's requested offset is sampled uniformly inside that strength's
bounds. The same ten proposals and sampled noise directions are paired across
strengths. Wrist offsets stay zero; noise applies during closing, lifting,
holding and final scoring. Joint-limit clipping determines the actual command
offset. Every trial records the strength, random seed, requested offsets,
applied offsets, nominal/final commands and outcome. The stronger levels are
exploratory stress tests, not calibrated hardware-error distributions.
These are different strengths of the same noise type. Additional noise types
have not been selected or implemented in this collector.

### Why we start with this noise type

The meeting plan asks for bounded random changes to joint targets. Our fixed
offsets meet that initial experiment: they test whether a slightly different
finger configuration still grasps, lifts and holds the object. There is no
requirement in these notes to add every kind of noise before starting trials.
The 50 attempts verify a small collection example; they do not complete the
planned thousands of trajectories or establish realistic hardware error ranges.

More elaborate disturbances would answer additional questions:

- **Smoothly changing command jitter:** an offset that changes during the
  attempt, to test temporary control disturbances. We would specify and log its
  amplitude and how quickly it changes.
- **Observation noise:** changes to images, depth or measured state. Noise
  before HUG inference could change the proposed grasp; noise on the future
  predictor's inputs would test sensing robustness. Changing a saved observation
  alone does not change a physical attempt under the current open-loop controller.
- **Dynamics variation:** different object mass, friction or actuator response,
  to test whether a grasp remains reliable under different physical conditions.

These are possible later experiments, not implemented modes in this pilot.
Action and observation noise are separate in [Dactyl's simulation setup,
Section 3.2 and Appendix C.2](https://arxiv.org/pdf/1808.00177); changing physical
properties is studied in [dynamics randomization, Section IV.C](https://arxiv.org/pdf/1710.06537).
Those studies concern control robustness and do not validate our future failure
predictor. Our next practical priority is verified object and proposal diversity,
keeping zero-noise controls and testing each disturbance separately before
combining them. More complicated noise is not automatically more realistic.

### See the recorded noise

The local interactive explanation is saved at
`runs/noise_explainer_2026-10-10/index.html`. It embeds the five recorded seed-2
attempts and both replay videos, so the HTML can be opened directly in a browser
or shared as a file without the viewer server. It is a generated local artifact,
not included in this PR. Choose a strength, select `FFJ1` or `THJ3`, and scrub
the attempt to compare the nominal target, requested offset, clipped command
and measured angle. Controls select existing data; they do not rerun physics.

The plotted commands are endpoints of 40 ms control intervals; the controller
interpolates between endpoints during each interval. Measured joint positions
are logged at the interval end. In the first lift interval, the noisy thumb
target `THJ3` requests approximately **−15.9°** and clips to its **−12.0°**
lower limit. Commanded angles and measured motion remain distinct, as described
in the [MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).

Here is the same HUG proposal, generated with seed 2, under two conditions:

- **No noise:** `batch/episode_000006` lifted the cube 147.97 mm and held it for two qualifying seconds. Its label is **1**. Video (`runs/noisy_pilot_2026-10-09/batch/episode_000006/rollout.mp4`), result (`runs/noisy_pilot_2026-10-09/batch/episode_000006/result.json`).
- **Noise bounded by ±0.6 radians:** `stress_batch/episode_000005` reached only 52.84 mm of lift and had zero qualifying hold time. Its label is **0**. The scorer recorded `failed_acquisition`, meaning it did not reach the required lift even though there was some hand/object contact. Video (`runs/noisy_pilot_2026-10-09/stress_batch/episode_000005/rollout.mp4`), result (`runs/noisy_pilot_2026-10-09/stress_batch/episode_000005/result.json`).

One saved command makes the noise concrete. During the first lift interval, 2.96–3.00 seconds, finger joint `FFJ1` had a nominal target of **0.734880638 radians**. The noisy trial added **−0.593244796 radians**, giving a commanded target of **0.141635842 radians**. That is approximately **42.1° − 34.0° = 8.1°**. The command was inside the joint's 0–1.5708-radian limits, so the requested and applied offsets were equal. Saved noisy actions (`runs/noisy_pilot_2026-10-09/stress_batch/episode_000005/actions.npz`).

Noise was requested for all finger joints in that trial; clipping limited or removed some offsets. This one joint illustrates the recorded arithmetic; it does not establish that this joint alone caused the failure. The original HUG proposal remains the same, and the data distinguishes the nominal target, requested noise, applied noise and final command. Applied noise means the command offset after clipping; the measured joint motion is logged separately and need not equal that command.

The 0.6-radian setting is a strong stress condition. We have not established it as a realistic hardware error. These examples use one cube, one initial view and saved HUG proposals; they do not demonstrate generalization to new objects.

To watch the two saved simulations on the collection machine, open the
[local results viewer](http://127.0.0.1:8766/) and search for
`pilot_cube_hug_seed2`. Select `batch/episode_000006` for the successful attempt
or `stress_batch/episode_000005` for the failed attempt, then press Play.
The following direct video links also work while the local viewer is running:
[success replay](http://127.0.0.1:8766/files/batch/episode_000006/rollout.mp4) and
[failure replay](http://127.0.0.1:8766/files/stress_batch/episode_000005/rollout.mp4).
These are saved trajectory replays, not fresh physics runs, and the local server
is not a remotely shared website. If the viewer stops, open either local
`rollout.mp4` directly in a video player. To serve the saved collection again,
use the environment for Enoch's runtime branch and point it at your data folder:

```sh
python -m grasp_failure_prediction.evaluation.result_viewer \
  --runs-root /path/to/noisy_pilot_2026-10-09 --port 8766 --no-open
```

That viewer module comes from the separately referenced runtime branch; this
documentation PR provides the guide and source records.

## What the labels mean

**1 means the completed attempt passed the current scorer.** The scorer requires hand/object contact, at least 14 cm of lift, two consecutive qualifying hold seconds, no more than 2 cm from peak height to final height, no hand/table collision during pre-grasp or approach, and completion within the 10-second timeout. During hold, the permitted 2 cm drop tolerance gives a minimum height of 12 cm above the starting height.

**0 means a completed simulation attempt failed one or more scoring conditions.** A validation problem or runtime error receives no binary label; it must not become a training example labeled 0.

The labels describe the final attempt outcome. We still need to define and annotate failure onset and prediction timing before claiming the model can warn *before* failure. The HUG paper's benchmark success rule differs from our lift-and-hold scorer, so we should not compare their success rates directly.

## Where the files live

Paths below are relative to the original collection checkout, not the documentation worktree. They are local artifacts, not GitHub file links.

- Apple inputs and assets: `runs/hug_benchmark_review_2026-10-09/apple/`.
- Apple previews, provenance and inspection: `runs/hug_benchmark_review_2026-10-09/`.
- Pilot plans, episode index and summaries: `runs/noisy_pilot_2026-10-09/batch/` and `runs/noisy_pilot_2026-10-09/stress_batch/`.
- Each episode's `result.json`: success/failure and lift/hold measurements.
- Each episode's `collection_metadata.json`: label, source observation/proposal, noise settings and provenance.
- Each episode's `actions.npz`: commands, requested/applied noise and state information over time.
- Each episode's `trajectory.npz`: the recorded robot/object trajectory and contact diagnostics.
- Each episode's `scene.mjb` plus `actions.npz` integration states: saved-model/state replay material. `final_state.npz` contains final positions and velocities; it alone is not an exact replay snapshot. The two illustrated episodes also have `rollout.mp4` videos.

The pilot references the **initial RGB-D observation** in `runs/hug_integration_check/cube_observation/`. It does not record RGB-D at every timestep. Simulator-only state/contact information is useful for inspection and labeling; it is not automatically a sensor input for the future model. Final labels, future observations and noise seeds must not be predictor features.

`runs/` is ignored by Git and currently stored locally. A Git clone or pull request will not include these samples. Partners need an explicit data transfer or the recorded download source; GitHub is not a backup of these generated files.

## When Step 1 is complete

- [x] Identify the official training data and benchmark releases.
- [x] Explain pretrained HUG versus our future failure predictor.
- [x] Inspect a representative input and its matching simulation asset.
- [x] Record source revision, file hashes, license and local storage.
- [x] Recognize that robot trajectories and binary outcomes still need collection.
- [x] Record the benchmark split and the import gaps.
- [x] Prepare the worked example, saved videos and walkthrough for joint review.
- [ ] Each partner can explain the data choice and label rule in their own words.

Step 1's deliverable is this dataset review and agreed starting point. It does not require training a model, integrating every benchmark object or launching thousands of trials.

## What belongs in Step 2

The next step is to adapt one benchmark object's observation and geometry to the robot scene, verify camera/mesh alignment and contact handling, and inspect a small fixed-controller pilot. The apple needs that work because the current runner builds only the cube.

Keep the dataset-review documentation in one focused PR. Review the simulation import and pilot changes in a separate Step 2 PR, with the source and outcome definitions visible. Steps 3–6 will cover larger collection, labeling, data splits and predictor training after this small example is understood.
