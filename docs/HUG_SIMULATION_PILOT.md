# First connected simulation pilot

The pilot connects the partner's MuJoCo cube scene to observation validation,
official HUG preprocessing and CPU inference, Dex retargeting, and an
execution/scoring protocol. It is a functioning integration experiment, not yet
a validated HUG grasp-success benchmark.

As of 2026-10-02, a separate actuated Shadow Hand positive control successfully
lifts the cube 144.65 mm and holds it for two seconds. This uses an engineered
contact fit derived from HUG, unchanged collision shapes, and finite joint
torque limits. Three identical-start runs succeeded; opening the thumb failed.
The unchanged corrected HUG grasp still fails acquisition. See
[the debugging record](HUG_ALIGNMENT_DEBUG.md#actuated-shadow-baseline-successful-pickup-2026-10-02)
for the fixes, limitations, and reproduction commands.

## Original integration result (before alignment/controller fixes)

- A rendered 224px RGB image, metric depth, camera intrinsics, segmentation-derived
  cube mask, and explicit selected pixel passed observation validation.
- Backprojected cube pixels matched the known simulated cube surface with maximum
  error 0.00001168 m before depth was quantized to HUG's millimeter PNG format.
- Official HUG generated one real proposal with seed 42 and 50 sampling steps on
  CPU, float32. No substitute model or synthetic grasp was used for this attempt.
- Dex converted its landmarks to 24 Shadow Hand joint positions.
- The runner executed 143 steps and scored `no_contact`, with no object acquisition.
- Mean fingertip retargeting residual was 0.15866 m, maximum 0.18549 m. The
  wrist-to-palm correspondence and hand coordinate conventions require review.
  This result must not be labeled a demonstrated failure of HUG's grasp quality.

The camera observation looks plausible with HUG's predicted human-hand skeleton
projected onto it. That visual check is useful but does not establish 3D contact.
The original runner also prescribes the hand's motion kinematically; it does not
model an actuator-limited robot controller. The later diagnostic runner uses
force-limited joint servos and an ideal arm carrier. That is still an experimental
simulation controller, not a calibrated physical Shadow Hand.

## Source and assets

Partner source: `eval-infra-setup` at
`07d4b811dd7f5b216339436dcd4305e6b61d269a`.
HUG source: `https://github.com/KevinyWu/hug` at
`8d1c52d4c24bfae5a369e32e3f134f5601a02630`.
Dex URDF: `https://github.com/dexsuite/dex-urdf` at
`7304c7fb59214dab870eca02cf26f76e944e12df`.
Checkpoint: `kevinywu/hug`, SHA256
`515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae`.
MANO assets were supplied separately by the user and retain their licenses.

Downloads, local dependency installs, and outputs are ignored under `runs/`.
The pilot used an isolated copy of the partner's source with our two observation
modules added; it did not change the partner's branch or merge either PR.

## Commands

These scripts require the partner's evaluation package and our observation
modules on the same Python import path. The capture and execution scripts also
require working macOS graphics access. Use a fresh output directory for each
capture; proposals and reports are protected from accidental overwriting.

```sh
python scripts/capture_sim_observation.py \
  --urdf-root /path/to/dex-urdf/robots/hands --output /path/to/cube_observation

python scripts/infer_sim_observation.py \
  --hug-root /path/to/hug --observation /path/to/cube_observation \
  --checkpoint /path/to/hug_full.safetensors --seed 42 --steps 50

python scripts/execute_sim_proposal.py \
  --observation /path/to/cube_observation --urdf-root /path/to/dex-urdf/robots/hands
```

Runtime: MuJoCo 3.3.7, PyTorch 2.11.0, Transformers 4.57.1,
torch-cluster 1.6.3 built locally for CPU, manotorch at `a2a70c5`,
chumpy at `580566e`, Dex Retargeting 0.5.0, Pinocchio 3.8.0.
Matching native dependencies included cmeel-urdfdom 4.0.1 and
cmeel-tinyxml2 10.0.0. These versions describe this local pilot rather than a
portable installation guarantee.

## Current gate

Coordinate conventions have been corrected and one feasible engineered Shadow
grasp now succeeds with an actuated controller. The remaining gate is to apply
a fixed, declared conversion/controller to fresh HUG proposals and test whether
the results are reliable across conditions. Any geometry correction must remain
traceable and must not be reported as an unchanged proposal. A connected
pipeline and one successful positive control are distinct from a validated
research evaluation.
