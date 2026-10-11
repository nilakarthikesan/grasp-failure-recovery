# Frozen model evaluation on fresh objects and scenes

Completed on 11 October 2026, after declaring the model, controller, scorer,
input cutoff and collection schedule. The saved original weights, training
normalization and failure alert threshold of 0.5 were unchanged. No training,
calibration or threshold selection used this collection.

The batch contains **2,080 completed attempts**, with **659 successes and
1,421 failures**, no runtime or validation errors and no excluded attempts.
Four geometries provide eight observation families and 32 fresh HUG proposals.
The two familiar objects are the original cube and apple. The two new objects
are a native HUG milk carton and mug. Each object has 520 attempts.

- `dataset.npz`: 2,080 × 1,440 float32 causal input values, binary outcome labels
  and source group IDs. Inputs contain twenty closing samples through 2.96 s:
  24 measured joint positions, 24 velocities and 24 actual commands. Labels and
  identities are separate arrays, outside X.
- `manifest.json`: input contract, provenance, source hashes, memberships,
  source novelty, regenerated task labels and the frozen execution reference.
- `predictions.csv`: 6,240 predictions from the three original models.
- `report.json`: full and grouped metrics, model/configuration hashes and runtime.
- `protocol/`: the original frozen plan, setup amendment, exact scene cases,
  source assets/attribution, pose rejection audit and replay selection policy.
- `replays/`: eight small MP4 examples and a media hash/selection manifest.
- `failure_categories.json` and `validation.json`: failure breakdown and
  verification evidence.

This is a separately versioned **evaluation only** artifact. The original
`load_dataset` is specific to the original training/validation/test pilot;
this bundle does not add rows to those partitions. Full trajectories, meshes
and HUG weights remain in the local experiment directory listed
in [the walkthrough](../../docs/FRESH_OBJECT_EVALUATION.md).

For a concrete model error, watch the [first successful milk carton attempt](replays/episode_001040.mp4).
The original threshold raises an alert at predicted failure probability 0.6085,
but the attempt meets the frozen success rule. The [first milk carton failure](replays/episode_001042.mp4)
has predicted failure probability 0.9923. These examples were selected by actual
outcome and collector order, rather than confidence.

## What the unchanged model did

On familiar geometries in fresh scenes, failure AUROC is **0.9173**, failure
recall **88.3%**, and false alarms among successful attempts **21.3%**.
Across the new geometries, AUROC is **0.5588**, recall **73.2%**, and false alarms
among successful attempts **75.5%**. These contrasts describe this batch and
eight observation families, not independent population estimates.

The milk carton has 245 successes and 275 failures. Its AUROC is 0.8121, but
the fixed threshold falsely flags 185 successful attempts and misses eight
failures. All 520 mug attempts fail; the model catches 315 and misses 205.
Every mug failure is an approach collision between hand and table, before the
closing prefix cutoff. Mug AUROC and success false alarm rate are undefined
because no mug succeeded. These attempts test eventual task labels after an
earlier violation, not warnings before future grasp loss.
Pooled new object ranking also reflects differences between the two objects;
it is not the mean of their individual ranking scores.

Overall, the full model's AUROC is 0.7737, log loss 0.6645 and Brier score
0.1865. Commands alone give AUROC 0.5524 and log loss 1.1081; the unchanged
training prevalence gives AUROC 0.5 and log loss 0.6584. The full model improves
ranking and Brier score over the constant comparison, but does not improve
overall log loss. Aggregate accuracy should not replace the object breakdowns.

These results show a limitation of transferring the current predictor and
threshold to new objects. They do not establish slip timing or deployment
reliability. Geometry and native mass change together for the new objects.

A separate read only audit reconstructed every raw prefix and frozen task
label, reproduced all 6,240 probabilities exactly without prediction helpers,
and recomputed all 87 grouped metric records. It verified original source
disjointness and all model, controller and configuration hashes.

## Setup rejection and sources

The initial salt shaker failed the existing stationary capture test in six
declared resting orientations. We retained that rejection and replaced it with
the milk carton before any labeled grasp attempts. It is not a failed grasp
label and is not among the 2,080 attempts.

New geometry comes from [HUG / 1M HuGs](https://huggingface.co/datasets/kevinywu/1m-hugs),
published by kevinywu, revision `56ea3f100b1a4ec38bb35e986c0dd14d38977283`,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Milk carton
asset: `val/medium_5/milk_box`; mug asset: `val/medium_5/mug`. The provenance
manifest also retains the rejected `val/small_4/salt_shaker` download.
No official benchmark grasp annotations or failure recordings were used.

See [the step by step guide](../../docs/FRESH_OBJECT_EVALUATION.md) for the
frozen evaluation command and the local viewer. All recorded labels regenerate
under the original scorer, and source observations/proposals are disjoint from
the original pilot. Predictions are computed offline from causal prefixes.
