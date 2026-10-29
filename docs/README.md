# Documentation

## Run and reproduce

- [Phase 1 end-to-end demo and meeting walkthrough](PHASE1_END_TO_END.md)
- [Phase 1, Part 1: known-object scene preparation](PHASE1_SCENE_PREPARATION.md)
- [Unified multi-provider evaluation command](GRASP_EVAL_CLI.md)
- [Pinned DexGraspNet replay](DEXGRASPNET_REPLAY.md)
- [ContactGen and UniDexGrasp integration](CONTACTGEN_AND_UNIDEXGRASP.md)
- [DexGrasp Anything GPU-worker handoff](DEXGRASP_ANYTHING_CLOUD_HANDOFF.md)
- [External provider worker contract](GRASP_PROVIDER_WORKERS.md)

- [Setup, simulation commands and replay](RUNNING_THE_PROJECT.md)
- [Noisy rollout collection and task labels](NOISY_ROLLOUT_COLLECTION.md)
- [Multi-scene collection and geometry provenance](MULTI_SCENE_COLLECTION.md)
- [Prediction inputs and grouped splits](DATA_SPLITS.md)
- [Baseline fitting and inference](PREDICTION_BASELINE.md)
- [Evaluation manifests and registries](EVALUATION_MANIFEST.md)
- [HUG observation validation](HUG_OBSERVATION_PREFLIGHT.md)
- [Linux container runtime](CONTAINERIZED_EVALUATION.md)
- [Working branch and local assets](WORKING_BRANCH.md)

## Research and completed evidence

- [Grasp mapping fidelity audit](GRASP_MAPPING_FIDELITY.md)
- [HUG individual mapping confirmation](HUG_MAPPING_CONFIRMATION.md)
- [HUG surface-aware correction and independent validation](HUG_SURFACE_REFINEMENT.md)
- [HUG fingertip/direction constrained correction](HUG_CONSTRAINED_REFINEMENT.md)

- [Research protocol](HUG_MONITOR_RESEARCH_PLAN.md)
- [Experiment results and evidence index](PAPER_PROGRESS.md)
- [Fresh HUG single-case CLI check](HUG_SINGLE_EVAL_2026-10-13.md)
- [Fresh-object evaluation](FRESH_OBJECT_EVALUATION.md)
- [Mug table-clearance correction](MUG_TABLE_CLEARANCE.md)
- [Working paper](paper/working_draft.tex)
- [Partner handoff](PARTNER_HANDOFF.md)
- [Recorded noise demo](demos/grasp-noise/README.md)

Versioned datasets, model coefficients, predictions and experiment manifests
are stored in [research_artifacts](../research_artifacts/). Raw simulation
recordings and third-party runtime assets remain local; see the collection guides.

## Development history

[The archive](archive/README.md) contains earlier designs, literature surveys,
educational walkthroughs and superseded plans. It supports tracing the project's
development and is separate from the current research protocol.

[Dated research decisions](RESEARCH_DECISIONS.md) preserve the evidence behind
changes. The [commit review](archive/COMMIT_REVIEW_2026-10-12.md) explains how all
91 commits through `6790b2d` contribute to the repository.

The [Panda training specification](PART_I_TRAINING_SPEC.md) documents supporting
implementation retained in `part1/`; it is separate from the evaluated
HUG/Shadow Hand predictor.


## Parallel generator validation and shared manuscript

[Coordination and three thread prompts](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md)
assign independent ContactGen, DexGrasp Anything and UniDexGrasp audits, shared
dependency ownership and locked subsection updates to the existing paper.
The prediction-centered opening and provider methods are drafted; actual-native
validation and mixed-source training remain pending where stated.
