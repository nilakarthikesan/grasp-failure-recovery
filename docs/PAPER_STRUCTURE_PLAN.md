# Paper structure and writing plan

## Current direction, 14 October 2026 — parallel audits and prediction-centered paper

The user authorizes parallel individual validation of ContactGen, DexGrasp
Anything and UniDexGrasp, with HUG/shared dependencies coordinated in the main
chat. The accepted working title is **Evaluating Simulation Diversity for
Grasp Failure Prediction**. The central question is whether training on proposals
from multiple generators improves generalization to unfamiliar objects and
grasp configurations relative to single-generator training. Mapping validity
supports this prediction study; it is not the sole research contribution.

The [working manuscript](paper/working_draft.tex) now contains an updated
abstract/introduction and source-specific mapping/verification subsections,
with completed HUG evidence and explicit missing actual-native evidence for the
other three sources. Historical HUG predictor/executor results are preserved as
development motivation. The one/two/three/four-source comparison remains
proposed, with matched training counts as a methods control and generator
identity separated from count. No training or new evaluation data occurs here.

Current structure: introduction/related work; common mapping validity method;
HUG/ContactGen/DexGrasp Anything/UniDexGrasp subsections; historical HUG execution
and development diagnostics; planned multiple-source training/generalization
comparison; limitations/evidence; references. Provider threads maintain only
their owned marked subsections using the shared lock/hash editor. The coordinator
owns abstract/introduction, common claims, bibliography and global ledgers.
[Assignments and coordination](PARALLEL_GENERATOR_VALIDATION_HANDOFF.md).

This is a growing research draft, not a completed comparison or submission-ready
PDF. The earlier diagnostic-only workshop plan below remains a dated alternative.
The broader direction supersedes its HUG-only research-question and exclusion of
four-generator work; it does not change the workshop's page/template rules or
claim future findings as completed. Final submission scope/format must be chosen
from evidence actually completed at submission time.

## Historical diagnostic-only submission plan (preserved)

Updated 14 October 2026. Target: the CoRL 2026 Sim2Real2Field workshop,
Late-Breaking track. The authors chose to aim for the 15 October deadline with
the completed diagnostics. The submission should present a simulation study of
warning reliability; the matched training-diversity comparison remains future
work under the [existing research protocol](HUG_MONITOR_RESEARCH_PLAN.md).

The working manuscript stays at [paper/working_draft.tex](paper/working_draft.tex).
This plan supplies its proposed structure and starter prose. The existing
manuscript has not yet been shortened or converted to the official template.

## Venue format and submission requirements

The [workshop call](https://sim2real2field-corl26.github.io/) confirms a
non-archival workshop, not a journal. Its Late-Breaking track allows four pages
excluding references. The eight-page Published track is for work already
accepted at another notable venue. The workshop explicitly welcomes preliminary
findings and negative results.

Use the linked [CoRL 2026 template](https://drive.google.com/file/d/1R9irIW0ImqeDHh5g8Ukm2XGy91sWxOsQ/view?usp=drive_link).
Its initial submission uses `\documentclass{article}` and
`\usepackage{corl_2026}`. It is single-column, with a 10-point Times body and
US-letter layout. Let the style control margins, fonts and spacing. Its default
mode anonymizes the author block; `[final]` and `[preprint]` reveal authors and
should not be used for the initial review version. The example recommends a
single-paragraph abstract of four to six sentences and two or three keywords.
The existing draft's generic article/geometry formatting must be replaced when
preparing the submission PDF, while retaining its source path.

The public deadline is **15 October 2026, 11:59 p.m. Central daylight time**,
equivalent to **16 October, 12:59 a.m. in New York**. Use this earlier public
deadline even though the current submission API is configured one hour later.
All authors need active OpenReview profiles, and at least one author must be
willing to review up to two workshop papers. The current
[workshop form](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Workshop/Sim2Real2Field)
collects the title, abstract, keywords, author profiles, track and PDF. Its
[public form specification](https://api2.openreview.net/invitations?id=robot-learning.org%2FCoRL%2F2026%2FWorkshop%2FSim2Real2Field%2F-%2FSubmission)
sets a 50 MB PDF limit; [OpenReview documents the units](https://docs.openreview.net/reference/api-v2/entities/invitation/specifiers).

The workshop says reviewing follows standard CoRL guidelines. The
[main-conference author instructions](https://www.corl.org/contributions/instruction-for-authors)
require anonymous review, a Limitations section and disclosure of generative-AI
use. Include limitations inside our four pages and prepare an accurate disclosure.
The workshop does not explicitly exempt appendices or currently offer a
supplementary-upload or AI-disclosure field. Do not assume the main conference's
appendix exclusions, supplementary ZIP or nine-page camera-ready allowance apply.
Keep the complete argument inside four pages, with references afterward; confirm
any optional appendix or supplement arrangement in the live form before relying
on it. Remove identifying repository links and acknowledgments from the review
PDF, and use an anonymous evidence link only if one is available and permitted.

## The story supported by the completed evidence

Proposed title: **When Successful Grasps Trigger Failure Warnings**.
A more descriptive alternative is **Diagnosing Pre-Lift Grasp Warning Reliability
Under Unfamiliar Execution Configurations**.

The current title, “Evaluating Simulation Diversity for Grasp Failure Warnings,”
suggests that the matched diversity comparison has already been evaluated. The
new title should describe the measured diagnostic rather than its proposed remedy.

Central claim: a pre-lift outcome predictor can rank development failures while
still warning on successful unfamiliar grasps. Correcting the executor exposes
this operating-point problem, and a wrist coverage/feasibility analysis bounds
what the subsequent diagnostic can explain.

Frame the paper around three contributions, without claiming a new learning
algorithm:

1. A reproducible HUG-to-Shadow-Hand simulation protocol and baseline that
   predicts eventual grasp/lift/hold completion from joint states and applied
   commands recorded before lift.
2. A frozen-model reliability diagnostic: development AUROC 0.8523, followed by
   warnings on all 20 successful corrected mugs and, in a separate protocol,
   all 13 successes in the eight-pair wrist diagnostic.
3. An input and static-feasibility audit that documents unfamiliar wrist
   commands and shows why the eight observed training wrist pairs cannot
   provide a safe familiar-command mug control under the declared clearance gate.

This most directly addresses the workshop's question about monitoring transfer
failure. The relevance is a simulation diagnostic of a monitoring problem on the
path toward deployment, not evidence of successful sim-to-real-to-field transfer.

## Example structures to borrow

These are full conference papers, not four-page workshop templates. Borrow their
argument order and compress it; do not import their length or contribution claims.

### Runtime monitoring example

[Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency
and Progress](https://proceedings.mlr.press/v270/agia25a.html), Agia et al.,
CoRL 2024 (proceedings published in 2025), has eight main pages. Its sections are
Introduction; Related Work; Problem Setup; Proposed Approach: Sentinel;
Experiments; Results; Conclusion. The
[paper PDF](https://arxiv.org/pdf/2410.04640) uses an early pipeline figure and
score plots with a warning threshold.

Borrow the clear problem definition and evaluation of both detection and false
alarms. Our closing-prefix target is eventual task completion: we have not
measured physical failure-onset times or positive slip-warning lead time.

### Grasp prediction and transfer example

[Grasp Stability Prediction with Sim-to-Real Transfer from Tactile Sensing](https://arxiv.org/pdf/2208.02885),
Si et al., IROS 2022, has an eight-page PDF including references. Its sections
are Introduction; Related Work; Simulation Framework; Sim-to-Real Grasping
Prediction; Experiments; Conclusions.

Borrow its separation of simulation setup, prediction method and evaluation.
Its real-robot experiments support transfer claims; our paper should instead
name its simulation scope in the abstract, setup and limitations.

### Failure detection and calibration example

[Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure
Detection for Imitation Learning Policies](https://arxiv.org/html/2503.08558v1),
Xu et al., RSS 2025, uses Introduction; Related Work; Problem Formulation;
Failure Detection Framework; Experiments; Results; Limitations; Conclusion.

Borrow its separation of scoring from the warning rule and its focused
evaluation questions. Our strongest question is whether a warning rule remains
useful when an unfamiliar task succeeds, rather than whether AUROC alone rises.

## Proposed four-page outline

The allocations below sum to four pages and include displays. They are layout
targets to verify in the official template, not guaranteed word counts.

1. **Title and abstract — about 0.35 page.** In roughly 120–160 words, state the
   problem, system, development result, corrected-mug result and simulation
   scope. Keep the proposed training comparison out of the abstract's results.
2. **Introduction and related work — about 0.55 page.** Explain why warnings
   must distinguish unsuccessful attempts from successful unfamiliar ones.
   Give a short account of outcome prediction and runtime monitoring, then the
   three bounded contributions. Integrate related work instead of allocating
   a separate survey section. Relevant existing references include Calandra,
   Gilday and recent runtime monitors; verify every citation before submission.
3. **Task, predictor and evaluation protocol — about 0.75 page.** Describe
   HUG proposals, retargeting, Shadow Hand and MuJoCo; define the grasp/lift/hold
   label and null-label exclusions; identify closing samples through 2.96 s and
   lift at 3.00 s. Summarize the state/command logistic baseline, commands-only
   reference, grouped retrospective split and frozen 0.5 warning threshold.
   Explicitly distinguish original v2, corrected v3 and paired v4 execution.
4. **Results and diagnosis — about 1.55 pages.** Organize by findings, not run
   chronology: development ranking signal; corrected-mug false warnings;
   unfamiliar wrist inputs and static feasibility; separate paired diagnostic.
   Explain that the original fresh mug batch failed during approach and could
   not test false warnings on successful mugs. Present counts at the fixed
   threshold as the main result; use AUROC as complementary evidence.
5. **Discussion, limitations and next experiment — about 0.60 page.** Explain
   that score attribution identifies model arithmetic rather than physical
   causation. Cover shared proposal families, inspected development data,
   different executor versions, simulation scope and cutoff-sensitive labels.
   Briefly propose calibration/representation checks and a matched coverage
   comparison, retaining their unexecuted status.
6. **Conclusion — about 0.20 page.** State the measured lesson: useful ranking
   can coexist with warnings on successful unfamiliar grasps. End with the
   specific next comparison needed to test a remedy.

Put references after the main text. The four-page argument must stand on its own.

## Displays and evidence to prioritize

**Figure 1: system and prediction timeline.** Show RGB-D/HUG proposal,
retargeted Shadow Hand execution, closing-prefix features, frozen predictor and
eventual outcome. Mark the prefix ending at 2.96 s and lift beginning at 3.00 s.
A saved successful-mug sequence with its warning can make the problem concrete.
Do not draw a recovery feedback loop: the predictor was scored offline and did
not change these trajectories.

**Table 1: warning counts in three separately labeled cohorts.** Use the
retrospective original test (58 successes, 22 false warnings; 202 failures,
187 warned and 15 missed), corrected v3 mugs (20 successes, 20 false warnings;
3 failures, all warned), and paired v4 mugs (13 successes, 13 false warnings;
3 failures, all warned). Identify the fixed threshold and shared families in
the caption. Never pool the latter cohorts into 33 independent successes: v4
reuses families and B reference trajectories from the corrected checks.

If space permits, add a small wrist-command plot showing the actual training
pairs and diagnostic commands. Keep it descriptive; neither paired posture is
familiar. A table and pipeline are sufficient if a third display crowds the text.

Use [PAPER_PROGRESS.md](PAPER_PROGRESS.md) as the claim index. Primary evidence
is in the baseline, corrected-mug, wrist-input, feasibility and paired bundles
under `research_artifacts/`. Keep hashes, replay tolerances, exhaustive pair
values and development chronology in those records. Do not depend on uploading
a separate supplement to make the main finding understandable.

## Starter prose for author review

The following is a proposed opening, not a replacement of the current manuscript.
Its contribution sentence and citations should be reviewed by both authors.

### Proposed introduction opening

Simulation makes it practical to collect labeled grasp attempts for learning an
outcome monitor. A useful monitor must flag unsuccessful attempts while allowing
successful unfamiliar grasps to proceed. Evaluating only failure ranking can
obscure whether a fixed warning rule makes that distinction. We study this
problem in a HUG-based Shadow Hand grasp, lift and hold task in MuJoCo, using
joint states and applied commands recorded before lifting. Our frozen predictor
has useful ranking on a retrospective development test, yet warns on every
successful mug in a separately versioned executor diagnostic. We analyze the
associated wrist inputs and feasible comparison configurations to identify what
this result establishes and which training comparisons remain necessary.

### Proposed abstract

Pre-lift grasp monitors must distinguish unsuccessful attempts from successful
unfamiliar configurations. We study a logistic outcome predictor using joint
states and applied commands from HUG-based Shadow Hand execution in MuJoCo.
The predictor achieves failure-ranking AUROC 0.8523 on a retrospective
development test, but, after a separately versioned mug-clearance correction,
its unchanged warning rule flags all 23 completed mugs, including 20 successes.
Wrist commands in every successful corrected mug fall outside the original
training ranges, and a static audit finds no eligible mug control among the
eight observed training wrist pairs under the declared clearance gate.
A separate diagnostic with two feasible unfamiliar postures for each of eight
proposals also warns on all 13 successes and three failures. These related,
simulation-only diagnostics expose an operating-point reliability problem;
they motivate controlled calibration, representation and coverage comparisons
without establishing a causal remedy or physical transfer.

## Writing order through the deadline

**14 October: establish the argument and draft the core.** Review the proposed
title, central claim and counts together. Prepare the result table and timeline
first, then shorten the setup and write the results around those displays.
Convert the existing manuscript in place to the official initial-submission
template. Preserve model weights, labels, protocol versions and experiment
records; no new training study is required for this diagnostic submission.

**15 October: complete the explanation and verify the PDF.** Write the short
introduction/related work and discussion/limitations, then finalize the abstract
and conclusion. Verify source references, numerical denominators, anonymization,
figure captions and the four-page boundary. Compile the existing `.tex` with the
desktop compiler after changes and inspect its PDF for readability.

**15 October, before submission: resolve author details and upload.** Confirm
author profiles, authorship, reviewer commitment and the exact live form fields.
Prepare an accurate account of AI assistance for any applicable disclosure.
Aim for a first verified upload by 9:00 p.m. New York time to leave a buffer
before the 12:59 a.m. deadline. Both authors should review the submission; verify the uploaded PDF, author list
and track.

The first writing milestone is a reviewed core result and an official-template
draft, not an additional experiment. Provider adapters, mapping audits and
four-generator generalization stay outside this paper's central contribution.
