"""Save parallel work assignments and update the existing, single-file manuscript."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
TMP = Path(__file__).parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def block(name, text):
    return f'% BEGIN GFP SECTION {name}\n'+text.strip()+'\n'+f'% END GFP SECTION {name}\n'

ABSTRACT = r'''
\begin{abstract}
Grasp-failure predictors trained on simulated attempts may encounter unfamiliar
objects and configurations whose coverage depends on the grasp proposal source.
We investigate whether training with proposals from multiple generators improves
generalization relative to single-generator training, using HUG, ContactGen,
DexGrasp Anything and UniDexGrasp as candidate sources for a common Shadow Hand
task in MuJoCo. Before collecting comparison data, we audit each source's native
representation, coordinate conversion, articulation and object-relative surface
geometry, separating mapping fidelity from eventual task success.
Completed HUG development checks support numerical reconstruction and kinematic
composition, but reject strict surface preservation across six detailed cases
and three finer repeats; a constrained correction accepts no fitted candidate
and retains all six original poses. Historical HUG prediction diagnostics also
show that useful failure ranking can coexist with warnings on successful
unfamiliar attempts. The multiple-generator training comparison and independent
generalization evaluation remain unexecuted, so improved prediction is a
hypothesis rather than a reported result.
\end{abstract}
'''

INTRO = r'''
\section{Introduction}
Simulation makes it practical to collect labeled grasp attempts for training an
outcome predictor. The resulting training distribution depends on which grasps
are proposed, how those proposals are translated to the simulated hand, and how
the hand executes them. A predictor exposed to several proposal sources might
encounter a broader range of grasp configurations than one trained from a single
source. Broader coverage need not improve prediction: additional sources can
introduce redundant examples or different translation errors.

Our central research question is:
\begin{quote}
Does training a grasp-failure predictor on simulated grasps from multiple
generators improve generalization to unfamiliar objects and grasp configurations
compared with single-generator training?
\end{quote}
We propose comparisons using one, two, three and four proposal sources under a
common hand, task, execution protocol and predictor specification. Matched
selected training counts distinguish source diversity from supplying more
examples; they are an experimental control rather than the research question.
Object-held-out and generator-held-out tests address distinct generalization
claims. These comparisons have not yet been executed.

HUG, ContactGen, DexGrasp Anything and UniDexGrasp supply different native
representations and generation stages \cite{hug,contactgen,dga,unidex}.
Their published grasp-quality results do not certify our translations.
We audit each proposal against its own declared native configuration before
interpreting simulator outcomes. Mapping validation supports the prediction
study: it neither requires every proposal to succeed nor guarantees downstream
generalization. The target is eventual grasp, lift and hold completion from a
closing prefix, rather than slip onset or a certified stable-grip decision.
'''

COMMON = r'''
\section{Grasp sources and simulator mapping}
\subsection{Validation framework}
Each source record identifies its input geometry, physical scale, coordinate
frames, hand assets, checkpoint, configuration and output stage. We first
reconstruct the native proposal and independently check transformation,
joint-order and forward-kinematic consistency. We then compare native and
translated configurations in a shared object frame, reporting per-finger
landmark errors, object proximity, penetration, lost and added near-surface
regions, and reliable surface-direction comparisons. Visual and collision
surfaces are distinct references, with unmatched regions and sampling uncertainty
retained in every report.

Numerical decoding, geometric preservation and execution adequacy are separate
claims. Agreement of joints or fingertips does not establish equivalent
collision surfaces, contact forces or task outcomes. Poor native proposals
remain in the audit; successful lifting is not a mapping acceptance criterion.
Thresholds and sampling assumptions are declared explicitly, and inspected
development cases are distinguished from later independent validation.
Our common executor differs from source-specific policies and quality tests.

For the current HUG development audit, a frozen strict encoded-surface claim
is refuted by a positive same-part lost/added proximity-area lower bound or a
reliable object-facing sign reversal, under declared numerical assumptions.
An absent sampled reversal cannot certify continuous normal agreement. That
claim defines strict preservation, not a universal tolerance for all useful
retargeters. An approximate simulator-pipeline claim would require separately
justified distortion budgets and independent adequacy evidence; it cannot
retroactively turn the recorded HUG failures into passes.

Provider audits and their evidence-backed manuscript subsections can proceed
in parallel. Shared retargeters, assets, measurement code and execution settings
are versioned dependencies: a change requires reviewing affected audits.
The current phase uses static reconstruction of saved inputs and manufactured
checks. It generates no new evaluation episodes, physical outcome labels or
training data. Actual-native validation remains pending where authentic source
outputs or assets have not been established.
'''

HUG = r'''
\subsection{HUG}
HUG supplies MANO hand hypotheses from observations \cite{hug}. Our conversion
places the native wrist at the Shadow palm position and retargets object-relative
landmarks through Dex Retargeting \cite{dex}. Native and target hands have
different surface geometry, so numerical reconstruction and correct forward
kinematics are necessary but insufficient evidence of grasp preservation.

All 32 saved native exports reconstruct exactly in the recorded 3D audit;
this does not assert all-32 camera-projection equivalence. Detailed surface
measurements on six existing apple, milk-box and mug development cases, together
with three finer repeats, reject the declared strict preservation claim.
Definite lost/added proximity regions and reliable sampled orientation reversals
remain counterexamples. The nine resolutions are repeated measurements of six
grasps, not nine independent physical failures.

An experimental local fit adds five individual fingertip nonworsening
constraints, a separate strict 10 mm mean-error guard and native-relative facing
targets, including away-facing targets. Final engineering assessment separately
declares a 10 nm arithmetic allowance; original tip errors are approximation
errors, rather than scientific acceptance margins. None of the six fits provides
an eligible improving candidate; each returns its exact original joint/root pose.
Independent source checks pass, while the unchanged surface audit returns zero
PASS and nine FAIL. Pointwise orientation separately has seven FAIL and two
UNRESOLVED results. This is not a repaired mapping or evidence of global
infeasibility. Physical contact and failure-label preservation remain unconfirmed.
'''

CONTACT = r'''
\subsection{ContactGen}
ContactGen predicts object-centric contact, hand-part and direction maps,
then optimizes a human hand pose \cite{contactgen}. Our worker retains refined
MANO vertices and joint frames. The adapter restores vertex-mean object
centering, applies the declared mesh-to-object transform, constructs thumb-first
landmarks and uses the MANO-to-Shadow path shared with HUG. The shaped wrist
origin comes from joint frame 0 rather than optimized translation alone.

Current static checks support transform composition at two prescribed wrist
states, with residual at most $1.21\times10^{-16}$ m. This is algebraic evidence;
no authentic native ContactGen proposal has been validated in the current audit.
Independent pose/shape reconstruction and translated surface fidelity remain
pending. The current worker also needs preservation of sampled input points and
predicted contact/part/direction maps before a contact-to-native solver claim
can be substantiated. ContactGen's bone-local direction ray is distinct from an
actual surface normal. HUG's measured errors are not ContactGen results.
'''

DGA = r'''
\subsection{DexGrasp Anything}
DexGrasp Anything uses a diffusion model with physical constraints
\cite{dga}. The pinned implementation exports a denormalized articulated
Shadow configuration containing translation, a 6D rotation representation and
24 joints, including both wrists. The root is the forearm; conversion must
compose the actual wrist-dependent forearm-to-palm transform and preserve
normalization, scale and the selected sampling/optimization stage.

A dedicated static bridge candidate was examined on zero and 64 deterministic
joint configurations, with maximum fingertip separation of 0.100 mm. This is
synthetic kinematic evidence, not native proposal validation. The source-specific
decoder, inference worker and accepted registered mapping remain pending.
The older 22-finger/free-palm assumption is not the pinned source contract.
Collision-surface fidelity and actual generated-output preservation have not
been established. Native quality tests are separate from our common task labels.
'''

UNI = r'''
\subsection{UniDexGrasp}
UniDexGrasp separates diverse pose generation from a goal-conditioned execution
policy \cite{unidex}. Our worker exports the pre-adaptation IPDF/Glow proposal
stage; it does not reproduce the complete learned execution system. The adapter
inverts canonicalization, composes input-to-object and the actual source palm
offset, and maps 22 named finger articulations into 24 target joints with fixed
target wrists. Raw and adapted outputs remain distinct stages.

The available DexGraspNet-reference kinematic bridge was checked on 112 synthetic
configurations, with maximum fingertip discrepancy of 1.751 mm. Two corner
configurations exceed target limits and are rejected rather than clipped.
This reference certificate is not validation of UniDexGrasp's actual generation
hand asset or a real native proposal, neither of which has been established in
the current evidence. The certificate does not cover collision surfaces,
friction or inertia, so reference FK agreement cannot certify contact fidelity.
'''

PLANNED = r'''
\section{Planned multiple-generator comparison}
\paragraph{Status and validity gate.}
The source-diversity study remains proposed. Native mapping claims and their
limits must be documented before collecting comparison labels. The current
HUG strict-preservation failures and other providers' missing actual-native
evidence cannot be presented as completed four-source validation. A declared
approximate-pipeline study needs its own justified adequacy checks.

\paragraph{Compare source count and source identity.}
Compare predictors trained from one, two, three and four proposal generators.
With four sources there are 15 nonempty source subsets: four single-source,
six two-source, four three-source and one four-source condition. Evaluating
only one nested source chain would confound source count with generator identity.
Whether to execute all subsets or preregister a feasible balanced subset remains
an experimental-design choice, not a completed experiment. Matched selected
training counts control for added data; source diversity does not itself imply
better coverage or improved prediction.

Keep the canonical hand, physical object pool, task/executor and labels,
predictor inputs/architecture and tuning budget consistent. Declare source
quotas, class-mixture handling and grouping before collection; fit normalization
on training only. Retain inference cost, generated/selected/rejected counts and
mapping refusal reasons. An adapter rejection is not a physical task-failure
label. The generic provider recorder still needs the closing velocity and
applied-command fields needed by the historical predictor; input compatibility
must be established before using that predictor as a reference.

\paragraph{Generalization and baselines.}
Use one independent object-held-out evaluation pool across conditions; keep
related observation/proposal families and their disturbances in the same split.
Held-out-generator evaluation is a separate transfer question. Tune operating
points and calibration on validation only, freeze models before test scoring,
and report false warnings on successful tasks together with missed failures,
ranking, calibration and uncertainty respecting shared families. Undefined
failure recall when a group has no failures remains explicit.

Use the same predictor specification for the primary source-diversity ablation.
Architecture comparisons and compatible external predictors answer additional
questions; published scores on different tasks/hands/test sets are not direct
baselines. The original logistic predictor and its 0.5 threshold remain a
historical reference, not evidence for the new comparison. Neither always-warning
nor always-clearing behavior meets both objectives. No source mixture, new
predictor, calibration or fresh holdout has been fitted or scored in this phase.
'''

BIB = r'''
\bibitem{contactgen}
S. Liu, Y. Zhou, J. Yang, S. Gupta, and S. Wang.
\emph{ContactGen: Generative Contact Modeling for Grasp Generation}.
ICCV, 2023. \url{https://arxiv.org/abs/2310.03740}.

\bibitem{dga}
Y. Zhong, Q. Jiang, J. Yu, and Y. Ma.
\emph{DexGrasp Anything: Towards Universal Robotic Dexterous Grasping
with Physics Awareness}.
CVPR, 2025. \url{https://arxiv.org/abs/2503.08257v2}.

\bibitem{unidex}
Y. Xu et al.
\emph{UniDexGrasp: Universal Robotic Dexterous Grasping via Learning
Diverse Proposal Generation and Goal-Conditioned Policy}.
CVPR, 2023. \url{https://arxiv.org/abs/2303.00938v2}.

'''

def main():
    paper = ROOT/'docs/paper/working_draft.tex'
    old = paper.read_text()
    oldhash = sha(paper)
    assert '% BEGIN GFP SECTION' not in old
    text = old.replace('Evaluating Simulation Diversity for Grasp Failure Warnings',
                       'Evaluating Simulation Diversity for Grasp Failure Prediction',1)
    text = text.replace(r'\date{12 October 2026}',r'\date{14 October 2026}',1)
    a = text.index(r'\begin{abstract}');b = text.index(r'\end{abstract}',a)+len(r'\end{abstract}')
    text = text[:a]+block('abstract',ABSTRACT)+text[b:]
    a=text.index(r'\section{Question and relation to prior work}')
    b=text.index('Failure prediction and recovery are established topics.',a)
    related_end=text.index(r'\section{Execution, labels and predictor}',b)
    related=text[b:related_end].strip()
    related=related.replace('Our candidate contribution is a reproducible, embodiment-specific investigation\nof which simulated variations improve warnings at a matched training-data\nbudget.', 'Our proposed contribution is a controlled study of multi-source simulation\ncoverage for failure prediction, supported by explicit embodiment-specific\nmapping audits.')
    related=related.replace('our candidate\ncontribution concerns a controlled HUG collection and representation comparison.', 'our proposed\ncontribution concerns source-diversity and representation comparisons in a\ncommon simulation task.')
    introduction=INTRO.strip()+'\n\n'+r'\subsection{Related work}'+'\n'+related+'\n'
    mapping='\n'+block('mapping_common',COMMON)+'\n'+block('hug',HUG)+'\n'+block('contactgen',CONTACT)+'\n'+block('dexgrasp_anything',DGA)+'\n'+block('unidexgrasp',UNI)+'\n'
    text=text[:a]+block('introduction',introduction)+mapping+text[related_end:]
    text=text.replace(r'\section{Execution, labels and predictor}',r'''\section{Historical HUG execution and predictor reference}
The following protocol and fitted predictor describe the completed HUG-only
development study. They motivate the broader question but are not mixed-source
training results. Controller versions and their original labels remain distinct;
the common prospective protocol must be declared before new collection.''',1)
    text=text.replace(r'\section{Completed evidence}',r'\section{Historical HUG development evidence}',1)
    a=text.index(r'\section{Planned comparison and decision criteria}')
    b=text.index(r'\section{Limitations and evidence availability}',a)
    text=text[:a]+block('planned_comparison',PLANNED)+'\n'+text[b:]
    a=text.index(r'\section{Limitations and evidence availability}')
    b=text.index(r'\begin{thebibliography}',a)
    limitations=text[a:b]
    limitations=limitations.replace('Our completed diagnostic concerns one hand, fixed task definitions and reused', 'The four-source training/generalization comparison remains unexecuted and\nactual-native mapping validation is incomplete for three sources. HUG\nretargeting preserves neither all proximity relationships nor all reliable\ndirections under the declared strict development claim. These limitations\ncannot be hidden by numerical/FK passage or successful task outcomes.\nOur historical predictor diagnostic concerns one hand, fixed task definitions and reused',1)
    limitations+='The latest static mapping records are in '+r'\path{hug_constrained_refinement_2026-10-14}'+'.\nThe manuscript grows alongside provider-specific audit reports; parallel prose\nupdates do not modify frozen experimental inputs or turn pending tests into results.\n'
    text=text[:a]+block('limitations',limitations)+'\n'+text[b:]
    a=text.index(r'\begin{thebibliography}');b=text.index(r'\end{thebibliography}',a)+len(r'\end{thebibliography}')
    bib=text[a:b].replace(r'\bibitem{dex}',BIB.strip()+'\n\n'+r'\bibitem{dex}',1)
    text=text[:a]+block('bibliography',bib)+text[b:]
    historical_old=old[old.index(r'\paragraph{Original pilot.}'):old.index(r'\section{Planned comparison and decision criteria}')]
    historical_new=text[text.index(r'\paragraph{Original pilot.}'):text.index('% BEGIN GFP SECTION planned_comparison')]
    assert historical_old.rstrip()==historical_new.rstrip(), 'Historical results must remain exact'
    paper.write_text(text)
    (ROOT/'scripts/update_paper_section.py').write_bytes((TMP/'update_paper_section.py').read_bytes())
    tests=(TMP/'test_update_paper_section.py').read_text()
    # The installed test locates the script through the repository root.
    tests=tests.replace('SCRIPT = Path(__file__).with_name("update_paper_section.py")',
                        'SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_paper_section.py"')
    (ROOT/'tests/test_paper_section_updates.py').write_text(tests)
    docs=ROOT/'docs/parallel_validation';docs.mkdir(exist_ok=True)
    for name in ('CONTACTGEN_VALIDATION_HANDOFF.md','UNIDEXGRASP_VALIDATION_HANDOFF.md','DEXGRASP_ANYTHING_VALIDATION_HANDOFF.md'):
        (docs/name).write_bytes((TMP/name).read_bytes())
    (ROOT/'docs/PARALLEL_GENERATOR_VALIDATION_HANDOFF.md').write_bytes((TMP/'PARALLEL_GENERATOR_VALIDATION_HANDOFF.md').read_bytes())
    evidence=ROOT/'research_artifacts/parallel_validation_coordination_2026-10-14';evidence.mkdir(exist_ok=True)
    (evidence/'paper_section_update_verification.json').write_bytes((TMP/'paper_section_update_verification.json').read_bytes())
    (evidence/'manuscript_before_parallel_update.tex').write_text(old)
    (evidence/'install_parallel_handoffs_and_paper.py').write_bytes(Path(__file__).read_bytes())
    verification={
        'schema':'parallel_generator_handoffs_and_paper_initialization_v1',
        'date':'2026-10-14','previous_branch_commit':'0db42ca9d768f760ec61bd67fa6c7e6d3711e33b',
        'paper_path':str(paper),'paper_before_sha256':oldhash,'paper_after_sha256':sha(paper),
        'historical_HUG_result_text_preserved_exact':True,
        'initialization_before_user_threads_launch':True,
        'source_ids':['hug','contactgen','dexgrasp_anything','unidexgrasp'],
        'new_generator_inference':False,'physics_steps':0,'new_eval_data':False,'predictor_training':False,
        'future_paper_updates':'shared lock+reread+expected owned-section hash',
        'installed_helper_sha256':sha(ROOT/'scripts/update_paper_section.py'),
        'paper_compile_status':'PENDING',
        'note':'The previous HUG 114-file snapshot includes its historical paper. This separately authorized manuscript revision is not a rerun or revision of that static experiment.'}
    (evidence/'initialization_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
