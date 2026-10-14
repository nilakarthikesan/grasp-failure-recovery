# HUG constrained refinement evidence

14 October 2026. All six saved-grasp fits are complete. The solver returned 0 eligible surface corrections and retained 6 exact original poses. Independent original/final source preflight passes all six; the unchanged six-coarse/three-fine strict mapping audit returns 0 PASS, 9 FAIL and 0 UNRESOLVED.

[The method](../../docs/HUG_CONSTRAINED_REFINEMENT.md), [independent report](validation/REPORT.md)
and [all-case summary](validation/summary.json) distinguish engineering
source checks, candidate selection and strict scientific mapping acceptance.
A retained original is an explicit refusal, not a corrected mapping.

`implementation/` preserves the earlier implementation-only checkpoint,
whose pending status was accurate at its checkpoint. Its immutable manifest
binds the isolated API, 50 new manufactured tests and 120 installed API checks.
`fit/` preserves complete canonical runtime V2.1 singleton reports, the merged
candidate and all recorded major-callback histories, compressed byte-exact.
[The clarified fit report](fit/REPORT_COUNTER_CLARIFIED.md) is the reader
version. The original draft/report and its manifest remain byte-exact;
a separately bound note corrects only the wording of the 120-iteration
budget. Actual solver iterations and major callbacks are separate counters.
The pre-audit fit summary retains its then-pending audit status; the completed
independent result is in `validation/summary.json`.
`validation/` holds the independent all-six preflight and full nine-resolution
reports, unchanged rule, summary, scripts and diagnostic figures. `direction/`
retains the actual normal helper, manufactured fixtures and baseline parity.
`history/` preserves interrupted, import-aborted and unexecuted versions plus
separate measured performance fixtures; none is pooled into mapping results.

[results_manifest.json](results_manifest.json) hashes all stored evidence.
[compression_bindings.json](compression_bindings.json) binds the full original
canonical report/log bytes to compressed copies. Licensed source assets,
saved scene binaries and query arrays stay in their existing runtime paths.
Frozen scripts retain the recorded absolute input paths; adapting paths
requires a separately recorded reproduction configuration.

All 114 original protected source/model/paper/input bytes remain unchanged;
the installed new API/test are separately bound. Canonical fit guard selftests
total 18, with zero actual physics calls; the independent guards also report
zero. No inference, eval episode, outcome/label, predictor training or advance
to another generator occurred. The default mapper does not invoke this API.
