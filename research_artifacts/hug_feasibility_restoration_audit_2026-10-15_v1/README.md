# Conditional coarse HUG audit — not run

15 October 2026. This directory owns the planned independent coarse audit of the bounded restoration study. It performed **zero new coarse grids, zero fine grids, zero orientation queries, zero FK and zero physics steps**. The scientific acceptance rule remains SHA-256 `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`.

The source-bound result is `NOT_RUN_NO_ELIGIBLE_CANDIDATE`. All twelve restored checkpoints are reported tip-feasible by the fit output, but none passes the predeclared improvement/nonregression gate. All six final poses retain their pooled references. Independent saved-output replay checks the exact rank comparisons, protected orientation Boolean masks, and identity of selected q/root/domain fields against those previously audited references.

The replay passes 226 checks. Reasons overlap across endpoints:

- 12 of 12 lack strict improvement under the unchanged five-component exact rank.
- 10 of 12 invalidate at least one previously valid protected orientation row.
- 6 of 12 lose at least one previously passing protected orientation row.

An invalid orientation row is not a proven sign reversal. Some retained rows can improve while the complete gate still refuses the candidate. These counts are fitting-row results, not full-grid measurements. The separate numerical review owns actual source/FK/cap/trust verification; this audit's replay uses saved JSON only.

The previous nine strict FAIL results remain **historical evidence for the unchanged selected pooled poses**, not newly measured results. Refused restored endpoints receive no coarse/fine full-grid status here. A tip restoration pass is not mapping acceptance.

## Prepared but unexecuted code

`prepare_wrapper.py` derives `validate_candidate.py` from the prior frozen direct-constraint audit. Text-parity assertions preserve `full_fields`, `status`, and the original grid/proximity block byte for byte. It changes only eligible-case/source/trust preflight, the density-2-only schedule and explicit short circuit after a definite proximity failure. If used later, that short circuit would report orientation `NOT_RUN`; otherwise it would call the same historical orientation routine. No finer execution is automatic.

`protocol.DRAFT.json` is deliberately **not frozen and not executable**. Its candidate bindings and eligibility-dependent indices remain unset. No `protocol.json`, geometry report or preflight scene exists here. The wrapper was compiled as Python text but was never imported or executed; no claim of a completed numeric wrapper validation is made.

`record_skip.py` is a standard-library saved-output replay; `audit_skip_receipt.json` binds the exact candidate, fit protocol, pooled reference, prior audit summary and preparation files. The receipt explicitly distinguishes zero fresh grids from nine historical failures.

No shared source, documents, mapper, generator output, outcome labels or evaluation dataset were changed by this audit.
