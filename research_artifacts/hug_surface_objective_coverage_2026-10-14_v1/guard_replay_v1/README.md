# Saved HUG direction safeguard replay, V1

This separately frozen, post-hoc replay calls the new conservative direction selection guard on existing saved arrays. It creates no candidate pose, geometric query, optimizer run, simulation episode or label.

All six original identity controls are accepted. All six previously selected poses are refused, although their five original-relative fingertip caps still pass in the saved arithmetic. Each selected pose invalidates at least one originally valid fixed direction bin and worsens at least one available part's native-relative weighted direction SSE. report.json preserves every part and exact refusal reason.

Identity acceptance means that the safeguard recognizes its own baseline. It does not mean the original grasp is a faithful mapping: all nine original strict geometry audits remain FAIL. A refused candidate is not a native physical grasp failure, and no rejection becomes a training label.

The new safeguard uses per-part weighted direction SSE, not the global soft-L1 objective. Every fixed native score, including away-facing scores, remains a target; positive site weights are copied with no renormalization. No missing anatomical mass is redistributed. Any originally valid fixed site becoming invalid is refused, separately from whether its part's SSE increases. This is a conservative selection condition with zero SSE increase allowance; it does not certify unobserved sites, complete surfaces, unseen objects or physical outcomes.

The replay protocol was frozen before the first guard call, after inspection of the existing fitting/audit results. Expected controls are stated prospectively within this explicitly post-hoc diagnostic. The source guard, candidate report, import dependencies and prior coverage analysis are hash-bound. The seven inputs remain read-only.

Replay with the recorded project Python and PYTHONPATH from protocol.json, passing replay_direction_guard.py --root /absolute/project-work/checkout --output /a/new/report.json. The script refuses changed bound inputs, changed replay source and existing outputs. Required package imports are real; no dependency stubs or runtime installations were used.

No new HUG fit has been run with this safeguard. No production mapper was changed or promoted by this replay.
