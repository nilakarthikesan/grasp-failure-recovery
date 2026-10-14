# HUG constrained refinement evidence

Implementation checkpoint, 14 October 2026. The isolated API and 50 new
manufactured tests are implemented; the installed API plus the existing
surface-refinement/guard tests pass all 120 checks. These are mathematical/API
checks, not certification of saved-grasp geometry.

The canonical joint-constrained experiment uses the six existing seed-0 HUG
apple/milk-box/mug proposals under runtime V2.1, protocol e42b74b787a8a1a5b5a661596466fe66b24c14eea39db25ad039fda2a462b8ff.
Its complete fits and unchanged independent six-coarse/three-fine audit are
**PENDING** at this checkpoint. No result from an incomplete search is accepted.

[Implementation evidence](implementation/manifest.json) binds the frozen API,
tests, settings and manufactured verification. The experimental API is identical
to the frozen solver used by the running experiment. The default mapper and
grasp-eval do not invoke it. No new generator inference, physical evaluation
episode, task label, predictor training or other-generator mapping occurs.

The completed experiment and independent reports will be added separately.
Historical protocols and incomplete runtime preparations will retain their
status; no previous scientific result is rewritten or pooled.
