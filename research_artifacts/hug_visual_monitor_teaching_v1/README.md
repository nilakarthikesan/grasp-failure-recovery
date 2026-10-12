# HUG visual monitor teaching record

Prepared 12 October 2026. This record binds existing local cube recordings,
their shared HUG proposal, source outcomes and explanatory images to the
[learning guide](../../docs/HUG_VISUAL_MONITOR_LEARNING_GUIDE.md).

It is **teaching documentation**, not a visual training dataset, new rollout
collection, trained monitor, VLA integration or reproduced external evaluation.
Original simulator data and the original joint predictor remain unchanged.

The success has 147.97 mm peak lift and two qualifying hold seconds. The
failure has 52.84 mm peak lift and zero qualifying hold seconds, despite
contact and some lift. It is not an annotated empty grasp or slip-onset example.
Both use HUG seed 2; their finger-offset bounds are 0 and ±0.6 radians.

`manifest.json` records absolute local paths, SHA-256 hashes, historical
execution identity, labels and three primary reading sources. Raw files remain
in ignored `runs/` directories in the runtime checkout; a clone of this
artifact does not include them. Explanatory videos and proposal overlays must
not be treated as clean model inputs.

The guide separates future-outcome forecasting from acquisition detection and
describes a proposed real-only, targeted-simulation and size-matched random
simulation comparison. No scores or improvement claims were generated here.
