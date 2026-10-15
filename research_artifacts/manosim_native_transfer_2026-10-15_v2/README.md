# Literal exported shape: MANOSim transfer v2

The immutable v1 attempt stopped before any forward because exported HUG beta
differs from the nominal upstream shape file. Both development records contain
the same ten float32 coefficients. Their equality to BF16 rounding of the nominal
file is an observation, not proof of checkpoint origin.

Stage 1 freezes `build_protocol.json` before one neutral, right-only call through
MANOSim commit c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac's `_build_side`.
`shape_build.py` observes original producer arguments/returns without replacing
them. The upstream myhand configuration rules remain unchanged; bone mass and
inertia are shape-dependent outputs. Manufactured cube/tetra controls qualify
the installed Trimesh5.1.1 mass-property API. No model is built in those controls.

`run_shape_build.py` refuses MuJoCo model construction/forward/step, generation,
retargeting and collection. It runs one subprocess with a 600-second cap and no
retry. Generated assets and neutral producer geometry stay under ignored
`build_run_v1/raw/`; the tracked builder report preserves their write-time hashes.
A build receipt establishes asset provenance, not fidelity or physical validity.

Stage 2 requires independent readback and a **separate** native audit freeze
binding the generated shape, two XMLs, sixteen OBJs and producer report before
any native pose or field query. Keep original HUG geometry, pose, frame, fields,
part labels and acceptance limits. No collection or training is permitted here.
The two inspected cases are development cases, not held-out generalization.
Manuscript files are outside this work.
