# Saved-array evidence figures

These exact executed scripts render already sealed arrays and scalar reports
with NumPy/Matplotlib. They never instantiate MANO/MuJoCo, evaluate a skin,
query geometry, integrate physics or fit a model. PNGs and render receipts
stay under ignored local `runs`; licensed derived hand geometry is not added
to Git. Each receipt hashes every input, output and the exact renderer source.

`render_saved.py` produced the local
`runs/manosim_saved_validation_figures_2026-10-15_v2` images: source skin
motion/parity/derivative evidence and the retained original stock contact
failure. Its v2 corrects the initial plot's signed side1 world-z force to the
positive contact-normal magnitude and displays the complete data ranges.
Earlier illustration files remain separate; no scientific result changed.

`compare_contacts.py` produced
`runs/manosim_saved_contact_comparison_2026-10-15_v1/contact_model_evidence.png`.
The stock finer-step model remains NOT_QUALIFIED, while the separately
declared compliant model passes its eight synthetic conditions and four
timestep comparisons. These are different model assumptions; neither graph
proves real-hand calibration or complete native MANO contact.

The plots were visually inspected. Paths are explicit host-local provenance;
scripts require fresh output directories and will refuse to overwrite prior
renders. The hand figure describes the skin/state-function qualification,
not dynamic actuator tracking or the later engine-surface bridge.
