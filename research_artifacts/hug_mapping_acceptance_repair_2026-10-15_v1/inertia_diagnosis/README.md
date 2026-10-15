# Source-only inertia precision diagnosis

The unchanged milk source inertia tensor reconstructed by installed MuJoCo3.3.7 `mju_eig3` differs by5.829067976798074e−12 kg m², agreeing with the assembled bridge residual within1.27e−21. NumPy's symmetric eigensolver reconstructs it within9.76e−19. This diagnostic builds no model, performs no FK or physics and changes no source asset or acceptance tolerance. The separate implementation proposal canonicalizes only the copied tensor representation before compilation.

The source code's Jacobi decomposition and early stopping are documented in `source_provenance.json`; measurements and unchanged raw tensor are in `diagnosis.json`.
