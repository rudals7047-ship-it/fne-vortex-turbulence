# FNE-II: Vortex Topology and Turbulent Dissipation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20745146.svg)](https://doi.org/10.5281/zenodo.20745146)

Companion study to the granular force-network ensemble work
(**FNE-I**: [fne-contact-realization](https://github.com/rudals7047-ship-it/fne-contact-realization),
Zenodo [10.5281/zenodo.20514147](https://doi.org/10.5281/zenodo.20514147)).

## Summary

We test whether the topological organization of the vorticity field carries
predictive information about future local dissipation in turbulence, beyond
what low-order field-geometry features provide. Using a coarse-grained
Eulerian-cell representation of forced isotropic turbulence (Johns Hopkins
Turbulence Database, 128³ subdomain, Re_λ ≈ 433, 100 frames), we build a
vortex skeleton graph and adapt the relational-advantage framework of FNE-I.

**Main findings:**
- Relational (graph) features add predictive value for future dissipation
  beyond persistence and field geometry, at every horizon tested
  (ΔR²(R|Y,O) > 0, growing with horizon up to ~2 Kolmogorov times).
- The gain survives five robustness checks, including a strengthened
  geometric baseline that explicitly includes vorticity and
  velocity-gradient-tensor invariants (~70% of the signal retained).
- CRU_turb ≈ 0.58: field geometry recovers only about half the variance in
  skeleton degree, placing turbulence between glassy dynamics (CRU ≈ 0) and
  granular force networks (CRU ≈ 0.94–0.99) — interpreted, as a hypothesis,
  as an *epistemic* rather than *ontic* underdetermination.

## Repository structure

```
paper/      manuscript (LaTeX source + compiled PDF)
code/       analysis pipeline (JHTDB cutout → skeleton graph → forecasting)
figures/    main figure
docs/       research protocol, draft notes
```

## Reproducing

Analysis uses the public JHTDB forced isotropic turbulence dataset. The code
reads a JHTDB token from the `JHTDB_TOKEN` environment variable, falling back
to the public testing token (limited to small queries):

```python
import os
os.environ['JHTDB_TOKEN'] = '<your JHTDB token>'  # request one from JHTDB
```

Large intermediate caches (`*.npz`) are not tracked; the pipeline regenerates
them from the database.

## Status

Preprint / working manuscript. Feedback welcome.

## Citation

Kim, Kyeongmin. *Field Geometry Incompletely Reconstructs Realized Vortical
Adjacency in Turbulent Dissipation Forecasting.* Zenodo, 2026.
DOI: [10.5281/zenodo.20745146](https://doi.org/10.5281/zenodo.20745146)

## License

- Code (`code/`): MIT — see [`LICENSE_CODE_MIT`](LICENSE_CODE_MIT).
- Manuscript and figures (`paper/`, `figures/`): CC BY 4.0 — see [`LICENSE_MANUSCRIPT_CC_BY`](LICENSE_MANUSCRIPT_CC_BY).
